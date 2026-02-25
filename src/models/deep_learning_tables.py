import torch
import torch.nn as nn
import torch.nn.functional as F

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse projection onto the probability simplex.

    Unlike softmax, which always distributes mass across all features,
    sparsemax produces exact zeros for features below the support threshold τ.
    This is the mathematical engine behind TabNet's interpretable masks.
    """
    z = z - z.max(dim=dim, keepdim=True).values  # numerical stability
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    n = z.shape[dim]
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    k = k.view(shape)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    support = (1 + k * z_sorted > cumsum)
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1).float()
    tau_idx = (k_z - 1).long().clamp(max=n - 1)
    tau = (cumsum.gather(dim, tau_idx) - 1) / k_z
    return torch.clamp(z - tau, min=0)


class GLUBlock(nn.Module):
    """Gated Linear Unit block: (Wx + b) · σ(Vx + c).

    The gating mechanism acts as a learned filter: the first half of the linear
    projection produces a value, the second half (after sigmoid) decides how much
    of that value to pass through. Residual connection + √0.5 normalisation stabilises
    training in deep stacks.
    """
    def __init__(self, in_dim: int, out_dim: int, fc: nn.Linear | None = None):
        super().__init__()
        self.fc = fc if fc is not None else nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = nn.BatchNorm1d(out_dim * 2)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.fc(x))
        h_val, h_gate = h[:, :self.out_dim], h[:, self.out_dim:]
        return h_val * torch.sigmoid(h_gate)


class FeatureTransformer(nn.Module):
    """Two-layer GLU stack: one shared layer (same weights across all steps)
    + one step-specific layer. The shared layer learns general feature embeddings;
    the step-specific layer adapts them to the current step's focus.
    """
    def __init__(self, n_features: int, n_d: int, shared_fc: nn.Linear):
        super().__init__()
        self.shared_block    = GLUBlock(n_features, n_d, fc=shared_fc)
        self.specific_block  = GLUBlock(n_d, n_d)
        self.scale = (0.5 ** 0.5)  # residual normalisation factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared_block(x)
        return (h + self.specific_block(h)) * self.scale


class AttentiveTransformer(nn.Module):
    """Maps the previous step's representation → a sparse feature selection mask.

    Prior scales P penalise features that were already heavily attended in previous
    steps, enforcing diversity across the N_steps selections.
    """
    def __init__(self, n_d: int, n_features: int):
        super().__init__()
        self.fc = nn.Linear(n_d, n_features, bias=False)
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, h_prev: torch.Tensor, prior_scales: torch.Tensor) -> torch.Tensor:
        return sparsemax(prior_scales * self.bn(self.fc(h_prev)))


class ScratchTabNet(nn.Module):
    """TabNet regressor built from scratch.

    Parameters
    ----------
    n_features : number of input features
    n_d        : width of the feature transformer output (= n_a in the paper)
    n_steps    : number of sequential attention steps
    gamma      : feature reuse coefficient (1.0 = no reuse, 1.3 = mild reuse)
    lambda_s   : entropy regularisation weight for sparsity
    """
    def __init__(
        self,
        n_features: int,
        n_d: int = 16,
        n_steps: int = 3,
        gamma: float = 1.3,
        lambda_s: float = 1e-4,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_d        = n_d
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.lambda_s   = lambda_s

        self.initial_bn = nn.BatchNorm1d(n_features)

        # One shared FC whose weights are reused by every FeatureTransformer
        self.shared_fc = nn.Linear(n_features, n_d * 2, bias=False)

        self.feature_transformers  = nn.ModuleList([
            FeatureTransformer(n_features, n_d, self.shared_fc) for _ in range(n_steps)
        ])
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(n_d, n_features) for _ in range(n_steps)
        ])
        self.final_fc = nn.Linear(n_d, 1)

        # Store masks after each forward pass for interpretability
        self._masks: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._masks = []
        x_bn = self.initial_bn(x)

        # Bootstrap: process full (unmasked) input to seed the first attentive step
        h = self.feature_transformers[0](x_bn)
        aggregated = torch.zeros(x.shape[0], self.n_d, device=x.device)
        prior_scales = torch.ones(x.shape[0], self.n_features, device=x.device)

        entropy_loss = torch.tensor(0.0, device=x.device)

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](h, prior_scales)
            self._masks.append(mask.detach())

            # Penalise reuse of already-selected features
            prior_scales = prior_scales * (self.gamma - mask)

            # Process masked input
            h = self.feature_transformers[step](mask * x_bn)
            aggregated += F.relu(h)

            # Sparsity regularisation: encourage masks to be as concentrated as possible
            entropy_loss -= (mask * torch.log(mask + 1e-15)).sum(dim=-1).mean()

        pred = self.final_fc(aggregated).squeeze(-1)
        return pred, self.lambda_s * entropy_loss

    @property
    def feature_importance(self) -> torch.Tensor:
        """Average attention weight per feature across all steps and samples."""
        if not self._masks:
            raise RuntimeError("Run a forward pass first.")
        return torch.stack(self._masks).mean(dim=0).mean(dim=0)  # (n_features,)

    @property
    def step_masks(self) -> torch.Tensor:
        """Per-step average attention weights: shape (n_steps, n_features)."""
        if not self._masks:
            raise RuntimeError("Run a forward pass first.")
        return torch.stack(self._masks).mean(dim=1)  # (n_steps, n_features)


class ExU(nn.Module):
    """Exp-Centered Unit: output_j = (x - b_j) * exp(w_j).

    Each unit j has its own learned bias b_j (the activation centre) and
    weight w_j (the amplification factor). Initialising b_j ~ N(0, 0.5)
    distributes unit thresholds across the input range so that both negative
    and positive scaled values receive non-zero responses. exp(w_j) is always
    positive, so the sign of the output is determined by (x - b_j); the
    subsequent ReLU then selects which units fire.
    """
    def __init__(self, out_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_features))
        self.b = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.w, mean=4.0, std=0.5)   # high initial amplification
        nn.init.normal_(self.b, std=0.5)              # spread thresholds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,),  b/w: (H,)  ->  (B, H)
        return (x.unsqueeze(-1) - self.b) * torch.exp(self.w)


class FeatureNN(nn.Module):
    """One sub-network per feature: maps scalar x_j -> f_j(x_j).

    Architecture: ExU(H) -> ReLU -> Dropout -> Linear(H, 1)
    """
    def __init__(self, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.exu     = ExU(hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.exu(x))                         # (B, H)
        return self.linear(self.dropout(h)).squeeze(-1)  # (B,)


class ScratchNAM(nn.Module):
    """Neural Additive Model: y_hat = bias + sum_j f_j(x_j)."""
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            FeatureNN(hidden, dropout) for _ in range(n_features)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        contribs = [net(x[:, j]) for j, net in enumerate(self.feature_nets)]
        return self.bias + sum(contribs), contribs

    @torch.no_grad()
    def shape_function(self, j: int, n_pts: int = 300):
        """Evaluate f_j over a dense standardised grid."""
        grid = torch.linspace(-3.5, 3.5, n_pts)
        vals = self.feature_nets[j](grid)
        return grid.numpy(), vals.numpy()
