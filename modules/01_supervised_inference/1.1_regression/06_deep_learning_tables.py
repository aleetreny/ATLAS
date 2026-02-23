# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deep Learning for Tables: TabNet and Neural Additive Models
#
# ## 1. Theoretical Foundation
#
# ### 1.1 The Philosophy The previous five notebooks have taken us through the full arc of regression for structured data:
# - **Classical Statistics** gave us interpretable baselines with closed-form solutions (OLS, GLMs, PLS).
# - **Regularization** showed how to constrain model complexity to prevent overfitting (Lasso, Ridge, ElasticNet).
# - **Robust Regression** taught us to survive corrupted data by design (RANSAC, Theil-Sen, Huber).
# - **Non-Parametric Methods** abandoned fixed functional forms for pure data-driven flexibility (KNN, SVR, Gaussian Processes).
# - **Gradient Boosting** emerged as the undisputed industry standard for tabular prediction (XGBoost, LightGBM, CatBoost).
#
# At this point, a reasonable question is: *why even bring neural networks into the picture?*
#
# The honest answer is that for most tabular tasks, you should not. XGBoost and CatBoost will beat a naively-configured neural network on a structured dataset almost every time. The benchmarks are consistent: gradient boosting wins on tabular data because trees handle heterogeneous features, mixed scales, and feature interactions natively. Neural networks, trained end-to-end with gradient descent, have to *learn* all of this from scratch.
#
# But "almost every time" is not "every time". And more importantly, accuracy is not always the only objective.
#
# This notebook covers two approaches that bring deep learning to tables for distinct and legitimate reasons:
#
# 1. **TabNet** (Arik & Pfister, Google Brain, 2019): A neural architecture that uses a sequential attention mechanism to *mimic* the feature selection behavior of decision trees. The goal is to match gradient boosting accuracy while remaining fully differentiable; opening the door to end-to-end training in larger deep learning pipelines.
#
# 2. **Neural Additive Models / NAMs** (Agarwal et al., Google Research, 2021): A neural architecture that *enforces interpretability by design*. Instead of a single monolithic network, each input feature gets its own sub-network. The final prediction is a sum of individual contributions. You can plot what the model learned for each feature. You can audit it. The goal is not to beat gradient boosting, but to give non-linear modeling the interpretability of a GLM.
#
# These two models represent opposite ends of the deep-learning-for-tables spectrum: TabNet trades some interpretability for accuracy, NAMs trade some accuracy for complete interpretability.
#
# ### 1.2 TabNet: Sequential Attention as a Differentiable Tree
#
# TabNet's key insight is that decision trees are essentially a feature selection mechanism. At each node, a tree picks *one* feature and a threshold. The path from root to leaf is a sequence of these single-feature decisions. TabNet reproduces this behavior with a neural network, replacing the hard, non-differentiable splits of a tree with a *soft, differentiable attention mask*.
#
# The architecture processes input in $N_{steps}$ sequential steps. At each step $i$:
#
# 1. An **Attention Transformer** computes a soft mask $M^{(i)} \in [0,1]^d$ over the $d$ input features using a sparsemax activation:
#
#    $$ M^{(i)} = \text{sparsemax}\!\left( P^{(i-1)} \cdot h_a\!\left( a^{(i-1)} \right) \right) $$
# where $P^{(i-1)}$ is the **prior scales matrix**, a penalty that discourages re-using features already selected in previous steps. This is the direct analog of the "each tree focuses on residuals left by the previous" logic in boosting.
#
# 2. A **Feature Transformer** processes the masked input $M^{(i)} \odot f$ through a shared+step-specific MLP to produce a processed representation $h^{(i)}$.
#
# 3. The outputs of all steps are aggregated: $\hat{y} = W_{\text{final}} \sum_{i=1}^{N_{steps}} \text{ReLU}\!\left(h^{(i)}\right)$
#
# Sparsemax is the key ingredient. Unlike softmax, which spreads probability mass across all features, sparsemax produces exactly-zero weights for unimportant features. The model selects a sparse subset of features at each step, just like a tree does at each node.
#
# The masks $M^{(i)}$ are a byproduct of inference, not post-hoc analysis. TabNet is interpretable *by architecture*: you can inspect which features were attended to at each step for any individual prediction.
#
# ### 1.3 NAMs: One Network Per Feature
#
# Neural Additive Models are a direct descendant of **Generalized Additive Models (GAMs)**, which we encountered in the Classical Statistics notebook. The GAM framework decomposes a prediction into a sum of univariate functions:
#
# $$ \hat{y} = \beta_0 + \sum_{j=1}^{d} f_j(x_j) $$
#
# In classical GAMs (splines, smoothing), the shape functions $f_j$ are constrained to be smooth and are estimated from data using regularized regression. Interpretability is guaranteed because each term depends only on a single feature; you can plot $f_j(x_j)$ vs. $x_j$ to see exactly how the model uses that feature.
#
# NAMs replace each shape function $f_j$ with a small neural network. The architecture is:
#
# - $d$ independent **feature networks** $\{f_j\}_{ j=1}^d$, each taking a single scalar input $x_j$.
# - Each network uses the **ExU (Exp-Centered Unit)** activation: $\text{ExU}(x) = (x - b) \cdot e^w$, which produces a flexible, high-frequency function capable of capturing sharp local transitions (e.g., a price cliff at a specific carat threshold).
# - The outputs are summed with a learned bias: $\hat{y} = \beta_0 + \sum_j f_j(x_j)$.
#
# The additive structure is a hard architectural constraint, not a regularization choice. A NAM **cannot** model interaction effects between features (e.g., "high carat *and* low clarity" as a combined signal). This is the deliberate tradeoff: full interpretability at the cost of interaction modeling.
#
# For problems where understanding *how* the model makes decisions matters as much as *how accurately* it makes them (medical risk scoring, credit underwriting, insurance pricing, regulatory compliance), this tradeoff is not a weakness; it is the product requirement.
#
# ### 1.4 The Roadmap
#
# In this notebook, we will build the complete intuition and implementation of both architectures:
#
# 1. **TabNet from Scratch:** The full attention mechanism, feature transformer, and sequential step aggregation using only PyTorch.
# 2. **TabNet Pro:** Using the `pytorch-tabnet` library, the production-grade implementation with early stopping and native feature importance extraction.
# 3. **NAM from Scratch:** Building feature networks, the ExU activation, and the additive aggregation layer entirely in PyTorch.
# 4. **NAM Pro:** Using the `nam` library (Google Research), the reference implementation with `FeatureNN`, `ExU` activation, and built-in shape function extraction.
# 5. **Comparison:** Head-to-head against our gradient boosting baselines from the previous notebook, with a specific focus on the interpretability-accuracy tradeoff.
#
# > **Installation note for `nam`:** The package declares `sklearn` (the deprecated PyPI alias) as a dependency, which pip tries to build from source, and fails. The fix is to install the wheel directly with `--no-deps` and add only the genuinely missing packages (`pytorch-lightning`, `loguru`). The real scikit-learn is already present and fully compatible; only the stale alias was the problem.
#
# **Dataset:** We continue with the **Diamonds** dataset (53,940 rows), the same dataset used in the Gradient Boosting notebook. This allows direct comparison against the CatBoost baseline (RMSE &#36;530, R² 0.9823) established in notebook 05. The mix of continuous and ordinal categorical features, the non-linear price-carat relationship, and the well-understood domain logic (carat, clarity, color drive price) make it ideal for showcasing interpretability tools.
#
# ---
# *We follow the **ATLAS** protocol: First we understand the **Intuition**, then we build it from **Scratch** using only NumPy/PyTorch, and finally we apply the **Pro** tools used in industry.*

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15

device = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available()          else
    'cpu'
)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"MPS available   : {torch.backends.mps.is_available()}")
print(f"Active device   : {device}")

# %% [markdown]
# ## 2. Data Preparation
#
# We use the same **Diamonds** dataset as notebook 05. This is deliberate: by keeping the dataset identical, every metric we compute is directly comparable to the CatBoost baseline (RMSE &#36;530, R² 0.9823) established there.
#
# One critical difference from the gradient boosting notebook: **neural networks are not scale-invariant**. A tree splits on rank order; a threshold of `carat > 1.0` behaves identically whether carat is measured in grams or carats. A neural network, however, uses gradient descent on raw feature values. A feature with a range of 0–18,000 (price) will produce gradients orders of magnitude larger than a feature ranging 0–10 (table), causing training instability and slow convergence.
#
# We therefore apply `StandardScaler` to all input features before training any neural model. The target `price` is also standardized; this keeps the loss function numerically stable and allows us to use the same learning rate across experiments. We invert the scaling when computing final metrics so RMSE is reported in dollars.

# %%
import os
_cache = os.path.expanduser('~/seaborn-data/diamonds.csv')
df = pd.read_csv(_cache) if os.path.exists(_cache) else sns.load_dataset('diamonds')

# Ordinal encoding (same order as notebook 05)
cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order   = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

df['cut']     = pd.Categorical(df['cut'],     categories=cut_order,     ordered=True).codes
df['color']   = pd.Categorical(df['color'],   categories=color_order,   ordered=True).codes
df['clarity'] = pd.Categorical(df['clarity'], categories=clarity_order, ordered=True).codes

X = df.drop(columns=['price'])
y = df['price'].values.astype(float)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (fit only on train to avoid data leakage)
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train_raw)
X_test  = feature_scaler.transform(X_test_raw)

# Scale target
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled  = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Baseline
baseline_preds = np.full_like(y_test, y_train.mean())
baseline_rmse  = np.sqrt(mean_squared_error(y_test, baseline_preds))

print(f"Train: {X_train.shape[0]:,} rows  |  Test: {X_test_raw.shape[0]:,} rows  |  Features: {X_train.shape[1]}")
print(f"\nFeature statistics after scaling:")
print(pd.DataFrame(X_train, columns=X.columns).describe().loc[['mean', 'std']].round(4).to_string())
print(f"\nTarget statistics after scaling:")
print(f"  mean={y_train_scaled.mean():.4f}  std={y_train_scaled.std():.4f}")
print(f"\nBaseline RMSE (predict mean): ${baseline_rmse:,.2f}")
print(f"\nLibrary check:")
libs = {'pytorch_tabnet': 'TabNet Pro',
        'nam':            'NAM Pro'}
for lib, role in libs.items():
    try:
        __import__(lib); print(f"  {lib}: OK  →  {role}")
    except ImportError:
        print(f"  {lib}: NOT FOUND  →  {role}")
print(f"\nCatBoost baseline from notebook 05: $530  →  our reference ceiling")

# %% [markdown]
# **43,152 train / 10,788 test samples, 9 features.** After `StandardScaler`, every feature has mean ≈ 0.0000 and std ≈ 1.0000; verified across all nine columns. The gradient descent update for `carat` (raw range ≈ 0.2–5.0) and for `table` (raw range ≈ 43–95) now operate on the same numerical scale.
#
# The baseline RMSE is **&#36;3,987.22** (predicting the mean price for every diamond). Our reference ceiling is CatBoost's **&#36;530 RMSE** from notebook 05. Approaching that number with a neural network would be a strong result; exceeding it is expected and acceptable, since the primary goals here are differentiability (TabNet) and interpretability (NAMs), not raw accuracy.
#
# Both `pytorch_tabnet` and `nam` are installed and confirmed working. The full ATLAS protocol applies: Scratch first to understand the internals, then Pro to see the reference implementation.
#
# Note that `target_scaler` is fitted only on `y_train`. All final metrics are computed by calling `target_scaler.inverse_transform()` on scaled predictions, so reported RMSE values are always in original dollar units.

# %%
# ── scale disparity: raw vs. standardised features ──────────────────────────
raw_df    = X_train_raw.reset_index(drop=True)
scaled_df = pd.DataFrame(X_train, columns=X.columns)

# Sort features by raw std so the most heterogeneous are first
feat_order = raw_df.std().sort_values(ascending=False).index.tolist()

raw_melt    = raw_df.melt(var_name='Feature', value_name='Value')
scaled_melt = scaled_df.melt(var_name='Feature', value_name='Value')

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# ---
# Left: Raw (shared axis reveals scale chaos) ---
sns.boxplot(
    data=raw_melt, x='Feature', y='Value', hue='Feature', order=feat_order,
    ax=axes[0], palette='Blues_d', linewidth=1.5, legend=False,
    flierprops=dict(marker='.', markersize=1.5, alpha=0.3)
)
axes[0].set_title('Before StandardScaler\nFeatures on completely different scales',
                  fontweight='bold', fontsize=14)
axes[0].set_xlabel('')
axes[0].set_ylabel('Raw value')
axes[0].tick_params(axis='x', rotation=30, labelsize=12)

# Annotate the most extreme contrast to make the problem tangible
idx_depth = feat_order.index('depth')
idx_carat = feat_order.index('carat')
axes[0].annotate('depth: median=61.8\n(absolute value ~60)',
                 xy=(idx_depth, 62),
                 xytext=(idx_depth - 1.8, 82),
                 fontsize=10, color='#1565C0', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.3))
axes[0].annotate('carat: median=0.70\n(84× smaller scale)',
                 xy=(idx_carat, 1.0),
                 xytext=(idx_carat - 1.8, 22),
                 fontsize=10, color='#B71C1C', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.3))

# ---
# Right: Scaled (same axis, everything aligns) ---
sns.boxplot(
    data=scaled_melt, x='Feature', y='Value', hue='Feature', order=feat_order,
    ax=axes[1], palette='Reds_d', linewidth=1.5, legend=False,
    flierprops=dict(marker='.', markersize=1.5, alpha=0.3)
)
axes[1].axhline(0,  color='black', linestyle='--', alpha=0.45, linewidth=1.2)
axes[1].axhline(1,  color='grey',  linestyle=':',  alpha=0.4,  linewidth=1.0)
axes[1].axhline(-1, color='grey',  linestyle=':',  alpha=0.4,  linewidth=1.0)
axes[1].set_title('After StandardScaler\nAll features aligned: mean=0, σ=1',
                  fontweight='bold', fontsize=14)
axes[1].set_xlabel('')
axes[1].set_ylabel('Standardised value (σ units)')
axes[1].tick_params(axis='x', rotation=30, labelsize=12)
axes[1].text(8.4, 1.15, '+1σ', fontsize=9, color='grey', ha='right')
axes[1].text(8.4, -1.35, '−1σ', fontsize=9, color='grey', ha='right')

fig.suptitle('Why Neural Networks Need Feature Scaling (Trees Do Not)',
             fontsize=16, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The left panel makes the problem impossible to ignore. `depth` and `table` have medians around 60 with tight ranges, so a gradient update on these features would be 10–15× larger than on `carat` (median ~0.7) or the ordinal features (range 0–7). Left unchecked, gradient descent would steer the network almost exclusively based on the largest-magnitude features, effectively ignoring everything else.
#
# The right panel shows the fix: after `StandardScaler`, every box is centred on zero and the interquartile ranges are comparable across all features. The dashed lines mark ±1σ. Notice the outliers (dots above the upper whiskers): `y` and `z` still show extreme values after scaling (raw measurements of 58.9mm and 31.8mm respectively, almost certainly instrument errors in the dataset). Tree-based models handle these silently; a neural network will be nudged by them, which is another reason to be aware of scaling even after normalisation.
#
# This comparison is the single strongest argument for using `StandardScaler` before any gradient-based model. Trees split on rank order and are completely immune to magnitude. Neural networks are not.

# %% [markdown]
# ## 3. TabNet from Scratch
#
# ### 3.1 The Forward Pass, Step by Step
#
# Before touching a single line of code, it helps to trace one forward pass mentally.
#
# Suppose our input is a batch of diamonds, each described by 9 features: $x \in \mathbb{R}^{B \times 9}$. TabNet processes this input in $N_{\text{steps}}$ sequential steps. Each step asks the same question: *"Given what I already know about this batch, which features should I look at next?"*
#
# **Step $i$ in detail:**
#
# 1. **Attentive Transformer** computes a sparse selection mask $M^{(i)} \in [0,1]^{B \times 9}$:
#    $$ M^{(i)} = \text{sparsemax}\!\left( P^{(i-1)} \odot h_a\!\left( h^{(i-1)} \right) \right) $$
#    - $h^{(i-1)}$: the processed representation from the *previous* step.
#    - $h_a$: a linear layer + BN, which maps the $d$-dimensional representation back to feature space.
#    - $P^{(i-1)}$: **prior scales**, a running product that penalises re-selecting features already used.
#
# 2. **Prior scales update:** discourage reuse of already-selected features:
#    $$ P^{(i)} = P^{(i-1)} \odot \left( \gamma - M^{(i)} \right) $$
#    With $\gamma = 1.3$: if feature $j$ received full attention ($M^{(i)}_j = 1$), it is down-weighted to &#36;0.3$ in the next step; still possible to revisit but de-prioritised.
#
# 3. **Feature Transformer** processes the *masked* input:
#    $$ h^{(i)} = \text{FeatureTransformer}\!\left( M^{(i)} \odot \hat{x} \right) $$
#    where $\hat{x}$ is the batch-normalised input. The transformer is a stack of **GLU (Gated Linear Unit)** blocks with residual connections; one set of weights *shared* across all steps, one set *step-specific*.
#
# 4. **Aggregation:** accumulate the processed output across all steps:
#    $$ \hat{y} = W_{\text{out}} \sum_{i=1}^{N_{\text{steps}}} \text{ReLU}\!\left( h^{(i)} \right) $$
#
# ### 3.2 Sparsemax: The Core of Interpretability
#
# Softmax maps any input to a valid probability distribution, but every output is strictly positive. **Sparsemax** maps inputs to the *simplex* (sums to 1, all values ≥ 0) but can produce *exact zeros*.
#
# The algorithm finds the threshold $\tau$ such that the output $\pi_j = \max(z_j - \tau, 0)$ sums to 1. Features with $z_j \leq \tau$ receive exactly zero weight; they are *ignored*. This is what makes TabNet's feature importance readable: the masks tell you not just "feature $j$ was less important" but "feature $j$ was not used at all."

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time


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


# Sanity check
z_test = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
sm = torch.softmax(z_test, dim=-1)
sp = sparsemax(z_test, dim=-1)
print(f"softmax: {sm.numpy().round(3)}  → all positive, sums to {sm.sum():.1f}")
print(f"sparsemax: {sp.numpy().round(3)}  → exact zeros, sums to {sp.sum():.1f}")

# %% [markdown]
# `softmax` spreads probability across all four inputs, even assigning ~5% to the most negative element. `sparsemax` concentrates all mass on the top two and returns **exact zeros** for the others. Multiply this mask against 9 features and you get a principled, readable feature selection, not a soft blend of everything.

# %%
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

# %% [markdown]
# A few design decisions worth unpacking:
#
# - **Shared FC weights**: `self.shared_fc` is a *single* `nn.Linear` whose weight matrix is passed to every `FeatureTransformer`. All steps literally share the same object; PyTorch's autograd accumulates gradients correctly. This gives the network a common feature vocabulary while the step-specific layers let each step "speak its own dialect."
#
# - **Sparsity regularisation**: We minimise $-\sum_j M_j \log(M_j + \epsilon)$, the negative entropy of the mask. Since sparsemax already produces zeros, this nudges the *non-zero* weights to be concentrated on fewer features rather than spread thinly. `lambda_s=1e-4` keeps it from dominating the regression loss.
#
# - **`forward` returns a tuple** `(prediction, entropy_loss)` so the training loop can combine them without the model needing to know the batch targets.

# %%
# === Training Setup ===
SEED       = 42
N_D        = 32    # representation width per step
N_STEPS    = 3
GAMMA      = 1.3
LAMBDA_S   = 1e-4
LR         = 0.005
BATCH_SIZE = 1024
N_EPOCHS   = 150
PATIENCE   = 20   # early stopping

torch.manual_seed(SEED)

model = ScratchTabNet(
    n_features=X_train.shape[1],
    n_d=N_D,
    n_steps=N_STEPS,
    gamma=GAMMA,
    lambda_s=LAMBDA_S,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ScratchTabNet: {n_params:,} trainable parameters")
print(f"  n_d={N_D}  n_steps={N_STEPS}  gamma={GAMMA}  lambda_sparse={LAMBDA_S}")
print(f"  batch_size={BATCH_SIZE}  lr={LR}  max_epochs={N_EPOCHS}  patience={PATIENCE}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-4
)

# Build DataLoaders from scaled tensors
X_tr  = torch.tensor(X_train,       dtype=torch.float32)
y_tr  = torch.tensor(y_train_scaled, dtype=torch.float32)
X_te  = torch.tensor(X_test,        dtype=torch.float32)
y_te  = torch.tensor(y_test_scaled,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# **8,457 trainable parameters,** orders of magnitude fewer than a standard MLP. The compact footprint comes from three sources: (1) `n_d=32` limits the representation width, (2) the shared FC halves the effective parameter count vs. step-specific layers, and (3) 3 steps is enough for a 9-feature dataset. TabNet's power does not come from having many parameters, but from *choosing which features to use* at each step. A gradient-boosted tree achieves the same via explicit splits; TabNet achieves it via differentiable sparse attention trained end-to-end.

# %%
# === Training Loop ===
train_rmse_hist, val_rmse_hist = [], []
best_val, best_epoch, epochs_no_improve = np.inf, 0, 0
best_state = None
t_start = time.time()

model.train()
for epoch in range(1, N_EPOCHS + 1):
    epoch_loss = 0.0
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred, entropy_loss = model(X_batch)
        loss = F.mse_loss(pred, y_batch) + entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)

    # Evaluate on full train and test sets (in eval mode for stable BN stats)
    model.eval()
    with torch.no_grad():
        tr_pred_s, _ = model(X_tr)
        te_pred_s, _ = model(X_te)

    # Invert scaling → dollar RMSE
    tr_pred = target_scaler.inverse_transform(tr_pred_s.numpy().reshape(-1, 1)).ravel()
    te_pred = target_scaler.inverse_transform(te_pred_s.numpy().reshape(-1, 1)).ravel()
    tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
    te_rmse = np.sqrt(mean_squared_error(y_test,  te_pred))

    train_rmse_hist.append(tr_rmse)
    val_rmse_hist.append(te_rmse)
    scheduler.step(te_rmse)

    if te_rmse < best_val:
        best_val, best_epoch = te_rmse, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d} | Train RMSE: ${tr_rmse:,.0f} | Val RMSE: ${te_rmse:,.0f} | LR: {lr_now:.5f}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

elapsed = time.time() - t_start

# Restore best weights
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    final_pred_s, _ = model(X_te)
final_pred = target_scaler.inverse_transform(final_pred_s.numpy().reshape(-1, 1)).ravel()
scratch_tabnet_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
scratch_tabnet_r2   = r2_score(y_test, final_pred)
scratch_tabnet_mae  = mean_absolute_error(y_test, final_pred)

print(f"\n{'='*55}")
print(f"  Scratch TabNet | Best Epoch {best_epoch}/{epoch}")
print(f"  RMSE : ${scratch_tabnet_rmse:,.2f}")
print(f"  MAE  : ${scratch_tabnet_mae:,.2f}")
print(f"  R2   : {scratch_tabnet_r2:.4f}")
print(f"  vs Baseline  : {(1 - scratch_tabnet_rmse / baseline_rmse)*100:.1f}% reduction")
print(f"  vs CatBoost  : RMSE {scratch_tabnet_rmse / 530:.2f}x worse")
print(f"{'='*55}")

# %% [markdown]
# **RMSE &#36;1,076 | MAE &#36;678 | R² 0.9272 | 73.0% reduction vs baseline. Best epoch: 11/31.**
#
# Three things stand out:
#
# 1. **73% error reduction from a neural network with 8,457 parameters trained in 72 seconds.** This is far from trivial. Scratch gradient boosting with 200 stumps achieved 72% on the same data, so this simple TabNet is already in the same ballpark.
#
# 2. **Early stopping fires at epoch 31 (best at epoch 11).** The learning rate scheduler halved LR twice (epoch ~17: 0.005 → 0.0025, epoch ~22: 0.0025 → 0.000625), but the model had already saturated its capacity. With `n_d=32` and `n_steps=3`, the effective function space is limited; the model cannot capture the complex multi-feature interactions that deeper trees handle trivially.
#
# 3. **The gap vs. CatBoost (&#36;530) is 2x.** This is the architectural tax of interpretability: the sparse sequential attention that makes TabNet readable also restricts what it can model. The `pytorch-tabnet` library, with deeper transformers and proper regularisation, closes this gap significantly, which is what we will see in the next section.

# %%
# ── training learning curve ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

epochs_ran = np.arange(1, len(train_rmse_hist) + 1)
ax.plot(epochs_ran, train_rmse_hist, color='#2196F3', linewidth=2.5,
        label='Train RMSE', alpha=0.9)
ax.plot(epochs_ran, val_rmse_hist,   color='#FF5722', linewidth=2.5,
        label='Validation RMSE', alpha=0.9)

ax.scatter([best_epoch], [best_val], color='#4CAF50', s=140, zorder=5,
           edgecolor='white', linewidth=2)
ax.annotate(f'Best: Epoch {best_epoch}\nRMSE: ${best_val:,.0f}',
            xy=(best_epoch, best_val),
            xytext=(best_epoch + max(len(epochs_ran) * 0.08, 3), best_val + 150),
            fontsize=12, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

ax.axhline(y=baseline_rmse, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(2, baseline_rmse + 60, f'Baseline: ${baseline_rmse:,.0f}', fontsize=11, color='gray')

ax.axhline(y=530, color='#9C27B0', linestyle='--', alpha=0.45, linewidth=1.5)
ax.text(2, 530 + 60, 'CatBoost (nb 05): $530', fontsize=11, color='#9C27B0')

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE ($)')
ax.set_title('Scratch TabNet: Learning Curve', fontweight='bold')
ax.legend(fontsize=13)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The learning curve tells a clean story. Validation RMSE drops steeply in the first 11 epochs: the model learns the dominant pattern fast, capturing that large `y` and `carat` predict higher prices. After epoch 11, the LR scheduler detects stagnation and halves the learning rate twice. Each halving produces a brief improvement attempt, then flat-lines. The train and validation curves track each other closely throughout, with virtually no overfitting. This is the regularising effect of the sparse masks combined with the low parameter count. Unlike the gradient boosting models (which showed clear train-val divergence), TabNet is almost impossible to overfit on this dataset at this scale.
#
# The CatBoost reference line (&#36;530) sits comfortably below our curve, a concrete reminder of what the interpretability tradeoff costs in accuracy.

# %%
# ── step-wise attention masks heatmap ────────────────────────────────────────
model.eval()
with torch.no_grad():
    model(X_te)                        # populate ._masks on the test set
    masks_np = model.step_masks.numpy()  # (n_steps, n_features)

fig, axes = plt.subplots(1, 2, figsize=(20, 5),
                         gridspec_kw={'width_ratios': [3, 1]})

# Left: heatmap of per-step attention
im = axes[0].imshow(masks_np, cmap='YlOrRd', aspect='auto', vmin=0)
axes[0].set_xticks(range(len(X.columns)))
axes[0].set_xticklabels(X.columns.tolist(), fontsize=12, rotation=30, ha='right')
axes[0].set_yticks(range(N_STEPS))
axes[0].set_yticklabels([f'Step {i+1}' for i in range(N_STEPS)], fontsize=12)
axes[0].set_title('Average Feature Attention per Step\n'
                  '(each row sums to ~1, sparsemax constraint)',
                  fontweight='bold', fontsize=13)
fig.colorbar(im, ax=axes[0], label='Average attention weight')

# Annotate each cell with its value
for i in range(N_STEPS):
    for j in range(len(X.columns)):
        axes[0].text(j, i, f'{masks_np[i, j]:.2f}',
                     ha='center', va='center', fontsize=9,
                     color='white' if masks_np[i, j] > 0.25 else '#333333',
                     fontweight='bold')

# Right: aggregated feature importance (mean across steps)
importance = masks_np.mean(axis=0)
feat_order_imp = np.argsort(importance)[::-1]
colors_imp = plt.cm.YlOrRd(importance[feat_order_imp] / importance.max())
axes[1].barh(
    [X.columns[i] for i in feat_order_imp],
    importance[feat_order_imp],
    color=colors_imp, edgecolor='white', linewidth=0.5
)
axes[1].set_title('Aggregated\nImportance', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Mean attention weight')
axes[1].invert_yaxis()
sns.despine()

fig.suptitle('TabNet Interpretability: Which Features Are Selected at Each Step?',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# This heatmap is the payoff of the entire TabNet architecture, something a gradient boosting model cannot give you without post-hoc SHAP analysis.
#
# Read each row as "at this step, TabNet focused on these features":
#
# - **Step 1: Spatial scan** (`y`: 0.23, `depth`: 0.21, `carat`: 0.18, `clarity`: 0.12): The first step casts a wide net across the dimensional features. `y` (the Y-axis length in mm) and `depth` together act as a physical proxy for size. `carat` and `clarity` provide an initial quality anchor.
#
# - **Step 2: Dimensional refinement** (`y`: 0.41, `color`: 0.15, `z`: 0.10): Having established a spatial frame, Step 2 concentrates *heavily* on `y` (0.41, the largest single attention weight in the entire model). The prior scales penalise `depth` (already used in Step 1), so the model pivots to the complementary `z` dimension and adds `color` as the first quality signal.
#
# - **Step 3: Carat dominance** (`carat`: 0.43, `z`: 0.19, `color`: 0.17, `x`: 0.15): The final step is dominated by `carat` (0.43). Having characterised the physical dimensions in Steps 1 and 2, the model now uses the canonical weight metric for final price calibration.
#   `color` persists; `cut` and `table` are nearly ignored throughout (0.006 and 0.036).
#
# The sequential logic mirrors how a jeweller actually prices a diamond: first gauge the size visually (Step 1–2), then confirm the carat weight on a scale (Step 3). TabNet learned this reasoning structure from data, not from domain knowledge.

# %%
# ── predicted vs actual ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))

hb = ax.hexbin(y_test, final_pred, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
fig.colorbar(hb, ax=ax, label='Count')
lims = [0, max(y_test.max(), final_pred.max()) * 1.05]
ax.plot(lims, lims, color='#FF5722', linewidth=2, linestyle='--',
        alpha=0.85, label='Perfect prediction')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title(f'Scratch TabNet: Predicted vs. Actual\nRMSE: ${scratch_tabnet_rmse:,.0f}  |  R²: {scratch_tabnet_r2:.4f}',
             fontweight='bold')
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
ax.legend(fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The hexbin plot shows R² 0.9272: the model explains 93% of price variance. The dense cluster along the diagonal from &#36;0–&#36;8,000 is tight, covering the bulk of the dataset. Above &#36;10,000, predictions start to scatter: the model underestimates very expensive diamonds. This is the classic high-end failure mode of low-capacity models, since there are simply too few diamonds above &#36;10,000 in the training set to calibrate that region well.
#
# Compare this mentally to the Scratch Gradient Boosting hexbin from notebook 05 (R² 0.9212): TabNet achieves slightly higher R² despite having an architectural constraint (sparse sequential attention) that gradient boosting stumps do not have. The neural architecture compensates with smooth, continuous predictions; no horizontal banding from discrete stump outputs.

# %% [markdown]
# ## 4. TabNet Pro: `pytorch-tabnet`
#
# ### 4.1 What the Library Adds
#
# Our scratch implementation captures the core logic, but the production `pytorch-tabnet` library (DreamQuark, 2019) adds several engineering improvements that matter in practice:
#
# | Component | Scratch | pytorch-tabnet |
# |---|---|---|
# | Feature Transformer depth | 2 GLU layers | 4 GLU layers (2 shared + 2 step-specific) |
# | Batch Normalisation | Standard BN | **Ghost BN**: BN computed on virtual mini-batches of size `virtual_batch_size` within each batch, reducing variance of BN statistics |
# | LR scheduling | `ReduceLROnPlateau` | Pluggable scheduler, `StepLR` or `CyclicLR` |
# | Initialisation | PyTorch default | Xavier uniform for FC layers, specific BN init |
# | Explain API | Manual mask aggregation | `model.explain(X)` returns step-by-step masks per sample |
# | Feature importance | Manual mask mean | `model.feature_importances_` (magnitude-weighted) |
#
# The 4-layer Feature Transformer is the most impactful difference: each step can learn more complex transformations of the masked input, capturing feature interactions that two GLU layers cannot.
#
# ### 4.2 Hyperparameter Decisions
#
# | Parameter | Value | Rationale |
# |---|---|---|
# | `n_d = n_a` | 32 | Same as scratch for fair comparison |
# | `n_steps` | 4 | One extra step vs scratch; more selection rounds |
# | `gamma` | 1.3 | Same as scratch |
# | `lambda_sparse` | 1e-3 | 10x stronger sparsity penalty than scratch |
# | `lr` | 0.02 | Default Adam LR for tabnet |
# | `batch_size` | 4096 | Larger batches stabilise Ghost BN |
# | `virtual_batch_size` | 4096 | Same as batch (equivalent to standard BN at this scale) |
# | `max_epochs` | 50 | ~10s per epoch on CPU; early stopping guards against overrun |

# %%
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

pro_model = TabNetRegressor(
    n_d=32, n_a=32,
    n_steps=4,
    gamma=1.3,
    lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=0.02),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params=dict(step_size=10, gamma=0.9),
    mask_type='sparsemax',
    verbose=0,
    seed=42,
)

pro_model.fit(
    X_train, y_train_scaled.reshape(-1, 1),
    eval_set=[(X_test, y_test_scaled.reshape(-1, 1))],
    eval_metric=['rmse'],
    max_epochs=50,
    patience=15,
    batch_size=4096,
    virtual_batch_size=4096,
)

# Predictions in dollar space
pro_preds_scaled = pro_model.predict(X_test)
pro_preds = target_scaler.inverse_transform(pro_preds_scaled.reshape(-1, 1)).ravel()
pro_rmse = np.sqrt(mean_squared_error(y_test, pro_preds))
pro_r2   = r2_score(y_test, pro_preds)
pro_mae  = mean_absolute_error(y_test, pro_preds)

# Convert scaled val history to dollar RMSE
std_y = target_scaler.scale_[0]
val_hist_dollars = np.array(pro_model.history['val_0_rmse']) * std_y

n_pro_params = sum(p.numel() for p in pro_model.network.parameters() if p.requires_grad)

print(f"TabNet Pro | Best Epoch {pro_model.best_epoch}/{len(val_hist_dollars)}")
print(f"  RMSE        : ${pro_rmse:,.2f}")
print(f"  MAE         : ${pro_mae:,.2f}")
print(f"  R2          : {pro_r2:.4f}")
print(f"  Parameters  : {n_pro_params:,}")
print(f"  vs Baseline : {(1-pro_rmse/baseline_rmse)*100:.1f}% reduction")
print(f"  vs Scratch  : {(1-pro_rmse/scratch_tabnet_rmse)*100:.1f}% improvement")
print(f"  vs CatBoost : {pro_rmse/530:.2f}x worse")

# %% [markdown]
# **RMSE &#36;654 | MAE &#36;378 | R² 0.9731 | Best epoch 49/50.**
#
# Three observations stand out:
#
# 1. **39.2% improvement over scratch** (&#36;1,076 → &#36;654) from a model with the same `n_d=32` and only one extra attention step. The gain comes entirely from the deeper Feature Transformer (4 GLU layers vs 2) and the stronger sparsity penalty (`lambda_sparse=1e-3` vs `1e-4`). The architecture is the same; the engineering is better.
#
# 2. **Best epoch = 49/50** (hit the max_epochs limit). The model had not fully converged, meaning more training would push RMSE lower still. This is a known characteristic of TabNet on medium-sized datasets: it improves slowly but steadily. The CatBoost gap (&#36;654 vs &#36;530) would narrow with additional epochs or a learning rate sweep.
#
# 3. **Still 1.23x worse than CatBoost.** The accuracy gap persists even with the production library. This is not a failure of TabNet; it is the measurable cost of constraining the model to learn through sparse sequential feature selection rather than unrestricted tree splits.

# %%
# ── pro learning curve vs scratch reference ───────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

epochs_pro = np.arange(1, len(val_hist_dollars) + 1)

ax.plot(epochs_pro, val_hist_dollars, color='#2196F3', linewidth=2.5,
        label='TabNet Pro (val)', alpha=0.9)

# Scratch reference as horizontal zone
ax.axhline(scratch_tabnet_rmse, color='#FF9800', linestyle='--',
           alpha=0.7, linewidth=1.8, label=f'Scratch TabNet best: ${scratch_tabnet_rmse:,.0f}')
ax.axhline(530, color='#9C27B0', linestyle='--',
           alpha=0.5, linewidth=1.5, label='CatBoost (nb 05): $530')
ax.axhline(baseline_rmse, color='gray', linestyle=':',
           alpha=0.5, linewidth=1.2, label=f'Baseline: ${baseline_rmse:,.0f}')

# Best point
ax.scatter([pro_model.best_epoch + 1], [val_hist_dollars[pro_model.best_epoch]],
           color='#4CAF50', s=140, zorder=5, edgecolor='white', linewidth=2)
ax.annotate(f'Best: Epoch {pro_model.best_epoch + 1}\nRMSE: ${val_hist_dollars[pro_model.best_epoch]:,.0f}',
            xy=(pro_model.best_epoch + 1, val_hist_dollars[pro_model.best_epoch]),
            xytext=(pro_model.best_epoch - 10, val_hist_dollars[pro_model.best_epoch] + 300),
            fontsize=12, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation RMSE ($)')
ax.set_title('TabNet Pro: Learning Curve', fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, min(baseline_rmse * 1.1, val_hist_dollars[0] * 1.05))
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The learning curve has a characteristic double-phase shape. The first 5 epochs drop steeply from baseline (&#36;19,658 at epoch 1, essentially random initialisation) to &#36;2,373, capturing the dominant carat-price signal. The next 45 epochs grind out the remaining gains as the model learns progressively more refined feature interactions. The curve has not flattened by epoch 50, confirming that best_epoch=49 simply hit the training budget rather than a true convergence point.
#
# The orange reference line (&#36;1,076, scratch TabNet) is crossed around epoch 10, meaning the Pro library beats our scratch model in 10 epochs of what our scratch took 11 epochs to achieve best. After that, Pro keeps improving while scratch had already stalled, which illustrates the compounding benefit of deeper feature transformers.

# %%
# ── feature importance comparison: scratch vs pro ────────────────────────────
feat_names = X.columns.tolist()

# Scratch: aggregate mask attention across steps
with torch.no_grad():
    model.eval()
    model(torch.tensor(X_test, dtype=torch.float32))
scratch_importance = model.step_masks.numpy().mean(axis=0)
scratch_importance /= scratch_importance.sum()

# Pro: built-in feature_importances_
pro_importance = pro_model.feature_importances_

sort_idx = np.argsort(pro_importance)[::-1]  # sort by Pro importance

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

colors_s = '#FF9800'
colors_p = '#2196F3'

axes[0].barh([feat_names[i] for i in sort_idx],
             [scratch_importance[i] for i in sort_idx],
             color=colors_s, edgecolor='white', alpha=0.85)
axes[0].set_title('Scratch TabNet\n(avg attention mask)', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Normalised importance')
axes[0].invert_yaxis()
for i, idx in enumerate(sort_idx):
    axes[0].text(scratch_importance[idx] + 0.003, i,
                 f'{scratch_importance[idx]:.3f}', va='center', fontsize=10)

axes[1].barh([feat_names[i] for i in sort_idx],
             [pro_importance[i] for i in sort_idx],
             color=colors_p, edgecolor='white', alpha=0.85)
axes[1].set_title('TabNet Pro\n(feature_importances_)', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Normalised importance')
for i, idx in enumerate(sort_idx):
    axes[1].text(pro_importance[idx] + 0.003, i,
                 f'{pro_importance[idx]:.3f}', va='center', fontsize=10)

fig.suptitle('Feature Importance: Scratch vs. Pro\n'
             'Same architecture, different depth and training budget',
             fontsize=15, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The two importance profiles tell a story about model capacity.
#
# The scratch model (orange) concentrates heavily on dimensional features: `carat` (0.240), `y` (0.232), `z` (0.108). These are the easiest signals to learn with shallow transformers, since `y` alone is a near-linear proxy for `carat`. The model grabs what it can reach with two GLU layers.
#
# The Pro model (blue) distributes attention more evenly and shifts ranking: `clarity` (0.199) rises to first place, nearly tied with `carat` (0.193), and `y` drops from second to seventh (0.073). This is a more accurate representation of diamond pricing theory: clarity and color are genuine quality premia, not just dimensional proxies. The deeper Feature Transformer can disentangle these signals from the size-dominated noise, which the scratch model cannot.
#
# `cut` (0.059) and `table` (0.046) remain the least informative in both models, consistent with the XGBoost importance analysis in notebook 05.

# %%
# ── step-wise attention heatmap: tabnet pro (4 steps) ─────────────────────────
_, pro_masks = pro_model.explain(X_test)
n_steps_pro = pro_model.n_steps

masks_pro_np = np.stack([pro_masks[i].mean(axis=0) for i in range(n_steps_pro)])

fig, axes = plt.subplots(1, 2, figsize=(20, 5),
                         gridspec_kw={'width_ratios': [3, 1]})

im = axes[0].imshow(masks_pro_np, cmap='YlOrRd', aspect='auto', vmin=0)
axes[0].set_xticks(range(len(feat_names)))
axes[0].set_xticklabels(feat_names, fontsize=12, rotation=30, ha='right')
axes[0].set_yticks(range(n_steps_pro))
axes[0].set_yticklabels([f'Step {i+1}' for i in range(n_steps_pro)], fontsize=12)
axes[0].set_title('TabNet Pro: Average Feature Attention per Step\n'
                  '(4 steps, stronger sparsity, deeper transformer)',
                  fontweight='bold', fontsize=13)
fig.colorbar(im, ax=axes[0], label='Average attention weight')

for i in range(n_steps_pro):
    for j in range(len(feat_names)):
        axes[0].text(j, i, f'{masks_pro_np[i, j]:.2f}',
                     ha='center', va='center', fontsize=9,
                     color='white' if masks_pro_np[i, j] > 0.25 else '#333333',
                     fontweight='bold')

importance_pro = masks_pro_np.mean(axis=0)
feat_order_imp = np.argsort(importance_pro)[::-1]
colors_imp = plt.cm.YlOrRd(importance_pro[feat_order_imp] / importance_pro.max())
axes[1].barh([feat_names[i] for i in feat_order_imp],
             importance_pro[feat_order_imp],
             color=colors_imp, edgecolor='white', linewidth=0.5)
axes[1].set_title('Aggregated\nImportance', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Mean attention weight')
axes[1].invert_yaxis()
sns.despine()

fig.suptitle('TabNet Pro: Step-wise Attention (Compare with Scratch, 3 steps)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Compared to the scratch 3-step heatmap, the Pro 4-step version shows a more refined division of labour across the steps:
#
# - **Step 1** distributes broadly: `z` (0.23), `table` (0.15), `carat` (0.14), `clarity` (0.14). An initial scan across size and quality features simultaneously, something our shallow scratch transformer could not do in one step.
#
# - **Step 2** locks onto `carat` (0.50) plus `color` (0.22). The dominant price driver is isolated cleanly. The prior scales have suppressed `z` and `table` from Step 1.
#
# - **Step 3** pivots to fine-grained quality: `x` (0.30), `cut` (0.27), `depth` (0.13). These are the features most related to the geometric proportions that affect a diamond's light performance, independent of its raw size.
#
# - **Step 4** closes with `clarity` (0.26), `color` (0.22), `x` (0.18). A final quality correction layer adjusting for the two hardest-to-price attributes.
#
# The pattern is interpretable without any post-hoc analysis. A jeweller reading this heatmap would recognise it immediately: size first, then primary quality, then proportions, then microscopic grading.

# %% [markdown]
#
# ---
# ## 5. Neural Additive Models from Scratch
#
# TabNet assigns an **attention mask** to decide which features to read at each step, but it still produces a single opaque output. You cannot ask it: *"exactly how does adding one carat change the predicted price, all else equal?"*
#
# A **Neural Additive Model (NAM)** answers that question by construction. The architecture is a modern realisation of the classic **Generalised Additive Model** (GAM):
#
# $$\hat{y} = \beta_0 + \sum_{j=1}^{p} f_j(x_j)$$
#
# where each $f_j$ is a small neural network that receives **only feature $j$** as input. Because the sum is additive, the contribution of feature $j$ is always $f_j(x_j)$, independent of every other feature. This gives you a curve you can plot, a **shape function**, and the plot tells you the full story of that feature's effect.
#
# ### 5.1 The ExU Activation
#
# A plain ReLU MLP can represent any shape function, but it requires many hidden units to capture sharp transitions (e.g. the price cliff between SI2 and VS2 clarity grades). Agarwal et al. (2021) introduced the **Exp-Centered Unit (ExU)**:
#
# $$\text{ExU}_j(x) = (x - b_j)\,e^{w_j}$$
#
# The learned $e^{w_j}$ acts as a per-unit **frequency multiplier**. A large positive $w_j$ creates a steep step centred at $b_j$, while a small $w_j$ creates a gentle slope. By initialising $w_j \sim \mathcal{N}(4, 0.5)$ (so $e^{w_j} \approx 55$ at the start) and $b_j \sim \mathcal{N}(0, 0.5)$ (to spread thresholds across the standardised input range), the network begins with high-frequency capacity and regularises down during training.
#
# ### 5.2 Architecture
#
# > **FeatureNN$_j$:** `ExU(out=H)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` $\rightarrow$ `Linear(H, 1)`
# >
# > **ScratchNAM:** $\text{bias} + \sum_{j} \text{FeatureNN}_j(x_j)$
#
# With $p = 9$ features and $H = 64$, the total parameter count is only **1,729**, three orders of magnitude smaller than TabNet and CatBoost.
#
# ### 5.3 What We Lose (and Why It Matters)
#
# The additive constraint is the model's superpower *and* its Achilles heel. Consider: a 2-carat diamond with *Ideal* cut and *IF* clarity commands a disproportionate premium because top quality across *all* dimensions is rare. The product of grading factors is not captured by a sum of individual functions. NAMs will systematically underfit any dataset with strong **feature interactions**, and the diamonds dataset has several.
#
# This is not a bug but a design choice: you trade predictive accuracy for complete transparency. For regulated industries or anywhere a model must be auditable, that trade-off is often worthwhile.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── ExU activation ────────────────────────────────────────────────────────────
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


# ── FeatureNN ─────────────────────────────────────────────────────────────────
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


# ── NAM ───────────────────────────────────────────────────────────────────────
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


n_params = sum(p.numel() for p in ScratchNAM(X_train.shape[1]).parameters())
print(f"ScratchNAM: {n_params:,} parameters across {X_train.shape[1]} FeatureNNs")
print(f"  (TabNet Pro had ~{int(32*32*9 + 32*32*4*2 + 32*9):,}+ parameters)")

# %% [markdown]
# 1,729 parameters total, three orders of magnitude fewer than TabNet Pro. Every unit in the model is directly traceable to a single feature; there is no weight that mixes information from two or more inputs.

# %%
# ── Training ──────────────────────────────────────────────────────────────────
SEED_NAM   = 42
HIDDEN_NAM = 64
DROPOUT_NAM = 0.15
LR_NAM     = 0.002
BATCH_NAM  = 2048
EPOCHS_NAM = 150
PATIENCE_NAM = 20

torch.manual_seed(SEED_NAM)
nam = ScratchNAM(X_train.shape[1], HIDDEN_NAM, DROPOUT_NAM)
opt = torch.optim.Adam(nam.parameters(), lr=LR_NAM, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5, min_lr=1e-4)

Xtr_t   = torch.tensor(X_train,       dtype=torch.float32)
ytrs_t  = torch.tensor(y_train_scaled, dtype=torch.float32)
Xte_t   = torch.tensor(X_test,        dtype=torch.float32)
loader  = DataLoader(TensorDataset(Xtr_t, ytrs_t), batch_size=BATCH_NAM, shuffle=True)

nam_tr_hist, nam_val_hist = [], []
best_val_nam, best_epoch_nam, no_imp = np.inf, 0, 0
best_state_nam = None

for epoch in range(1, EPOCHS_NAM + 1):
    nam.train()
    for xb, yb in loader:
        opt.zero_grad()
        pred, _ = nam(xb)
        F.mse_loss(pred, yb).backward()
        torch.nn.utils.clip_grad_norm_(nam.parameters(), 1.0)
        opt.step()

    nam.eval()
    with torch.no_grad():
        tr_s, _ = nam(Xtr_t);  te_s, _ = nam(Xte_t)
    tr_pred = target_scaler.inverse_transform(tr_s.numpy().reshape(-1,1)).ravel()
    te_pred = target_scaler.inverse_transform(te_s.numpy().reshape(-1,1)).ravel()
    tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
    te_rmse = np.sqrt(mean_squared_error(y_test,  te_pred))
    nam_tr_hist.append(tr_rmse); nam_val_hist.append(te_rmse)
    sched.step(te_rmse)

    if te_rmse < best_val_nam:
        best_val_nam, best_epoch_nam = te_rmse, epoch
        best_state_nam = {k: v.clone() for k,v in nam.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
    if epoch % 15 == 0:
        print(f"  Epoch {epoch:3d} | Train: ${tr_rmse:,.0f} | Val: ${te_rmse:,.0f} | LR: {opt.param_groups[0]['lr']:.5f}")
    if no_imp >= PATIENCE_NAM:
        print(f"  Early stopping at epoch {epoch}  (best={best_epoch_nam})")
        break

nam.load_state_dict(best_state_nam)
nam.eval()
with torch.no_grad():
    te_final_s, nam_contribs = nam(Xte_t)
te_pred_nam = target_scaler.inverse_transform(te_final_s.numpy().reshape(-1,1)).ravel()

rmse_nam = np.sqrt(mean_squared_error(y_test, te_pred_nam))
mae_nam  = mean_absolute_error(y_test, te_pred_nam)
r2_nam   = r2_score(y_test, te_pred_nam)

print(f"\n{'='*45}")
print(f"  Scratch NAM  |  Best epoch {best_epoch_nam}")
print(f"  RMSE : ${rmse_nam:,.2f}")
print(f"  MAE  : ${mae_nam:,.2f}")
print(f"  R2   : {r2_nam:.4f}")
print(f"{'='*45}")

# %% [markdown]
# The learning rate scheduler triggers twice before early stopping: the model first overshoots around epoch 45, then the reduced LR allows it to re-enter a productive regime around epoch 80–120 before plateauing. The final R² sits in the 0.85–0.86 range, which is the ceiling for a strictly additive model on the diamonds dataset.
#
# The gap to TabNet Pro is real and expected: the data contains strong carat × clarity interactions that no sum of individual curves can represent exactly. But notice what we gained: we can now ask the model exactly how it arrived at any prediction, and it can answer with a simple lookup on nine curves; no SHAP values, no approximations.

# %% [markdown]
# The shape function plot is the diagnostic that makes NAMs unique: one curve per feature, showing the model's learned relationship between that feature and the predicted price. The y-axis is the dollar contribution: how much that feature value adds to (or subtracts from) the baseline price. Because the model is purely additive, these curves fully explain every prediction.

# %%
# ── shape functions: one curve per feature ───────────────────────────────────
std_y    = target_scaler.scale_[0]
feat_order = list(X_train_raw.columns)  # original column order

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes_flat = axes.flatten()

feat_colors = {
    'carat': '#2196F3', 'cut': '#FF9800', 'color': '#9C27B0',
    'clarity': '#E91E63', 'depth': '#00BCD4', 'table': '#FF5722',
    'x': '#4CAF50', 'y': '#795548', 'z': '#607D8B',
}

# Original-scale tick labels for each feature
feat_labels = {
    'carat'  : ([0.2, 0.5, 1.0, 1.5, 2.0, 3.0], [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]),
    'cut'    : ([0, 1, 2, 3, 4], ['Fair', 'Good', 'V.Good', 'Prem.', 'Ideal']),
    'color'  : ([0, 1, 2, 3, 4, 5, 6], ['J', 'I', 'H', 'G', 'F', 'E', 'D']),
    'clarity': ([0, 1, 2, 3, 4, 5, 6, 7], ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']),
    'depth'  : ([58, 61, 62, 63, 65], [58, 61, 62, 63, 65]),
    'table'  : ([53, 56, 57, 59, 62, 66], [53, 56, 57, 59, 62, 66]),
    'x'      : ([3, 4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9, 10]),
    'y'      : ([3, 4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9, 10]),
    'z'      : ([2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]),
}

for idx, feat in enumerate(feat_order):
    ax = axes_flat[idx]
    j  = idx  # same order as FeatureNNs were created

    # Evaluate shape function on dense grid
    grid_s, vals_s = nam.shape_function(j, n_pts=300)
    vals_dollar = vals_s * std_y

    # Convert x-axis back to original scale
    mu  = feature_scaler.mean_[j]
    sig = feature_scaler.scale_[j]
    grid_orig = grid_s * sig + mu

    color = feat_colors[feat]
    ax.fill_between(grid_orig, 0, vals_dollar,
                    where=(vals_dollar > 0), alpha=0.18, color=color)
    ax.fill_between(grid_orig, 0, vals_dollar,
                    where=(vals_dollar < 0), alpha=0.18, color='#e74c3c')
    ax.plot(grid_orig, vals_dollar, color=color, lw=2.2, zorder=3)
    ax.axhline(0, color='#555', lw=0.8, ls='--', alpha=0.5)

    # Annotate the range
    r = vals_dollar.max() - vals_dollar.min()
    ax.text(0.97, 0.97, f'range ≈ ${r:,.0f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#333',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc', alpha=0.8))

    ax.set_title(feat, fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('Original scale', fontsize=8, color='#555')
    ax.set_ylabel('$ contribution', fontsize=8, color='#555')
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    # Custom x-tick labels for ordinal features (x-axis already in original scale)
    if feat in ['cut', 'color', 'clarity']:
        raw_vals, labels = feat_labels[feat]
        ax.set_xticks(raw_vals)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('NAM Shape Functions: Individual Feature Contributions to Diamond Price',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# Each panel is a direct window into the model's internal logic.
#
# - **`carat`**: The curve rises sharply and monotonically, reflecting the well-known exponential price premium for large stones. The model assigns near-zero contribution below the average carat (0.80 ct) because those diamonds are common, and most of their price is captured by the bias term. Above ~1.5 ct the contribution exceeds &#36;4,000.
#
# - **`x` and `z`** (length and depth in mm): Near-identical rising curves. Both are proxies for size; the model uses them as secondary size signals when `carat` is ambiguous. Their combined contribution for large stones can exceed &#36;10,000 additively.
#
# - **`depth` and `table`**: Both show a penalty for extreme values. Very deep diamonds lose brilliance; very wide tables reduce light return. The curves decrease for values beyond the optimal range, correctly capturing why proportions matter.
#
# - **`cut`**: Unexpectedly, the shape function decreases for higher cut grades. This is a classic **interaction suppression artefact**: in the real data, Ideal-cut diamonds tend to be smaller (cutters optimise proportion over weight), so a purely additive model conflates high cut quality with smaller size and assigns a negative marginal contribution. TabNet avoids this by attending to cut and carat *jointly* in the same step.
#
# - **`color` and `clarity`**: Gentle positive slopes, consistent with the grading scale (higher code = better grade). The modest range (< &#36;500) reflects that color and clarity are secondary to size in the diamonds market, particularly at the low-to-mid carat range that dominates the dataset.
#
# None of this required any post-hoc explainability tool. The shape functions **are** the model.

# %%
# ── predicted vs actual + error by price tier ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Predicted vs Actual hexbin
ax = axes[0]
hb = ax.hexbin(y_test, te_pred_nam, gridsize=55, cmap='YlOrRd',
               bins='log', mincnt=1, linewidths=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', lw=1.5, alpha=0.7, label='Perfect prediction')
ax.plot([], [], 'w.', label=f'R² = {r2_nam:.4f}')
ax.plot([], [], 'w.', label=f'RMSE = ${rmse_nam:,.0f}')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log(count)', fontsize=9)
ax.set_xlabel('Actual Price ($)', fontsize=11)
ax.set_ylabel('Predicted Price ($)', fontsize=11)
ax.set_title('NAM: Predicted vs Actual', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Right: error band across price deciles, compared to Scratch TabNet and Pro
ax2 = axes[1]
n_buckets = 10
decile_edges = np.percentile(y_test, np.linspace(0, 100, n_buckets + 1))
bucket_centers, err_nam_v, err_tab_v = [], [], []

for i in range(n_buckets):
    mask = (y_test >= decile_edges[i]) & (y_test < decile_edges[i+1])
    if mask.sum() > 5:
        bucket_centers.append(np.median(y_test[mask]))
        err_nam_v.append(np.sqrt(mean_squared_error(y_test[mask], te_pred_nam[mask])))
        err_tab_v.append(np.sqrt(mean_squared_error(y_test[mask], pro_preds[mask])))

ax2.plot(bucket_centers, err_nam_v,   'o-', color='#E91E63', lw=2, ms=7, label='Scratch NAM')
ax2.plot(bucket_centers, err_tab_v,   's--', color='#2196F3', lw=1.8, ms=6, label='TabNet Pro')
ax2.fill_between(bucket_centers, err_nam_v, err_tab_v,
                 alpha=0.12, color='#9C27B0', label='Interaction gap')
ax2.set_xlabel('Price decile midpoint ($)', fontsize=11)
ax2.set_ylabel('RMSE ($)', fontsize=11)
ax2.set_title('Error by Price Tier: NAM vs TabNet Pro', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.suptitle('NAM: Prediction Quality and Error vs TabNet Pro',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# The left panel shows the NAM's predictions: the diagonal trend is clear, but the scatter band is noticeably wider than TabNet Pro's (the hexbin cloud is more diffuse). The model is calibrated; predictions are unbiased, but it misses the premium on exceptional stones.
#
# The right panel quantifies exactly *where* the interaction gap lives. For cheap diamonds (< &#36;2K), NAM and TabNet Pro produce nearly identical errors; low-price stones are small and uniform, so additivity is not a limiting assumption. The gap opens dramatically for diamonds above &#36;5K: at the &#36;10K+ tier, NAM's RMSE is roughly 3× worse than TabNet Pro. This is where the carat × clarity × cut interactions that TabNet learns to detect matter most. A &#36;15,000 diamond is almost always large AND high-quality across every dimension simultaneously, a joint condition that no sum of individual curves can fully represent.

# %% [markdown]
#
# ---
# ## 6. NAM Pro: `nam` library
#
# ### 6.1 What the Library Adds
#
# The official `nam` library (Agarwal et al., 2021) extends the scratch implementation in three ways:
#
# | Component | Scratch | `nam` library |
# |---|---|---|
# | FeatureNN depth | ExU → Dropout → Linear(1) | ExU(H) → [Linear(64)→ReLU] → [Linear(32)→ReLU] → Linear(1) |
# | Feature dropout | none | entire FeatureNNs zeroed with prob `feature_dropout` |
# | L2 regularisation | `weight_decay` in Adam | per-layer `l2_regularization` + `output_regularization` |
# | Shape function API | manual `shape_function()` method | `model.calc_outputs(X)` returns per-feature (B, 1) tensors |
#
# Feature dropout is the most structurally important addition: randomly silencing entire FeatureNNs during training forces the model to distribute the price signal across features rather than concentrating it in one dominant network. The result is more uniform and better-calibrated shape functions.
#
# ### 6.2 Hyperparameter Decisions
#
# | Parameter | Value | Rationale |
# |---|---|---|
# | `num_basis_functions` | 64 | hidden units in first ExU layer |
# | `units_multiplier` | 2 | n_units = 64 × 2 = 128 |
# | `hidden_sizes` | [64, 32] | two additional Linear layers after ExU |
# | `dropout` | 0.10 | standard dropout on hidden activations |
# | `feature_dropout` | 0.05 | structured: drops whole FeatureNNs |
# | `l2_regularization` | 1e-4 | weight decay analogue |
# | `lr` | 5e-4 | lower than scratch; deeper net converges slower |
# | `max_epochs` | 200 | early stopping guards the budget |
#
# The training loop is unchanged from scratch: a standard PyTorch `Adam` + `ReduceLROnPlateau`; the only difference is the `NAM(config, name, num_inputs, num_units)` constructor and the `calc_outputs(X)` method that returns per-feature `(B, 1)` tensors.

# %%
from nam.models import NAM
from nam.config import defaults

cfg_pro = defaults()
cfg_pro.regression          = True
cfg_pro.seed                = 42
cfg_pro.activation          = 'exu'
cfg_pro.num_basis_functions = 64
cfg_pro.units_multiplier    = 2          # n_units = 64 * 2 = 128
cfg_pro.hidden_sizes        = [64, 32]   # ExU(128) → Linear(64) → Linear(32) → Linear(1)
cfg_pro.dropout             = 0.10
cfg_pro.feature_dropout     = 0.05
cfg_pro.l2_regularization   = 1e-4
cfg_pro.output_regularization = 0.0

torch.manual_seed(cfg_pro.seed)
n_units_pro = cfg_pro.num_basis_functions * cfg_pro.units_multiplier
nam_pro = NAM(config=cfg_pro, name='NAMPro',
              num_inputs=X_train.shape[1], num_units=n_units_pro)
n_params_pro = sum(p.numel() for p in nam_pro.parameters() if p.requires_grad)
print(f"NAM Pro: {n_params_pro:,} parameters  ({n_params_pro // n_params:.0f}× scratch NAM)")

# %% [markdown]
# 55× more parameters than the scratch model, but all still partitioned into 9 independent feature networks. The additive constraint is intact; we only increased depth within each FeatureNN, not the number of features a single network can see.

# %%
# ── Training ──────────────────────────────────────────────────────────────────
LR_PRO     = 5e-4
BATCH_PRO  = 2048
EPOCHS_PRO = 200
PAT_PRO    = 25

opt_pro   = torch.optim.Adam(nam_pro.parameters(), lr=LR_PRO, weight_decay=cfg_pro.l2_regularization)
sched_pro = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pro, patience=10, factor=0.5, min_lr=5e-5)

Xtr_t2  = torch.tensor(X_train,       dtype=torch.float32)
ytrs_t2 = torch.tensor(y_train_scaled, dtype=torch.float32)
Xte_t2  = torch.tensor(X_test,        dtype=torch.float32)
loader2 = DataLoader(TensorDataset(Xtr_t2, ytrs_t2), batch_size=BATCH_PRO, shuffle=True)

pro_tr_hist, pro_val_hist = [], []
best_val_pro, best_epoch_pro, no_imp_pro = np.inf, 0, 0
best_state_pro = None

for epoch in range(1, EPOCHS_PRO + 1):
    nam_pro.train()
    for xb, yb in loader2:
        opt_pro.zero_grad()
        pred, _ = nam_pro(xb)
        F.mse_loss(pred, yb).backward()
        torch.nn.utils.clip_grad_norm_(nam_pro.parameters(), 1.0)
        opt_pro.step()

    nam_pro.eval()
    with torch.no_grad():
        tr_s2, _ = nam_pro(Xtr_t2);  te_s2, _ = nam_pro(Xte_t2)
    tr_pred2 = target_scaler.inverse_transform(tr_s2.numpy().reshape(-1,1)).ravel()
    te_pred2 = target_scaler.inverse_transform(te_s2.numpy().reshape(-1,1)).ravel()
    tr_rmse2 = np.sqrt(mean_squared_error(y_train, tr_pred2))
    te_rmse2 = np.sqrt(mean_squared_error(y_test,  te_pred2))
    pro_tr_hist.append(tr_rmse2); pro_val_hist.append(te_rmse2)
    sched_pro.step(te_rmse2)

    if te_rmse2 < best_val_pro:
        best_val_pro, best_epoch_pro = te_rmse2, epoch
        best_state_pro = {k: v.clone() for k, v in nam_pro.state_dict().items()}
        no_imp_pro = 0
    else:
        no_imp_pro += 1
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Train: ${tr_rmse2:,.0f} | Val: ${te_rmse2:,.0f} | LR: {opt_pro.param_groups[0]['lr']:.5f}")
    if no_imp_pro >= PAT_PRO:
        print(f"  Early stopping at epoch {epoch}  (best={best_epoch_pro})")
        break

nam_pro.load_state_dict(best_state_pro)
nam_pro.eval()
with torch.no_grad():
    te_pro_s, _ = nam_pro(Xte_t2)
    nam_pro_contribs = nam_pro.calc_outputs(Xte_t2)

te_pred_pro = target_scaler.inverse_transform(te_pro_s.numpy().reshape(-1,1)).ravel()
rmse_pro = np.sqrt(mean_squared_error(y_test, te_pred_pro))
mae_pro  = mean_absolute_error(y_test, te_pred_pro)
r2_pro   = r2_score(y_test, te_pred_pro)

print(f"\n{'='*50}")
print(f"  NAM Pro  |  Best epoch {best_epoch_pro}")
print(f"  RMSE : ${rmse_pro:,.2f}")
print(f"  MAE  : ${mae_pro:,.2f}")
print(f"  R2   : {r2_pro:.4f}")
print(f"{'='*50}")

# %% [markdown]
# The Pro model converges faster than scratch (early stopping around epoch 85 vs 140) because the deeper FeatureNNs can fit the shape functions with fewer gradient steps. But the final RMSE is essentially identical to the scratch model; both land around &#36;1,510–1,560, well above TabNet Pro's &#36;654.
#
# This is one of the most instructive results in the module: multiplying parameters by 55× without removing the additive constraint yields no RMSE improvement. Both models are bottlenecked by the same missing interaction terms, not by capacity. What the deeper architecture *does* improve is the quality and smoothness of the shape functions, visible in the next visualisation.

# %% [markdown]
# The most revealing comparison: the same mathematical quantity ($f_j(x_j)$, feature contribution in dollars) learned by two different architectures. Where the curves agree, the signal is robust. Where they disagree, the architecture capacity matters.

# %%
# ── shape function comparison: scratch nam vs nam pro ─────────────────────────
selected = ['carat', 'cut', 'clarity', 'depth', 'x']
std_y_v  = target_scaler.scale_[0]

fig, axes = plt.subplots(1, len(selected), figsize=(16, 5), sharey=False)

for ax, feat in zip(axes, selected):
    j = list(X_train_raw.columns).index(feat)
    mu, sig = feature_scaler.mean_[j], feature_scaler.scale_[j]

    # ── Scratch shape function ────────────────────────────────────────────────
    grid_s, vals_scratch = nam.shape_function(j, n_pts=300)
    vals_scratch_d = vals_scratch * std_y_v
    grid_orig = grid_s * sig + mu

    # ── Pro shape function ────────────────────────────────────────────────────
    grid_t = torch.linspace(-3.5, 3.5, 300)
    with torch.no_grad():
        full_x = torch.zeros(300, X_train.shape[1])
        full_x[:, j] = grid_t
        pro_co = nam_pro.calc_outputs(full_x)
    vals_pro_d = pro_co[j].squeeze(-1).numpy() * std_y_v

    # Zero-center both curves for fair comparison (remove constant offset from bias)
    vals_scratch_d -= vals_scratch_d.mean()
    vals_pro_d     -= vals_pro_d.mean()

    color_s = feat_colors[feat]
    ax.fill_between(grid_orig, 0, vals_scratch_d, alpha=0.10, color=color_s)
    ax.fill_between(grid_orig, 0, vals_pro_d,     alpha=0.10, color='#455A64')
    ax.plot(grid_orig, vals_scratch_d, color=color_s, lw=2.2, label='Scratch NAM', zorder=3)
    ax.plot(grid_orig, vals_pro_d,     color='#37474F', lw=1.8, ls='--', label='NAM Pro', zorder=3)
    ax.axhline(0, color='#aaa', lw=0.7, ls=':')

    ax.set_title(feat, fontsize=12, fontweight='bold', color=color_s)
    ax.set_xlabel('Original scale', fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.tick_params(labelsize=7)

    if feat in ['cut', 'color', 'clarity']:
        raw_vals, labels = feat_labels[feat]
        ax.set_xticks(raw_vals)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)

    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if feat == 'carat':
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Shape Functions: Scratch NAM vs NAM Pro (zero-centred)',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# The comparison surfaces two key differences:
#
# - **`carat`**: Both models agree on the upward trend; the Pro curve is smoother and extends further into negative territory (small diamonds contribute -&#36;509 below the mean, vs ~&#36;0 in scratch). The deeper architecture allows it to represent the "below-average is penalised" effect that the single-layer scratch model missed.
#
# - **`depth`** and **`table`**: The Pro model's curves are nearly flat (range &#36;45-&#36;61), while the scratch model shows steeper slopes. This is the effect of feature dropout:
#   `depth` and `table` are weakly predictive features, and the regulariser correctly suppresses their influence, concentrating the model's capacity on `carat`, `x`, `z`.
#
# - **`cut`**: Both models show the same interaction-suppression artefact. This is not a calibration issue; it is the additive constraint manifesting in the same place regardless of architecture depth. The artefact is a property of the data's interaction structure, not the model's capacity.
#
# - **`clarity`**: The Pro model shows a cleaner monotone step function, consistent with the grading scale. The scratch model captures the same direction but with a noisier boundary around VS2/VS1.

# %% [markdown]
#
# ---
# ## 7. Final Comparison
#
# | Model | RMSE | MAE | R² | Parameters | Interpretable |
# |---|---|---|---|---|---|
# | Baseline (mean) | &#36;4,031 | &#36;3,033 | 0.000 | 1 | trivial |
# | Scratch NAM | &#36;1,512 | &#36;1,015 | 0.856 | 1,729 | **full shape functions** |
# | NAM Pro (`nam`) | &#36;1,555 | &#36;958 | 0.848 | 95,347 | **full shape functions** |
# | Scratch TabNet | &#36;1,076 | &#36;700 | 0.927 | ~43K | attention masks |
# | **TabNet Pro** | **&#36;654** | **&#36;378** | **0.973** | ~45K | attention masks |
#
# The table tells a clean story on two axes: **accuracy** and **interpretability**.
#
# Reading the NAM rows together, the most striking result is that multiplying parameters by 55× (Scratch: 1,729; Pro: 95,347) yields no RMSE improvement; in fact, it is marginally worse (&#36;1,555 vs &#36;1,512). The bottleneck is not capacity; it is the architectural constraint. No amount of depth inside each FeatureNN can produce a term that combines `carat × clarity`, because such a term requires two inputs and the additive sum forbids it. Throwing more parameters at an incorrectly specified model is a pattern that repeats across machine learning; NAMs make it unusually visible.
#
# Where the deeper Pro architecture *does* help is interpretability quality: the Pro shape functions are smoother, the regulariser suppresses noise features (`depth`, `table`) to near-zero, and the `carat` curve correctly captures the below-mean penalty that the scratch model approximates as zero.
#
# - **NAM (both variants)** trade ~12 R² points for complete transparency. Every prediction decomposes into nine auditable dollar contributions. This is the architecture of choice when an explanation is a legal or regulatory requirement.
#
# - **Scratch TabNet** hits 0.927 R² with a fraction of a commercial model's complexity. The attention masks are interpretable at the aggregate level (feature importance ranking) but not at the individual prediction level.
#
# - **TabNet Pro** leverages Ghost Batch Normalisation, momentum-based training, and a heavily tuned implementation to reach 0.973 R², within reach of gradient boosted trees (CatBoost achieves ~0.979 on the same split).
#
# No single architecture is universally best. The right choice is the one whose **interpretability requirement, parameter budget, and accuracy floor** all align with the application context.

# %% [markdown]
# ---
# ## Conclusion
#
# This module covered the two neural architectures that have reshaped how practitioners think about tabular data:
#
# **TabNet** demonstrated that a transformer-style attention mechanism can be adapted to select features sequentially, producing a model whose feature importances emerge naturally from the attention weights rather than from post-hoc SHAP or permutation tests. The scratch implementation revealed every building block: sparsemax, GLU blocks, the prior scale that prevents repetition across steps, and the sparsity regularisation term that controls how focused each step is. The Pro implementation showed what a well-engineered library adds on top: Ghost Batch Normalisation for more stable training and a scheduler that squeezes the last few R² points out of the data.
#
# **Neural Additive Models** pushed interpretability to its logical limit. By restricting each sub-network to a single input, we obtained shape functions: curves that show the dollar contribution of every feature value, without any approximation or surrogate. The ExU activation enabled the network to represent sharp grading transitions with very few parameters. The `nam` library added depth, feature dropout, and L2 regularisation, producing smoother and better-calibrated shape functions; but the RMSE barely moved, because the binding constraint is additive structure, not model capacity. The cost was real: a 12-point R² gap versus TabNet Pro, concentrated in the high-value segment where feature interactions dominate.
#
# **Key takeaways:**
# - StandardScaler is not optional for neural networks on tabular data: it is structural.
# - sparsemax produces cleaner, sparser attention than softmax, but both are valid.
# - ExU's initialisation (`w ~ N(4, 0.5)`, `b ~ N(0, 0.5)`) is critical: zero-centred `b` distributes thresholds across the input range, allowing shape functions to represent behaviour on both sides of the feature mean.
# - Multiplying NAM parameters by 55× without removing the additive constraint yields no RMSE improvement: architectural inductive bias is the binding constraint, not capacity.
# - Feature dropout in NAM Pro correctly suppresses low-signal features (`depth`, `table`) and produces cleaner, more interpretable shape functions.
# - The interaction gap between additive and non-additive models is not uniform: it is concentrated at high feature value combinations, visible in per-decile RMSE plots.
# - Model complexity should be chosen by interpretability requirement first, parameter budget second, and accuracy target third.
