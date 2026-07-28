"""Data for the ATLAS 'Energy Based Models' article, module 3.2.

An energy model is the shortest possible answer to "what is a density": pick any
function E(x) you like, call exp(-E(x)) unnormalised probability, and divide by
its integral. The integral is the whole difficulty. Nobody computes it, every
method in the literature is a way of not computing it, and the price of not
computing it is never shown because in the dimensions people care about it
cannot be shown.

In two dimensions it can be shown. That is the reason this module measures on a
plane: the partition function is a sum over a grid, and the spine
(generative_common.py) measures that the sum reproduces a closed form mass to
2.8e-15. So the article can do the thing the field cannot do, which is train the
same network the honest way and the cheap way and score both against the exact
answer.

Six blocks, every number in the page measured here:

A. The constant nobody computes, computed. One architecture, one
   initialisation, one number of updates, nine arms: exact maximum likelihood
   with log Z by quadrature, contrastive divergence with k Langevin steps from
   the data for k in 1, 5, 20, 100, and persistent contrastive divergence with a
   replay buffer at the same four k. Scored by the EXACT KL to the truth, over
   three seeds, so the gap between two arms can be read against the wobble
   between seeds.
B. The gradient, directly. At four points along the exact run, the exact
   gradient of the log likelihood by quadrature against the CD-k estimate of the
   same gradient, for four k and three batch sizes, with repeats. Cosine
   similarity and norm ratio, and the cosine of the AVERAGE over repeats, which
   is the part Monte Carlo noise cannot explain and bias can.
C. The sampler is the model. Unadjusted Langevin has a stationary distribution
   that is not the target. On the true energy the target is known exactly, so
   the error is measurable: KL against step size over two orders of magnitude,
   the fitted exponent, then the Metropolis adjusted version on the same grid of
   step sizes with its acceptance rates. Plus the closed form for a Gaussian,
   which the simulation has to reproduce.
D. The noise is not optional. The same chains with the noise term deleted.
E. A classifier is already an energy model. The module's shared judge, with
   E(x) = -logsumexp(logits), against four kinds of thing that are not digits.
   Nobody trained it for that.
F. The export: the float16 weights the browser will run, the energy and the
   truth on a grid, and reference values computed on the QUANTISED weights, so
   the page can check its own forward pass and its own log Z.

Nothing here publishes a wall clock time. Where a cost is compared, it is
compared as a count of energy evaluations, which is the same number on every
machine. The thread count goes into meta because it changes the order of the
floating point reductions.

Run from anywhere: `python src/utils/generate_ebm_data.py`.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import generative_common as gc              # noqa: E402

OUT = ROOT / "ebm" / "data"
OUT.mkdir(parents=True, exist_ok=True)

THREADS = gc.threads()
SEED = gc.SEED
rnd, arr, mat, sci = gc.rnd, gc.arr, gc.mat, gc.sci

# ---------------------------------------------------------------- the model
HIDDEN = 64                # 2 -> 64 -> 64 -> 1, softplus, so E is smooth in x
TAU = 2.0                  # E(x) = mlp(x) + |x|^2 / (2 tau^2): the anchor that
#                            keeps exp(-E) integrable no matter what the mlp does
N_TRAIN = 4000
N_HELD = 2000

# ------------------------------------------------------------ block A sizes
SEEDS = (19, 23, 29)
UPDATES = 600
BATCH = 256
LR = 3e-3
TRAIN_M = 128              # the grid the exact arm normalises on while training
SCORE_M = 512              # the grid every arm is finally scored on
TRACE_M = 256              # the grid the training curve is measured on
TRACE_EVERY = 100
CD_KS = (1, 5, 20, 100)
CD_STEP = 0.01             # the Langevin step the negative chains use
BUFFER = 1024              # the persistent replay buffer

# ------------------------------------------------------------ block B sizes
GRAD_KS = (1, 5, 20, 100)
GRAD_BATCHES = (64, 256, 1024)
GRAD_REPEATS = 8
GRAD_M = 512
GRAD_CHECK_M = 256
CKPT_FRACTIONS = (0.0, 0.125, 0.5, 1.0)

# ------------------------------------------------------------ block C sizes
CHAINS = 20000
BURN = 500                 # 5 / (eps * ring curvature) at the smallest step, so the
#                            chain has left the exact draw it started from
KEEP = 500
THIN = 5
EPS_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2)
SAMPLER_SEEDS = (19, 23, 29)
BINS = 24                  # bins per side of the picture square
SUB = 42                   # fine cells per bin when the bin's exact mass is taken
ANNULUS = (1.2, 2.8)       # the band that contains the ring and nothing else
RBINS = 32
RTARGET_M = 2048
ESCAPE = 50.0              # a walker past this is absorbed and counted as outside
RCLAMP = 1e-9
GAUSS_WALKERS = 200000     # the one dimensional closed form check

# ------------------------------------------------------------ block D sizes
D_CHAINS = 5000
D_STEPS = 600

# ------------------------------------------------------------ block E sizes
POOL_N = 600

# ------------------------------------------------------------ block F sizes
EXPORT_M = 128
SAMPLE_N = 900
VIEW = gc.TOY["view"]
DOMAIN = gc.TOY["domain"]

STAGE_SECONDS = {}


def stage(name, t0):
    """Elapsed seconds, printed and never exported: a wall clock is a fact
    about one machine. What gets exported is counts of energy evaluations."""
    dt = time.time() - t0
    STAGE_SECONDS[name] = dt
    print(f"    [{name} took {dt:.1f} s on {THREADS} threads]")
    return time.time()


# ================================================================= the network
def make_net(seed):
    torch.manual_seed(int(seed))
    return torch.nn.Sequential(
        torch.nn.Linear(2, HIDDEN), torch.nn.Softplus(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Softplus(),
        torch.nn.Linear(HIDDEN, 1))


def weights_of(net):
    return [p.detach().numpy().copy().astype(np.float64) for p in net.parameters()]


def net_from(W):
    net = make_net(SEED)
    with torch.no_grad():
        for p, v in zip(net.parameters(), W):
            p.copy_(torch.tensor(np.asarray(v), dtype=torch.float32))
    return net


def energy_t(net, x):
    """The energy in torch: the mlp plus the quadratic anchor."""
    return net(x).squeeze(-1) + (x * x).sum(-1) / (2.0 * TAU * TAU)


def softplus_np(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def energy_np(W, X):
    """The same energy in numpy, which is what everything downstream reads."""
    W1, b1, W2, b2, W3, b3 = W
    X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
    h = softplus_np(X @ np.asarray(W1).T + np.asarray(b1))
    h = softplus_np(h @ np.asarray(W2).T + np.asarray(b2))
    e = (h @ np.asarray(W3).T + np.asarray(b3)).ravel()
    return e + (X * X).sum(1) / (2.0 * TAU * TAU)


def flat_grad(net):
    return torch.cat([p.grad.reshape(-1) for p in net.parameters()]).numpy().astype(np.float64)


def quantised_weights(spec, w):
    """The weights read back out of the float16 payload, in the shape energy_np
    wants. export_half writes each (in, out) matrix transposed and flattened, so
    reshaping to (out, in) recovers exactly the numbers the browser will read."""
    out = []
    for s in spec:
        nin, nout = s["shape"]
        out.append(w[s["w_offset"]:s["w_offset"] + s["w_size"]].reshape(nout, nin))
        out.append(w[s["b_offset"]:s["b_offset"] + s["b_size"]])
    return out


def model_langevin(net, x, k, eps, gen):
    """k steps of unadjusted Langevin on the model's own energy."""
    for _ in range(int(k)):
        x = x.detach().requires_grad_(True)
        g, = torch.autograd.grad(energy_t(net, x).sum(), x)
        x = (x - eps * g + math.sqrt(2.0 * eps) * torch.randn(x.shape, generator=gen)).detach()
    return x


_GRIDS = {}


def grid_of(m, half=None):
    key = (m, half)
    if key not in _GRIDS:
        _GRIDS[key] = gc.toy_grid(m) if half is None else gc.toy_grid(m, half=half)
    return _GRIDS[key]


def score(W, m=SCORE_M):
    """log Z by quadrature, and from it the EXACT KL to the truth.

    The model is a density on the quadrature square: q(x) = exp(-E(x)) / Z with
    Z the sum below and zero outside. `leak_nats` further down measures what the
    same energy would have put outside that square, which is the size of the
    thing this definition sweeps up.
    """
    X, cell, _ = grid_of(m)
    E = energy_np(W, X)
    logz = float(logsumexp(-E) + math.log(cell))
    log_q = -E - logz
    return {"kl": float(gc.toy_kl(log_q, m=m)), "logz": logz, "log_q": log_q}


def heldout_logp(W, Xhe, logz):
    v = -energy_np(W, Xhe) - logz
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def spread(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.max() - v.min())


print("=" * 72)
print("ATLAS / energy based models: the constant nobody computes, computed")
print("=" * 72)
print(f"  torch {torch.__version__}, numpy {np.__version__}, {THREADS} threads")

# ============================================================== 0. the truth
gc.banner("the truth, and what the grid under it is worth")
T0 = time.time()
QUAD = gc.toy_quadrature_report()
for row in QUAD:
    print(f"  a {row['m']} by {row['m']} grid integrates the closed form to "
          f"{row['mass']:.15f}, off by {row['error']:.3e}")
print("  the error stops falling because it is already the floating point floor, "
      "not because the grid has stopped improving: a finer grid buys nothing here")

ENT = {m: float(gc.toy_entropy(m)) for m in (512, 1024, 2048)}
CEILING = -ENT[1024]
print(f"  the truth's entropy is {ENT[1024]:.6f} nats "
      f"({ENT[512]:.6f} at 512, {ENT[2048]:.6f} at 2048), so no model on this data "
      f"can have a mean log density above {CEILING:.6f}")

HOLES = [{"r": r, "mass": sci(gc.toy_mass(lambda X, rr=r: (X ** 2).sum(1) < rr * rr))}
         for r in (1.0, 1.2, 1.4)]
for h in HOLES:
    print(f"  the hole in the middle of the ring: {h['mass']:.3e} of the mass "
          f"inside r = {h['r']}")

# The KL machinery, checked against the only case whose answer is known without
# any model at all: the truth scored against itself.
XG, CELLG, _ = grid_of(SCORE_M)
KL_SELF = float(gc.toy_kl(gc.toy_logpdf(XG), m=SCORE_M))
print(f"  the truth scored against itself gives KL {KL_SELF:.3e}, which is the "
      f"noise floor of every KL below")

# the mass the picture square leaves out, two ways
XV, CELLV, _ = grid_of(BINS * SUB, half=VIEW)
MASS_VIEW = float(gc.toy_pdf(XV).sum() * CELLV)
OUT_VIEW_A = 1.0 - MASS_VIEW
OUT_VIEW_B = float(gc.toy_mass(lambda X: (np.abs(X) > VIEW).any(1), m=1024))
print(f"  the {2*VIEW:g} by {2*VIEW:g} square the pictures show leaves "
      f"{OUT_VIEW_A:.3e} of the mass outside (the domain grid says "
      f"{OUT_VIEW_B:.3e}, a difference of {abs(OUT_VIEW_A-OUT_VIEW_B):.1e})")
T0 = stage("truth", T0)


# ================================ the true energy, its gradient, and the guards
def rot(deg):
    return gc._rot(deg)


def components(X):
    """The three pieces of the truth, each already multiplied by its weight.

    Written out here rather than called from the spine because the sampler needs
    each piece separately: the gradient of a mixture is the weighted average of
    the gradients of its parts. Their sum is asserted against the spine's own
    density below, so this is a second implementation and not a copy with a
    typo in it.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
    R = gc.TOY["ring"]
    r = np.maximum(np.sqrt((X ** 2).sum(1)), 1e-12)
    ring = (R["weight"] * np.exp(-0.5 * ((r - R["r0"]) / R["sigma"]) ** 2)
            / (2.0 * np.pi * r * R["sigma"] * np.sqrt(2.0 * np.pi)))
    cols = [ring]
    for name in ("blob_a", "blob_b"):
        B = gc.TOY[name]
        Q = rot(B["angle"])
        s = np.asarray(B["sd"], dtype=float)
        d = (X - np.asarray(B["mean"], dtype=float)) @ Q
        cols.append(B["weight"] * np.exp(-0.5 * ((d / s) ** 2).sum(1))
                    / (2.0 * np.pi * s[0] * s[1]))
    return np.column_stack(cols)


def true_grad_logp(X):
    """grad log p of the mixture, in closed form."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
    C = components(X)
    p = np.maximum(C.sum(1), 1e-300)
    R = gc.TOY["ring"]
    r = np.maximum(np.sqrt((X ** 2).sum(1)), RCLAMP)
    radial = -(r - R["r0"]) / R["sigma"] ** 2 - 1.0 / r
    G = C[:, 0:1] * (radial[:, None] * (X / r[:, None]))
    for i, name in enumerate(("blob_a", "blob_b"), start=1):
        B = gc.TOY[name]
        Q = rot(B["angle"])
        s = np.asarray(B["sd"], dtype=float)
        d = (X - np.asarray(B["mean"], dtype=float)) @ Q
        G = G + C[:, i:i + 1] * (-(d / s ** 2) @ Q.T)
    return G / p[:, None]


def true_logp_torch(x):
    R = gc.TOY["ring"]
    r = torch.clamp(torch.sqrt((x * x).sum(-1)), min=1e-12)
    p = (R["weight"] * torch.exp(-0.5 * ((r - R["r0"]) / R["sigma"]) ** 2)
         / (2.0 * math.pi * r * R["sigma"] * math.sqrt(2.0 * math.pi)))
    for name in ("blob_a", "blob_b"):
        B = gc.TOY[name]
        Q = torch.tensor(rot(B["angle"]), dtype=torch.float64)
        s = torch.tensor(np.asarray(B["sd"], dtype=float), dtype=torch.float64)
        mu = torch.tensor(np.asarray(B["mean"], dtype=float), dtype=torch.float64)
        d = (x - mu) @ Q
        p = p + (B["weight"] * torch.exp(-0.5 * ((d / s) ** 2).sum(-1))
                 / (2.0 * math.pi * s[0] * s[1]))
    return torch.log(p)


gc.banner("the gradient of the true log density, three ways")
PROBE = gc.toy_sample(400, seed=101)
PROBE = np.vstack([PROBE, np.array([[0.0, 1.0], [3.0, -3.0], [-1.5, 1.5], [4.0, 4.0]])])
g_closed = true_grad_logp(PROBE)
xt = torch.tensor(PROBE, dtype=torch.float64, requires_grad=True)
g_auto, = torch.autograd.grad(true_logp_torch(xt).sum(), xt)
g_auto = g_auto.numpy()
H = 1e-5
g_fd = np.column_stack([
    (gc.toy_logpdf(PROBE + H * np.eye(2)[j]) - gc.toy_logpdf(PROBE - H * np.eye(2)[j])) / (2 * H)
    for j in range(2)])
GRAD_GUARD = {
    "closed_vs_autograd": sci(np.abs(g_closed - g_auto).max()),
    "closed_vs_differences": sci(np.abs(g_closed - g_fd).max()),
    "components_vs_spine": sci(np.abs(components(PROBE).sum(1) - gc.toy_pdf(PROBE)).max()),
    "step": H, "points": int(len(PROBE)),
}
print(f"  closed form against torch autograd on {len(PROBE)} points: "
      f"{GRAD_GUARD['closed_vs_autograd']:.3e}")
print(f"  closed form against central differences at h = {H}: "
      f"{GRAD_GUARD['closed_vs_differences']:.3e}")
print(f"  and the three pieces sum to the spine's own density to "
      f"{GRAD_GUARD['components_vs_spine']:.3e}")
T0 = stage("gradient guards", T0)


# ==================================== A. the constant nobody computes, computed
def data_for(seed):
    X = gc.toy_sample(N_TRAIN + N_HELD, seed=5000 + int(seed))
    return X[:N_TRAIN], X[N_TRAIN:]


def train(mode, k, seed):
    """One arm. Same architecture, same initialisation, same update count.

    mode is 'exact' (log Z by quadrature, so the gradient is the real one),
    'cd' (k Langevin steps started at the data) or 'pcd' (k Langevin steps
    started wherever the replay buffer left off).
    """
    key = (f"ebm-arm-v2-{mode}-k{k}-s{seed}-h{HIDDEN}-tau{TAU}-u{UPDATES}-b{BATCH}"
           f"-lr{LR}-gm{TRAIN_M}-eps{CD_STEP}-buf{BUFFER}-n{N_TRAIN}"
           f"-tr{TRACE_M}x{TRACE_EVERY}")

    def build():
        Xtr, _ = data_for(seed)
        net = make_net(seed)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        gen = torch.Generator().manual_seed(4000 + int(seed))
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        gX, gcell, _ = grid_of(TRAIN_M)
        GX = torch.tensor(gX, dtype=torch.float32)
        logcell = math.log(gcell)
        buf = (torch.randn((BUFFER, 2), generator=gen) * TAU)
        marks = sorted({int(round(f * UPDATES)) for f in CKPT_FRACTIONS})
        ckpts, trace = {}, []
        for t in range(UPDATES):
            if t in marks:
                ckpts[t] = weights_of(net)
            idx = torch.randint(0, len(Xt), (BATCH,), generator=gen)
            xb = Xt[idx]
            opt.zero_grad()
            if mode == "exact":
                Eg = energy_t(net, GX)
                loss = energy_t(net, xb).mean() + (torch.logsumexp(-Eg, 0) + logcell)
            else:
                if mode == "cd":
                    x0 = xb.clone()
                else:
                    bidx = torch.randint(0, BUFFER, (BATCH,), generator=gen)
                    x0 = buf[bidx].clone()
                neg = model_langevin(net, x0, k, CD_STEP, gen)
                if mode == "pcd":
                    buf[bidx] = neg
                loss = energy_t(net, xb).mean() - energy_t(net, neg).mean()
            loss.backward()
            opt.step()
            if t == 0 or (t + 1) % TRACE_EVERY == 0:
                trace.append({"update": t + 1,
                              "kl": rnd(score(weights_of(net), TRACE_M)["kl"], 5)})
        ckpts[UPDATES] = weights_of(net)
        return {"W": weights_of(net), "ckpts": ckpts, "trace": trace}

    return gc.cached(key, build, quiet=True)


gc.banner("A. one network, nine ways of paying for the normalising constant")
print(f"  {HIDDEN} by {HIDDEN} softplus energy plus a |x|^2/(2*{TAU}^2) anchor, "
      f"{UPDATES} Adam updates of batch {BATCH} at lr {LR}, {len(SEEDS)} seeds")
print(f"  the exact arm normalises on a {TRAIN_M} by {TRAIN_M} grid every update; "
      f"the CD arms take k Langevin steps of {CD_STEP} from the data; the PCD arms "
      f"take them from a buffer of {BUFFER} that never restarts")

ARMS = [("exact", 0)] + [("cd", k) for k in CD_KS] + [("pcd", k) for k in CD_KS]
XSPAN = grid_of(EXPORT_M, half=VIEW)[0]        # where the energy span is read off
FITS, ROWS = {}, []
for mode, k in ARMS:
    kls, held, spans, ses = [], [], [], []
    for s in SEEDS:
        got = train(mode, k, s)
        FITS[(mode, k, s)] = got
        sc = score(got["W"], SCORE_M)
        _, Xhe = data_for(s)
        mu, se = heldout_logp(got["W"], Xhe, sc["logz"])
        kls.append(sc["kl"])
        held.append(mu)
        ses.append(se)
        Ev = energy_np(got["W"], XSPAN)
        spans.append(float(Ev.max() - Ev.min()))
    name = "exact" if mode == "exact" else f"{mode}-{k}"
    # The currency is energy evaluations, which is the same number everywhere.
    evals = UPDATES * TRAIN_M * TRAIN_M if mode == "exact" else UPDATES * k * BATCH
    row = {
        "arm": name, "mode": mode, "k": int(k),
        "kl": rnd(float(np.mean(kls)), 5), "kl_seeds": arr(kls, 5),
        "kl_spread": rnd(spread(kls), 5),
        "heldout": rnd(float(np.mean(held)), 4), "heldout_seeds": arr(held, 4),
        "heldout_spread": rnd(spread(held), 4), "heldout_se": rnd(float(np.mean(ses)), 4),
        "energy_span": rnd(float(np.mean(spans)), 3),
        "negative_evals": int(evals),
        "trace": FITS[(mode, k, SEEDS[0])]["trace"],
    }
    ROWS.append(row)
    print(f"  {name:>8}: KL {row['kl']:.5f} (seeds {', '.join(f'{v:.5f}' for v in kls)}, "
          f"spread {row['kl_spread']:.5f}), held out log density {row['heldout']:.4f} "
          f"against the ceiling {CEILING:.4f}")

BY = {r["arm"]: r for r in ROWS}
KL_SPREADS = [r["kl_spread"] for r in ROWS]
FLOOR_A = float(np.median(KL_SPREADS))
print(f"\n  the seed to seed spread of the KL runs from {min(KL_SPREADS):.5f} to "
      f"{max(KL_SPREADS):.5f}, median {FLOOR_A:.5f}: that is the floor under every "
      f"comparison in this table")

CD1_BIAS = BY["cd-1"]["kl"] - BY["exact"]["kl"]
PCD1_BIAS = BY["pcd-1"]["kl"] - BY["exact"]["kl"]
CD100_BIAS = BY["cd-100"]["kl"] - BY["exact"]["kl"]
print(f"  CD-1 costs {CD1_BIAS:.4f} nats of KL against the exact arm "
      f"({BY['cd-1']['kl']:.5f} against {BY['exact']['kl']:.5f}), which is "
      f"{abs(CD1_BIAS)/max(FLOOR_A,1e-12):.0f} times the seed floor")
if CD100_BIAS > 0:
    print(f"  a hundred steps close most of it but not all: CD-100 is still "
          f"{CD100_BIAS:.4f} nats above the exact arm")
else:
    print(f"  and against the script, CD-100 scores {abs(CD100_BIAS):.4f} nats BELOW "
          f"the exact arm: at this budget more sampling stopped being the "
          f"limitation, so the measurement wins and the sentence changes")
best_cd = min((r for r in ROWS if r["mode"] == "cd"), key=lambda r: r["kl"])
best_pcd = min((r for r in ROWS if r["mode"] == "pcd"), key=lambda r: r["kl"])
if best_pcd["kl"] < best_cd["kl"]:
    print(f"  the best persistent arm ({best_pcd['arm']}, {best_pcd['kl']:.5f}) beats the "
          f"best data started one ({best_cd['arm']}, {best_cd['kl']:.5f}) by "
          f"{best_cd['kl']-best_pcd['kl']:.5f} nats")
else:
    print(f"  the best data started arm ({best_cd['arm']}, {best_cd['kl']:.5f}) beats the "
          f"best persistent one ({best_pcd['arm']}, {best_pcd['kl']:.5f}) by "
          f"{best_pcd['kl']-best_cd['kl']:.5f} nats, which is not the usual story and is "
          f"what this run measured")
print(f"  cost in energy evaluations: the exact arm spends "
      f"{BY['exact']['negative_evals']:,}, CD-1 spends {BY['cd-1']['negative_evals']:,} and "
      f"CD-100 spends {BY['cd-100']['negative_evals']:,}. The exact arm is not the "
      f"expensive one in two dimensions, it is the impossible one in two hundred")
T0 = stage("block A", T0)

# ======================================================== B. the gradient, directly
gc.banner("B. the exact gradient of the log likelihood, and what CD-k sends instead")
CKPT_ARM = FITS[("exact", 0, SEEDS[0])]
XTR0, XHE0 = data_for(SEEDS[0])


def exact_gradient(W, Xdata, m, chunk=65536):
    """grad of the negative log likelihood, with the negative phase done by
    quadrature instead of by sampling.

    The weights of the negative phase are a softmax over the grid, so it is
    accumulated in chunks: exact, and with memory that does not depend on m.
    """
    net = net_from(W)
    Xd = torch.tensor(np.asarray(Xdata), dtype=torch.float32)
    net.zero_grad()
    energy_t(net, Xd).mean().backward()
    g_data = flat_grad(net)
    X, cell, _ = grid_of(m)
    E = energy_np(W, X)
    w = np.exp(-E - logsumexp(-E))
    net.zero_grad()
    for s in range(0, len(X), chunk):
        xb = torch.tensor(X[s:s + chunk], dtype=torch.float32)
        wb = torch.tensor(w[s:s + chunk], dtype=torch.float32)
        (-(wb * energy_t(net, xb)).sum()).backward()
    return g_data + flat_grad(net), g_data


def cd_gradient(W, g_data, k, batch, rep, Xtr):
    net = net_from(W)
    gen = torch.Generator().manual_seed(90000 + 977 * rep + 13 * k + batch)
    idx = torch.randint(0, len(Xtr), (batch,), generator=gen)
    x0 = torch.tensor(np.asarray(Xtr)[idx.numpy()], dtype=torch.float32)
    neg = model_langevin(net, x0, k, CD_STEP, gen)
    net.zero_grad()
    (-energy_t(net, neg).mean()).backward()
    return g_data + flat_grad(net)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def gradient_block():
    rows, norms = [], []
    marks = sorted(CKPT_ARM["ckpts"].keys())
    for t in marks:
        W = CKPT_ARM["ckpts"][t]
        g_ex, g_data = exact_gradient(W, XTR0, GRAD_M)
        g_chk, _ = exact_gradient(W, XTR0, GRAD_CHECK_M)
        norms.append({"update": int(t), "norm": sci(np.linalg.norm(g_ex)),
                      "data_norm": sci(np.linalg.norm(g_data)),
                      "grid_check_cos": rnd(cosine(g_ex, g_chk), 6),
                      "grid_check_ratio": rnd(float(np.linalg.norm(g_chk)
                                                    / max(np.linalg.norm(g_ex), 1e-30)), 6)})
        for k in GRAD_KS:
            for b in GRAD_BATCHES:
                G = np.stack([cd_gradient(W, g_data, k, b, r, XTR0)
                              for r in range(GRAD_REPEATS)])
                cs = np.array([cosine(g, g_ex) for g in G])
                rt = np.array([np.linalg.norm(g) / max(np.linalg.norm(g_ex), 1e-30)
                               for g in G])
                gm = G.mean(0)
                rows.append({
                    "update": int(t), "k": int(k), "batch": int(b),
                    "cos": rnd(float(cs.mean()), 4), "cos_sd": rnd(float(cs.std(ddof=1)), 4),
                    "ratio": rnd(float(rt.mean()), 4), "ratio_sd": rnd(float(rt.std(ddof=1)), 4),
                    "cos_of_mean": rnd(cosine(gm, g_ex), 4),
                    "ratio_of_mean": rnd(float(np.linalg.norm(gm)
                                               / max(np.linalg.norm(g_ex), 1e-30)), 4),
                })
    return {"rows": rows, "exact": norms}


GRAD = gc.cached(
    f"ebm-grad-v2-u{UPDATES}-m{GRAD_M}-{GRAD_CHECK_M}-k{GRAD_KS}-b{GRAD_BATCHES}"
    f"-r{GRAD_REPEATS}-s{SEEDS[0]}-eps{CD_STEP}-h{HIDDEN}-tau{TAU}-n{N_TRAIN}",
    gradient_block, quiet=True)

print(f"  {GRAD_REPEATS} repeats of every (k, batch) pair at "
      f"{len(GRAD['exact'])} points along the exact run, all against the same "
      f"data term, so the only thing that differs is how the negative phase is taken")
for e in GRAD["exact"]:
    print(f"  after {e['update']:>4} updates the exact gradient has norm {e['norm']:.3e} "
          f"(the same gradient on a {GRAD_CHECK_M} grid agrees to a cosine of "
          f"{e['grid_check_cos']:.6f})")
MID = sorted({r["update"] for r in GRAD["rows"]})[1]
for k in GRAD_KS:
    r = [x for x in GRAD["rows"] if x["update"] == MID and x["k"] == k
         and x["batch"] == GRAD_BATCHES[-1]][0]
    print(f"  at update {MID}, k = {k:>3}, batch {r['batch']}: cosine "
          f"{r['cos']:.4f} plus or minus {r['cos_sd']:.4f}, norm ratio {r['ratio']:.4f}; "
          f"averaging the {GRAD_REPEATS} repeats lifts the cosine to {r['cos_of_mean']:.4f}")
LAST = sorted({r["update"] for r in GRAD["rows"]})[-1]
e_first = [e for e in GRAD["exact"] if e["update"] == 0][0]
e_last = [e for e in GRAD["exact"] if e["update"] == LAST][0]
r_last = [x for x in GRAD["rows"] if x["update"] == LAST and x["k"] == 1
          and x["batch"] == GRAD_BATCHES[-1]][0]
shrink = float(e_first["norm"]) / max(float(e_last["norm"]), 1e-30)
if shrink > 3.0:
    print(f"  by the end of the exact run its own gradient has shrunk by a factor of "
          f"{shrink:.0f}, to {e_last['norm']:.3e}, while CD-1 still pulls with "
          f"{r_last['ratio_of_mean']:.1f} times that norm. The cosine there means "
          f"nothing (it is measured against something on its way to zero) and the norm "
          f"is the whole point: CD's fixed point is somewhere else")
else:
    print(f"  at the end of the exact run its gradient has NOT collapsed "
          f"({e_last['norm']:.3e} against {e_first['norm']:.3e} at the start, a factor "
          f"of {shrink:.2f}), so this run has not reached the optimum and the sentence "
          f"about CD pulling away from a stationary point is not one this measurement "
          f"can make. CD-1's norm there is {r_last['ratio_of_mean']:.2f} times the "
          f"exact one")
T0 = stage("block B", T0)

# ============================================== C. the sampler is the model
gc.banner("C. what unadjusted Langevin actually samples from")
XEDGES = np.linspace(-VIEW, VIEW, BINS + 1)
REDGES = np.linspace(ANNULUS[0], ANNULUS[1], RBINS + 1)


def bin_targets():
    X, cell, _ = grid_of(BINS * SUB, half=VIEW)
    p = (gc.toy_pdf(X) * cell).reshape(BINS * SUB, BINS * SUB)
    T = p.reshape(BINS, SUB, BINS, SUB).sum((1, 3))
    return T, float(max(1.0 - T.sum(), 0.0))


TARGET2D, TARGET_OUT = bin_targets()


def radial_target(m):
    X, cell, _ = grid_of(m)
    p = gc.toy_pdf(X) * cell
    r = np.sqrt((X ** 2).sum(1))
    m_in = (r >= ANNULUS[0]) & (r < ANNULUS[1])
    h = np.histogram(r[m_in], bins=REDGES, weights=p[m_in])[0]
    w = p[m_in]
    mu = float((w * r[m_in]).sum() / w.sum())
    sd = float(np.sqrt((w * (r[m_in] - mu) ** 2).sum() / w.sum()))
    return h / h.sum(), mu, sd, float(w.sum())


TARGET_R, RING_MU, RING_SD, RING_MASS = radial_target(RTARGET_M)
TARGET_R_C, _, RING_SD_C, _ = radial_target(RTARGET_M // 2)
print(f"  the band {ANNULUS[0]} to {ANNULUS[1]} holds {RING_MASS:.4f} of the truth, "
      f"with radius {RING_MU:.5f} plus or minus {RING_SD:.5f} "
      f"(a {RTARGET_M//2} grid says {RING_SD_C:.5f})")

XD, CELLD, _ = grid_of(1024)
_lab = components(XD).argmax(1)
_pd = gc.toy_pdf(XD) * CELLD
REGION_MASS = [float(_pd[_lab == i].sum()) for i in range(3)]
print(f"  by nearest component the truth splits {REGION_MASS[0]:.4f} ring, "
      f"{REGION_MASS[1]:.4f} round blob, {REGION_MASS[2]:.4f} thin blob")


def kl_hist(counts, outside, total):
    q = np.concatenate([counts.ravel(), [outside]]) / total
    p = np.concatenate([TARGET2D.ravel(), [max(TARGET_OUT, 1e-300)]])
    m = q > 0
    return float((q[m] * np.log(q[m] / p[m])).sum())


def kl_radial(counts):
    tot = counts.sum()
    if tot == 0:
        return float("nan")
    q = counts / tot
    m = q > 0
    return float((q[m] * np.log(q[m] / np.maximum(TARGET_R[m], 1e-300))).sum())


NBINS_USED = int((TARGET2D > 1e-12).sum()) + 1


def collect(x, acc, series=True):
    H = np.histogram2d(x[:, 0], x[:, 1], bins=[XEDGES, XEDGES])[0]
    acc["counts"] += H
    acc["outside"] += len(x) - H.sum()
    acc["total"] += len(x)
    r = np.sqrt((x ** 2).sum(1))
    m = (r >= ANNULUS[0]) & (r < ANNULUS[1])
    acc["r"] += np.histogram(r[m], bins=REDGES)[0]
    acc["rs"] += float(r[m].sum())
    acc["rss"] += float((r[m] ** 2).sum())
    acc["rn"] += int(m.sum())
    acc["walkers"] = max(acc["walkers"], len(x))
    if not series:
        return
    # Two walkers of the same chain one thinning apart are not two samples. The
    # lag one correlation of the radius and of the angle, accumulated online,
    # is what turns a sample count into an effective sample count below.
    c = x[:, 0] / np.maximum(r, 1e-12)
    for tag, cur in (("r", r), ("c", c)):
        prev = acc["prev"].get(tag)
        if prev is not None:
            a = acc["ac"][tag]
            a[0] += float(cur.sum())
            a[1] += float(prev.sum())
            a[2] += float((cur * prev).sum())
            a[3] += float((cur * cur).sum())
            a[4] += float((prev * prev).sum())
            a[5] += len(cur)
        acc["prev"][tag] = cur


def new_acc():
    return {"counts": np.zeros((BINS, BINS)), "outside": 0.0, "total": 0,
            "r": np.zeros(RBINS), "rs": 0.0, "rss": 0.0, "rn": 0, "walkers": 0,
            "prev": {}, "ac": {"r": [0.0] * 6, "c": [0.0] * 6}}


def lag_one(a):
    n = a[5]
    if n < 2:
        return 0.0
    mx, my = a[0] / n, a[1] / n
    vx, vy = a[3] / n - mx * mx, a[4] / n - my * my
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(np.clip((a[2] / n - mx * my) / math.sqrt(vx * vy), -0.999, 0.999))


def effective_n(acc, tag, total):
    """The usual (1 - rho) / (1 + rho) correction, floored at the number of
    walkers (a chain is worth at least the one sample it started from) and
    capped at the number of samples actually taken."""
    rho = lag_one(acc["ac"][tag])
    n = total * (1.0 - rho) / (1.0 + rho)
    return float(min(max(n, min(acc["walkers"], total)), total)), rho


def run_chains(X0, eps, kind, seed, burn=BURN, keep=KEEP, thin=THIN):
    """kind is 'ula', 'mala' or 'descent'. Absorbing anything past ESCAPE, which
    is reported rather than hidden: at the biggest steps that is the answer."""
    rs = np.random.RandomState(int(seed))
    x = np.array(X0, dtype=np.float64, copy=True)
    live = np.ones(len(x), dtype=bool)
    lp = gc.toy_logpdf(x)
    g = true_grad_logp(x)
    acc = new_acc()
    accepted = proposed = 0
    inner = 0
    for t in range(burn + keep):
        if kind == "descent":
            step = eps * g
            x = np.where(live[:, None], x + step, x)
        else:
            z = rs.randn(*x.shape)
            prop = x + eps * g + math.sqrt(2.0 * eps) * z
            prop = np.nan_to_num(prop, nan=ESCAPE * 2, posinf=ESCAPE * 2, neginf=-ESCAPE * 2)
            if kind == "ula":
                x = np.where(live[:, None], prop, x)
            else:
                lp2 = gc.toy_logpdf(prop)
                g2 = true_grad_logp(prop)
                fwd = -((prop - x - eps * g) ** 2).sum(1) / (4.0 * eps)
                bwd = -((x - prop - eps * g2) ** 2).sum(1) / (4.0 * eps)
                la = lp2 - lp + bwd - fwd
                take = (np.log(np.maximum(rs.rand(len(x)), 1e-300)) < la) & live
                accepted += int(take.sum())
                proposed += int(live.sum())
                x = np.where(take[:, None], prop, x)
                lp = np.where(take, lp2, lp)
                g = np.where(take[:, None], g2, g)
        if kind != "mala":
            live &= (np.abs(x) <= ESCAPE).all(1)
            g = true_grad_logp(x)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        inner += int((np.sqrt((x ** 2).sum(1)) < 0.05).sum())
        if t >= burn and (t - burn) % thin == 0:
            collect(x, acc)
    lab0 = components(X0).argmax(1)
    lab1 = components(x).argmax(1)
    return {
        "acc": acc, "final": x, "escaped": float(1.0 - live.mean()),
        "accept": (accepted / proposed) if proposed else None,
        "switch": float((lab0 != lab1).mean()),
        "weights": [float((lab1 == i).mean()) for i in range(3)],
        "inner": inner,
    }


def summarise(out):
    a = out["acc"]
    rn = max(a["rn"], 1)
    mu = a["rs"] / rn
    sd = math.sqrt(max(a["rss"] / rn - mu * mu, 0.0))
    n_ang, rho_ang = effective_n(a, "c", a["total"])
    n_rad, rho_rad = effective_n(a, "r", a["total"])
    return {
        "kl_grid": kl_hist(a["counts"], a["outside"], a["total"]),
        "kl_ring": kl_radial(a["r"]),
        # the plug in KL of a histogram reads (bins - 1) / (2 n) too high, and
        # the n in that formula is the effective one, not the sample count
        "floor_grid": (NBINS_USED - 1) / (2.0 * n_ang),
        "floor_ring": (RBINS - 1) / (2.0 * n_rad),
        "neff_angle": n_ang, "neff_radius": n_rad,
        "rho_angle": rho_ang, "rho_radius": rho_rad,
        "ring_sd": sd, "ring_mu": mu,
        "outside": a["outside"] / a["total"],
        "escaped": out["escaped"], "switch": out["switch"],
        "weights": out["weights"], "accept": out["accept"],
        "samples": int(a["total"]), "inner": out["inner"],
    }


def sampler_arm(kind, eps, seed):
    key = (f"ebm-chain-v3-{kind}-e{eps}-s{seed}-c{CHAINS}-b{BURN}-k{KEEP}-t{THIN}"
           f"-bins{BINS}-r{RBINS}-{ANNULUS}-esc{ESCAPE}")

    def build():
        X0 = gc.toy_sample(CHAINS, seed=7000 + int(seed))
        return summarise(run_chains(X0, eps, kind, 20000 + int(seed)))

    return gc.cached(key, build, quiet=True)


def floor_arm(seed):
    """The same estimator on exact independent samples: what the histogram KL
    reads when the sampler is perfect.

    Twice, because the number that matters is not the number of samples. Once
    with every sample independent, which is the optimistic end, and once with
    only as many samples as there are chains, which is what a run whose walkers
    never forget where they started is really worth. Both are compared with the
    textbook (bins - 1) / (2 n), which is the third way of getting the same
    number.
    """
    key = (f"ebm-floor-v3-s{seed}-c{CHAINS}-k{KEEP}-t{THIN}-bins{BINS}-r{RBINS}"
           f"-{ANNULUS}")

    def build():
        rs = np.random.RandomState(600 + int(seed))
        acc, one = new_acc(), new_acc()
        for i in range(KEEP // THIN):
            X = gc.toy_sample(CHAINS, seed=int(rs.randint(1, 2 ** 31 - 1)))
            collect(X, acc)
            if i == 0:
                collect(X, one, series=False)
        rn = max(acc["rn"], 1)
        mu = acc["rs"] / rn
        return {"kl_grid": kl_hist(acc["counts"], acc["outside"], acc["total"]),
                "kl_ring": kl_radial(acc["r"]),
                "kl_grid_chains": kl_hist(one["counts"], one["outside"], one["total"]),
                "kl_ring_chains": kl_radial(one["r"]),
                "rho_angle": lag_one(acc["ac"]["c"]),
                "ring_sd": math.sqrt(max(acc["rss"] / rn - mu * mu, 0.0)),
                "samples": int(acc["total"])}

    return gc.cached(key, build, quiet=True)


FLOORS = [floor_arm(s) for s in SAMPLER_SEEDS]
FLOOR_GRID = float(np.mean([f["kl_grid"] for f in FLOORS]))
FLOOR_RING = float(np.mean([f["kl_ring"] for f in FLOORS]))
FLOOR_GRID_C = float(np.mean([f["kl_grid_chains"] for f in FLOORS]))
FLOOR_RING_C = float(np.mean([f["kl_ring_chains"] for f in FLOORS]))
N_SAMP = FLOORS[0]["samples"]
FLOOR_FORMULA = {"grid_all": sci((NBINS_USED - 1) / (2.0 * N_SAMP)),
                 "grid_chains": sci((NBINS_USED - 1) / (2.0 * CHAINS)),
                 "ring_all": sci((RBINS - 1) / (2.0 * N_SAMP)),
                 "ring_chains": sci((RBINS - 1) / (2.0 * CHAINS)),
                 "measured_rho_of_independent_draws": sci(float(np.mean(
                     [f["rho_angle"] for f in FLOORS])))}
print(f"  {CHAINS:,} chains, {BURN} steps of burn in, {KEEP} kept and thinned by "
      f"{THIN}, so {N_SAMP:,} samples per arm")
print(f"  on {N_SAMP:,} EXACT and independent samples the estimator still reads "
      f"{FLOOR_GRID:.5f} nats on the {BINS} by {BINS} grid and {FLOOR_RING:.5f} in the "
      f"radial band; the textbook bias says {FLOOR_FORMULA['grid_all']:.2e} and "
      f"{FLOOR_FORMULA['ring_all']:.2e}")
print(f"  on only {CHAINS:,} of them, which is what a run whose walkers never forget "
      f"where they started is worth, it reads {FLOOR_GRID_C:.5f} and "
      f"{FLOOR_RING_C:.5f} (formula {FLOOR_FORMULA['grid_chains']:.2e} and "
      f"{FLOOR_FORMULA['ring_chains']:.2e})")
print(f"  so each arm below carries its own floor, computed from the lag one "
      f"correlation its own walkers actually have")

SAMPLER_ROWS = []
for kind in ("ula", "mala"):
    for eps in EPS_GRID:
        got = [sampler_arm(kind, eps, s) for s in SAMPLER_SEEDS]
        row = {
            "sampler": kind, "eps": eps,
            "kl_grid": rnd(float(np.mean([g["kl_grid"] for g in got])), 5),
            "kl_grid_spread": rnd(spread([g["kl_grid"] for g in got]), 5),
            "kl_ring": rnd(float(np.mean([g["kl_ring"] for g in got])), 5),
            "kl_ring_spread": rnd(spread([g["kl_ring"] for g in got]), 5),
            "floor_grid": sci(float(np.mean([g["floor_grid"] for g in got]))),
            "floor_ring": sci(float(np.mean([g["floor_ring"] for g in got]))),
            "neff_angle": int(round(float(np.mean([g["neff_angle"] for g in got])))),
            "neff_radius": int(round(float(np.mean([g["neff_radius"] for g in got])))),
            "ring_sd": rnd(float(np.mean([g["ring_sd"] for g in got])), 5),
            "ring_inflation": rnd(float(np.mean([g["ring_sd"] for g in got])) / RING_SD, 4),
            "outside": sci(float(np.mean([g["outside"] for g in got]))),
            "escaped": sci(float(np.mean([g["escaped"] for g in got]))),
            "switch": sci(float(np.mean([g["switch"] for g in got]))),
            "inner_hits": int(sum(g["inner"] for g in got)),
            "weights": arr(np.mean([g["weights"] for g in got], axis=0), 4),
            "accept": (rnd(float(np.mean([g["accept"] for g in got])), 4)
                       if got[0]["accept"] is not None else None),
        }
        SAMPLER_ROWS.append(row)
        tail = "" if row["accept"] is None else f", acceptance {row['accept']*100:.1f}%"
        print(f"  {kind:>5} at step {eps:g}: KL {row['kl_grid']:.5f} "
              f"(spread {row['kl_grid_spread']:.5f}, floor {row['floor_grid']:.1e}), "
              f"radial KL {row['kl_ring']:.5f} (floor {row['floor_ring']:.1e}), "
              f"ring width {row['ring_sd']:.5f} against the true {RING_SD:.5f}{tail}")


INNER = sum(r["inner_hits"] for r in SAMPLER_ROWS)
if INNER == 0:
    print(f"  and no walker of any arm ever entered r < 0.05, where the ring's 1/r "
          f"term is singular and the gradient is clamped at {RCLAMP:g}: the clamp is "
          f"never reached, so nothing below depends on it")
else:
    print(f"  {INNER:,} walker visits landed inside r < 0.05, where the ring's 1/r term "
          f"is singular and the gradient is clamped at {RCLAMP:g}: that clamp is load "
          f"bearing and the numbers around it are worth less than the others")


def fit_exponent(rows, field, floor_field):
    """log KL against log step, over the steps where the KL is well clear of the
    arm's OWN estimator floor. Returns the exponent, its R squared, and which
    points it used, because a fit on three points is a different claim from a
    fit on six."""
    pts = [(r["eps"], r[field]) for r in rows
           if r[field] > 3.0 * r[floor_field] and r[field] > 0]
    if len(pts) < 3:
        return {"points": len(pts), "exponent": None, "r2": None,
                "note": "not enough steps above the floor to fit anything"}
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = float(((y - pred) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    return {"points": len(pts), "exponent": rnd(float(coef[0]), 3),
            "intercept": rnd(float(coef[1]), 3),
            "r2": rnd(1.0 - ss / tot if tot > 0 else float("nan"), 5),
            "eps_min": min(p[0] for p in pts), "eps_max": max(p[0] for p in pts)}


ULA_ROWS = [r for r in SAMPLER_ROWS if r["sampler"] == "ula"]
MALA_ROWS = [r for r in SAMPLER_ROWS if r["sampler"] == "mala"]
FIT_GRID = fit_exponent(ULA_ROWS, "kl_grid", "floor_grid")
FIT_RING = fit_exponent(ULA_ROWS, "kl_ring", "floor_ring")
if FIT_GRID["exponent"] is not None:
    print(f"  the {BINS} by {BINS} KL of unadjusted Langevin goes as step to the "
          f"{FIT_GRID['exponent']} (R squared {FIT_GRID['r2']}, over the "
          f"{FIT_GRID['points']} steps above the floor)")
if FIT_RING["exponent"] is not None:
    print(f"  the radial KL, whose floor is {FLOOR_RING:.6f} and which therefore reaches "
          f"further down, goes as step to the {FIT_RING['exponent']} "
          f"(R squared {FIT_RING['r2']}, {FIT_RING['points']} steps)")

worst_mala = max(MALA_ROWS, key=lambda r: r["kl_ring"])
worst_ula = max(ULA_ROWS, key=lambda r: r["kl_ring"])
big = EPS_GRID[-1]
mala_big = [r for r in MALA_ROWS if r["eps"] == big][0]
ula_big = [r for r in ULA_ROWS if r["eps"] == big][0]
print(f"  at the same step of {big:g} the correction is the whole difference: "
      f"unadjusted reads {ula_big['kl_ring']:.5f} in the radial band and adjusted "
      f"{mala_big['kl_ring']:.5f}, with {mala_big['accept']*100:.1f}% of its proposals "
      f"accepted. The worst MALA arm anywhere is {worst_mala['kl_ring']:.5f} against "
      f"the worst unadjusted one's {worst_ula['kl_ring']:.5f}")
best_ula = min(ULA_ROWS, key=lambda r: r["kl_grid"])
best_ula_ring = min(ULA_ROWS, key=lambda r: r["kl_ring"])
QUIET = [r for r in SAMPLER_ROWS if r["ring_inflation"] < 1.5]
NOISY = [r for r in SAMPLER_ROWS if r["ring_inflation"] >= 1.5]
if QUIET:
    print(f"  no chain crosses between the ring and the blobs while it is still "
          f"sampling anything: over the {len(QUIET)} arms whose ring is within half "
          f"again of its true width, the largest fraction that changes component is "
          f"{max(float(r['switch']) for r in QUIET):.1e}. Started in the wrong "
          f"proportions, no number of steps would fix them")
if NOISY:
    worst_switch = max(NOISY, key=lambda r: float(r["switch"]))
    print(f"  the {len(NOISY)} arms that have blown the ring open do move walkers "
          f"between components (up to {float(worst_switch['switch']):.2f} at "
          f"{worst_switch['sampler']} step {worst_switch['eps']:g}), which is not "
          f"mixing: it is the discretisation throwing them out of the mode")


def gaussian_check():
    """The one case where the stationary distribution of unadjusted Langevin is
    known in closed form: a Gaussian. For target N(0, 1/lam), the chain's own
    variance is 2 / (lam (2 - eps lam)), so its ratio to the target is
    1 / (1 - eps lam / 2). This is the only claim in block C that can be checked
    against algebra instead of against another simulation."""
    lam = 1.0 / gc.TOY["ring"]["sigma"] ** 2
    rows = []
    for eps in EPS_GRID:
        rs = np.random.RandomState(31)
        x = rs.randn(GAUSS_WALKERS) / math.sqrt(lam)
        s2 = 0.0
        n = 0
        for t in range(400 + 200):
            x = x - eps * lam * x + math.sqrt(2.0 * eps) * rs.randn(GAUSS_WALKERS)
            if t >= 400:
                s2 += float((x ** 2).mean())
                n += 1
        pred = 2.0 / (2.0 - eps * lam) if eps * lam < 2.0 else float("inf")
        rows.append({"eps": eps, "measured": rnd(s2 / n * lam, 5),
                     "closed_form": rnd(pred, 5) if np.isfinite(pred) else None})
    return {"lambda": rnd(lam, 4), "rows": rows}


GAUSS = gc.cached(f"ebm-gauss-v2-{EPS_GRID}-{GAUSS_WALKERS}", gaussian_check, quiet=True)
for g in GAUSS["rows"]:
    cf = "unstable" if g["closed_form"] is None else f"{g['closed_form']:.5f}"
    print(f"  a Gaussian of the ring's own curvature at step {g['eps']:g}: variance ratio "
          f"{g['measured']:.5f} measured, {cf} in closed form")


def cold_start():
    """The same sampler, started from the wrong place. Nothing else changes."""
    key = f"ebm-cold-v3-c{CHAINS}-b{BURN}-k{KEEP}-t{THIN}-bins{BINS}"

    def build():
        rs = np.random.RandomState(77)
        X0 = rs.uniform(-VIEW, VIEW, (CHAINS, 2))
        out = {}
        for eps in (1e-2, 3e-2):
            out[eps] = summarise(run_chains(X0, eps, "ula", 31000))
        return out

    return gc.cached(key, build, quiet=True)


COLD = cold_start()
for eps, c in sorted(COLD.items()):
    print(f"  started uniformly on the picture square instead of at the truth, "
          f"unadjusted Langevin at step {eps:g} ends with weights "
          f"{c['weights'][0]:.3f} / {c['weights'][1]:.3f} / {c['weights'][2]:.3f} against "
          f"the true {REGION_MASS[0]:.3f} / {REGION_MASS[1]:.3f} / {REGION_MASS[2]:.3f}, "
          f"and KL {c['kl_grid']:.4f}")
T0 = stage("block C", T0)

# ============================================== D. the noise is not optional
gc.banner("D. the same chains with the noise deleted")
RING_MIN = 0.5 * (gc.TOY["ring"]["r0"]
                  + math.sqrt(gc.TOY["ring"]["r0"] ** 2 - 4 * gc.TOY["ring"]["sigma"] ** 2))


def descent_arm(eps, seed):
    key = f"ebm-descent-v3-e{eps}-s{seed}-c{D_CHAINS}-st{D_STEPS}-bins{BINS}"

    def build():
        X0 = gc.toy_sample(D_CHAINS, seed=7000 + int(seed))
        out = run_chains(X0, eps, "descent", 40000 + int(seed),
                         burn=D_STEPS - 1, keep=1, thin=1)
        x = out["final"]
        # a second, shorter run to say how much is still moving at the end
        out2 = run_chains(X0, eps, "descent", 40000 + int(seed),
                          burn=D_STEPS - 51, keep=1, thin=1)
        move = float(np.abs(x - out2["final"]).max())
        r = np.sqrt((x ** 2).sum(1))
        band = (r >= ANNULUS[0]) & (r < ANNULUS[1])
        s = summarise(out)
        s["distinct_3"] = int(len(np.unique(np.round(x, 3), axis=0)))
        s["distinct_6"] = int(len(np.unique(np.round(x, 6), axis=0)))
        s["ring_radius"] = float(r[band].mean()) if band.any() else float("nan")
        s["ring_radius_sd"] = float(r[band].std()) if band.any() else float("nan")
        s["moved_last_50"] = move
        # A step that has stopped moving because it is tiny is not a step that
        # has arrived. What says arrived is that the gradient it is following
        # has gone to zero, so that is what gets measured.
        s["grad_end"] = float(np.linalg.norm(true_grad_logp(x), axis=1).mean())
        s["endpoints"] = x[:600]
        return s

    return gc.cached(key, build, quiet=True)


D_ROWS = []
for eps in EPS_GRID:
    got = [descent_arm(eps, s) for s in SAMPLER_SEEDS]
    row = {
        "eps": eps,
        "kl_grid": rnd(float(np.mean([g["kl_grid"] for g in got])), 4),
        "kl_grid_spread": rnd(spread([g["kl_grid"] for g in got]), 4),
        "kl_ring": rnd(float(np.mean([g["kl_ring"] for g in got])), 4),
        "distinct_3": int(round(float(np.mean([g["distinct_3"] for g in got])))),
        "distinct_6": int(round(float(np.mean([g["distinct_6"] for g in got])))),
        "ring_sd": sci(float(np.mean([g["ring_sd"] for g in got]))),
        "ring_radius": rnd(float(np.mean([g["ring_radius"] for g in got])), 5),
        "moved_last_50": sci(float(np.mean([g["moved_last_50"] for g in got]))),
        "grad_end": sci(float(np.mean([g["grad_end"] for g in got]))),
        "weights": arr(np.mean([g["weights"] for g in got], axis=0), 4),
    }
    D_ROWS.append(row)
    print(f"  step {eps:g}: radial KL {row['kl_ring']:.4f} against the best Langevin "
          f"arm's {best_ula_ring['kl_ring']:.5f}, ring width {row['ring_sd']:.2e} against the "
          f"true {RING_SD:.5f}, {row['distinct_3']:,} distinct endpoints of "
          f"{D_CHAINS:,} at three decimals, ending on a gradient of "
          f"{row['grad_end']:.1e} after moving {row['moved_last_50']:.1e} over the "
          f"last fifty steps")
# The claim about where the walkers land belongs to an arm that has actually
# stopped moving, which is measured rather than assumed.
D_SHOW_EPS = EPS_GRID[len(EPS_GRID) // 2]
D_SHOW = descent_arm(D_SHOW_EPS, SAMPLER_SEEDS[0])
D_CONV = min(D_ROWS, key=lambda r: float(r["grad_end"]))
print(f"  the arm that has actually arrived is step {D_CONV['eps']:g}, whose walkers "
      f"end on a gradient of {D_CONV['grad_end']:.1e} and moved "
      f"{D_CONV['moved_last_50']:.1e} over their last fifty steps: there every walker "
      f"that started on the ring sits at radius {D_CONV['ring_radius']:.5f}, and "
      f"solving (r - r0)/sigma^2 + 1/r = 0 gives {RING_MIN:.5f}, a difference of "
      f"{abs(D_CONV['ring_radius']-RING_MIN):.1e}")
if D_CONV["distinct_3"] > 0.1 * D_CHAINS:
    print(f"  so 'the walkers collapse onto the modes' is not quite what happens, and "
          f"the measurement is more interesting than the slogan: the ring's minimum is "
          f"a CIRCLE, a one dimensional set, so {D_CONV['distinct_3']:,} of the "
          f"{D_CHAINS:,} endpoints are still distinct at three decimals. They are all "
          f"at one radius, with a spread of {float(D_CONV['ring_sd']):.1e} against the "
          f"true {RING_SD:.5f}. The two blobs are the part that does collapse to a "
          f"point each")
else:
    print(f"  the endpoints collapse to {D_CONV['distinct_3']} distinct points of "
          f"{D_CHAINS:,}")
T0 = stage("block D", T0)

# ================================== E. a classifier is already an energy model
gc.banner("E. the energy nobody trained: -logsumexp of a classifier's logits")


def judge_logits():
    """The shared judge, with its logits.

    generative_common.judge() returns probabilities, and a softmax divides out
    exactly the quantity this section is about: logsumexp of the logits is the
    constant it throws away. The weights are in the module cache under the key
    judge() writes, so this calls judge() first (which guarantees the entry
    exists) and then reads that same entry back. Nothing here retrains anything
    and nothing here writes to the spine.
    """
    j = gc.judge()
    key = f"judge-{gc.JUDGE['hidden']}-{gc.JUDGE['epochs']}-{gc.JUDGE['seed']}-v1"

    def missing():
        raise RuntimeError("the judge's own cache entry disappeared between two calls")

    raw = gc.cached(key, missing, quiet=True)
    W1, b1, W2, b2 = raw["W"]

    def logits(X):
        h = np.maximum(np.asarray(X, dtype=float).reshape(-1, W1.shape[1]) @ W1.T + b1, 0.0)
        return h @ W2.T + b2

    return j, logits


JUDGE, LOGITS = judge_logits()
FLAT, YD, TR, TE = gc.digits14()
XR = FLAT[TE]
rsE = np.random.RandomState(SEED)
MU, SD = float(FLAT[TR].mean()), float(FLAT[TR].std())
G_RAW = rsE.normal(MU, SD, (POOL_N, FLAT.shape[1]))
POOLS = {
    "real": XR[:POOL_N],
    "uniform": rsE.uniform(0.0, 1.0, (POOL_N, FLAT.shape[1])),
    # the same uniform noise with its total ink scaled to the digits', which is
    # the control: it changes the one thing the energy turns out to care about
    "uniform_matched": rsE.uniform(0.0, 2.0 * MU, (POOL_N, FLAT.shape[1])),
    "gaussian": G_RAW,
    "gaussian_clipped": np.clip(G_RAW, 0.0, 1.0),
    "shuffled": np.array([rsE.permutation(row) for row in XR[:POOL_N]]),
    "fashion": gc.fashion14(POOL_N, seed=SEED)[0],
}


def energy_of(X):
    z = LOGITS(X)
    zm = z.max(1, keepdims=True)
    return -(zm.ravel() + np.log(np.exp(z - zm).sum(1)))


def confidence_of(X):
    z = LOGITS(X)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).max(1)


def auc_two_ways(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    a = float(roc_auc_score(y, s))
    r = rankdata(s)
    b = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
              / (len(pos) * len(neg)))
    return a, b


E_REAL = energy_of(POOLS["real"])
C_REAL = confidence_of(POOLS["real"])
POOL_ROWS, AUC_GAP = [], 0.0
ALL_E, ALL_INK = [], []
for name, X in POOLS.items():
    e = energy_of(X)
    c = confidence_of(X)
    ink = np.asarray(X).sum(1)
    ALL_E.append(e)
    ALL_INK.append(ink)
    row = {"pool": name, "n": int(len(X)),
           "energy": rnd(float(e.mean()), 4), "energy_sd": rnd(float(e.std(ddof=1)), 4),
           "ink": rnd(float(ink.mean()), 3),
           "confidence": rnd(float(c.mean()), 4)}
    if name != "real":
        a1, a2 = auc_two_ways(-E_REAL, -e)
        c1, c2 = auc_two_ways(C_REAL, c)
        AUC_GAP = max(AUC_GAP, abs(a1 - a2), abs(c1 - c2))
        row["auc_energy"] = rnd(a1, 4)
        row["auc_confidence"] = rnd(c1, 4)
    POOL_ROWS.append(row)
    extra = "" if name == "real" else (f", AUC {row['auc_energy']:.4f} by energy and "
                                       f"{row['auc_confidence']:.4f} by confidence")
    print(f"  {name:>16}: mean energy {row['energy']:>8.4f} plus or minus "
          f"{row['energy_sd']:.4f}, mean confidence {row['confidence']:.4f}{extra}")

_z = LOGITS(XR)
_z = np.exp(_z - _z.max(1, keepdims=True))
SOFT_GUARD = float(np.abs(_z / _z.sum(1, keepdims=True) - JUDGE["proba"](XR)).max())
INK_CORR = float(np.corrcoef(np.concatenate(ALL_E), np.concatenate(ALL_INK))[0, 1])
print(f"  the judge gets {JUDGE['accuracy']*100:.2f}% of the {len(TE)} held out digits "
      f"right and is {JUDGE['real_confidence']*100:.2f}% confident on them; nobody asked "
      f"it for a density and nobody trained it on anything that is not a digit")
FAKES = [r for r in POOL_ROWS if r["pool"] != "real"]
wins = [r for r in FAKES if r["auc_energy"] > r["auc_confidence"]]
useful = [r for r in FAKES if r["auc_energy"] > 0.5]
backwards = [r for r in FAKES if r["auc_energy"] < 0.5]
print(f"  energy scores above chance on {len(useful)} of the {len(FAKES)} pools and "
      f"beats the softmax confidence on {len(wins)} of them")
if backwards:
    worst = min(FAKES, key=lambda r: r["auc_energy"])
    print(f"  and on {len(backwards)} it is BACKWARDS: {worst['pool']} scores "
          f"{worst['auc_energy']:.4f}, which means this energy prefers that pool to real "
          f"digits ({worst['energy']:.4f} against {POOL_ROWS[0]['energy']:.4f}). The "
          f"story that a classifier is a free density model does not survive contact "
          f"with the measurement, and the reason is in the next line")
print(f"  across all {sum(r['n'] for r in POOL_ROWS):,} images the energy correlates "
      f"{INK_CORR:.4f} with total ink, and the pools run from {min(r['ink'] for r in POOL_ROWS):.2f} "
      f"to {max(r['ink'] for r in POOL_ROWS):.2f} of it: what -logsumexp mostly measures "
      f"here is how much of the image is switched on")
mat_row = [r for r in FAKES if r["pool"] == "uniform_matched"][0]
uni_row = [r for r in FAKES if r["pool"] == "uniform"][0]
print(f"  the control says the same thing from the other side: uniform noise with the "
      f"digits' own ink scores {mat_row['auc_energy']:.4f} where the same noise at full "
      f"brightness scores {uni_row['auc_energy']:.4f}, one knob changed")
hard = min(FAKES, key=lambda r: abs(r["auc_energy"] - 0.5))
print(f"  the pool it separates least is {hard['pool']} at {hard['auc_energy']:.4f}, "
      f"which is the honest edge of the claim and is measured rather than skipped")
T0 = stage("block E", T0)

# ============================================== F. what the browser will run
gc.banner("F. the weights, the grids and the references the page checks itself with")
EXPORT_ARMS = {"exact": FITS[("exact", 0, SEEDS[0])]["W"], "cd1": FITS[("cd", 1, SEEDS[0])]["W"]}
XE, CELLE, _ = grid_of(EXPORT_M, half=VIEW)
PROBES = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-1.414, -1.414], [1.2, 1.2],
                   [-2.55, 2.55], [2.55, -2.55], [3.4, -1.7], [0.5, 0.5],
                   [4.0, 4.0], [-4.0, -4.0], [6.0, 0.0]])
MODELS = {}
for name, W in EXPORT_ARMS.items():
    spec, b64, v16 = gc.export_half([("l1", np.asarray(W[0]).T, W[1]),
                                     ("l2", np.asarray(W[2]).T, W[3]),
                                     ("l3", np.asarray(W[4]).T, W[5])])
    w = gc.decode_half(b64, len(v16))
    # Everything the page will quote is measured on THESE numbers, the quantised
    # ones, not on the ones training left behind: float16 is what the browser has.
    Wq = quantised_weights(spec, w)
    dense_check = float(np.abs(gc.dense(w, spec[0], PROBES)
                               - (PROBES @ np.asarray(Wq[0]).T + Wq[1])).max())
    full = score(W, SCORE_M)
    quant = score(Wq, SCORE_M)
    _, Xhe0 = data_for(SEEDS[0])
    hq, hqse = heldout_logp(Wq, Xhe0, quant["logz"])
    hf, _ = heldout_logp(W, Xhe0, full["logz"])
    Eg = energy_np(Wq, XE)
    e16 = np.array(Eg, dtype=np.float32).astype(np.float16).astype(np.float64)
    egrid_b64, _ = gc.export_flat(Eg)
    # how far the same energy would leak outside the square the constant is taken over
    Xw, cellw, _ = grid_of(EXPORT_M * 4, half=DOMAIN * 1.6)
    lz_wide = float(logsumexp(-energy_np(Wq, Xw)) + math.log(cellw))
    MODELS[name] = {
        "layers": spec, "count": int(len(v16)), "weights_b64": b64,
        "kl": rnd(full["kl"], 5), "kl_quantised": rnd(quant["kl"], 5),
        "heldout": rnd(hf, 4), "heldout_quantised": rnd(hq, 4), "heldout_se": rnd(hqse, 4),
        "logz": {str(m): rnd(float(logsumexp(-energy_np(Wq, grid_of(m)[0]))
                                   + math.log(grid_of(m)[1])), 6)
                 for m in (64, 128, 256, 512, 1024)},
        "logz_train_grid": rnd(float(logsumexp(-energy_np(Wq, grid_of(TRAIN_M)[0]))
                                     + math.log(grid_of(TRAIN_M)[1])), 6),
        "leak_nats": sci(lz_wide - quant["logz"]),
        "energy_b64": egrid_b64,
        "energy_lo": rnd(float(Eg.min()), 4), "energy_hi": rnd(float(Eg.max()), 4),
        "grid_quant_max": sci(float(np.abs(e16 - Eg).max())),
        "probe_energy": arr(energy_np(Wq, PROBES), 6),
        "quant_energy_shift": sci(float(np.abs(energy_np(Wq, XE) - energy_np(W, XE)).max())),
        "dense_check": sci(dense_check),
    }
    m = MODELS[name]
    print(f"  {name}: {m['count']:,} float16 numbers; KL {m['kl']:.5f} before quantising "
          f"and {m['kl_quantised']:.5f} after, held out log density {m['heldout_quantised']:.4f} "
          f"plus or minus {m['heldout_se']:.4f} against the ceiling {CEILING:.4f}")
    print(f"     log Z on the {TRAIN_M} grid {m['logz_train_grid']:.6f}, on 512 "
          f"{m['logz']['512']:.6f}, on 1024 {m['logz']['1024']:.6f}; the same energy "
          f"would put {m['leak_nats']:.2e} nats of mass outside the square the constant "
          f"is taken over")

TRUTH_LOG = np.log(np.maximum(gc.toy_pdf(XE), 1e-300))
TRUTH_LOG = np.maximum(TRUTH_LOG, -20.0)
truth_b64, truth16 = gc.export_flat(TRUTH_LOG)
SAMPLE = gc.toy_sample(SAMPLE_N, seed=SEED)
print(f"  the truth's log density on the same {EXPORT_M} by {EXPORT_M} grid, clipped at "
      f"-20 and written as float16 (worst rounding "
      f"{np.abs(truth16.astype(np.float64) - TRUTH_LOG).max():.2e})")

# ==================================================================== the JSON
payload = {
    "meta": {
        "seed": SEED, "seeds": list(SEEDS), "threads": THREADS,
        "torch": torch.__version__, "numpy": np.__version__,
        "hidden": HIDDEN, "tau": TAU, "updates": UPDATES, "batch": BATCH, "lr": LR,
        "train_grid": TRAIN_M, "score_grid": SCORE_M, "trace_grid": TRACE_M,
        "cd_step": CD_STEP, "buffer": BUFFER, "n_train": N_TRAIN, "n_held": N_HELD,
        "domain": DOMAIN, "view": VIEW,
        "note": "every figure on the page was measured by this file. No wall clock "
                "time is published: cost is reported as a count of energy "
                "evaluations, which is the same number on any machine, and the "
                "thread count is here because it changes the order of the floating "
                "point reductions.",
    },
    "truth": {
        "quadrature": QUAD,
        "entropy": {str(m): rnd(v, 6) for m, v in ENT.items()},
        "ceiling": rnd(CEILING, 6),
        "kl_self": sci(KL_SELF),
        "holes": HOLES,
        "outside_view": sci(OUT_VIEW_A),
        "outside_view_check": sci(OUT_VIEW_B),
        "region_mass": arr(REGION_MASS, 4),
        "ring_radius": rnd(RING_MU, 5), "ring_width": rnd(RING_SD, 5),
        "ring_width_coarse": rnd(RING_SD_C, 5), "band": list(ANNULUS),
        "band_mass": rnd(RING_MASS, 5),
        "ring_minimum": rnd(RING_MIN, 5),
        "sample": mat(SAMPLE, 3),
    },
    "arms": {
        "rows": ROWS,
        "floor": rnd(FLOOR_A, 5),
        "cd1_bias": rnd(CD1_BIAS, 5),
        "pcd1_bias": rnd(PCD1_BIAS, 5),
        "cd100_bias": rnd(CD100_BIAS, 5),
        "trace_every": TRACE_EVERY,
    },
    "gradient": {
        "rows": GRAD["rows"], "exact": GRAD["exact"],
        "repeats": GRAD_REPEATS, "batches": list(GRAD_BATCHES), "ks": list(GRAD_KS),
        "grid": GRAD_M, "check_grid": GRAD_CHECK_M,
        "params": int(sum(p.numel() for p in make_net(SEED).parameters())),
    },
    "sampler": {
        "rows": SAMPLER_ROWS,
        "chains": CHAINS, "burn": BURN, "kept": KEEP, "thin": THIN,
        "samples": int(N_SAMP), "bins": BINS, "rbins": RBINS,
        "bins_used": NBINS_USED,
        "floor_grid": sci(FLOOR_GRID), "floor_ring": sci(FLOOR_RING),
        "floor_grid_chains": sci(FLOOR_GRID_C), "floor_ring_chains": sci(FLOOR_RING_C),
        "floor_formula": FLOOR_FORMULA,
        "floor_grid_spread": sci(spread([f["kl_grid"] for f in FLOORS])),
        "fit_grid": FIT_GRID, "fit_ring": FIT_RING,
        "gaussian": GAUSS,
        "cold": [{"eps": e, "kl_grid": rnd(c["kl_grid"], 4), "weights": arr(c["weights"], 4)}
                 for e, c in sorted(COLD.items())],
    },
    "nonoise": {
        "rows": D_ROWS, "chains": D_CHAINS, "steps": D_STEPS,
        "ring_minimum": rnd(RING_MIN, 5),
        "endpoints": mat(D_SHOW["endpoints"], 3),
        "endpoints_eps": D_SHOW_EPS,
    },
    "classifier": {
        "accuracy": rnd(JUDGE["accuracy"], 4),
        "real_confidence": rnd(JUDGE["real_confidence"], 4),
        "hidden": JUDGE["hidden"], "pool_n": POOL_N,
        "pixel_mean": rnd(MU, 4), "pixel_sd": rnd(SD, 4),
        "ink_correlation": rnd(INK_CORR, 4),
        "rows": POOL_ROWS,
    },
    "model": {
        "format": "float16 little endian, base64; each layer's weights are row major "
                  "over output units, then its biases. The energy is "
                  "mlp(x) + |x|^2 / (2 tau^2).",
        "tau": TAU, "hidden": HIDDEN, "activation": "softplus",
        "grid": {"m": EXPORT_M, "half": VIEW, "cell": sci(CELLE),
                 "truth_logp_b64": truth_b64, "truth_clip": -20.0},
        "probes": mat(PROBES, 3),
        "arms": MODELS,
    },
    "guards": {
        "true_gradient": GRAD_GUARD,
        "radial_target_two_grids": sci(float(np.abs(TARGET_R - TARGET_R_C).max())),
        "ring_width_two_grids": sci(abs(RING_SD - RING_SD_C)),
        "auc_two_ways": sci(AUC_GAP),
        "softmax_vs_judge": sci(SOFT_GUARD),
        "kl_self": sci(KL_SELF),
        "digits_assert": {"worst": gc.DIGITS.get("assert_worst"),
                          "floor": gc.DIGITS.get("assert_floor")},
    },
}

def scrub(root):
    """json.dumps writes NaN and Infinity happily and JSON.parse refuses them,
    so a single division by zero six blocks up would take the whole page down
    with a syntax error. Anything not finite becomes null here, and says where
    it was, because a null in the JSON is a measurement that did not happen and
    somebody has to know."""
    found = []

    def walk(node, where):
        if isinstance(node, dict):
            return {k: walk(v, f"{where}.{k}") for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(v, f"{where}[{i}]") for i, v in enumerate(node)]
        if isinstance(node, float) and not math.isfinite(node):
            found.append(where)
            return None
        return node

    return walk(root, ""), found


payload, NOT_FINITE = scrub(payload)
if NOT_FINITE:
    print(f"  WARNING: {len(NOT_FINITE)} value(s) were not finite and went out as "
          f"null: {', '.join(NOT_FINITE[:8])}")
else:
    print("  every number in the payload is finite, so the page can parse it")
path = gc.write_json(OUT / "ebm.json", payload)
print("  stages: " + ", ".join(f"{k} {v:.0f}s" for k, v in STAGE_SECONDS.items()))
print(f"  total {sum(STAGE_SECONDS.values())/60:.1f} minutes on {THREADS} threads, "
      f"which is a fact about this machine and appears nowhere in the JSON")
