"""Data for the ATLAS denoising diffusion article: the noise that can be undone.

Every model in module 3.2 fought the same fight: the density it wanted was
intractable, so it approximated (a bound, an unnormalised score, a
diffeomorphism that cannot tear). Diffusion changes the question. It does not
try to write the density down at all. It destroys the data with a process whose
every step is a gaussian, and learns to undo one step at a time, which turns one
impossible problem into a thousand easy ones.

The reason this article can say more than a survey can is that the forward
process is closed form on the module's toy density:

    q(x_t | x_0) = N(sqrt(a_t) x_0, (1 - a_t) I)

so the marginal q(x_t) is the truth SCALED and CONVOLVED with a gaussian. Both
operations stay exact on the quadrature grid, which means this file can compute,
rather than estimate:

A. The forward marginal at every noise level, two independent ways, and the
   mass guard that catches the one thing an FFT convolution can get wrong.
B. What the noise actually destroys: the exact KL from the truth to a standard
   normal along the whole schedule, and the exact rate at which the ring's hole
   fills in. That hole is the same one the flows article measured and could not
   open, so the two articles are talking about the same object.
C. The prior term everybody assumes is zero: KL(q(x_T) || N(0, I)), exactly,
   against schedule and against T.
D. The TRUE score field, which the closed form gives us, against the one a
   network learns. Nobody doing this on images can make this comparison; here
   it is a subtraction.
E. Sampling: ancestral against deterministic, from a thousand steps down to
   two, scored against a floor measured from real data split in two.
F. The digits, small enough to sample in the browser from float16 weights, with
   the shared judge AND the ink guard that judge now travels with.

Run from anywhere: `python src/utils/generate_ddpm_data.py`.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from scipy.special import i0e

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.generative_common import (          # noqa: E402
    ROOT as REPO, SEED, TOY, arr, banner, c2st, cached, digits14, export_flat,
    export_half, judge, judge_ink_check, mat, rbf_mmd2, real_vs_real_floor, rnd, sci,
    threads, toy_entropy, toy_grid, toy_kl, toy_pdf, toy_quadrature_report,
    toy_sample, write_json, disc_weights, toy_hole_exact)

OUT = REPO / "ddpm" / "data"
THREADS = threads()

# ---------------------------------------------------------------- configuration
STEPS = 1000                  # the schedule the article calls "the full one"
GRID = 1024                   # the quadrature grid every exact number lives on
SHOW_TS = [0, 50, 100, 200, 400, 700, 999]
HID = 64
EPOCHS = 400
N_TRAIN, N_TEST = 20000, 5000
SEEDS = [19, 23, 29]
SEED_MAIN = SEEDS[0]

print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED_MAIN}")


# ================================================================ the schedules
def schedule(name, steps=STEPS):
    """beta_t, and the cumulative alpha the closed form actually needs."""
    if name == "linear":
        beta = np.linspace(1e-4, 0.02, steps)
    elif name == "cosine":
        # Nichol and Dhariwal's, written from its definition rather than copied
        s = 0.008
        t = np.linspace(0, 1, steps + 1)
        f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        ab = f / f[0]
        beta = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    elif name == "quadratic":
        beta = np.linspace(1e-4 ** 0.5, 0.02 ** 0.5, steps) ** 2
    else:
        raise ValueError(name)
    alpha = 1.0 - beta
    return beta, np.cumprod(alpha)


BETA, ABAR = schedule("linear")


# ============================================ the forward marginal, exactly
def _kernel(sigma, m, half):
    """An isotropic gaussian of width `sigma` sampled on the grid, in the
    wrapped order an FFT expects.

    The order is the whole trick. `fftfreq` already returns the wrapped
    coordinates, so applying a shift on top of it translates the answer by half
    a grid, which is a mistake that looks like a slightly blurry result rather
    than like an error. The two route guard below is what catches it.
    """
    off = np.fft.fftfreq(m, d=1.0 / m) * (2.0 * half / m)
    k1 = np.exp(-0.5 * (off / sigma) ** 2)
    k1 = k1 / k1.sum()
    return np.outer(k1, k1)


def forward_grid(t, m=GRID, abar=None):
    """q(x_t) on the quadrature grid, exactly.

    q(x_t) = (truth scaled by sqrt(abar)) convolved with N(0, (1-abar) I), and
    both halves are exact: the scaled truth is evaluated in closed form at every
    grid point (no samples, so no splatting and none of its variance), and the
    convolution is a multiplication in Fourier space.
    """
    abar = ABAR if abar is None else abar
    a = float(abar[t])
    half = TOY["domain"]
    X, cell, shape = toy_grid(m)
    sa = np.sqrt(a)
    # density of sqrt(a) * x0, which is p(u / sqrt(a)) / a in two dimensions
    field = (toy_pdf(X / sa) / a).reshape(shape)
    sig = np.sqrt(max(1.0 - a, 1e-12))
    h = 2.0 * half / m
    if sig < 0.75 * h:
        return field, cell, shape          # narrower than a cell: no blur to add
    K = _kernel(sig, m, half)
    dens = np.real(np.fft.ifft2(np.fft.fft2(field) * np.fft.fft2(K)))
    return np.maximum(dens, 0.0), cell, shape


def _ring_profile(t, rho, abar=None, nr=4001):
    """The ring's contribution to q(x_t), at the radii `rho`, in closed form up
    to one dimensional quadrature.

    The ring is rotationally symmetric and so is the blur, so the result is a
    function of the radius alone and the angular integral has a name: it is a
    modified Bessel function. What is left is an integral over the ring's own
    radius, done here on a fine one dimensional grid, with I0 in its
    exponentially scaled form so the exponent never overflows.
    """
    abar = ABAR if abar is None else abar
    a = float(abar[t])
    sa = np.sqrt(a)
    v = max(1.0 - a, 1e-12)
    R = TOY["ring"]
    r0, sg = sa * R["r0"], sa * R["sigma"]
    lo, hi = max(r0 - 9.0 * sg, 0.0), r0 + 9.0 * sg
    r = np.linspace(lo, hi, nr)
    pr = np.exp(-0.5 * ((r - r0) / sg) ** 2)
    pr = pr / np.trapezoid(pr, r)
    rho = np.asarray(rho, dtype=float).ravel()
    out = np.empty(len(rho))
    step = max(1, int(2e7 // nr))                 # keep the outer product small
    for s in range(0, len(rho), step):
        rr = rho[s:s + step]
        z = np.outer(rr, r) / v
        logk = (-(rr[:, None] ** 2 + r[None, :] ** 2) / (2 * v) + z
                + np.log(np.maximum(i0e(z), 1e-300)))
        out[s:s + step] = np.trapezoid(np.exp(logk) * pr[None, :] / (2 * np.pi * v), r, axis=1)
    return out


def forward_direct(pts, t, abar=None, profile=None):
    """q(x_t) at chosen points, in closed form.

    This is the PRIMARY route, and the grid convolution is the check on it,
    which is the opposite of how this file started. The reason is block A: a
    fixed quadrature grid cannot follow a density that is being scaled towards
    a delta, and by the end of the schedule the scaled truth is three cells
    wide. This route has no grid to lose: the blobs are gaussians convolved
    with a gaussian, which is a gaussian, and the ring reduces to a radial
    profile that is the same at every angle.

    `profile` optionally supplies a precomputed (radii, values) pair to
    interpolate the ring from, which is what makes a million point grid cheap.
    """
    abar = ABAR if abar is None else abar
    a = float(abar[t])
    sa = np.sqrt(a)
    v = max(1.0 - a, 1e-12)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = np.zeros(len(pts))

    for name in ("blob_a", "blob_b"):
        B = TOY[name]
        th = np.radians(B["angle"])
        Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        s = np.asarray(B["sd"], dtype=float)
        C = a * (Q @ np.diag(s ** 2) @ Q.T) + v * np.eye(2)
        mu = sa * np.asarray(B["mean"], dtype=float)
        d = pts - mu
        q = (d @ np.linalg.inv(C) * d).sum(1)
        out += B["weight"] * np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(np.linalg.det(C)))

    rho = np.sqrt((pts ** 2).sum(1))
    if profile is None:
        ring = _ring_profile(t, rho, abar)
    else:
        # A cubic spline, not a linear interpolation. The ring's profile at the
        # start of the schedule has features at the scale of its own width, and
        # straight lines between samples cost 2.2e-04 there, which is far too
        # much to keep calling the result exact. Cubic is fourth order in the
        # same spacing and costs nothing extra; it can undershoot past the tail,
        # so the result is clipped at zero, which a density may be.
        spl, rmax = profile
        ring = np.where(rho <= rmax, spl(np.minimum(rho, rmax)), 0.0)
        ring = np.maximum(ring, 0.0)
    return out + TOY["ring"]["weight"] * ring


def forward_domain(t, abar=None):
    """Half width of a square that holds essentially all of q(x_t).

    The truth is inside 6.5, the scaling shrinks that by sqrt(abar), and the
    blur adds a gaussian of width sqrt(1 - abar): eight of those is far past
    anything that matters, and the mass guard checks it rather than trusting it.
    """
    abar = ABAR if abar is None else abar
    a = float(abar[t])
    return float(np.sqrt(a) * TOY["domain"] + 8.0 * np.sqrt(max(1.0 - a, 1e-12)))


def forward_field(t, m=GRID, abar=None):
    """q(x_t) on a grid sized for THIS t, plus the grid and the cell area.

    Returns (X, dens, cell, shape, half). Every exact integral in this file runs
    on one of these: a grid that follows the density instead of a fixed one that
    the density walks out of.
    """
    half = forward_domain(t, abar)
    e = np.linspace(-half, half, m + 1)
    c = 0.5 * (e[1:] + e[:-1])
    gx, gy = np.meshgrid(c, c, indexing="ij")
    X = np.c_[gx.ravel(), gy.ravel()]
    rmax = float(np.sqrt(2.0) * half) * 1.001
    pr_r = np.linspace(0.0, rmax, 12000)
    prof = (CubicSpline(pr_r, _ring_profile(t, pr_r, abar)), rmax)
    dens = forward_direct(X, t, abar, profile=prof)
    return X, dens, float((2.0 * half / m) ** 2), (m, m), half


def log_grid(dens):
    return np.log(np.maximum(np.asarray(dens).ravel(), 1e-300))


# ============================================================== A. the guards
banner("A. the forward marginal, two ways, and what the grid is worth")
quad = toy_quadrature_report()
print(f"  the truth integrates to {quad[1]['mass']} on the {GRID} grid "
      f"(error {quad[1]['error']})")

guard_rows = []
for t in SHOW_TS:
    X, dens, cell, shape, half = forward_field(t)
    mass = float(dens.sum() * cell)
    rs = np.random.RandomState(5 + t)
    idx = rs.choice(len(X), 300, replace=False)
    # (i) the interpolated ring against the Bessel quadrature evaluated exactly
    exact = forward_direct(X[idx], t)
    got = dens[idx]
    keep = exact > 1e-6
    interp = float(np.max(np.abs(got[keep] - exact[keep]) / exact[keep])) if keep.any() else 0.0
    # (ii) the grid convolution, which is only meaningful while the grid can
    #      still resolve the scaled truth: that is the whole point of block A
    fft_rel = None
    a = float(ABAR[t])
    if np.sqrt(a) * TOY["domain"] > 40.0 * (2.0 * TOY["domain"] / GRID):
        fdens, fcell, fshape = forward_grid(t)
        Xf, _, _ = toy_grid(GRID)
        ex = forward_direct(Xf[idx], t)
        gv = fdens.ravel()[idx]
        k2 = ex > 1e-6
        fft_rel = float(np.max(np.abs(gv[k2] - ex[k2]) / ex[k2])) if k2.any() else 0.0
    guard_rows.append({"t": t, "abar": rnd(a, 6), "half": rnd(half, 3),
                       "mass": rnd(mass, 8), "interp": sci(interp),
                       "fft": None if fft_rel is None else sci(fft_rel)})
    print(f"  t={t:>4}: abar {a:.6f}, box {half:5.2f}, mass {mass:.8f}, "
          f"interpolation {interp:.2e}, "
          + (f"grid convolution {fft_rel:.2e}" if fft_rel is not None
             else "grid convolution not attempted (the scaled truth is under 40 cells wide)"))

worst_mass = max(abs(r["mass"] - 1.0) for r in guard_rows)
worst_interp = max(r["interp"] for r in guard_rows)
fft_checked = [r for r in guard_rows if r["fft"] is not None]
worst_fft = max(r["fft"] for r in fft_checked) if fft_checked else None
assert worst_mass < 1e-5, (
    f"q(x_t) loses {worst_mass:.2e} of its mass on its own box, so either the box is too "
    f"small or the closed form is wrong")
assert worst_interp < 1e-6, (
    f"interpolating the ring's radial profile costs {worst_interp:.2e}, which is too much "
    f"to call the result exact")
assert worst_fft is None or worst_fft < 5e-3, (
    f"the grid convolution and the closed form disagree by {worst_fft:.2e} where the grid "
    f"should still be able to resolve the scaled truth")
print(f"\n  worst mass error {worst_mass:.2e}, worst interpolation {worst_interp:.2e}, "
      f"worst grid convolution {worst_fft:.2e} over {len(fft_checked)} of {len(SHOW_TS)} levels")

CEILING = toy_entropy(GRID)
print(f"  the truth's entropy, the ceiling for the whole module: {CEILING:.6f} nats")


# ================================================ B. what the noise destroys
banner("B. what the noise destroys, exactly, and how fast")


def kl_to_normal(t, m=512, abar=None):
    """KL(q(x_t) || N(0, I)), exactly.

    This is the quantity the whole method is betting on. Training assumes that
    by the end of the schedule the data is gone and sampling may start from a
    standard normal; that assumption is a number, and this computes it instead
    of asserting it.
    """
    X, dens, cell, shape, half = forward_field(t, m, abar)
    lq = log_grid(dens)
    lp = -0.5 * (X ** 2).sum(1) - np.log(2.0 * np.pi)
    return float((dens * (lq - lp)).sum() * cell)


def hole_mass(t, radius=1.2, m=512, abar=None):
    """The mass q(x_t) puts inside the ring's hole.

    The flows article measured exactly this circle: the truth puts 1.417e-07
    there and the deepest flow it could fit put 0.001004, because a smooth
    invertible map cannot tear a hole open. Diffusion fills it on purpose, and
    this is the rate.
    """
    X, dens, cell, shape, half = forward_field(t, m, abar)
    # Weighted by the fraction of each cell inside the circle, not by whether
    # its centre is: at t=0 the mass in there is 1.4e-07, and a staircase drawn
    # over the boundary is then the entire answer. The flows article published a
    # number that moved with the resolution for exactly this reason.
    return float((dens * disc_weights(X, np.sqrt(cell), radius)).sum() * cell)


TRACK = [0, 10, 25, 50, 100, 150, 200, 300, 400, 550, 700, 850, 999]
destroy = []
for t in TRACK:
    k = kl_to_normal(t)
    h = hole_mass(t)
    destroy.append({"t": t, "abar": rnd(float(ABAR[t]), 6), "kl_normal": rnd(k, 5),
                    "hole": sci(h)})
    print(f"  t={t:>4}: abar {ABAR[t]:.6f}  KL to a standard normal {k:9.5f} nats  "
          f"hole {h:.3e}")

TRUTH_HOLE = destroy[0]["hole"]
FLOWS_HOLE = 0.001004                 # what the flows article's deepest flow managed
half_t = next((r["t"] for r in destroy if r["kl_normal"] <= destroy[0]["kl_normal"] / 2), None)
fill_t = next((r["t"] for r in destroy if r["hole"] >= FLOWS_HOLE), None)
print(f"\n  at t=0 the hole holds {TRUTH_HOLE:.3e}, which is the truth's own number")
print(f"  the KL to a standard normal is halved by t={half_t}")
print(f"  the hole passes {FLOWS_HOLE} (what the flows article's best flow put there) "
      f"at t={fill_t}, and it gets there by blurring rather than by tearing")


# ============================================ C. the term everyone assumes away
banner("C. the prior term, which is assumed to be zero and is not")
prior = {}
for name in ("linear", "cosine", "quadratic"):
    b, ab = schedule(name)
    rows = []
    for T in (100, 250, 500, 1000):
        k = kl_to_normal(T - 1, 512, ab)
        rows.append({"T": T, "abar_T": sci(float(ab[T - 1])), "kl": sci(k)})
    prior[name] = {"rows": rows, "abar_end": sci(float(ab[-1])),
                   "kl_end": sci(kl_to_normal(STEPS - 1, 512, ab))}
    print(f"  {name:<10}: abar at the end {ab[-1]:.3e}, "
          f"KL(q(x_T) || N(0,I)) = {prior[name]['kl_end']:.3e} nats")
    for r in rows:
        print(f"      T={r['T']:>4}: abar {r['abar_T']:.3e}, prior term {r['kl']:.3e} nats")

best_prior = min(prior, key=lambda n: prior[n]["kl_end"])
print(f"\n  the schedule that leaves least behind at t=T is {best_prior} "
      f"({prior[best_prior]['kl_end']:.3e} nats)")


# ================================================= D. the score, true and learned
banner("D. the true score field against a learned one")


def true_score(pts, t, abar=None):
    """grad log q(x_t), exactly.

    Every term of q is differentiable in closed form. The blobs are gaussians,
    whose gradient is the gaussian times minus the inverse covariance applied to
    the offset. The ring is a radial profile, so its gradient points along the
    radius with magnitude the profile's own derivative, and a cubic spline hands
    that over for free. This is the field a network is trying to learn, and on
    images nobody can write it down.
    """
    abar = ABAR if abar is None else abar
    a = float(abar[t])
    sa, v = np.sqrt(a), max(1.0 - a, 1e-12)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    dens = np.zeros(len(pts))
    grad = np.zeros_like(pts)

    for name in ("blob_a", "blob_b"):
        B = TOY[name]
        th = np.radians(B["angle"])
        Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        sd = np.asarray(B["sd"], dtype=float)
        C = a * (Q @ np.diag(sd ** 2) @ Q.T) + v * np.eye(2)
        Ci = np.linalg.inv(C)
        d = pts - sa * np.asarray(B["mean"], dtype=float)
        val = (B["weight"] * np.exp(-0.5 * (d @ Ci * d).sum(1))
               / (2.0 * np.pi * np.sqrt(np.linalg.det(C))))
        dens += val
        grad += -(d @ Ci) * val[:, None]

    rho = np.sqrt((pts ** 2).sum(1))
    rmax = float(rho.max()) * 1.001 + 1e-9
    pr_r = np.linspace(0.0, rmax, 12000)
    spl = CubicSpline(pr_r, _ring_profile(t, pr_r, abar))
    w = TOY["ring"]["weight"]
    dens += w * np.maximum(spl(rho), 0.0)
    unit = pts / np.maximum(rho, 1e-12)[:, None]
    grad += w * spl(rho, 1)[:, None] * unit
    return grad / np.maximum(dens, 1e-300)[:, None]


class Score(nn.Module):
    """eps_theta(x, t): the standard parameterisation, predicting the noise that
    was added rather than the score itself. They differ by a known factor, which
    is what makes the comparison in this block possible at all."""

    def __init__(self, hid=HID):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(),
                                 nn.Linear(hid, 2))

    @staticmethod
    def embed(t):
        """Sinusoidal in log space, which is what makes one network cover a
        thousand noise levels instead of memorising a few."""
        f = torch.exp(torch.linspace(0.0, 6.0, 8, device=t.device))
        a = t[:, None].float() / STEPS * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.embed(t)], 1))


def train_score(seed, epochs=EPOCHS, hid=HID, abar=None, steps=STEPS, tag="linear"):
    ab = ABAR if abar is None else abar

    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        X = torch.tensor(toy_sample(N_TRAIN, seed=seed), dtype=torch.float32)
        abt = torch.tensor(ab, dtype=torch.float32)
        net = Score(hid)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=g)
            for s in range(0, len(X), 512):
                xb = X[perm[s:s + 512]]
                t = torch.randint(0, steps, (len(xb),), generator=g)
                eps = torch.randn(xb.shape, generator=g)
                at = abt[t][:, None]
                xt = at.sqrt() * xb + (1 - at).sqrt() * eps
                opt.zero_grad()
                loss = ((net(xt, t) - eps) ** 2).mean()
                loss.backward()
                opt.step()
            sched.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()],
                "loss": float(loss)}

    out = cached(f"ddpm-toy-{tag}-s{seed}-h{hid}-e{epochs}-k{steps}-n{N_TRAIN}-v1", build)
    net = Score(hid)
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out


NET, NET_META = train_score(SEED_MAIN)
print(f"  trained: final batch loss {NET_META['loss']:.5f}")


def learned_score(net, pts, t, abar=None):
    ab = ABAR if abar is None else abar
    v = max(1.0 - float(ab[t]), 1e-12)
    with torch.no_grad():
        x = torch.tensor(np.asarray(pts, dtype=float), dtype=torch.float32)
        tt = torch.full((len(x),), int(t), dtype=torch.long)
        eps = net(x, tt).numpy()
    return -eps / np.sqrt(v)            # score = -eps / sigma_t, by definition


score_rows = []
for t in TRACK:
    X, dens, cell, shape, half = forward_field(t, 256)
    keep = dens > (dens.max() * 1e-4)          # where there is anything to learn
    P = X[keep]
    w = dens[keep] * cell
    w = w / w.sum()
    S = true_score(P, t)
    L = learned_score(NET, P, t)
    dot = (S * L).sum(1)
    ns, nl = np.linalg.norm(S, axis=1), np.linalg.norm(L, axis=1)
    cos = dot / np.maximum(ns * nl, 1e-30)
    rel = np.linalg.norm(L - S, axis=1) / np.maximum(ns, 1e-30)
    score_rows.append({
        "t": t, "cos": rnd(float((cos * w).sum()), 4),
        "rel": rnd(float((rel * w).sum()), 4),
        "true_norm": rnd(float((ns * w).sum()), 3),
        "learned_norm": rnd(float((nl * w).sum()), 3),
    })
    print(f"  t={t:>4}: cosine {score_rows[-1]['cos']:+.4f}, relative error "
          f"{score_rows[-1]['rel']:.4f}, the true score has length "
          f"{score_rows[-1]['true_norm']:.3f} on average")

# A guard the result handed over for free. By the end of the schedule q(x_t) is
# a standard normal, whose score is -x, so the mean length of the true score has
# to be E|x| for a 2-D standard normal, which is sqrt(pi/2). If the closed form
# above were wrong anywhere, there is no reason it would land on that constant.
END_NORM = float(np.sqrt(np.pi / 2.0))
end_row = next(r for r in score_rows if r["t"] == TRACK[-1])
end_gap = abs(end_row["true_norm"] - END_NORM)
assert end_gap < 5e-3, (
    f"at t={TRACK[-1]} the true score should have mean length sqrt(pi/2) = {END_NORM:.4f}, "
    f"because q(x_t) is a standard normal by then, and it has {end_row['true_norm']}")
print(f"  and the guard the answer gives away: at t={TRACK[-1]} the true score's mean length is "
      f"{end_row['true_norm']} against sqrt(pi/2) = {END_NORM:.4f}, which is what it must be "
      f"once the data is gone")

worst = max(score_rows, key=lambda r: r["rel"])
best = min(score_rows, key=lambda r: r["rel"])
print(f"\n  hardest at t={worst['t']} (relative error {worst['rel']}, true length "
      f"{worst['true_norm']}), easiest at t={best['t']} ({best['rel']})")


# ============================================ E. sampling, and what steps buy
banner("E. sampling: ancestral against deterministic, from 1000 steps down to 2")


def sample(net, n, steps_used, seed, mode="ancestral", abar=None, eta=None):
    """Run the reverse process with `steps_used` of the schedule's levels.

    Ancestral is the original: each step adds noise back. Deterministic is DDIM,
    the same trained network with the noise term removed, which is a different
    sampler for the SAME model rather than a different model.
    """
    ab = ABAR if abar is None else abar
    idx = np.linspace(len(ab) - 1, 0, steps_used).round().astype(int)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    with torch.no_grad():
        for j, t in enumerate(idx):
            a_t = float(ab[t])
            a_prev = float(ab[idx[j + 1]]) if j + 1 < len(idx) else 1.0
            tt = torch.full((n,), int(t), dtype=torch.long)
            eps = net(x, tt)
            x0 = (x - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)
            if mode == "ancestral":
                # the variance of q(x_{t-1} | x_t, x_0), which is the posterior
                # the model is trained to match
                var = (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev) if a_prev < 1.0 else 0.0
                mean = (np.sqrt(a_prev) * (1 - a_t / a_prev) / (1 - a_t) * x0
                        + np.sqrt(a_t / a_prev) * (1 - a_prev) / (1 - a_t) * x)
                x = mean + (np.sqrt(var) * torch.randn(x.shape, generator=g) if var > 0 else 0.0)
            else:
                x = np.sqrt(a_prev) * x0 + np.sqrt(max(1 - a_prev, 0.0)) * eps
    return x.numpy()


REAL = toy_sample(4000, seed=777)
GAMMA = 1.0 / (2.0 * float(np.median(
    np.linalg.norm(REAL[:800, None, :] - REAL[None, :800, :], axis=2)) ** 2))
FLOOR = real_vs_real_floor(REAL, GAMMA, splits=5, n=800, seed=31)
print(f"  the floor, from real draws split in two {FLOOR['splits']} times: two sample statistic "
      f"{FLOOR['mmd2']} (largest {FLOOR['mmd2_max']}), separable at {FLOOR['c2st']}")

STEP_LIST = [2, 5, 10, 25, 50, 100, 250, 1000]
steps_rows = []
for mode in ("ancestral", "deterministic"):
    for k in STEP_LIST:
        S = sample(NET, 800, k, 4242, mode)
        m = rbf_mmd2(S, REAL[:1600], GAMMA)
        c = c2st(S, REAL[:800], seed=1)
        hole = float(((S ** 2).sum(1) <= 1.2 ** 2).mean())
        steps_rows.append({"mode": mode, "steps": k, "mmd2": sci(m), "c2st": rnd(c, 4),
                           "hole_share": sci(hole),
                           "resolves": bool(abs(m) > FLOOR["mmd2_max"])})
        print(f"  {mode:<14} {k:>4} steps: two sample {m:+.3e}, separable at {c:.4f}, "
              f"{hole * 100:5.2f}% of draws inside the hole")

# "The best" is the wrong reading once a curve is inside its own floor: picking
# the smallest |mmd2| there is picking the luckiest seed. The claim that means
# something is the FIRST step count from which it is inside the floor and stays
# inside for every larger count.
def enters(mode):
    rows = [r for r in steps_rows if r["mode"] == mode]
    for i, r in enumerate(rows):
        if all(abs(x["mmd2"]) <= FLOOR["mmd2_max"] for x in rows[i:]):
            return r["steps"]
    return None


enter_anc, enter_det = enters("ancestral"), enters("deterministic")
print(f"\n  ancestral is inside the floor from {enter_anc} steps onwards")
print(f"  deterministic is inside the floor from {enter_det} steps onwards")

# The hole needs its own measurement, because a share cannot be resolved below
# one draw: 800 draws cannot tell 0.001 from 0.0001, and both of those are
# interesting numbers here. The flows article's deepest flow put 0.001006 in
# this circle and could not do better, because a smooth invertible map cannot
# tear a gaussian open. A diffusion sampler is not a diffeomorphism, so the
# question is whether it actually does better or merely could.
TRUTH_HOLE_SHARE = 1.433262e-07
N_HOLE = 200000
hole_rows = []
for mode in ("ancestral", "deterministic"):
    S = sample(NET, N_HOLE, 100, 9091, mode)
    inside = int(((S ** 2).sum(1) <= 1.2 ** 2).sum())
    share = inside / N_HOLE
    # a share from n draws has this much resolution, and below it nothing is said
    res = 1.0 / N_HOLE
    hole_rows.append({"mode": mode, "n": N_HOLE, "inside": inside, "share": sci(share),
                      "resolution": sci(res), "vs_flow": rnd(share / 0.001006, 4),
                      "vs_truth": sci(share / TRUTH_HOLE_SHARE)})
    print(f"  {mode:<14}: {inside} of {N_HOLE:,} draws inside the hole, a share of {share:.3e} "
          f"(resolution {res:.0e}), against 0.001006 for the flows article's deepest flow "
          f"and {TRUTH_HOLE_SHARE:.3e} for the truth")




# ============================================ F. the same thing on 196 pixels
banner("F. the wall: what pixel space costs at 14 by 14")
X_DIG, Y_DIG, TR, TE = digits14()
J = judge()
print(f"  the module's shared judge: {J['accuracy'] * 100:.2f}% on the held out digits and "
      f"{J['real_confidence']:.4f} confident on them")

DIG_STEPS = 200
_, DABAR = schedule("cosine", DIG_STEPS)


class PixelScore(nn.Module):
    def __init__(self, dim, hid, depth):
        super().__init__()
        layers = [nn.Linear(dim + 32, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        layers += [nn.Linear(hid, dim)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def embed(t):
        f = torch.exp(torch.linspace(0.0, float(np.log(200.0)), 16, device=t.device))
        a = t[:, None].float() / DIG_STEPS * f[None, :] * np.pi
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.embed(t)], 1))


def train_pixels(hid, depth, epochs, seed=SEED_MAIN):
    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        Xt = torch.tensor(X_DIG[TR] * 2.0 - 1.0, dtype=torch.float32)
        abt = torch.tensor(DABAR, dtype=torch.float32)
        net = PixelScore(Xt.shape[1], hid, depth)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for s in range(0, len(Xt), 128):
                xb = Xt[perm[s:s + 128]]
                t = torch.randint(0, DIG_STEPS, (len(xb),), generator=g)
                eps = torch.randn(xb.shape, generator=g)
                at = abt[t][:, None]
                xt = at.sqrt() * xb + (1 - at).sqrt() * eps
                opt.zero_grad()
                loss = ((net(xt, t) - eps) ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.detach())
            sch.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()], "loss": last}

    out = cached(f"ddpm-pix-cos{DIG_STEPS}-s{seed}-h{hid}-d{depth}-e{epochs}-v2", build)
    net = PixelScore(X_DIG.shape[1], hid, depth)
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out, sum(p.numel() for p in net.parameters())


def sample_pixels(net, n, steps_used, seed):
    """The reverse process on pixels, with the one correction that a clamp
    forces.

    Clamping the predicted clean image to the range a picture can have is
    standard and necessary. What is NOT optional is recomputing the noise from
    the clamped image: the update is built on the identity
    x_t = sqrt(a) x0 + sqrt(1-a) eps, so keeping the old eps after changing x0
    breaks it, and the chain leaves the data manifold on the first step and
    never comes back. Measured: without this, 200 steps return grey mush with a
    mean pixel of 0.48 against 0.13 for a real digit.
    """
    idx = np.linspace(DIG_STEPS - 1, 0, steps_used).round().astype(int)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, X_DIG.shape[1], generator=g)
    with torch.no_grad():
        for j, t in enumerate(idx):
            a_t = float(DABAR[t])
            a_prev = float(DABAR[idx[j + 1]]) if j + 1 < len(idx) else 1.0
            eps = net(x, torch.full((n,), int(t), dtype=torch.long))
            x0 = ((x - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)).clamp(-1, 1)
            eps = (x - np.sqrt(a_t) * x0) / np.sqrt(max(1 - a_t, 1e-12))
            x = np.sqrt(a_prev) * x0 + np.sqrt(max(1 - a_prev, 0.0)) * eps
    return ((x.numpy() + 1.0) / 2.0).clip(0, 1)


REALPIX = X_DIG[TE]
PGAMMA = 1.0 / (2.0 * float(np.median(
    np.linalg.norm(REALPIX[:300, None, :] - REALPIX[None, :300, :], axis=2)) ** 2))
PFLOOR = real_vs_real_floor(REALPIX, PGAMMA, splits=5, n=250, seed=17)
print(f"  the pixel floor, real against real: two sample {PFLOOR['mmd2']} "
      f"(largest {PFLOOR['mmd2_max']}), separable at {PFLOOR['c2st']}")

LADDER = [(192, 2, 600), (384, 3, 2000), (512, 3, 4000)]
ladder_rows, pools, nets = [], [], {}
for hid, depth, ep in LADDER:
    net, meta, npar = train_pixels(hid, depth, ep)
    S = sample_pixels(net, 250, 100, 7)
    m = rbf_mmd2(S, REALPIX[:500], PGAMMA)
    c = c2st(S, REALPIX[:250], seed=2)
    key = f"{npar // 1000}k"
    ladder_rows.append({"hidden": hid, "depth": depth, "epochs": ep, "params": npar,
                        "kb_half": rnd(npar * 2 / 1024, 1), "loss": rnd(meta["loss"], 4),
                        "mmd2": sci(m), "c2st": rnd(c, 4), "ink": sci(float(S.mean())),
                        "resolves": bool(abs(m) > PFLOOR["mmd2_max"])})
    pools.append((key, S))
    nets[npar] = (net, S)
    print(f"  {npar:>7,} weights ({npar * 2 / 1024:6.1f} kB in float16), {ep:>4} epochs: "
          f"two sample {m:+.4f}, separable at {c:.4f}, mean pixel {S.mean():.4f}")

REAL_INK = float(REALPIX.mean())
print(f"  a real held out digit has mean pixel {REAL_INK:.4f}")
INK = judge_ink_check(pools, REALPIX, J)
print(f"\n  the ink guard on the shared judge: confidence correlates {INK['corr']} with ink; "
      f"{'CONFOUNDED' if INK['confounded'] else 'not confounded'}")

BIG = max(nets)
best_row = max(ladder_rows, key=lambda r: r["params"])
print(f"\n  The toy on this page runs from {sum(p.numel() for p in NET.parameters()):,} weights. "
      f"Reaching {best_row['mmd2']} on 196 pixels took {BIG:,}, which is "
      f"{BIG / sum(p.numel() for p in NET.parameters()):,.0f} times as many and "
      f"{best_row['kb_half']} kB: too much to ship in a page, and that is the measurement, "
      f"not an excuse. It is also the argument for doing the diffusion somewhere smaller.")

# What the page shows instead of weights: the pictures themselves, which cost
# 196 numbers each rather than three quarters of a million.
STRIP = 24
show = nets[BIG][1][:STRIP]
strip_b64, strip_v = export_flat(show.ravel().tolist())
real_b64, real_v = export_flat(REALPIX[:STRIP].ravel().tolist())
print(f"  the page ships {STRIP} samples and {STRIP} real digits as pixels: "
      f"{len(strip_v) + len(real_v):,} numbers, {(len(strip_b64) + len(real_b64)) / 1024:.0f} kB")


# ============================================================== G. the scoreboard
banner("G. the module, side by side, on the same density")
import json as _json

board = {"ceiling": rnd(CEILING, 6)}
fj = REPO / "flows" / "data" / "flows.json"
if fj.exists():
    fb = _json.loads(fj.read_text("utf-8")).get("board", {})
    for k in ("flow", "vae", "ebm", "truth_hole"):
        if k in fb:
            board[k] = fb[k]
            print(f"  from the flows article: {k} = {fb[k]}")

# This article's own row is a different SHAPE, and that is the point rather than
# an omission. A flow reports an exact log density because a change of variables
# is an identity; a variational autoencoder reports a bound it can name the gap
# of; a diffusion sampler reports neither, because there is no closed form for
# the density its reverse chain induces. What it has instead is samples, so the
# only honest column it can fill is the sample one, read against a floor.
board["ddpm"] = {
    "exact": None,
    "why_no_exact": "the reverse chain has no closed form density, so this article cannot put "
                    "a number in the column the flow filled exactly and the autoencoder "
                    "bounded; what it has is samples",
    "mmd2_best": min((r["mmd2"] for r in steps_rows), key=abs),
    "floor": FLOOR["mmd2_max"],
    "enter_deterministic": enter_det, "enter_ancestral": enter_anc,
    "hole": hole_rows[0]["share"] if hole_rows else None,
    "hole_deterministic": hole_rows[1]["share"] if len(hole_rows) > 1 else None,
}
print(f"  this article: no exact density (the reverse chain has none), sample statistic "
      f"{board['ddpm']['mmd2_best']} against a floor of {board['ddpm']['floor']}")

# ================================================ what the browser runs
banner("H. exporting the toy process so the page can run it")
# the module's export contract, unchanged: W is (inputs, outputs) and is
# written row major over OUTPUT units, so the page reads a layer with a stride
blocks = []
for i, lin in enumerate([NET.net[0], NET.net[2], NET.net[4], NET.net[6]]):
    blocks.append((f"l{i}", lin.weight.detach().numpy().T, lin.bias.detach().numpy()))
net_spec, net_b64, net_v = export_half(blocks)
print(f"  the score network: {len(net_v):,} numbers, float16 base64 is {len(net_b64) / 1024:.0f} kB")

# The page recomputes the exact density itself rather than being handed a
# picture of it: the two blobs are gaussians convolved with a gaussian, which is
# a gaussian and is four lines of JavaScript, and the ring is radial, so all it
# needs is that one radial profile per level. 400 radii instead of a 64 by 64
# field per level, and what it draws is exact rather than resampled.
PAGE_TS = [0, 100, 250, 400, 600, 800, 999]
RHO_MAX, RHO_N = 9.0, 1000
rho_axis = np.linspace(0.0, RHO_MAX, RHO_N)
prof_flat, deriv_flat = [], []
for t in PAGE_TS:
    # The DERIVATIVE ships too, and that is not a convenience. The page needs
    # the score, which is the profile's slope, and a slope recovered by
    # differencing a linearly interpolated table is a staircase: with 400
    # samples the page's score came out 168% wrong while its density was only
    # 6% wrong. Shipping the spline's own derivative costs one more array and
    # makes both of them right.
    spl = CubicSpline(rho_axis, _ring_profile(t, rho_axis))
    prof_flat.extend(spl(rho_axis).tolist())
    deriv_flat.extend(spl(rho_axis, 1).tolist())
prof_b64, prof_v = export_flat(prof_flat)
deriv_b64, deriv_v = export_flat(deriv_flat)
print(f"  the ring's radial profile and its slope at {len(PAGE_TS)} levels: "
      f"{len(prof_v) + len(deriv_v):,} numbers, "
      f"{(len(prof_b64) + len(deriv_b64)) / 1024:.0f} kB")

# And a probe the page checks itself against: if it has read the weights or the
# schedule wrongly, these will not match, however plausible its pictures look.
# The probe points are DRAWN FROM q(x_t) rather than spread over the square, and
# that is not a convenience either. Uniform points land where the density is
# 1e-25, twenty orders of magnitude under the median, and comparing a score
# there measures what float16 can carry rather than whether the page read the
# file correctly: the first version of this guard failed at 168% on a point
# whose density it had itself rounded to zero. Sampling the points the way the
# process does puts the guard where every number in the article is evaluated.
rs = np.random.RandomState(404)
base_x0 = toy_sample(40, seed=404)
probe = {"points": [], "t": PAGE_TS, "q": [], "score": [], "eps": []}
for t in PAGE_TS:
    a = float(ABAR[t])
    pts = np.sqrt(a) * base_x0 + np.sqrt(max(1 - a, 1e-12)) * rs.randn(40, 2)
    probe["points"].append(mat(pts, 4))
    # scientific notation, not six decimal places: a density of 8e-25 written
    # to six decimals is the number zero, and then nothing can be checked
    probe["q"].append([sci(v, 6) for v in forward_direct(pts, t)])
    probe["score"].append(mat(true_score(pts, t), 4))
    probe["eps"].append(mat(-learned_score(NET, pts, t)
                           * np.sqrt(max(1 - a, 1e-12)), 4))
print(f"  a {len(base_x0)} point probe at each level, drawn from q(x_t) itself so the guard "
      f"sits where the article's numbers are")

payload = {
    "meta": {"seed": SEED_MAIN, "seeds": SEEDS, "threads": THREADS,
             "torch": torch.__version__, "steps": STEPS, "grid": GRID,
             "hidden": HID, "epochs": EPOCHS, "n_train": N_TRAIN,
             "toy_params": int(sum(p.numel() for p in NET.parameters())),
             "ceiling": rnd(CEILING, 6), "show_ts": SHOW_TS, "track": TRACK,
             "digit_steps": DIG_STEPS, "digit_side": 14, "strip": STRIP,
             "real_ink": rnd(REAL_INK, 4),
             "judge_accuracy": rnd(J["accuracy"], 4),
             "judge_confidence": rnd(J["real_confidence"], 4),
             "note": "q(x_t) is the truth scaled and blurred, so every number here that calls "
                     "itself exact is a closed form or a quadrature on a box sized for that "
                     "noise level, never an average over samples"},
    "quadrature": quad,
    "guards": {"rows": guard_rows, "worst_mass": sci(worst_mass),
               "worst_interp": sci(worst_interp),
               "worst_fft": None if worst_fft is None else sci(worst_fft),
               "fft_levels": len(fft_checked), "levels": len(SHOW_TS),
               "end_score_norm": rnd(end_row["true_norm"], 4),
               "end_score_target": rnd(END_NORM, 4)},
    "destroy": {"rows": destroy, "half_t": half_t, "fill_t": fill_t,
                "truth_hole": sci(TRUTH_HOLE_SHARE), "flow_hole": 0.001006},
    "prior": prior, "best_prior": best_prior,
    "score": {"rows": score_rows, "hardest": worst["t"], "easiest": best["t"],
              "worst_rel": rnd(worst["rel"], 4), "best_rel": rnd(best["rel"], 4)},
    "sampling": {"rows": steps_rows, "floor": FLOOR, "gamma": sci(GAMMA),
                 "enter_ancestral": enter_anc, "enter_deterministic": enter_det,
                 "hole": hole_rows},
    "pixels": {"ladder": ladder_rows, "floor": PFLOOR, "gamma": sci(PGAMMA),
               "ink": INK, "real_ink": rnd(REAL_INK, 4),
               "biggest": BIG, "toy_params": int(sum(p.numel() for p in NET.parameters())),
               "samples_b64": strip_b64, "real_b64": real_b64},
    "board": board,
    "net": {"format": "float16 little-endian, base64; dense layers in the order of the spec, "
                      "row major over output units, then the bias",
            "b64": net_b64, "count": len(net_v), "spec": net_spec,
            "hidden": HID, "steps": STEPS,
            "abar": arr(ABAR[PAGE_TS], 8), "abar_all": arr(ABAR, 8),
            "page_ts": PAGE_TS,
            "ring_b64": prof_b64, "ringd_b64": deriv_b64,
            "rho_max": RHO_MAX, "rho_n": RHO_N,
            "probe": probe},
}
write_json(OUT / "ddpm.json", payload)
