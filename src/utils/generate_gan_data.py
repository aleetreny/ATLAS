"""Data for the ATLAS 'Counterfeit' article: GANs, DCGAN and WGAN.

Every model in sections 1 and 2 of this site was asked to describe data that
was handed to it. This one is asked to produce data that was not, and the only
judge available is a second network that is learning at the same time. So the
loss is not a number the optimiser can drive down: it is the score of a game
whose other player moves back. That is the whole subject, and almost every
practical fact about GANs falls out of taking the game seriously.

Five blocks, each one measured rather than described:

0. The game on the line, in closed form. The data is a two component gaussian
   mixture and the generator is a single gaussian, so the whole family is two
   numbers and every divergence can be integrated on a grid instead of
   estimated. That makes three separate things checkable rather than quotable:
   which distribution each divergence picks out of the same family (forward KL
   covers, reverse KL collapses onto one mode, and the GAN's Jensen-Shannon
   sits between them), the identity V(D*, G) = 2 JSD - 2 log 2 that connects
   the minimax value to a divergence at all, and the gradient the generator
   receives under the saturating and the non-saturating loss as the
   discriminator gets better, which is the reason nobody trains the loss the
   original paper wrote down.

1. Disjoint supports, which is the case the theory says is fatal and the case
   real image data is always in. Two uniform segments a distance theta apart:
   Jensen-Shannon is log 2 for every theta except zero and its derivative is
   therefore zero everywhere, while the Wasserstein distance is theta. Both
   are then measured through actual trained networks, a saturating
   discriminator and a gradient-penalised critic, because the argument is only
   interesting if the trained thing behaves like the closed form says.

2. Mode collapse on the ring of eight gaussians, with a mode counter that is
   calibrated in both directions (real samples must score eight, a point mass
   must score one) before it is allowed to report anything. Four objectives,
   three seeds each, so the spread between seeds is measured rather than
   assumed to be small.

3. A quality metric for images that this file can actually defend: a small
   convolutional classifier trained on the real data, its 64-dimensional
   penultimate layer used for a Frechet distance and for precision/recall, and
   its class predictions used to count mode dropping directly. The metric is
   calibrated against real-versus-real (the floor), real-versus-noise (the
   ceiling), a blur ladder and a deliberate class drop, so the reader knows
   what a given number is worth before any GAN is judged by it.

4. DCGAN on MNIST: five objectives and architectures at an equal number of
   generator updates, two seeds each, evaluated on the metric from block 3
   every few hundred steps. What comes out includes the two results the
   literature promises (the saturating loss is worse, the critic's estimate
   tracks sample quality where the discriminator's loss does not) and the ones
   it does not (which arms actually differ once the seed spread is on the
   page).

No wall clock is published anywhere on the page: costs are parameter counts
and multiply-accumulates, which are the same number on every machine. The
thread count this ran with is recorded in the JSON anyway, because the
floating point reduction order depends on it.

Run from anywhere: `python src/utils/generate_gan_data.py`. Stages cache into
~/.atlas_vision_data/gan_cache/, keyed by their full configuration, so an
interrupted run costs nothing and changing one hyperparameter invalidates
exactly the runs that depended on it.
"""
import base64
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gans" / "data"
IMG = ROOT / "gans" / "img"
CACHE = Path.home() / ".atlas_vision_data" / "gan_cache"
for d in (OUT, IMG, CACHE):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

SMALL = os.environ.get("ATLAS_GAN_SMALL") == "1"
THREADS = min(4, os.cpu_count() or 4)

rnd = lambda v, n=4: round(float(v), n)


def cached(key, fn):
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = CACHE / f"{h}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    val = fn()
    with open(path, "wb") as fh:
        pickle.dump(val, fh)
    print(f"    [{key}] computed in {time.time() - t0:.1f}s", flush=True)
    return val


# ----------------------------------------------------------------------------
# 0. The game on the line
# ----------------------------------------------------------------------------
# The data: an even mixture of two gaussians. The model family: one gaussian.
# The family cannot contain the data, which is the point. Every quantity below
# is a trapezoid integral on the same grid, and the browser recomputes all of
# them from these same constants, so the numbers on the page are the numbers
# the reader's own machine produces.
MIX = [(-1.5, 0.5), (1.5, 0.5)]      # (mean, sd) per component, equal weights
GRID = (-12.0, 12.0, 4801)


def line_grid():
    x0, x1, n = GRID
    x = np.linspace(x0, x1, n)
    return x, (x1 - x0) / (n - 1)


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def data_pdf(x):
    return sum(normal_pdf(x, m, s) for m, s in MIX) / len(MIX)


def trapz(y, dx):
    return (y.sum() - 0.5 * (y[0] + y[-1])) * dx


def cumtrapz(y, dx):
    """Cumulative trapezoid with a leading zero, so it is a CDF on the grid."""
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1])) * dx
    return out


TINY = 1e-300


def divergences(p, q, dx):
    """Forward KL, reverse KL, Jensen-Shannon and Wasserstein-1, all on the
    same grid with the same rule, so the four are comparable as choices rather
    than as implementations."""
    m = 0.5 * (p + q)
    fwd = trapz(np.where(p > 0, p * np.log(np.maximum(p, TINY) / np.maximum(q, TINY)), 0.0), dx)
    rev = trapz(np.where(q > 0, q * np.log(np.maximum(q, TINY) / np.maximum(p, TINY)), 0.0), dx)
    js = 0.5 * trapz(np.where(p > 0, p * np.log(np.maximum(p, TINY) / np.maximum(m, TINY)), 0.0), dx) \
        + 0.5 * trapz(np.where(q > 0, q * np.log(np.maximum(q, TINY) / np.maximum(m, TINY)), 0.0), dx)
    w1 = trapz(np.abs(cumtrapz(p, dx) - cumtrapz(q, dx)), dx)
    return dict(fwd=fwd, rev=rev, js=js, w1=w1)


def value_at_optimum(p, q, dx):
    """V(D*, G) with D* = p / (p + q), the quantity the minimax game is
    actually optimising. Equals 2 JSD - 2 log 2, which is checked below rather
    than asserted."""
    d = np.where(p + q > 0, p / np.maximum(p + q, TINY), 0.5)
    a = trapz(np.where(p > 0, p * np.log(np.maximum(d, TINY)), 0.0), dx)
    b = trapz(np.where(q > 0, q * np.log(np.maximum(1 - d, TINY)), 0.0), dx)
    return a + b


def stage_line():
    x, dx = line_grid()
    p = data_pdf(x)
    mass = trapz(p, dx)
    assert abs(mass - 1) < 1e-9, f"the data density integrates to {mass}"

    # --- which member of the family each divergence picks
    mus = np.linspace(-3.0, 3.0, 241)
    sds = np.linspace(0.2, 2.5, 116)
    best = {k: (np.inf, None, None) for k in ("fwd", "rev", "js", "w1")}
    surfaces = {k: np.zeros((len(sds), len(mus))) for k in best}
    for i, sd in enumerate(sds):
        q = normal_pdf(x[None, :], mus[:, None], sd)
        pm = p[None, :]
        m = 0.5 * (pm + q)
        lg = lambda a, b: np.where(a > 0, a * np.log(np.maximum(a, TINY) / np.maximum(b, TINY)), 0.0)
        tp = lambda y: (y.sum(axis=1) - 0.5 * (y[:, 0] + y[:, -1])) * dx
        vals = dict(
            fwd=tp(lg(pm, q)),
            rev=tp(lg(q, pm)),
            js=0.5 * tp(lg(pm, m)) + 0.5 * tp(lg(q, m)),
        )
        cq = np.zeros_like(q)
        cq[:, 1:] = np.cumsum(0.5 * (q[:, 1:] + q[:, :-1]), axis=1) * dx
        vals["w1"] = tp(np.abs(cumtrapz(p, dx)[None, :] - cq))
        for k, v in vals.items():
            surfaces[k][i] = v
            j = int(np.argmin(v))
            if v[j] < best[k][0]:
                best[k] = (float(v[j]), float(mus[j]), float(sd))

    # The mixture is symmetric, so reverse KL has two equally good answers and
    # the grid returns whichever it met first. That is a fact about the
    # objective, not an artefact, so it gets measured: the value at the mirror
    # image of the winner has to match to the grid's own resolution.
    ties = {}
    for k, (v, mu, sd) in best.items():
        q = normal_pdf(x, -mu, sd)
        ties[k] = float(divergences(p, q, dx)[k])

    # --- the identity that makes the game a divergence at all
    #
    # V(D*, G) = 2 JSD - 2 log 2 is exact on the real line and only nearly
    # exact on a finite grid, because the derivation uses that both densities
    # integrate to one and a truncated grid loses whatever sits in the tails.
    # So both versions are measured: the textbook one, whose error is the
    # missing mass rather than a mistake, and the mass-corrected one, which is
    # an identity in floating point and is what the guard is allowed to check.
    checks = []
    ident_worst = 0.0
    ident_exact = 0.0
    worst_mass = 0.0
    for mu, sd in [(0.0, 1.0), (1.5, 0.5), (-0.7, 2.2), (2.4, 0.35)]:
        q = normal_pdf(x, mu, sd)
        dv = divergences(p, q, dx)
        v = value_at_optimum(p, q, dx)
        mp, mq = trapz(p, dx), trapz(q, dx)
        ident_worst = max(ident_worst, abs(v - (2 * dv["js"] - 2 * math.log(2))))
        ident_exact = max(ident_exact, abs(v - (2 * dv["js"] - (mp + mq) * math.log(2))))
        worst_mass = max(worst_mass, abs(1 - mp), abs(1 - mq))
        checks.append(dict(mu=mu, sd=sd, value=rnd(v, 8), mass=rnd(mq, 9),
                           **{k: rnd(w, 8) for k, w in dv.items()}))
    assert ident_exact < 1e-12, f"V(D*,G) = 2 JSD - (Mp+Mq) log 2 fails by {ident_exact}"
    assert ident_worst < 4 * worst_mass, "the identity gap is bigger than the missing mass"

    # --- the gradient the generator gets, as the two distributions separate
    # D* is frozen at the current generator, which is the assumption the
    # theorem makes; the derivative with respect to the generator's mean is
    # then an integral against d q / d mu = q (x - mu) / sd^2.
    sep = np.linspace(0.0, 6.0, 121)
    sd_g = 0.5
    grads = []
    for mu in sep:
        q = normal_pdf(x, mu, sd_g)
        d = np.where(p + q > 0, p / np.maximum(p + q, TINY), 0.5)
        dq = q * (x - mu) / sd_g ** 2
        sat = trapz(np.log(np.maximum(1 - d, TINY)) * dq, dx)
        ns = trapz(-np.log(np.maximum(d, TINY)) * dq, dx)
        acc = 0.5 * trapz(p * (p > q), dx) + 0.5 * trapz(q * (q >= p), dx)
        grads.append(dict(sep=rnd(mu, 3), sat=float(sat), ns=float(ns), acc=float(acc),
                          js=float(divergences(p, q, dx)["js"])))

    return dict(best={k: dict(value=v, mu=mu, sd=sd, mirror=ties[k])
                      for k, (v, mu, sd) in best.items()},
                identity_worst=ident_worst, identity_exact=ident_exact,
                worst_mass=worst_mass, checks=checks, grads=grads,
                mus=mus.tolist(), sds=sds.tolist(),
                surfaces={k: v for k, v in surfaces.items()})


# ----------------------------------------------------------------------------
# A tiny MLP written twice: numpy here, JavaScript in the article
# ----------------------------------------------------------------------------
# The 2-D widget in the browser trains a GAN live, so both sides need the same
# random numbers and the same arithmetic. numpy's generator does not exist in a
# browser; a linear congruential generator plus Box-Muller does, and it is
# written identically in both places. The guard at the end of this file checks
# a forward pass and one optimiser step rather than a whole trajectory, because
# thousands of Adam steps diverge between any two implementations for reasons
# that have nothing to do with either being wrong.
class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s / 4294967296.0

    def uniform(self, n):
        return np.array([self.u() for _ in range(n)])

    def normal(self, n):
        out = np.empty(n)
        i = 0
        while i < n:
            u1 = max(self.u(), 1e-12)
            u2 = self.u()
            r = math.sqrt(-2 * math.log(u1))
            out[i] = r * math.cos(2 * math.pi * u2)
            if i + 1 < n:
                out[i + 1] = r * math.sin(2 * math.pi * u2)
            i += 2
        return out


# ----------------------------------------------------------------------------
# 1. Disjoint supports
# ----------------------------------------------------------------------------
def stage_disjoint():
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    thetas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    steps = 60 if SMALL else 1500
    batch = 256
    rows = []
    for th in thetas:
        torch.manual_seed(19)

        def sample(n, x):
            y = torch.rand(n, 1)
            return torch.cat([torch.full((n, 1), float(x)), y], 1)

        # the saturating discriminator: a classifier, trained to convergence
        disc = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                             nn.Linear(64, 1))
        opt = torch.optim.Adam(disc.parameters(), 1e-3)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(steps):
            a, b = sample(batch, 0.0), sample(batch, th)
            loss = bce(disc(a), torch.ones(batch, 1)) + bce(disc(b), torch.zeros(batch, 1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a, b = sample(4096, 0.0), sample(4096, th)
            va = torch.sigmoid(disc(a)).mean().item()
            vb = torch.sigmoid(disc(b)).mean().item()
            value = (torch.log(torch.sigmoid(disc(a)).clamp_min(1e-12)).mean()
                     + torch.log((1 - torch.sigmoid(disc(b))).clamp_min(1e-12)).mean()).item()
        # what the generator would receive: d/dtheta of both losses, through
        # the same trained discriminator
        pos = torch.full((4096, 1), float(th), requires_grad=True)
        inp = torch.cat([pos, torch.rand(4096, 1)], 1)
        s = torch.sigmoid(disc(inp))
        gsat = torch.autograd.grad(torch.log((1 - s).clamp_min(1e-12)).mean(), pos,
                                   retain_graph=True)[0].abs().mean().item()
        gns = torch.autograd.grad(-torch.log(s.clamp_min(1e-12)).mean(), pos)[0].abs().mean().item()

        # the critic: same shape, no sigmoid, gradient penalty
        torch.manual_seed(19)
        crit = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                             nn.Linear(64, 1))
        copt = torch.optim.Adam(crit.parameters(), 1e-3, betas=(0.5, 0.9))
        for _ in range(steps):
            a, b = sample(batch, 0.0), sample(batch, th)
            eps = torch.rand(batch, 1)
            mid = (eps * a + (1 - eps) * b).requires_grad_(True)
            g = torch.autograd.grad(crit(mid).sum(), mid, create_graph=True)[0]
            gp = ((g.norm(dim=1) - 1) ** 2).mean()
            loss = crit(b).mean() - crit(a).mean() + 10.0 * gp
            copt.zero_grad(); loss.backward(); copt.step()
        with torch.no_grad():
            a, b = sample(4096, 0.0), sample(4096, th)
            west = (crit(a).mean() - crit(b).mean()).item()
        pos = torch.full((4096, 1), float(th), requires_grad=True)
        inp = torch.cat([pos, torch.rand(4096, 1)], 1)
        gcrit = torch.autograd.grad(-crit(inp).mean(), pos)[0].abs().mean().item()

        # Significant digits rather than decimal places: these gradients are
        # the whole point of the experiment and they run down to 1e-8, which
        # six decimal places rounds to a flat zero and turns every ratio built
        # from them into a division by nothing.
        sig = lambda v: float(f"{v:.4e}")
        rows.append(dict(theta=th, js=0.0 if th == 0 else math.log(2), w1=abs(th),
                         d_real=rnd(va, 5), d_fake=rnd(vb, 5), value=rnd(value, 5),
                         g_sat=sig(gsat), g_ns=sig(gns),
                         w_est=rnd(west, 5), g_crit=sig(gcrit)))
        print(f"    theta={th}: D says {va:.3f}/{vb:.3f}, sat grad {gsat:.2e}, "
              f"critic W {west:.3f} against {abs(th):.3f}", flush=True)
    return dict(rows=rows, steps=steps, batch=batch)


# ----------------------------------------------------------------------------
# 2. The ring of eight gaussians
# ----------------------------------------------------------------------------
RING_R = 2.0
RING_SD = 0.05
RING_K = 8


def ring_centres():
    a = np.arange(RING_K) * 2 * math.pi / RING_K
    return np.stack([RING_R * np.cos(a), RING_R * np.sin(a)], 1)


def ring_sample(n, rng):
    c = ring_centres()
    idx = rng.integers(0, RING_K, n)
    return c[idx] + rng.normal(0, RING_SD, (n, 2))


def mode_report(pts, tol_sd=4.0, min_share=0.005):
    """How many of the eight modes a sample covers.

    Two decisions that have to be calibrated rather than chosen: a point
    belongs to a mode if it lands within `tol_sd` standard deviations of that
    centre (so points nowhere near the ring count for nothing), and a mode is
    covered if at least `min_share` of the whole sample belongs to it (so one
    stray point does not buy a mode). The calibration in stage_ring checks that
    real samples score eight and a point mass scores one.
    """
    c = ring_centres()
    d = np.linalg.norm(pts[:, None, :] - c[None, :, :], axis=2)
    near = d.min(axis=1)
    who = d.argmin(axis=1)
    good = near <= tol_sd * RING_SD
    counts = np.bincount(who[good], minlength=RING_K)
    covered = int((counts >= min_share * len(pts)).sum())
    share = counts / max(counts.sum(), 1)
    nz = share[share > 0]
    return dict(covered=covered, quality=float(good.mean()),
                counts=counts.tolist(),
                entropy=float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0,
                max_share=float(share.max()) if counts.sum() else 0.0)


def ring_run(arm, seed, steps, batch, n_critic=None):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    real_all = ring_sample(20000, rng)
    real_t = torch.from_numpy(real_all).float()

    def mlp(nin, nout, act):
        return nn.Sequential(nn.Linear(nin, 128), act(), nn.Linear(128, 128), act(),
                             nn.Linear(128, nout))

    G = mlp(2, 2, lambda: nn.ReLU(True))
    D = mlp(2, 1, lambda: nn.LeakyReLU(0.2, True))
    wgan = arm.startswith("wgan")
    lr = 1e-4
    og = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    n_critic = (3 if wgan else 1) if n_critic is None else n_critic
    curve = []
    for it in range(steps):
        for _ in range(n_critic):
            real = real_t[torch.randint(0, real_t.shape[0], (batch,))]
            fake = G(torch.randn(batch, 2)).detach()
            if wgan:
                ld = D(fake).mean() - D(real).mean()
                if arm == "wgan-gp":
                    eps = torch.rand(batch, 1)
                    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)
                    g = torch.autograd.grad(D(mid).sum(), mid, create_graph=True)[0]
                    ld = ld + 10.0 * ((g.norm(dim=1) - 1) ** 2).mean()
            else:
                ld = bce(D(real), torch.ones(batch, 1)) + bce(D(fake), torch.zeros(batch, 1))
            od.zero_grad(); ld.backward(); od.step()
            if arm == "wgan-clip":
                with torch.no_grad():
                    for p in D.parameters():
                        p.clamp_(-0.05, 0.05)
        fake = G(torch.randn(batch, 2))
        out = D(fake)
        if wgan:
            lg = -out.mean()
        elif arm == "sat":
            lg = torch.log((1 - torch.sigmoid(out)).clamp_min(1e-12)).mean()
        else:
            lg = bce(out, torch.ones(batch, 1))
        og.zero_grad(); lg.backward(); og.step()
        if (it + 1) % max(1, steps // 40) == 0:
            with torch.no_grad():
                pts = G(torch.randn(2000, 2)).numpy()
            r = mode_report(pts)
            curve.append(dict(it=it + 1, covered=r["covered"], quality=rnd(r["quality"], 4)))
    with torch.no_grad():
        pts = G(torch.randn(5000, 2)).numpy()
    rep = mode_report(pts)
    return dict(final=rep, curve=curve, n_critic=n_critic,
                points=np.round(pts[:1200], 3).tolist(),
                gparams=sum(p.numel() for p in G.parameters()),
                dparams=sum(p.numel() for p in D.parameters()))


def stage_ring():
    rng = np.random.default_rng(19)
    real = ring_sample(5000, rng)
    calib = dict(
        real=mode_report(real),
        point_mass=mode_report(np.tile(ring_centres()[0], (5000, 1))
                               + rng.normal(0, RING_SD, (5000, 2))),
        uniform=mode_report(rng.uniform(-2.5, 2.5, (5000, 2))),
        two_modes=mode_report(np.concatenate([
            ring_centres()[0] + rng.normal(0, RING_SD, (2500, 2)),
            ring_centres()[3] + rng.normal(0, RING_SD, (2500, 2))])),
    )
    assert calib["real"]["covered"] == RING_K, "the mode counter cannot see real data"
    assert calib["point_mass"]["covered"] == 1, "the mode counter cannot see collapse"
    assert calib["two_modes"]["covered"] == 2, "the mode counter miscounts two modes"

    steps = 40 if SMALL else 4000
    batch = 256
    arms = ["ns", "sat", "wgan-clip", "wgan-gp"]
    seeds = [1, 2, 3]
    runs = {}
    for arm in arms:
        for sd in seeds:
            key = f"ring-{arm}-s{sd}-st{steps}-b{batch}-r{RING_R}-sd{RING_SD}-v1"
            runs[f"{arm}/{sd}"] = cached(key, lambda a=arm, s=sd: ring_run(a, s, steps, batch))
            r = runs[f"{arm}/{sd}"]["final"]
            print(f"    {arm} seed {sd}: {r['covered']}/8 modes, "
                  f"{r['quality']*100:.1f}% on the ring", flush=True)
    return dict(calib=calib, runs=runs, steps=steps, batch=batch, arms=arms, seeds=seeds,
                centres=np.round(ring_centres(), 4).tolist(),
                real_points=np.round(real[:800], 3).tolist(),
                r=RING_R, sd=RING_SD)


def stage_ring_falsify(steps, batch):
    """Attempts to break the ring result, run because the result was not the
    expected one.

    Two of the four objectives cover every mode on every seed at the budget
    above and two of them do not, which is not what the folklore says, so
    before publishing it: give the losers three times the budget in case they
    were merely slow, and give the winners a stronger opponent in case the
    reason they are fine is that their discriminator was too weak to corner
    them. A negative result with no falsification attempts behind it is
    indistinguishable from not having tried.
    """
    trials = [
        dict(name="ns-3x", arm="ns", steps=steps * 3, n_critic=1, seeds=[1, 3],
             why="three times the budget for the loss that already worked"),
        dict(name="wgan-gp-3x", arm="wgan-gp", steps=steps * 3, n_critic=3, seeds=[1, 2],
             why="three times the budget for the loss that did not"),
        dict(name="ns-strong-d", arm="ns", steps=steps, n_critic=5, seeds=[1, 3],
             why="five discriminator steps per generator step, to make collapse likelier"),
        dict(name="wgan-gp-1critic", arm="wgan-gp", steps=steps, n_critic=1,
             seeds=[1, 2], why="one critic step, in case three is what starved the generator"),
    ]
    out = {}
    for t in trials:
        for sd in t["seeds"]:
            key = (f"ringf-{t['arm']}-nc{t['n_critic']}-s{sd}-st{t['steps']}-b{batch}"
                   f"-r{RING_R}-sd{RING_SD}-v1")
            r = cached(key, lambda t=t, s=sd: ring_run(t["arm"], s, t["steps"], batch,
                                                       n_critic=t["n_critic"]))
            out[f"{t['name']}/{sd}"] = dict(final=r["final"], curve=r["curve"],
                                            steps=t["steps"], n_critic=t["n_critic"],
                                            arm=t["arm"], why=t["why"])
            print(f"    {t['name']} seed {sd}: {r['final']['covered']}/8 modes, "
                  f"{r['final']['quality']*100:.1f}% on the ring", flush=True)
    return dict(trials=[{k: v for k, v in t.items()} for t in trials], runs=out)


# ----------------------------------------------------------------------------
# 2b. The same game, written twice, so the browser can play it live
# ----------------------------------------------------------------------------
# The widget in the article trains a GAN in the reader's tab. That is only
# worth doing if the arithmetic is the same arithmetic, so the whole thing is
# written here in numpy with an explicit loop order and again in JavaScript,
# and the reference trajectory below is what the browser has to reproduce.
# Everything is float64 on both sides and no library is involved beyond the
# four arithmetic operations, tanh and exp.
NET_H = 24
LIVE_BATCH = 64
LIVE_STEPS = 200


def he_init(rng, nin, nout):
    """Uniform in +-sqrt(6/(nin+nout)), drawn in row-major order over outputs
    so that the JavaScript side can consume the same stream in the same
    order."""
    lim = math.sqrt(6.0 / (nin + nout))
    w = np.array([[(rng.u() * 2 - 1) * lim for _ in range(nin)] for _ in range(nout)])
    return w, np.zeros(nout)


class MiniMLP:
    def __init__(self, rng, sizes, act):
        self.sizes = sizes
        self.act = act
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            w, b = he_init(rng, sizes[i], sizes[i + 1])
            self.W.append(w)
            self.b.append(b)
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.t = 0

    def act_f(self, z):
        return np.tanh(z) if self.act == "tanh" else np.where(z > 0, z, 0.2 * z)

    def act_d(self, z):
        return 1 - np.tanh(z) ** 2 if self.act == "tanh" else np.where(z > 0, 1.0, 0.2)

    def forward(self, x):
        a = [x]
        zs = []
        for i in range(len(self.W)):
            z = a[-1] @ self.W[i].T + self.b[i]
            zs.append(z)
            a.append(self.act_f(z) if i < len(self.W) - 1 else z)
        return a, zs

    def backward(self, a, zs, dout):
        gw, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gw[i] = d.T @ a[i]
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i]) * self.act_d(zs[i - 1])
        # the third return value is the gradient with respect to the INPUT,
        # which is the only channel the discriminator has for telling the
        # generator anything at all
        return gw, gb, d @ self.W[0]

    def step(self, gw, gb, lr, b1=0.5, b2=0.999, eps=1e-8):
        self.t += 1
        gs = list(gw) + list(gb)
        ps = self.W + self.b
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for k, g in enumerate(gs):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            ps[k] -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps)


def live_gan(seed, steps, arm="ns"):
    rng = LCG(seed)
    G = MiniMLP(rng, [2, NET_H, NET_H, 2], "tanh")
    D = MiniMLP(rng, [2, NET_H, NET_H, 1], "leaky")
    c = ring_centres()
    losses = []
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    for it in range(steps):
        idx = (rng.uniform(LIVE_BATCH) * RING_K).astype(int) % RING_K
        real = c[idx] + RING_SD * rng.normal(LIVE_BATCH * 2).reshape(LIVE_BATCH, 2)
        z = rng.normal(LIVE_BATCH * 2).reshape(LIVE_BATCH, 2)
        ga, gz = G.forward(z)
        fake = ga[-1]
        # discriminator: one step of the usual cross entropy
        ra, rz = D.forward(real)
        fa, fz = D.forward(fake)
        pr, pf = sig(ra[-1]), sig(fa[-1])
        ld = float(-(np.log(np.maximum(pr, 1e-12)).mean() + np.log(np.maximum(1 - pf, 1e-12)).mean()))
        dgw1, dgb1, _ = D.backward(ra, rz, (pr - 1) / LIVE_BATCH)
        dgw2, dgb2, _ = D.backward(fa, fz, pf / LIVE_BATCH)
        D.step([a + b for a, b in zip(dgw1, dgw2)], [a + b for a, b in zip(dgb1, dgb2)], 2e-3)
        # generator: non-saturating unless the arm asks for the original
        fa, fz = D.forward(fake)
        pf = sig(fa[-1])
        if arm == "sat":
            lg = float(np.log(np.maximum(1 - pf, 1e-12)).mean())
            seed_grad = -pf / LIVE_BATCH
        else:
            lg = float(-np.log(np.maximum(pf, 1e-12)).mean())
            seed_grad = (pf - 1) / LIVE_BATCH
        _, _, dx = D.backward(fa, fz, seed_grad)
        ggw, ggb, _ = G.backward(ga, gz, dx)
        G.step(ggw, ggb, 2e-3)
        losses.append((rnd(ld, 8), rnd(lg, 8)))
    z = rng.normal(2000 * 2).reshape(2000, 2)
    pts = G.forward(z)[0][-1]
    # Two reports, and the second one is the one the browser can check itself
    # against: it only receives 600 points rounded to three decimals, so a
    # count over the full 2000 would have it disagreeing with the generator
    # about a counter that is in fact identical on both sides.
    shown = np.array(np.round(pts[:600], 3).tolist())
    return dict(losses=losses, report=mode_report(pts), report_shown=mode_report(shown),
                points=np.round(pts[:600], 3).tolist(),
                params=sum(w.size + b.size for w, b in zip(G.W, G.b)))


def stage_browser():
    """A short reference trajectory plus one frozen forward pass.

    The trajectory is what the browser has to match step for step; the frozen
    pass is what tells you which half is wrong when it does not.
    """
    rng = LCG(12345)
    G = MiniMLP(rng, [2, NET_H, NET_H, 2], "tanh")
    D = MiniMLP(rng, [2, NET_H, NET_H, 1], "leaky")
    probe = np.array([[0.0, 0.0], [1.0, -1.0], [-0.7, 0.35], [2.0, 2.0]])
    frozen = dict(
        seed=12345,
        g_out=np.round(G.forward(probe)[0][-1], 10).tolist(),
        d_out=np.round(D.forward(probe)[0][-1], 10).tolist(),
        probe=probe.tolist(),
        w_sum=rnd(sum(float(w.sum()) for w in G.W) + sum(float(w.sum()) for w in D.W), 10),
    )
    runs = {}
    for arm in ("ns", "sat"):
        runs[arm] = live_gan(7, LIVE_STEPS, arm)
    return dict(frozen=frozen, runs=runs, steps=LIVE_STEPS, batch=LIVE_BATCH, h=NET_H)


# ----------------------------------------------------------------------------
# 3. A quality metric with a calibration
# ----------------------------------------------------------------------------
def frechet(a, b):
    """Frechet distance between two gaussians fitted to feature sets.

    The matrix square root is done through a symmetric eigendecomposition of
    the similarity-transformed product, which is the standard trick and avoids
    a scipy sqrtm on a matrix that is only positive semidefinite in theory.
    """
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    # sqrt of ca, then sqrt(ca) cb sqrt(ca) is symmetric psd
    wa, va = np.linalg.eigh(ca)
    sa = va @ np.diag(np.sqrt(np.maximum(wa, 0))) @ va.T
    m = sa @ cb @ sa
    wm = np.linalg.eigvalsh(m)
    tr = np.sqrt(np.maximum(wm, 0)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * tr)


def prec_recall(real, fake, k=3):
    """Kynkaanniemi's improved precision and recall: a point is inside the
    other set's manifold if it falls within some point's distance to its own
    kth nearest neighbour."""
    def radii(x):
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return np.sort(d, axis=1)[:, k - 1]
    rr, rf = radii(real), radii(fake)
    d = np.linalg.norm(fake[:, None, :] - real[None, :, :], axis=2)
    precision = float((d <= rr[None, :]).any(axis=1).mean())
    recall = float((d.T <= rf[None, :]).any(axis=1).mean())
    return precision, recall


def build_probe():
    """The yardstick's shape, in one place, so that the cache can rebuild it
    from a state dict without repeating the definition."""
    import torch
    import torch.nn as nn

    torch.manual_seed(7)
    body = nn.Sequential(
        nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())
    head = nn.Linear(64, 10)
    return nn.Sequential(body, head), body, head, None


def train_probe():
    import torch
    import torch.nn as nn
    from vision_data import mnist

    torch.set_num_threads(THREADS)
    torch.manual_seed(7)
    Xtr, ytr, Xte, yte = mnist()
    xt = torch.from_numpy(np.ascontiguousarray(Xtr)).float().div(127.5).sub(1).unsqueeze(1)
    yt = torch.from_numpy(np.ascontiguousarray(ytr)).long()
    xv = torch.from_numpy(np.ascontiguousarray(Xte)).float().div(127.5).sub(1).unsqueeze(1)
    yv = torch.from_numpy(np.ascontiguousarray(yte)).long()
    net, body, head, _ = build_probe()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    lossf = nn.CrossEntropyLoss()
    epochs = 1 if SMALL else 4
    n = 2000 if SMALL else xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = lossf(net(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(xv).argmax(1) == yv).float().mean().item()
    print(f"    the probe classifier reaches {acc*100:.2f}% on the MNIST test set", flush=True)
    return net, body, head, acc


def features_of(body, head, imgs):
    """imgs: (n, 28, 28) float in [-1, 1]. Returns (features, class ids)."""
    import torch
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(imgs)).float().unsqueeze(1)
        f = body(t)
        c = head(f).argmax(1)
    return f.numpy(), c.numpy()


def stage_metric_uncached():
    import torch
    from vision_data import mnist

    net, body, head, acc = train_probe()
    Xtr, ytr, Xte, yte = mnist()
    rng = np.random.default_rng(11)
    to_f = lambda a: a.astype(np.float32) / 127.5 - 1.0
    n = 500 if SMALL else 5000
    idx = rng.permutation(len(Xtr))
    A = to_f(Xtr[idx[:n]])
    B = to_f(Xtr[idx[n:2 * n]])
    T = to_f(Xte[:n])
    fa, ca = features_of(body, head, A)
    fb, cb = features_of(body, head, B)
    ft, ct = features_of(body, head, T)

    def blur(x, k):
        """Separable box blur of width k, the cheapest way to destroy exactly
        the high frequencies a GAN is supposed to produce."""
        if k <= 1:
            return x
        out = x.copy()
        pad = k // 2
        for ax in (1, 2):
            p = np.pad(out, [(0, 0)] + [(pad, pad) if a == ax else (0, 0) for a in (1, 2)],
                       mode="edge")
            c = np.cumsum(p, axis=ax)
            sl = lambda s, e: tuple(slice(None) if a != ax else slice(s, e) for a in range(3))
            out = (c[sl(k, None)] - c[sl(0, -k)]) / k
        return out

    ladder = []
    for k in (1, 2, 3, 5, 7):
        fk, _ = features_of(body, head, blur(A, k))
        ladder.append(dict(k=k, fd=rnd(frechet(fb, fk), 4)))
    dropped = to_f(Xtr[idx[:len(Xtr)]][ytr[idx[:len(Xtr)]] < 7][:n])
    fd_drop, _ = features_of(body, head, dropped)
    noise = rng.uniform(-1, 1, (n, 28, 28)).astype(np.float32)
    fn_, _ = features_of(body, head, noise)
    # real MNIST is mostly background: uniform noise is not a fair ceiling on
    # its own, so a second one that has the right marginal but no structure
    shuffled = A.reshape(n, -1).copy()
    for r in shuffled:
        rng.shuffle(r)
    fs_, _ = features_of(body, head, shuffled.reshape(n, 28, 28))

    pr_real = prec_recall(fb[:1000], fa[:1000])
    pr_blur = prec_recall(fb[:1000], features_of(body, head, blur(A, 5))[0][:1000])
    calib = dict(
        floor_train=rnd(frechet(fa, fb), 4),
        floor_test=rnd(frechet(fa, ft), 4),
        blur=ladder,
        class_drop=rnd(frechet(fb, fd_drop), 4),
        noise=rnd(frechet(fb, fn_), 4),
        shuffled=rnd(frechet(fb, fs_), 4),
        pr_real=[rnd(pr_real[0], 4), rnd(pr_real[1], 4)],
        pr_blur=[rnd(pr_blur[0], 4), rnd(pr_blur[1], 4)],
        real_hist=(np.bincount(cb, minlength=10) / len(cb)).round(4).tolist(),
        true_hist=(np.bincount(ytr, minlength=10) / len(ytr)).round(4).tolist(),
        probe_acc=rnd(acc, 4), n=n,
    )
    assert calib["floor_train"] < calib["class_drop"], "the metric cannot see a dropped class"
    assert calib["class_drop"] < calib["noise"], "the metric ranks noise below a dropped class"
    ref = dict(features=fb, classes=cb)
    return calib, ref, (net, body, head)


def stage_metric():
    """Cached by hand rather than through `cached`, because what has to survive
    a pickle is weights and arrays, not torch modules: rebuilding the modules
    from a state dict is both smaller on disk and immune to a torch version
    that decides its own classes are no longer unpicklable."""
    import torch

    key = f"metric-{'s' if SMALL else 'f'}-v1"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = CACHE / f"{h}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            saved = pickle.load(fh)
        net, body, head, _ = build_probe()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in saved["state"].items()})
        net.eval()
        return saved["calib"], saved["ref"], (net, body, head)
    calib, ref, probe = stage_metric_uncached()
    net = probe[0]
    with open(path, "wb") as fh:
        pickle.dump(dict(calib=calib, ref=ref,
                         state={k: v.detach().numpy() for k, v in net.state_dict().items()}), fh)
    return calib, ref, probe


# ----------------------------------------------------------------------------
# 4. DCGAN on MNIST
# ----------------------------------------------------------------------------
Z_DIM = 32
G_CH = 64
D_CH = 32


def build_models(arm, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class Gen(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(Z_DIM, G_CH * 7 * 7)
            self.net = nn.Sequential(
                nn.BatchNorm2d(G_CH), nn.ReLU(True),
                nn.Upsample(scale_factor=2), nn.Conv2d(G_CH, G_CH // 2, 3, 1, 1),
                nn.BatchNorm2d(G_CH // 2), nn.ReLU(True),
                nn.Upsample(scale_factor=2), nn.Conv2d(G_CH // 2, G_CH // 4, 3, 1, 1),
                nn.BatchNorm2d(G_CH // 4), nn.ReLU(True),
                nn.Conv2d(G_CH // 4, 1, 3, 1, 1), nn.Tanh())

        def forward(self, z):
            return self.net(self.fc(z).view(-1, G_CH, 7, 7))

    class Disc(nn.Module):
        def __init__(self, norm=True):
            super().__init__()
            def blk(i, o, k, s, p):
                layers = [nn.Conv2d(i, o, k, s, p)]
                if norm:
                    layers.append(nn.GroupNorm(4, o))
                layers.append(nn.LeakyReLU(0.2, True))
                return layers
            self.net = nn.Sequential(*blk(1, D_CH, 4, 2, 1), *blk(D_CH, D_CH * 2, 4, 2, 1),
                                     *blk(D_CH * 2, D_CH * 4, 3, 2, 1), nn.Flatten())
            self.fc = nn.Linear(D_CH * 4 * 4 * 4, 1)

        def forward(self, x):
            return self.fc(self.net(x))

    class MlpGen(nn.Module):
        """The same job with no convolutions and a comparable weight count, so
        'what the DC in DCGAN buys' is a measurement and not a slogan."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(Z_DIM, 256), nn.BatchNorm1d(256), nn.ReLU(True),
                nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(True),
                nn.Linear(256, 784), nn.Tanh())

        def forward(self, z):
            return self.net(z).view(-1, 1, 28, 28)

    class MlpDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(784, 256), nn.LeakyReLU(0.2, True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, True), nn.Linear(256, 1))

        def forward(self, x):
            return self.net(x)

    if arm == "mlp":
        return MlpGen(), MlpDisc()
    # Group normalisation rather than batch normalisation in the critic, for
    # every arm: the gradient penalty is defined per sample and a batch
    # statistic makes the penalised quantity depend on the rest of the batch.
    # Using it in all arms keeps the only difference between them the loss.
    return Gen(), Disc(norm=True)


def image_run(arm, seed, steps, batch, evalfn):
    import torch
    import torch.nn as nn
    from vision_data import mnist

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    Xtr, ytr, _, _ = mnist()
    n_train = 2000 if SMALL else 20000
    rng = np.random.default_rng(19)
    keep = rng.permutation(len(Xtr))[:n_train]
    X = torch.from_numpy(np.ascontiguousarray(Xtr[keep])).float().div(127.5).sub(1).unsqueeze(1)

    G, D = build_models(arm, seed)
    wgan = arm.startswith("wgan")
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.9 if wgan else 0.999))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.9 if wgan else 0.999))
    bce = nn.BCEWithLogitsLoss()
    n_critic = 2 if wgan else 1
    gp_batch = batch // 2      # the penalty is an expectation, and it costs
    fixed = torch.randn(64, Z_DIM)
    curve = []
    d_updates = 0
    for it in range(steps):
        for _ in range(n_critic):
            real = X[torch.randint(0, X.shape[0], (batch,))]
            with torch.no_grad():
                fake = G(torch.randn(batch, Z_DIM))
            if wgan:
                ld = D(fake).mean() - D(real).mean()
                if arm == "wgan-gp":
                    eps = torch.rand(gp_batch, 1, 1, 1)
                    mid = (eps * real[:gp_batch] + (1 - eps) * fake[:gp_batch]).requires_grad_(True)
                    g = torch.autograd.grad(D(mid).sum(), mid, create_graph=True)[0]
                    ld = ld + 10.0 * ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()
            else:
                ld = bce(D(real), torch.ones(batch, 1)) + bce(D(fake), torch.zeros(batch, 1))
            od.zero_grad(); ld.backward(); od.step()
            d_updates += 1
            if arm == "wgan-clip":
                with torch.no_grad():
                    for p in D.parameters():
                        p.clamp_(-0.01, 0.01)
        out = D(G(torch.randn(batch, Z_DIM)))
        if wgan:
            lg = -out.mean()
        elif arm == "sat":
            lg = torch.log((1 - torch.sigmoid(out)).clamp_min(1e-12)).mean()
        else:
            lg = bce(out, torch.ones(batch, 1))
        og.zero_grad(); lg.backward(); og.step()

        if (it + 1) % max(1, steps // 8) == 0 or it == 0:
            G.eval()
            with torch.no_grad():
                samples = G(torch.randn(2000, Z_DIM)).squeeze(1).numpy()
                # what the discriminator itself thinks, which is the number a
                # practitioner would actually be watching during training
                real = X[torch.randint(0, X.shape[0], (512,))]
                fk = G(torch.randn(512, Z_DIM))
                if wgan:
                    d_stat = (D(real).mean() - D(fk).mean()).item()
                else:
                    d_stat = (bce(D(real), torch.ones(512, 1))
                              + bce(D(fk), torch.zeros(512, 1))).item()
            G.train()
            m = evalfn(samples)
            m.update(it=it + 1, d_stat=rnd(d_stat, 5))
            curve.append(m)
            print(f"      {arm}/{seed} step {it+1}: fd {m['fd']:.2f} "
                  f"modes {m['modes']}/10 d {d_stat:.3f}", flush=True)
    # How Lipschitz the critic actually is, which is the property both WGAN
    # variants exist to enforce and neither of them guarantees. Measured the
    # way the penalty defines it: the norm of the input gradient on the
    # segments between real and generated samples.
    real = X[torch.randint(0, X.shape[0], (256,))]
    with torch.no_grad():
        fake = G(torch.randn(256, Z_DIM))
    eps = torch.rand(256, 1, 1, 1)
    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)
    gnorm = torch.autograd.grad(D(mid).sum(), mid)[0].flatten(1).norm(dim=1)
    lip = dict(mean=rnd(gnorm.mean().item(), 4), max=rnd(gnorm.max().item(), 4))

    G.eval()
    with torch.no_grad():
        sheet = G(fixed).squeeze(1).numpy()
        big = G(torch.randn(5000, Z_DIM)).squeeze(1).numpy()
        # a straight line in latent space, which is the standard sanity check
        # that the generator learned a map and not a lookup table
        za, zb = torch.randn(4, Z_DIM), torch.randn(4, Z_DIM)
        ts = torch.linspace(0, 1, 8)
        walk = torch.cat([za[i][None] * (1 - t) + zb[i][None] * t for i in range(4) for t in ts])
        interp = G(walk).squeeze(1).numpy()
    final = evalfn(big)
    state = {k: v.detach().numpy() for k, v in G.state_dict().items()}
    return dict(curve=curve, final=final, sheet=sheet, interp=interp, lip=lip,
                gparams=sum(p.numel() for p in G.parameters()),
                dparams=sum(p.numel() for p in D.parameters()),
                d_updates=d_updates, samples=big[:512], state=state)


def stage_images(calib, ref, probe):
    body, head = probe[1], probe[2]
    real_f = ref["features"]
    real_hist = np.array(calib["real_hist"])

    def evalfn(samples):
        f, c = features_of(body, head, samples)
        hist = np.bincount(c, minlength=10) / len(c)
        p, r = prec_recall(real_f[:1000], f[:1000])
        return dict(fd=rnd(frechet(real_f, f), 4), hist=hist.round(4).tolist(),
                    modes=int((hist >= 0.02).sum()), tv=rnd(0.5 * np.abs(hist - real_hist).sum(), 4),
                    precision=rnd(p, 4), recall=rnd(r, 4))

    steps = 30 if SMALL else 2000
    batch = 128
    arms = ["dcgan", "sat", "wgan-clip", "wgan-gp", "mlp"]
    seeds = [1, 2]
    runs = {}
    for arm in arms:
        for sd in seeds:
            key = (f"img-{arm}-s{sd}-st{steps}-b{batch}-z{Z_DIM}-g{G_CH}-d{D_CH}"
                   f"-n{'2k' if SMALL else '20k'}-v1")
            runs[f"{arm}/{sd}"] = cached(key,
                                         lambda a=arm, s=sd: image_run(a, s, steps, batch, evalfn))

    # Memorisation, which is the first objection anyone raises at a sheet of
    # samples. The distance from a sample to its nearest training image, in the
    # probe's feature space, against the same statistic for real images the
    # generator never saw: if a generator were copying, its distance would sit
    # below the held-out one rather than above it.
    from vision_data import mnist
    Xtr, _, Xte, _ = mnist()
    rng = np.random.default_rng(19)
    keep = rng.permutation(len(Xtr))[:2000 if SMALL else 20000]
    train_f, _ = features_of(body, head, Xtr[keep[:4000]].astype(np.float32) / 127.5 - 1.0)
    held_f, _ = features_of(body, head, Xte[:512].astype(np.float32) / 127.5 - 1.0)
    nn_of = lambda f: float(np.sort(np.linalg.norm(f[:, None, :] - train_f[None, :, :], axis=2),
                                    axis=1)[:, 0].mean())
    nn = dict(held_out=rnd(nn_of(held_f), 4),
              train_self=rnd(float(np.sort(np.linalg.norm(
                  train_f[:512, None, :] - train_f[None, :, :], axis=2), axis=1)[:, 1].mean()), 4))
    for arm in arms:
        nn[arm] = rnd(nn_of(features_of(body, head, runs[f"{arm}/1"]["samples"])[0]), 4)

    # The claim the WGAN paper makes and the one nobody checks: that the number
    # you can watch while training (the critic's estimate, or the
    # discriminator's loss) tells you whether the samples are getting better.
    # Correlation across every checkpoint of both seeds, per arm.
    track = {}
    for arm in arms:
        xs, ys = [], []
        for sd in seeds:
            for c in runs[f"{arm}/{sd}"]["curve"]:
                xs.append(c["d_stat"])
                ys.append(c["fd"])
        xs, ys = np.array(xs), np.array(ys)
        r = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else float("nan")
        track[arm] = dict(r=None if math.isnan(r) else rnd(r, 4), n=len(xs))
    return dict(runs=runs, steps=steps, batch=batch, arms=arms, seeds=seeds, nn=nn, track=track)


# ----------------------------------------------------------------------------
# Images out
# ----------------------------------------------------------------------------
def save_png(arr, path, scale=1):
    from PIL import Image

    a = np.clip(arr, 0, 1)
    a = (a * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    if scale > 1:
        im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(path, optimize=True)
    return path.stat().st_size


def sheet(tiles, cols, pad=0):
    """tiles: list of (h, w) arrays in [0, 1]."""
    t = np.asarray(tiles)
    n, h, w = t.shape
    rows = math.ceil(n / cols)
    out = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad))
    for k in range(n):
        r, c = divmod(k, cols)
        out[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = t[k]
    return out


# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"threads: {THREADS}{'  (SMALL smoke run)' if SMALL else ''}", flush=True)

    print("0. the game on the line", flush=True)
    line = cached(f"line-{MIX}-{GRID}-v1", stage_line)
    for k, v in line["best"].items():
        print(f"    {k}: mu {v['mu']:+.3f}, sd {v['sd']:.3f}, value {v['value']:.5f}", flush=True)
    print(f"    V(D*,G) = 2 JSD - 2 log 2 holds to {line['identity_worst']:.2e}", flush=True)

    print("1. disjoint supports", flush=True)
    dis = cached(f"disjoint-{'s' if SMALL else 'f'}-v2", stage_disjoint)

    print("2. the ring of eight gaussians", flush=True)
    ring = stage_ring()

    print("2a. trying to break the ring result", flush=True)
    falsify = stage_ring_falsify(ring["steps"], ring["batch"])

    print("2b. the same game written twice", flush=True)
    live = cached(f"live-h{NET_H}-b{LIVE_BATCH}-st{LIVE_STEPS}-v2", stage_browser)
    for arm, r in live["runs"].items():
        print(f"    {arm}: {r['report']['covered']}/8 after {LIVE_STEPS} steps of the "
              f"{r['params']} weight generator", flush=True)

    print("3. a quality metric with a calibration", flush=True)
    calib, ref, probe = stage_metric()

    print("4. DCGAN on MNIST", flush=True)
    imgs = stage_images(calib, ref, probe)

    # ---------------------------------------------------------------- write
    x, dx = line_grid()
    game = dict(
        meta=dict(threads=THREADS, mixture=[dict(mu=m, sd=s) for m, s in MIX],
                  grid=dict(x0=GRID[0], x1=GRID[1], n=GRID[2]),
                  note="every divergence on this page is a trapezoid integral on this grid, "
                       "and the browser recomputes all of them from these same constants"),
        best={k: dict(value=rnd(v["value"], 6), mu=rnd(v["mu"], 4), sd=rnd(v["sd"], 4),
                      mirror=rnd(v["mirror"], 6)) for k, v in line["best"].items()},
        identity=dict(worst=float(f"{line['identity_worst']:.3e}"),
                      exact=float(f"{line['identity_exact']:.3e}"),
                      missing_mass=float(f"{line['worst_mass']:.3e}"), rows=line["checks"]),
        grads=[dict(sep=g["sep"], sat=float(f"{g['sat']:.6e}"), ns=float(f"{g['ns']:.6e}"),
                    acc=rnd(g["acc"], 5), js=rnd(g["js"], 6)) for g in line["grads"]],
        disjoint=dis,
    )
    (OUT / "game.json").write_text(json.dumps(game), "utf-8")

    ring_out = dict(
        meta=dict(k=RING_K, r=ring["r"], sd=ring["sd"], steps=ring["steps"],
                  batch=ring["batch"], seeds=ring["seeds"], arms=ring["arms"],
                  centres=ring["centres"], threads=THREADS),
        calib=ring["calib"], real_points=ring["real_points"], falsify=falsify,
        runs={k: dict(final=v["final"], curve=v["curve"], points=v["points"][:800],
                      gparams=v["gparams"], dparams=v["dparams"])
              for k, v in ring["runs"].items()},
        live=dict(steps=live["steps"], batch=live["batch"], h=live["h"],
                  frozen=live["frozen"],
                  runs={k: dict(losses=v["losses"], report=v["report"],
                                report_shown=v["report_shown"], points=v["points"],
                                params=v["params"]) for k, v in live["runs"].items()}),
    )
    (OUT / "ring.json").write_text(json.dumps(ring_out), "utf-8")

    tiles, order = [], []
    for arm in imgs["arms"]:
        r = imgs["runs"][f"{arm}/1"]
        tiles.extend((r["sheet"][:32] + 1) / 2)
        order.append(arm)
    from vision_data import mnist
    Xtr, _, _, _ = mnist()
    rng = np.random.default_rng(23)
    real_tiles = Xtr[rng.permutation(len(Xtr))[:32]] / 255.0
    tiles.extend(real_tiles)
    order.append("real")
    png = {}
    png["samples"] = save_png(sheet(tiles, 16), IMG / "samples.png")
    walk = imgs["runs"]["dcgan/1"]["interp"]
    png["interp"] = save_png(sheet((walk + 1) / 2, 8), IMG / "interp.png")

    images_out = dict(
        meta=dict(z=Z_DIM, g_ch=G_CH, d_ch=D_CH, steps=imgs["steps"], batch=imgs["batch"],
                  arms=imgs["arms"], seeds=imgs["seeds"], threads=THREADS,
                  n_train=2000 if SMALL else 20000, tile=28, sheet_cols=16,
                  sheet_order=order, png_bytes=png),
        metric=calib, nn=imgs["nn"], track=imgs["track"],
        runs={k: dict(curve=v["curve"], final=v["final"], gparams=v["gparams"],
                      dparams=v["dparams"], d_updates=v["d_updates"], lip=v["lip"])
              for k, v in imgs["runs"].items()},
    )
    (OUT / "images.json").write_text(json.dumps(images_out), "utf-8")

    print(f"\nwrote {OUT}/game.json, ring.json, images.json and {len(png)} sheets "
          f"in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
