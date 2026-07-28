"""Data for the ATLAS 'VQ-VAE' article: the code becomes a number of bits.

Every code this module has produced so far is a point in R^k. A variational
autoencoder narrows it to a distribution over R^k, but it is still a real
vector, and a real vector has no size: writing one down takes as many bits as
you are willing to spend. A VQ-VAE replaces the point by an ENTRY IN A FINITE
ALPHABET, and the moment it does, the code has a length. Four cells drawn from
a codebook of 64 is 24 bits, exactly, and 24 bits is a thing a competitor can
be handed too.

Six blocks, all measured here, nothing cited:

A. Vector quantisation is k-means, on the closed form 2-D density the whole
   module shares. Rate is log2(K) bits, distortion is measured on held out
   samples AND computed exactly by quadrature against the truth, and the
   honest baseline (the same bits spent one coordinate at a time) is fitted
   with the same effort. The gap between the two curves is measured in bits.
B. The gradient that does not exist. Nearest neighbour is piecewise constant,
   so the derivative of the reconstruction term with respect to the encoder
   output is exactly zero almost everywhere. This block measures that: it
   perturbs the encoder output and counts how often the loss changes at all,
   it computes the exact distance to the Voronoi boundary in closed form, and
   it compares the straight-through estimator against the soft relaxation that
   IS differentiable, across nine temperatures.
C. Codebook collapse, measured. Gradient codebook loss or EMA, with or without
   a k-means initialisation, three seeds each, plus a sweep of the commitment
   cost. Codes ever used, perplexity of the usage, dead codes, error.
D. The same number of bits, honestly. A continuous bottleneck whose code is
   quantised to exactly the same total, two ways (post hoc, and trained with
   quantisation noise so the comparison is not a straw man), same data, same
   training budget, same bits, three seeds so there is a floor.
E. A codebook cannot sample by itself. Uniform code grids, decoded and scored
   by the module's shared judge, against a fitted autoregressive prior over
   the cells and against real digits, Fashion-MNIST and pixel noise.
F. The export: encoder, decoder, codebook and prior in float16, with reference
   inputs and outputs computed ON THE QUANTISED WEIGHTS, because those are the
   numbers the browser will have.

Run from anywhere: `python src/utils/generate_vqvae_data.py`.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans as scipy_kmeans
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import generative_common as gc                       # noqa: E402
from src.utils.generative_common import (                           # noqa: E402
    SEED, arr, banner, cached, decode_half, dense, digits14, export_flat,
    export_half, fashion14, judge, mat, rnd, sci, threads, toy_grid, toy_pdf,
    toy_sample, write_json)

# The output directory is NOT created at import time: a generator that is
# already running owns its output directory, and creating it early is how a
# 109 minute run once died at the write step.
OUT = ROOT / "vq-vae" / "data"

# ============================================================== configuration
# Everything that decides a number lives here, so the smoke copy is a diff of
# this block and nothing else, and so every cache key can quote it whole.
THREADS = threads()

# ---- A: the 2-D rate distortion sweep
A_KS = [2, 4, 8, 16, 32, 64, 128, 256]     # 1 to 8 bits
A_SCALAR_EXP = [0, 1, 2, 3, 4, 5, 6, 7]    # 1 to 128 levels per coordinate
A_N_FIT = 20000
A_N_TEST = 20000
A_RESTARTS = 20            # the number of restarts is part of the figure
A_SCIPY_RESTARTS = 10      # the second solver, on the first seed only
A_SEEDS = [0, 1, 2]
A_SHOW_K = 64              # the same codebook size the digit model uses
A_DRAW_KS = [4, 16, 64]    # the assignments the page ships, a subset of A_KS
A_POINTS = 600             # points drawn in the picture
A_POINT_DP = 3             # and the decimals they travel with
A_GRID = 512               # quadrature grid for the exact distortion
A_GRID_REFINE = [512, 1024, 2048]   # the same integral three times, for the floor

# ---- the digit model
P = 196
CELLS = 4
DIM = 8
K = 64
HID = 128
EPOCHS = 80
BATCH = 128
LR = 2e-3
BETA = 0.25
EMA_DECAY = 0.99
EMA_EPS = 1e-5
MONITOR = 4
SEEDS = [19, 20, 21]
BETAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

# ---- B: the gradient probes
B_BATCH = 200
B_DIRS = 4
B_EPS = [1e-6, 3.16e-6, 1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3,
         1e-2, 3.16e-2, 1e-1, 3.16e-1, 1.0]
B_TAUS = [1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1, 3.16e-1, 1.0, 3.16, 10.0]

# ---- D: the same bits, twice
D_LEVELS = 256             # 8 bits per continuous dimension
D_BUDGETS = [{"k": 64, "dims": 3}, {"k": 256, "dims": 4}]   # 24 bits and 32 bits

# ---- E: the prior and the samplers
E_HID = 128
E_EPOCHS = 120
E_VAL = 200                # of the training grids, held back to pick the epoch
E_SAMPLES = 500
E_SEEDS = [0, 1, 2]

# ---- gates
JSON_LIMIT_KB = 700
BUDGET_S = 20 * 60         # below the 25 minute budget, so it can actually fire

assert set(A_DRAW_KS) <= set(A_KS), "the page can only draw codebooks that were fitted"
assert A_SHOW_K in A_KS, "the codebook the page highlights has to be one that was fitted"

T0 = time.time()
STAGE = []


def stage(name):
    """A running clock for the operator. No wall clock second is published:
    thread counts change the order of the floating point reductions, so the
    number of threads is what goes in the JSON and seconds stay in stdout."""
    STAGE.append((name, time.time() - T0))
    print(f"\n[{time.time() - T0:7.1f}s] {name}")


def spread(vals, n=6, sig=False):
    """Mean, extremes and width of a quantity over seeds: the floor under
    every comparison in this file.

    `sig` switches to three significant digits, which is the only honest way to
    write a quantity that can legitimately come out at 1e-30: rounding one of
    those to six decimals writes a measured number as a flat zero.
    """
    v = [float(x) for x in vals]
    f = (lambda x: sci(x)) if sig else (lambda x: rnd(x, n))
    return {"mean": f(np.mean(v)), "min": f(min(v)), "max": f(max(v)),
            "spread": f(max(v) - min(v)), "seeds": len(v)}


def plural(n, word):
    """Counts that reach prose need their singular branch, always."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def bits_of(k):
    b = int(round(math.log2(k)))
    assert 2 ** b == k, f"{k} is not a power of two, so log2(K) is not a bit count"
    return b


def entropy_bits(counts):
    """The entropy of a usage histogram, in bits. This is the rate an ideal
    entropy coder would pay, which is at most log2(K) and equals it only when
    every code is used exactly as often as every other."""
    c = np.asarray(counts, dtype=float)
    p = c / max(c.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def perplexity(counts):
    return float(2.0 ** entropy_bits(counts))


# Configuration is checked here rather than three minutes into the run, where a
# missing showcase codebook would cost the whole of block A.
assert A_SHOW_K in A_KS, "the showcase codebook is not one of the ones being fitted"
assert set(A_DRAW_KS) <= set(A_KS), "the page is asked to draw a codebook nobody fits"
assert bits_of(A_SHOW_K) <= 2 * max(A_SCALAR_EXP), \
    "the scalar sweep does not reach the showcase budget"
for _b in D_BUDGETS:
    assert CELLS * bits_of(_b["k"]) == _b["dims"] * bits_of(D_LEVELS), \
        f"budget {_b} does not spend the same bits on both arms"
assert BETA in BETAS, "the beta sweep does not contain the beta the article uses"
assert len(BETAS) >= 5, "the commitment cost needs at least five rungs"
assert len(SEEDS) >= 2 and len(A_SEEDS) >= 2 and len(E_SEEDS) >= 2, \
    "every comparison in this file needs a noise floor under it"

banner("ATLAS / VQ-VAE: the code stops being a point and becomes a number of bits")
print(f"torch {torch.__version__} on {THREADS} threads, numpy {np.__version__}")

# ============================================================== the shared data
stage("the shared digits, asserted to be the ones the module already used")
FLAT, YD, TR, TE = digits14()
XT, XV = FLAT[TR], FLAT[TE]
print(f"  {len(XT)} digits to fit on, {len(XV)} held out, {P} pixels each, "
      f"and the autoencoders article's own float16 encoder reproduces its own "
      f"codes on them to {gc.DIGITS['assert_worst']}")
JUDGE = judge()
print(f"  the module's shared judge: {JUDGE['accuracy'] * 100:.2f}% on the held out "
      f"digits, mean confidence {JUDGE['real_confidence']:.4f} on them")

XT_T = torch.tensor(XT, dtype=torch.float32)
XV_T = torch.tensor(XV, dtype=torch.float32)


# ================================================================== BLOCK A
# Vector quantisation IS k-means: the objective a codebook minimises, mean
# squared distance from a point to its codeword, is the k-means objective with
# the codewords as centroids. So the first block of this article is a k-means
# sweep, and the only new thing is that the answer is read as a rate.
def scalar_levels(x, levels, seed):
    """The optimal scalar quantiser for one coordinate at `levels` levels, which
    is 1-D k-means (Lloyd-Max). Fitted with the same number of restarts as the
    vector codebook, because the number of restarts is part of the number."""
    if levels == 1:
        return np.array([float(x.mean())])
    km = KMeans(n_clusters=levels, n_init=A_RESTARTS, random_state=seed).fit(
        x.reshape(-1, 1))
    return np.sort(km.cluster_centers_.ravel())


def scalar_apply(x, lv):
    """Nearest level, by bisection on the midpoints. Exact and O(n log L)."""
    if len(lv) == 1:
        return np.full_like(x, lv[0])
    edges = 0.5 * (lv[1:] + lv[:-1])
    return lv[np.searchsorted(edges, x)]


def _block_a():
    grid, cell, _ = toy_grid(A_GRID)
    pgrid = toy_pdf(grid)
    rows_by_seed = {}
    scalar_by_seed = {}
    keep = {}
    for s in A_SEEDS:
        fit = toy_sample(A_N_FIT, seed=1000 + s)
        test = toy_sample(A_N_TEST, seed=2000 + s)
        vec = {}
        for k in A_KS:
            km = KMeans(n_clusters=k, n_init=A_RESTARTS, random_state=s).fit(fit)
            C = km.cluster_centers_
            d_fit = float((((fit - C[km.labels_]) ** 2).sum(1)).mean())
            # Two ways for the same fitted codebook: what sklearn reports and
            # what the centres it returned actually cost.
            inertia_gap = abs(d_fit - km.inertia_ / len(fit)) / max(d_fit, 1e-12)
            tree = cKDTree(C)
            dte, ite = tree.query(test)
            d_test = float((dte ** 2).mean())
            se = float((dte ** 2).std(ddof=1) / np.sqrt(len(test)))
            dg, ig = tree.query(grid)
            d_exact = float((pgrid * dg ** 2).sum() * cell)
            counts = np.bincount(ite, minlength=k)
            row = {"k": k, "bits": bits_of(k), "fit": d_fit, "test": d_test,
                   "test_se": se, "exact": d_exact, "inertia_gap": inertia_gap,
                   "usage_bits": entropy_bits(counts), "centres": C}
            if s == A_SEEDS[0]:
                # A second solver on the same points. scipy starts from random
                # data points rather than k-means++, so a gap here is a
                # statement about restarts and initialisation, not about the
                # objective, which is the same one.
                C2, _ = scipy_kmeans(fit, k, iter=A_SCIPY_RESTARTS, seed=s)
                d2 = float((cKDTree(C2).query(fit)[0] ** 2).mean())
                row["scipy_fit"] = d2
                row["scipy_gap"] = (d2 - d_fit) / d_fit
                if k <= 16:
                    # cKDTree against brute force, on the grid, so the exact
                    # integral does not rest on one nearest neighbour routine.
                    sub = grid[::37]
                    brute = np.argmin(((sub[:, None, :] - C[None, :, :]) ** 2).sum(-1), 1)
                    row["tree_vs_brute"] = int((brute != tree.query(sub)[1]).sum())
            vec[k] = row
        # The scalar quantiser is separable, so each coordinate is fitted once
        # per level count and every bit split is a sum of two numbers already
        # measured. That is also why its exact distortion is cheap.
        sc = {}
        for coord in (0, 1):
            for e in A_SCALAR_EXP:
                lv = scalar_levels(fit[:, coord], 2 ** e, seed=s)
                q_test = scalar_apply(test[:, coord], lv)
                q_grid = scalar_apply(grid[:, coord], lv)
                sc[(coord, e)] = {
                    "levels": lv,
                    "test": float(((test[:, coord] - q_test) ** 2).mean()),
                    "exact": float((pgrid * (grid[:, coord] - q_grid) ** 2).sum() * cell),
                    "usage_bits": entropy_bits(np.bincount(
                        np.searchsorted(0.5 * (lv[1:] + lv[:-1]), test[:, coord])
                        if len(lv) > 1 else np.zeros(len(test), dtype=int),
                        minlength=len(lv))),
                }
        rows_by_seed[s] = vec
        scalar_by_seed[s] = sc
        if s == A_SEEDS[0]:
            keep = {"fit": fit, "test": test}
    # the same integral on three grids, to measure its own floor: the
    # integrand is piecewise smooth (it kinks on every Voronoi boundary), so
    # unlike the normalising constant it does NOT converge exponentially
    refine = []
    C_show = rows_by_seed[A_SEEDS[0]][A_SHOW_K]["centres"]
    for m in A_GRID_REFINE:
        g2, c2, _ = toy_grid(m)
        d2g, _ = cKDTree(C_show).query(g2)
        refine.append({"m": m, "exact": float((toy_pdf(g2) * d2g ** 2).sum() * c2)})
    return {"vec": rows_by_seed, "sc": scalar_by_seed, "refine": refine, "keep": keep}


stage("A. vector quantisation is k-means, and the answer is a rate")
A = cached(
    "vqvae-A-v1-" + json.dumps({"ks": A_KS, "exp": A_SCALAR_EXP, "nfit": A_N_FIT,
                                "ntest": A_N_TEST, "restarts": A_RESTARTS,
                                "scipy": A_SCIPY_RESTARTS, "seeds": A_SEEDS,
                                "grid": A_GRID, "refine": A_GRID_REFINE}, sort_keys=True),
    _block_a)

# the scalar curve: for each total budget, the best split of the bits. The split
# is chosen per seed, so the baseline is the best scalar quantiser that seed
# could have built rather than the one the first seed happened to prefer.
scalar_curve = []
for b in range(0, 2 * max(A_SCALAR_EXP) + 1):
    per_seed, best_split = [], None
    for s in A_SEEDS:
        opts = []
        for ex in A_SCALAR_EXP:
            ey = b - ex
            if ey in A_SCALAR_EXP:
                d = A["sc"][s][(0, ex)]["test"] + A["sc"][s][(1, ey)]["test"]
                de = A["sc"][s][(0, ex)]["exact"] + A["sc"][s][(1, ey)]["exact"]
                ub = A["sc"][s][(0, ex)]["usage_bits"] + A["sc"][s][(1, ey)]["usage_bits"]
                opts.append((d, de, ex, ey, ub))
        if not opts:
            continue
        opts.sort()
        per_seed.append(opts[0])
        if s == A_SEEDS[0]:
            best_split = (opts[0][2], opts[0][3])
    if not per_seed:
        continue
    scalar_curve.append({
        "bits": b, "split": [best_split[0], best_split[1]],
        "per_seed": [o[0] for o in per_seed],
        "test": spread([o[0] for o in per_seed]),
        "exact": spread([o[1] for o in per_seed]),
        # what an entropy coder would actually pay for this grid, against the
        # b bits its index nominally costs
        "usage_bits": spread([o[4] for o in per_seed], 4),
    })

rows_a = []
for k in A_KS:
    b = bits_of(k)
    per = [A["vec"][s][k] for s in A_SEEDS]
    sc = next((r for r in scalar_curve if r["bits"] == b), None)
    # The scalar baseline has to exist at every budget the vector curve uses,
    # or the comparison quietly loses a row. It exists whenever the per axis
    # level counts can be split, which A_SCALAR_EXP has to cover.
    assert sc is not None, (
        f"no scalar quantiser at {b} bits: A_SCALAR_EXP = {A_SCALAR_EXP} cannot "
        f"split {b} bits across two axes")
    ratio = [sc_test / v["test"] for v, sc_test in zip(per, sc["per_seed"])]
    rows_a.append({
        "k": k, "bits": b,
        "vector_test": spread([v["test"] for v in per]),
        "vector_exact": spread([v["exact"] for v in per]),
        "vector_fit": spread([v["fit"] for v in per]),
        "scalar_test": sc["test"], "scalar_exact": sc["exact"],
        "scalar_split": sc["split"],
        "ratio": spread(ratio, 4),
        "usage_bits": spread([v["usage_bits"] for v in per], 4),
        "test_se": rnd(float(np.mean([v["test_se"] for v in per])), 6),
        "codebook": mat(A["vec"][A_SEEDS[0]][k]["centres"], 4),
    })
    r = rows_a[-1]
    print(f"  K = {k:>3} ({plural(b, 'bit')}): vector {r['vector_test']['mean']:.5f} "
          f"(exact {r['vector_exact']['mean']:.5f}), the same bits one coordinate "
          f"at a time {r['scalar_test']['mean']:.5f} with a {sc['split'][0]}+{sc['split'][1]} "
          f"split, a ratio of {r['ratio']['mean']:.3f} "
          f"(spread over {len(A_SEEDS)} seeds {r['ratio']['spread']:.3f})")

# how far apart the two curves are, measured in bits rather than in error
extra_bits = []
sc_bits = np.array([r["bits"] for r in scalar_curve], dtype=float)
sc_log = np.log2(np.array([r["test"]["mean"] for r in scalar_curve]))
for r in rows_a:
    target = math.log2(r["vector_test"]["mean"])
    hit = None
    for i in range(len(sc_bits) - 1):
        lo, hi = sc_log[i], sc_log[i + 1]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            hit = sc_bits[i] + (target - lo) / (hi - lo) * (sc_bits[i + 1] - sc_bits[i])
            break
    extra_bits.append({"bits": r["bits"],
                       "scalar_bits_needed": None if hit is None else rnd(hit, 3),
                       "extra": None if hit is None else rnd(hit - r["bits"], 3)})
have = [e for e in extra_bits if e["extra"] is not None]
if have:
    print(f"  read sideways instead of downwards: to reach what "
          f"{plural(have[-1]['bits'], 'bit')} of vector quantisation reaches, scalar "
          f"quantisation needs {have[-1]['scalar_bits_needed']} bits, which is "
          f"{have[-1]['extra']} bits of pure overhead for spending them one "
          f"coordinate at a time")
missing = [e["bits"] for e in extra_bits if e["extra"] is None]
if missing:
    print(f"  (at {', '.join(str(v) for v in missing)} bits the scalar curve never "
          f"reaches the vector one inside a sweep that runs to {int(sc_bits.max())} bits, "
          f"so no crossing is quoted there)")

# the slope of the curve, measured, not assumed. A straight line needs three
# points before it means anything, so below that the slope is not published.
SLOPE_FROM = 3
fit_rows = [r for r in rows_a if r["bits"] >= SLOPE_FROM]
sc_rows = [r for r in scalar_curve if r["bits"] >= SLOPE_FROM]
if len(fit_rows) >= 3 and len(sc_rows) >= 3:
    sl = float(np.polyfit([r["bits"] for r in fit_rows],
                          [math.log2(r["vector_test"]["mean"]) for r in fit_rows], 1)[0])
    sl_sc = float(np.polyfit([r["bits"] for r in sc_rows],
                             [math.log2(r["test"]["mean"]) for r in sc_rows], 1)[0])
    parallel = abs(sl - sl_sc) < 0.1
    print(f"  above {SLOPE_FROM} bits the vector curve falls {-sl:.3f} in log2 of the "
          f"error per bit and the scalar one {-sl_sc:.3f}: "
          + ("the two are close to parallel, so the gain is an offset and not a different "
             "exponent" if parallel else
             "the two are NOT parallel, so the gain grows with the rate rather than "
             "sitting at a constant number of bits"))
else:
    sl = sl_sc = None
    print(f"  the slope of the two curves is not published: above {SLOPE_FROM} bits there "
          f"are only {len(fit_rows)} and {len(sc_rows)} points, and a line through two "
          f"points is not a measurement")

ref = A["refine"]
q_floor = abs(ref[-1]["exact"] - ref[-2]["exact"]) / ref[-1]["exact"]
print(f"  the exact distortion is an integral over a function that kinks on every "
      f"Voronoi boundary, so it does NOT inherit the exponential convergence of the "
      f"normalising constant: at K = {A_SHOW_K} the three grids give "
      f"{', '.join('%.6f' % r['exact'] for r in ref)}, so the last refinement moves it "
      f"by {q_floor:.2e} and that is the floor of any comparison against it")

# what the page draws
fit0, test0 = A["keep"]["fit"], A["keep"]["test"]
draw_pts = np.round(test0[:A_POINTS], A_POINT_DP)
draw_assign, draw_mismatch = {}, {}
for k in A_DRAW_KS:
    Craw = A["vec"][A_SEEDS[0]][k]["centres"]
    Cr = np.round(Craw, 4)
    # The assignment is computed on the numbers the JSON actually ships, so the
    # browser reproduces it exactly; the mismatch against full precision is
    # measured rather than assumed to be zero.
    a_round = np.argmin(((draw_pts[:, None, :] - Cr[None, :, :]) ** 2).sum(-1), 1)
    a_full = np.argmin(((test0[:A_POINTS, None, :] - Craw[None, :, :]) ** 2).sum(-1), 1)
    draw_assign[k] = a_round
    draw_mismatch[k] = int((a_round != a_full).sum())
print(f"  the picture ships {A_POINTS} points at {A_POINT_DP} decimals and their "
      f"assignments computed on those same rounded numbers: against full precision "
      f"they differ on "
      f"{', '.join('%d point(s) at K = %d' % (draw_mismatch[k], k) for k in A_DRAW_KS)}")

show_split = next(r for r in scalar_curve if r["bits"] == bits_of(A_SHOW_K))["split"]
toy = {
    "config": {"ks": A_KS, "n_fit": A_N_FIT, "n_test": A_N_TEST,
               "restarts": A_RESTARTS, "scipy_restarts": A_SCIPY_RESTARTS,
               "seeds": len(A_SEEDS), "grid": A_GRID,
               "note": "distortion is the mean squared distance from a point to its "
                       "codeword; rate is log2(K) bits per point"},
    "rows": rows_a,
    "scalar_curve": [{"bits": r["bits"], "split": r["split"], "test": r["test"],
                      "exact": r["exact"], "usage_bits": r["usage_bits"]}
                     for r in scalar_curve],
    "showcase_note": "the split of the bits between the two coordinates is chosen "
                     "per seed, so the baseline is the best scalar quantiser that "
                     "seed could have built",
    "extra_bits": extra_bits,
    "slope": {"vector": None if sl is None else rnd(sl, 4),
              "scalar": None if sl_sc is None else rnd(sl_sc, 4),
              "from_bits": SLOPE_FROM},
    "quadrature": {"refine": [{"m": r["m"], "exact": rnd(r["exact"], 8)} for r in ref],
                   "floor_rel": sci(q_floor)},
    "showcase": {
        "k": A_SHOW_K, "bits": bits_of(A_SHOW_K),
        "split": show_split,
        "levels_x": arr(A["sc"][A_SEEDS[0]][(0, show_split[0])]["levels"], 4),
        "levels_y": arr(A["sc"][A_SEEDS[0]][(1, show_split[1])]["levels"], 4),
        "points": mat(draw_pts, A_POINT_DP),
        "assign": {str(k): [int(v) for v in draw_assign[k]] for k in A_DRAW_KS},
        "assign_ks": A_DRAW_KS,
        "round_mismatch": {str(k): draw_mismatch[k] for k in A_DRAW_KS},
    },
}


# ================================================================ the networks
def dist2(Z, E, chunk=4096):
    """Squared distances from every row of Z to every codeword, written as a
    difference rather than expanded.

    |z|^2 - 2 z.e + |e|^2 is faster and returns small negative numbers for a
    point sitting on top of a codeword, which is exactly the regime block B
    measures. Everything here is small enough that the honest form wins.
    """
    Zf = Z.reshape(-1, Z.shape[-1])
    out = []
    for s in range(0, len(Zf), chunk):
        b = Zf[s:s + chunk]
        out.append(((b.unsqueeze(1) - E.unsqueeze(0)) ** 2).sum(-1))
    return torch.cat(out, 0)


class VQ(nn.Module):
    """encoder 196 -> hid -> cells*d, a codebook of K vectors of length d, one
    nearest neighbour lookup per cell, decoder cells*d -> hid -> 196.

    The cells are separate code slots rather than image patches: nothing in the
    loss asks a code to own a region of the picture, and block F measures
    whether one ends up doing so anyway.
    """

    def __init__(self, p, hid, cells, d, k, seed, code_scale=1.0):
        super().__init__()
        torch.manual_seed(seed)
        self.p, self.cells, self.d, self.k = p, cells, d, k
        self.enc = nn.Sequential(nn.Linear(p, hid), nn.ReLU(), nn.Linear(hid, cells * d))
        self.dec = nn.Sequential(nn.Linear(cells * d, hid), nn.ReLU(), nn.Linear(hid, p))
        self.E = nn.Parameter(torch.randn(k, d) * code_scale)

    def encode(self, X):
        return self.enc(X).view(-1, self.cells, self.d)

    def assign(self, Z):
        with torch.no_grad():
            return dist2(Z, self.E).argmin(-1).view(Z.shape[0], self.cells)

    def decode_codes(self, Zq):
        return torch.sigmoid(self.dec(Zq.reshape(Zq.shape[0], -1)))

    def forward(self, X):
        ze = self.encode(X)
        idx = self.assign(ze)
        zq = self.E[idx]
        zst = ze + (zq - ze).detach()
        return ze, zq, idx, self.decode_codes(zst)


class Cont(nn.Module):
    """The same shape with a continuous bottleneck of `dims` numbers. Only the
    two matrices touching the bottleneck differ from VQ, and block D reports
    both parameter counts rather than claiming they match."""

    def __init__(self, p, hid, dims, seed, squash):
        super().__init__()
        torch.manual_seed(seed)
        self.dims, self.squash = dims, squash
        self.enc = nn.Sequential(nn.Linear(p, hid), nn.ReLU(), nn.Linear(hid, dims))
        self.dec = nn.Sequential(nn.Linear(dims, hid), nn.ReLU(), nn.Linear(hid, p))

    def code(self, X):
        z = self.enc(X)
        return torch.tanh(z) if self.squash else z

    def decode_codes(self, Z):
        return torch.sigmoid(self.dec(Z))


def vq_state(model):
    return [p.detach().numpy().copy() for p in
            (model.enc[0].weight, model.enc[0].bias, model.enc[2].weight,
             model.enc[2].bias, model.dec[0].weight, model.dec[0].bias,
             model.dec[2].weight, model.dec[2].bias, model.E)]


def vq_restore(cfg, blob):
    m = VQ(P, cfg["hid"], cfg["cells"], cfg["d"], cfg["k"], cfg["seed"])
    with torch.no_grad():
        for p, v in zip((m.enc[0].weight, m.enc[0].bias, m.enc[2].weight, m.enc[2].bias,
                         m.dec[0].weight, m.dec[0].bias, m.dec[2].weight, m.dec[2].bias,
                         m.E), blob["W"]):
            p.copy_(torch.tensor(v))
    return m


def vq_eval(model, X):
    with torch.no_grad():
        ze = model.encode(X)
        idx = model.assign(ze)
        y = model.decode_codes(model.E[idx])
        mse = float(((y - X) ** 2).mean())
    counts = np.bincount(idx.reshape(-1).numpy(), minlength=model.k)
    return mse, counts, idx


def train_vq_raw(cfg):
    torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    model = VQ(P, cfg["hid"], cfg["cells"], cfg["d"], cfg["k"], cfg["seed"],
               cfg["code_scale"])
    with torch.no_grad():
        z0 = model.encode(XT_T)
    init = {"code_rms": float(model.E.detach().pow(2).mean().sqrt()),
            "enc_rms": float(z0.detach().pow(2).mean().sqrt())}
    if cfg["kmeans_init"]:
        km = KMeans(n_clusters=cfg["k"], n_init=10, random_state=cfg["seed"]).fit(
            z0.reshape(-1, cfg["d"]).numpy())
        with torch.no_grad():
            model.E.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    if cfg["ema"]:
        model.E.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"])
    cs = torch.ones(cfg["k"])
    ew = model.E.detach().clone()
    ever = np.zeros(cfg["k"], dtype=bool)
    hist = []
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(XT_T), generator=g)
        for s in range(0, len(XT_T), cfg["batch"]):
            xb = XT_T[perm[s:s + cfg["batch"]]]
            ze = model.encode(xb)
            idx = model.assign(ze)
            ever |= np.bincount(idx.reshape(-1).numpy(), minlength=cfg["k"]) > 0
            zq = model.E[idx]
            zst = ze + (zq - ze).detach()
            y = model.decode_codes(zst)
            rec = ((y - xb) ** 2).mean()
            commit = ((ze - zq.detach()) ** 2).mean()
            loss = rec + cfg["beta"] * commit
            if not cfg["ema"]:
                loss = loss + ((zq - ze.detach()) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg["ema"]:
                with torch.no_grad():
                    oh = F.one_hot(idx.reshape(-1), cfg["k"]).float()
                    cs.mul_(cfg["decay"]).add_(oh.sum(0), alpha=1 - cfg["decay"])
                    ew.mul_(cfg["decay"]).add_(oh.t() @ ze.detach().reshape(-1, cfg["d"]),
                                               alpha=1 - cfg["decay"])
                    tot = cs.sum()
                    sm = (cs + EMA_EPS) / (tot + cfg["k"] * EMA_EPS) * tot
                    model.E.copy_(ew / sm.unsqueeze(1))
        if ep == 0 or (ep + 1) % cfg["monitor"] == 0:
            tr_mse, counts, _ = vq_eval(model, XT_T)
            te_mse, _, _ = vq_eval(model, XV_T)
            hist.append({"epoch": ep + 1, "train": tr_mse, "test": te_mse,
                         "active": int((counts > 0).sum()),
                         "perplexity": perplexity(counts),
                         "ever": int(ever.sum())})
    tr_mse, counts, _ = vq_eval(model, XT_T)
    te_mse, counts_te, _ = vq_eval(model, XV_T)
    return {"W": vq_state(model), "history": hist, "init": init,
            "counts": counts.tolist(), "counts_test": counts_te.tolist(),
            "ever": int(ever.sum()),
            "final": {"train": tr_mse, "test": te_mse,
                      "active": int((counts > 0).sum()),
                      "dead": int((counts == 0).sum()),
                      "perplexity": perplexity(counts),
                      "usage_bits": entropy_bits(counts)}}


def vq_cfg(seed, ema=False, kmeans_init=False, beta=BETA, k=K, epochs=EPOCHS):
    return {"kind": "vq", "cells": CELLS, "d": DIM, "k": k, "hid": HID,
            "epochs": epochs, "batch": BATCH, "lr": LR, "beta": beta, "ema": ema,
            "decay": EMA_DECAY, "eps": EMA_EPS, "kmeans_init": kmeans_init,
            "code_scale": 1.0, "monitor": MONITOR, "seed": seed}


def train_vq(cfg):
    return cached("vqvae-net-v1-" + json.dumps(cfg, sort_keys=True),
                  lambda: train_vq_raw(cfg), quiet=True)


def train_cont_raw(cfg):
    # The noise has to be one quantisation bin wide, and the bin is only known
    # in advance when the code is squashed into a fixed range.
    assert not cfg["noise"] or cfg["squash"], \
        "training with quantisation noise needs a code whose range is known"
    torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    model = Cont(P, cfg["hid"], cfg["dims"], cfg["seed"], cfg["squash"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    step = 2.0 / (cfg["levels"] - 1)
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(XT_T), generator=g)
        for s in range(0, len(XT_T), cfg["batch"]):
            xb = XT_T[perm[s:s + cfg["batch"]]]
            z = model.code(xb)
            if cfg["noise"]:
                # The standard trick from the compression literature: adding
                # uniform noise of one quantisation bin makes the network learn
                # a code that survives being rounded, which is the fairest
                # possible version of the continuous competitor.
                z = z + (torch.rand(z.shape, generator=g) - 0.5) * step
            y = model.decode_codes(z)
            loss = ((y - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    W = [p.detach().numpy().copy() for p in
         (model.enc[0].weight, model.enc[0].bias, model.enc[2].weight, model.enc[2].bias,
          model.dec[0].weight, model.dec[0].bias, model.dec[2].weight, model.dec[2].bias)]
    with torch.no_grad():
        ztr = model.code(XT_T).numpy()
        zte = model.code(XV_T).numpy()
        free = float(((model.decode_codes(model.code(XV_T)) - XV_T) ** 2).mean())
    return {"W": W, "ztr": ztr, "zte": zte, "free": free}


def cont_cfg(seed, dims, squash, noise, epochs=EPOCHS):
    return {"kind": "cont", "dims": dims, "hid": HID, "epochs": epochs, "batch": BATCH,
            "lr": LR, "levels": D_LEVELS, "squash": squash, "noise": noise, "seed": seed}


def train_cont(cfg):
    return cached("vqvae-cont-v1-" + json.dumps(cfg, sort_keys=True),
                  lambda: train_cont_raw(cfg), quiet=True)


# a determinism guard: the same configuration twice, bit for bit, on a run
# short enough to be free. It is what catches a probe that leaves state behind.
stage("the determinism guard: the same configuration twice, bit for bit")
det_cfg = vq_cfg(SEEDS[0], epochs=4)
det_a = train_vq_raw(det_cfg)
det_b = train_vq_raw(det_cfg)
det_worst = max(float(np.abs(a - b).max()) for a, b in zip(det_a["W"], det_b["W"]))
assert det_worst == 0.0, f"training is not deterministic: {det_worst}"
print(f"  two independent runs of the same {det_cfg['epochs']} epoch configuration agree "
      f"to {det_worst:.1e} on every one of "
      f"{sum(a.size for a in det_a['W']):,} weights")


# ================================================================== BLOCK C
# Four ways to keep a codebook, three seeds each, and then what the commitment
# cost does. This runs before block B because it decides which model the rest
# of the article (and the browser) uses.
stage("C. codebook collapse, measured four ways and over a sweep of beta")
VARIANTS = [
    {"id": "grad", "label": "codebook loss by gradient", "ema": False, "kmeans_init": False},
    {"id": "grad_km", "label": "gradient, k-means init", "ema": False, "kmeans_init": True},
    {"id": "ema", "label": "EMA codebook", "ema": True, "kmeans_init": False},
    {"id": "ema_km", "label": "EMA, k-means init", "ema": True, "kmeans_init": True},
]
collapse_rows = []
blobs = {}
for v in VARIANTS:
    runs = [train_vq(vq_cfg(s, ema=v["ema"], kmeans_init=v["kmeans_init"])) for s in SEEDS]
    blobs[v["id"]] = runs
    hlen = len(runs[0]["history"])
    hist = []
    for i in range(hlen):
        pts = [r["history"][i] for r in runs]
        hist.append({"epoch": pts[0]["epoch"],
                     "train": rnd(float(np.mean([p["train"] for p in pts])), 6),
                     "test": rnd(float(np.mean([p["test"] for p in pts])), 6),
                     "active": rnd(float(np.mean([p["active"] for p in pts])), 2),
                     "ever": rnd(float(np.mean([p["ever"] for p in pts])), 2),
                     "perplexity": rnd(float(np.mean([p["perplexity"] for p in pts])), 3)})
    row = {"id": v["id"], "label": v["label"], "ema": v["ema"],
           "kmeans_init": v["kmeans_init"],
           "test": spread([r["final"]["test"] for r in runs]),
           "train": spread([r["final"]["train"] for r in runs]),
           "active": spread([r["final"]["active"] for r in runs], 2),
           "dead": spread([r["final"]["dead"] for r in runs], 2),
           "ever": spread([r["ever"] for r in runs], 2),
           "perplexity": spread([r["final"]["perplexity"] for r in runs], 3),
           "usage_bits": spread([r["final"]["usage_bits"] for r in runs], 4),
           "history": hist,
           "usage": runs[0]["counts"]}
    collapse_rows.append(row)
    print(f"  {v['label']:<26}: {row['active']['mean']:.1f} of {K} codes alive at the end "
          f"({row['ever']['mean']:.1f} ever used), perplexity "
          f"{row['perplexity']['mean']:.2f}, held out error {row['test']['mean']:.6f} "
          f"(seed spread {row['test']['spread']:.6f})")

init0 = blobs["grad"][0]["init"]
print(f"  the codebook starts at rms {init0['code_rms']:.3f} while the untrained "
      f"encoder puts out rms {init0['enc_rms']:.3f}: a "
      f"{init0['code_rms'] / max(init0['enc_rms'], 1e-9):.1f} to 1 mismatch is the whole "
      f"mechanism of collapse, and the k-means arm is the same run with that mismatch "
      f"removed at step 0 and nothing else changed")

best_variant = min(collapse_rows, key=lambda r: r["test"]["mean"])
worst_variant = max(collapse_rows, key=lambda r: r["test"]["mean"])
floor_c = max(r["test"]["spread"] for r in collapse_rows)
sep = worst_variant["test"]["mean"] - best_variant["test"]["mean"]
if abs(sep) > floor_c:
    print(f"  the best of the four is {best_variant['label']} and the worst is "
          f"{worst_variant['label']}, {sep:.6f} apart against a worst seed spread of "
          f"{floor_c:.6f}, so the ordering survives the noise")
else:
    print(f"  the four are {sep:.6f} apart from best to worst against a worst seed "
          f"spread of {floor_c:.6f}: this comparison does NOT resolve, and the page "
          f"has to say so rather than crown one")

beta_rows = []
for b in BETAS:
    runs = [train_vq(vq_cfg(s, ema=False, kmeans_init=False, beta=b)) for s in SEEDS]
    beta_rows.append({
        "beta": b,
        "test": spread([r["final"]["test"] for r in runs]),
        "active": spread([r["final"]["active"] for r in runs], 2),
        "dead": spread([r["final"]["dead"] for r in runs], 2),
        "perplexity": spread([r["final"]["perplexity"] for r in runs], 3),
        "usage_bits": spread([r["final"]["usage_bits"] for r in runs], 4),
    })
    r = beta_rows[-1]
    print(f"  beta = {b:<5}: {r['active']['mean']:.1f} codes alive, perplexity "
          f"{r['perplexity']['mean']:.2f}, held out error {r['test']['mean']:.6f} "
          f"(spread {r['test']['spread']:.6f})")
b_best = min(beta_rows, key=lambda r: r["test"]["mean"])
b_floor = max(r["test"]["spread"] for r in beta_rows)
b_span = max(r["test"]["mean"] for r in beta_rows) - min(r["test"]["mean"] for r in beta_rows)
if b_span > b_floor:
    print(f"  the sweep moves the error by {b_span:.6f} from end to end against a worst "
          f"seed spread of {b_floor:.6f}, and the best is beta = {b_best['beta']}")
else:
    print(f"  the whole sweep moves the error by {b_span:.6f} while the seed spread alone "
          f"is {b_floor:.6f}: beta does not decide anything here, and saying so is the "
          f"result")

MAIN_ID = best_variant["id"]
MAIN_CFG = vq_cfg(SEEDS[0], ema=best_variant["ema"], kmeans_init=best_variant["kmeans_init"])
MAIN = blobs[MAIN_ID][0]
MODEL = vq_restore(MAIN_CFG, MAIN)
print(f"  the model the rest of the article and the browser use is the winner of this "
      f"block: {best_variant['label']}, seed {SEEDS[0]}")


# ========================================================= the browser's copy
# This comes before the sampling block on purpose: what the page will show has
# to be measured on the weights the page will have, and float16 is enough to
# change which codeword a cell picks.
stage("the float16 copy of the model, and what quantising it costs")
blocks = [("enc1", MODEL.enc[0].weight.detach().numpy().T, MODEL.enc[0].bias.detach().numpy()),
          ("enc2", MODEL.enc[2].weight.detach().numpy().T, MODEL.enc[2].bias.detach().numpy()),
          ("dec1", MODEL.dec[0].weight.detach().numpy().T, MODEL.dec[0].bias.detach().numpy()),
          ("dec2", MODEL.dec[2].weight.detach().numpy().T, MODEL.dec[2].bias.detach().numpy())]
spec, b64, v16 = export_half(blocks)
W16 = decode_half(b64, len(v16))
cb_b64, cb16 = export_flat(MODEL.E.detach().numpy())
E16 = decode_half(cb_b64, cb16.size).reshape(K, DIM)


def encode16(X):
    h = np.maximum(dense(W16, spec[0], X), 0.0)
    return dense(W16, spec[1], h).reshape(-1, CELLS, DIM)


def assign16(Z):
    d = ((Z[:, :, None, :] - E16[None, None, :, :]) ** 2).sum(-1)
    return d.argmin(-1)


def decode16(idx):
    idx = np.asarray(idx)
    zq = E16[idx].reshape(len(idx), CELLS * DIM)
    h = np.maximum(dense(W16, spec[2], zq), 0.0)
    return 1.0 / (1.0 + np.exp(-dense(W16, spec[3], h)))


_, counts32, IDX_TE32 = vq_eval(MODEL, XV_T)
Z16 = encode16(FLAT[TE])
I16 = assign16(Z16)
Y16 = decode16(I16)
G_TR = assign16(encode16(FLAT[TR]))
G_TE = I16
counts_tr = np.bincount(G_TR.reshape(-1), minlength=K)
with torch.no_grad():
    Z32 = MODEL.encode(XV_T).numpy()
    Y32 = MODEL.decode_codes(MODEL.E[IDX_TE32]).numpy()
mse32 = float(((Y32 - FLAT[TE]) ** 2).mean())
mse16 = float(((Y16 - FLAT[TE]) ** 2).mean())
idx_moved = int((I16 != IDX_TE32.numpy()).sum())
quant = {"code_shift": sci(float(np.abs(Z16 - Z32).max())),
         "index_changes": idx_moved,
         "index_total": int(G_TE.size),
         "mse_full": rnd(mse32, 6), "mse_quantised": rnd(mse16, 6)}
print(f"  {len(v16):,} weights plus {cb16.size} codebook numbers. float16 moves the "
      f"encoder output by at most {quant['code_shift']} and that is enough to change "
      f"{idx_moved} of {G_TE.size} held out codeword choices, which moves the held out "
      f"error from {mse32:.6f} to {mse16:.6f}. Everything the page will show is measured "
      f"on these numbers from here on.")
print(f"  of the {K} codes, {int((counts_tr > 0).sum())} are used on the training digits "
      f"through the quantised encoder, and the busiest takes "
      f"{counts_tr.max() / counts_tr.sum() * 100:.1f}% of the {counts_tr.sum():,} cells")


# ================================================================== BLOCK B
# The gradient that does not exist.
def block_b(cfg, blob):
    """All of block B for one trained checkpoint, in float64 so that a
    gradient which really is zero reads as zero rather than as underflow."""
    assert (cfg["cells"], cfg["d"], cfg["k"]) == (CELLS, DIM, K), \
        "this block reads the grid shape from the module constants"
    model = vq_restore(cfg, blob).double()
    before = [p.detach().numpy().copy() for p in model.parameters()]
    X = torch.tensor(FLAT[TE][:B_BATCH], dtype=torch.float64)
    with torch.no_grad():
        ze0 = model.encode(X)
        idx0 = model.assign(ze0)
        base_y = model.decode_codes(model.E[idx0])
        base_loss = ((base_y - X) ** 2).mean(1)

    # what torch itself says about the hard path
    ze = ze0.detach().clone().requires_grad_(True)
    idx = model.assign(ze)
    y = model.decode_codes(model.E[idx])
    hard_loss = ((y - X) ** 2).mean(1).sum()
    hard_g = torch.autograd.grad(hard_loss, ze, allow_unused=True)[0]
    hard_max = 0.0 if hard_g is None else float(hard_g.abs().max())

    # the straight-through gradient, and a finite difference check of it
    ze = ze0.detach().clone().requires_grad_(True)
    zq = model.E[model.assign(ze)]
    zst = ze + (zq - ze).detach()
    st_loss = ((model.decode_codes(zst) - X) ** 2).mean(1).sum()
    g_st = torch.autograd.grad(st_loss, ze)[0].detach()
    gen = torch.Generator().manual_seed(11)
    u = torch.randn(zq.shape, generator=gen, dtype=torch.float64)
    u = u / u.reshape(len(u), -1).norm(dim=1).view(-1, 1, 1)
    h = 1e-6
    with torch.no_grad():
        lp = ((model.decode_codes(zq + h * u) - X) ** 2).mean(1).sum()
        lm = ((model.decode_codes(zq - h * u) - X) ** 2).mean(1).sum()
    fd = float((lp - lm) / (2 * h))
    an = float((g_st * u).sum())
    fd_gap = abs(fd - an) / max(abs(an), 1e-30)

    # The exact distance to the wall of the Voronoi cell, in closed form. The
    # wall between codewords i and j is the plane where the two squared
    # distances are equal, so the perpendicular distance to it is
    # (d2_j - d2_i) / (2 |e_j - e_i|). The smallest of those over j is the
    # distance to the cell itself, and moving along that wall's normal crosses
    # it first, before any other: for any other wall k the crossing happens at
    # t_k / (n_j . n_k), which is at least t_k, which is at least t_j.
    with torch.no_grad():
        d2 = dist2(ze0.detach(), model.E).reshape(len(X), CELLS, -1)
        di = d2.gather(2, idx0.unsqueeze(-1))
        diff = model.E.unsqueeze(0) - model.E[idx0].unsqueeze(2)      # (n, cells, K, d)
        nrm = diff.norm(dim=-1)
        t = (d2 - di) / (2 * nrm.clamp_min(1e-30))
        t.scatter_(2, idx0.unsqueeze(-1), float("inf"))
        margin, jstar = t.min(2)
        normal = torch.gather(diff, 2, jstar.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, DIM)).squeeze(2)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    def flips(direction, eps):
        """Fraction of (image, cell) pairs whose codeword changes, and whose
        reconstruction loss changes at all, when that one cell is moved by eps
        along `direction`. The two numbers must agree exactly."""
        chg_idx = chg_loss = tot = 0
        for c in range(CELLS):
            z = ze0.detach().clone()
            z[:, c, :] = z[:, c, :] + eps * direction[:, c, :]
            with torch.no_grad():
                idx = model.assign(z)
                loss = ((model.decode_codes(model.E[idx]) - X) ** 2).mean(1)
            chg_idx += int((idx[:, c] != idx0[:, c]).sum())
            chg_loss += int((loss != base_loss).sum())
            tot += len(X)
        return chg_idx, chg_loss, tot

    gen = torch.Generator().manual_seed(23)
    families = {}
    mism = 0
    # a cell whose straight-through gradient is zero has no direction to be
    # pushed in, so it cannot flip and would bias that family downwards
    zero_grad = int((g_st.norm(dim=-1) == 0).sum())
    for name in ("random", "gradient", "normal"):
        rows = []
        for eps in B_EPS:
            ci = cl = tt = 0
            reps = B_DIRS if name == "random" else 1
            for r in range(reps):
                if name == "random":
                    u = torch.randn(ze0.shape, generator=gen, dtype=torch.float64)
                    u = u / u.norm(dim=-1, keepdim=True)
                elif name == "gradient":
                    u = g_st / g_st.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                else:
                    u = normal
                a, b, t_ = flips(u, eps)
                ci += a
                cl += b
                tt += t_
            mism += abs(ci - cl)
            rows.append({"eps": eps, "changed": rnd(ci / tt, 6),
                         "loss_changed": rnd(cl / tt, 6)})
        quiet = [r["eps"] for r in rows if r["changed"] == 0.0]
        families[name] = {"rows": rows,
                          "largest_quiet_eps": max(quiet) if quiet else None,
                          "first_moving_eps": next((r["eps"] for r in rows
                                                    if r["changed"] > 0), None)}

    # the closed form and the simulation, on the same quantity
    with torch.no_grad():
        below = model.assign(ze0 + (margin * (1 - 1e-6)).unsqueeze(-1) * normal)
        above = model.assign(ze0 + (margin * (1 + 1e-6)).unsqueeze(-1) * normal)
    exact_ok = float(((below == idx0) & (above != idx0)).double().mean())

    # the soft relaxation, which is differentiable, at every temperature in the
    # sweep: softmax(-|z - e|^2 / tau) against the codebook, which converges to
    # the hard lookup in the forward direction as tau goes to zero
    taus = []
    for tau in B_TAUS:
        ze = ze0.detach().clone().requires_grad_(True)
        d2s = dist2(ze, model.E).reshape(len(X), CELLS, -1)
        w = torch.softmax(-d2s / tau, dim=-1)
        zs = w @ model.E
        soft_loss = ((model.decode_codes(zs) - X) ** 2).mean(1)
        g_soft = torch.autograd.grad(soft_loss.sum(), ze)[0].detach()
        a = g_st.reshape(len(X), -1)
        b = g_soft.reshape(len(X), -1)
        cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-300)
        ratio = b.norm(dim=1) / a.norm(dim=1).clamp_min(1e-300)
        with torch.no_grad():
            gap = float((zs - model.E[idx0]).norm(dim=-1).mean()
                        / model.E[idx0].norm(dim=-1).mean())
        taus.append({"tau": tau, "cosine": float(cos.mean()),
                     "cosine_sd": float(cos.std()),
                     "norm_ratio": float(ratio.mean()),
                     "forward_gap": gap,
                     "soft_loss": float(soft_loss.mean()),
                     "hard_loss": float(base_loss.mean())})

    after = [p.detach().numpy().copy() for p in model.parameters()]
    left_behind = max(float(np.abs(a - b).max()) for a, b in zip(before, after))
    return {"hard_max": hard_max, "st_norm": float(g_st.norm(dim=(1, 2)).mean()),
            "fd_gap": fd_gap, "fd_value": fd, "families": families,
            "margin": {"min": float(margin.min()), "median": float(margin.median()),
                       "max": float(margin.max()), "mean": float(margin.mean())},
            "exact_ok": exact_ok, "index_loss_mismatch": mism, "taus": taus,
            "left_behind": left_behind, "zero_grad_cells": zero_grad,
            "code_rms": float(model.E[idx0].norm(dim=-1).mean())}


stage("B. the gradient that does not exist, and the two estimators that pretend it does")
B_RUNS = [cached("vqvae-B-v1-" + json.dumps({"cfg": vq_cfg(s, ema=best_variant["ema"],
                                                           kmeans_init=best_variant["kmeans_init"]),
                                             "batch": B_BATCH, "dirs": B_DIRS,
                                             "eps": B_EPS, "taus": B_TAUS},
                                            sort_keys=True),
                 (lambda s=s: block_b(vq_cfg(s, ema=best_variant["ema"],
                                             kmeans_init=best_variant["kmeans_init"]),
                                      train_vq(vq_cfg(s, ema=best_variant["ema"],
                                                      kmeans_init=best_variant["kmeans_init"])))),
                 quiet=True)
         for s in SEEDS]
B0 = B_RUNS[0]
assert all(r["left_behind"] == 0.0 for r in B_RUNS), "a probe left the model changed"
assert all(r["index_loss_mismatch"] == 0 for r in B_RUNS), \
    "the loss moved without the codeword moving, which cannot happen"
print(f"  torch's own answer for the gradient of the reconstruction term with respect to "
      f"the encoder output, through the hard nearest neighbour, is exactly "
      f"{B0['hard_max']:.1f} on all {B_BATCH * CELLS * DIM:,} entries: there is no path")
print(f"  and it is not a rounding effect. Moving one cell's encoder output and decoding "
      f"again, the loss is bit for bit identical whenever the codeword does not change: "
      f"{sum(r['index_loss_mismatch'] for r in B_RUNS)} disagreements out of "
      f"{len(B_RUNS) * len(B_EPS) * (B_DIRS + 2) * B_BATCH * CELLS:,} probes")
for name in ("random", "gradient", "normal"):
    q = [v for v in (r["families"][name]["largest_quiet_eps"] for r in B_RUNS)
         if v is not None]
    f0 = [v for v in (r["families"][name]["first_moving_eps"] for r in B_RUNS)
          if v is not None]
    label = {"random": "a random direction", "gradient": "the direction training moves in",
             "normal": "the worst possible direction"}[name]
    quiet_txt = (f"nothing at all changes up to eps = {min(q):.2e}" if q else
                 f"something already changes at the smallest eps tried, {min(B_EPS):.0e}")
    move_txt = (f"{min(f0):.2e}" if f0 else
                f"never inside a sweep that reaches {max(B_EPS):.0e}")
    print(f"    along {label}: {quiet_txt}, and the first eps that moves anything is "
          f"{move_txt}")
mg = [r["margin"] for r in B_RUNS]
print(f"  the closed form says why: the median distance from an encoder output to the "
      f"wall of its own Voronoi cell is {np.mean([m['median'] for m in mg]):.4f} against "
      f"codewords of length {np.mean([r['code_rms'] for r in B_RUNS]):.3f}, and the "
      f"closest any of {B_BATCH * CELLS} cells gets to a wall is "
      f"{min(m['min'] for m in mg):.2e}")
print(f"  pushing exactly along the wall's normal flips the codeword exactly at that "
      f"distance on {min(r['exact_ok'] for r in B_RUNS) * 100:.1f}% of the cells, which "
      f"is the closed form and the simulation agreeing on the same number")
# and the two agree a second way: the smallest margin in the batch has to sit
# between the largest step that changed nothing and the first step that did
bracket_ok = 0
for r in B_RUNS:
    fam = r["families"]["normal"]
    lo = fam["largest_quiet_eps"] or 0.0
    hi = fam["first_moving_eps"] or float("inf")
    bracket_ok += int(lo < r["margin"]["min"] <= hi)
assert bracket_ok == len(B_RUNS), \
    "the closed form margin is not bracketed by the simulation that measures it"
print(f"  and on all {len(B_RUNS)} checkpoints the smallest margin in the batch lands "
      f"between the largest step that changed nothing and the first step that did, which "
      f"is the second way those two agree")
print(f"  the straight-through gradient is the decoder's own gradient copied past the "
      f"quantiser, and a central difference on the decoder confirms it to "
      f"{max(r['fd_gap'] for r in B_RUNS):.2e} relative")
print("  the soft relaxation, which really is differentiable, per temperature:")
# One table, averaged over the checkpoints, and every sentence below plus the
# JSON reads from it: a "best tau" taken from one seed and a table taken from
# three is two different claims wearing the same name.
TAU_ROWS = []
for i in range(len(B_TAUS)):
    cs = [r["taus"][i]["cosine"] for r in B_RUNS]
    nr = [r["taus"][i]["norm_ratio"] for r in B_RUNS]
    fg = [r["taus"][i]["forward_gap"] for r in B_RUNS]
    sl_ = [r["taus"][i]["soft_loss"] for r in B_RUNS]
    TAU_ROWS.append({"tau": B_RUNS[0]["taus"][i]["tau"],
                     "cosine": spread(cs, 4), "norm_ratio": spread(nr, 8),
                     "forward_gap": spread(fg, 8), "soft_loss": spread(sl_, 6)})
    print(f"    tau = {TAU_ROWS[-1]['tau']:<7.4g}: forward gap {np.mean(fg):.2e}, "
          f"cosine with straight-through {np.mean(cs):+.4f} "
          f"(spread {max(cs) - min(cs):.4f}), norm ratio {np.mean(nr):.3e}")
tau_cos = max(TAU_ROWS, key=lambda t: t["cosine"]["mean"])
tau_ratio = min(TAU_ROWS,
                key=lambda t: abs(math.log(max(t["norm_ratio"]["mean"], 1e-300))))
tau_small = TAU_ROWS[0]
where = ("at the small end" if tau_cos["tau"] == tau_small["tau"] else
         "not at the small end")
print(f"  read that column by column. As tau falls the FORWARD converges (the gap reaches "
      f"{tau_small['forward_gap']['mean']:.1e} at tau = {tau_small['tau']}), and the "
      f"GRADIENT does not: its norm goes to {tau_small['norm_ratio']['mean']:.2e} of the "
      f"straight-through one. The best agreement in direction is "
      f"{tau_cos['cosine']['mean']:+.4f} at tau = {tau_cos['tau']}, which is {where}, and "
      f"the closest the two norms ever get is a factor of "
      f"{tau_ratio['norm_ratio']['mean']:.3g} at tau = {tau_ratio['tau']}.")
if tau_small["norm_ratio"]["mean"] < 0.5 and tau_cos["cosine"]["mean"] < 0.999:
    print("  so the straight-through estimator is NOT the zero temperature limit of the "
          "soft relaxation. It is the separate decision to pretend the Jacobian is the "
          "identity, and no temperature in this sweep reproduces it.")
else:
    print(f"  against the plan, the soft relaxation DOES approach the straight-through "
          f"estimator here: cosine {tau_cos['cosine']['mean']:.4f} and norm ratio "
          f"{tau_small['norm_ratio']['mean']:.3f} at the small end. The article follows "
          f"the measurement.")


# ================================================================== BLOCK D
stage("D. the same number of bits, spent two ways")


def quantise_uniform(Z, lo, hi, levels):
    step = (hi - lo) / (levels - 1)
    clipped = np.clip(Z, lo, hi)
    q = lo + np.rint((clipped - lo) / step) * step
    return q, float(np.mean((Z < lo) | (Z > hi))), step


def code_symbols(Z, lo, hi, levels):
    step = (hi - lo) / (levels - 1)
    return np.rint((np.clip(Z, lo, hi) - lo) / step).astype(int)


bits_rows = []
for bud in D_BUDGETS:
    kk, dims = bud["k"], bud["dims"]
    total_bits = CELLS * bits_of(kk)
    assert total_bits == dims * bits_of(D_LEVELS), \
        "the two arms are not spending the same number of bits"
    vq_runs = [train_vq(vq_cfg(s, ema=best_variant["ema"],
                               kmeans_init=best_variant["kmeans_init"], k=kk))
               for s in SEEDS]
    vq_bits = []
    for r in vq_runs:
        c = np.array(r["counts_test"], dtype=float)
        vq_bits.append(CELLS * entropy_bits(c))
    arms = {"vq": {"label": f"{plural(CELLS, 'cell')} from a codebook of {kk}",
                   "test": spread([r["final"]["test"] for r in vq_runs]),
                   "entropy_bits": spread(vq_bits, 3),
                   "free": None, "clipped": 0.0}}
    for tag, squash, noise in (("post", False, False), ("noise", True, True)):
        runs = [train_cont(cont_cfg(s, dims, squash, noise)) for s in SEEDS]
        tests, ents, clips, frees = [], [], [], []
        for r in runs:
            lo = r["ztr"].min(0) if not squash else np.full(dims, -1.0)
            hi = r["ztr"].max(0) if not squash else np.full(dims, 1.0)
            q, clip, _ = quantise_uniform(r["zte"], lo, hi, D_LEVELS)
            m = Cont(P, HID, dims, 0, squash)
            with torch.no_grad():
                for p, v in zip((m.enc[0].weight, m.enc[0].bias, m.enc[2].weight,
                                 m.enc[2].bias, m.dec[0].weight, m.dec[0].bias,
                                 m.dec[2].weight, m.dec[2].bias), r["W"]):
                    p.copy_(torch.tensor(v))
                y = m.decode_codes(torch.tensor(q, dtype=torch.float32))
                tests.append(float(((y - XV_T) ** 2).mean()))
            sym = code_symbols(r["zte"], lo, hi, D_LEVELS)
            ents.append(sum(entropy_bits(np.bincount(sym[:, j], minlength=D_LEVELS))
                            for j in range(dims)))
            clips.append(clip)
            frees.append(r["free"])
        arms[tag] = {
            "label": (f"{plural(dims, 'number')} at {bits_of(D_LEVELS)} bits each, "
                      + ("quantised afterwards" if tag == "post"
                         else "trained with the noise")),
            "test": spread(tests), "entropy_bits": spread(ents, 3),
            "free": spread(frees), "clipped": rnd(float(np.mean(clips)), 5)}
    # parameter counts, because they are the thing that is NOT matched. The
    # codebook is counted apart from the network, the way the prose says it.
    n_vq = (P * HID + HID) + (HID * CELLS * DIM + CELLS * DIM) + \
           (CELLS * DIM * HID + HID) + (HID * P + P)
    n_ct = (P * HID + HID) + (HID * dims + dims) + (dims * HID + HID) + (HID * P + P)
    winner = min(arms.items(), key=lambda kv: kv[1]["test"]["mean"])
    floor_d = max(a["test"]["spread"] for a in arms.values())
    others = sorted((v["test"]["mean"] for k_, v in arms.items() if k_ != winner[0]))
    row = {"bits": total_bits, "k": kk, "dims": dims, "arms": arms,
           "params_vq": n_vq, "params_cont": n_ct, "codebook_params": kk * DIM,
           "winner": winner[0], "margin": rnd(others[0] - winner[1]["test"]["mean"], 6),
           "floor": rnd(floor_d, 6),
           "resolved": bool(abs(others[0] - winner[1]["test"]["mean"]) > floor_d)}
    bits_rows.append(row)
    print(f"  {total_bits} bits per image, three ways:")
    for tag in ("vq", "post", "noise"):
        a = arms[tag]
        extra = "" if a["free"] is None else \
            f", and {a['free']['mean']:.6f} if it is allowed to keep the numbers whole"
        print(f"    {a['label']:<52}: {a['test']['mean']:.6f} "
              f"(seed spread {a['test']['spread']:.6f}){extra}")
    print(f"    the winner is {arms[winner[0]]['label']}, ahead by {row['margin']:.6f} "
          f"against a worst seed spread of {floor_d:.6f}"
          + (", so the ordering survives the noise" if row["resolved"] else
             ", which is INSIDE the noise, so this budget does not decide anything"))
    print(f"    what is matched: the data, the held out set, the architecture outside the "
          f"bottleneck, the optimiser, {plural(EPOCHS, 'epoch')}, batch {BATCH}, "
          f"{plural(len(SEEDS), 'seed')}, and {total_bits} bits exactly. What is NOT "
          f"matched: the parameter count ({n_vq:,} against {n_ct:,}, because the decoder "
          f"of the discrete arm reads {CELLS * DIM} numbers and the continuous one reads "
          f"{dims}), and the codebook itself, {kk * DIM} more numbers of shared model "
          f"state that the bit count does not charge for, because it is sent once and "
          f"not per image")
    print(f"    measured entropies rather than nominal bits: neither can exceed "
          f"{total_bits} by construction, and measured, the discrete code spends "
          f"{arms['vq']['entropy_bits']['mean']:.3f} bits and the post hoc continuous one "
          f"{arms['post']['entropy_bits']['mean']:.3f}. Both are sums of per cell "
          f"marginals, so both are upper bounds on the joint")
    if arms["post"]["clipped"] > 0:
        print(f"    the continuous arm's range comes from the TRAINING codes, so "
              f"{arms['post']['clipped'] * 100:.2f}% of the held out code entries land "
              f"outside it and get clipped: that cost is inside the number above")


# ================================================================== BLOCK E
stage("E. a codebook cannot sample by itself")


class Prior(nn.Module):
    """An autoregressive prior over the cells: cell t from the cells before it,
    a masked one-hot input and a position marker, one shared MLP."""

    def __init__(self, cells, k, hid, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.cells, self.k = cells, k
        self.net = nn.Sequential(nn.Linear(cells * k + cells, hid), nn.ReLU(),
                                 nn.Linear(hid, k))

    def forward(self, X):
        return self.net(X)


def prior_inputs(G, cells, k):
    n = len(G)
    oh = np.zeros((n, cells, k), dtype=np.float32)
    oh[np.arange(n)[:, None], np.arange(cells)[None, :], G] = 1.0
    X = np.zeros((n, cells, cells * k + cells), dtype=np.float32)
    for t in range(cells):
        if t:
            X[:, t, :t * k] = oh[:, :t, :].reshape(n, -1)
        X[:, t, cells * k + t] = 1.0
    return X


def train_prior_raw(cfg):
    torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"] + 3)
    fit, val = G_TR[:-E_VAL], G_TR[-E_VAL:]
    Xf = torch.tensor(prior_inputs(fit, CELLS, K).reshape(-1, CELLS * K + CELLS))
    yf = torch.tensor(fit.reshape(-1))
    Xv = torch.tensor(prior_inputs(val, CELLS, K).reshape(-1, CELLS * K + CELLS))
    yv = torch.tensor(val.reshape(-1))
    Xt = torch.tensor(prior_inputs(G_TE, CELLS, K).reshape(-1, CELLS * K + CELLS))
    yt = torch.tensor(G_TE.reshape(-1))
    net = Prior(CELLS, K, cfg["hid"], cfg["seed"])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    best = (1e9, None, 0)
    curve = []
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(Xf), generator=g)
        for s in range(0, len(Xf), 256):
            i = perm[s:s + 256]
            opt.zero_grad()
            F.cross_entropy(net(Xf[i]), yf[i]).backward()
            opt.step()
        with torch.no_grad():
            v = float(F.cross_entropy(net(Xv), yv)) / math.log(2)
            t = float(F.cross_entropy(net(Xt), yt)) / math.log(2)
        curve.append({"epoch": ep + 1, "val": v, "test": t})
        if v < best[0]:
            best = (v, [p.detach().numpy().copy() for p in net.parameters()], ep + 1)
    with torch.no_grad():
        for p, v in zip(net.parameters(), best[1]):
            p.copy_(torch.tensor(v))
        test_bits = float(F.cross_entropy(net(Xt), yt)) / math.log(2)
        train_bits = float(F.cross_entropy(net(Xf), yf)) / math.log(2)
    return {"W": best[1], "val_bits": best[0], "test_bits": test_bits,
            "train_bits": train_bits, "epoch": best[2], "curve": curve}


def train_prior(cfg):
    return cached("vqvae-prior-v1-" + json.dumps(cfg, sort_keys=True),
                  lambda: train_prior_raw(cfg), quiet=True)


PRIOR_RUNS = [train_prior({"hid": E_HID, "epochs": E_EPOCHS, "val": E_VAL,
                           "cells": CELLS, "k": K, "seed": s,
                           "main": json.dumps(MAIN_CFG, sort_keys=True)})
              for s in E_SEEDS]
P_BEST = min(PRIOR_RUNS, key=lambda r: r["val_bits"])

# the counting reference: independent marginals per cell, add one smoothing
marg = np.stack([np.bincount(G_TR[:, c], minlength=K) + 1.0 for c in range(CELLS)])
marg = marg / marg.sum(1, keepdims=True)
marg_bits = float(-np.log2(marg[np.arange(CELLS)[None, :], G_TE]).mean())
uniform_bits = float(bits_of(K))


def sample_prior(W, n, seed):
    rs = np.random.RandomState(seed)
    W1, b1, W2, b2 = W
    G = np.zeros((n, CELLS), dtype=int)
    for t in range(CELLS):
        X = prior_inputs(G, CELLS, K)[:, t, :]
        h = np.maximum(X @ W1.T + b1, 0.0)
        z = h @ W2.T + b2
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(1, keepdims=True)
        cum = p.cumsum(1)
        u = rs.rand(n, 1)
        G[:, t] = (u > cum).sum(1)
    return np.clip(G, 0, K - 1)


def decode_grids(G):
    """Through the float16 decoder, which is the one the page will run."""
    return decode16(np.asarray(G))


def decode_grids32(G):
    with torch.no_grad():
        return MODEL.decode_codes(MODEL.E[torch.tensor(np.asarray(G))]).numpy()


def score(imgs):
    p = JUDGE["proba"](imgs)
    lab = p.argmax(1)
    h = np.bincount(lab, minlength=10) / len(lab)
    return {"confidence": float(p.max(1).mean()),
            "class_entropy": entropy_bits(np.bincount(lab, minlength=10)),
            "classes": [int(v) for v in np.bincount(lab, minlength=10)],
            "top_class_share": float(h.max())}


samplers = {}
half_gap = 0.0
for i, s in enumerate(E_SEEDS):
    r = np.random.RandomState(100 + s)
    Gu = r.randint(0, K, size=(E_SAMPLES, CELLS))
    samplers.setdefault("uniform", []).append(score(decode_grids(Gu)))
    # the same sample through the full precision decoder, so the page's numbers
    # and the trained model's numbers are known to be the same claim
    half_gap = max(half_gap, abs(samplers["uniform"][-1]["confidence"]
                                 - score(decode_grids32(Gu))["confidence"]))
    Gm = np.stack([r.choice(K, size=E_SAMPLES, p=marg[c]) for c in range(CELLS)], 1)
    samplers.setdefault("marginal", []).append(score(decode_grids(Gm)))
    samplers.setdefault("prior", []).append(
        score(decode_grids(sample_prior(PRIOR_RUNS[i]["W"], E_SAMPLES, 200 + s))))
samplers["reconstruction"] = [score(decode_grids(G_TE))]
samplers["real"] = [score(FLAT[TE])]
FASH, _ = fashion14(n=len(TE), seed=SEED)
samplers["fashion"] = [score(FASH)]
samplers["noise"] = [score(np.random.RandomState(4).rand(len(TE), P))]

prior_out = {
    "uniform_bits": uniform_bits,
    "marginal_bits": rnd(marg_bits, 4),
    "ar_bits": spread([r["test_bits"] for r in PRIOR_RUNS], 4),
    "ar_train_bits": spread([r["train_bits"] for r in PRIOR_RUNS], 4),
    "ar_epoch": [int(r["epoch"]) for r in PRIOR_RUNS],
    "cells": CELLS, "k": K,
    "saved_bits_per_image": rnd(CELLS * (uniform_bits
                                         - float(np.mean([r["test_bits"]
                                                          for r in PRIOR_RUNS]))), 3),
    "saved_bits_marginal": rnd(CELLS * (uniform_bits - marg_bits), 3),
    "samplers": {name: {"confidence": spread([v["confidence"] for v in vals], 4),
                        "class_entropy": spread([v["class_entropy"] for v in vals], 3),
                        "top_class_share": spread([v["top_class_share"] for v in vals], 4),
                        "classes": vals[0]["classes"]}
                 for name, vals in samplers.items()},
    "judge_accuracy": rnd(JUDGE["accuracy"], 4),
    "judge_real_confidence": rnd(JUDGE["real_confidence"], 4),
    "half_precision_gap": sci(half_gap),
    "samples": E_SAMPLES,
}
# the judge scoring real held out digits here must reproduce the number the
# shared module measured when it trained the judge, or the two are not the
# same object
judge_gap = abs(prior_out["samplers"]["real"]["confidence"]["mean"]
                - JUDGE["real_confidence"])
assert judge_gap < 1e-4, f"the shared judge does not reproduce its own number: {judge_gap}"
print(f"  a uniform prior over the grid costs {uniform_bits:.3f} bits per cell by "
      f"definition. Counting the cells one at a time and ignoring how they combine gets "
      f"{marg_bits:.4f}, and the fitted autoregressive prior gets "
      f"{prior_out['ar_bits']['mean']:.4f} on the held out grids (spread "
      f"{prior_out['ar_bits']['spread']:.4f} over {plural(len(E_SEEDS), 'seed')})")
ar_mean = prior_out["ar_bits"]["mean"]
if ar_mean > marg_bits + prior_out["ar_bits"]["spread"]:
    print(f"  and against the plan, the fitted prior is WORSE than counting each cell on "
          f"its own ({ar_mean:.4f} against {marg_bits:.4f} bits per cell, with a seed "
          f"spread of {prior_out['ar_bits']['spread']:.4f} under it). With {CELLS} cells "
          f"there is very little joint structure left for it to find, and the honest "
          f"compression number for this page is the counting one, "
          f"{prior_out['saved_bits_marginal']:.2f} bits per image")
else:
    print(f"  the joint structure is worth something: the autoregressive prior beats "
          f"counting each cell on its own by {marg_bits - ar_mean:.4f} bits per cell, "
          f"which is {CELLS * (marg_bits - ar_mean):.2f} bits per image on top of what "
          f"the marginals already save, and {prior_out['saved_bits_per_image']:.2f} bits "
          f"per image against the uniform code the rate of this model is quoted in")
for name in ("real", "reconstruction", "prior", "marginal", "uniform", "fashion", "noise"):
    v = prior_out["samplers"][name]
    print(f"    {name:<15}: judge confidence {v['confidence']['mean']:.4f} "
          f"(spread {v['confidence']['spread']:.4f}), the most popular of the ten classes "
          f"takes {v['top_class_share']['mean'] * 100:.1f}% of them")
c_real = prior_out["samplers"]["real"]["confidence"]["mean"]
c_uni = prior_out["samplers"]["uniform"]["confidence"]["mean"]
c_ar = prior_out["samplers"]["prior"]["confidence"]["mean"]
c_noise = prior_out["samplers"]["noise"]["confidence"]["mean"]
floor_e = prior_out["samplers"]["prior"]["confidence"]["spread"]
if c_ar - c_uni > floor_e:
    print(f"  the prior does move the score, from {c_uni:.4f} to {c_ar:.4f} against a "
          f"seed spread of {floor_e:.4f}")
else:
    print(f"  the prior moves the score by {c_ar - c_uni:+.4f} against a seed spread of "
          f"{floor_e:.4f}: it does not resolve, and the page says so")
print(f"  and read it against the right reference: real held out digits score {c_real:.4f} "
      f"and uniform pixel noise already scores {c_noise:.4f}, so the interesting distance "
      f"is from {c_noise:.4f} to {c_real:.4f} and the samples sit at {c_ar:.4f}. "
      f"A confidence like that is not a digit, it is a judge with ten choices and no "
      f"option to abstain.")


# ================================================================== BLOCK F
stage("F. what the browser gets, assembled")

# Does a cell own a region of the picture? Measured, not assumed: sweep one
# cell over the whole codebook with the rest of the grid held fixed and see
# where the picture moves.
foot = np.zeros((CELLS, P))
for c in range(CELLS):
    acc = []
    for g in G_TE[:16]:
        grids = np.tile(g, (K, 1))
        grids[:, c] = np.arange(K)
        acc.append(decode16(grids).std(0))
    foot[c] = np.mean(acc, 0)
quad = np.zeros((CELLS, 4))
side = int(round(math.sqrt(P)))
half = side // 2
img_q = np.zeros((side, side), dtype=int)
img_q[:half, :half], img_q[:half, half:] = 0, 1
img_q[half:, :half], img_q[half:, half:] = 2, 3
for c in range(CELLS):
    for q_ in range(4):
        quad[c, q_] = foot[c][img_q.ravel() == q_].sum()
quad = quad / quad.sum(1, keepdims=True)
even = 100.0 / 4
strongest = quad.max(1)
if strongest.max() < 2 * even / 100:
    verdict = (f"and measured, none does: the strongest quadrant of a cell takes between "
               f"{strongest.min() * 100:.1f}% and {strongest.max() * 100:.1f}% of the "
               f"change it causes, against the {even:.0f}% of a cell that owns nothing. "
               f"Changing one of the {CELLS} codes repaints the whole digit")
else:
    verdict = (f"and yet one does: the strongest quadrant of a cell takes up to "
               f"{strongest.max() * 100:.1f}% of the change it causes, against the "
               f"{even:.0f}% of a cell that owns nothing, so locality appears here "
               f"without anything asking for it")
print(f"  nothing in the loss asks a code to own a corner of the image, {verdict}.")

SHOW = [int(np.where(YD[TE] == c)[0][0]) for c in range(10) if (YD[TE] == c).any()]
REF = SHOW[:6]
prior_blocks = [("p1", P_BEST["W"][0].T, P_BEST["W"][1]),
                ("p2", P_BEST["W"][2].T, P_BEST["W"][3])]
p_spec, p_b64, p_v16 = export_half(prior_blocks)
PW16 = decode_half(p_b64, len(p_v16))
ref_in = prior_inputs(G_TE[:4], CELLS, K).reshape(-1, CELLS * K + CELLS)
ref_logits = dense(PW16, p_spec[1], np.maximum(dense(PW16, p_spec[0], ref_in), 0.0))
# the prior the page will sample from is the quantised one, so the sample score
# it produces is measured here too rather than assumed to be the same
W16_prior = [PW16[p_spec[0]["w_offset"]:p_spec[0]["w_offset"] + p_spec[0]["w_size"]]
             .reshape(p_spec[0]["shape"][1], p_spec[0]["shape"][0]),
             PW16[p_spec[0]["b_offset"]:p_spec[0]["b_offset"] + p_spec[0]["b_size"]],
             PW16[p_spec[1]["w_offset"]:p_spec[1]["w_offset"] + p_spec[1]["w_size"]]
             .reshape(p_spec[1]["shape"][1], p_spec[1]["shape"][0]),
             PW16[p_spec[1]["b_offset"]:p_spec[1]["b_offset"] + p_spec[1]["b_size"]]]
prior_half = score(decode_grids(sample_prior(W16_prior, E_SAMPLES, 900)))
prior_full = score(decode_grids(sample_prior(P_BEST["W"], E_SAMPLES, 900)))
prior_half_gap = abs(prior_half["confidence"] - prior_full["confidence"])
print(f"  the prior the page ships is the float16 one: sampling {E_SAMPLES} grids from it "
      f"scores {prior_half['confidence']:.4f} against {prior_full['confidence']:.4f} from "
      f"the weights training produced, a gap of {prior_half_gap:.2e}")

net_out = {
    "format": "float16 little-endian, base64; each layer's weights are row major over "
              "output units, then its biases",
    "arch": {"pixels": P, "hidden": HID, "cells": CELLS, "dim": DIM, "k": K,
             "bits": CELLS * bits_of(K),
             "encoder": f"{P} -> {HID} relu -> {CELLS * DIM}, read as {CELLS} vectors "
                        f"of {DIM}",
             "decoder": f"{CELLS * DIM} -> {HID} relu -> {P} sigmoid"},
    "layers": spec, "count": int(len(v16)), "weights_b64": b64,
    "codebook_b64": cb_b64, "codebook_count": int(cb16.size),
    "codebook": mat(E16, 4),
    "usage": [int(v) for v in counts_tr],
    "quant": quant,
    "footprint": mat(foot, 5),
    "quadrant_share": mat(quad, 4),
    "reference": [{"index": int(i), "label": int(YD[TE][i]),
                   "pixels": arr(FLAT[TE][i], 4),
                   "z": arr(Z16[i], 4),
                   "codes": [int(v) for v in I16[i]],
                   "decoded": arr(Y16[i], 4)} for i in REF],
    "showcase": [{"index": int(i), "label": int(YD[TE][i]),
                  "pixels": arr(FLAT[TE][i], 4),
                  "codes": [int(v) for v in I16[i]]} for i in SHOW],
    "prior": {"layers": p_spec, "count": int(len(p_v16)), "weights_b64": p_b64,
              "input": f"{CELLS * K} masked one-hot entries then {CELLS} position flags",
              "reference_logits": mat(ref_logits, 4),
              "reference_grids": [[int(v) for v in g] for g in G_TE[:4]],
              "test_bits": rnd(P_BEST["test_bits"], 4),
              "sample_confidence": rnd(prior_half["confidence"], 4)},
}

guards = {
    "quadrature_mass": sci(abs(float((toy_pdf(toy_grid(A_GRID)[0]).sum()
                                      * toy_grid(A_GRID)[1]) - 1.0))),
    "exact_vs_holdout": [
        {"k": r["k"],
         "gap": rnd(abs(r["vector_exact"]["mean"] - r["vector_test"]["mean"]), 6),
         "sampling_se": r["test_se"]} for r in rows_a],
    "quadrature_floor_rel": sci(q_floor),
    "inertia_gap": sci(max(A["vec"][s][k]["inertia_gap"] for s in A_SEEDS for k in A_KS)),
    "scipy_gap": {str(k): rnd(A["vec"][A_SEEDS[0]][k]["scipy_gap"], 5) for k in A_KS},
    "tree_vs_brute": int(sum(A["vec"][A_SEEDS[0]][k].get("tree_vs_brute", 0)
                             for k in A_KS)),
    "assign_round_mismatch": {str(k): draw_mismatch[k] for k in A_DRAW_KS},
    "determinism_worst": det_worst,
    "probe_left_behind": max(r["left_behind"] for r in B_RUNS),
    "index_loss_mismatch": int(sum(r["index_loss_mismatch"] for r in B_RUNS)),
    "boundary_exact_fraction": rnd(min(r["exact_ok"] for r in B_RUNS), 6),
    "margin_bracketed": bracket_ok,
    "finite_difference_gap": sci(max(r["fd_gap"] for r in B_RUNS)),
    "hard_gradient_max": max(r["hard_max"] for r in B_RUNS),
    "float16_index_changes": idx_moved,
    "float16_code_shift": quant["code_shift"],
    "judge_reproduces_itself": sci(judge_gap),
    "half_precision_sample_gap": sci(half_gap),
    "half_precision_prior_gap": sci(prior_half_gap),
    "zero_gradient_cells": int(max(r["zero_grad_cells"] for r in B_RUNS)),
}
worst_scipy = max(A["vec"][A_SEEDS[0]][k]["scipy_gap"] for k in A_KS)
best_scipy = min(A["vec"][A_SEEDS[0]][k]["scipy_gap"] for k in A_KS)
print(f"  the second k-means solver, started from random data points with "
      f"{A_SCIPY_RESTARTS} restarts instead of k-means++ with {A_RESTARTS}, lands between "
      f"{best_scipy * 100:+.2f}% and {worst_scipy * 100:+.2f}% of the published "
      f"distortion: the number of restarts is part of the figure, and this is how much")
print(f"  cKDTree and brute force disagree on {guards['tree_vs_brute']} of the grid "
      f"assignments they were both asked for")

out = {
    "meta": {
        "seed": SEED, "seeds": SEEDS, "threads": THREADS,
        "torch": torch.__version__, "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "digits": {"n": int(len(FLAT)), "train": int(len(TR)), "test": int(len(TE)),
                   "pixels": P},
        "training": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "beta": BETA,
                     "ema_decay": EMA_DECAY, "hidden": HID},
        "main_variant": MAIN_ID,
        "reuse_check": "the digits are the ones the autoencoders article fitted on, "
                       "asserted by running that article's own float16 encoder on them",
        "note": "no wall clock second is published anywhere in this file: the thread "
                "count above changes the order of the floating point reductions, so "
                "times are not comparable between machines and are not quoted",
    },
    "toy": toy,
    "gradient": {
        "batch": B_BATCH, "directions": B_DIRS, "seeds": len(B_RUNS),
        "hard_gradient_max": max(r["hard_max"] for r in B_RUNS),
        "st_norm": spread([r["st_norm"] for r in B_RUNS], 6),
        "margin": {k_: spread([r["margin"][k_] for r in B_RUNS], 6)
                   for k_ in ("min", "median", "mean", "max")},
        "code_rms": spread([r["code_rms"] for r in B_RUNS], 4),
        "families": {
            name: {
                "rows": [{"eps": B_RUNS[0]["families"][name]["rows"][i]["eps"],
                          "changed": spread([r["families"][name]["rows"][i]["changed"]
                                             for r in B_RUNS], 6)}
                         for i in range(len(B_EPS))],
                "largest_quiet_eps": min([r["families"][name]["largest_quiet_eps"]
                                          for r in B_RUNS
                                          if r["families"][name]["largest_quiet_eps"]]
                                         or [None]),
                "first_moving_eps": min([r["families"][name]["first_moving_eps"]
                                         for r in B_RUNS
                                         if r["families"][name]["first_moving_eps"]]
                                        or [None]),
            } for name in ("random", "gradient", "normal")},
        "taus": TAU_ROWS,
        "hard_loss": rnd(float(np.mean([r["taus"][0]["hard_loss"] for r in B_RUNS])), 6),
        "best_cosine_tau": tau_cos["tau"],
    },
    "collapse": {"k": K, "cells": CELLS, "dim": DIM, "epochs": EPOCHS,
                 "variants": collapse_rows, "beta": beta_rows,
                 "init": {"code_rms": rnd(init0["code_rms"], 4),
                          "encoder_rms": rnd(init0["enc_rms"], 4)},
                 "winner": MAIN_ID, "floor": rnd(floor_c, 6)},
    "bits": {"rows": bits_rows, "levels": D_LEVELS,
             "note": "same data, same held out set, same architecture outside the "
                     "bottleneck, same optimiser, same epochs, same seeds, same bits; "
                     "the parameter counts are not the same and are reported"},
    "prior": prior_out,
    "net": net_out,
    "guards": guards,
}

path = write_json(OUT / "vqvae.json", out)
kb = path.stat().st_size / 1024
if kb > JSON_LIMIT_KB:
    print(f"  OVER BUDGET: {kb:.0f} kB against a limit of {JSON_LIMIT_KB} kB. Drop, in "
          f"this order: the prior weights ({len(p_b64) / 1024:.0f} kB), the footprint "
          f"map, the showcase pixels (the page can fetch digits from the autoencoders "
          f"article), and the toy codebooks above K = {A_SHOW_K}.")
else:
    print(f"  under the {JSON_LIMIT_KB} kB budget with {JSON_LIMIT_KB - kb:.0f} kB to spare")

banner("what the measurements said, including where they contradicted the plan")
beaten = [r for r in rows_a if r["ratio"]["min"] > 1.0]
if len(beaten) == len(rows_a):
    print(f"1. Vector quantisation beats spending the same bits one coordinate at a time "
          f"at every one of the {len(rows_a)} rates measured and on every seed, by a "
          f"factor of {rows_a[0]['ratio']['mean']:.2f} at {rows_a[0]['bits']} bit and "
          f"{rows_a[-1]['ratio']['mean']:.2f} at {rows_a[-1]['bits']} bits.")
else:
    lost = [r["bits"] for r in rows_a if r["ratio"]["min"] <= 1.0]
    print(f"1. Against the plan, vector quantisation does NOT win at every rate: at "
          f"{lost} bits at least one seed has the scalar quantiser level with it or "
          f"ahead. It wins at the other {len(beaten)} rates.")
worst_dir = B0["families"]["normal"]["first_moving_eps"]
print(f"2. The reconstruction term has no gradient with respect to the encoder output, "
      f"and that is measured rather than argued: "
      + (f"nothing changes at all until the output moves {worst_dir:.1e} in the worst "
         f"direction." if worst_dir is not None else
         f"in the worst direction something already moves at the smallest step tried, "
         f"{min(B_EPS):.0e}, so the flat region is narrower than this sweep resolves."))
print(f"3. The straight-through estimator is not the limit of the soft relaxation. As the "
      f"temperature falls the forward converges and the gradient dies: at tau = "
      f"{tau_small['tau']} the soft gradient is {tau_small['norm_ratio']['mean']:.1e} of "
      f"the straight-through one. Best direction agreement is "
      f"{tau_cos['cosine']['mean']:+.4f} at tau = {tau_cos['tau']}.")
print(f"4. Codebook health: {best_variant['label']} wins the four way comparison, and "
      f"{worst_variant['label']} loses it, {sep:.6f} apart against a seed spread of "
      f"{floor_c:.6f}.")
for i, row in enumerate(bits_rows):
    w = row["arms"][row["winner"]]
    print(f"5.{i + 1} At {row['bits']} bits the winner is {w['label']}, by "
          f"{row['margin']:.6f} against a floor of {row['floor']:.6f}"
          + ("." if row["resolved"] else ", which is inside the floor: unresolved."))
print(f"6. A codebook does not sample. Uniform grids score {c_uni:.4f} with the shared "
      f"judge and a fitted prior {c_ar:.4f}, against {c_real:.4f} for real digits and "
      f"{c_noise:.4f} for pixel noise.")

total = time.time() - T0
print(f"\ntotal {total:.0f} s ({total / 60:.1f} min) on {THREADS} threads")
if total > BUDGET_S:
    print(f"OVER THE {BUDGET_S / 60:.0f} MINUTE GATE. Real cuts, biggest first: drop "
          f"K = 256 from A_KS and from D_BUDGETS (about a third of block A and a fifth "
          f"of block D), cut A_RESTARTS from {A_RESTARTS} to 10, cut A_SEEDS and SEEDS "
          f"from 3 to 2 (a third of every training in C and D), cut A_N_FIT from "
          f"{A_N_FIT:,} to 8,000. Cosmetic afterwards: the scipy cross check and the "
          f"{A_GRID_REFINE} refinement study.")
