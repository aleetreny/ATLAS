"""Data for the ATLAS 'Alone' web article, the second of module 2.3.

The previous article ended on a defect: every classical rule estimates its
yardstick from the sample that contains the suspect, so a batch of eight
planted bottles hid itself completely inside 178 real ones. Four detectors
answer that, and no two of them mean the same thing by "strange":

  * an ISOLATION FOREST asks how few random cuts it takes to separate a point
    from the rest. Nothing about density, nothing about a centre.
  * LOCAL OUTLIER FACTOR compares a point's density with the density of its own
    neighbours, so "strange" is relative to the neighbourhood and not to the
    dataset.
  * a ONE-CLASS SVM draws a boundary around the data in a kernel space and asks
    only which side of it a point falls on.
  * the MINIMUM COVARIANCE DETERMINANT keeps the previous article's ellipse and
    fixes its estimator: fit the mean and the covariance on the tightest half of
    the data, then measure everybody against that.

Results worth knowing before reading the code. Three of them are not what the
plan said, and in each case the plan is what changed:

  * the forest's cuts are axis parallel and it shows, but NOT where it was
    supposed to. The intended demonstration, two clusters on a diagonal with the
    empty corners between them, did not reproduce: the forest scores an empty
    corner most of the way to a genuine outlier and gets 0.99 average precision
    on that dataset. Where it does show is on ONE round blob: past four standard
    deviations the score along an axis stops moving entirely while the score
    along the diagonal keeps climbing.
  * "more data makes an isolation forest worse" is a claim about CLUSTERED
    anomalies, not about size. On 4,120 rows with a dense cluster of anomalies
    it reproduces hard (1.0000 at 16 samples per tree against 0.36 at all of
    them); on 252 rows with scattered ones it does not happen at all, and every
    step up the ladder helps. Both are published.
  * the one-class SVM's nu property is two inequalities and they do not fare the
    same. The support vector half holds in 24 settings of 24. The outlier half
    holds in 12, and the failures are a function of gamma alone, with a worst
    overshoot of nearly eight times what nu promised.
  * a global k-nearest-neighbour distance and LOF disagree by design on data
    with two densities, and every one of the distance's mistakes is a member of
    the loose cluster.
  * MCD finds the eight planted bottles the previous article's Mahalanobis
    could not see, on exactly the same rows, asserted against that article's
    JSON. On two clusters the same repair makes things worse, because a robust
    estimator fixes the estimator and not the model.
  * no detector wins on all five datasets, the ranking swaps, and the isolation
    forest never comes first while never collapsing either.

The isolation forest is written twice, here and in the page, over the same
linear congruential generator, so the browser builds the same forest this file
does and the guard can compare tree for tree rather than distribution for
distribution.

No wall clock is published anywhere in this article.

Run from anywhere: `python src/utils/generate_isolation_data.py`.
"""
import json
import math
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.covariance import MinCovDet
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "isolation" / "data"
OUT.mkdir(parents=True, exist_ok=True)

DP = 4
rnd = lambda v, n=4: round(float(v), n)
SEED = 19

print("=" * 70)
print("Alone: four detectors, four meanings of strange")
print("=" * 70)


# ============================================================= the shared rng
# Written identically in `isolation/js/forest.js`. numpy's generator cannot be
# reproduced in a browser, and a forest whose trees the page cannot rebuild is
# a forest the page can only illustrate.
class LCG:
    """s <- (1664525 s + 1013904223) mod 2^32, the Numerical Recipes constants."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next_u32(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def unit(self):
        return self.next_u32() / 4294967296.0

    def below(self, n):
        return int(self.unit() * n)


# ======================================================== the isolation forest
def c_factor(n):
    """The average path length of an unsuccessful search in a binary search
    tree, which is what a subtree of n points would have cost on average. This
    is the normaliser that turns a depth into a score."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649015329) - 2.0 * (n - 1) / n


def build_tree(X, idx, depth, limit, rng):
    """A node is (feature, split, left, right) or a leaf ('leaf', size).

    The order of the draws is the contract with the browser: one draw for the
    feature, one for the split, then the left subtree and then the right."""
    n = len(idx)
    if depth >= limit or n <= 1:
        return {"leaf": n}
    f = rng.below(X.shape[1])
    col = X[idx, f]
    lo, hi = float(col.min()), float(col.max())
    if hi <= lo:
        return {"leaf": n}
    split = lo + rng.unit() * (hi - lo)
    left = idx[col < split]
    right = idx[col >= split]
    if len(left) == 0 or len(right) == 0:
        return {"leaf": n}
    return {"f": f, "s": split,
            "l": build_tree(X, left, depth + 1, limit, rng),
            "r": build_tree(X, right, depth + 1, limit, rng)}


def sample_without_replacement(n, k, rng):
    """A partial Fisher-Yates over 0..n-1, written the same way on both sides."""
    order = list(range(n))
    k = min(k, n)
    for i in range(k):
        j = i + rng.below(n - i)
        order[i], order[j] = order[j], order[i]
    return order[:k]


def build_forest(X, n_trees, psi, seed):
    rng = LCG(seed)
    limit = int(math.ceil(math.log2(max(2, min(psi, len(X))))))
    trees = []
    for _ in range(n_trees):
        take = np.array(sample_without_replacement(len(X), psi, rng))
        trees.append(build_tree(X, take, 0, limit, rng))
    return {"trees": trees, "psi": min(psi, len(X)), "limit": limit}


def path_length(node, x):
    d = 0
    while "leaf" not in node:
        node = node["l"] if x[node["f"]] < node["s"] else node["r"]
        d += 1
    return d + c_factor(node["leaf"])


def forest_score(forest, Q):
    c = c_factor(forest["psi"])
    out = np.empty(len(Q))
    for i, x in enumerate(Q):
        h = sum(path_length(t, x) for t in forest["trees"]) / len(forest["trees"])
        out[i] = 2.0 ** (-h / c) if c > 0 else 0.0
    return out


# ==================================================================== datasets
def two_density(seed=SEED):
    """The picture LOF was invented for: a tight cluster, a loose one, and
    points that are strange only relative to the tight one. A global distance
    has to call the loose cluster's own members stranger than those."""
    g = np.random.default_rng(seed)
    tight = g.normal([0, 0], 0.32, size=(120, 2))
    loose = g.normal([5.2, 5.2], 1.15, size=(120, 2))
    # six sitting a short walk outside the tight cluster, closer to its centre
    # than a typical member of the loose one is to its own
    ang = np.linspace(0, 2 * math.pi, 7)[:6] + 0.3
    near = np.c_[1.35 * np.cos(ang), 1.35 * np.sin(ang)]
    far = np.array([[2.6, 8.4], [8.7, 1.1], [-3.2, 4.6], [7.9, 8.9], [-2.4, -2.9], [9.1, 5.4]])
    X = np.vstack([tight, loose, near, far])
    y = np.r_[np.zeros(240), np.ones(12)]
    return X, y


def corners(seed=SEED + 1):
    """Two clusters on a diagonal, and points in the two empty corners. Any
    axis parallel method has to separate the corners from the clusters using
    only vertical and horizontal cuts, and it cannot."""
    g = np.random.default_rng(seed)
    a = g.normal([0, 0], 0.75, size=(100, 2))
    b = g.normal([10, 10], 0.75, size=(100, 2))
    odd = np.vstack([g.normal([0, 10], 0.55, size=(5, 2)),
                     g.normal([10, 0], 0.55, size=(5, 2))])
    X = np.vstack([a, b, odd])
    y = np.r_[np.zeros(200), np.ones(10)]
    return X, y


def downsampled(X, y, keep_class, anomaly_class, n_anom, seed):
    g = np.random.default_rng(seed)
    normal = X[y == keep_class] if np.isscalar(keep_class) else X[np.isin(y, keep_class)]
    pool = X[y == anomaly_class]
    take = g.permutation(len(pool))[:n_anom]
    Xo = np.vstack([normal, pool[take]])
    yo = np.r_[np.zeros(len(normal)), np.ones(n_anom)]
    return Xo, yo


wine = load_wine()
Xw, yw = downsampled(wine.data, wine.target, [0, 1], 2, 8, SEED)
cancer = load_breast_cancer()
# in this file malignant is 0 and benign is 1; the rare class is the anomaly
Xc, yc = downsampled(cancer.data, cancer.target, 1, 0, 18, SEED)

# The previous article's contaminated file, rows unchanged. If its JSON is
# there, reuse is asserted rather than declared.
prev = ROOT / "outliers" / "data" / "outliers.json"
planted_ok = None
if prev.exists():
    P = json.loads(prev.read_text(encoding="utf-8"))
    base = np.array(P["wine"]["matrix"])
    plant = np.array(P["many_d"]["masking"]["planted"])[:P["many_d"]["masking"]["lost"]]
    Xp = np.vstack([base, plant])
    yp = np.r_[np.zeros(len(base)), np.ones(len(plant))]
    planted_ok = {
        "wine_gap": float(np.abs(base - np.round(wine.data, DP)).max()),
        "n_planted": int(len(plant)),
        "prev_flagged": int(next(r["flagged"] for r in P["many_d"]["masking"]["rows"]
                                if r["m"] == len(plant))),
        "prev_cut": P["many_d"]["chi_cut"],
    }
    print(f"\nthe previous article's contaminated file, reused: {len(base)} real bottles plus "
          f"{len(plant)} planted, and the matrix matches its JSON to "
          f"{planted_ok['wine_gap']:.1e}")
    print(f"  that article measured {planted_ok['prev_flagged']} of the {len(plant)} flagged by "
          f"plain Mahalanobis")
else:
    raise SystemExit("outliers/data/outliers.json is missing: run generate_outliers_data.py first")

X2, y2 = two_density()
Xk, yk = corners()
# Rounded HERE, before anything is measured on them, because these two are the
# datasets the page holds and rebuilds its forests from. Measured on the
# unrounded arrays instead, the browser reproduced the corners exactly (its
# points are far apart relative to 1e-4) and missed the two densities by 6e-3,
# because in a cluster of standard deviation 0.32 the fourth decimal decides
# which side of a split a point falls on. The rounding is not a detail of the
# export, it is part of the experiment.
X2 = np.round(X2, DP)
Xk = np.round(Xk, DP)

SETS = [
    ("two densities", X2, y2, True),
    ("two clusters and two corners", Xk, yk, True),
    ("wine, cultivar 3 thinned", Xw, yw, False),
    ("biopsies, malignant thinned", Xc, yc, False),
    ("the planted batch", Xp, yp, False),
]
print("\ndatasets:")
for name, X, y, _ in SETS:
    print(f"  {name:<30} {X.shape[0]:>4} rows, {X.shape[1]:>2} columns, "
          f"{int(y.sum()):>3} anomalies ({y.mean():.1%})")

# =========================================================== the forest, checked
# My forest against scikit-learn's, on the same data with the same settings.
# They cannot be identical (different random streams) so what is compared is
# what a user would compare: the ranking each produces.
N_TREES, PSI = 100, 128
forest_check = []
for name, X, y, _ in SETS:
    mine = build_forest(X, N_TREES, PSI, SEED)
    s_mine = forest_score(mine, X)
    sk = IsolationForest(n_estimators=N_TREES, max_samples=min(PSI, len(X)),
                         random_state=SEED).fit(X)
    s_sk = -sk.score_samples(X)
    forest_check.append({
        "name": name,
        "ap_mine": rnd(average_precision_score(y, s_mine)),
        "ap_sklearn": rnd(average_precision_score(y, s_sk)),
        "auc_mine": rnd(roc_auc_score(y, s_mine)),
        "auc_sklearn": rnd(roc_auc_score(y, s_sk)),
        "rank_corr": rnd(float(np.corrcoef(
            np.argsort(np.argsort(s_mine)), np.argsort(np.argsort(s_sk)))[0, 1])),
    })
print(f"\nmy isolation forest against scikit-learn's ({N_TREES} trees, {PSI} samples):")
for r in forest_check:
    print(f"  {r['name']:<30} average precision {r['ap_mine']:.4f} against "
          f"{r['ap_sklearn']:.4f}, rank correlation {r['rank_corr']:.4f}")
worst_forest = min(forest_check, key=lambda r: r["rank_corr"])

# ---- seeds: how much of the difference above is just the random stream
seed_spread = {}
for name, X, y, _ in SETS[:2]:
    aps = [average_precision_score(y, forest_score(build_forest(X, N_TREES, PSI, s), X))
           for s in range(101, 111)]
    seed_spread[name] = {"mean": rnd(float(np.mean(aps))), "sd": rnd(float(np.std(aps))),
                         "min": rnd(min(aps)), "max": rnd(max(aps)), "n": len(aps)}
    print(f"  over ten seeds on the {name}: {seed_spread[name]['mean']:.4f} "
          f"plus or minus {seed_spread[name]['sd']:.4f}")

# ---- the axis parallel bias, measured where it actually shows
#
# The intended experiment was the two empty corners, and it did not reproduce:
# the forest scores a genuinely empty corner almost as high as a point outside
# everything, and gets average precision 0.99 on that dataset. The reason is
# that a subsample drawn mostly from one cluster isolates a corner in one cut,
# so 200 trees between them see it fine.
#
# Where the bias does show is on ONE blob, comparing points at the same distance
# from the centre in different directions. A cut is always vertical or
# horizontal, so a point straight out along an axis is shielded by everything
# the blob has at that height, and a point out along the diagonal is not.
blob_rng = np.random.default_rng(3)
BLOB = blob_rng.normal(0, 1.0, size=(1000, 2))
blob_forest = build_forest(BLOB, 300, 256, 5)
ANGLES = list(range(0, 91, 15))
RADII = [1.0, 2.0, 2.6, 3.2, 4.0, 5.0, 6.5]
radial = []
for r in RADII:
    pts = np.array([[r * math.cos(math.radians(a)), r * math.sin(math.radians(a))]
                    for a in ANGLES])
    s = forest_score(blob_forest, pts)
    radial.append({"r": r, "scores": [rnd(v, 5) for v in s],
                   "axis": rnd(float(s[0]), 5), "diag": rnd(float(s[3]), 5),
                   "gap": rnd(float(s[3] - s[0]), 5)})
print("\nthe axis parallel bias, on one gaussian blob (300 trees):")
for r in radial:
    print(f"  r = {r['r']:>4}: along an axis {r['axis']:.5f}, along the diagonal "
          f"{r['diag']:.5f}, the diagonal ahead by {r['gap']:+.5f}")
axis_sat = [r for r in radial if r["r"] >= 4.0]
axis_flat = max(abs(axis_sat[i]["axis"] - axis_sat[0]["axis"]) for i in range(len(axis_sat)))
diag_climb = axis_sat[-1]["diag"] - axis_sat[0]["diag"]
worst_gap = max(radial, key=lambda r: r["gap"])
print(f"  past r = {axis_sat[0]['r']} the axis score moves {axis_flat:.5f} in total while the "
      f"diagonal climbs {diag_climb:.5f}")

# The corners, kept because the dataset is on the scoreboard, and because the
# result there is the honest counterweight to the paragraph above.
grid_n = 61
gx = np.linspace(-3.5, 13.5, grid_n)
GX, GY = np.meshgrid(gx, gx)
grid_pts = np.c_[GX.ravel(), GY.ravel()]
corner_forest = build_forest(Xk, 200, PSI, SEED)
grid_scores = forest_score(corner_forest, grid_pts)
probes = {
    "inside a cluster": [0.0, 0.0],
    "the empty corner": [0.0, 10.0],
    "the other corner": [10.0, 0.0],
    "between the clusters": [5.0, 5.0],
    "well outside everything": [16.0, 16.0],
    "off one axis only": [16.0, 5.0],
}
probe_scores = {k: rnd(float(forest_score(corner_forest, np.array([v]))[0]), 5)
                for k, v in probes.items()}
print("\nthe forest's score at six places on the corners dataset:")
for k, v in probe_scores.items():
    print(f"  {k:<26} {v:.5f}")
corner_gap = probe_scores["the empty corner"] - probe_scores["inside a cluster"]
outside_gap = probe_scores["well outside everything"] - probe_scores["inside a cluster"]
print(f"  a genuinely empty corner scores {corner_gap:+.5f} against a cluster centre, while a "
      f"point outside everything scores {outside_gap:+.5f}: a factor of "
      f"{outside_gap / corner_gap if corner_gap else float('inf'):.1f}")
# and what the detector does with the ten planted corner points
sk_corner = IsolationForest(n_estimators=200, max_samples=PSI, random_state=SEED).fit(Xk)
corner_ap = {"mine": rnd(average_precision_score(yk, forest_score(corner_forest, Xk))),
             "sklearn": rnd(average_precision_score(yk, -sk_corner.score_samples(Xk)))}

# ---- the subsample sweep, on two datasets because one of them says no
#
# The original paper's claim is that a smaller subsample scores BETTER, and the
# mechanism it gives is masking: a cluster of anomalies large enough to look
# like a small cluster of ordinary points. So the claim is not about size, it is
# about whether the anomalies are clustered, and the honest way to publish it is
# to run it on one dataset of each kind.
def psi_sweep(X, y, grid, seeds=5, trees=N_TREES):
    rows = []
    for psi in grid:
        if psi > len(X):
            psi = len(X)
        aps = [average_precision_score(y, forest_score(build_forest(X, trees, psi, s), X))
               for s in range(201, 201 + seeds)]
        rows.append({"psi": int(psi), "ap": rnd(float(np.mean(aps))),
                     "sd": rnd(float(np.std(aps)))})
        if psi == len(X):
            break
    return rows


# a cluster of anomalies inside a big sample, which is the paper's own picture
mask_rng = np.random.default_rng(11)
Xm = np.vstack([mask_rng.normal([0, 0], 1.0, size=(4000, 2)),
                mask_rng.normal([4.5, 4.5], 0.25, size=(120, 2))])
ym = np.r_[np.zeros(4000), np.ones(120)]
psi_sets = {
    "scattered anomalies, 252 rows": psi_sweep(X2, y2, [8, 16, 32, 64, 128, 256, len(X2)]),
    "a cluster of anomalies, 4,120 rows":
        psi_sweep(Xm, ym, [16, 32, 64, 128, 256, 512, 1024, 2048, len(Xm)], seeds=3),
}
psi_summary = {}
print("\nsubsample size against average precision:")
for key, rows in psi_sets.items():
    best = max(rows, key=lambda r: r["ap"])
    full = rows[-1]
    psi_summary[key] = {"best": best, "full": full,
                        "cost": rnd(best["ap"] - full["ap"]),
                        "reproduces": bool(best["ap"] - full["ap"] > 3 * max(best["sd"], full["sd"],
                                                                             1e-9))}
    print(f"  {key}:")
    for r in rows:
        print(f"    {r['psi']:>5} samples: {r['ap']:.4f} plus or minus {r['sd']:.4f}")
    print(f"    best at {best['psi']}, using all {full['psi']} costs "
          f"{best['ap'] - full['ap']:+.4f}  "
          f"({'the paper reproduces' if psi_summary[key]['reproduces'] else 'it does not'})")
psi_rows = psi_sets["scattered anomalies, 252 rows"]
best_psi = psi_summary["scattered anomalies, 252 rows"]["best"]
full_psi = psi_summary["scattered anomalies, 252 rows"]["full"]
MASK_SPEC = {"n_normal": 4000, "n_anom": 120, "centre": [4.5, 4.5], "spread": 0.25,
             "normal_spread": 1.0, "seed": 11}

# ---- the forest the page will build, and its reference scores
PAGE_TREES, PAGE_PSI, PAGE_SEED = 100, 64, 7
page_forests = {}
for key, X in [("two densities", X2), ("two clusters and two corners", Xk)]:
    f = build_forest(X, PAGE_TREES, PAGE_PSI, PAGE_SEED)
    probe = X[::17]
    page_forests[key] = {
        "trees": PAGE_TREES, "psi": PAGE_PSI, "seed": PAGE_SEED,
        "limit": f["limit"],
        "c_psi": rnd(c_factor(f["psi"]), 8),
        "probe_index": [int(i) for i in range(0, len(X), 17)],
        "probe_score": [rnd(v, 8) for v in forest_score(f, probe)],
        "ap": rnd(average_precision_score(y2 if key == "two densities" else yk,
                                          forest_score(f, X))),
    }
c_table = [{"n": n, "c": rnd(c_factor(n), 8)} for n in [1, 2, 4, 8, 16, 64, 128, 256, 1024]]

# ============================================================ local outlier factor
def knn_distance(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, k]


K_GRID = [3, 5, 8, 10, 15, 20, 30, 45, 60]
lof_rows = []
for k in K_GRID:
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(X2)
    s_lof = -lof.negative_outlier_factor_
    s_knn = knn_distance(X2, k)
    lof_rows.append({
        "k": k,
        "ap_lof": rnd(average_precision_score(y2, s_lof)),
        "ap_knn": rnd(average_precision_score(y2, s_knn)),
        "auc_lof": rnd(roc_auc_score(y2, s_lof)),
        "auc_knn": rnd(roc_auc_score(y2, s_knn)),
    })
best_lof = max(lof_rows, key=lambda r: r["ap_lof"])
best_knn = max(lof_rows, key=lambda r: r["ap_knn"])
print("\nLOF against a plain k-nearest-neighbour distance on the two densities:")
for r in lof_rows:
    print(f"  k = {r['k']:>2}: LOF {r['ap_lof']:.4f}, kNN distance {r['ap_knn']:.4f}")
print(f"  best LOF {best_lof['ap_lof']:.4f} at k = {best_lof['k']}, best kNN "
      f"{best_knn['ap_knn']:.4f} at k = {best_knn['k']}")

# what the two disagree about: the loose cluster's own members
K_SHOW = best_lof["k"]
lof_show = LocalOutlierFactor(n_neighbors=K_SHOW).fit(X2)
s_lof = -lof_show.negative_outlier_factor_
s_knn = knn_distance(X2, K_SHOW)
n_anom = int(y2.sum())
top_lof = set(np.argsort(-s_lof)[:n_anom].tolist())
top_knn = set(np.argsort(-s_knn)[:n_anom].tolist())
truth = set(np.where(y2 == 1)[0].tolist())
loose = set(range(120, 240))
disagree = {
    "k": K_SHOW, "n_anom": n_anom,
    "lof_right": len(top_lof & truth), "knn_right": len(top_knn & truth),
    "lof_loose": len(top_lof & loose), "knn_loose": len(top_knn & loose),
    "lof_scores": [rnd(v, 5) for v in s_lof],
    "knn_scores": [rnd(v, 5) for v in s_knn],
}
print(f"  taking the top {n_anom} by each score at k = {K_SHOW}: LOF gets "
      f"{disagree['lof_right']} of the {n_anom} right, the kNN distance "
      f"{disagree['knn_right']}; of the ones they get wrong, "
      f"{disagree['knn_loose']} of the distance's are members of the loose cluster against "
      f"{disagree['lof_loose']} of LOF's")

# the reference the browser has to reproduce, entry by entry
lof_ref = {str(k): [rnd(v, 6) for v in
                    -LocalOutlierFactor(n_neighbors=k).fit(X2).negative_outlier_factor_]
           for k in K_GRID}

# ============================================================== the one-class SVM
NU_GRID = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
GAMMA_GRID = [0.02, 0.1, 0.5, 2.0]
def rebuild_decision(sv, alpha, rho, gamma, Q):
    """The decision function as the PAGE will compute it: from the exported
    support vectors, the exported coefficients and the exported offset.

    This is not the same array as `m.decision_function(X)`, and the difference
    matters more than it looks. The two agree to about 1.5e-8, which is nothing;
    but average precision is a function of an ORDERING, and an ordering is not
    continuous in the scores. On this dataset a perturbation of 1.5e-8 swaps one
    adjacent pair near the top of the ranking and moves average precision by
    0.0049, which is larger than several of the gaps this article reports. So
    every number the page will show is measured on the reconstructed function.
    """
    K = np.exp(-gamma * ((Q[:, None, :] - X2[sv][None, :, :]) ** 2).sum(-1))
    return K @ np.array(alpha) - rho


svm_models = {}
nu_rows = []
ap_drift = []
for nu in NU_GRID:
    for gamma in GAMMA_GRID:
        m = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf", tol=1e-8).fit(X2)
        f_fit = m.decision_function(X2)
        sv_i = [int(i) for i in m.support_]
        al = [rnd(float(v), 8) for v in m.dual_coef_[0]]
        rho_v = rnd(float(-m.intercept_[0]), 8)
        f = rebuild_decision(sv_i, al, rho_v, gamma, X2)
        ap_drift.append({"nu": nu, "gamma": gamma,
                         "worst_score_gap": float(np.abs(f - f_fit).max()),
                         "ap_fit": rnd(average_precision_score(y2, -f_fit), 6),
                         "ap_export": rnd(average_precision_score(y2, -f), 6)})
        # Indexed by position in the two grids, never by a stringified float.
        # Python writes str(2.0) as "2.0" and JavaScript writes String(2.0) as
        # "2", so a key built from the numbers on one side cannot be looked up
        # from the other: the widget silently found no model at gamma 2 and drew
        # nothing, which is the same family of bug as the "1e-06" against
        # "0.000001" this repo has paid for once already.
        key = f"{NU_GRID.index(nu)}|{GAMMA_GRID.index(gamma)}"
        svm_models[key] = {
            "nu": nu, "gamma": gamma,
            "sv": sv_i, "alpha": al, "rho": rho_v,
            "ap": rnd(average_precision_score(y2, -f)),
            "outside": int((f < 0).sum()),
            "n_sv": int(len(m.support_)),
            "probe": [rnd(float(v), 6) for v in f[::11]],
        }
        nu_rows.append({"nu": nu, "gamma": gamma,
                        "frac_out": rnd((f < 0).mean(), 5),
                        "frac_sv": rnd(len(m.support_) / len(X2), 5),
                        "ap": svm_models[key]["ap"]})
# The nu property is two inequalities and they do not fare the same. The lower
# bound (nu is at most the fraction of support vectors) is a statement about the
# dual's box constraint and holds exactly. The upper bound (nu is at least the
# fraction of points left outside) is a statement about the exact optimum, and
# a peaked kernel plus a stopping tolerance is enough to break it: the two are
# therefore counted separately rather than as one claim.
lower_ok = [r for r in nu_rows if r["nu"] <= r["frac_sv"] + 1e-9]
upper_ok = [r for r in nu_rows if r["frac_out"] <= r["nu"] + 1e-9]
holds = [r for r in nu_rows if r in lower_ok and r in upper_ok]
worst_upper = max(nu_rows, key=lambda r: r["frac_out"] / r["nu"])
by_gamma = []
for gamma in GAMMA_GRID:
    rows_g = [r for r in nu_rows if r["gamma"] == gamma]
    by_gamma.append({"gamma": gamma,
                     "upper_ok": sum(1 for r in rows_g if r["frac_out"] <= r["nu"] + 1e-9),
                     "lower_ok": sum(1 for r in rows_g if r["nu"] <= r["frac_sv"] + 1e-9),
                     "n": len(rows_g),
                     "worst": rnd(max(r["frac_out"] / r["nu"] for r in rows_g), 3)})
print(f"\nthe one-class SVM's nu property, over {len(nu_rows)} settings:")
print(f"  nu <= fraction of support vectors:      holds in {len(lower_ok)} of {len(nu_rows)}")
print(f"  fraction left outside <= nu:            holds in {len(upper_ok)} of {len(nu_rows)}")
print(f"  the worst breach is nu {worst_upper['nu']} gamma {worst_upper['gamma']}: "
      f"{worst_upper['frac_out']:.4f} outside where nu promises at most {worst_upper['nu']}, "
      f"a factor of {worst_upper['frac_out'] / worst_upper['nu']:.1f}")
print("  and the breach is entirely a function of gamma:")
for v in by_gamma:
    print(f"    gamma {v['gamma']:<5}: the upper bound holds in {v['upper_ok']} of {v['n']}, "
          f"worst overshoot {v['worst']:.2f} times nu")
for r in nu_rows[::4]:
    print(f"  nu {r['nu']:<5} gamma {r['gamma']:<5}: outside {r['frac_out']:.4f}, "
          f"support vectors {r['frac_sv']:.4f}")
svm_best = max(nu_rows, key=lambda r: r["ap"])
svm_worst = min(nu_rows, key=lambda r: r["ap"])
print(f"  average precision runs from {svm_worst['ap']:.4f} (nu {svm_worst['nu']}, gamma "
      f"{svm_worst['gamma']}) to {svm_best['ap']:.4f} (nu {svm_best['nu']}, gamma "
      f"{svm_best['gamma']})")

# a raster of the decision function for the widget's default, so the page has
# something to check its own kernel arithmetic against
svm_probe_grid = np.c_[np.linspace(-4, 10, 15).repeat(15),
                       np.tile(np.linspace(-4, 10, 15), 15)]
svm_raster = {}
for key, m in list(svm_models.items()):
    svm_raster[key] = [rnd(float(v), 6) for v in
                       rebuild_decision(m["sv"], m["alpha"], m["rho"], m["gamma"],
                                        svm_probe_grid)]
worst_drift = max(ap_drift, key=lambda r: abs(r["ap_fit"] - r["ap_export"]))
print(f"\n  scoring the exported model instead of the fitted one:")
print(f"    the two decision functions differ by at most "
      f"{max(r['worst_score_gap'] for r in ap_drift):.2e}")
print(f"    and their average precisions by up to "
      f"{abs(worst_drift['ap_fit'] - worst_drift['ap_export']):.4f} "
      f"(nu {worst_drift['nu']}, gamma {worst_drift['gamma']}), because average precision is a "
      f"function of an ordering")
assert all("|" in k and k.split("|")[0].isdigit() for k in svm_models), \
    "the model keys have to be index pairs, or the browser cannot look them up"

# ============================================================================ MCD
def cstep(X, support):
    """One concentration step: fit on the support, score everybody, keep the h
    smallest. Written here and in the page, and the page's version has to reach
    the same fixed point from the same start."""
    h = len(support)
    mu = X[support].mean(0)
    S = np.cov(X[support].T, ddof=1)
    d = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(S), X - mu)
    return np.sort(np.argsort(d)[:h]), mu, S, d


mcd_sets = {}
for name, X, y, _ in SETS:
    n, p = X.shape
    h = (n + p + 1) // 2
    ref = MinCovDet(support_fraction=h / n, random_state=SEED).fit(X)
    d_mcd = ref.mahalanobis(X)
    mu_full = X.mean(0)
    S_full = np.cov(X.T, ddof=1)
    d_full = np.einsum("ij,jk,ik->i", X - mu_full, np.linalg.inv(S_full), X - mu_full)
    # Is scikit-learn's answer a fixed point of the concentration step? That is
    # a property of the solution and the browser can check it without having to
    # reproduce a random restart.
    #
    # It has to be `raw_support_`, not `support_`. The second one is the set
    # after the reweighting step that MinCovDet runs on top of the MCD, which is
    # a different estimator and is not a fixed point of the plain step: checking
    # the wrong one reported a failure on the planted batch that was not there.
    sup0 = np.sort(np.where(ref.raw_support_)[0])
    sup1, _, _, _ = cstep(X, sup0)
    mcd_sets[name] = {
        "h": h, "n": n, "p": p,
        "fixed_point": bool(np.array_equal(sup0, sup1)),
        "ap_mcd": rnd(average_precision_score(y, d_mcd)),
        "ap_full": rnd(average_precision_score(y, d_full)),
        "auc_mcd": rnd(roc_auc_score(y, d_mcd)),
        "auc_full": rnd(roc_auc_score(y, d_full)),
        "support": [int(i) for i in sup0] if p <= 2 else None,
    }
    print(f"\nMCD on the {name}: h = {h} of {n}; average precision {mcd_sets[name]['ap_mcd']:.4f} "
          f"against {mcd_sets[name]['ap_full']:.4f} for the full-sample ellipse"
          + ("; scikit-learn's support is a fixed point of the concentration step"
             if mcd_sets[name]["fixed_point"] else "; and its support is NOT a fixed point"))

# the concentration steps the widget animates, on the two-density set
mcd_steps = []
n2 = len(X2)
h2 = (n2 + 2 + 1) // 2
sup = np.arange(n2)                      # start from everything, which is the
mu = X2.mean(0)                          # previous article's estimator exactly
S = np.cov(X2.T, ddof=1)
d = np.einsum("ij,jk,ik->i", X2 - mu, np.linalg.inv(S), X2 - mu)
sup = np.sort(np.argsort(d)[:h2])
for step in range(9):
    mu = X2[sup].mean(0)
    S = np.cov(X2[sup].T, ddof=1)
    d = np.einsum("ij,jk,ik->i", X2 - mu, np.linalg.inv(S), X2 - mu)
    nxt = np.sort(np.argsort(d)[:h2])
    mcd_steps.append({
        "step": step,
        "mu": [rnd(v) for v in mu],
        "cov": [[rnd(v, 6) for v in row] for row in S],
        "det": rnd(float(np.linalg.det(S)), 8),
        "ap": rnd(average_precision_score(y2, d)),
        "caught": int(sum(1 for i in np.argsort(-d)[:int(y2.sum())] if y2[i] == 1)),
        "changed": int(len(set(nxt.tolist()) ^ set(sup.tolist())) // 2),
        "support": [int(i) for i in sup],
    })
    if np.array_equal(nxt, sup):
        break
    sup = nxt
print(f"\nthe concentration step on the two densities, h = {h2} of {n2}:")
for r in mcd_steps:
    print(f"  step {r['step']}: determinant {r['det']:.6g}, average precision {r['ap']:.4f}, "
          f"{r['changed']} rows swapped")
mcd_converged = len(mcd_steps)

# ================================================================ the scoreboard
def all_scores(X, y):
    out = {}
    f = build_forest(X, 200, min(PSI, len(X)), SEED)
    out["forest"] = forest_score(f, X)
    k = min(20, len(X) - 1)
    out["lof"] = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    out["knn"] = knn_distance(X, k)
    # a scale-free gamma so one dataset's units do not decide the comparison
    med = np.median(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)) if len(X) < 700 else None
    gamma = 1.0 / med if med else "scale"
    out["ocsvm"] = -OneClassSVM(nu=0.1, gamma=gamma).fit(X).decision_function(X)
    n, p = X.shape
    hh = (n + p + 1) // 2
    out["mcd"] = MinCovDet(support_fraction=hh / n, random_state=SEED).fit(X).mahalanobis(X)
    mu = X.mean(0)
    S = np.cov(X.T, ddof=1)
    out["mahalanobis"] = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(S), X - mu)
    return out


DETECTORS = ["forest", "lof", "knn", "ocsvm", "mcd", "mahalanobis"]
board = []
for name, X, y, _ in SETS:
    s = all_scores(X, y)
    row = {"name": name, "n": len(X), "p": X.shape[1], "anom": int(y.sum())}
    for d_name in DETECTORS:
        row[d_name] = rnd(average_precision_score(y, s[d_name]))
        row[f"{d_name}_auc"] = rnd(roc_auc_score(y, s[d_name]))
    row["winner"] = max(DETECTORS, key=lambda d: row[d])
    row["loser"] = min(DETECTORS, key=lambda d: row[d])
    row["baseline"] = rnd(float(y.mean()))
    board.append(row)
print("\nthe scoreboard, average precision (a coin would score the anomaly rate):")
print(f"  {'dataset':<30} " + " ".join(f"{d:>12}" for d in DETECTORS))
for r in board:
    print(f"  {r['name']:<30} " + " ".join(f"{r[d]:>12.4f}" for d in DETECTORS)
          + f"   winner: {r['winner']}")
winners = sorted({r["winner"] for r in board})
never_wins = [d for d in DETECTORS if d not in winners]
print(f"  {len(winners)} different detectors win the {len(board)} datasets: {', '.join(winners)}")

# ---- the threshold is a separate decision from the score
CONT = [0.01, 0.02, 0.05, 0.1, 0.2]
cont_rows = []
s_forest = all_scores(X2, y2)["forest"]
for c in CONT:
    thr = np.quantile(s_forest, 1 - c)
    flagged = s_forest > thr
    tp = int((flagged & (y2 == 1)).sum())
    cont_rows.append({"contamination": c, "flagged": int(flagged.sum()), "caught": tp,
                      "missed": int(y2.sum()) - tp,
                      "precision": rnd(tp / max(1, flagged.sum())),
                      "recall": rnd(tp / y2.sum())})
ap_fixed = rnd(average_precision_score(y2, s_forest))
print(f"\nthe same forest scores, five thresholds; average precision is "
      f"{ap_fixed:.4f} at every one of them:")
for r in cont_rows:
    print(f"  contamination {r['contamination']:<5}: flags {r['flagged']:>3}, catches "
          f"{r['caught']:>2} of {int(y2.sum())}, precision {r['precision']:.3f}")

# ==================================================================== the cost
cost_rows = []
for n in [250, 1000, 10000, 100000]:
    psi = min(256, n)
    cost_rows.append({
        "n": n,
        "forest": 200 * psi * int(math.ceil(math.log2(psi))),   # build
        "forest_score": 200 * n * int(math.ceil(math.log2(psi))),
        "lof": n * n,                                            # the distance table
        "ocsvm": n * n,
        "mcd": n * 13 * 13 + 13 ** 3 // 3,
    })
print("\ncost, counted (no clock anywhere):")
for r in cost_rows:
    print(f"  n = {r['n']:>7}: forest builds in {r['forest']:>12,} and scores in "
          f"{r['forest_score']:>14,}; a full distance table is {r['lof']:>14,}")

# ===================================================================== the file
out = {
    "sets": {
        "two densities": {"points": [[rnd(a), rnd(b)] for a, b in X2],
                          "labels": [int(v) for v in y2],
                          "tight": 120, "loose": 240},
        "two clusters and two corners": {"points": [[rnd(a), rnd(b)] for a, b in Xk],
                                         "labels": [int(v) for v in yk]},
    },
    "forest": {
        "check": forest_check, "worst": worst_forest["name"], "seeds": seed_spread,
        "n_trees": N_TREES, "psi": PSI,
        "c_table": c_table,
        "page": page_forests,
        "probes": {"points": probes, "scores": probe_scores,
                   "corner_gap": rnd(corner_gap, 5), "outside_gap": rnd(outside_gap, 5),
                   "ratio": rnd(outside_gap / corner_gap, 3) if corner_gap else None,
                   "ap": corner_ap},
        "psi_sweep": {"rows": psi_rows, "best": best_psi, "full": full_psi,
                      "sets": psi_sets, "summary": psi_summary, "mask_spec": MASK_SPEC},
        "radial": {"rows": radial, "angles": ANGLES, "radii": RADII,
                   "axis_flat": rnd(axis_flat, 5), "diag_climb": rnd(diag_climb, 5),
                   "worst": worst_gap, "trees": 300, "psi": 256, "n": len(BLOB)},
        "grid": {"n": grid_n, "lo": -3.5, "hi": 13.5,
                 "scores": [rnd(v, 5) for v in grid_scores],
                 "trees": 200, "psi": PSI, "seed": SEED},
    },
    "lof": {"rows": lof_rows, "k_grid": K_GRID, "best_lof": best_lof, "best_knn": best_knn,
            "disagree": disagree, "ref": lof_ref},
    "svm": {"nu_grid": NU_GRID, "gamma_grid": GAMMA_GRID, "models": svm_models,
            "rows": nu_rows, "holds": len(holds), "total": len(nu_rows),
            "lower_ok": len(lower_ok), "upper_ok": len(upper_ok),
            "worst_upper": worst_upper, "by_gamma": by_gamma,
            "best": svm_best, "worst": svm_worst,
            "raster": {"lo": -4, "hi": 10, "n": 15, "values": svm_raster},
            "ap_drift": {"rows": ap_drift, "worst": worst_drift,
                         "worst_score_gap": max(r["worst_score_gap"] for r in ap_drift)}},
    "mcd": {"sets": mcd_sets, "steps": mcd_steps, "converged": mcd_converged, "h": h2,
            "planted": planted_ok},
    "board": {"rows": board, "detectors": DETECTORS, "winners": winners,
              "never_wins": never_wins},
    "threshold": {"rows": cont_rows, "ap": ap_fixed, "anomalies": int(y2.sum())},
    "cost": {"rows": cost_rows},
    "meta": {"seed": SEED, "decimals": DP, "sklearn": sklearn.__version__,
             "scipy": scipy.__version__, "numpy": np.__version__,
             "lcg": "s <- (1664525 s + 1013904223) mod 2^32"},
}
path = OUT / "isolation.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
