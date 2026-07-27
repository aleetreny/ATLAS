"""Data for the ATLAS 'Assumptions' web article, the fifth of module 2.1.

The four articles before this one each answered "what is a cluster" and each
answer arrived with fine print: k-means needs a mean (so numbers, and squared
euclidean distance), DBSCAN needs one radius for the whole dataset, and every
one of them needs somebody to say k. This article drops one clause at a time,
k-medoids, k-modes, OPTICS and mean shift, and prices each removal on the very
same points the earlier articles used: the blobs of seed 19, the failure
gallery, the varying-density trap, and the 178 wines.

Everything the page draws it also computes: PAM, k-modes, OPTICS and mean shift
all run in the browser. This file ships the seeded datasets, the reference
answers the page checks itself against at load time (scikit-learn for OPTICS
and mean shift, the Rust FasterPAM and the kmodes package for the other two),
and the measurements that need a library or a long sweep.

Three results here are worth knowing before reading the code:

  * the break-even distance of an outlier is arithmetic, not folklore. Spending
    a centre on a lone outlier costs whatever the remaining points pay for
    having one centre fewer; keeping it costs d^2 for k-means and d for PAM.
    Both predictions are computed and then checked against a sweep.
  * k-modes is NOT the most accurate way to cluster categories here. K-means on
    a one-hot encoding beats it, and both beat k-medoids. What k-modes owns is
    that its answer cannot depend on how the categories were numbered, and the
    sweep of 40 relabellings measures exactly what that is worth.
  * OPTICS with its default xi is worse than a well-chosen DBSCAN radius on
    every dataset here except the one that has no good radius at all, which is
    the one it was invented for.

Run from anywhere: `python src/utils/generate_assumptions_data.py`.
"""
import itertools
import json
import time
from pathlib import Path

import kmedoids
import numpy as np
from kmodes.kmodes import KModes
from scipy.spatial.distance import cdist
from sklearn.cluster import (DBSCAN, HDBSCAN, KMeans, MeanShift, OPTICS,
                             cluster_optics_dbscan, estimate_bandwidth)
from sklearn.datasets import load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assumptions" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
ari = lambda a, b: float(adjusted_rand_score(a, b))
SEED = 19
t_start = time.perf_counter()

# Two of the sweeps below are most of the runtime, and rerunning the whole file
# to move a decimal in a printed line is a waste of a coffee. Finished sweeps
# are cached outside the repo under a key that names their entire
# configuration, so changing any of it invalidates the cache by itself.
CACHE = Path.home() / ".atlas_vision_data" / "assumptions_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def cached(key, fn):
    path = CACHE / f"{key}.json"
    if path.exists():
        print(f"  (cached: {key})")
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


# =============================================================== the algorithms
# The three the browser will run itself, written here first so that the numbers
# on the page and the numbers in this file come out of the same definition.
# Deterministic, all of them: no seed, no restarts, nothing to disagree about.

def pam(D, k, max_iter=100):
    """Classic PAM: BUILD a set of k medoids greedily, then SWAP the single
    (medoid, non-medoid) pair that lowers the total distance most, until no
    swap helps. Returns (medoids, labels, loss)."""
    n = len(D)
    med = [int(np.argmin(D.sum(axis=1)))]              # BUILD: the most central point
    while len(med) < k:
        d_near = D[med].min(axis=0)
        gains = np.maximum(d_near[None, :] - D, 0).sum(axis=1)
        gains[med] = -1
        med.append(int(np.argmax(gains)))
    med = sorted(med)
    for _ in range(max_iter):
        cost = D[med].min(axis=0).sum()
        best = None
        for i in range(k):
            others = [m for j, m in enumerate(med) if j != i]
            d_other = D[others].min(axis=0)             # cost if this medoid leaves
            cand = np.minimum(d_other[None, :], D).sum(axis=1)
            cand[med] = np.inf
            h = int(np.argmin(cand))
            if best is None or cand[h] < best[0]:
                best = (float(cand[h]), i, h)
        if best is None or best[0] >= cost - 1e-12:
            break
        med[best[1]] = best[2]
        med = sorted(med)
    labels = np.argmin(D[med], axis=0)
    return np.array(med), labels, float(D[med].min(axis=0).sum())


def cao_init(A, k):
    """Cao et al. (2009): the densest row first, then the row maximising
    density times distance to the nearest mode already chosen. Deterministic,
    and it depends on the categories only through how often they occur, never
    through the numbers they were given."""
    n, m = A.shape
    freq = []
    for j in range(m):
        v, c = np.unique(A[:, j], return_counts=True)
        freq.append(dict(zip(v.tolist(), (c / n).tolist())))
    dens = np.array([sum(freq[j][int(A[i, j])] for j in range(m)) / m for i in range(n)])
    modes = [A[int(np.argmax(dens))].copy()]
    while len(modes) < k:
        score = np.array([min(int((A[i] != mo).sum()) for mo in modes) * dens[i] for i in range(n)])
        modes.append(A[int(np.argmax(score))].copy())
    return np.array(modes)


def kmodes(A, k, max_iter=100):
    """K-modes: Hamming distance, and a centre that is the most common category
    in each column. Ties go to whichever value appears first inside the
    cluster, so that renumbering the categories cannot change the answer."""
    modes = cao_init(A, k)
    labels = np.full(len(A), -1)
    for _ in range(max_iter):
        d = np.stack([(A != modes[c]).sum(axis=1) for c in range(k)], axis=1)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for c in range(k):
            rows = A[labels == c]
            if not len(rows):
                continue
            for j in range(A.shape[1]):
                vals, counts = np.unique(rows[:, j], return_counts=True)
                tied = set(vals[counts == counts.max()].tolist())
                modes[c, j] = next(v for v in rows[:, j].tolist() if v in tied)
    cost = int(sum(int((A[i] != modes[labels[i]]).sum()) for i in range(len(A))))
    return modes, labels, cost


def optics(X, min_samples):
    """OPTICS exactly as scikit-learn orders it: repeatedly take the unprocessed
    point with the smallest reachability (ties to the lowest index, which is
    what np.argmin does), then relax its neighbours' reachability against its
    core distance. Returns (ordering, reachability, core_distances)."""
    n = len(X)
    D = cdist(X, X)
    core = np.sort(D, axis=1)[:, min_samples - 1]
    reach = np.full(n, np.inf)
    processed = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)
    for t in range(n):
        idx = np.where(~processed)[0]
        p = int(idx[np.argmin(reach[idx])])
        processed[p] = True
        order[t] = p
        un = np.where(~processed)[0]
        if len(un):
            rd = np.maximum(D[p][un], core[p])
            upd = rd < reach[un]
            reach[un[upd]] = rd[upd]
    return order, reach, core


def mean_shift(X, bw, max_iter=300):
    """Mean shift with a flat kernel, following scikit-learn's implementation
    closely enough that the two agree point for point: every point is a seed,
    each seed walks to the mean of the window around it until it stops moving,
    identical landing places collapse, the survivors are taken strongest first
    (ties by coordinate, descending, which is what sorting tuples does), and
    every point joins the nearest survivor.

    The intensity that ranks a peak is the size of the LAST window measured,
    which is the one around the previous position, not the final one. That
    detail decides ties, and ties decide how many modes come out."""
    stop = 1e-3 * bw
    peaks = {}
    for i in range(len(X)):
        p = X[i].copy()
        it = 0
        while True:
            inside = X[np.linalg.norm(X - p, axis=1) <= bw]
            if not len(inside):
                break
            old = p
            p = inside.mean(axis=0)
            if np.linalg.norm(p - old) <= stop or it == max_iter:
                break
            it += 1
        if len(inside):
            peaks[tuple(p)] = len(inside)
    ranked = sorted(peaks.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    centers = []
    for p, _ in ranked:
        p = np.array(p)
        if all(np.linalg.norm(p - q) > bw for q in centers):
            centers.append(p)
    centers = np.array(centers)
    return centers, np.argmin(cdist(X, centers), axis=1)


# ================================================================ the datasets
# Every one of these is rebuilt with the construction of the article that first
# used it, and then checked against the JSON that article shipped. If a single
# coordinate had drifted, every comparison on this page would be meaningless.
Xb, yb = make_blobs(n_samples=300, centers=[[-3.2, 1.8], [1.8, 3.0], [0.6, -2.6]],
                    cluster_std=[0.85, 1.05, 0.9], random_state=SEED)
Xb = np.round(Xb, 3)

rs = np.random.RandomState(SEED)
t1 = rs.uniform(0, 2 * np.pi, 150)
t2 = rs.uniform(0, 2 * np.pi, 150)
Xr = np.round(np.vstack([
    np.column_stack([1.1 * np.cos(t1), 1.1 * np.sin(t1)]) + rs.normal(0, 0.11, (150, 2)),
    np.column_stack([3.1 * np.cos(t2), 3.1 * np.sin(t2)]) + rs.normal(0, 0.11, (150, 2)),
]), 3)
yr = np.array([0] * 150 + [1] * 150)
Xu = np.round(np.vstack([
    rs.normal([-2.4, 0], [0.4, 0.4], (60, 2)),
    rs.normal([1.6, 0.4], [1.9, 1.9], (240, 2)),
]), 3)
yu = np.array([0] * 60 + [1] * 240)

Xa, ya = make_blobs(n_samples=300, random_state=170)
Xa_raw = Xa @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])
Xa = np.round(Xa_raw, 3)

rs2 = np.random.RandomState(SEED)
Xv = np.round(np.vstack([
    rs2.normal([-0.8, 0.5], 0.14, (100, 2)),
    rs2.normal([0.01, -0.09], 0.14, (100, 2)),
    rs2.normal([4.0, 0.2], 1.15, (150, 2)),
]), 3)
yv = np.array([0] * 100 + [1] * 100 + [2] * 150)

published = {}
prev = json.loads((ROOT / "kmeans" / "data" / "clusters.json").read_text(encoding="utf-8"))
dens = json.loads((ROOT / "dbscan" / "data" / "density.json").read_text(encoding="utf-8"))
gal = {g["name"]: g for g in prev["gallery"]}
checks = {
    "blobs": np.array_equal(Xb, np.array(prev["blobs"]["points"])),
    "rings": np.array_equal(Xr, np.array(gal["rings"]["points"])),
    "unequal": np.array_equal(Xu, np.array(gal["unequal"]["points"])),
    "anisotropic": np.array_equal(Xa, np.array(gal["anisotropic"]["points"])),
    "trap": np.array_equal(Xv, np.array(dens["datasets"]["varying"]["points"])),
}
print("same points as the earlier articles shipped:",
      ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in checks.items()))
assert all(checks.values()), "a shared dataset drifted; every comparison below would be a lie"

wine = load_wine()
Xw = np.round(StandardScaler().fit_transform(wine.data), 4)
yw = wine.target
proj = PCA(n_components=2).fit_transform(Xw)


def trap_metrics(labels, y=yv):
    """The two measures the DBSCAN article used on this dataset, kept because
    the ARI here rewards deleting the diffuse cluster outright: calling all 150
    of its points noise scores a perfect 1.000."""
    labels = np.asarray(labels)

    def major(cls):
        ids = labels[(y == cls) & (labels >= 0)]
        return int(np.bincount(ids).argmax()) if len(ids) else -9 - cls
    sep = 1 if major(0) != major(1) else 0
    idsC = labels[(y == 2) & (labels >= 0)]
    rec = float(np.bincount(idsC).max() / (y == 2).sum()) if len(idsC) else 0.0
    return sep, rnd(rec, 3)


# ======================================================= 1. the centre as a row
print("\n--- k-medoids ---")
D_blobs = cdist(Xb, Xb)
med, lab_pam, loss_pam = pam(D_blobs, 3)
ref_fast = kmedoids.fasterpam(D_blobs, 3, random_state=SEED)
km_blobs = KMeans(3, n_init=10, random_state=SEED).fit(Xb)
print(f"blobs: PAM loss {loss_pam:.4f} at medoids {med.tolist()}; "
      f"FasterPAM {ref_fast.loss:.4f}; same partition {ari(lab_pam, ref_fast.labels):.4f}")
print(f"  ARI against the truth: k-means {ari(yb, km_blobs.labels_):.3f}, PAM {ari(yb, lab_pam):.3f}")

# Is the swap heuristic finding the best set of medoids or just a good one? On
# a subsample every possible set can be tried, so the question has an answer.
sub = Xb[::8]
Ds = cdist(sub, sub)
brute = min((float(Ds[list(c)].min(axis=0).sum()), c) for c in itertools.combinations(range(len(sub)), 3))
_, _, loss_sub = pam(Ds, 3)
print(f"  brute force over the {len(list(itertools.combinations(range(len(sub)), 3))):,} possible triples of a "
      f"{len(sub)}-point subsample: {brute[0]:.4f}; PAM found {loss_sub:.4f}")

medoids_out = {
    "medoids": [int(i) for i in med],
    "loss": rnd(loss_pam, 3),
    "ari": rnd(ari(yb, lab_pam), 3),
    "km_ari": rnd(ari(yb, km_blobs.labels_), 3),
    "km_centers": [[rnd(c[0], 3), rnd(c[1], 3)] for c in km_blobs.cluster_centers_],
    "fasterpam_loss": rnd(ref_fast.loss, 3),
    "fasterpam_agrees": rnd(ari(lab_pam, ref_fast.labels), 4),
    "brute_force": {"n": int(len(sub)), "triples": int(len(list(itertools.combinations(range(len(sub)), 3)))),
                    "optimum": rnd(brute[0], 4), "pam": rnd(loss_sub, 4)},
}

# ------------------------------------------------- what one outlier is worth
# Spending a centre on a single distant point costs whatever the other 300 pay
# for having one centre fewer. Keeping it costs d^2 (k-means) or d (PAM). Both
# break-evens are arithmetic; the sweep below checks them.
J3 = float(km_blobs.inertia_)
J2 = float(KMeans(2, n_init=10, random_state=SEED).fit(Xb).inertia_)
L3 = loss_pam
_, _, L2 = pam(D_blobs, 2)
sizes = np.bincount(km_blobs.labels_)
HOME = int(np.argmax(km_blobs.cluster_centers_[:, 1]))    # the top-right blob
ORIGIN = km_blobs.cluster_centers_[HOME]
DIR = np.array([0.6, 0.8])
m_home = int(sizes[HOME])
pred_km = float(np.sqrt((J2 - J3) * (m_home + 1) / m_home))
pred_pam = float(L2 - L3)
# There is a second, earlier break, and it belongs to the search rather than to
# the objective. PAM's greedy BUILD hands the outlier a medoid as soon as that
# buys more than a third blob medoid would; escaping means giving the outlier
# back, which costs its full distance at once, and a single swap can only offer
# the best third medoid to the PAIR it is already stuck with.
pair, _, _ = pam(D_blobs, 2)
L3_from_pair = float(min(D_blobs[list(pair) + [h]].min(axis=0).sum()
                         for h in range(len(Xb)) if h not in pair))
pred_pam_build = L2 - L3_from_pair
# And the mirror image of it: a search that starts from three good medoids has
# to hand one of them to the outlier to isolate it, which leaves the other two,
# not the best two. That is where FasterPAM's random restarts give way.
L2_restricted = float(min(D_blobs[[m for m in med if m != drop]].min(axis=0).sum() for drop in med))
pred_pam_good = L2_restricted - L3
print(f"  one centre fewer costs the 300 points {J2 - J3:.2f} of inertia and {L2 - L3:.2f} of distance")
print(f"  predicted break: k-means at d = {pred_km:.2f}, the medoid objective at d = {pred_pam:.2f}, "
      f"a search holding the outlier from the start at d = {pred_pam_build:.2f}, "
      f"one that has to give a good medoid away at d = {pred_pam_good:.2f}")

# The sweep runs on exactly the grid the slider will offer, so the reference the
# widget quotes is one the reader can actually land on.
GRID = [round(v, 2) for v in list(np.arange(0.5, 60.01, 0.5)) + list(np.arange(62.5, 600.01, 2.5))]
# Parallel arrays rather than 336 little objects: the same numbers at a fifth
# of the bytes, and the widget indexes them by slider position anyway.
def run_outlier_sweep():
    out = {"km": [], "pam": [], "fast": [], "km_alone": [], "pam_alone": [], "fast_alone": [],
           "disagreements": 0}
    for d in GRID:
        P = np.vstack([Xb, ORIGIN + DIR * d])
        km = KMeans(3, n_init=10, random_state=SEED).fit(P)
        Dp = cdist(P, P)
        mp, lp, lossp = pam(Dp, 3)
        fp = kmedoids.fasterpam(Dp, 3, random_state=SEED)
        if abs(fp.loss - lossp) > 1e-6:
            out["disagreements"] += 1
        fl = np.asarray(fp.labels)
        out["km"].append(rnd(ari(yb, km.labels_[:300]), 3))
        out["pam"].append(rnd(ari(yb, lp[:300]), 3))
        out["fast"].append(rnd(ari(yb, fl[:300]), 3))
        out["km_alone"].append(int((km.labels_ == km.labels_[300]).sum() == 1))
        out["pam_alone"].append(int((lp == lp[300]).sum() == 1))
        out["fast_alone"].append(int((fl == fl[300]).sum() == 1))
    return out


sweep = cached(f"outlier-s{SEED}-n{len(GRID)}-to{GRID[-1]}-home{HOME}", run_outlier_sweep)
disagreements = sweep.pop("disagreements")
first = lambda key: next((GRID[i] for i, v in enumerate(sweep[key]) if v), None)
brk_km, brk_pam, brk_fast = first("km_alone"), first("pam_alone"), first("fast_alone")
print(f"  measured break: k-means at d = {brk_km}, classic PAM at d = {brk_pam}, "
      f"FasterPAM at d = {brk_fast} ({disagreements} of {len(GRID)} sweep points where "
      f"FasterPAM found a cheaper set than the BUILD-and-SWAP written here)")

outlier_out = {
    "origin": [rnd(ORIGIN[0], 3), rnd(ORIGIN[1], 3)],
    "dir": [DIR[0], DIR[1]],
    "home_size": m_home,
    "grid": GRID,
    # the per-position "is it alone" flags did their job above, producing the
    # three break distances; shipping 336 of each would be shipping the working
    "sweep": {k: v for k, v in sweep.items() if not k.endswith("_alone")},
    "J3": rnd(J3, 2), "J2": rnd(J2, 2), "L3": rnd(L3, 2), "L2": rnd(L2, 2),
    "L3_from_pair": rnd(L3_from_pair, 2),
    "L2_restricted": rnd(L2_restricted, 2),
    "pred_km": rnd(pred_km, 2),
    "pred_pam": rnd(pred_pam, 2),
    "pred_pam_build": rnd(pred_pam_build, 2),
    "pred_pam_good": rnd(pred_pam_good, 2),
    "break_km": brk_km,
    "break_pam": brk_pam,
    "break_fast": brk_fast,
    "fasterpam_disagreements": disagreements,
}

# --------------------------------------------- the wine, under three metrics
print("\nwine, 178 bottles, 13 standardised dimensions:")
km_wine = KMeans(3, n_init=10, random_state=SEED).fit(Xw)
print(f"  k-means (article 12's run, reproduced): ARI {ari(yw, km_wine.labels_):.3f}")
wine_metrics = []
for key, metric, label in [("euclidean", "euclidean", "euclidean"),
                           ("manhattan", "cityblock", "manhattan"),
                           ("cosine", "cosine", "cosine")]:
    Dw = cdist(Xw, Xw, metric=metric)
    mw, lw, lossw = pam(Dw, 3)
    fw = kmedoids.fasterpam(Dw, 3, random_state=SEED)
    wine_metrics.append({
        "key": key, "label": label,
        "medoids": [int(i) for i in mw],
        "medoid_classes": [int(yw[i]) for i in mw],
        "labels": [int(v) for v in lw],
        "ari": rnd(ari(yw, lw), 3),
        "loss": rnd(lossw, 3),
        "fasterpam_loss": rnd(fw.loss, 3),
    })
    r = wine_metrics[-1]
    print(f"  PAM {label:<10} ARI {r['ari']:.3f}  exemplar bottles {r['medoids']} "
          f"(cultivars {r['medoid_classes']})  loss {r['loss']} vs FasterPAM {r['fasterpam_loss']}")

wine_out = {
    "features": [[rnd(v) for v in row] for row in Xw],
    "proj": [[rnd(a, 3), rnd(b, 3)] for a, b in proj],
    "true": [int(v) for v in yw],
    "km_ari": rnd(ari(yw, km_wine.labels_), 3),
    "metrics": wine_metrics,
    "gmm_ari": 0.915,      # article 13, same bottles, quoted for the table
    "ward_ari": 0.790,     # article 15, same bottles
}

# ================================================= 2. no mean to be had at all
print("\n--- k-modes ---")
# The same 178 bottles with every measurement replaced by which third of the
# range it falls in: low, middle, high. Thirteen categorical columns, and the
# arithmetic that k-means needs is gone.
codes = np.zeros_like(wine.data, dtype=int)
for j in range(wine.data.shape[1]):
    qs = np.quantile(wine.data[:, j], [1 / 3, 2 / 3])
    codes[:, j] = np.digitize(wine.data[:, j], qs)

modes, lab_modes, cost_modes = kmodes(codes, 3)
pkg = KModes(n_clusters=3, init="Cao", n_init=1, random_state=SEED).fit(codes)
print(f"k-modes: cost {cost_modes}, ARI {ari(yw, lab_modes):.3f}  "
      f"(kmodes package, same init: cost {int(pkg.cost_)}, ARI {ari(yw, pkg.labels_):.3f})")

onehot = np.zeros((len(codes), codes.shape[1] * 3))
for j in range(codes.shape[1]):
    onehot[np.arange(len(codes)), j * 3 + codes[:, j]] = 1
# Thirty restarts rather than ten everywhere the browser reruns the same fit:
# on 13 integer columns with masses of ties, ten k-means++ starts do not always
# reach the same basin, and a page whose live answer sits outside the range the
# text quotes is a page arguing with itself.
N_INIT_CAT = 30
km_codes = KMeans(3, n_init=N_INIT_CAT, random_state=SEED).fit(codes.astype(float))
km_oh = KMeans(3, n_init=N_INIT_CAT, random_state=SEED).fit(onehot)
D_ham = cdist(onehot, onehot, metric="cityblock") / 2
med_ham, lab_ham, loss_ham = pam(D_ham, 3)
print(f"  k-means on the codes as numbers: ARI {ari(yw, km_codes.labels_):.3f}")
print(f"  k-means on the one-hot columns:  ARI {ari(yw, km_oh.labels_):.3f}")
print(f"  k-medoids with hamming:          ARI {ari(yw, lab_ham):.3f}")

# Renumbering the categories is a choice nobody remembers making. K-modes cannot
# see it. K-means on the codes can see nothing else.
rsp = np.random.RandomState(7)
perms, perm_rows = [], []
identical = 0
for t in range(40):
    P = [rsp.permutation(3) for _ in range(13)]
    permuted = np.stack([P[j][codes[:, j]] for j in range(13)], axis=1)
    _, lm, cm = kmodes(permuted, 3)
    a_modes = ari(yw, lm)
    a_km = ari(yw, KMeans(3, n_init=N_INIT_CAT, random_state=SEED).fit(permuted.astype(float)).labels_)
    if ari(lab_modes, lm) > 0.9999 and cm == cost_modes:
        identical += 1
    perms.append([[int(v) for v in p] for p in P])
    perm_rows.append({"modes": rnd(a_modes, 3), "km": rnd(a_km, 3), "cost": int(cm)})
km_range = [min(r["km"] for r in perm_rows), max(r["km"] for r in perm_rows)]
print(f"  40 relabellings: k-modes returned the identical partition {identical}/40 times; "
      f"k-means on the codes ranged {km_range[0]:.3f} to {km_range[1]:.3f} "
      f"(mean {np.mean([r['km'] for r in perm_rows]):.3f})")

modes_out = {
    "codes": [[int(v) for v in row] for row in codes],
    "levels": 3,
    "kmodes": {"labels": [int(v) for v in lab_modes], "cost": cost_modes,
               "ari": rnd(ari(yw, lab_modes), 3),
               "modes": [[int(v) for v in row] for row in modes],
               "pkg_cost": int(pkg.cost_)},
    "km_codes": {"labels": [int(v) for v in km_codes.labels_], "ari": rnd(ari(yw, km_codes.labels_), 3)},
    "km_onehot": {"labels": [int(v) for v in km_oh.labels_], "ari": rnd(ari(yw, km_oh.labels_), 3)},
    "medoids_hamming": {"labels": [int(v) for v in lab_ham], "ari": rnd(ari(yw, lab_ham), 3),
                        "medoids": [int(i) for i in med_ham], "loss": rnd(loss_ham, 1)},
    "perms": perms,
    "perm_rows": perm_rows,
    "identical": identical,
    "km_range": km_range,
    "km_mean": rnd(float(np.mean([r["km"] for r in perm_rows])), 3),
    "n_relabellings": int(6 ** 13),
}

# ======================================================= 3. every radius at once
print("\n--- OPTICS ---")
MIN_SAMPLES = 5
optics_sets = {}
for name, X, y in [("blobs", Xb, yb), ("rings", Xr, yr), ("trap", Xv, yv)]:
    order, reach, core = optics(X, MIN_SAMPLES)
    ref = OPTICS(min_samples=MIN_SAMPLES).fit(X)
    same = bool(np.array_equal(order, ref.ordering_))
    gap = float(np.abs(np.nan_to_num(reach, posinf=0) - np.nan_to_num(ref.reachability_, posinf=0)).max())
    # Points whose reachability ties can be visited in either order, so the
    # ordering is allowed to differ. The curve the page draws is not: that is
    # the thing to compare, and it is identical to the last decimal.
    curve = float(np.abs(np.nan_to_num(reach[order], posinf=0)
                         - np.nan_to_num(ref.reachability_[ref.ordering_], posinf=0)).max())
    swaps = int((order != ref.ordering_).sum())
    print(f"{name:<6} ordering identical to scikit-learn's: {same}"
          f"{'' if same else f' ({swaps} tied points visited in another order)'}, "
          f"largest reachability difference {gap:.2e}, largest difference in the drawn curve {curve:.2e}")

    # every horizontal cut of that ordering, on the slider's own grid
    cuts = []
    cut_grid = [round(v, 3) for v in np.arange(0.05, 2.001, 0.025)]
    for eps in cut_grid:
        lab = cluster_optics_dbscan(reachability=ref.reachability_, core_distances=ref.core_distances_,
                                    ordering=ref.ordering_, eps=eps)
        row = {"eps": eps, "n_clusters": int(lab.max()) + 1, "n_noise": int((lab == -1).sum()),
               "ari": rnd(ari(y, lab), 3)}
        if name == "trap":
            row["separated"], row["recovered"] = trap_metrics(lab)
        cuts.append(row)
    best_cut = max(cuts, key=lambda r: r["ari"])

    # and the extraction that reads the shape of the plot instead of its height
    xis = []
    for xi in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        op = OPTICS(min_samples=MIN_SAMPLES, xi=xi, min_cluster_size=0.05).fit(X)
        row = {"xi": xi, "labels": [int(v) for v in op.labels_],
               "n_clusters": int(op.labels_.max()) + 1, "n_noise": int((op.labels_ == -1).sum()),
               "ari": rnd(ari(y, op.labels_), 3),
               "hierarchy": [[int(a), int(b)] for a, b in op.cluster_hierarchy_]}
        if name == "trap":
            row["separated"], row["recovered"] = trap_metrics(op.labels_)
        xis.append(row)
    best_xi = max(xis, key=lambda r: r["ari"])
    print(f"       best cut: eps {best_cut['eps']} -> {best_cut['n_clusters']} clusters, "
          f"ARI {best_cut['ari']}   |   best xi: {best_xi['xi']} -> {best_xi['n_clusters']} clusters, "
          f"ARI {best_xi['ari']}")
    if name == "trap":
        sep_cuts = [c for c in cuts if c["separated"]]
        print(f"       cuts that separate the tight pair: eps {sep_cuts[0]['eps']} to {sep_cuts[-1]['eps']}, "
              f"and the most of the diffuse cluster any of them keeps is "
              f"{max(c['recovered'] for c in sep_cuts):.2f}")

    optics_sets[name] = {
        "ordering": [int(v) for v in ref.ordering_],
        # six decimals, not four: the browser's own OPTICS is checked against
        # this array at load time, and a guard cannot be sharper than the
        # precision of what it compares against
        "reachability": [None if not np.isfinite(v) else rnd(v, 6) for v in ref.reachability_[ref.ordering_]],
        "matches_sklearn": same,
        "ordering_swaps": swaps,
        "curve_gap": rnd(curve, 20),
        "cut_grid": cut_grid,
        "cuts": cuts,
        "best_cut": best_cut,
        "xis": xis,
    }

# the promise the DBSCAN article left open, answered on its own two measures
hdb = dens["hdbscan"]["10"]
trap_xi = max(optics_sets["trap"]["xis"], key=lambda r: min(r["separated"], r["recovered"]))
trap_cut = max(optics_sets["trap"]["cuts"], key=lambda r: min(r["separated"], r["recovered"]))
print(f"the trap, on the DBSCAN article's two measures: best single radius keeps "
      f"{trap_cut['recovered']:.2f} of the diffuse cluster, HDBSCAN {hdb['recovered']}, "
      f"OPTICS at xi {trap_xi['xi']} keeps {trap_xi['recovered']:.2f}")
# the previous article's own answer to this dataset, so the page can quote it
# without a second fetch and without anyone retyping it
optics_sets["trap"]["hdbscan_recovered"] = hdb["recovered"]

# ============================================================= 4. nobody says k
print("\n--- mean shift ---")
shift_sets = {}
BW_GRID = [round(v, 2) for v in np.arange(0.30, 4.001, 0.05)]
for name, X, y, k in [("blobs", Xb, yb, 3), ("rings", Xr, yr, 2), ("unequal", Xu, yu, 2), ("trap", Xv, yv, 3)]:
    auto = float(estimate_bandwidth(X, quantile=0.3, random_state=SEED))

    def sweep_bandwidths():
        rows_ = []
        for bw in BW_GRID:
            ms_ = MeanShift(bandwidth=bw, bin_seeding=False).fit(X)
            rows_.append({"bw": bw, "modes": int(len(ms_.cluster_centers_)),
                          "ari": rnd(ari(y, ms_.labels_), 3)})
        return rows_

    rows = cached(f"shift-{name}-s{SEED}-n{len(BW_GRID)}-{BW_GRID[0]}to{BW_GRID[-1]}", sweep_bandwidths)
    ms_auto = MeanShift(bandwidth=auto, bin_seeding=False).fit(X)
    best = max(rows, key=lambda r: r["ari"])
    right_k = [r for r in rows if r["modes"] == k]
    # the browser runs this itself, so check the two implementations agree first
    mism = 0
    for bw in [1.0, 1.5, 2.0, 2.5, 3.0]:
        c1, l1 = mean_shift(X, bw)
        c2 = MeanShift(bandwidth=bw, bin_seeding=False).fit(X)
        if len(c1) != len(c2.cluster_centers_) or ari(l1, c2.labels_) < 0.9999:
            mism += 1
    print(f"{name:<8} auto bandwidth {auto:.3f} -> {len(ms_auto.cluster_centers_)} modes, "
          f"ARI {ari(y, ms_auto.labels_):.3f} | best over the sweep {best['ari']} at bw {best['bw']} "
          f"| {len(right_k)} of {len(rows)} bandwidths give exactly {k} modes | "
          f"{mism}/5 disagreements with scikit-learn")
    shift_sets[name] = {
        "true_k": k,
        "auto_bw": rnd(auto, 3),
        "auto_modes": int(len(ms_auto.cluster_centers_)),
        "auto_ari": rnd(ari(y, ms_auto.labels_), 3),
        "sweep": rows,
        "best": best,
        "right_k_range": [right_k[0]["bw"], right_k[-1]["bw"]] if right_k else None,
        "sklearn_mismatches": mism,
    }

# ============================================================== 5. the scoreboard
# The module, on one table, with every number measured on the coordinates this
# page ships rather than quoted from anywhere.
print("\n--- the scoreboard (ARI, measured on the shipped coordinates) ---")
board = []
for name, X, y, k in [("blobs", Xb, yb, 3), ("anisotropic", Xa, ya, 3), ("rings", Xr, yr, 2),
                      ("unequal", Xu, yu, 2), ("trap", Xv, yv, 3)]:
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED).fit(X)
    D = cdist(X, X)
    _, lp, _ = pam(D, k)
    auto = float(estimate_bandwidth(X, quantile=0.3, random_state=SEED))
    ms = MeanShift(bandwidth=auto, bin_seeding=False).fit(X)
    op = OPTICS(min_samples=MIN_SAMPLES, xi=0.05, min_cluster_size=0.05).fit(X)
    # DBSCAN and HDBSCAN, swept, so the density row is the best a radius can do
    db_best = max((DBSCAN(eps=float(e), min_samples=MIN_SAMPLES).fit(X).labels_ for e in np.arange(0.05, 2.0, 0.025)),
                  key=lambda l: ari(y, l))
    hd = HDBSCAN(min_cluster_size=10, copy=True).fit(X)
    row = {
        "name": name, "k": k, "n": int(len(X)),
        "kmeans": rnd(ari(y, km.labels_), 3),
        "gmm": rnd(ari(y, gm.predict(X)), 3),
        "pam": rnd(ari(y, lp), 3),
        "dbscan_best": rnd(ari(y, db_best), 3),
        "hdbscan": rnd(ari(y, hd.labels_), 3),
        "meanshift": rnd(ari(y, ms.labels_), 3),
        "meanshift_modes": int(len(ms.cluster_centers_)),
        "optics": rnd(ari(y, op.labels_), 3),
    }
    board.append(row)
    print(f"  {name:<12} kmeans {row['kmeans']:>6}  gmm {row['gmm']:>6}  pam {row['pam']:>6}  "
          f"dbscan* {row['dbscan_best']:>6}  hdbscan {row['hdbscan']:>6}  "
          f"meanshift {row['meanshift']:>6} ({row['meanshift_modes']} modes)  optics {row['optics']:>6}")

# The k-means article publishes 0.529 for the sheared blobs and this file
# measures a different number, because it measures on the rounded coordinates
# the page draws. Worth measuring rather than waving at: the two nearest optima
# are a few hundredths apart, so the fourth decimal of the data decides.
km_raw = KMeans(3, n_init=10, random_state=SEED).fit(Xa_raw)
km_rnd = KMeans(3, n_init=10, random_state=SEED).fit(Xa)
knife = {
    "raw_ari": rnd(ari(ya, km_raw.labels_), 3),
    "rounded_ari": rnd(ari(ya, km_rnd.labels_), 3),
    "raw_inertia": rnd(km_raw.inertia_, 2),
    "rounded_inertia": rnd(km_rnd.inertia_, 2),
}
print(f"  sheared blobs, unrounded: ARI {knife['raw_ari']} at inertia {knife['raw_inertia']}; "
      f"rounded: ARI {knife['rounded_ari']} at inertia {knife['rounded_inertia']}")

# ============================================================ 6. what it costs
print("\n--- seconds, same 10-D blobs, k = 5, best of three ---")
# Best of three, because a wall clock on a machine that is also running a
# browser is not a measurement: the same row came out 3 to 5 times slower on a
# busy run than on a quiet one, and the minimum is the honest estimate of the
# work rather than of the contention.
REPEATS = 3


def time_costs():
    rows = []
    for n in [500, 2000, 5000]:
        Xs, ys = make_blobs(n_samples=n, centers=5, n_features=10, cluster_std=1.5, random_state=SEED)
        Dn = cdist(Xs, Xs)
        bw = estimate_bandwidth(Xs, quantile=0.3, random_state=SEED)
        best = {"kmeans": np.inf, "pam": np.inf, "optics": np.inf, "meanshift": np.inf}
        for _ in range(REPEATS):
            t0 = time.perf_counter(); KMeans(5, n_init=10, random_state=SEED).fit(Xs)
            best["kmeans"] = min(best["kmeans"], time.perf_counter() - t0)
            t0 = time.perf_counter(); kmedoids.fasterpam(Dn, 5, random_state=SEED)
            best["pam"] = min(best["pam"], time.perf_counter() - t0)
            t0 = time.perf_counter(); OPTICS(min_samples=5).fit(Xs)
            best["optics"] = min(best["optics"], time.perf_counter() - t0)
            t0 = time.perf_counter(); MeanShift(bandwidth=bw, bin_seeding=False).fit(Xs)
            best["meanshift"] = min(best["meanshift"], time.perf_counter() - t0)
        rows.append({"n": n, "repeats": REPEATS, "matrix_mb": rnd(Dn.nbytes / 1e6, 1),
                     **{k: rnd(v, 2) for k, v in best.items()}})
        r = rows[-1]
        print(f"  n={n:>5}  kmeans {r['kmeans']:>6}  pam {r['pam']:>6} (+{r['matrix_mb']} MB of distances)  "
              f"optics {r['optics']:>6}  meanshift {r['meanshift']:>6}")
    return rows


cost_rows = cached(f"cost-s{SEED}-r{REPEATS}", time_costs)

out = {
    "datasets": {name: {"points": [[rnd(a, 3), rnd(b, 3)] for a, b in X],
                        "true_labels": [int(v) for v in y]}
                 for name, X, y in [("blobs", Xb, yb), ("rings", Xr, yr),
                                    ("unequal", Xu, yu), ("trap", Xv, yv)]},
    "medoids": medoids_out,
    "outlier": outlier_out,
    "wine": wine_out,
    "modes": modes_out,
    "optics": optics_sets,
    "shift": shift_sets,
    "board": board,
    "knife_edge": knife,
    "cost": cost_rows,
    "meta": {"seed": SEED, "min_samples": MIN_SAMPLES,
             "shared_datasets_verified": {k: bool(v) for k, v in checks.items()},
             "seconds_to_generate": rnd(time.perf_counter() - t_start, 1)},
}
path = OUT / "relaxations.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB) in "
      f"{time.perf_counter() - t_start:.0f} s")
