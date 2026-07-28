"""Data for the ATLAS 'Affinities' web article, the sixth and last of module 2.1.

Every algorithm this module has published so far reads coordinates: a mean, a
radius, a covariance, a medoid. The two left on the map do not. Affinity
propagation and spectral clustering both start by throwing the points away and
keeping the n by n table of how similar each pair is, and everything they do
afterwards happens on that table. What that buys is the two things the module
has been paying for all along: nobody has to say k, and a cluster no longer has
to be a blob. What it costs is n squared, and a new parameter each.

Measured on the same points as the five articles before: the blobs of seed 19,
the failure gallery, the varying-density trap and the 178 bottles of wine, all
rebuilt here and asserted coordinate by coordinate against the JSON those
articles ship.

Results worth knowing before reading the code:

  * affinity propagation does not ask for k, and its default answer for k is
    wrong on every dataset here: 9 to 16 clusters where the truth is 2 or 3.
    The dial that replaces k is not a dial for k, and the sweep shows why: it
    is not monotone, and half the values of k are unreachable on the rings.
  * where it does land on the right k, its message passing finds exactly the
    optimum of the objective it defines: the same exemplars as PAM on the same
    similarities, and the exact best of every subset on a small enough sample.
  * spectral clustering solves both datasets that beat this module: the rings
    (1.000 against k-means' -0.003) and the trap the density article ended on
    (1.000). It also has three different textbook last steps that are not the
    same algorithm, and they disagree by 0.19 of ARI on one dataset here.
  * the relaxation everybody calls an approximation is, on the 16-point graph
    where all 32,767 splits can be enumerated, exact: the eigenvector's best
    threshold IS the best cut. Its eigenvalue, though, is not reachable.

No wall clock is published anywhere in this article. Times are the one number
that changes between machines, so the cost section counts similarities and
message updates instead, which are the same everywhere.

Run from anywhere: `python src/utils/generate_affinities_data.py`.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.cluster import (AffinityPropagation, DBSCAN, HDBSCAN, KMeans,
                             MeanShift, OPTICS, SpectralClustering,
                             estimate_bandwidth)
from sklearn.datasets import load_wine, make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "affinities" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
ari = lambda a, b: float(adjusted_rand_score(a, b))
SEED = 19
KNN = 10                      # neighbours in the default graph, everywhere
MIN_SAMPLES = 5               # the previous article's minPts, so its board rows reproduce
t_start = time.perf_counter()

CACHE = Path.home() / ".atlas_vision_data" / "affinities_cache"
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
# The two the browser will run itself, written here first so that the page and
# this file come out of one definition rather than two.

def affinity_propagation(S, damping=0.9, max_iter=2000, conv_iter=50, trace=False):
    """Frey and Dueck's message passing, following scikit-learn's loop line for
    line so that the two cannot disagree: the same rolling convergence window,
    the same re-election of each cluster's exemplar at the end, and the same
    decree that an exemplar belongs to its own cluster (without which an
    exemplar can be assigned to the next one over, because its own preference
    is often lower than its similarity to a neighbour).

    scikit-learn also perturbs S by about machine epsilon to break exact ties.
    That is not reproduced, and the check below measures that it changes
    nothing here: six random seeds give the same exemplars."""
    n = len(S)
    A = np.zeros((n, n))
    R = np.zeros((n, n))
    ind = np.arange(n)
    e = np.zeros((n, conv_iter), dtype=bool)
    counts = []
    it_used, converged = max_iter, False
    for it in range(max_iter):
        tmp = A + S
        I = tmp.argmax(axis=1)
        Y = tmp[ind, I]
        tmp[ind, I] = -np.inf
        Y2 = tmp.max(axis=1)
        tmp = S - Y[:, None]
        tmp[ind, I] = S[ind, I] - Y2
        R = damping * R + (1 - damping) * tmp

        tmp = np.maximum(R, 0)
        tmp.flat[::n + 1] = R.flat[::n + 1]
        tmp = tmp - tmp.sum(axis=0)                 # minus the new availabilities
        dA = np.diag(tmp).copy()
        tmp = tmp.clip(0, np.inf)
        tmp.flat[::n + 1] = dA
        A = damping * A - (1 - damping) * tmp

        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % conv_iter] = E
        K = int(E.sum())
        if trace:
            counts.append(K)
        if it >= conv_iter:
            se = e.sum(axis=1)
            if int(((se == conv_iter) | (se == 0)).sum()) == n and K > 0:
                it_used, converged = it + 1, True
                break
    ex = np.flatnonzero((np.diag(A) + np.diag(R)) > 0)
    if not len(ex):
        return np.array([], int), np.zeros(n, int), it_used, converged, counts
    c = S[:, ex].argmax(axis=1)
    c[ex] = np.arange(len(ex))
    for k in range(len(ex)):
        ii = np.where(c == k)[0]
        ex[k] = ii[S[np.ix_(ii, ii)].sum(axis=0).argmax()]
    c = S[:, ex].argmax(axis=1)
    c[ex] = np.arange(len(ex))
    labels = ex[c]
    centres = np.unique(labels)
    return centres, np.searchsorted(centres, labels), it_used, converged, counts


def net_similarity(S, exemplars, preference):
    """What affinity propagation maximises: every point's similarity to its
    exemplar, plus the preference paid once per exemplar.

    The exemplars' own columns are left out of the first sum rather than read
    off a diagonal, for two reasons: an exemplar's similarity to itself IS the
    preference, which the second term already charges, and a max over the
    exemplar rows would happily assign an exemplar to a closer neighbour and
    count somebody else's similarity in its place."""
    E = list(exemplars)
    if not len(E):
        return -np.inf
    others = np.setdiff1d(np.arange(len(S)), E)
    total = preference * len(E)
    if len(others):
        total += float(S[np.ix_(E, others)].max(axis=0).sum())
    return total


def pam(D, k, max_iter=100):
    """The previous article's PAM, kept here to ask one question: given the
    same similarities and the same k, does message passing find a better set of
    centres than build-and-swap does?"""
    med = [int(np.argmin(D.sum(axis=1)))]
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
            d_other = D[others].min(axis=0)
            cand = np.minimum(d_other[None, :], D).sum(axis=1)
            cand[med] = np.inf
            h = int(np.argmin(cand))
            if best is None or cand[h] < best[0]:
                best = (float(cand[h]), i, h)
        if best is None or best[0] >= cost - 1e-12:
            break
        med[best[1]] = best[2]
        med = sorted(med)
    return np.array(med), np.argmin(D[med], axis=0), float(D[med].min(axis=0).sum())


def knn_graph(X, k):
    """k nearest neighbours, symmetrised with an OR and left unweighted, which
    is what scikit-learn's `affinity='nearest_neighbors'` builds.

    Sorted by SQUARED distance, which the browser also uses because it saves a
    square root per pair. That is not a free choice: the two orderings are the
    same ordering in exact arithmetic and not always in floating point, since
    sqrt is monotone but collapses neighbouring floats onto each other. On the
    trap at 20 neighbours the two versions differ by exactly one edge, which is
    enough to move the sixth eigenvalue in the third decimal and to make a
    guard fire that has nothing wrong with it."""
    D = cdist(X, X, "sqeuclidean")
    A = np.zeros_like(D)
    for i in range(len(X)):
        A[i, np.argsort(D[i], kind="stable")[1:k + 1]] = 1.0
    return np.maximum(A, A.T)


def rbf_graph(X, gamma):
    """A gaussian on every pair. The diagonal is zeroed: a self-loop changes
    every degree by the same 1 and no cut by anything, and the browser has to
    build the same matrix this does."""
    W = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    np.fill_diagonal(W, 0.0)
    return W


def laplacian_eigen(W):
    """Eigenvalues of the symmetric normalised Laplacian, ascending, with the
    eigenvectors of the matrix actually iterated in the browser."""
    d = W.sum(axis=1)
    dinv = 1 / np.sqrt(np.maximum(d, 1e-12))
    M = (W * dinv[:, None]) * dinv[None, :]
    vals, vecs = eigh(M)
    return 1 - vals[::-1], vecs[:, ::-1], d


def spectral(W, k, seed=SEED, restarts=30):
    """Ng, Jordan and Weiss: the k smallest eigenvectors of the normalised
    Laplacian, rows rescaled to unit length, k-means on those rows."""
    lam, vecs, _ = laplacian_eigen(W)
    U = vecs[:, :k]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    return KMeans(k, n_init=restarts, random_state=seed).fit(U).labels_, lam


def ncut(W, mask):
    """Shi and Malik's normalised cut of a bipartition."""
    cut = W[mask][:, ~mask].sum()
    a, b = W[mask].sum(), W[~mask].sum()
    if a == 0 or b == 0:
        return np.inf
    return float(cut / a + cut / b)


# ================================================================ the datasets
# Rebuilt with the constructions of the articles that first used them, then
# checked against the JSON those articles shipped. Without this the whole
# article is a comparison between different data.
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
Xa = np.round(Xa @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]), 3)

rs2 = np.random.RandomState(SEED)
Xv = np.round(np.vstack([
    rs2.normal([-0.8, 0.5], 0.14, (100, 2)),
    rs2.normal([0.01, -0.09], 0.14, (100, 2)),
    rs2.normal([4.0, 0.2], 1.15, (150, 2)),
]), 3)
yv = np.array([0] * 100 + [1] * 100 + [2] * 150)

prev = json.loads((ROOT / "kmeans" / "data" / "clusters.json").read_text(encoding="utf-8"))
dens = json.loads((ROOT / "dbscan" / "data" / "density.json").read_text(encoding="utf-8"))
relax = json.loads((ROOT / "assumptions" / "data" / "relaxations.json").read_text(encoding="utf-8"))
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

SETS = [("blobs", Xb, yb, 3), ("anisotropic", Xa, ya, 3), ("rings", Xr, yr, 2),
        ("unequal", Xu, yu, 2), ("trap", Xv, yv, 3)]
BY = {name: (X, y, k) for name, X, y, k in SETS}

# ======================================================== 1. messages and k
print("\n--- affinity propagation ---")
SIM = {}
for name, X, y, k in SETS + [("wine", Xw, yw, 3)]:
    S = -cdist(X, X, "sqeuclidean")
    iu = np.triu_indices(len(X), 1)
    SIM[name] = (S, float(np.median(S[iu])), float(S[iu].min()))

ap_default = {}
for name, X, y, k in SETS + [("wine", Xw, yw, 3)]:
    S, med, lo = SIM[name]
    Sp = S.copy()
    np.fill_diagonal(Sp, med)
    ex, lab, its, conv, _ = affinity_propagation(Sp)
    ap_default[name] = {
        "median_pref": rnd(med, 3), "min_pref": rnd(lo, 3), "true_k": k,
        "k": int(len(ex)), "ari": rnd(ari(y, lab), 3), "iters": int(its),
    }
    print(f"  {name:<12} default preference (the median, {med:>9.2f}) -> {len(ex):>2} clusters, "
          f"ARI {ari(y, lab):.3f} against a true k of {k}, in {its} iterations")

# The tie-break noise scikit-learn adds: does it decide anything here?
noise_stable = True
for name in ["blobs", "rings"]:
    S, med, lo = SIM[name]
    outs = set()
    for rsd in range(6):
        a = AffinityPropagation(damping=0.9, preference=med, max_iter=2000,
                                convergence_iter=50, random_state=rsd).fit(BY[name][0])
        outs.add(tuple(sorted(a.cluster_centers_indices_.tolist())))
    noise_stable &= len(outs) == 1
print(f"  scikit-learn's random tie-break gives one answer over six seeds: {noise_stable}")

# ------------------------------------------------- the dial that replaces k
# The grid the slider will offer, geometric in the size of the preference so
# that the whole staircase fits: from twenty times the most negative similarity
# (where everything is one cluster) up to the median (scikit-learn's default).
def pref_grid(med, lo, n=56):
    return [round(-v, 3) for v in np.geomspace(abs(med), 20 * abs(lo), n)][::-1]


ap_sweep = {}
for name, X, y, k in SETS:
    S, med, lo = SIM[name]
    grid = pref_grid(med, lo)

    def run():
        out = {"k": [], "ari": [], "iters": [], "converged": []}
        for p in grid:
            Sp = S.copy()
            np.fill_diagonal(Sp, p)
            ex, lab, its, conv, _ = affinity_propagation(Sp, max_iter=1200, conv_iter=40)
            out["k"].append(int(len(ex)))
            out["ari"].append(rnd(ari(y, lab), 3))
            out["iters"].append(int(its))
            out["converged"].append(bool(conv))
        return out

    sw = cached(f"apsweep-{name}-s{SEED}-n{len(grid)}-v1", run)
    ks = sw["k"]
    reachable = sorted(set(ks))
    missing = [v for v in range(min(ks), max(ks) + 1) if v not in reachable]
    drops = sum(1 for i in range(len(ks) - 1) if ks[i] > ks[i + 1])
    # The least drastic setting among the ties: the grid runs from the most
    # negative preference upwards, so the LAST index reaching the best score is
    # the one closest to a default a reader would actually type.
    best_i = max(i for i, v in enumerate(sw["ari"]) if v == max(sw["ari"]))
    ap_sweep[name] = {
        "grid": grid, **sw,
        # `reachable` stays although only `missing` is quoted: it is the
        # trace for a claim the page makes in bold, which is the one exception
        # to shipping nothing the page does not read.
        "reachable": reachable, "missing": missing, "drops": drops,
        "best": {"pref": grid[best_i], "k": ks[best_i], "ari": sw["ari"][best_i], "index": best_i},
    }
    print(f"  {name:<12} sweep of {len(grid)} preferences: k from {min(ks)} to {max(ks)}, "
          f"{len(missing)} value(s) of k unreachable {missing}, {drops} step(s) where a higher "
          f"preference gives FEWER clusters; best ARI {sw['ari'][best_i]} at k={ks[best_i]}")

# --------------------------------------------------------------- the damping
# The parameter that exists only to make the iteration converge, and changes
# the answer. Below scikit-learn's floor of 0.5 it does not converge at all.
DAMP = [round(v, 2) for v in np.arange(0.30, 0.96, 0.05)]
DAMP_PREF = ap_sweep["blobs"]["best"]["pref"]


def damping_sweep():
    S, med, lo = SIM["blobs"]
    Sp = S.copy()
    np.fill_diagonal(Sp, DAMP_PREF)
    rows = []
    for d in DAMP:
        ex, lab, its, conv, counts = affinity_propagation(Sp, damping=d, max_iter=2000,
                                                          conv_iter=50, trace=True)
        rows.append({"damping": d, "k": int(len(ex)), "iters": int(its), "converged": bool(conv),
                     "ari": rnd(ari(yb, lab), 3),
                     "swing": int(max(counts[-200:]) - min(counts[-200:])),
                     "tail": [int(v) for v in counts[-8:]]})
    return rows


damp_rows = cached(f"damping-s{SEED}-p{DAMP_PREF}-n{len(DAMP)}-v1", damping_sweep)
for r in damp_rows:
    if r["converged"]:
        print(f"  damping {r['damping']:.2f}: {r['k']:>3} clusters, ARI {r['ari']:>6}, "
              f"converged in {r['iters']} iterations")
    else:
        print(f"  damping {r['damping']:.2f}: never settled in 2000 iterations; over the last 200 "
              f"of them the number of points electing themselves swings by {r['swing']}")
damped_ok = [r for r in damp_rows if r["converged"]]
answers = sorted({r["k"] for r in damped_ok})
print(f"  among the {len(damped_ok)} damping values that converge, {len(answers)} different "
      f"answers for k: {answers}")

# ------------------------------------- the objective, and who else can find it
print("\n--- the objective, and two searches for it ---")
S, med, lo = SIM["blobs"]
Sp = S.copy()
np.fill_diagonal(Sp, DAMP_PREF)
ex_ap, lab_ap, its_ap, _, _ = affinity_propagation(Sp)
k_ap = len(ex_ap)
D2 = cdist(Xb, Xb, "sqeuclidean")
med_pam, lab_pam, loss_pam = pam(D2, k_ap)
net_ap = net_similarity(Sp, ex_ap, DAMP_PREF)
net_pam = net_similarity(Sp, med_pam, DAMP_PREF)
same_set = sorted(ex_ap.tolist()) == sorted(med_pam.tolist())
print(f"  at preference {DAMP_PREF}, message passing elects {sorted(ex_ap.tolist())} "
      f"(net similarity {net_ap:.2f}), and PAM on the same similarities with the same k picks "
      f"{sorted(med_pam.tolist())} (net {net_pam:.2f}): same set {same_set}")

# On a subsample every subset can be tried, so "is this the optimum" has an answer.
SUB_STRIDE = 10
sub = Xb[::SUB_STRIDE]
Ssub = -cdist(sub, sub, "sqeuclidean")
nsub = len(sub)
MAXK = 6


def brute_force():
    out = {}
    for tag, p in [("chosen", DAMP_PREF), ("median", med)]:
        Sp2 = Ssub.copy()
        np.fill_diagonal(Sp2, p)
        best = (-np.inf, None)
        tried = 0
        for kk in range(1, MAXK + 1):
            for c in itertools.combinations(range(nsub), kk):
                v = net_similarity(Sp2, c, p)
                tried += 1
                if v > best[0]:
                    best = (v, list(c))
        ex2, lab2, _, _, _ = affinity_propagation(Sp2, max_iter=4000, conv_iter=100)
        out[tag] = {"preference": rnd(p, 3), "subsets": tried,
                    "best_net": rnd(best[0], 4), "best_k": len(best[1]),
                    "ap_net": rnd(net_similarity(Sp2, ex2, p), 4), "ap_k": int(len(ex2)),
                    "same": sorted(ex2.tolist()) == best[1]}
    return out


brute = cached(f"brute-s{SEED}-n{nsub}-k{MAXK}-p{DAMP_PREF}-v2", brute_force)
for tag, b in brute.items():
    gap = b["best_net"] - b["ap_net"]
    print(f"  {nsub} points at the {tag} preference, every one of the {b['subsets']:,} subsets of "
          f"up to {MAXK}: best net {b['best_net']} at k={b['best_k']}; message passing "
          f"{b['ap_net']} at k={b['ap_k']} ({'the same set' if b['same'] else f'a different set, short by {gap:.4f}'})"
          + (", and the cap on k is doing the work" if b["best_k"] == MAXK else ""))

# ------------------------------------------------------------ the wine, tuned
S, medw, low = SIM["wine"]
grid_w = pref_grid(medw, low, 48)
wine_rows = []
for p in grid_w:
    Sp2 = S.copy()
    np.fill_diagonal(Sp2, p)
    ex2, lab2, _, _, _ = affinity_propagation(Sp2, max_iter=1200, conv_iter=40)
    wine_rows.append({"pref": p, "k": int(len(ex2)), "ari": rnd(ari(yw, lab2), 3)})
w_best = max(wine_rows, key=lambda r: r["ari"])
w_k3 = [r for r in wine_rows if r["k"] == 3]
km_wine = KMeans(3, n_init=30, random_state=SEED).fit(Xw)
_, lab_pw, _ = pam(cdist(Xw, Xw), 3)
wine_out = {
    "n": int(len(Xw)), "features": int(Xw.shape[1]),
    "best": w_best,
    "at_k3": max(w_k3, key=lambda r: r["ari"]) if w_k3 else None,
    "kmeans_ari": rnd(ari(yw, km_wine.labels_), 3),
    "pam_ari": rnd(ari(yw, lab_pw), 3),
    "default_k": ap_default["wine"]["k"], "default_ari": ap_default["wine"]["ari"],
}
print(f"  wine: best ARI over the sweep {w_best['ari']} at k={w_best['k']}; at k=3 "
      f"{wine_out['at_k3']['ari'] if w_k3 else 'never reached'}; k-means {wine_out['kmeans_ari']}, "
      f"PAM {wine_out['pam_ari']}")

# ---------------------------------------------- what the browser is checked on
ap_checks = []
for name in ["blobs", "rings", "trap"]:
    S, med, lo = SIM[name]
    X, y, k = BY[name]
    for tag, p in [("median", med), ("swept", ap_sweep[name]["grid"][20]),
                   ("lowest", ap_sweep[name]["grid"][0])]:
        Sp2 = S.copy()
        np.fill_diagonal(Sp2, p)
        ex2, lab2, its2, conv2, _ = affinity_propagation(Sp2)
        sk = AffinityPropagation(damping=0.9, preference=p, max_iter=2000,
                                 convergence_iter=50, random_state=0).fit(X)
        skx = sorted(sk.cluster_centers_indices_.tolist()) if sk.cluster_centers_indices_ is not None else []
        ap_checks.append({
            "dataset": name, "preference": rnd(p, 3), "damping": 0.9,
            "exemplars": [int(v) for v in ex2], "k": int(len(ex2)), "iters": int(its2),
            "ari": rnd(ari(y, lab2), 4),
            "sklearn_same": sorted(ex2.tolist()) == skx,
        })
mismatch = [c for c in ap_checks if not c["sklearn_same"]]
print(f"  the loop written here and scikit-learn's elect the same exemplars in "
      f"{len(ap_checks) - len(mismatch)} of {len(ap_checks)} checks")

# ============================================================ 2. the graph
print("\n--- spectral clustering ---")
KNN_GRID = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80]
GAMMA_GRID = [round(float(v), 4) for v in np.geomspace(0.02, 60, 26)]
spectral_sets = {}
for name, X, y, k in SETS:
    knn_rows, rbf_rows = [], []
    for kk in KNN_GRID:
        W = knn_graph(X, kk)
        lab, lam = spectral(W, k)
        comp = int((lam < 1e-8).sum())
        knn_rows.append({"k": kk, "ari": rnd(ari(y, lab), 3), "components": comp})
    for g in GAMMA_GRID:
        W = rbf_graph(X, g)
        lab, lam = spectral(W, k)
        rbf_rows.append({"gamma": g, "ari": rnd(ari(y, lab), 3),
                         "components": int((lam < 1e-8).sum())})
    W = knn_graph(X, KNN)
    lab, lam = spectral(W, k)
    km = KMeans(k, n_init=30, random_state=SEED).fit(X)
    best_knn = max(knn_rows, key=lambda r: r["ari"])
    band = [r["k"] for r in knn_rows if r["ari"] >= best_knn["ari"] - 0.001]
    spectral_sets[name] = {
        "true_k": k,
        "knn_sweep": knn_rows, "rbf_sweep": rbf_rows,
        "ari": rnd(ari(y, lab), 3), "kmeans_ari": rnd(ari(y, km.labels_), 3),
        "components": int((lam < 1e-8).sum()),
        # six decimals: the browser's own eigenvalues are checked against these
        "eigenvalues": [rnd(v, 6) for v in lam[:9]],
        "best_knn": best_knn, "knn_band": [min(band), max(band)],
        "best_rbf": max(rbf_rows, key=lambda r: r["ari"]),
    }
    print(f"  {name:<12} at {KNN} neighbours: ARI {ari(y, lab):.3f} against k-means' "
          f"{ari(y, km.labels_):.3f}; graph in {spectral_sets[name]['components']} piece(s); "
          f"best over the neighbour sweep {best_knn['ari']} at k={best_knn['k']}")

# the wine, which has no picture on the page but belongs in the comparison
Ww = knn_graph(Xw, KNN)
lab_w, lam_w = spectral(Ww, 3)
spectral_wine = {"ari": rnd(ari(yw, lab_w), 3), "kmeans_ari": rnd(ari(yw, km_wine.labels_), 3),
                 "eigenvalues": [rnd(v, 6) for v in lam_w[:9]],
                 "components": int((lam_w < 1e-8).sum())}
print(f"  {'wine':<12} at {KNN} neighbours: ARI {spectral_wine['ari']} against k-means' "
      f"{spectral_wine['kmeans_ari']}")

# --------------------------------------------------- three textbook last steps
print("  the three last steps in the literature, on the same graph:")
variants = {}
for name, X, y, k in SETS + [("wine", Xw, yw, 3)]:
    W = knn_graph(X, KNN)
    d = W.sum(axis=1)
    dinv = 1 / np.sqrt(np.maximum(d, 1e-12))
    lam, vecs, _ = laplacian_eigen(W)
    U = vecs[:, :k]
    rows = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    njw = KMeans(k, n_init=30, random_state=SEED).fit(rows).labels_
    walk = KMeans(k, n_init=30, random_state=SEED).fit(U * dinv[:, None]).labels_
    L = np.diag(d) - W
    _, e2 = eigh(L)
    unnorm = KMeans(k, n_init=30, random_state=SEED).fit(e2[:, :k]).labels_
    sk = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=KNN,
                            random_state=SEED, assign_labels="kmeans").fit(X)
    variants[name] = {"njw": rnd(ari(y, njw), 3), "walk": rnd(ari(y, walk), 3),
                      "unnorm": rnd(ari(y, unnorm), 3), "sklearn": rnd(ari(y, sk.labels_), 3)}
    # Two spreads, because they are two different claims: how far apart the
    # three textbook last steps are, and how far scikit-learn's own pipeline
    # (which scales the eigenvectors by the degree rather than the row norm)
    # sits from all of them. Quoting the second while naming the first three is
    # how a true number ends up in a false sentence.
    three = [variants[name][k2] for k2 in ("njw", "walk", "unnorm")]
    variants[name]["spread"] = rnd(max(three) - min(three), 3)
    variants[name]["spread_with_sklearn"] = rnd(max(three + [variants[name]["sklearn"]])
                                                - min(three + [variants[name]["sklearn"]]), 3)
    print(f"    {name:<12} row-normalised {variants[name]['njw']:>6}  random walk "
          f"{variants[name]['walk']:>6}  unnormalised {variants[name]['unnorm']:>6}  "
          f"(spread {variants[name]['spread']:.3f})  scikit-learn "
          f"{variants[name]['sklearn']:>6}  (spread with it {variants[name]['spread_with_sklearn']:.3f})")
worst_variant = max(variants.items(), key=lambda kv: kv[1]["spread"])
worst_sklearn = max(variants.items(), key=lambda kv: kv[1]["spread_with_sklearn"])
print(f"    the three textbook versions are furthest apart on the {worst_variant[0]} "
      f"({worst_variant[1]['spread']}); scikit-learn's own is furthest from them on the "
      f"{worst_sklearn[0]} ({worst_sklearn[1]['spread_with_sklearn']})")

# ------------------------------------------------------------- the eigengap
print("  the eigengap heuristic, and how far you look:")
eigengap = {}
for name, X, y, k in SETS + [("wine", Xw, yw, 3)]:
    W = knn_graph(X, KNN)
    lam, _, _ = laplacian_eigen(W)
    picks = {}
    for horizon in [4, 6, 9]:
        gaps = np.diff(lam[:horizon])
        picks[str(horizon)] = int(np.argmax(gaps)) + 1
    eigengap[name] = {"true_k": k, "lam": [rnd(v, 6) for v in lam[:9]], "picks": picks}
    print(f"    {name:<12} true k {k}: looking at the first 4 eigenvalues it says "
          f"{picks['4']}, at 6 it says {picks['6']}, at 9 it says {picks['9']}")
for horizon in ["4", "6", "9"]:
    right = sum(1 for v in eigengap.values() if v["picks"][horizon] == v["true_k"])
    print(f"    right on {right} of {len(eigengap)} datasets when looking at {horizon} eigenvalues")

# --------------------------------------------- what the browser is checked on
spectral_checks = []
for name in ["blobs", "rings", "trap"]:
    X, y, k = BY[name]
    for mode, param in [("knn", 5), ("knn", KNN), ("knn", 20), ("rbf", 1.0), ("rbf", 5.0)]:
        W = knn_graph(X, param) if mode == "knn" else rbf_graph(X, param)
        lab, lam = spectral(W, k)
        spectral_checks.append({"dataset": name, "mode": mode, "param": param,
                                "eigenvalues": [rnd(v, 6) for v in lam[:6]],
                                "ari": rnd(ari(y, lab), 4)})

# ================================================ 3. the relaxation, exhausted
print("\n--- the relaxation, against every possible cut ---")
CUT_STRIDE = 19
CUT_GAMMA = 0.15
cut_pts = Xb[::CUT_STRIDE]
ncut_n = len(cut_pts)
Wc = rbf_graph(cut_pts, CUT_GAMMA)
lam_c, vec_c, deg_c = laplacian_eigen(Wc)
fiedler = vec_c[:, 1] / np.sqrt(deg_c)              # the random-walk eigenvector
best_cut, all_cuts = (np.inf, None), []
for r in range(1, ncut_n):
    for c in itertools.combinations(range(ncut_n), r):
        if 0 not in c:
            continue                                # each split counted once
        mask = np.zeros(ncut_n, bool)
        mask[list(c)] = True
        v = ncut(Wc, mask)
        all_cuts.append(v)
        if v < best_cut[0]:
            best_cut = (v, mask.copy())
order = np.argsort(fiedler)
sweep_best = (np.inf, None, None)
threshold_rows = []
for t in range(1, ncut_n):
    mask = np.zeros(ncut_n, bool)
    mask[order[:t]] = True
    v = ncut(Wc, mask)
    cutpoint = float((fiedler[order[t - 1]] + fiedler[order[t]]) / 2)
    threshold_rows.append({"t": t, "ncut": rnd(v, 6), "at": rnd(cutpoint, 6)})
    if v < sweep_best[0]:
        sweep_best = (v, mask.copy(), t)
sign_mask = fiedler > 0
sign_cut = ncut(Wc, sign_mask)
same_as_optimum = bool(np.array_equal(best_cut[1], sweep_best[1])
                       or np.array_equal(best_cut[1], ~sweep_best[1]))
rank_of_sign = int(sum(1 for v in all_cuts if v < sign_cut - 1e-12) + 1)
print(f"  {ncut_n} points, {len(all_cuts):,} splits: the best normalised cut is {best_cut[0]:.6f}")
print(f"  the eigenvector's best threshold gives {sweep_best[0]:.6f} "
      f"({'the same split' if same_as_optimum else 'a different split'}), "
      f"its sign gives {sign_cut:.6f} (rank {rank_of_sign} of {len(all_cuts):,}), "
      f"and the relaxed value it is a relaxation of is {lam_c[1]:.6f}")
cut_out = {
    "points": [[rnd(a, 3), rnd(b, 3)] for a, b in cut_pts],
    "gamma": CUT_GAMMA, "stride": CUT_STRIDE, "n": ncut_n, "splits": len(all_cuts),
    "optimum": rnd(best_cut[0], 6), "optimum_mask": [int(v) for v in best_cut[1]],
    "sweep_best": rnd(sweep_best[0], 6), "sweep_t": int(sweep_best[2]),
    "sweep_is_optimum": same_as_optimum,
    "sign": rnd(sign_cut, 6), "sign_rank": rank_of_sign,
    "lambda2": rnd(lam_c[1], 6),
    "eigenvalues": [rnd(v, 6) for v in lam_c[:6]],
    "fiedler": [rnd(v, 6) for v in fiedler],
    "thresholds": threshold_rows,
    "histogram": {"min": rnd(min(all_cuts), 6), "max": rnd(max(all_cuts), 6),
                  "median": rnd(float(np.median(all_cuts)), 6)},
}

# ============================================================== 4. the bill
print("\n--- the bill, counted rather than timed ---")
cost_rows = []
for n in [178, 300, 1000, 10_000, 100_000]:
    cost_rows.append({"n": n, "entries": n * n, "gb": rnd(n * n * 8 / 1e9, 4)})
    print(f"  n = {n:>7,}: {n * n:>15,} similarities, {n * n * 8 / 1e9:>9.3f} GB at eight bytes each")
ap_updates = {"n": len(Xb), "iters": int(its_ap), "updates": int(2 * its_ap * len(Xb) ** 2)}
km_ops = {"k": 3, "passes": int(KMeans(3, n_init=1, random_state=SEED).fit(Xb).n_iter_),
          "per_pass": 3 * len(Xb)}
print(f"  affinity propagation on the 300 blobs: {ap_updates['iters']} iterations of two n by n "
      f"message updates each, {ap_updates['updates']:,} in total")
print(f"  k-means on the same points: {km_ops['passes']} passes of {km_ops['per_pass']} distances, "
      f"{km_ops['passes'] * km_ops['per_pass']:,} in total")

# ========================================================= 5. the scoreboard
print("\n--- the module's scoreboard, every cell measured on the shipped coordinates ---")


def board_row(name, X, y, k):
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED).fit(X)
    D = cdist(X, X)
    _, lp, _ = pam(D, k)
    auto = float(estimate_bandwidth(X, quantile=0.3, random_state=SEED))
    ms = MeanShift(bandwidth=auto, bin_seeding=False).fit(X)
    op = OPTICS(min_samples=MIN_SAMPLES, xi=0.05, min_cluster_size=0.05).fit(X)
    db_best = max((DBSCAN(eps=float(e), min_samples=MIN_SAMPLES).fit(X).labels_
                   for e in np.arange(0.05, 2.0, 0.025)), key=lambda l: ari(y, l))
    hd = HDBSCAN(min_cluster_size=10, copy=True).fit(X)
    S, med, lo = SIM[name]
    Sp2 = S.copy()
    np.fill_diagonal(Sp2, med)
    ex2, lab2, _, _, _ = affinity_propagation(Sp2)
    sp_lab, _ = spectral(knn_graph(X, KNN), k)
    return {
        "name": name, "k": k, "n": int(len(X)),
        "kmeans": rnd(ari(y, km.labels_), 3), "gmm": rnd(ari(y, gm.predict(X)), 3),
        "pam": rnd(ari(y, lp), 3), "dbscan_best": rnd(ari(y, db_best), 3),
        "hdbscan": rnd(ari(y, hd.labels_), 3), "meanshift": rnd(ari(y, ms.labels_), 3),
        "meanshift_modes": int(len(ms.cluster_centers_)),
        "optics": rnd(ari(y, op.labels_), 3),
        "ap": rnd(ari(y, lab2), 3), "ap_k": int(len(ex2)),
        "spectral": rnd(ari(y, sp_lab), 3),
    }


board = cached(f"board-s{SEED}-knn{KNN}-m{MIN_SAMPLES}-v3",
               lambda: [board_row(n, X, y, k) for n, X, y, k in SETS])
# The previous article published seven of these ten columns on the very same
# coordinates. Reusing a number is a claim, so it is checked rather than stated.
old = {r["name"]: r for r in relax["board"]}
drift = []
for r in board:
    for key in ["kmeans", "gmm", "pam", "dbscan_best", "hdbscan", "meanshift", "optics"]:
        if abs(r[key] - old[r["name"]][key]) > 0.0005:
            label = {"kmeans": "k-means", "gmm": "the mixture", "pam": "k-medoids",
                     "dbscan_best": "DBSCAN", "hdbscan": "HDBSCAN",
                     "meanshift": "mean shift", "optics": "OPTICS"}[key]
            drift.append(f"{label} on the {r['name']}, {r[key]:.3f} here against "
                         f"{old[r['name']][key]:.3f} published")
    print(f"  {r['name']:<12} kmeans {r['kmeans']:>6}  medoids {r['pam']:>6}  dbscan* "
          f"{r['dbscan_best']:>6}  meanshift {r['meanshift']:>6}  optics {r['optics']:>6}  "
          f"affinity {r['ap']:>6} ({r['ap_k']} clusters)  spectral {r['spectral']:>6}")

# --------------------------------------------- the three that do not ask for k
nok = []
for name, X, y, k in SETS:
    W = knn_graph(X, KNN)
    lam, _, _ = laplacian_eigen(W)
    gap_k = int(np.argmax(np.diff(lam[:6]))) + 1
    row_b = [r for r in board if r["name"] == name][0]
    published = relax["shift"].get(name, {}).get("auto_modes")
    row = {"name": name, "true_k": k, "ap_k": row_b["ap_k"], "gap_k": gap_k,
           "shift_k": row_b["meanshift_modes"], "components": int((lam < 1e-8).sum())}
    if published is not None and published != row["shift_k"]:
        drift.append(f"the number of mean shift modes on the {name}, {row['shift_k']} here "
                     f"against {published} published")
    nok.append(row)
    print(f"  {name:<12} true k {k}: mean shift said {row['shift_k']}, affinity propagation "
          f"{row['ap_k']}, the eigengap {row['gap_k']}")
score = {key: sum(1 for r in nok if r[key] == r["true_k"]) for key in ["ap_k", "gap_k", "shift_k"]}
print(f"  right answers out of {len(nok)}: mean shift {score['shift_k']}, "
      f"affinity propagation {score['ap_k']}, eigengap {score['gap_k']}")
print(f"  the eight numbers the previous article published on these same coordinates reproduce "
      f"here: {'all of them' if not drift else 'NO: ' + '; '.join(drift)}")

# =================================================================== the file
out = {
    "datasets": {name: {"points": [[rnd(a, 3), rnd(b, 3)] for a, b in X],
                        "true_labels": [int(v) for v in y], "k": k}
                 for name, X, y, k in SETS},
    "ap": {
        "default": ap_default, "sweep": ap_sweep, "damping": {
            "grid": DAMP, "rows": damp_rows, "preference": DAMP_PREF,
            "answers": answers, "floor": 0.5,
        },
        "objective": {
            "preference": DAMP_PREF, "exemplars": [int(v) for v in ex_ap], "k": int(k_ap),
            "net": rnd(net_ap, 3), "iters": int(its_ap),
            "pam_medoids": [int(v) for v in med_pam], "pam_net": rnd(net_pam, 3),
            "same_set": same_set, "ari_ap": rnd(ari(yb, lab_ap), 3),
            "ari_pam": rnd(ari(yb, lab_pam), 3), "brute": brute, "sub_n": nsub, "max_k": MAXK,
            "sub_stride": SUB_STRIDE,
        },
        "wine": wine_out, "checks": ap_checks, "noise_stable": noise_stable,
    },
    "spectral": {
        "knn": KNN, "sets": spectral_sets, "wine": spectral_wine, "variants": variants,
        "worst_variant": worst_variant[0], "worst_sklearn": worst_sklearn[0],
        "eigengap": eigengap, "checks": spectral_checks,
        "knn_grid": KNN_GRID, "gamma_grid": GAMMA_GRID,
    },
    "cut": cut_out,
    "cost": {"rows": cost_rows, "ap": ap_updates, "kmeans": km_ops},
    "board": board, "nok": nok, "nok_score": score,
    "meta": {
        "seed": SEED, "knn": KNN, "damping": 0.9, "min_samples": MIN_SAMPLES,
        "shared_datasets_verified": {k: bool(v) for k, v in checks.items()},
        # No seconds anywhere, not even here: this article's whole cost section
        # is counted rather than timed, and a stray wall clock in the JSON is a
        # wall clock somebody will quote one day.
        "board_drift": drift,
        "sklearn": sklearn.__version__,
        "sklearn_previous": "1.8.0",
    },
}
path = OUT / "affinity.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB) in "
      f"{time.perf_counter() - t_start:.0f} s")
