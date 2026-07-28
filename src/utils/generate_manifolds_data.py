"""Data for the ATLAS 'Unrolling' article: Isomap, LLE and self-organising maps.

Everything in module 2.2 so far has approximated a row as a weighted sum of a
few atoms, which is a claim that the data lies near a flat slice of the space it
was measured in. This article is about what to do when it does not, and it is
built on the one dataset where the right answer is known exactly.

A Swiss roll is a rectangle rolled up. Because it is a developable surface, the
distance between two points measured ALONG the sheet has a closed form: the arc
length of the spiral between them, and the height between them, combined by
Pythagoras. So there is a true two-dimensional chart to compare against, and
every claim about "recovering the manifold" becomes a number instead of a
picture that looks about right.

Five blocks:

1. The roll, its exact geodesics, and how badly a straight line through the
   ambient space misrepresents them. This is the whole motivation, measured.
2. The neighbourhood graph as a function of k, and the short circuit: the exact
   k at which one edge jumps between two turns of the roll and takes the
   geodesic estimate down with it. Also the k below which the graph falls into
   pieces and Isomap has nothing to work with at all.
3. Isomap against LLE against PCA on the same points, scored four ways, plus
   the residual variance curve that puts a number on the intrinsic dimension.
4. The case Isomap is not built for: the same roll with a rectangular hole
   punched in it, where every geodesic has to detour and the flat chart Isomap
   assumes exists does not.
5. A self-organising map, which is the odd one out: it does not embed the data,
   it moves a grid of prototypes into the data and then reads positions off the
   grid. Quantisation error and topographic error, and the same recovery score
   as everyone else so the comparison is fair.

Plus the same three methods on 2,000 MNIST digits, where the answer is not
known and the only honest score is whether the two dimensions kept enough to
classify with.

Run from anywhere: `python src/utils/generate_manifolds_data.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.vision_data import mnist, subset            # noqa: E402

OUT = ROOT / "manifolds" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 19
rnd = lambda v, n=4: round(float(v), n)
arr = lambda a, n=4: [round(float(v), n) for v in np.asarray(a).ravel()]
mat = lambda A, n=4: [[round(float(v), n) for v in row] for row in np.asarray(A)]

N_ROLL = 500                      # the browser rebuilds every embedding at this size
KS = list(range(3, 31))
K_SHOW = 5                        # inside the window the sweep below measures

print("=" * 72)
print("ATLAS / manifolds: Isomap, LLE and the self-organising map")
print("=" * 72)


# ----------------------------------------------------------------- shared maths
def arclength(t):
    """Distance travelled along the spiral (t cos t, t sin t) from the origin.

    The speed of that curve is sqrt(1 + t^2), so the arc length integrates in
    closed form. This is the only reason the article can score anything: it is
    the true horizontal coordinate of the sheet, before anybody rolled it up.
    """
    t = np.asarray(t, dtype=float)
    return 0.5 * (t * np.sqrt(1 + t ** 2) + np.arcsinh(t))


def swiss_roll(n, seed, hole=None):
    rs = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rs.rand(n * 3))
    h = 21.0 * rs.rand(n * 3)
    s = arclength(t)
    keep = np.ones(len(t), dtype=bool)
    if hole is not None:
        s0, s1, h0, h1 = hole
        keep = ~((s > s0) & (s < s1) & (h > h0) & (h < h1))
    t, h, s = t[keep][:n], h[keep][:n], s[keep][:n]
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    return np.round(X, 4), np.round(np.column_stack([s, h]), 4)


def geodesic_matrix(chart):
    """True distance along the sheet: the flat distance in the unrolled chart."""
    return pairwise_distances(np.asarray(chart, dtype=float))


def knn_graph(X, k):
    """Symmetric k-nearest-neighbour graph with Euclidean edge weights.

    Symmetric because that is what every implementation does and because an
    asymmetric graph makes 'shortest path' depend on which end you start from,
    which is not a property a distance is allowed to have.
    """
    D = pairwise_distances(X)
    n = len(X)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = D[rows, cols]
    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return G.maximum(G.T), D


def graph_distances(G):
    return shortest_path(G, method="D", directed=False)


def procrustes_fit(A, B):
    """What is left of A after B has been rotated, reflected, scaled and shifted
    onto it, as a share of A's own scatter. Identical to the distances article's,
    deliberately: the two are scoring the same kind of claim."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    na, nb = np.sqrt((Ac ** 2).sum()), np.sqrt((Bc ** 2).sum())
    if na == 0 or nb == 0:
        return 1.0
    Ac, Bc = Ac / na, Bc / nb
    U, sv, Vt = np.linalg.svd(Bc.T @ Ac)
    return float(((Ac - sv.sum() * (Bc @ (U @ Vt))) ** 2).sum())


def double_centre(D):
    D2 = np.asarray(D, dtype=float) ** 2
    return -0.5 * (D2 - D2.mean(1, keepdims=True) - D2.mean(0, keepdims=True)
                   + D2.mean())


def seeded_block(n, k, seed):
    """The starting block for every iterative eigensolver here, written out as a
    linear congruential generator because the browser has to produce the SAME
    one. numpy's normal draw is not reproducible in JavaScript, and where the
    spectrum is nearly degenerate the starting block decides which of several
    equally valid answers comes back, so this is not a detail."""
    Q = np.empty((n, k))
    s = np.uint32(seed)
    a = np.uint32(1664525)
    c = np.uint32(1013904223)
    for i in range(n):
        for j in range(k):
            s = np.uint32(s * a + c)
            Q[i, j] = float(s) / 4294967296.0 - 0.5
    return Q


def top_eigen(B, k, iters=120, seed=7):
    """Block power iteration for the top k eigenpairs of a symmetric matrix.

    The browser cannot diagonalise a 500 by 500 matrix in a slider callback, and
    it does not have to: a two-dimensional embedding needs two eigenvectors.
    Written here in the same form the browser runs so the two can be compared.
    """
    Q, _ = np.linalg.qr(seeded_block(len(B), k, seed))
    for _ in range(iters):
        Q, _ = np.linalg.qr(B @ Q)
    T = Q.T @ B @ Q
    w, V = np.linalg.eigh(T)
    order = np.argsort(-w)
    return w[order], Q @ V[:, order]


def isomap_own(X, k, dim=2):
    G, _ = knn_graph(X, k)
    ncomp, labels = connected_components(G, directed=False)
    Dg = graph_distances(G)
    if ncomp > 1:
        return None, Dg, ncomp
    w, V = top_eigen(double_centre(Dg), dim)
    return V * np.sqrt(np.maximum(w, 0)), Dg, ncomp


def bottom_eigen(M, k, iters=80, shift=1e-6, seed=7):
    """Block INVERSE power iteration for the k smallest eigenpairs.

    Block power iteration on (cI - M) is the obvious way to reach the small end
    of a spectrum and it does not work: with c a Gershgorin bound the ratio
    between the two leading eigenvalues of cI - M is within a hair of 1, so 300
    iterations left LLE's answer 0.97 away from scikit-learn's, which reads
    exactly like a wrong formula and is a wrong solver. Factorise once and
    iterate on the inverse instead: the same three eigenvectors come back in
    tens of iterations because the ratios are now enormous.
    """
    n = len(M)
    A = np.asarray(M, dtype=float) + shift * np.eye(n)
    L = np.linalg.cholesky(A)
    Q, _ = np.linalg.qr(seeded_block(n, k, seed))
    for _ in range(iters):
        Y = np.linalg.solve(L.T, np.linalg.solve(L, Q))
        Q, _ = np.linalg.qr(Y)
    T = Q.T @ np.asarray(M, dtype=float) @ Q
    w, V = np.linalg.eigh(T)
    return w, Q @ V                      # ascending


def lle_matrix(X, k, reg=1e-3):
    """Roweis and Saul's local weights: what each point costs to rebuild out of
    its neighbours, assembled into M = (I-W)'(I-W)."""
    n = len(X)
    D = pairwise_distances(X)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    W = np.zeros((n, n))
    for i in range(n):
        Z = X[idx[i]] - X[i]
        C = Z @ Z.T
        C = C + np.eye(k) * reg * np.trace(C)
        w = np.linalg.solve(C, np.ones(k))
        W[i, idx[i]] = w / w.sum()
    M = np.eye(n) - W
    return M.T @ M


def lle_own(X, k, dim=2, reg=1e-3, exact=False):
    """The directions those weights leave alone. The first eigenvector is the
    constant one, at eigenvalue zero: it carries nothing and every
    implementation throws it away."""
    M = lle_matrix(X, k, reg)
    n = len(X)
    if exact:
        w, V = np.linalg.eigh(M)
    else:
        w, V = bottom_eigen(M, dim + 2)
    return V[:, 1:dim + 1] * np.sqrt(n)


def som_train(X, rows, cols, epochs, seed, sigma0=None, lr0=0.5, pca_init=False,
              want_init=False):
    """A Kohonen map, deterministic given the seed, so the browser can be asked
    to reproduce the prototypes rather than something that merely looks like
    them. Both sources of randomness (which points become the starting
    prototypes, and the order the samples arrive in) are drawn here and
    exported, because numpy's generator cannot be reproduced in JavaScript and a
    training run that starts somewhere else is a different training run."""
    rs = np.random.RandomState(seed)
    n, p = X.shape
    if pca_init:
        pc = PCA(n_components=2, random_state=seed).fit(X)
        a = np.linspace(-2, 2, rows) * np.sqrt(pc.explained_variance_[0])
        b = np.linspace(-2, 2, cols) * np.sqrt(pc.explained_variance_[1])
        P = (X.mean(0) + a[:, None, None] * pc.components_[0]
             + b[None, :, None] * pc.components_[1]).reshape(rows * cols, p)
        init_idx = []
    else:
        init_idx = rs.choice(n, rows * cols, replace=False)
        P = X[init_idx].astype(float).copy()
    gy, gx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    grid = np.column_stack([gy.ravel(), gx.ravel()]).astype(float)
    Gd2 = ((grid[:, None, :] - grid[None, :, :]) ** 2).sum(2)
    sigma0 = sigma0 if sigma0 is not None else max(rows, cols) / 2.0
    total = epochs * n
    step = 0
    orders = [rs.permutation(n) for _ in range(epochs)]
    for e in range(epochs):
        for i in orders[e]:
            frac = step / total
            sig = sigma0 * (0.5 / sigma0) ** frac
            lr = lr0 * (0.01 / lr0) ** frac
            b = int(((P - X[i]) ** 2).sum(1).argmin())
            hcoef = np.exp(-Gd2[b] / (2 * sig * sig))
            P += (lr * hcoef)[:, None] * (X[i] - P)
            step += 1
    if want_init:
        return P, grid, [o.tolist() for o in orders], [int(v) for v in init_idx]
    return P, grid, [o.tolist() for o in orders]


def som_scores(X, P, grid, rows, cols):
    D = pairwise_distances(X, P)
    order = np.argsort(D, axis=1)
    best, second = order[:, 0], order[:, 1]
    qe = float(D[np.arange(len(X)), best].mean())
    adj = (np.abs(grid[best] - grid[second]).sum(1) <= 1.0001)
    te = float(1.0 - adj.mean())
    return qe, te, best


# =========================================================== 1. the roll itself
print("\n" + "=" * 72)
print("a rectangle, rolled up")
X, CHART = swiss_roll(N_ROLL, SEED)
GEO = geodesic_matrix(CHART)
AMB = pairwise_distances(X)
iu = np.triu_indices(N_ROLL, 1)
ratio = GEO[iu] / np.maximum(AMB[iu], 1e-9)
worst = int(np.argmax(ratio))
amb_corr = float(np.corrcoef(AMB[iu], GEO[iu])[0, 1])
# the pair the ambient space lies about most: near in space, far along the sheet
wi, wj = iu[0][worst], iu[1][worst]
print(f"  {N_ROLL} points, chart {CHART[:,0].min():.1f} to {CHART[:,0].max():.1f} "
      f"along the sheet and 0 to {CHART[:,1].max():.1f} across it")
print(f"  straight line against distance along the sheet: correlation {amb_corr:.4f}")
print(f"  worst pair: {AMB[wi,wj]:.3f} apart in space, {GEO[wi,wj]:.3f} along the "
      f"sheet, a factor of {ratio[worst]:.1f}")

# PCA is the whole of module 2.2 so far, applied to a case it cannot win
pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X)
pca_disp = procrustes_fit(CHART, pca2)
pca_tw = float(trustworthiness(X, pca2, n_neighbors=12))
print(f"  PCA to two dimensions: disparity {pca_disp:.4f}, trustworthiness {pca_tw:.4f}")

# ==================================================== 2. the graph, and its k
print("\n" + "=" * 72)
print("the neighbourhood graph, and the edge that ruins it")
SHORT_FACTOR = 2.0
k_rows = []
first_short = None
first_connected = None
for k in KS:
    G, D = knn_graph(X, k)
    ncomp, _ = connected_components(G, directed=False)
    coo = G.tocoo()
    keep = coo.row < coo.col
    er, ec = coo.row[keep], coo.col[keep]
    edge_ratio = GEO[er, ec] / np.maximum(coo.data[keep], 1e-9)
    shorts = int((edge_ratio > SHORT_FACTOR).sum())
    Dg = graph_distances(G) if ncomp == 1 else None
    row = {"k": k, "components": int(ncomp), "edges": int(len(er)),
           "short_circuits": shorts,
           "worst_edge_ratio": rnd(float(edge_ratio.max()), 3)}
    if Dg is not None:
        row["geodesic_corr"] = rnd(float(np.corrcoef(Dg[iu], GEO[iu])[0, 1]), 5)
        row["geodesic_rmse"] = rnd(float(np.sqrt(((Dg[iu] - GEO[iu]) ** 2).mean())), 4)
        Z, _, _ = isomap_own(X, k)
        row["disparity"] = rnd(procrustes_fit(CHART, Z), 5)
        row["trustworthiness"] = rnd(float(trustworthiness(X, Z, n_neighbors=12)), 5)
        if first_connected is None:
            first_connected = k
    if shorts > 0 and first_short is None:
        first_short = k
    k_rows.append(row)
    print(f"  k {k:>2}: {ncomp:>2} component(s), {len(er):>5} edges, "
          f"{shorts:>3} short circuits (worst ratio {edge_ratio.max():6.2f})"
          + (f", geodesic corr {row['geodesic_corr']:.5f}, disparity {row['disparity']:.5f}"
             if Dg is not None else ""))
print(f"  the graph first connects at k = {first_connected}; the first short "
      f"circuit appears at k = {first_short}")

# ================================================ 3. three methods, four scores
print("\n" + "=" * 72)
print("Isomap, LLE and PCA on the same 500 points")
iso_own, Dg_show, _ = isomap_own(X, K_SHOW)
iso_sk = Isomap(n_neighbors=K_SHOW, n_components=2).fit_transform(X)
iso_agree = procrustes_fit(iso_sk, iso_own)
print(f"  own Isomap against scikit-learn's: disparity {iso_agree:.3e}")

# LLE gets its own k, and the reason is a measurement rather than a preference.
# The two eigenvectors LLE keeps are the second and third smallest of a matrix
# whose smallest eigenvalues, at small k, sit within a factor of a few of each
# other and of nothing: at k=5 they are around 1e-9, and a solver that has to
# separate them by iterating is separating numerical noise. Any k where the
# browser's iteration and an exact diagonalisation land in the same place is a k
# where the answer exists; below it, the answer is whatever the solver did.
lle_rows = []
lle_k = None
for k in [5, 6, 8, 10, 12, 16, 20, 24]:
    M = lle_matrix(X, k)
    w_ex, V_ex = np.linalg.eigh(M)
    Ze = V_ex[:, 1:3] * np.sqrt(N_ROLL)
    Zi = lle_own(X, k)
    Zsk = LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                 random_state=SEED, reg=1e-3).fit_transform(X)
    agree_iter = procrustes_fit(Ze, Zi)
    row = {
        "k": k,
        "eig2": float(f"{w_ex[1]:.3e}"), "eig3": float(f"{w_ex[2]:.3e}"),
        "eig4": float(f"{w_ex[3]:.3e}"),
        "gap": rnd(float(w_ex[3] / max(w_ex[2], 1e-30)), 3),
        "iter_vs_exact": float(f"{agree_iter:.3e}"),
        "sk_vs_exact": float(f"{procrustes_fit(Zsk, Ze):.3e}"),
        "disparity": rnd(procrustes_fit(CHART, Ze), 5),
        "trustworthiness": rnd(float(trustworthiness(X, Ze, n_neighbors=12)), 5),
    }
    lle_rows.append(row)
    # A disparity of 1e-5 between two solvers means the two configurations agree
    # to about three parts in a thousand, which is determined by any standard
    # this article applies to anything. Below that k the answer is not merely
    # imprecise: at k=5 the two land half a configuration apart.
    if lle_k is None and agree_iter < 1e-5:
        lle_k = k
    print(f"  LLE k={k:>2}: bottom eigenvalues {row['eig2']:.2e} {row['eig3']:.2e} "
          f"{row['eig4']:.2e}; iteration against exact {row['iter_vs_exact']:.1e}, "
          f"scikit-learn against exact {row['sk_vs_exact']:.1e}; "
          f"disparity {row['disparity']:.5f}")
lle_best = min(lle_rows, key=lambda r: r["disparity"])
print(f"  LLE's own best k by disparity is {lle_best['k']} "
      f"({lle_best['disparity']:.5f}); the smallest k where the answer is "
      f"determined at all is {lle_k}")
lle_own_z = lle_own(X, lle_k)
lle_sk = LocallyLinearEmbedding(n_neighbors=lle_k, n_components=2,
                                random_state=SEED, reg=1e-3).fit_transform(X)
lle_agree = procrustes_fit(lle_sk, lle_own_z)
print(f"  own LLE at k={lle_k} against scikit-learn's: disparity {lle_agree:.3e}")

methods = {}
for name, Z in [("pca", pca2), ("isomap", iso_own), ("lle", lle_own_z)]:
    Dz = pairwise_distances(Z)
    methods[name] = {
        "disparity": rnd(procrustes_fit(CHART, Z), 5),
        "trustworthiness": rnd(float(trustworthiness(X, Z, n_neighbors=12)), 5),
        "geodesic_corr": rnd(float(np.corrcoef(Dz[iu], GEO[iu])[0, 1]), 5),
        "coords": mat(Z, 4),
    }
    print(f"  {name:<7} disparity {methods[name]['disparity']:.5f}  "
          f"trustworthiness {methods[name]['trustworthiness']:.5f}  "
          f"geodesic correlation {methods[name]['geodesic_corr']:.5f}")

# residual variance against dimension: the elbow is the intrinsic dimension
resid = []
Bg = double_centre(Dg_show)
wg, Vg = np.linalg.eigh(Bg)
og = np.argsort(-wg)
wg, Vg = wg[og], Vg[:, og]
for d in range(1, 8):
    Zd = Vg[:, :d] * np.sqrt(np.maximum(wg[:d], 0))
    Dd = pairwise_distances(Zd)
    r = float(np.corrcoef(Dd[iu], Dg_show[iu])[0, 1])
    resid.append({"dim": d, "residual": rnd(1 - r ** 2, 5)})
# The elbow is not where the drop is biggest, which is always the first step and
# would call every dataset one-dimensional. It is the last dimension whose
# arrival still buys a tenth of what the first one bought.
drops = [resid[i]["residual"] - resid[i + 1]["residual"] for i in range(len(resid) - 1)]
elbow = 1
for i, d in enumerate(drops):
    if d >= 0.1 * drops[0]:
        elbow = i + 2
# built outside the f-string: a Python 3.11 f-string cannot nest the same quote
resid_txt = ", ".join(str(r["dim"]) + ":" + str(r["residual"]) for r in resid)
print(f"  residual variance by dimension: {resid_txt}")
print(f"  biggest drop after dimension {elbow}")

# ====================================================== 4. the roll with a hole
print("\n" + "=" * 72)
print("the same roll with a hole punched in it, which Isomap is not built for")
HOLE = (30.0, 55.0, 5.0, 16.0)
XH, CHARTH = swiss_roll(N_ROLL, SEED, hole=HOLE)
GEOH = geodesic_matrix(CHARTH)          # still the flat distance: the wrong answer
isoh, Dgh, ncomph = isomap_own(XH, K_SHOW)
hole_disp = procrustes_fit(CHARTH, isoh)
hole_tw = float(trustworthiness(XH, isoh, n_neighbors=12))
iuh = np.triu_indices(len(XH), 1)
# A graph path is a chain of chords across a curved sheet, so it undershoots the
# true arc by a fixed amount whatever the sheet looks like. Measuring the detour
# means subtracting that bias, which the intact roll supplies at the same k and
# the same sample size: one thing changed, everything else held.
ratio_intact = float(np.mean(Dg_show[iu] / np.maximum(GEO[iu], 1e-9)))
ratio_hole = float(np.mean(Dgh[iuh] / np.maximum(GEOH[iuh], 1e-9)))
detour = ratio_hole - ratio_intact
# a pair has to detour when its graph path is more than a tenth longer than the
# intact roll's own overhead would explain
thresh = ratio_intact * 1.1
crossing = int(((Dgh[iuh] / np.maximum(GEOH[iuh], 1e-9)) > thresh).sum())
print(f"  disparity {hole_disp:.5f} against {methods['isomap']['disparity']:.5f} "
      f"on the intact roll, trustworthiness {hole_tw:.5f}")
print(f"  graph path over flat distance: {ratio_intact:.4f} on the intact roll, "
      f"{ratio_hole:.4f} with the hole, a detour of {detour:+.4f}")
print(f"  {crossing} of {len(iuh[0])} pairs walk more than {thresh:.3f} times "
      f"the flat distance, which the intact roll never asks of anybody")

# ============================================================ 5. a moving grid
print("\n" + "=" * 72)
print("a self-organising map: a grid pulled into the data")
# The sheet is about four times as long as it is wide, and a grid has a fixed
# shape that the reader has to choose BEFORE seeing the answer. So the aspect
# ratio is swept at a nearly constant number of cells, which isolates the one
# thing being asked about: does guessing the shape wrong cost anything.
SOM_SHAPES = [(12, 12), (9, 16), (7, 21), (6, 24), (4, 36)]
SOM_EPOCHS = 12
Xs = np.round((X - X.mean(0)) / X.std(0).max(), 4)
sheet_aspect = float((CHART[:, 0].max() - CHART[:, 0].min())
                     / (CHART[:, 1].max() - CHART[:, 1].min()))
som_rows_out = []
for r, c in SOM_SHAPES:
    Pg, gg, _ = som_train(Xs, r, c, SOM_EPOCHS, SEED)
    q, t_, b = som_scores(Xs, Pg, gg, r, c)
    som_rows_out.append({
        "rows": r, "cols": c, "cells": r * c, "aspect": rnd(c / r, 3),
        "quantisation": rnd(q, 4), "topographic": rnd(t_, 4),
        "disparity": rnd(procrustes_fit(CHART, gg[b]), 5),
        "trustworthiness": rnd(float(trustworthiness(Xs, gg[b], n_neighbors=12)), 5),
        "ties": int(len(Xs) - len(np.unique(b))),
    })
    print(f"    {r:>2}x{c:<2} ({r*c:>3} cells, aspect {c/r:5.2f}): quantisation "
          f"{q:.4f}, topographic {t_:.4f}, disparity "
          f"{som_rows_out[-1]['disparity']:.5f}, trustworthiness "
          f"{som_rows_out[-1]['trustworthiness']:.5f}")
# Before publishing "a self-organising map does not unroll a Swiss roll", try to
# make it. Four more knobs, each pushed the way that would help if the schedule
# were the problem: a principal component initialisation instead of random
# prototypes, three times the epochs, and twice the starting neighbourhood.
print("\n  trying to make it work, which is the only way to publish that it does not:")
attempts = []
for label, kw in [
    ("random init, 12 epochs", dict(epochs=12, sigma0=6.0)),
    ("random init, 40 epochs", dict(epochs=40, sigma0=6.0)),
    ("random init, wide start", dict(epochs=40, sigma0=12.0)),
    ("pca init, 12 epochs", dict(epochs=12, sigma0=6.0, pca_init=True)),
    ("pca init, 40 epochs", dict(epochs=40, sigma0=12.0, pca_init=True)),
]:
    for r, c in [(12, 12), (6, 24)]:
        Pa, ga, _ = som_train(Xs, r, c, kw["epochs"], SEED, sigma0=kw["sigma0"],
                              pca_init=kw.get("pca_init", False))
        q, t_, b = som_scores(Xs, Pa, ga, r, c)
        cc = abs(float(np.corrcoef(ga[b][:, 1], CHART[:, 0])[0, 1]))
        attempts.append({"label": label, "rows": r, "cols": c,
                         "epochs": kw["epochs"], "sigma0": kw["sigma0"],
                         "quantisation": rnd(q, 4), "topographic": rnd(t_, 4),
                         "disparity": rnd(procrustes_fit(CHART, ga[b]), 5),
                         "long_axis_corr": rnd(cc, 4)})
        print(f"    {label:<24} {r:>2}x{c:<2}: quantisation {q:.4f}, topographic "
              f"{t_:.4f}, disparity {attempts[-1]['disparity']:.5f}, long axis "
              f"|corr| {cc:.4f}")
best_attempt = min(attempts, key=lambda a: a["disparity"])
print(f"  best of {len(attempts)} attempts: {best_attempt['label']} at "
      f"{best_attempt['rows']}x{best_attempt['cols']}, disparity "
      f"{best_attempt['disparity']} against Isomap's "
      f"{methods['isomap']['disparity']}")

best_shape = min(som_rows_out, key=lambda r: r["disparity"])
SOM_ROWS, SOM_COLS = best_shape["rows"], best_shape["cols"]
print(f"  the sheet's own aspect ratio is {sheet_aspect:.2f}; the grid that "
      f"recovers it best is {SOM_ROWS}x{SOM_COLS}, aspect {best_shape['aspect']}")
P, grid, orders, som_init = som_train(Xs, SOM_ROWS, SOM_COLS, SOM_EPOCHS,
                                      SEED, want_init=True)
qe, te, best = som_scores(Xs, P, grid, SOM_ROWS, SOM_COLS)
som_z = grid[best] + 0.0
som_disp = procrustes_fit(CHART, som_z)
som_tw = float(trustworthiness(Xs, som_z, n_neighbors=12))
# and the plainest check that the grid really did unroll: does walking along its
# long axis walk along the sheet
som_corr = float(np.corrcoef(grid[best][:, 1], CHART[:, 0])[0, 1])
print(f"  {SOM_ROWS}x{SOM_COLS} grid, {SOM_EPOCHS} epochs: quantisation error "
      f"{qe:.4f}, topographic error {te:.4f}")
print(f"  grid position against the true chart: disparity {som_disp:.5f}, "
      f"trustworthiness {som_tw:.5f}, and the grid's long axis tracks the "
      f"sheet's at correlation {som_corr:.4f} "
      f"({len(Xs) - len(np.unique(best))} points share a cell with another)")

# ================================================= 6. and the same on real data
print("\n" + "=" * 72)
print("2,000 MNIST digits, where nobody knows the right answer")
Xtr, ytr, _, _ = mnist()
Xd, yd = subset(Xtr, ytr, 2000, seed=SEED)
DIG = Xd.reshape(len(Xd), -1) / 255.0
rs = np.random.RandomState(SEED)
perm = rs.permutation(len(DIG))
tr, te_idx = perm[:1400], perm[1400:]
digit_rows = []
for name, Z in [
    ("pca", PCA(n_components=2, random_state=SEED).fit_transform(DIG)),
    ("isomap", Isomap(n_neighbors=10, n_components=2).fit_transform(DIG)),
    ("lle", LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                   random_state=SEED).fit_transform(DIG)),
]:
    knn = KNeighborsClassifier(n_neighbors=5).fit(Z[tr], yd[tr])
    acc = float((knn.predict(Z[te_idx]) == yd[te_idx]).mean())
    tw = float(trustworthiness(DIG, Z, n_neighbors=12))
    digit_rows.append({"method": name, "accuracy": rnd(acc, 4),
                       "trustworthiness": rnd(tw, 5),
                       "coords": mat(Z / np.abs(Z).max(), 4)})
    print(f"  {name:<7} 5-nn accuracy on the 2-D map {acc:.4f}, "
          f"trustworthiness {tw:.5f}")
knn_full = KNeighborsClassifier(n_neighbors=5).fit(DIG[tr], yd[tr])
acc_full = float((knn_full.predict(DIG[te_idx]) == yd[te_idx]).mean())
print(f"  the same classifier on all 784 pixels: {acc_full:.4f}")

# ===================================================================== write it
out = {
    "meta": {
        "seed": SEED, "n": N_ROLL, "k_show": K_SHOW,
        "short_factor": SHORT_FACTOR,
        "sklearn": __import__("sklearn").__version__,
        "note": "the browser rebuilds the graph, the shortest paths, the "
                "double centring, the block power iteration, the LLE weights "
                "and the SOM training loop itself and checks against the "
                "references here",
    },
    "roll": {
        "points": mat(X, 4),
        "chart": mat(CHART, 4),
        "scaled": mat(Xs, 4),
        "ambient_corr": rnd(amb_corr, 5),
        "worst_pair": [int(wi), int(wj)],
        "worst_ambient": rnd(AMB[wi, wj], 4),
        "worst_geodesic": rnd(GEO[wi, wj], 4),
        "worst_ratio": rnd(ratio[worst], 3),
        "geo_mean": rnd(float(GEO[iu].mean()), 4),
        "amb_mean": rnd(float(AMB[iu].mean()), 4),
        "chart_span": [rnd(CHART[:, 0].min(), 3), rnd(CHART[:, 0].max(), 3),
                       rnd(CHART[:, 1].min(), 3), rnd(CHART[:, 1].max(), 3)],
    },
    "graph": {
        "ks": KS, "rows": k_rows,
        # The k dial opens well past the cliff on purpose: the section is about
        # the cliff, and a widget whose opening state is already the answer
        # leaves its own "snap to the best" button with nothing to do.
        "opens_at": 12,
        "first_connected": first_connected,
        "first_short_circuit": first_short,
    },
    "methods": methods,
    "residual": {"rows": resid, "elbow": elbow},
    "lle": {"k": lle_k, "rows": lle_rows, "best_k": lle_best["k"],
            "best_disparity": lle_best["disparity"]},
    "checks": {
        "isomap_vs_sklearn": float(f"{iso_agree:.3e}"),
        "lle_vs_sklearn": float(f"{lle_agree:.3e}"),
    },
    "hole": {
        "box": list(HOLE),
        "points": mat(XH, 4),
        "chart": mat(CHARTH, 4),
        "coords": mat(isoh, 4),
        "disparity": rnd(hole_disp, 5),
        "trustworthiness": rnd(hole_tw, 5),
        "ratio_intact": rnd(ratio_intact, 4),
        "ratio_hole": rnd(ratio_hole, 4),
        "detour": rnd(detour, 4),
        "threshold": rnd(thresh, 4),
        "crossing_pairs": crossing,
        "total_pairs": int(len(iuh[0])),
        "components": int(ncomph),
    },
    "som": {
        "rows": SOM_ROWS, "cols": SOM_COLS, "epochs": SOM_EPOCHS,
        "prototypes": mat(P, 4),
        "orders": orders,
        "init_index": som_init,
        "sigma0": rnd(max(SOM_ROWS, SOM_COLS) / 2.0, 4),
        "lr0": 0.5,
        "quantisation": rnd(qe, 4),
        "topographic": rnd(te, 4),
        "disparity": rnd(som_disp, 5),
        "trustworthiness": rnd(som_tw, 5),
        "long_axis_corr": rnd(som_corr, 4),
        "ties": int(len(Xs) - len(np.unique(best))),
        "sheet_aspect": rnd(sheet_aspect, 3),
        "by_shape": som_rows_out,
        "shapes": [[r, c] for r, c in SOM_SHAPES],
        "attempts": attempts,
        "best_attempt": best_attempt,
    },
    "digits": {
        "n": int(len(DIG)), "train": int(len(tr)), "test": int(len(te_idx)),
        "k": 10, "knn": 5,
        "rows": digit_rows,
        "labels": [int(v) for v in yd],
        "full_accuracy": rnd(acc_full, 4),
    },
}
path = OUT / "manifolds.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size/1024:.1f} kB)")
