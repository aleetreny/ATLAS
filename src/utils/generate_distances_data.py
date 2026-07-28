"""Data for the ATLAS 'Distances' article: multidimensional scaling and factor
analysis, the two things that happen when you stop trusting the data matrix.

The previous article factorised X itself. This one throws X away twice, in two
different directions, and measures what survives each time.

* Throw away the coordinates and keep the pairwise distances. Classical MDS
  rebuilds a configuration from the distance matrix alone, and the
  configuration it rebuilds is PCA's, to machine precision. That is not a
  resemblance, it is an identity, and it is checked here before it is claimed.
* Throw away the belief that every measured variance is signal. Factor analysis
  splits each variable's variance into a part shared with the others and a part
  that belongs to that variable alone, which makes it a generative model with a
  likelihood, unlike PCA, which is a rotation.

Five blocks:

1. The 178 wines as a distance matrix. Double centring, the eigendecomposition,
   and the identity against PCA. The raw matrix is asserted to be the SAME one
   the directions article ships, coordinate for coordinate, so "the same wine"
   is a check rather than an intention.
2. Six distances over those wines, and how much of each one is not Euclidean at
   all, measured as the share of eigenvalue mass that comes back negative. Zero
   is a theorem for the Euclidean case and a measurement here.
3. Metric against non-metric MDS under a monotone distortion of the distances.
   Non-metric MDS only reads the ORDER of the dissimilarities, so a distortion
   that leaves the order alone should cost it nothing. That is the claim, and
   the sweep over the distortion exponent is the test.
4. Factor analysis against PCA on a designed model where the truth is known,
   with a control arm whose only difference is that the per-variable noise is
   equal instead of unequal. With equal noise the two methods must agree (it is
   a theorem); the gap in the other arm is therefore the price of one
   assumption, isolated.
5. A dissimilarity nobody has coordinates for: how often a k-nearest-neighbour
   classifier confuses each pair of MNIST digit classes. Mapping that is the
   thing MDS can do and PCA cannot, because there is no data matrix to rotate.

Run from anywhere: `python src/utils/generate_distances_data.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import smacof as sk_smacof
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.vision_data import mnist, subset            # noqa: E402

OUT = ROOT / "distances" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 19
rnd = lambda v, n=4: round(float(v), n)
arr = lambda a, n=4: [round(float(v), n) for v in np.asarray(a).ravel()]
mat = lambda A, n=4: [[round(float(v), n) for v in row] for row in np.asarray(A)]

print("=" * 72)
print("ATLAS / distances: MDS and factor analysis")
print("=" * 72)


# ----------------------------------------------------------------- shared maths
def double_centre(D):
    """B = -0.5 J D^2 J, the matrix whose eigenvectors are classical MDS.

    Written out rather than called from a library because the browser runs the
    same three lines and the scrolly draws them one at a time.
    """
    D2 = np.asarray(D, dtype=float) ** 2
    rowm = D2.mean(1, keepdims=True)
    colm = D2.mean(0, keepdims=True)
    return -0.5 * (D2 - rowm - colm + D2.mean())


def lingoes(D):
    """The smallest constant that has to be added to every squared distance
    before the set can be embedded in a Euclidean space at all.

    Adding c to every off-diagonal squared distance adds c/2 to every
    eigenvalue of B except the one the centring pins at zero, so the constant
    is exactly minus twice the most negative eigenvalue. One line of algebra,
    and the check below adds it and measures that nothing negative is left.
    """
    w, _ = eig_desc(double_centre(D))
    c = float(max(0.0, -2.0 * w.min()))
    D2 = np.asarray(D, dtype=float) ** 2 + c
    np.fill_diagonal(D2, 0.0)
    w2, _ = eig_desc(double_centre(np.sqrt(D2)))
    left = float(np.abs(np.minimum(w2, 0)).sum() / np.abs(w2).sum())
    return c, left


def eig_desc(A):
    """Eigenvalues and eigenvectors of a symmetric matrix, largest first."""
    w, V = np.linalg.eigh(np.asarray(A, dtype=float))
    order = np.argsort(-w)
    return w[order], V[:, order]


def classical_mds(D, k):
    w, V = eig_desc(double_centre(D))
    pos = np.maximum(w[:k], 0.0)
    return V[:, :k] * np.sqrt(pos), w


def fix_signs(A):
    """A configuration is only defined up to the sign of each axis, so pin the
    sign by the entry of largest magnitude, the same way the PCA article does."""
    A = np.asarray(A, dtype=float).copy()
    for j in range(A.shape[1]):
        col = A[:, j]
        if col[np.abs(col).argmax()] < 0:
            A[:, j] = -col
    return A


def pdist_matrix(X, metric="euclidean"):
    D = pairwise_distances(X, metric=metric)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def stress1(D, Z):
    """Kruskal's stress-1: the residual between the target dissimilarities and
    the distances actually realised in the map, as a share of the map's size."""
    d = pdist_matrix(Z)
    iu = np.triu_indices_from(D, 1)
    num = ((D[iu] - d[iu]) ** 2).sum()
    den = (d[iu] ** 2).sum()
    return float(np.sqrt(num / den))


def procrustes_fit(A, B):
    """Rotate, reflect, scale and shift B onto A, and report what is left.

    Returned as a share of A's own scatter, so 0 means the two configurations
    are the same picture and 1 means B carries nothing about A.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    normA = np.sqrt((Ac ** 2).sum())
    normB = np.sqrt((Bc ** 2).sum())
    if normA == 0 or normB == 0:
        return 1.0
    Ac = Ac / normA
    Bc = Bc / normB
    U, s, Vt = np.linalg.svd(Bc.T @ Ac)
    R = U @ Vt
    scale = s.sum()
    return float(((Ac - scale * (Bc @ R)) ** 2).sum())


def pav(y, w):
    """Pool adjacent violators: the same isotonic regression the calibration
    article used on probabilities, here on distances. Written out because the
    browser has to run exactly this and because a library call would hide the
    only interesting fact about non-metric MDS."""
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()
    n = len(y)
    vals = y.copy()
    wts = w.copy()
    size = np.ones(n, dtype=int)
    k = 0
    for i in range(n):
        vals[k] = y[i]
        wts[k] = w[i]
        size[k] = 1
        k += 1
        while k > 1 and vals[k - 2] > vals[k - 1]:
            tot = wts[k - 2] + wts[k - 1]
            vals[k - 2] = (wts[k - 2] * vals[k - 2] + wts[k - 1] * vals[k - 1]) / tot
            wts[k - 2] = tot
            size[k - 2] += size[k - 1]
            k -= 1
    out = np.empty(n)
    j = 0
    for i in range(k):
        out[j:j + size[i]] = vals[i]
        j += size[i]
    return out


def smacof(D, Z0, n_iter, nonmetric=False, order=None):
    """One Guttman transform per iteration, and nothing else.

    Deterministic given the starting configuration, which is why the browser can
    be asked to reproduce the answer to eight decimals rather than to within
    some hand-waved tolerance. `nonmetric` swaps the target dissimilarities for
    the isotonic regression of the current distances onto the ORDER of the
    originals, which is Kruskal's idea inside Guttman's update.
    """
    n = len(D)
    iu = np.triu_indices(n, 1)
    Z = np.array(Z0, dtype=float)
    target = D.copy()
    trace = []
    for _ in range(n_iter):
        d = pdist_matrix(Z)
        if nonmetric:
            dv = d[iu][order]
            hat = pav(dv, np.ones(len(dv)))
            # keep the scale fixed, or the whole configuration collapses to a
            # point and reports a stress of zero
            scale = np.sqrt((d[iu] ** 2).sum() / max((hat ** 2).sum(), 1e-12))
            hat = hat * scale
            flat = np.empty(len(dv))
            flat[order] = hat
            target = np.zeros_like(D)
            target[iu] = flat
            target = target + target.T
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d > 1e-12, target / np.maximum(d, 1e-12), 0.0)
        B = -ratio
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(1))
        Z = (B @ Z) / n
        trace.append(stress1(target, Z))
    return Z, trace, target


def subspace_gap(A, B):
    """1 minus the mean squared cosine of the principal angles between the two
    column spaces: 0 when they are the same plane, 1 when they are orthogonal.
    Comparing loadings entry by entry would be meaningless, because a factor
    model is only ever identified up to a rotation of its factors."""
    Qa = np.linalg.qr(np.asarray(A, dtype=float))[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=float))[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(1.0 - (s ** 2).mean())


def varimax(L, iters=200, tol=1e-12):
    """Kaiser's rotation. It cannot change the fit, which is the point."""
    L = np.asarray(L, dtype=float).copy()
    p, k = L.shape
    R = np.eye(k)
    last = 0.0
    for _ in range(iters):
        Lam = L @ R
        U, s, Vt = np.linalg.svd(
            L.T @ (Lam ** 3 - Lam @ np.diag((Lam ** 2).sum(0)) / p))
        R = U @ Vt
        if abs(s.sum() - last) < tol * max(s.sum(), 1e-12):
            break
        last = s.sum()
    return L @ R, R


def simplicity(L):
    """Varimax's own criterion: the variance of the squared loadings within each
    column. High means every variable belongs to one factor and not the others,
    which is what 'interpretable' means when anyone says it about a factor."""
    L2 = np.asarray(L, dtype=float) ** 2
    return float(sum(((c ** 2).mean() - (c.mean()) ** 2) for c in L2.T))


def fa_em(S, k, iters=5000, tol=1e-12):
    """Rubin and Thayer's EM for the factor model, on a covariance matrix.

    Written out for the same reason as the double centring: the browser runs
    this loop, and a fit the reader cannot check is a fit the reader has to take
    on trust. Checked against scikit-learn's FactorAnalysis below.
    """
    S = np.asarray(S, dtype=float)
    p = S.shape[0]
    psi = np.diag(S).astype(float).copy()
    w, V = eig_desc(S)
    L = V[:, :k] * np.sqrt(np.maximum(w[:k], 1e-6))
    prev = -np.inf
    ll = prev
    used = iters
    for it in range(iters):
        inv_psi = 1.0 / np.maximum(psi, 1e-10)
        M = np.eye(k) + (L.T * inv_psi) @ L
        Minv = np.linalg.inv(M)
        beta = Minv @ (L.T * inv_psi)                       # k x p
        SB = S @ beta.T                                     # p x k
        Ezz = Minv + beta @ S @ beta.T                      # k x k
        L = SB @ np.linalg.inv(Ezz)
        psi = np.maximum(np.diag(S) - np.einsum("ij,ji->i", L, beta @ S), 1e-8)
        Sigma = L @ L.T + np.diag(psi)
        sign, logdet = np.linalg.slogdet(Sigma)
        ll = float(-0.5 * (logdet + np.trace(np.linalg.solve(Sigma, S))))
        if abs(ll - prev) < tol * max(abs(ll), 1.0):
            used = it + 1
            break
        prev = ll
    return L, psi, ll, used


def loglik(S, Sigma, n):
    """Gaussian log likelihood of a covariance model, per the usual constant."""
    p = S.shape[0]
    sign, logdet = np.linalg.slogdet(Sigma)
    return float(-0.5 * n * (p * np.log(2 * np.pi) + logdet
                             + np.trace(np.linalg.solve(Sigma, S))))


# ============================================================ 1. the wine again
print("\n" + "=" * 72)
print("the 178 wines, as a distance matrix")
wine = load_wine()
NAMES = list(wine.feature_names)
XW = np.round(wine.data, 3)
YW = wine.target

# Reuse is demonstrated, not declared: the matrix this article standardises is
# asserted to be the identical rounded matrix the directions article shipped.
prev = json.loads((ROOT / "directions" / "data" / "directions.json").read_text("utf-8"))
prev_matrix = np.array(prev["spectrum"]["matrix"], dtype=float)
assert prev_matrix.shape == XW.shape, "the wine changed shape between articles"
worst_reuse = float(np.abs(prev_matrix - XW).max())
assert worst_reuse == 0.0, f"the wine moved by {worst_reuse}"
print(f"  the raw matrix matches directions.json exactly ({XW.shape[0]} rows, "
      f"{XW.shape[1]} columns, worst difference {worst_reuse:.0e})")

# Standardised with the population sd, rounded once, and measured afterwards:
# the browser only ever has these rounded numbers.
XS = np.round((XW - XW.mean(0)) / XW.std(0), 4)
N = len(XS)
D_EUC = pdist_matrix(XS)

pca = PCA(n_components=13, random_state=SEED).fit(XS)
scores = pca.transform(XS)
coords, eigs = classical_mds(D_EUC, 13)
gap_coords = float(np.abs(fix_signs(coords) - fix_signs(scores)).max())
# The eigenvalues of the double centred matrix are n times PCA's variances,
# which is the same statement seen from the other side.
gap_eigs = float(np.abs(eigs[:13] / N - pca.explained_variance_ * (N - 1) / N).max())
print(f"  classical MDS against PCA scores: worst coordinate differs by {gap_coords:.3e}")
print(f"  eigenvalues over n against PCA variances: worst {gap_eigs:.3e}")
neg_euclid = float(np.abs(np.minimum(eigs, 0)).sum() / np.abs(eigs).sum())
print(f"  negative eigenvalue mass of a Euclidean distance: {neg_euclid:.3e}")

# a small corner of the matrix, for the scrolly to show as numbers
SMALL = [int(np.where(YW == c)[0][j]) for c in range(3) for j in range(2)]
small_D = D_EUC[np.ix_(SMALL, SMALL)]
small_B = double_centre(small_D)

wine_out = {
    "n": N,
    "features": NAMES,
    "matrix": mat(XS, 4),
    "classes": [int(v) for v in YW],
    "class_names": ["cultivar 1", "cultivar 2", "cultivar 3"],
    "eigenvalues": arr(eigs, 5),
    "evr": arr(np.maximum(eigs, 0) / np.maximum(eigs, 0).sum(), 6),
    "mds_pca_gap": float(f"{gap_coords:.3e}"),
    "eig_variance_gap": float(f"{gap_eigs:.3e}"),
    "negative_mass": float(f"{neg_euclid:.3e}"),
    "small_index": SMALL,
    "small_labels": [f"{'ABC'[YW[i]]}{i}" for i in SMALL],
    "small_D": mat(small_D, 3),
    "small_B": mat(small_B, 3),
    "d_max": rnd(D_EUC.max(), 4),
    "d_mean": rnd(D_EUC[np.triu_indices(N, 1)].mean(), 4),
}

# ====================================================== 2. six kinds of distance
print("\n" + "=" * 72)
print("which of these dissimilarities is a distance in some Euclidean space")
METRICS = [
    ("euclidean", "euclidean", "straight line, the one PCA already answers"),
    ("cityblock", "cityblock", "add the differences instead of squaring them"),
    ("sqrt cityblock", None, "the same numbers under a square root"),
    ("chebyshev", "chebyshev", "the single worst measurement decides"),
    ("cosine", "cosine", "angle between bottles, ignoring how big they are"),
    ("correlation", "correlation", "shape of the profile, ignoring level too"),
]
metric_rows = []
for label, name, blurb in METRICS:
    if name is None:
        Dm = np.sqrt(pdist_matrix(XS, "cityblock"))
    else:
        Dm = pdist_matrix(XS, name)
    w, V = eig_desc(double_centre(Dm))
    neg = float(np.abs(np.minimum(w, 0)).sum() / np.abs(w).sum())
    Z2 = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    cail, left = lingoes(Dm)
    metric_rows.append({
        "key": label,
        "blurb": blurb,
        "negative_mass": float(f"{neg:.3e}"),
        "neg_count": int((w < -1e-9).sum()),
        "most_negative": rnd(w.min(), 4),
        "largest": rnd(w.max(), 4),
        "evr2": rnd(float(np.maximum(w[:2], 0).sum() / np.maximum(w, 0).sum()), 4),
        "stress2": rnd(stress1(Dm, Z2), 4),
        "lingoes": rnd(cail, 4),
        "lingoes_left": float(f"{left:.3e}"),
        "d_mean": rnd(float(Dm[np.triu_indices(len(Dm), 1)].mean()), 4),
        "eigenvalues": arr(w[:12], 4),
        "eig_last": arr(w[-4:], 4),
    })
    print(f"  {label:<15} negative mass {neg:9.3e}  ({int((w < -1e-9).sum()):>3} "
          f"negative eigenvalues)  2-D stress {metric_rows[-1]['stress2']:.4f}  "
          f"Lingoes constant {cail:.4f} (leaves {left:.1e})")

# ================================================ 3. metric against non-metric
print("\n" + "=" * 72)
print("a monotone distortion of the distances, and who notices")
TRUTH = np.round(scores[:, :2], 4)
D_TRUTH = pdist_matrix(TRUTH)
D_TRUTH = D_TRUTH / D_TRUTH.max()
rs = np.random.RandomState(SEED)
INIT = np.round(rs.normal(scale=0.2, size=(N, 2)), 4)
SM_ITERS = 200
GAMMAS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0]
iu = np.triu_indices(N, 1)
ORDER = np.argsort(D_TRUTH[iu], kind="stable")

gamma_rows = []
trace_one = []
for g in GAMMAS:
    Dg = D_TRUTH ** g
    Dg = Dg / Dg.max()
    Zm, trace_m, _ = smacof(Dg, INIT, SM_ITERS)
    if g == 1.0:
        trace_one = trace_m
    Zn, trace_n, hat = smacof(Dg, INIT, SM_ITERS, nonmetric=True, order=ORDER)
    row = {
        "gamma": g,
        "metric_stress": rnd(trace_m[-1], 5),
        "nonmetric_stress": rnd(trace_n[-1], 5),
        "metric_disparity": rnd(procrustes_fit(TRUTH, Zm), 5),
        "nonmetric_disparity": rnd(procrustes_fit(TRUTH, Zn), 5),
    }
    gamma_rows.append(row)
    print(f"  gamma {g:<4}  metric: stress {row['metric_stress']:.5f} "
          f"disparity {row['metric_disparity']:.5f}   "
          f"non-metric: stress {row['nonmetric_stress']:.5f} "
          f"disparity {row['nonmetric_disparity']:.5f}")

# A monotone distortion cannot move an ordering, so the non-metric column above
# is the SAME map nine times over. That is the theorem, and stating it without
# the other half would be selling it: noise moves the ordering, and the moment
# it does, reading only the ordering stops being free. So measure that too.
print("\n  and what happens when the distortion is not monotone, but noise:")
noise_rows = []
for eps in [0.0, 0.02, 0.05, 0.1, 0.2]:
    rn = np.random.RandomState(77)
    Dn = D_TRUTH + rn.normal(0, eps, D_TRUTH.shape)
    Dn = np.abs(0.5 * (Dn + Dn.T))
    np.fill_diagonal(Dn, 0.0)
    Dn = Dn / Dn.max()
    on = np.argsort(Dn[iu], kind="stable")
    Zm, tm, _ = smacof(Dn, INIT, SM_ITERS)
    Zn, tn, _ = smacof(Dn, INIT, SM_ITERS, nonmetric=True, order=on)
    # how badly the noise shuffled the ordering, as a rank correlation
    r1 = np.argsort(np.argsort(D_TRUTH[iu]))
    r2 = np.argsort(np.argsort(Dn[iu]))
    tau = float(np.corrcoef(r1, r2)[0, 1])
    noise_rows.append({
        "eps": eps, "rank_corr": rnd(tau, 5),
        "metric_disparity": rnd(procrustes_fit(TRUTH, Zm), 5),
        "nonmetric_disparity": rnd(procrustes_fit(TRUTH, Zn), 5),
        "metric_stress": rnd(tm[-1], 5), "nonmetric_stress": rnd(tn[-1], 5),
    })
    print(f"    noise {eps:<5} rank correlation {tau:.5f}   metric disparity "
          f"{noise_rows[-1]['metric_disparity']:.5f}   non-metric "
          f"{noise_rows[-1]['nonmetric_disparity']:.5f}")

# an independent solver on the undistorted case, because a stress this article
# publishes has to come from something other than this file's own loop
Zsk = sk_smacof(D_TRUTH, n_components=2, random_state=SEED, n_init=8,
                max_iter=600, eps=1e-10, normalized_stress=True)[0]
sk_stress = stress1(D_TRUTH, Zsk)
own = next(r for r in gamma_rows if r["gamma"] == 1.0)
print(f"  scikit-learn MDS on the undistorted matrix: stress {sk_stress:.5f} "
      f"against this file's {own['metric_stress']:.5f}")

# the browser draws its own Shepard diagram, so the only thing shipped is the
# check that this file's pool-adjacent-violators agrees with a library's
SHEP_GAMMA = 3.0
Dg = D_TRUTH ** SHEP_GAMMA
Dg = Dg / Dg.max()
Zn, _, hat = smacof(Dg, INIT, SM_ITERS, nonmetric=True, order=ORDER)
dz = pdist_matrix(Zn)
iso = IsotonicRegression().fit(Dg[iu], dz[iu])
iso_gap = float(np.abs(iso.predict(Dg[iu])
                       - pav(dz[iu][ORDER], np.ones(len(ORDER)))[np.argsort(ORDER)]).max())
print(f"  own pool-adjacent-violators against scikit-learn's isotonic fit: "
      f"worst {iso_gap:.3e}")

# and the claim behind the identical column: every non-metric run above is the
# same configuration, because a monotone distortion cannot reach the ordering
nm_maps = []
for g in GAMMAS:
    Dq = D_TRUTH ** g
    Dq = Dq / Dq.max()
    nm_maps.append(smacof(Dq, INIT, SM_ITERS, nonmetric=True, order=ORDER)[0])
nm_spread = float(max(np.abs(m - nm_maps[0]).max() for m in nm_maps))
print(f"  the {len(GAMMAS)} non-metric maps differ from each other by at most "
      f"{nm_spread:.3e}")

scaling_out = {
    "truth": mat(TRUTH, 4),
    "init": mat(INIT, 4),
    "iters": SM_ITERS,
    "gammas": GAMMAS,
    "rows": gamma_rows,
    "noise_rows": noise_rows,
    "shepard_gamma": SHEP_GAMMA,
    "sklearn_stress": rnd(sk_stress, 5),
    "pav_gap": float(f"{iso_gap:.3e}"),
    "nonmetric_spread": float(f"{nm_spread:.3e}"),
    "trace_metric": arr(trace_one[:60], 5),
    "d_truth_small": mat(D_TRUTH[np.ix_(SMALL, SMALL)], 4),
}

# ================================================= 4. factor analysis vs PCA
print("\n" + "=" * 72)
print("a model of the noise, against a rotation that has none")
NF, PF, KF = 300, 8, 2
rf = np.random.RandomState(SEED)
L_TRUE = np.round(rf.uniform(0.5, 1.4, (PF, KF)) * rf.choice([-1, 1], (PF, KF)), 3)
Zf = rf.normal(size=(NF, KF))
COMMON = np.round(Zf @ L_TRUE.T, 4)                 # the part every variable shares
NOISE = np.round(rf.normal(size=(NF, PF)), 4)       # fixed; the slider scales it
BASE_PSI = np.round(rf.uniform(0.30, 0.75, PF), 3)  # the quiet variables' noise sd
LOUD = 0
SDS = [round(0.2 + 0.2 * i, 1) for i in range(15)]  # the slider's own grid


def build(sd_loud, equal=False):
    psi_sd = BASE_PSI.copy()
    if equal:
        psi_sd = np.full(PF, float(np.round(BASE_PSI.mean(), 3)))
    psi_sd[LOUD] = sd_loud
    return np.round(COMMON + NOISE * psi_sd, 4), psi_sd


def compare(X, psi_sd, standardise=False):
    # Dividing a column by its own standard deviation rescales the true loading
    # of that variable by exactly the same factor, so the truth to compare
    # against has to be rescaled too. Comparing a rescaled fit to an unscaled
    # truth measures the rescaling and calls it an error: it made factor
    # analysis look worse than PCA, which is the opposite of what it does.
    truth = L_TRUE
    true_psi = np.asarray(psi_sd, dtype=float) ** 2
    if standardise:
        sd = X.std(0)
        X = (X - X.mean(0)) / sd
        truth = L_TRUE / sd[:, None]
        true_psi = true_psi / sd ** 2
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / len(Xc)
    p = PCA(n_components=KF, random_state=SEED).fit(Xc)
    Lp = (p.components_ * np.sqrt(p.explained_variance_)[:, None]).T
    Lf, psi, ll, iters_used = fa_em(S, KF)
    fsk = FactorAnalysis(n_components=KF, random_state=SEED,
                         max_iter=3000, tol=1e-8).fit(Xc)
    off = ~np.eye(p.n_features_in_, dtype=bool)
    resid_p = float(np.sqrt(((S - (Lp @ Lp.T + np.diag(np.diag(S - Lp @ Lp.T))))[off] ** 2).mean()))
    resid_f = float(np.sqrt(((S - (Lf @ Lf.T + np.diag(psi)))[off] ** 2).mean()))
    # With the noise made equal on every variable there is nothing left to
    # correlate against, so the field is null rather than a NaN on the page.
    pcorr = (rnd(float(np.corrcoef(psi, true_psi)[0, 1]), 4)
             if np.std(true_psi) > 1e-12 else None)
    return {
        "pca_gap": rnd(subspace_gap(truth, Lp), 5),
        "fa_gap": rnd(subspace_gap(truth, Lf), 5),
        "pca_loud_share": rnd(float((Lp[LOUD] ** 2).sum() / (Lp ** 2).sum()), 4),
        "fa_loud_share": rnd(float((Lf[LOUD] ** 2).sum() / (Lf ** 2).sum()), 4),
        "pca_loud_loading": rnd(float(np.sqrt((Lp[LOUD] ** 2).sum())), 4),
        "fa_loud_loading": rnd(float(np.sqrt((Lf[LOUD] ** 2).sum())), 4),
        "psi_true": arr(true_psi, 4),
        "psi_fa": arr(psi, 4),
        "psi_share": arr(psi / np.diag(S), 4),
        "psi_err": rnd(float(np.abs(psi - true_psi).max()), 4),
        "psi_rel_err": rnd(float((np.abs(psi - true_psi) / true_psi).max()), 4),
        "psi_corr": pcorr,
        "fa_converged": int(iters_used < 5000),
        "off_resid_pca": rnd(resid_p, 5),
        "off_resid_fa": rnd(resid_f, 5),
        "fa_iters": iters_used,
        "sk_psi_gap": float(f"{np.abs(psi - fsk.noise_variance_).max():.3e}"),
        "loadings_pca": mat(Lp, 4),
        "loadings_fa": mat(Lf, 4),
        "_share_raw": psi / np.diag(S),
        "_pc1_raw": Lp[:, 0] / np.linalg.norm(Lp[:, 0]),
    }


sweep = []
for sd in SDS:
    X, psi_sd = build(sd)
    r = compare(X, psi_sd)
    r["sd"] = sd
    # The widget fits both methods itself and draws its own loadings, so the
    # reference only has to carry the numbers a guard compares.
    for k in ("loadings_pca", "loadings_fa", "_share_raw", "_pc1_raw"):
        r.pop(k)
    sweep.append(r)
    print(f"  loud variable sd {sd:<4} subspace error: PCA {r['pca_gap']:.5f}  "
          f"FA {r['fa_gap']:.5f}   PCA puts {r['pca_loud_share']*100:5.2f}% of its "
          f"loading mass on it, FA {r['fa_loud_share']*100:5.2f}%   "
          f"(own EM against sklearn: {r['sk_psi_gap']})")

# the control: one thing changed, the noise made equal, and the theorem says the
# two methods must now agree
CTRL_SD = 2.6
X_un, psi_un = build(CTRL_SD)
X_eq, psi_eq = build(float(np.round(BASE_PSI.mean(), 3)), equal=True)
ctrl_un = compare(X_un, psi_un)
ctrl_eq = compare(X_eq, psi_eq)
print(f"\n  control arm, noise equal on every variable:")
print(f"    unequal noise: PCA subspace error {ctrl_un['pca_gap']:.5f}, FA {ctrl_un['fa_gap']:.5f}")
print(f"    equal noise:   PCA subspace error {ctrl_eq['pca_gap']:.5f}, FA {ctrl_eq['fa_gap']:.5f}")
# what standardising does to the story, which is the obvious objection
std_un = compare(X_un, psi_un, standardise=True)
print(f"    unequal noise, standardised first: PCA {std_un['pca_gap']:.5f}, "
      f"FA {std_un['fa_gap']:.5f}")
# and the reason FA barely moves: the share of each variable's variance it calls
# noise is the same number before and after standardising, to machine precision
scale_inv = float(np.abs(ctrl_un["_share_raw"] - std_un["_share_raw"]).max())
scale_inv_pca = float(np.abs(np.abs(ctrl_un["_pc1_raw"]) - np.abs(std_un["_pc1_raw"])).max())
print(f"    factor analysis calls the same SHARE of each variable noise either "
      f"way, to {scale_inv:.1e}; PCA's first direction moves by {scale_inv_pca:.4f}")

# rotation indeterminacy, on the wine, where the loadings have names
XWc = XS - XS.mean(0)
SW = XWc.T @ XWc / len(XWc)
K_WINE = 3
Lw, psiw, llw, itw = fa_em(SW, K_WINE)
Lrot, R = varimax(Lw)
sigma_before = Lw @ Lw.T + np.diag(psiw)
sigma_after = Lrot @ Lrot.T + np.diag(psiw)
rot_fit_gap = float(np.abs(sigma_before - sigma_after).max())
rot_load_change = float(np.abs(np.abs(Lrot) - np.abs(Lw)).max())
print(f"\n  varimax on the wine's {K_WINE} factors: the implied covariance moves by "
      f"{rot_fit_gap:.3e}, the loadings by up to {rot_load_change:.4f}")
print(f"    simplicity {simplicity(Lw):.4f} -> {simplicity(Lrot):.4f}")

# how many factors the wine wants, by a criterion PCA does not have
fa_k_rows = []
for k in range(1, 7):
    Lk, psik, llk, itk = fa_em(SW, k)
    npar = PF * 0 + 13 * k + 13 - k * (k - 1) // 2
    ll_full = loglik(SW, Lk @ Lk.T + np.diag(psik), N)
    bic = float(-2 * ll_full + npar * np.log(N))
    fa_k_rows.append({"k": k, "loglik": rnd(ll_full, 2), "params": int(npar),
                      "bic": rnd(bic, 2)})
best_k = min(fa_k_rows, key=lambda r: r["bic"])
print(f"  the wine by BIC: k = {best_k['k']} "
      f"({', '.join(str(r['bic']) for r in fa_k_rows)})")

# the uniquenesses on the wine, in the language of the measurements
uni_order = np.argsort(-psiw / np.diag(SW))
fa_out = {
    "n": NF, "p": PF, "k": KF, "loud": LOUD,
    "sds": SDS,
    "common": mat(COMMON, 4),
    "noise": mat(NOISE, 4),
    "base_psi_sd": arr(BASE_PSI, 3),
    "true_loadings": mat(L_TRUE, 3),
    "sweep": sweep,
    "control": {
        "sd": CTRL_SD,
        "unequal": {kk: ctrl_un[kk] for kk in
                    ("pca_gap", "fa_gap", "pca_loud_share", "fa_loud_share",
                     "off_resid_pca", "off_resid_fa", "psi_corr", "psi_err",
                     "loadings_pca", "loadings_fa")},
        "equal": {kk: ctrl_eq[kk] for kk in
                  ("pca_gap", "fa_gap", "pca_loud_share", "fa_loud_share",
                   "off_resid_pca", "off_resid_fa", "psi_corr", "psi_err")},
        "standardised": {kk: std_un[kk] for kk in
                         ("pca_gap", "fa_gap", "pca_loud_share", "fa_loud_share")},
        "scale_invariance": float(f"{scale_inv:.3e}"),
        "pca_direction_moves": rnd(scale_inv_pca, 4),
        "loud_index": LOUD,
    },
    "wine": {
        "k": K_WINE,
        "loadings": mat(Lw, 4),
        "varimax": mat(Lrot, 4),
        "uniquenesses": arr(psiw, 4),
        "uniqueness_share": arr(psiw / np.diag(SW), 4),
        "rot_fit_gap": float(f"{rot_fit_gap:.3e}"),
        "rot_load_change": rnd(rot_load_change, 4),
        "simplicity_before": rnd(simplicity(Lw), 4),
        "simplicity_after": rnd(simplicity(Lrot), 4),
        "noisiest": NAMES[int(uni_order[0])],
        "noisiest_share": rnd(float((psiw / np.diag(SW))[uni_order[0]]), 4),
        "cleanest": NAMES[int(uni_order[-1])],
        "cleanest_share": rnd(float((psiw / np.diag(SW))[uni_order[-1]]), 4),
        "by_k": fa_k_rows,
        "best_k": best_k["k"],
        "iters": itw,
    },
}

# ================================== 5. a dissimilarity with no coordinates at all
print("\n" + "=" * 72)
print("ten digits, mapped from how often a classifier mixes them up")
Xtr, ytr, Xte, yte = mnist()
Xd, yd = subset(Xtr, ytr, 2000, seed=SEED)
Xq, yq = subset(Xte, yte, 2000, seed=SEED)
FLAT = Xd.reshape(len(Xd), -1) / 255.0
FLATQ = Xq.reshape(len(Xq), -1) / 255.0
knn = KNeighborsClassifier(n_neighbors=5).fit(FLAT, yd)
pred = knn.predict(FLATQ)
C = np.zeros((10, 10))
for t, p_ in zip(yq, pred):
    C[t, p_] += 1
acc = float((pred == yq).mean())
counts = C.sum(1)
# Shepard's conversion of a confusion table into a dissimilarity, which is the
# oldest use multidimensional scaling was ever put to: how far apart two things
# are is the log of how much likelier the classifier is to get each of them
# right than to swap them. Zero confusions would send a pair to infinity, so a
# pair the classifier never mixes up counts as half a mistake, which is the
# smallest number of mistakes the experiment could have distinguished from none.
P = C / counts[:, None]
sym = 0.5 * (P + P.T)
floor = 0.5 / counts.mean()
DIG = np.log(np.sqrt(np.outer(np.diag(P), np.diag(P))) / np.maximum(sym, floor))
np.fill_diagonal(DIG, 0.0)
DIG = np.round(DIG / DIG.max(), 4)
never = int(((C + C.T)[np.triu_indices(10, 1)] == 0).sum())
print(f"  {never} of 45 class pairs were never once confused in either direction")
wD, VD = eig_desc(double_centre(DIG))
neg_dig = float(np.abs(np.minimum(wD, 0)).sum() / np.abs(wD).sum())
Zdig = fix_signs(VD[:, :2] * np.sqrt(np.maximum(wD[:2], 0)))
DIG_INIT = np.round(np.random.RandomState(3).normal(scale=0.2, size=(10, 2)), 4)
Zdig_nm, tr_nm, _ = smacof(DIG, DIG_INIT, 300, nonmetric=True,
                           order=np.argsort(DIG[np.triu_indices(10, 1)], kind="stable"))
iu10 = np.triu_indices(10, 1)
pairs = sorted(((DIG[i, j], i, j) for i, j in zip(*iu10)))
print(f"  5-nearest-neighbour accuracy on {len(FLATQ)} held out digits: {acc:.4f}")
print(f"  most confusable pair: {pairs[0][1]} and {pairs[0][2]} at {pairs[0][0]:.4f}; "
      f"least: {pairs[-1][1]} and {pairs[-1][2]} at {pairs[-1][0]:.4f}")
print(f"  negative eigenvalue mass of the confusion dissimilarity: {neg_dig:.4f} "
      f"({int((wD < -1e-9).sum())} negative eigenvalues)")

digits_out = {
    "n_train": int(len(FLAT)), "n_test": int(len(FLATQ)), "k": 5,
    "accuracy": rnd(acc, 4),
    "confusion": [[int(v) for v in row] for row in C],
    "dissimilarity": mat(DIG, 4),
    "init": mat(DIG_INIT, 4),
    "smacof_iters": 300,
    "eigenvalues": arr(wD, 4),
    "negative_mass": rnd(neg_dig, 4),
    "neg_count": int((wD < -1e-9).sum()),
    "coords": mat(Zdig, 4),
    "coords_nonmetric": mat(Zdig_nm, 4),
    "stress_classical": rnd(stress1(DIG, Zdig), 4),
    "stress_nonmetric": rnd(tr_nm[-1], 4),
    "closest": [int(pairs[0][1]), int(pairs[0][2])],
    "closest_d": rnd(pairs[0][0], 4),
    "furthest": [int(pairs[-1][1]), int(pairs[-1][2])],
    "furthest_d": rnd(pairs[-1][0], 4),
    "lingoes": rnd(lingoes(DIG)[0], 4),
    "lingoes_left": float(f"{lingoes(DIG)[1]:.3e}"),
    "never_confused": never,
    "top_pairs": [{"a": int(i), "b": int(j), "d": rnd(d, 4)} for d, i, j in pairs[:5]],
    "evr2": rnd(float(np.maximum(wD[:2], 0).sum() / np.maximum(wD, 0).sum()), 4),
}

# ===================================================================== write it
out = {
    "meta": {
        "seed": SEED,
        "wine_rounding": 4,
        "note": "every figure on the page is computed by this file; the browser "
                "redoes the double centring, the eigendecomposition, the SMACOF "
                "iterations and the EM factor analysis itself and checks against "
                "the references here",
        "sklearn": __import__("sklearn").__version__,
        "reuse_check": "the raw wine matrix is asserted identical to the one "
                       "directions/data/directions.json ships",
    },
    "wine": wine_out,
    "metrics": {"rows": metric_rows},
    "scaling": scaling_out,
    "factor": fa_out,
    "digits": digits_out,
}
path = OUT / "distances.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size/1024:.1f} kB)")
