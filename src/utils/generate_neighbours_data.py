"""Data for the ATLAS 'Neighbours' article: t-SNE and UMAP.

Every method in this module so far has tried to reproduce DISTANCES: PCA
minimises squared error against them, classical scaling inverts them, Isomap
replaces them with better ones and then inverts those. This article gives up on
distances entirely. It keeps only the question of who is whose neighbour, turns
that into a probability distribution over pairs, and matches probabilities. That
fixes the fragility of the previous three and introduces a completely different
family of ways to be misled, all of which are measurable and all of which are
measured here.

Seven blocks:

1. Why the previous methods had to fail on anything high dimensional, from
   first principles and with a number: k mutually equidistant points need k-1
   dimensions, so a plane can hold three, and everything past that has to be
   distorted. Measured as the spread of the distances a plane actually achieves.
2. The perplexity binary search, exported so the browser can be checked to have
   solved the same equation.
3. A t-SNE that the page runs itself, from a starting configuration and a
   schedule this file also runs, so the two trajectories are the same one.
4. The finding that matters most and is least often shown: run the same t-SNE
   on data with NO structure at all and it returns clusters. Counted, not
   described, with the same counting applied to the real digits as a control.
5. Cluster sizes and the gaps between clusters, against blobs whose radii and
   separations are known by construction. Neither survives.
6. Ten seeds, and what changes between them.
7. The scoreboard on 2,000 MNIST digits, the same draw the three previous
   articles used, with the same score module 2.2 has been using throughout.

UMAP is not reimplemented here. Its embeddings come from the reference package,
and the article says so: reproducing its negative sampling schedule in a browser
and calling the result UMAP would be a claim this atlas could not check.

Run from anywhere: `python src/utils/generate_neighbours_data.py`.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.vision_data import mnist, subset            # noqa: E402

OUT = ROOT / "neighbours" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 19
rnd = lambda v, n=4: round(float(v), n)
arr = lambda a, n=4: [round(float(v), n) for v in np.asarray(a).ravel()]
mat = lambda A, n=4: [[round(float(v), n) for v in row] for row in np.asarray(A)]

N_LIVE = 500           # what the browser reruns
N_BIG = 2000           # the scoreboard, same draw as the three previous articles
N_PCS = 30
ITERS = 1000
EXAGG_ITERS = 250
EXAGG = 12.0
LR = 200.0
PERPS = [2, 5, 10, 20, 30, 50, 80]
PERP_SHOW = 30

print("=" * 72)
print("ATLAS / neighbours: t-SNE and UMAP")
print("=" * 72)


# ----------------------------------------------------------------- shared maths
def seeded_block(n, k, seed):
    """The same linear congruential generator the browser runs, so a starting
    configuration is a shared fact rather than two similar random draws."""
    Q = np.empty((n, k))
    # plain Python integers with an explicit mask: numpy's uint32 does the same
    # arithmetic and warns about it on every step
    s = int(seed) & 0xFFFFFFFF
    for i in range(n):
        for j in range(k):
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            Q[i, j] = s / 4294967296.0 - 0.5
    return Q


def seeded_normal(n, d, seed, scale=1.0):
    """Standard normals from the same generator, by Box and Muller.

    The alternative is shipping the matrix, and for the two synthetic datasets
    on this page that is a quarter of a megabyte the reader would download in
    order to be told there is nothing in it. Both ends draw two uniforms per
    entry, in row major order, so the two arrays are the same array.
    """
    out = np.empty((n, d))
    s = int(seed) & 0xFFFFFFFF
    for i in range(n):
        for j in range(d):
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            u1 = max(s / 4294967296.0, 1e-12)
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            u2 = s / 4294967296.0
            out[i, j] = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2) * scale
    return out


def sq_dists(X):
    s = (X ** 2).sum(1)
    D = s[:, None] + s[None, :] - 2 * X @ X.T
    np.maximum(D, 0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def conditional_p(D2, perplexity, tol=1e-10, iters=200):
    """One bandwidth per point, chosen so that the neighbour distribution it
    induces has exactly the requested perplexity.

    Perplexity is two to the power of the entropy: the effective number of
    neighbours. Solving for it point by point is what lets t-SNE use one number
    in a dense region and a very different one in a sparse region, and it is the
    reason the algorithm has no notion of scale left over afterwards.
    """
    n = len(D2)
    P = np.zeros((n, n))
    betas = np.zeros(n)
    logU = np.log(perplexity)
    for i in range(n):
        lo, hi = -np.inf, np.inf
        beta = 1.0
        Di = np.delete(D2[i], i)
        for _ in range(iters):
            Pi = np.exp(-Di * beta)
            ssum = Pi.sum()
            if ssum <= 1e-300:
                H = 0.0
                Pi = np.zeros_like(Pi)
            else:
                H = np.log(ssum) + beta * (Di * Pi).sum() / ssum
                Pi = Pi / ssum
            diff = H - logU
            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        betas[i] = beta
        P[i, np.arange(n) != i] = Pi
    return P, betas


def joint_p(X, perplexity):
    P, betas = conditional_p(sq_dists(X), perplexity)
    P = (P + P.T) / (2 * len(P))
    np.maximum(P, 1e-12, out=P)
    return P, betas


def tsne(P, Y0, n_iter=ITERS, exagg_iters=EXAGG_ITERS, exagg=EXAGG, lr=LR,
         record=None):
    """van der Maaten's reference gradient descent, written out.

    Everything about it that could drift between two implementations has been
    pinned: the starting configuration is passed in, the gains rule and the two
    momenta are the published ones, and the early exaggeration switches off at a
    fixed iteration rather than on a convergence test.
    """
    n = len(P)
    Y = np.array(Y0, dtype=float)
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    trace = []
    for it in range(n_iter):
        num = 1.0 / (1.0 + sq_dists(Y))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        Pe = P * exagg if it < exagg_iters else P
        PQ = (Pe - Q) * num
        grad = 4.0 * ((np.diag(PQ.sum(1)) - PQ) @ Y)
        mom = 0.5 if it < exagg_iters else 0.8
        inc = (np.sign(grad) != np.sign(dY))
        gains = np.where(inc, gains + 0.2, gains * 0.8)
        np.maximum(gains, 0.01, out=gains)
        dY = mom * dY - lr * gains * grad
        Y = Y + dY
        Y = Y - Y.mean(0)
        if record is not None and (it + 1) in record:
            trace.append({"iter": it + 1, "kl": rnd(float((P * np.log(P / Q)).sum()), 6)})
    num = 1.0 / (1.0 + sq_dists(Y))
    np.fill_diagonal(num, 0.0)
    Q = np.maximum(num / num.sum(), 1e-12)
    kl = float((P * np.log(P / Q)).sum())
    return Y, kl, trace


def procrustes_fit(A, B):
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


def knn_sets(X, k=10):
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def knn_overlap(A, B):
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(A, B)]))


def count_blobs(Y, mult, min_samples=8):
    """How many separated clumps a reader would count, made mechanical.

    The radius is a multiple of the median distance to the tenth nearest
    neighbour, so the rule is scale free and applies unchanged to every
    embedding on this page. The multiple itself is not chosen by eye: it is
    calibrated below on an embedding whose answer is known to be three.
    """
    D = pairwise_distances(Y)
    np.fill_diagonal(D, np.inf)
    kd = np.sort(D, axis=1)[:, 9]
    eps = mult * float(np.median(kd))
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Y)
    return int(len(set(lab) - {-1})), float((lab == -1).mean()), eps


def hopkins(X, m=200, seed=0):
    """Clustering tendency: near 0.5 when points are spread out with no clumps,
    near 1 when they are clumped.

    Drawn from the same statistic Lawson and Jurs published in 1990. It is the
    only honest way to ask whether a picture has structure in it, because "does
    this look like clusters to you" is exactly the question at issue.
    """
    rs = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    m = min(m, n // 3)
    lo, hi = X.min(0), X.max(0)
    U = rs.uniform(lo, hi, size=(m, d))
    idx = rs.choice(n, m, replace=False)
    Du = pairwise_distances(U, X).min(1)
    Dw = pairwise_distances(X[idx], X)
    Dw[np.arange(m), idx] = np.inf
    Dw = Dw.min(1)
    return float(Du.sum() / (Du.sum() + Dw.sum()))


# ================================================ 1. why a plane is not enough
print("\n" + "=" * 72)
print("what a plane cannot hold")
simplex_rows = []
for k in range(3, 13):
    # k mutually equidistant points live in k-1 dimensions and nowhere smaller
    V = np.eye(k)
    V = V - V.mean(0)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    Xs = (U[:, :k - 1] * s[:k - 1])
    D = pairwise_distances(Xs)
    iu = np.triu_indices(k, 1)
    assert np.allclose(D[iu], D[iu][0]), "the simplex is not equilateral"
    # the best plane available, by metric scaling on those distances
    Z = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    d2 = pairwise_distances(Z)[iu]
    simplex_rows.append({
        "k": k, "dims_needed": k - 1,
        "spread": rnd(float(d2.std() / d2.mean()), 5),
        "ratio": rnd(float(d2.max() / max(d2.min(), 1e-12)), 3),
    })
    print(f"  {k:>2} equidistant points need {k-1:>2} dimensions; squeezed into 2 their "
          f"distances spread by {simplex_rows[-1]['spread']*100:6.2f}% and the largest "
          f"is {simplex_rows[-1]['ratio']:.2f} times the smallest")

# and the other half of the same story: how much room there is at radius r
volume_rows = []
rs = np.random.RandomState(SEED)
for d in [2, 3, 5, 10, 20, 30, 50]:
    Xd = rs.normal(size=(1500, d))
    Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
    Dd = pairwise_distances(Xd)
    iu = np.triu_indices(1500, 1)
    v = Dd[iu]
    volume_rows.append({"d": d, "mean": rnd(v.mean(), 4), "sd": rnd(v.std(), 4),
                        "cv": rnd(float(v.std() / v.mean()), 5)})
    print(f"  {d:>2} dimensions: pairwise distances {v.mean():.3f} give or take "
          f"{v.std():.3f}, a spread of {v.std()/v.mean()*100:5.2f}%")

# The clump counter's one free parameter, pinned from BOTH sides on cases whose
# answers are known, which is the only way a count of clusters is allowed to be
# used as evidence about clusters. It has to call a plain two-dimensional
# gaussian one clump (there are no clusters in it) and a t-SNE of three well
# separated blobs three (there are exactly three). Anything in between is a
# measurement rather than a choice.
print("\n" + "=" * 72)
print("pinning the clump counter from both sides")
_C = np.zeros((3, N_PCS))
_C[1, 0] = 40.0
_C[2, 1] = 80.0
_CAL = np.round(np.vstack([_C[i] + seeded_normal(160, N_PCS, 700 + i, r)
                           for i, r in enumerate([1.0, 2.0, 5.0])]), 3)
_Pcal, _ = joint_p(np.round(_CAL, 3), PERP_SHOW)
_Ycal, _, _ = tsne(_Pcal, np.round(seeded_block(len(_CAL), 2, 7) * 0.0002, 8))
FLAT2D = seeded_normal(N_LIVE, 2, 55)
MULT = None
for m in [round(0.6 + 0.1 * i, 1) for i in range(40)]:
    lo_ok = count_blobs(FLAT2D, m)[0] <= 1
    hi_ok = count_blobs(_Ycal, m)[0] == 3 and count_blobs(_Ycal, m)[1] < 0.05
    if lo_ok and hi_ok:
        MULT = m
        break
if MULT is None:
    MULT = 1.5
    print("  no multiple satisfies both ends; falling back to 1.5 and saying so")
HOP_FLAT = hopkins(FLAT2D, seed=3)
HOP_UNIF = hopkins(np.random.RandomState(56).uniform(size=(N_LIVE, 2)), seed=3)
# (uniform stays a numpy draw: it is quoted once, not recomputed anywhere)
print(f"  a radius of {MULT} times the median distance to the tenth nearest "
      f"neighbour calls a two-dimensional gaussian "
      f"{count_blobs(FLAT2D, MULT)[0]} clump and three blobs "
      f"{count_blobs(_Ycal, MULT)[0]}")
print(f"  and the clustering tendency of a picture with no clusters in it: "
      f"{HOP_FLAT:.4f} for a gaussian, {HOP_UNIF:.4f} for a uniform square, "
      f"both at n = {N_LIVE}")


# ============================================================= 2. the digits
print("\n" + "=" * 72)
print("2,000 MNIST digits, and 500 of them for the page to work on")
Xtr, ytr, _, _ = mnist()
Xd, yd = subset(Xtr, ytr, N_BIG, seed=SEED)
FLAT = Xd.reshape(len(Xd), -1) / 255.0
pca_big = PCA(n_components=N_PCS, random_state=SEED).fit(FLAT)
SC_BIG = pca_big.transform(FLAT)
print(f"  {N_PCS} principal components keep "
      f"{pca_big.explained_variance_ratio_.sum()*100:.2f}% of the variance")

sub = np.sort(np.concatenate([np.where(yd == c)[0][:N_LIVE // 10] for c in range(10)]))
SC = np.round(SC_BIG[sub], 3)                # what the browser downloads
YS = yd[sub]
FLAT_SUB = FLAT[sub]
print(f"  the page's subset: {len(SC)} digits, {SC.shape[1]} coordinates each")

# ------------------------------------------------- the perplexity search itself
perp_ref = {}
for p in PERPS:
    P, betas = joint_p(SC, p)
    ent = []
    D2 = sq_dists(SC)
    for i in range(len(SC)):
        Di = np.delete(D2[i], i)
        Pi = np.exp(-Di * betas[i])
        Pi = Pi / Pi.sum()
        nz = Pi[Pi > 0]
        ent.append(float(np.exp(-(nz * np.log(nz)).sum())))
    perp_ref[p] = {
        "betas": arr(betas, 6),
        "worst_perplexity_error": float(f"{max(abs(e - p) for e in ent):.3e}"),
        "sigma_min": rnd(float(1 / np.sqrt(2 * betas.max())), 5),
        "sigma_max": rnd(float(1 / np.sqrt(2 * betas.min())), 5),
        "p_sum": rnd(float(P.sum()), 8),
    }
    print(f"  perplexity {p:>2}: every point's neighbour distribution hits it to "
          f"{perp_ref[p]['worst_perplexity_error']}, bandwidths from "
          f"{perp_ref[p]['sigma_min']} to {perp_ref[p]['sigma_max']} "
          f"(a factor of {perp_ref[p]['sigma_max']/perp_ref[p]['sigma_min']:.1f})")

# =============================================== 3. the run the browser repeats
print("\n" + "=" * 72)
print("the t-SNE this page runs")
Y0 = np.round(seeded_block(len(SC), 2, 7) * 0.0002, 8)
CHECKS = [50, 100, 250, 400, 600, 1000]
runs = {}
for p in PERPS:
    P, _ = joint_p(SC, p)
    Y, kl, trace = tsne(P, Y0, record=set(CHECKS))
    nb, noise, eps = count_blobs(Y, MULT)
    tw = float(trustworthiness(SC, Y, n_neighbors=12))
    kn = KNeighborsClassifier(n_neighbors=5)
    acc = float(np.mean([
        kn.fit(np.delete(Y, i, 0), np.delete(YS, i)).predict(Y[i:i + 1])[0] == YS[i]
        for i in range(0, len(Y), 5)]))
    hop = hopkins(Y, seed=3)
    runs[p] = {
        "kl": rnd(kl, 6), "trace": trace, "clusters": nb, "noise": rnd(noise, 4),
        "trustworthiness": rnd(tw, 5), "accuracy": rnd(acc, 4), "hopkins": rnd(hop, 4),
    }
    print(f"  perplexity {p:>2}: KL {kl:.5f}, trustworthiness {tw:.5f}, "
          f"{nb} clumps, a five nearest neighbour vote on the map gets {acc*100:.1f}% "
          f"right, clustering tendency {hop:.4f}")

# an independent implementation, for the one thing it can honestly be asked
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Ysk = TSNE(n_components=2, perplexity=PERP_SHOW, init="pca", random_state=SEED,
               max_iter=1000).fit_transform(SC)
sk_tw = float(trustworthiness(SC, Ysk, n_neighbors=12))
own_tw = runs[PERP_SHOW]["trustworthiness"]
print(f"  scikit-learn's own t-SNE at perplexity {PERP_SHOW}: trustworthiness "
      f"{sk_tw:.5f} against this file's {own_tw}")

# How much of a nudge it takes to change the answer, which turns out to be the
# most practically important number on this page. Two implementations of this
# algorithm cannot agree on the bandwidths to better than about a part in ten
# million, because one sums with numpy's pairwise summation and the other adds
# left to right. So perturb the input by a part in a million and measure what
# comes out the far end: if the answer moves, then "the same t-SNE" is a phrase
# that needs a version number attached to it.
print("\n" + "=" * 72)
print("what a nudge of one part in a million does")
SC_EPS = np.round(SC + seeded_normal(len(SC), N_PCS, 999) * 1e-6, 9)
P_EPS, beta_eps = joint_p(SC_EPS, PERP_SHOW)
Y_EPS, kl_eps, _ = tsne(P_EPS, Y0)
P_REF, beta_ref = joint_p(SC, PERP_SHOW)
Y_REF, kl_ref, _ = tsne(P_REF, Y0)
nudge = {
    "input_change": float(f"{np.abs(SC_EPS - SC).max():.3e}"),
    "beta_change": float(f"{np.abs(beta_eps - beta_ref).max():.3e}"),
    "p_change": float(f"{np.abs(P_EPS - P_REF).max():.3e}"),
    "kl_ref": rnd(kl_ref, 6), "kl_nudged": rnd(kl_eps, 6),
    "kl_change": rnd(abs(kl_eps - kl_ref), 6),
    "disparity": rnd(procrustes_fit(Y_REF, Y_EPS), 5),
    "knn_overlap": rnd(knn_overlap(knn_sets(Y_REF), knn_sets(Y_EPS)), 5),
}
print(f"  moving every coordinate by at most {nudge['input_change']} moves the "
      f"bandwidths by {nudge['beta_change']} and the neighbour probabilities by "
      f"{nudge['p_change']}")
print(f"  and the finished map by {nudge['disparity']} in shape, with the divergence "
      f"going from {nudge['kl_ref']} to {nudge['kl_nudged']}; "
      f"{nudge['knn_overlap']*100:.1f}% of each point's ten nearest neighbours survive")


# =========================================== 4. structure that is not there
print("\n" + "=" * 72)
print("the same algorithm on data with nothing in it")
NOISE = np.round(seeded_normal(N_LIVE, N_PCS, 101, scale=float(SC.std())), 3)
hop_in_noise = hopkins(NOISE, seed=3)
hop_in_digits = hopkins(SC, seed=3)
print(f"  before any of this: clustering tendency {hop_in_noise:.4f} in the noise "
      f"itself and {hop_in_digits:.4f} in the digits themselves")
noise_rows = []
for p in PERPS:
    P, _ = joint_p(NOISE, p)
    Y, kl, _ = tsne(P, Y0)
    nb, left, eps = count_blobs(Y, MULT)
    real = runs[p]["clusters"]
    hop = hopkins(Y, seed=3)
    noise_rows.append({"perplexity": p, "clusters": nb, "noise_share": rnd(left, 4),
                       "kl": rnd(kl, 6), "real_clusters": real,
                       "hopkins": rnd(hop, 4)})
    print(f"  perplexity {p:>2}: {nb} apparent clumps in pure noise "
          f"(the real digits give {real}), clustering tendency {hop:.4f}, KL {kl:.5f}")

# The famous version of this claim uses a smaller sample in more dimensions at a
# very low perplexity, so before reporting that the effect did not appear, try
# it where it is supposed to live.
print("\n  and the same question where the folklore version of it lives:")
folklore = []
for nn, dd, pp in [(100, 30, 2), (500, 100, 2), (100, 100, 2), (100, 100, 5),
                   (200, 200, 2)]:
    Xf = np.round(seeded_normal(nn, dd, 202 + nn + dd), 3)
    Pf, _ = joint_p(Xf, pp)
    Yf, klf, _ = tsne(Pf, np.round(seeded_block(nn, 2, 7) * 0.0002, 8))
    nbf, leftf, _ = count_blobs(Yf, MULT)
    folklore.append({"n": nn, "d": dd, "perplexity": pp, "clusters": nbf,
                     "noise_share": rnd(leftf, 4), "hopkins": rnd(hopkins(Yf, seed=3), 4)})
    print(f"    {nn:>3} points in {dd:>3} dimensions at perplexity {pp}: "
          f"{nbf} apparent clumps")


# ================================= 5. sizes and gaps, against a known geometry
print("\n" + "=" * 72)
print("three blobs whose radii and separations are known")
RADII = [1.0, 2.0, 5.0]
CENTRES = np.zeros((3, N_PCS))
CENTRES[1, 0] = 40.0
CENTRES[2, 1] = 80.0
per = 160
BLOBS = np.round(np.vstack([CENTRES[i] + seeded_normal(per, N_PCS, 700 + i, RADII[i])
                            for i in range(3)]), 3)
BLAB = np.repeat(np.arange(3), per)


def blob_geometry(Y):
    cent = np.array([Y[BLAB == i].mean(0) for i in range(3)])
    rad = [float(np.sqrt(((Y[BLAB == i] - cent[i]) ** 2).sum(1)).mean()) for i in range(3)]
    sep = pairwise_distances(cent)
    return rad, sep


true_rad, true_sep = blob_geometry(BLOBS)
blob_rows = []
for p in PERPS:
    P, _ = joint_p(BLOBS, p)
    Y, kl, _ = tsne(P, np.round(seeded_block(len(BLOBS), 2, 7) * 0.0002, 8))
    rad, sep = blob_geometry(Y)
    blob_rows.append({
        "perplexity": p,
        "radius_ratio": rnd(rad[2] / rad[0], 3),
        "true_radius_ratio": rnd(true_rad[2] / true_rad[0], 3),
        "sep_ratio": rnd(sep[0, 2] / sep[0, 1], 3),
        "true_sep_ratio": rnd(true_sep[0, 2] / true_sep[0, 1], 3),
    })
    print(f"  perplexity {p:>2}: the widest blob is {rad[2]/rad[0]:.2f} times the "
          f"narrowest on the page against {true_rad[2]/true_rad[0]:.2f} in the data; "
          f"the far gap is {sep[0,2]/sep[0,1]:.2f} times the near one against "
          f"{true_sep[0,2]/true_sep[0,1]:.2f}")

# ============================================================== 6. ten seeds
print("\n" + "=" * 72)
print("ten seeds")
P30, _ = joint_p(SC, PERP_SHOW)
seed_ys = []
for s in range(10):
    Y0s = np.round(seeded_block(len(SC), 2, 11 + s) * 0.0002, 8)
    Y, kl, _ = tsne(P30, Y0s)
    seed_ys.append(Y)
pairs_disp = []
pairs_knn = []
base_sets = [knn_sets(Y) for Y in seed_ys]
for a in range(10):
    for b in range(a + 1, 10):
        pairs_disp.append(procrustes_fit(seed_ys[a], seed_ys[b]))
        pairs_knn.append(knn_overlap(base_sets[a], base_sets[b]))
print(f"  between any two runs: shape disparity {np.mean(pairs_disp):.4f} "
      f"(worst {max(pairs_disp):.4f}), and {np.mean(pairs_knn)*100:.1f}% of each point's "
      f"ten nearest neighbours are the same")

# ============================================================ 7. the scoreboard
print("\n" + "=" * 72)
print("the scoreboard, on all 2,000")
try:
    import umap                                                # noqa: E402
    HAVE_UMAP = True
except Exception as exc:                                       # pragma: no cover
    print(f"  umap-learn unavailable ({exc})")
    HAVE_UMAP = False

rsp = np.random.RandomState(SEED)
perm = rsp.permutation(N_BIG)
tr, te = perm[:1400], perm[1400:]
board = []


def score(name, Y, note=""):
    knn = KNeighborsClassifier(n_neighbors=5).fit(Y[tr], yd[tr])
    acc = float((knn.predict(Y[te]) == yd[te]).mean())
    tw = float(trustworthiness(SC_BIG, Y, n_neighbors=12))
    Dz = pairwise_distances(Y)
    Dx = pairwise_distances(SC_BIG)
    iu = np.triu_indices(len(Y), 1)
    gc = float(np.corrcoef(Dz[iu], Dx[iu])[0, 1])
    board.append({"method": name, "accuracy": rnd(acc, 4), "trustworthiness": rnd(tw, 5),
                  "global_corr": rnd(gc, 5), "note": note})
    print(f"  {name:<10} 5-nn on the map {acc:.4f}, trustworthiness {tw:.5f}, "
          f"agreement with the original distances {gc:.5f}")
    return Y


score("pca", PCA(n_components=2, random_state=SEED).fit_transform(SC_BIG))
Pbig, _ = joint_p(np.round(SC_BIG, 3), PERP_SHOW)
Ybig, kl_big, _ = tsne(Pbig, np.round(seeded_block(N_BIG, 2, 7) * 0.0002, 8))
score("t-sne", Ybig, f"perplexity {PERP_SHOW}, KL {kl_big:.4f}")
umap_big = None
umap_runs = {}
if HAVE_UMAP:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        umap_big = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                             random_state=SEED).fit_transform(SC_BIG)
        score("umap", umap_big, "15 neighbours, minimum distance 0.1")
        # the same three questions asked of UMAP that were asked of t-SNE
        for nn in [5, 15, 50]:
            Yu = umap.UMAP(n_neighbors=nn, min_dist=0.1, n_components=2,
                           random_state=SEED).fit_transform(SC)
            Yn = umap.UMAP(n_neighbors=nn, min_dist=0.1, n_components=2,
                           random_state=SEED).fit_transform(NOISE)
            Yb = umap.UMAP(n_neighbors=nn, min_dist=0.1, n_components=2,
                           random_state=SEED).fit_transform(BLOBS)
            rad, sep = blob_geometry(Yb)
            umap_runs[nn] = {
                "clusters": count_blobs(Yu, MULT)[0],
                "noise_clusters": count_blobs(Yn, MULT)[0],
                "hopkins": rnd(hopkins(Yu, seed=3), 4),
                "noise_hopkins": rnd(hopkins(Yn, seed=3), 4),
                "trustworthiness": rnd(float(trustworthiness(SC, Yu, n_neighbors=12)), 5),
                "radius_ratio": rnd(rad[2] / rad[0], 3),
                "sep_ratio": rnd(sep[0, 2] / sep[0, 1], 3),
            }
            if nn == 15:
                umap_runs[nn]["coords"] = mat(Yu / np.abs(Yu).max(), 4)
                umap_runs[nn]["noise_coords"] = mat(Yn / np.abs(Yn).max(), 4)
            print(f"  umap {nn:>2} neighbours: {umap_runs[nn]['clusters']} clumps on the "
                  f"digits, {umap_runs[nn]['noise_clusters']} on pure noise, "
                  f"trustworthiness {umap_runs[nn]['trustworthiness']}, widest blob "
                  f"{umap_runs[nn]['radius_ratio']} times the narrowest")
        # ten seeds of UMAP, the same question asked of t-SNE
        useeds = [umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            random_state=100 + s).fit_transform(SC) for s in range(10)]
    ud = []
    uk = []
    usets = [knn_sets(Y) for Y in useeds]
    for a in range(10):
        for b in range(a + 1, 10):
            ud.append(procrustes_fit(useeds[a], useeds[b]))
            uk.append(knn_overlap(usets[a], usets[b]))
    print(f"  umap over ten seeds: shape disparity {np.mean(ud):.4f}, neighbour "
          f"overlap {np.mean(uk)*100:.1f}%")
else:
    ud = [float("nan")]
    uk = [float("nan")]

# ===================================================================== write it
out = {
    "meta": {
        "seed": SEED, "n_live": len(SC), "n_big": N_BIG, "pcs": N_PCS,
        "iters": ITERS, "exagg_iters": EXAGG_ITERS, "exagg": EXAGG, "lr": LR,
        "perplexities": PERPS, "perplexity_show": PERP_SHOW, "checks": CHECKS,
        "sklearn": __import__("sklearn").__version__,
        "umap": (umap.__version__ if HAVE_UMAP else None),
        "variance_kept": rnd(float(pca_big.explained_variance_ratio_.sum()), 5),
        "note": "the browser solves the same perplexity search and runs the same "
                "gradient descent from the same starting configuration; UMAP is "
                "not reimplemented and its embeddings come from the package",
    },
    "crowding": {"simplex": simplex_rows, "volume": volume_rows},
    "digits": {
        "scores": mat(SC, 3),
        "labels": [int(v) for v in YS],
        "init": mat(Y0, 8),
        "perplexity": {k: ({kk: vv for kk, vv in v.items()
                            if kk != "betas" or k == PERP_SHOW})
                       for k, v in perp_ref.items()},
        "runs": {str(k): v for k, v in runs.items()},
        "sklearn_trustworthiness": rnd(sk_tw, 5),
    },
    "noise": {
        "seed": 101, "scale": rnd(float(SC.std()), 6),
        "rows": noise_rows,
        "folklore": folklore,
        "hopkins_input": rnd(hop_in_noise, 4),
        "hopkins_digits_input": rnd(hop_in_digits, 4),
        "hopkins_flat_2d": rnd(HOP_FLAT, 4),
        "hopkins_uniform_2d": rnd(HOP_UNIF, 4),
        "counter_multiple": MULT,
    },
    "blobs": {
        "seeds": [700, 701, 702], "per": per, "centres": mat(CENTRES, 3),
        "labels": [int(v) for v in BLAB],
        "radii": RADII,
        "true_radius_ratio": rnd(true_rad[2] / true_rad[0], 3),
        "true_sep_ratio": rnd(true_sep[0, 2] / true_sep[0, 1], 3),
        "rows": blob_rows,
        "init": mat(np.round(seeded_block(len(BLOBS), 2, 7) * 0.0002, 8), 8),
    },
    "nudge": nudge,
    "seeds": {
        "n": 10,
        "tsne_disparity": rnd(float(np.mean(pairs_disp)), 5),
        "tsne_disparity_worst": rnd(float(np.max(pairs_disp)), 5),
        "tsne_knn_overlap": rnd(float(np.mean(pairs_knn)), 5),
        "umap_disparity": rnd(float(np.mean(ud)), 5),
        "umap_knn_overlap": rnd(float(np.mean(uk)), 5),
    },
    "board": board,
    "umap": {"runs": {str(k): v for k, v in umap_runs.items()},
             "shown": 15,
             "big": mat(umap_big / np.abs(umap_big).max(), 3) if HAVE_UMAP else None},
    "big": {"tsne": mat(Ybig / np.abs(Ybig).max(), 3),
            "labels": [int(v) for v in yd]},
}
path = OUT / "neighbours.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size/1024:.1f} kB)")
