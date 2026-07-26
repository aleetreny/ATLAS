"""Data for the ATLAS 'Gaussian Mixture Models' web article.

The article is the promised sequel to k-means: same blobs (seed 19), same
failure gallery, same 178 wines, so every comparison is against numbers the
reader has already met. EM itself runs live in the browser; this file ships
the seeded datasets and scikit-learn reference answers the page checks
itself against, plus the measurements that need a real library: BIC/AIC
across k, the four covariance shapes on the sheared data, the gallery
revisited, and the wine bottles with their soft memberships.

Run from anywhere: `python src/utils/generate_gmm_data.py`.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gmm" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19


def cov2(m):
    """A 2x2 covariance as [c00, c01, c11]."""
    return [rnd(m[0][0]), rnd(m[0][1]), rnd(m[1][1])]


def pack(gm, X, y):
    labels = gm.predict(X)
    return {
        "weights": [rnd(w) for w in gm.weights_],
        "means": [[rnd(a, 3), rnd(b, 3)] for a, b in gm.means_],
        "covs": [cov2(c) for c in covs_of(gm)],
        "labels": [int(v) for v in labels],
        "ari": rnd(adjusted_rand_score(y, labels), 3),
    }


def covs_of(gm):
    ct = gm.covariance_type
    k, d = gm.means_.shape
    if ct == "full":
        return gm.covariances_
    if ct == "tied":
        return [gm.covariances_] * k
    if ct == "diag":
        return [np.diag(c) for c in gm.covariances_]
    return [np.eye(d) * c for c in gm.covariances_]  # spherical


# ------------------------------------------------------------- the main blobs
Xb, yb = make_blobs(n_samples=300, centers=[[-3.2, 1.8], [1.8, 3.0], [0.6, -2.6]],
                    cluster_std=[0.85, 1.05, 0.9], random_state=SEED)
gm_ref = GaussianMixture(3, covariance_type="full", n_init=10, random_state=SEED).fit(Xb)
print(f"blobs: mean log-likelihood {gm_ref.score(Xb):.4f}, "
      f"ARI {adjusted_rand_score(yb, gm_ref.predict(Xb)):.3f}")

blobs = {
    "points": [[rnd(a, 3), rnd(b, 3)] for a, b in Xb],
    "true_labels": [int(v) for v in yb],
    "ref_loglik": rnd(gm_ref.score(Xb)),          # per-sample mean, the JS check target
    "ref": pack(gm_ref, Xb, yb),
}

# ------------------------------------------------- BIC and AIC across k
# Unlike inertia, these have a minimum: the likelihood's reward for fitting is
# charged for every parameter it spends. Labels ship so a slider can show the
# clustering each k buys.
em_k = []
for k in range(1, 9):
    gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED).fit(Xb)
    em_k.append({
        "k": k,
        "bic": rnd(gm.bic(Xb), 1),
        "aic": rnd(gm.aic(Xb), 1),
        "labels": [int(v) for v in gm.predict(Xb)],
        "means": [[rnd(a, 3), rnd(b, 3)] for a, b in gm.means_],
        "covs": [cov2(c) for c in gm.covariances_],
    })
    print(f"  k={k}  BIC {em_k[-1]['bic']:9.1f}  AIC {em_k[-1]['aic']:9.1f}")
best_bic = min(em_k, key=lambda r: r["bic"])
print(f"  BIC minimum at k={best_bic['k']}")

# ------------------------------------------ the four covariance wardrobes
# On the sheared data that broke k-means: the whole point of the covariance
# matrix is that the model can learn the shear instead of suffering it.
Xa, ya = make_blobs(n_samples=300, random_state=170)
Xa = Xa @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])
shapes = {
    "points": [[rnd(a, 3), rnd(b, 3)] for a, b in Xa],
    "true_labels": [int(v) for v in ya],
    "variants": [],
}
print("\ncovariance shapes on the sheared blobs (k-means got ARI 0.529 here):")
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(3, covariance_type=ct, n_init=10, random_state=SEED).fit(Xa)
    v = pack(gm, Xa, ya)
    v["type"] = ct
    v["n_params"] = int(gm._n_parameters())
    v["bic"] = rnd(gm.bic(Xa), 1)
    shapes["variants"].append(v)
    print(f"  {ct:<10} ARI {v['ari']:.3f}  params {v['n_params']:>3}  BIC {v['bic']:9.1f}")

# ------------------------------------------------- the gallery, revisited
rs = np.random.RandomState(SEED)
t1 = rs.uniform(0, 2 * np.pi, 150)
t2 = rs.uniform(0, 2 * np.pi, 150)
Xr = np.vstack([
    np.column_stack([1.1 * np.cos(t1), 1.1 * np.sin(t1)]) + rs.normal(0, 0.11, (150, 2)),
    np.column_stack([3.1 * np.cos(t2), 3.1 * np.sin(t2)]) + rs.normal(0, 0.11, (150, 2)),
])
yr = np.array([0] * 150 + [1] * 150)
Xu = np.vstack([
    rs.normal([-2.4, 0], [0.4, 0.4], (60, 2)),
    rs.normal([1.6, 0.4], [1.9, 1.9], (240, 2)),
])
yu = np.array([0] * 60 + [1] * 240)

KM_ARI = {"anisotropic": 0.529, "rings": -0.003, "unequal": 0.306}  # from the k-means article
gallery = []
print("\nthe failure gallery, refit with full-covariance mixtures:")
for name, Xg, yg, k in [("anisotropic", Xa, ya, 3), ("rings", Xr, yr, 2), ("unequal", Xu, yu, 2)]:
    gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED).fit(Xg)
    g = pack(gm, Xg, yg)
    g["name"] = name
    g["k"] = k
    g["km_ari"] = KM_ARI[name]
    g["points"] = [[rnd(a, 3), rnd(b, 3)] for a, b in Xg]
    g["true_labels"] = [int(v) for v in yg]
    gallery.append(g)
    print(f"  {name:<12} k-means ARI {g['km_ari']:>6}  ->  GMM ARI {g['ari']:.3f}")

# ---------------------------------------------------------------- wine, soft
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
yw = wine.target
print("\nwine, 13 standardised dimensions, 178 bottles:")
fits = {}
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(3, covariance_type=ct, n_init=10, random_state=SEED).fit(Xw)
    fits[ct] = gm
    print(f"  {ct:<10} ARI {adjusted_rand_score(yw, gm.predict(Xw)):.3f}  "
          f"params {gm._n_parameters():>4}  BIC {gm.bic(Xw):9.1f}")
best_ct = min(fits, key=lambda ct: fits[ct].bic(Xw))
gm_w = fits[best_ct]
labels_w = gm_w.predict(Xw)
probs_w = gm_w.predict_proba(Xw)
ari_w = adjusted_rand_score(yw, labels_w)

# relabel components to cultivars, as the k-means article did
C = np.zeros((3, 3), dtype=int)
for c, t in zip(labels_w, yw):
    C[c][t] += 1
row, col = linear_sum_assignment(-C)
mapping = dict(zip(row, col))
agree = sum(C[r][mapping[r]] for r in range(3))
inv = np.argsort([mapping[r] for r in range(3)])   # column order: cultivar -> component
probs_rel = probs_w[:, inv]
print(f"  chosen by BIC: {best_ct}. ARI {ari_w:.3f}, {agree} of 178 bottles agree after relabelling")

# the k-means run of the previous article, reproduced, for the borderline six
km_w = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xw)
Ck = np.zeros((3, 3), dtype=int)
for c, t in zip(km_w.labels_, yw):
    Ck[c][t] += 1
rk, ck = linear_sum_assignment(-Ck)
km_map = dict(zip(rk, ck))
km_rel = np.array([km_map[v] for v in km_w.labels_])
km_wrong = [int(i) for i in np.where(km_rel != yw)[0]]
sure = probs_rel.max(axis=1)
print(f"  k-means' {len(km_wrong)} disagreements get mean top membership {sure[km_wrong].mean():.3f} "
      f"from the mixture (page mean {sure.mean():.3f})")

pca = PCA(n_components=2).fit(Xw)
wine_out = {
    "proj": [[rnd(a, 3), rnd(b, 3)] for a, b in pca.transform(Xw)],
    "true": [int(v) for v in yw],
    "gmm": [int(mapping[v]) for v in labels_w],
    "probs": [[rnd(p, 3) for p in row_] for row_ in probs_rel],
    "ari": rnd(ari_w, 3),
    "agree": int(agree),
    "chosen_type": best_ct,
    "compare_ari": {ct: rnd(adjusted_rand_score(yw, fits[ct].predict(Xw)), 3) for ct in fits},
    "compare_bic": {ct: rnd(fits[ct].bic(Xw), 1) for ct in fits},
    "km_wrong": km_wrong,
    "km_ari": 0.897,
    "classes": [str(c) for c in wine.target_names],
}

out = {
    "blobs": blobs,
    "em_k": em_k,
    "shapes": shapes,
    "gallery": gallery,
    "wine": wine_out,
    "versions": {"seed": SEED},
}
(OUT / "mixtures.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'mixtures.json'}  ({(OUT / 'mixtures.json').stat().st_size / 1024:.1f} kB)")
