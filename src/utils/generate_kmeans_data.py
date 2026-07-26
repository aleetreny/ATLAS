"""Data for the ATLAS 'K-Means' web article, the first of module 2.

Almost everything on the page runs live in the browser: Lloyd's algorithm is
thirty lines of JavaScript and watching it move is the article. What this file
provides is (a) the seeded datasets, so what the text describes is what is on
screen, and (b) scikit-learn reference results so the browser implementation
can be checked against the real thing at load time.

The one genuinely offline measurement is the wine experiment: the same 178
bottles from the first article, clustered blind on all 13 standardised
features, then compared against the cultivar labels k-means never saw.

Run from anywhere: `python src/utils/generate_kmeans_data.py`.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kmeans" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19

# ------------------------------------------------------------- the main blobs
# Three clusters a human can see, so every claim about the algorithm can be
# checked by eye before it is checked by number.
Xb, yb = make_blobs(n_samples=300, centers=[[-3.2, 1.8], [1.8, 3.0], [0.6, -2.6]],
                    cluster_std=[0.85, 1.05, 0.9], random_state=SEED)

km_ref = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xb)
print(f"blobs: 300 points, sklearn inertia {km_ref.inertia_:.2f}")

blobs = {
    "points": [[rnd(a, 3), rnd(b, 3)] for a, b in zip(Xb[:, 0], Xb[:, 1])],
    "true_labels": [int(v) for v in yb],
    "ref_inertia": rnd(km_ref.inertia_, 2),
    "ref_centers": [[rnd(c[0], 3), rnd(c[1], 3)] for c in km_ref.cluster_centers_],
}

# ----------------------------------------------- random init vs k-means++
# 300 restarts each, single run per restart (n_init=1), final inertia recorded.
# The article's claim is not "random init fails", it is "random init sometimes
# lands in a bad basin and k-means++ almost never does", which is a pair of
# DISTRIBUTIONS and has to be shown as one.
runs = {"random": [], "kmeanspp": []}
for i in range(300):
    for name, init in [("random", "random"), ("kmeanspp", "k-means++")]:
        km = KMeans(n_clusters=3, init=init, n_init=1, random_state=1000 + i).fit(Xb)
        runs[name].append(rnd(km.inertia_, 2))
for name, v in runs.items():
    v = np.array(v)
    good = float((v < km_ref.inertia_ * 1.01).mean())
    print(f"  {name:<9} median {np.median(v):8.1f}  worst {v.max():8.1f}  lands in the best basin {good * 100:.0f}% of the time")

init_runs = {k: sorted(v) for k, v in runs.items()}

# ------------------------------------------------------------- choosing k
# Inertia always falls with k, which is why the elbow is a squint; the
# silhouette actually peaks. Both measured on the same blobs.
choose_k = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xb)
    choose_k.append({
        "k": k,
        "inertia": rnd(km.inertia_, 1),
        "silhouette": rnd(silhouette_score(Xb, km.labels_), 4),
    })
    print(f"  k={k}  inertia {km.inertia_:8.1f}  silhouette {choose_k[-1]['silhouette']:.4f}")
best_sil = max(choose_k, key=lambda r: r["silhouette"])
print(f"  silhouette peaks at k={best_sil['k']}")

# ------------------------------------------------------------ failure gallery
# Three shapes that violate k-means' assumptions in three different ways.
rs = np.random.RandomState(SEED)

# anisotropic: the classic shear from the scikit-learn documentation, at a seed
# where the failure actually happens (with a friendlier seed the sheared blobs
# stay separable by spheres and the panel would demonstrate nothing).
Xa, ya = make_blobs(n_samples=300, random_state=170)
Xa = Xa @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])

# rings
t1 = rs.uniform(0, 2 * np.pi, 150)
t2 = rs.uniform(0, 2 * np.pi, 150)
Xr = np.vstack([
    np.column_stack([1.1 * np.cos(t1), 1.1 * np.sin(t1)]) + rs.normal(0, 0.11, (150, 2)),
    np.column_stack([3.1 * np.cos(t2), 3.1 * np.sin(t2)]) + rs.normal(0, 0.11, (150, 2)),
])
yr = np.array([0] * 150 + [1] * 150)

# unequal sizes and variances
Xu = np.vstack([
    rs.normal([-2.4, 0], [0.4, 0.4], (60, 2)),
    rs.normal([1.6, 0.4], [1.9, 1.9], (240, 2)),
])
yu = np.array([0] * 60 + [1] * 240)

gallery = []
for name, Xg, yg, k in [("anisotropic", Xa, ya, 3), ("rings", Xr, yr, 2), ("unequal", Xu, yu, 2)]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xg)
    ari = adjusted_rand_score(yg, km.labels_)
    gallery.append({
        "name": name,
        "k": k,
        "points": [[rnd(a, 3), rnd(b, 3)] for a, b in Xg],
        "true_labels": [int(v) for v in yg],
        "km_labels": [int(v) for v in km.labels_],
        "centers": [[rnd(c[0], 3), rnd(c[1], 3)] for c in km.cluster_centers_],
        "ari": rnd(ari, 3),
    })
    print(f"  {name:<12} ARI against the truth: {ari:.3f}")

# ---------------------------------------------------------------- wine, blind
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
yw = wine.target
km_w = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xw)
ari_w = adjusted_rand_score(yw, km_w.labels_)

# match cluster ids to cultivars by majority vote, for the confusion table
from scipy.optimize import linear_sum_assignment  # noqa: E402

C = np.zeros((3, 3), dtype=int)
for c, t in zip(km_w.labels_, yw):
    C[c][t] += 1
row, col = linear_sum_assignment(-C)
mapping = dict(zip(row, col))
agree = sum(C[r][mapping[r]] for r in range(3))
print(f"\nwine, clustered blind on 13 features: ARI {ari_w:.3f}, {agree} of {len(yw)} bottles"
      f" match their cultivar after relabelling")

pca = PCA(n_components=2).fit(Xw)
Zw = pca.transform(Xw)
Zc = pca.transform(km_w.cluster_centers_)
wine_out = {
    "proj": [[rnd(a, 3), rnd(b, 3)] for a, b in Zw],
    "true": [int(v) for v in yw],
    "km": [int(mapping[v]) for v in km_w.labels_],  # relabelled to match cultivars
    "centers": [[rnd(a, 3), rnd(b, 3)] for a, b in Zc],
    "ari": rnd(ari_w, 3),
    "agree": int(agree),
    "n": int(len(yw)),
    "confusion": [[int(C[r][c]) for c in range(3)] for r in range(3)],
    "mapping": [int(mapping[r]) for r in range(3)],
    "classes": [str(c) for c in wine.target_names],
}

out = {
    "blobs": blobs,
    "init_runs": init_runs,
    "choose_k": choose_k,
    "gallery": gallery,
    "wine": wine_out,
    "versions": {"seed": SEED},
}
(OUT / "clusters.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'clusters.json'}  ({(OUT / 'clusters.json').stat().st_size / 1024:.1f} kB)")
