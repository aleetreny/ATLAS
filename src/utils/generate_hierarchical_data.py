"""Data for the ATLAS 'Agglomerative Clustering and BIRCH' web article.

The centrepiece is the full ward dendrogram of the 178 wines with a cut line
the reader drags; the browser rebuilds every tree from the scipy linkage
matrices shipped here, and ALSO runs its own Lance-Williams agglomeration to
check itself against scipy at load time. Coordinates are rounded before
anything is measured, so page and reference share the same numbers (the
DBSCAN article's load-time guard taught that lesson).

Also measured here: ARI against the cultivars for every linkage at every cut
k = 2..8 (the linkage choice dwarfs the choice of k), a 2-D bridge dataset
where single linkage chains across and ward does not, and the scaling story
that motivates BIRCH: scipy's ward against BIRCH's summarise-then-cluster on
growing blob datasets.

Run from anywhere: `python src/utils/generate_hierarchical_data.py`.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.datasets import load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hierarchical" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19

# ------------------------------------------------------------------- the wine
wine = load_wine()
Xw = np.round(StandardScaler().fit_transform(wine.data), 4)   # rounded FIRST
yw = wine.target
pca = PCA(n_components=2).fit(Xw)
proj = pca.transform(Xw)

LINKAGES = ["ward", "complete", "average", "single"]
wine_out = {
    "features": [[rnd(v) for v in row] for row in Xw],
    "true": [int(v) for v in yw],
    "proj": [[rnd(a, 3), rnd(b, 3)] for a, b in proj],
    "classes": [str(c) for c in wine.target_names],
    "linkages": {},
}
print("wine, 178 bottles, 13 standardised dimensions:")
for method in LINKAGES:
    Z = linkage(Xw, method=method)
    aris = {}
    for k in range(2, 9):
        labels = fcluster(Z, k, criterion="maxclust")
        aris[str(k)] = rnd(adjusted_rand_score(yw, labels), 3)
    best_k = max(aris, key=lambda k2: aris[k2])
    wine_out["linkages"][method] = {
        "Z": [[int(a), int(b), rnd(d, 4), int(s)] for a, b, d, s in Z],
        "ari_by_k": aris,
        "best_k": int(best_k),
    }
    print(f"  {method:<9} ARI at k=3: {aris['3']:>6}   best: {aris[best_k]} at k={best_k}")

# --------------------------------------------------------- the bridge dataset
# Two clear blobs and a thin chain of stepping stones between them: single
# linkage's downfall since 1951. Rounded before measuring, like everything.
rs = np.random.RandomState(SEED)
A = rs.normal([-3.2, 0], 0.55, (55, 2))
B = rs.normal([3.2, 0.4], 0.55, (55, 2))
bridge = np.column_stack([np.linspace(-2.2, 2.2, 12), np.zeros(12) + rs.normal(0, 0.05, 12)])
Xc = np.round(np.vstack([A, B, bridge]), 3)
yc = np.array([0] * 55 + [1] * 55 + [2] * 12)

chain_out = {"points": [[rnd(a, 3), rnd(b, 3)] for a, b in Xc],
             "true_labels": [int(v) for v in yc], "linkages": {}}
print("\nthe bridge, cut at k=2 (the 12 stepping stones count as their own class):")
for method in LINKAGES:
    Z = linkage(Xc, method=method)
    labels2 = fcluster(Z, 2, criterion="maxclust")
    # how the two blobs fare: are A and B separated at k=2?
    majA = np.bincount(labels2[yc == 0]).argmax()
    majB = np.bincount(labels2[yc == 1]).argmax()
    chain_out["linkages"][method] = {
        "Z": [[int(a), int(b), rnd(d, 4), int(s)] for a, b, d, s in Z],
        "labels_k2": [int(v) for v in labels2],
        "separated": int(majA != majB),
    }
    print(f"  {method:<9} A and B separated: {'yes' if majA != majB else 'NO, chained across the bridge'}")

# ----------------------------------------------------------- the scaling story
print("\nscaling: full ward against BIRCH summarise-then-cluster (5 blobs, 10-D):")
scale_rows = []
for n in [2000, 5000, 10000, 20000]:
    Xs, ys = make_blobs(n_samples=n, centers=5, n_features=10, cluster_std=1.5, random_state=SEED)
    t0 = time.perf_counter()
    agg = AgglomerativeClustering(n_clusters=5, linkage="ward").fit(Xs)
    t_agg = time.perf_counter() - t0
    t0 = time.perf_counter()
    bir = Birch(n_clusters=5, threshold=3.5).fit(Xs)
    t_bir = time.perf_counter() - t0
    scale_rows.append({
        "n": n,
        "agg_seconds": rnd(t_agg, 2),
        "birch_seconds": rnd(t_bir, 2),
        "agg_ari": rnd(adjusted_rand_score(ys, agg.labels_), 3),
        "birch_ari": rnd(adjusted_rand_score(ys, bir.labels_), 3),
        "birch_subclusters": int(len(bir.subcluster_centers_)),
    })
    r = scale_rows[-1]
    print(f"  n={n:>6,}  ward {r['agg_seconds']:>6.2f}s (ARI {r['agg_ari']})   "
          f"birch {r['birch_seconds']:>5.2f}s (ARI {r['birch_ari']}, {r['birch_subclusters']} subclusters)")

# birch alone where full ward is no longer reasonable
n_big = 200000
Xs, ys = make_blobs(n_samples=n_big, centers=5, n_features=10, cluster_std=1.5, random_state=SEED)
t0 = time.perf_counter()
bir = Birch(n_clusters=5, threshold=3.5).fit(Xs)
t_big = time.perf_counter() - t0
big = {
    "n": n_big,
    "birch_seconds": rnd(t_big, 2),
    "birch_ari": rnd(adjusted_rand_score(ys, bir.labels_), 3),
    "birch_subclusters": int(len(bir.subcluster_centers_)),
}
print(f"  n={n_big:,}  birch {big['birch_seconds']:.2f}s (ARI {big['birch_ari']}, {big['birch_subclusters']} subclusters)")

out = {
    "wine": wine_out,
    "chain": chain_out,
    "scale": scale_rows,
    "scale_big": big,
    "versions": {"seed": SEED},
}
(OUT / "trees.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'trees.json'}  ({(OUT / 'trees.json').stat().st_size / 1024:.1f} kB)")
