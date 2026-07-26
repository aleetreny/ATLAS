"""Data for the ATLAS 'DBSCAN and HDBSCAN' web article.

DBSCAN itself runs live in the browser (neighbourhood counting and a BFS are
forty lines of JavaScript), so this file ships the seeded datasets and
scikit-learn reference answers for spot-checking, plus the measurements that
need a real library: what k-means and the mixture score on each dataset (the
failure arc this article closes), the full epsilon story on the rings, the
varying-density trap where no single radius works at all, and HDBSCAN's
answer to it.

The trap construction was found by search, not placed by eye (the lesson of
this project): two tight clusters a bridgeable gap apart, plus one diffuse,
populous cluster far away. The epsilon that keeps the diffuse cluster whole
(0.64+) fuses the tight pair (from 0.48); the best compromise leaves the
diffuse cluster in pieces. HDBSCAN, which compares densities locally, finds
all three at once.

Run from anywhere: `python src/utils/generate_dbscan_data.py`.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dbscan" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19

# ---------------------------------------------------------------- the datasets
rs = np.random.RandomState(SEED)

# two moons: the shape the whole density idea is usually sold on
Xm, ym = make_moons(n_samples=300, noise=0.09, random_state=SEED)
Xm = Xm * 2.0

# the rings from the k-means failure gallery, exact same construction
t1 = rs.uniform(0, 2 * np.pi, 150)
t2 = rs.uniform(0, 2 * np.pi, 150)
Xr = np.vstack([
    np.column_stack([1.1 * np.cos(t1), 1.1 * np.sin(t1)]) + rs.normal(0, 0.11, (150, 2)),
    np.column_stack([3.1 * np.cos(t2), 3.1 * np.sin(t2)]) + rs.normal(0, 0.11, (150, 2)),
])
yr = np.array([0] * 150 + [1] * 150)

# the three blobs plus genuine background noise: 30 points from nowhere
Xb0, yb0 = make_blobs(n_samples=300, centers=[[-3.2, 1.8], [1.8, 3.0], [0.6, -2.6]],
                      cluster_std=[0.85, 1.05, 0.9], random_state=SEED)
Xnoise = rs.uniform([-6.5, -5.5], [5.0, 6.0], (30, 2))
Xn = np.vstack([Xb0, Xnoise])
yn = np.concatenate([yb0, np.full(30, -1)])

# the trap: two tight clusters a bridgeable gap apart, one diffuse far away
rs2 = np.random.RandomState(SEED)
Xv = np.vstack([
    rs2.normal([-0.8, 0.5], 0.14, (100, 2)),
    rs2.normal([0.01, -0.09], 0.14, (100, 2)),
    rs2.normal([4.0, 0.2], 1.15, (150, 2)),
])
yv = np.array([0] * 100 + [1] * 100 + [2] * 150)

DATASETS = {
    "moons": (Xm, ym),
    "rings": (Xr, yr),
    "blobs_noise": (Xn, yn),
    "varying": (Xv, yv),
}
# Round the coordinates BEFORE measuring anything: the browser re-runs DBSCAN
# on the 3-decimal points this file ships, and a distance sitting exactly on
# the radius can flip a border pair, so reference and page must share the
# very same numbers. (Found the honest way: the load-time guard fired.)
DATASETS = {name: (np.round(X, 3), y) for name, (X, y) in DATASETS.items()}
Xm, Xr, Xn, Xv = (DATASETS[n][0] for n in ["moons", "rings", "blobs_noise", "varying"])


def trap_metrics(labels, y):
    """separated: do the two tight clusters end up in different clusters?
    recovered: largest share of the diffuse cluster inside one real cluster."""
    def major(cls):
        ids = labels[(y == cls) & (labels >= 0)]
        return int(np.bincount(ids).argmax()) if len(ids) else -9 - cls
    sep = 1 if major(0) != major(1) else 0
    idsC = labels[(y == 2) & (labels >= 0)]
    rec = float(np.bincount(idsC).max() / (y == 2).sum()) if len(idsC) else 0.0
    return sep, rnd(rec, 2)


# ------------------------------------------ k-means and the mixture, for the arc
print("what the previous two articles' models score here (ARI):")
cross = {}
for name, (X, y) in DATASETS.items():
    k = len(set(y[y >= 0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED).fit(X)
    cross[name] = {
        "km": rnd(adjusted_rand_score(y, km.labels_), 3),
        "gmm": rnd(adjusted_rand_score(y, gm.predict(X)), 3),
    }
    print(f"  {name:<12} k-means {cross[name]['km']:>6}  mixture {cross[name]['gmm']:>6}")

# --------------------------------------------------- reference DBSCAN answers
CHECKS = {
    "moons": [(0.30, 5), (0.45, 5), (0.60, 10)],
    "rings": [(0.30, 4), (0.65, 4), (1.10, 4)],
    "blobs_noise": [(0.50, 5), (0.80, 8)],
    "varying": [(0.16, 5), (0.44, 5), (0.80, 5)],
}
refs = {}
print("\nreference DBSCAN answers (eps, minPts -> clusters / noise / ARI):")
for name, (X, y) in DATASETS.items():
    refs[name] = []
    for eps, mp in CHECKS[name]:
        db = DBSCAN(eps=eps, min_samples=mp).fit(X)
        entry = {
            "eps": eps, "min_samples": mp,
            "n_clusters": int(db.labels_.max()) + 1,
            "n_noise": int((db.labels_ == -1).sum()),
            "ari": rnd(adjusted_rand_score(y, db.labels_), 3),
        }
        if name == "blobs_noise":
            entry["impostors_caught"] = int(((db.labels_ == -1) & (y == -1)).sum())
        refs[name].append(entry)
        print(f"  {name:<12} eps {eps:.2f}  minPts {mp:>2}  ->  {entry['n_clusters']} clusters, "
              f"{entry['n_noise']:>3} noise, ARI {entry['ari']}")

# ------------------------------------------------------- the rings, swept
print("\nepsilon sweep on the rings (minPts 4):")
rings_sweep = []
for eps in np.arange(0.10, 1.61, 0.05):
    db = DBSCAN(eps=eps, min_samples=4).fit(Xr)
    rings_sweep.append({
        "eps": rnd(eps, 2),
        "n_clusters": int(db.labels_.max()) + 1,
        "n_noise": int((db.labels_ == -1).sum()),
        "ari": rnd(adjusted_rand_score(yr, db.labels_), 3),
    })
rings_best = max(rings_sweep, key=lambda s: s["ari"])
print(f"  best: ARI {rings_best['ari']} at eps {rings_best['eps']} "
      f"({rings_best['n_clusters']} clusters, {rings_best['n_noise']} noise)")

# --------------------------------------- the trap, swept: no epsilon works
print("\nepsilon sweep on the varying-density trap (minPts 5):")
trap_sweep = []
for eps in np.arange(0.08, 1.41, 0.04):
    db = DBSCAN(eps=eps, min_samples=5).fit(Xv)
    sep, rec = trap_metrics(db.labels_, yv)
    trap_sweep.append({
        "eps": rnd(eps, 2),
        "n_clusters": int(db.labels_.max()) + 1,
        "n_noise": int((db.labels_ == -1).sum()),
        "ari": rnd(adjusted_rand_score(yv, db.labels_), 3),
        "separated": sep,
        "recovered": rec,
    })
trap_best = max(trap_sweep, key=lambda s: min(s["separated"], s["recovered"]))
print(f"  best joint: eps {trap_best['eps']}  separated {trap_best['separated']}  "
      f"recovered {trap_best['recovered']}  ARI {trap_best['ari']}")

# ------------------------------------------------------------ HDBSCAN's answer
print("\nHDBSCAN on the trap:")
hdb_out = {}
for mcs in [10, 15, 25]:
    hdb = HDBSCAN(min_cluster_size=mcs).fit(Xv)
    labels = hdb.labels_
    sep, rec = trap_metrics(labels, yv)
    hdb_out[str(mcs)] = {
        "labels": [int(v) for v in labels],
        "n_clusters": int(labels.max()) + 1,
        "n_noise": int((labels == -1).sum()),
        "ari": rnd(adjusted_rand_score(yv, labels), 3),
        "separated": sep,
        "recovered": rec,
    }
    print(f"  min_cluster_size {mcs:>2}: {hdb_out[str(mcs)]['n_clusters']} clusters, "
          f"{hdb_out[str(mcs)]['n_noise']:>3} noise, ARI {hdb_out[str(mcs)]['ari']}, "
          f"separated {sep}, recovered {rec}")

out = {
    "datasets": {
        name: {
            "points": [[rnd(a, 3), rnd(b, 3)] for a, b in X],
            "true_labels": [int(v) for v in y],
        } for name, (X, y) in DATASETS.items()
    },
    "cross": cross,
    "refs": refs,
    "rings_sweep": rings_sweep,
    "rings_best": rings_best,
    "trap_sweep": trap_sweep,
    "trap_best": trap_best,
    "hdbscan": hdb_out,
    "versions": {"seed": SEED},
}
(OUT / "density.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'density.json'}  ({(OUT / 'density.json').stat().st_size / 1024:.1f} kB)")
