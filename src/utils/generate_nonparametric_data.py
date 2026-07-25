"""Data for the ATLAS 'Non-Parametric Regression' web article.

Two stages, deliberately.

The one-dimensional stage reuses the Auto MPG data from the first article, so
the reader meets the same banana-shaped cloud they already watched a straight
line fail on. There the question was which equation to assume; here it is what
happens when you assume none at all.

The spatial stage uses California Housing restricted to longitude and latitude,
where "nearby" has a physical meaning and a Gaussian process can draw a map of
its own ignorance.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "non-parametric-regression" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)


# ----------------------------------------------------------------- 1-D: mpg
mpg = pd.read_csv(ROOT / "datasets" / "mpg.csv").dropna(subset=["horsepower", "mpg"])
x = mpg["horsepower"].to_numpy(float)
y = mpg["mpg"].to_numpy(float)
order = np.argsort(x)
x, y = x[order], y[order]

rs = np.random.RandomState(7)
perm = rs.permutation(len(y))
n_test = len(y) // 4
test_i = np.sort(perm[:n_test])
train_i = np.sort(perm[n_test:])

# reference KNN fits, so the browser can check its own implementation
grid = np.linspace(x.min(), x.max(), 60)
knn_ref = {}
for k in [1, 5, 15, 50]:
    for w in ["uniform", "distance"]:
        m = KNeighborsRegressor(n_neighbors=k, weights=w).fit(x[train_i, None], y[train_i])
        pred = m.predict(x[test_i, None])
        knn_ref[f"{k}_{w}"] = {
            "rmse_test": rnd(float(np.sqrt(np.mean((y[test_i] - pred) ** 2))), 3),
            "curve": [rnd(v, 3) for v in m.predict(grid[:, None])],
        }


def nadaraya_watson(xt, yt, xq, h):
    d = xq[:, None] - xt[None, :]
    w = np.exp(-0.5 * (d / h) ** 2)
    s = w.sum(1)
    out = np.where(s > 1e-12, (w * yt[None, :]).sum(1) / np.maximum(s, 1e-12), yt.mean())
    return out


def local_linear(xt, yt, xq, h):
    """Weighted least squares in a moving window, CENTRED on the query point.
    Centring is what the companion notebook's comment claimed but its code
    never did, and it is exactly what removes the boundary bias."""
    out = np.empty(len(xq))
    for i, q in enumerate(xq):
        dx = xt - q
        w = np.exp(-0.5 * (dx / h) ** 2)
        sw = w.sum()
        if sw < 1e-12:
            out[i] = yt.mean()
            continue
        # solve [[S0,S1],[S1,S2]] [a,b] = [T0,T1] with dx centred at the query
        s0, s1, s2 = w.sum(), (w * dx).sum(), (w * dx**2).sum()
        t0, t1 = (w * yt).sum(), (w * dx * yt).sum()
        det = s0 * s2 - s1 * s1
        out[i] = t0 / s0 if abs(det) < 1e-12 else (t0 * s2 - t1 * s1) / det
    return out


# the boundary-bias demonstration, measured rather than asserted
bands = [0.06, 0.12, 0.25]
h_grid = [(x.max() - x.min()) * b for b in bands]
kernel_ref = {}
for b, h in zip(bands, h_grid):
    nw = nadaraya_watson(x[train_i], y[train_i], x[test_i], h)
    ll = local_linear(x[train_i], y[train_i], x[test_i], h)
    edge = (x[test_i] < np.percentile(x, 10)) | (x[test_i] > np.percentile(x, 90))
    kernel_ref[str(b)] = {
        "h": rnd(h, 3),
        "nw_rmse": rnd(float(np.sqrt(np.mean((y[test_i] - nw) ** 2))), 3),
        "ll_rmse": rnd(float(np.sqrt(np.mean((y[test_i] - ll) ** 2))), 3),
        "nw_rmse_edges": rnd(float(np.sqrt(np.mean((y[test_i][edge] - nw[edge]) ** 2))), 3),
        "ll_rmse_edges": rnd(float(np.sqrt(np.mean((y[test_i][edge] - ll[edge]) ** 2))), 3),
    }

mpg_out = {
    "points": [{"x": rnd(a, 1), "y": rnd(b, 1)} for a, b in zip(x, y)],
    "train_idx": [int(i) for i in train_i],
    "test_idx": [int(i) for i in test_i],
    "grid": [rnd(v, 2) for v in grid],
    "ref": {"knn": knn_ref, "kernel": kernel_ref},
}
(OUT / "mpg.json").write_text(json.dumps(mpg_out, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------- the curse of dimensionality
# Two exact statements, no simulation needed for the first:
#   to enclose a fraction r of a uniform cube in d dimensions, a sub-cube needs
#   edge length r**(1/d), which approaches 1 fast.
curse = {"edge": [], "nn_distance": []}
for d in [1, 2, 3, 5, 10, 20, 50, 100]:
    curse["edge"].append({"d": d, "r10": rnd(0.10 ** (1 / d), 4), "r01": rnd(0.01 ** (1 / d), 4)})

# and the median distance to the nearest of n=500 uniform neighbours, simulated
rs2 = np.random.RandomState(3)
for d in [1, 2, 3, 5, 10, 20, 50, 100]:
    n = 500
    dists = []
    for _ in range(30):
        P = rs2.random_sample((n, d))
        q = rs2.random_sample(d)
        dd = np.sqrt(((P - q) ** 2).sum(1))
        dists.append((dd.min(), np.median(dd)))
    dists = np.array(dists)
    curse["nn_distance"].append(
        {"d": d, "nearest": rnd(dists[:, 0].mean(), 4), "median": rnd(dists[:, 1].mean(), 4),
         "ratio": rnd(dists[:, 0].mean() / dists[:, 1].mean(), 4)}
    )
(OUT / "curse.json").write_text(json.dumps(curse, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------- 2-D: spatial payoff
cal = fetch_california_housing()
rs3 = np.random.RandomState(42)
idx = rs3.choice(len(cal.target), 1200, replace=False)
XY = cal.data[idx][:, [7, 6]]  # longitude, latitude
val = cal.target[idx]

# standardize the coordinates: a degree of longitude is about 88 km at this
# latitude against 111 km for a degree of latitude, so raw euclidean distance
# on lon/lat is anisotropic. The article says so out loud.
mu, sd = XY.mean(0), XY.std(0)
XYs = (XY - mu) / sd

gp = GaussianProcessRegressor(
    kernel=ConstantKernel(1.0) * RBF(length_scale=0.3) + WhiteKernel(noise_level=0.3),
    normalize_y=True,
    n_restarts_optimizer=2,
    random_state=0,
).fit(XYs[:400], val[:400])

gx = np.linspace(XYs[:, 0].min(), XYs[:, 0].max(), 60)
gy = np.linspace(XYs[:, 1].min(), XYs[:, 1].max(), 60)
GX, GY = np.meshgrid(gx, gy)
mean, std = gp.predict(np.column_stack([GX.ravel(), GY.ravel()]), return_std=True)

spatial_out = {
    "points": [{"lon": rnd(a, 3), "lat": rnd(b, 3), "v": rnd(c, 3)} for (a, b), c in zip(XY[:400], val[:400])],
    "grid": {
        "lon": [rnd(v * sd[0] + mu[0], 3) for v in gx],
        "lat": [rnd(v * sd[1] + mu[1], 3) for v in gy],
        "mean": [rnd(v, 3) for v in mean],
        "std": [rnd(v, 3) for v in std],
    },
    "kernel": str(gp.kernel_),
}
(OUT / "spatial.json").write_text(json.dumps(spatial_out, separators=(",", ":")), encoding="utf-8")

print("=== boundary bias: does local linear actually beat the kernel average at the edges? ===")
for b, r in kernel_ref.items():
    print(f"  h = {r['h']:6.1f} ({b} of the range)")
    print(f"     overall RMSE   nadaraya-watson {r['nw_rmse']:.3f}   local linear {r['ll_rmse']:.3f}")
    print(f"     EDGES only     nadaraya-watson {r['nw_rmse_edges']:.3f}   local linear {r['ll_rmse_edges']:.3f}")
print()
print("=== curse of dimensionality: edge length of a sub-cube holding 10% of the data ===")
for e in curse["edge"]:
    print(f"  d = {e['d']:3d}   edge {e['r10']:.3f} of the full range")
print()
print("=== nearest vs median neighbour distance, 500 uniform points ===")
for e in curse["nn_distance"]:
    print(f"  d = {e['d']:3d}   nearest {e['nearest']:.3f}   median {e['median']:.3f}   ratio {e['ratio']:.3f}")
print()
print("fitted GP kernel:", gp.kernel_)
for f in ["mpg.json", "curse.json", "spatial.json"]:
    print(f, (OUT / f).stat().st_size, "bytes")
