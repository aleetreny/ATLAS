"""Data for the ATLAS 'Robust Regression' web article.

Primary dataset: the Hertzsprung-Russell diagram of star cluster CYG OB1, 47
stars, from Rousseeuw & Leroy, "Robust Regression and Outlier Detection" (1987),
p.27 table 3. Values cross-checked between two independent publications of the
table (robustbase::starsCYG and HSAUR::CYGOB1) and found identical.

This dataset is the article's centrepiece because the failure is real, famous
and not manufactured: ordinary least squares reports that hotter stars are
DIMMER, which is astrophysically backwards, because four genuine red giants sit
at low temperature and high luminosity and drag the line over. The outliers are
not data-entry errors. They are real stars that do not belong to the
relationship being modelled, which is exactly the distinction the article is
about.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import theilslopes
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)

OUT = Path(__file__).resolve().parents[2] / "robust-regression" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

# ------------------------------------------------------------------ the stars
LOG_TE = [
    4.37, 4.56, 4.26, 4.56, 4.30, 4.46, 3.84, 4.57, 4.26, 4.37,
    3.49, 4.43, 4.48, 4.01, 4.29, 4.42, 4.23, 4.42, 4.23, 3.49,
    4.29, 4.29, 4.42, 4.49, 4.38, 4.42, 4.29, 4.38, 4.22, 3.48,
    4.38, 4.56, 4.45, 3.49, 4.23, 4.62, 4.53, 4.45, 4.53, 4.43,
    4.38, 4.45, 4.50, 4.45, 4.55, 4.45, 4.42,
]
LOG_LIGHT = [
    5.23, 5.74, 4.93, 5.74, 5.19, 5.46, 4.65, 5.27, 5.57, 5.12,
    5.73, 5.45, 5.42, 4.05, 4.26, 4.58, 3.94, 4.18, 4.18, 5.89,
    4.38, 4.22, 4.42, 4.85, 5.02, 4.66, 4.66, 4.90, 4.39, 6.05,
    4.42, 5.10, 5.22, 6.29, 4.34, 5.62, 5.10, 5.22, 5.18, 5.57,
    4.62, 5.06, 5.34, 5.34, 5.54, 4.98, 4.50,
]
assert len(LOG_TE) == len(LOG_LIGHT) == 47

# the four red giants, 1-indexed 11, 20, 30, 34 in the source table
GIANTS = [10, 19, 29, 33]

X = np.array(LOG_TE).reshape(-1, 1)
y = np.array(LOG_LIGHT)
clean = np.array([i for i in range(47) if i not in GIANTS])


def slope_of(model, Xa, ya):
    model.fit(Xa, ya)
    if hasattr(model, "estimator_"):
        return float(model.estimator_.coef_[0]), float(model.estimator_.intercept_)
    return float(model.coef_[0]), float(model.intercept_)


fits = {}
fits["ols"] = slope_of(LinearRegression(), X, y)
fits["ols_giants_removed"] = slope_of(LinearRegression(), X[clean], y[clean])
fits["ransac"] = slope_of(RANSACRegressor(random_state=0, residual_threshold=0.35), X, y)
fits["huber"] = slope_of(HuberRegressor(epsilon=1.35, max_iter=2000), X, y)

# Two different estimators both answer to the name "Theil-Sen".
#
# The classic one, and the one this article defines and implements, is the
# median of the slopes of all pairs of points; scipy.stats.theilslopes computes
# exactly that. scikit-learn's TheilSenRegressor instead takes the SPATIAL
# median of least-squares fits over random subsamples, which generalises to
# many features but is not the same number in one dimension. On this data the
# gap is real and large: 1.727 against 1.493. The browser is checked against
# the classic value, because that is what it claims to compute.
ts = theilslopes(y, X.ravel())
fits["theilsen"] = (float(ts[0]), float(ts[1]))
fits["theilsen_sklearn_spatial"] = slope_of(TheilSenRegressor(random_state=0), X, y)

# leverage: how much pull each x position has, regardless of its y
mx = X.ravel().mean()
sxx = ((X.ravel() - mx) ** 2).sum()
lev = 1 / 47 + (X.ravel() - mx) ** 2 / sxx
lev_cutoff = 2 * 2 / 47  # 2p/n with p = 2 parameters

stars_out = {
    "points": [
        {"x": rnd(a, 2), "y": rnd(b, 2), "giant": i in GIANTS, "id": i + 1}
        for i, (a, b) in enumerate(zip(LOG_TE, LOG_LIGHT))
    ],
    "giants": GIANTS,
    "leverage": [rnd(v, 4) for v in lev],
    "leverage_cutoff": rnd(lev_cutoff, 4),
    "ref": {k: {"slope": rnd(v[0]), "intercept": rnd(v[1])} for k, v in fits.items()},
}
(OUT / "stars.json").write_text(json.dumps(stars_out, separators=(",", ":")), encoding="utf-8")

print("=== CYG OB1: does OLS really report that hotter stars are dimmer? ===")
for k, (s, b) in fits.items():
    print(f"  {k:22s} slope = {s:+.4f}   intercept = {b:+.3f}")
print(f"  four giants at 1-indexed rows: {[i+1 for i in GIANTS]}")
print(f"  leverage cutoff 2p/n = {lev_cutoff:.4f}; giants' leverage = "
      f"{[round(lev[i],3) for i in GIANTS]}")
print(f"  max leverage among the other 43 = {max(lev[i] for i in clean):.3f}")


# ------------------------------------------- settling the Huber confound
# The notebook this article grows out of concluded that Huber FAILS on a dense
# structural-outlier cluster. But its implementation used a fixed delta of 1.5
# on unscaled data (y spanning about +-100) with 2000 gradient descent steps at
# lr 0.01. If nearly every residual starts in the linear regime and the
# optimizer never converges, a flat line proves nothing about the loss.
from sklearn.datasets import make_regression

Xr, yr = make_regression(n_samples=200, n_features=1, noise=10.0, random_state=42)
rs = np.random.RandomState(123)
Xr[:50] = 3 + 0.5 * rs.standard_normal((50, 1))
yr[:50] = -3 + 10 * rs.standard_normal(50)
truth_slope, truth_int = slope_of(LinearRegression(), Xr[50:], yr[50:])


def gd_huber(Xa, ya, delta, lr, iters):
    w, b = 0.0, 0.0
    n = len(ya)
    for _ in range(iters):
        r = ya - (Xa.ravel() * w + b)
        small = np.abs(r) <= delta
        gw = -(Xa.ravel()[small] * r[small]).sum() - (delta * np.sign(r[~small]) * Xa.ravel()[~small]).sum()
        gb = -r[small].sum() - (delta * np.sign(r[~small])).sum()
        w -= lr * gw / n
        b -= lr * gb / n
    return w, b


naive = gd_huber(Xr, yr, 1.5, 0.01, 2000)
converged = gd_huber(Xr, yr, 1.5, 0.01, 400000)
xm, xs_ = Xr.mean(), Xr.std()
ym, ys_ = yr.mean(), yr.std()
Xs = (Xr - xm) / xs_
ys = (yr - ym) / ys_
ws, bs = gd_huber(Xs, ys, 1.35, 0.05, 200000)
scaled_slope = ws * ys_ / xs_

print("\n=== Is Huber's failure real, or an artefact of the optimizer? ===")
print(f"  clean-data truth (points 50+)      slope = {truth_slope:+.3f}")
print(f"  OLS on everything                  slope = {slope_of(LinearRegression(), Xr, yr)[0]:+.3f}")
print(f"  naive GD huber (2k iters, raw)     slope = {naive[0]:+.3f}   <- the notebook's fit")
print(f"  same GD run to convergence         slope = {converged[0]:+.3f}")
print(f"  GD huber, standardized, converged  slope = {scaled_slope:+.3f}")
print(f"  sklearn HuberRegressor (raw)       slope = {slope_of(HuberRegressor(max_iter=5000), Xr, yr)[0]:+.3f}")
print(f"  sklearn TheilSen                   slope = {slope_of(TheilSenRegressor(random_state=0), Xr, yr)[0]:+.3f}")
print(f"  sklearn RANSAC                     slope = {slope_of(RANSACRegressor(random_state=0), Xr, yr)[0]:+.3f}")

print("\n  huber slope as a function of delta (standardized, converged):")
for d in [0.1, 0.35, 0.75, 1.35, 3.0, 10.0]:
    wd, _ = gd_huber(Xs, ys, d, 0.05, 120000)
    print(f"    delta={d:<6}  slope = {wd * ys_ / xs_:+.3f}")

for f in ["stars.json"]:
    print(f"\n{f}: {(OUT / f).stat().st_size} bytes")
