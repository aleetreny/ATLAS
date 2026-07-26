"""Data for the ATLAS 'SVR and Kernel Ridge' web article, which closes 1.1.

Same cars, same split as the non-parametric article (seed 7, quarter held
out), because this article's whole argument is a comparison: two more ways to
draw a curve through the banana, one that keeps every point on the payroll
(kernel ridge, closed form, runs live in the browser and is checked here) and
one that pays only the points that matter (SVR, whose epsilon-tube fits are
measured on a grid the page replays).

Also measured: the support-vector count as the tube widens (sparsity is the
product), the exact equivalence between kernel ridge and the Gaussian process
posterior mean from the same article-4 machinery, and a 5-fold benchmark on
all numeric features against ridge from article 2.

Run from anywhere: `python src/utils/generate_svr_data.py`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "svr-kernel-ridge" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

# ------------------------------------------------- the cars, article 4's split
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
xtr, ytr = x[train_i], y[train_i]
xte, yte = y[test_i] * 0 + x[test_i], y[test_i]
print(f"cars: {len(y)} rows, {len(xtr)} train / {len(xte)} test (article 4's split)")

# Standardise x for every kernel machine (gamma then means the same thing
# everywhere); predictions stay in mpg.
mu, sd = xtr.mean(), xtr.std()
ztr = (xtr - mu) / sd
zte = (xte - mu) / sd
grid = np.linspace(x.min(), x.max(), 90)
zgrid = (grid - mu) / sd

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

# ----------------------------------------------- the SVR grid the page replays
GAMMA = 0.5
EPS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
CS = [0.1, 1.0, 10.0, 100.0]
svr_grid = []
print(f"\nSVR (rbf, gamma {GAMMA}): {len(EPS)}x{len(CS)} fits")
for eps in EPS:
    for C in CS:
        m = SVR(kernel="rbf", gamma=GAMMA, C=C, epsilon=eps).fit(ztr[:, None], ytr)
        svr_grid.append({
            "eps": eps, "C": C,
            "curve": [rnd(v, 3) for v in m.predict(zgrid[:, None])],
            "sv": [int(i) for i in m.support_],
            "n_sv": int(len(m.support_)),
            "rmse_test": rnd(rmse(yte, m.predict(zte[:, None])), 3),
        })
best = min(svr_grid, key=lambda r: r["rmse_test"])
print(f"  best test RMSE {best['rmse_test']} at eps {best['eps']}, C {best['C']} "
      f"({best['n_sv']} of {len(ztr)} support vectors)")
for eps in EPS:
    row = next(r for r in svr_grid if r["eps"] == eps and r["C"] == 10.0)
    print(f"  eps {eps:>4}: {row['n_sv']:>3} SVs, test RMSE {row['rmse_test']}")

# --------------------------------- kernel ridge: references for the live page
KR_GAMMAS = [0.1, 0.5, 2.0]
KR_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
kr_refs = []
print("\nkernel ridge references (browser must reproduce):")
for g in KR_GAMMAS:
    for lam in KR_LAMBDAS:
        m = KernelRidge(kernel="rbf", gamma=g, alpha=lam).fit(ztr[:, None], ytr)
        kr_refs.append({
            "gamma": g, "lambda": lam,
            "rmse_test": rnd(rmse(yte, m.predict(zte[:, None])), 3),
            "curve": [rnd(v, 3) for v in m.predict(zgrid[:, None])],
        })
kb = min(kr_refs, key=lambda r: r["rmse_test"])
print(f"  best test RMSE {kb['rmse_test']} at gamma {kb['gamma']}, lambda {kb['lambda']}")

# ------------------------- the identity: kernel ridge IS the GP posterior mean
g_eq, lam_eq = 0.5, 0.1
kr_eq = KernelRidge(kernel="rbf", gamma=g_eq, alpha=lam_eq).fit(ztr[:, None], ytr)
# Same kernel (length_scale = 1/sqrt(2 gamma)), same ridge on the diagonal,
# same zero-mean prior: the two solve literally the same linear system.
ls = 1.0 / np.sqrt(2 * g_eq)
gp = GaussianProcessRegressor(kernel=RBF(length_scale=ls), alpha=lam_eq,
                              optimizer=None).fit(ztr[:, None], ytr)
gp_mean = gp.predict(zgrid[:, None])
kr_curve = kr_eq.predict(zgrid[:, None])
gap = float(np.max(np.abs(gp_mean - kr_curve)))
print(f"\nkernel ridge vs GP posterior mean, same kernel and noise: max gap {gap:.2e} mpg")
identity = {
    "gamma": g_eq, "lambda": lam_eq,
    "kr_curve": [rnd(v, 3) for v in kr_curve],
    "gp_mean": [rnd(v, 3) for v in gp_mean],
    "max_gap": rnd(gap, 10),
}

# --------------------------------------------------- benchmark, all features
FEATS = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
df = mpg.dropna(subset=FEATS + ["mpg"])
Xf = df[FEATS].to_numpy(float)
yf = df["mpg"].to_numpy(float)
MODELS = {
    "ridge (article 2)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "kernel ridge rbf": make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", gamma=0.1, alpha=0.1)),
    "SVR rbf": make_pipeline(StandardScaler(), SVR(kernel="rbf", gamma=0.1, C=10.0, epsilon=1.0)),
}
kf = KFold(n_splits=5, shuffle=True, random_state=7)
bench = []
print("\n5-fold benchmark on six numeric features:")
for name, model in MODELS.items():
    errs = []
    t0 = time.perf_counter()
    n_sv = []
    for tr, te in kf.split(Xf):
        model.fit(Xf[tr], yf[tr])
        errs.append(rmse(yf[te], model.predict(Xf[te])))
        if "SVR" in name:
            n_sv.append(len(model[-1].support_))
    secs = time.perf_counter() - t0
    bench.append({
        "name": name,
        "rmse": rnd(float(np.mean(errs)), 3),
        "rmse_sd": rnd(float(np.std(errs)), 3),
        "seconds": rnd(secs, 3),
        "n_sv": int(np.mean(n_sv)) if n_sv else None,
        "n_train": int(len(Xf) * 0.8),
    })
    print(f"  {name:<18} RMSE {bench[-1]['rmse']} ± {bench[-1]['rmse_sd']}  ({secs:.2f}s"
          f"{', ~' + str(bench[-1]['n_sv']) + ' SVs' if n_sv else ''})")

out = {
    "x_train": [rnd(v, 4) for v in ztr],       # standardised, what the page fits on
    "y_train": [rnd(v, 2) for v in ytr],
    "x_test": [rnd(v, 4) for v in zte],
    "y_test": [rnd(v, 2) for v in yte],
    "x_mu": rnd(mu, 4), "x_sd": rnd(sd, 4),
    "grid": [rnd(v, 2) for v in grid],
    "zgrid": [rnd(v, 4) for v in zgrid],
    "svr_gamma": GAMMA,
    "eps_values": EPS,
    "c_values": CS,
    "svr_grid": svr_grid,
    "kr_refs": kr_refs,
    "identity": identity,
    "bench": bench,
    "versions": {"seed": 7},
}
(OUT / "machines.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'machines.json'}  ({(OUT / 'machines.json').stat().st_size / 1024:.1f} kB)")
