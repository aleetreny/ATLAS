"""Data for the ATLAS 'Regularization' web article.

Ships raw, standardized matrices so the browser can refit Ridge / Lasso /
ElasticNet live at any alpha. Reference fits from scikit-learn travel with the
data so the JavaScript solvers can be checked against ground truth at runtime.

Convention note (important, and the reason alphas here look different from the
notebook's): every objective on the page uses the 1/(2n) scaling

    Lasso : (1/2n)||y - Xb||^2 + a*||b||_1        (identical to sklearn.Lasso)
    Ridge : (1/2n)||y - Xb||^2 + (a/2)*||b||^2    (sklearn.Ridge with alpha = a*n)

so a single alpha slider is directly comparable between the two penalties.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Lasso, Ridge

OUT = Path(__file__).resolve().parents[2] / "regularization" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=5: round(float(v), n)


# ---------------------------------------------------------------- 1. the monster
# 20 noisy samples of y = sin(x) + x/2 on [0, 5]: the notebook's setup, kept
# verbatim (seed 42) so the article and the notebook tell the same story.
rng = np.random.RandomState(42)
x_sine = np.sort(rng.uniform(0, 5, 20))
y_true = np.sin(x_sine) + x_sine / 2
y_sine = y_true + rng.normal(0, 0.5, 20)

DEGREE = 15
Xp = np.vstack([x_sine ** d for d in range(1, DEGREE + 1)]).T
mu, sd = Xp.mean(0), Xp.std(0)
Xp_std = (Xp - mu) / sd

ref_sine = {}
# the tiny alphas are here on purpose: they are where an ill-conditioned design
# makes solvers disagree, so the browser can check whether its answer is
# trustworthy enough to quote in the prose
for a in [1e-9, 1e-8, 1e-6, 1e-4, 0.001, 0.01, 0.1, 1.0]:
    # sklearn.Ridge alpha maps to our a*n under the 1/(2n) convention
    r = Ridge(alpha=a * len(y_sine)).fit(Xp_std, y_sine)
    l = Lasso(alpha=a, max_iter=200000, tol=1e-10).fit(Xp_std, y_sine)
    ref_sine[str(a)] = {
        "ridge_max_abs": rnd(np.abs(r.coef_).max(), 4),
        "lasso_max_abs": rnd(np.abs(l.coef_).max(), 4),
        "lasso_nonzero": int((np.abs(l.coef_) > 1e-8).sum()),
    }

# The "unpenalized" fit, quoted in the prose as the size of the explosion.
#
# Careful here. A degree-15 monomial basis on 20 points has a condition number
# past what double precision can resolve, so as alpha goes to zero the answer
# stops being well defined and starts depending on the solver: at alpha=1e-12
# scikit-learn reports max|coef| around 100,000 while an independent QR solver
# reports something quite different, and neither is more correct. alpha=1e-9 is
# the smallest penalty where independent solvers still agree to within 1%, so
# that is the number the article quotes.
ols_sine = Ridge(alpha=1e-9 * len(y_sine)).fit(Xp_std, y_sine)

sine_out = {
    "points": [{"x": rnd(a, 4), "y": rnd(b, 4)} for a, b in zip(x_sine, y_sine)],
    "degree": DEGREE,
    "poly_mu": [rnd(v, 8) for v in mu],
    "poly_sd": [rnd(v, 8) for v in sd],
    "ref": {
        "alphas": ref_sine,
        "ols_max_abs": rnd(np.abs(ols_sine.coef_).max(), 1),
        "ols_alpha": 1e-9,
    },
}
(OUT / "sine.json").write_text(json.dumps(sine_out, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------------------ 2. diabetes
# sklearn ships this dataset scaled to unit NORM, not unit variance, which makes
# alpha values behave unintuitively. Re-standardize to mean 0 / std 1 so the
# penalty treats every feature on equal terms (the article says so explicitly).
dia = load_diabetes()
Xd = (dia.data - dia.data.mean(0)) / dia.data.std(0)
yd = dia.target - dia.target.mean()

# honest split for the validation curve
perm = np.random.RandomState(7).permutation(len(yd))
n_test = len(yd) // 4
test_i, train_i = perm[:n_test], perm[n_test:]

ALPHAS = np.logspace(-3, 1.2, 60)
ref_paths = {"alphas": [rnd(a, 6) for a in ALPHAS], "lasso": [], "ridge": [], "curve": []}
for a in ALPHAS:
    la = Lasso(alpha=a, max_iter=200000, tol=1e-10).fit(Xd, yd)
    ri = Ridge(alpha=a * len(yd)).fit(Xd, yd)
    ref_paths["lasso"].append([rnd(c, 3) for c in la.coef_])
    ref_paths["ridge"].append([rnd(c, 3) for c in ri.coef_])
    # train / test RMSE for the U-curve, fitted on the training split only
    lt = Lasso(alpha=a, max_iter=200000, tol=1e-10).fit(Xd[train_i], yd[train_i])
    rmse = lambda idx: float(np.sqrt(np.mean((yd[idx] - lt.predict(Xd[idx])) ** 2)))
    ref_paths["curve"].append({"train": rnd(rmse(train_i), 3), "test": rnd(rmse(test_i), 3)})

NICE = {
    "age": "age", "sex": "sex", "bmi": "bmi", "bp": "blood pressure",
    "s1": "s1 · total chol.", "s2": "s2 · ldl", "s3": "s3 · hdl",
    "s4": "s4 · chol ratio", "s5": "s5 · log triglyc.", "s6": "s6 · glucose",
}
diabetes_out = {
    "features": list(dia.feature_names),
    "labels": [NICE[f] for f in dia.feature_names],
    "X": [[rnd(v, 4) for v in row] for row in Xd],
    "y": [rnd(v, 2) for v in yd],
    "train_idx": sorted(int(i) for i in train_i),
    "test_idx": sorted(int(i) for i in test_i),
    "ref": ref_paths,
}
(OUT / "diabetes.json").write_text(json.dumps(diabetes_out, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------- 3. the correlated twins
# Two nearly identical predictors (r ~ 0.99) plus pure-noise features. Lasso must
# choose one twin arbitrarily; ElasticNet shares the credit. This is the failure
# mode that justifies ElasticNet's existence, which the notebook never showed.
rng2 = np.random.RandomState(11)
n = 120
base = rng2.normal(0, 1, n)
twin_a = base + rng2.normal(0, 0.14, n)
twin_b = base + rng2.normal(0, 0.14, n)
noise_feats = rng2.normal(0, 1, (n, 4))
Xc = np.column_stack([twin_a, twin_b, noise_feats])
Xc = (Xc - Xc.mean(0)) / Xc.std(0)
# only the twins carry signal, split evenly between them
yc = 3.0 * base + rng2.normal(0, 1.0, n)
yc = yc - yc.mean()

la_c = Lasso(alpha=0.15, max_iter=100000).fit(Xc, yc)
en_c = ElasticNet(alpha=0.15, l1_ratio=0.35, max_iter=100000).fit(Xc, yc)
collinear_out = {
    "labels": ["twin A", "twin B", "noise 1", "noise 2", "noise 3", "noise 4"],
    "X": [[rnd(v, 4) for v in row] for row in Xc],
    "y": [rnd(v, 4) for v in yc],
    "corr_twins": rnd(np.corrcoef(Xc[:, 0], Xc[:, 1])[0, 1], 3),
    "ref": {
        "lasso": [rnd(c, 3) for c in la_c.coef_],
        "elasticnet": [rnd(c, 3) for c in en_c.coef_],
    },
}
(OUT / "collinear.json").write_text(json.dumps(collinear_out, separators=(",", ":")), encoding="utf-8")

for f in ["sine.json", "diabetes.json", "collinear.json"]:
    print(f, (OUT / f).stat().st_size, "bytes")
print("OLS max|coef| on the degree-15 monster:", sine_out["ref"]["ols_max_abs"])
print("lasso@0.1 nonzero of 15:", ref_sine["0.1"]["lasso_nonzero"])
print("twins corr:", collinear_out["corr_twins"])
print("lasso twins:", collinear_out["ref"]["lasso"][:2], "| enet twins:", collinear_out["ref"]["elasticnet"][:2])
