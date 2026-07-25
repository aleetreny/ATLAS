"""Generate the data JSONs for the classical-regression article.

Everything the article's JS computes live (OLS, polynomial, Poisson GLM) is ALSO
computed here as a reference so the JS implementations can be validated against
ground truth. PLS/PCA projections are precomputed (no need to ship 13-D math).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

DATA = Path(r"C:\Users\Z0058EYW\Workspace\Atlas\docs\classical-regression\data")
rnd = lambda x, n=4: round(float(x), n)

# ---------- 1. Auto MPG: horsepower vs mpg ----------
mpg = pd.read_csv(DATA / "mpg_raw.csv").dropna(subset=["horsepower", "mpg"])
pts = [{"hp": rnd(h, 1), "mpg": rnd(m, 1), "name": str(n)}
       for h, m, n in zip(mpg.horsepower, mpg.mpg, mpg.name)]

X = mpg[["horsepower"]].values
y = mpg["mpg"].values
ols = LinearRegression().fit(X, y)
r2_lin = r2_score(y, ols.predict(X))

# Polynomial fits, degree 1..9 with train/test split for the overfitting knob
rng = np.random.RandomState(42)
idx = rng.permutation(len(y))
n_test = int(len(y) * 0.25)
test_i, train_i = idx[:n_test], idx[n_test:]
poly_ref = []
for d in range(1, 10):
    Xp = np.hstack([X ** k for k in range(1, d + 1)])
    # normalize columns to keep the normal equation well-conditioned (same as JS will do)
    mu, sd = Xp.mean(0), Xp.std(0)
    Xn = (Xp - mu) / sd
    reg = LinearRegression().fit(Xn[train_i], y[train_i])
    poly_ref.append({
        "degree": d,
        "r2_train": rnd(r2_score(y[train_i], reg.predict(Xn[train_i]))),
        "r2_test": rnd(r2_score(y[test_i], reg.predict(Xn[test_i]))),
    })

deg2 = np.hstack([X, X ** 2])
ols2 = LinearRegression().fit(deg2, y)

mpg_out = {
    "points": pts,
    "meta": {"n": len(pts), "x": "horsepower", "y": "mpg"},
    "ref": {
        "ols": {"intercept": rnd(ols.intercept_), "slope": rnd(ols.coef_[0]), "r2": rnd(r2_lin)},
        "poly2": {"intercept": rnd(ols2.intercept_), "b1": rnd(ols2.coef_[0], 6), "b2": rnd(ols2.coef_[1], 8),
                   "r2": rnd(r2_score(y, ols2.predict(deg2)))},
        "poly_degrees": poly_ref,
        "test_idx": sorted(int(i) for i in test_i),
    },
}
(DATA / "mpg.json").write_text(json.dumps(mpg_out, separators=(",", ":")), encoding="utf-8")

# ---------- 2. Tips: total_bill vs party size (Poisson GLM) ----------
tips = pd.read_csv(DATA / "tips_raw.csv")
tpts = [{"bill": rnd(b, 2), "size": int(s)} for b, s in zip(tips.total_bill, tips["size"])]
Xt = tips[["total_bill"]].values
yt = tips["size"].values
# sklearn PoissonRegressor with tiny alpha ~ unregularized MLE
pois = PoissonRegressor(alpha=1e-12, max_iter=10000).fit(Xt, yt)
ols_t = LinearRegression().fit(Xt, yt)
tips_out = {
    "points": tpts,
    "meta": {"n": len(tpts), "x": "total_bill", "y": "size"},
    "ref": {
        "poisson": {"intercept": rnd(pois.intercept_, 6), "coef": rnd(pois.coef_[0], 6)},
        "ols": {"intercept": rnd(ols_t.intercept_, 6), "slope": rnd(ols_t.coef_[0], 6)},
    },
}
(DATA / "tips.json").write_text(json.dumps(tips_out, separators=(",", ":")), encoding="utf-8")

# ---------- 3. Wine: PCA vs PLS 2-D projections ----------
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
yw = wine.target
pca = PCA(n_components=2).fit(Xw)
Zpca = pca.transform(Xw)
pls = PLSRegression(n_components=2).fit(Xw, yw)
Zpls = pls.transform(Xw)
# correlation matrix for the multicollinearity heatmap (13x13)
corr = np.corrcoef(Xw, rowvar=False)
wine_out = {
    "features": list(wine.feature_names),
    "classes": [str(c) for c in wine.target_names],
    "points": [{"pca": [rnd(a, 3), rnd(b, 3)], "pls": [rnd(c, 3), rnd(d, 3)], "cls": int(t)}
                for (a, b), (c, d), t in zip(Zpca, Zpls, yw)],
    "corr": [[rnd(v, 3) for v in row] for row in corr],
    "ref": {
        "pca_explained_var": [rnd(v) for v in pca.explained_variance_ratio_],
    },
}
(DATA / "wine.json").write_text(json.dumps(wine_out, separators=(",", ":")), encoding="utf-8")

for f in ["mpg.json", "tips.json", "wine.json"]:
    print(f, (DATA / f).stat().st_size, "bytes")
print("OLS ref:", mpg_out["ref"]["ols"])
print("Poisson ref:", tips_out["ref"]["poisson"])
