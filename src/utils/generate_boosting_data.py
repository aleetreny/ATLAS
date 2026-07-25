"""Data for the ATLAS 'Gradient Boosting' web article.

An honesty note that shapes the whole design. xgboost, lightgbm and catboost
are not installed in this environment, so this script does NOT produce a
library shoot-out, and the article does not print one. Inventing timings for
software you did not run is exactly the kind of number that quietly poisons an
explainer. What is measured here is the real and teachable contrast that
scikit-learn alone can provide: exact split search against histogram binning,
which is precisely the algorithmic idea LightGBM introduced and XGBoost later
adopted as its default.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gradient-boosting" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

# ------------------------------------------------------------------ diamonds
csv = ROOT / "datasets" / "diamonds.csv"
if not csv.exists():
    import urllib.request

    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
    urllib.request.urlretrieve(url, csv)

df = pd.read_csv(csv)
# the notebook flags x/y/z zeros as measurement errors and then never acts on
# them; a stone cannot have a zero dimension, so they go
before = len(df)
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].reset_index(drop=True)
dropped = before - len(df)

CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = list("JIHGFED")
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df["cut_o"] = df["cut"].map({v: i for i, v in enumerate(CUT)})
df["color_o"] = df["color"].map({v: i for i, v in enumerate(COLOR)})
df["clarity_o"] = df["clarity"].map({v: i for i, v in enumerate(CLARITY)})

FEATURES = ["carat", "cut_o", "color_o", "clarity_o", "depth", "table", "x", "y", "z"]
LABELS = ["carat", "cut", "colour", "clarity", "depth %", "table %", "x mm", "y mm", "z mm"]
X = df[FEATURES].to_numpy(float)
y = df["price"].to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def bench(name, make, repeats=3):
    times, scores = [], []
    for s in range(repeats):
        m = make(s)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        times.append(time.perf_counter() - t0)
        scores.append(rmse(yte, m.predict(Xte)))
    return {
        "name": name,
        "rmse": rnd(float(np.mean(scores)), 1),
        "rmse_sd": rnd(float(np.std(scores)), 1),
        "seconds": rnd(float(np.mean(times)), 2),
        "seconds_sd": rnd(float(np.std(times)), 2),
    }


print("running the benchmark that can honestly be run here...")
results = [
    bench("ridge (a straight line)", lambda s: Ridge(alpha=1.0)),
    bench("one deep tree", lambda s: DecisionTreeRegressor(max_depth=12, random_state=s)),
    bench("random forest, 100 trees", lambda s: RandomForestRegressor(n_estimators=100, random_state=s, n_jobs=-1)),
    bench("gradient boosting, exact splits", lambda s: GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=s)),
    bench("gradient boosting, histogram", lambda s: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=s)),
]
for r in results:
    print(f"  {r['name']:34s} rmse {r['rmse']:8.1f}  fit {r['seconds']:6.2f}s")

speedup = results[3]["seconds"] / results[4]["seconds"]
print(f"\n  histogram binning is {speedup:.1f}x faster here, at rmse "
      f"{results[4]['rmse']} against {results[3]['rmse']}")

# ---------------------------------------------- learning curve / early stopping
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=7)

# The learning rate here is deliberately aggressive. At the gentle default of
# 0.1 this model simply never overfits within any tree budget worth plotting,
# which would make an "early stopping" chart that demonstrates nothing. Turning
# the rate up is what surfaces the U-shape, and that is itself the lesson: a
# small learning rate is the cheapest overfitting insurance boosting offers.
CURVE_LR = 0.6
CURVE_DEPTH = 6
gb = GradientBoostingRegressor(
    n_estimators=400, learning_rate=CURVE_LR, max_depth=CURVE_DEPTH, random_state=0
)
gb.fit(Xa, ya)
curve = {"train": [], "valid": [], "lr": CURVE_LR, "depth": CURVE_DEPTH}
for tr_pred, va_pred in zip(gb.staged_predict(Xa), gb.staged_predict(Xv)):
    curve["train"].append(rnd(rmse(ya, tr_pred), 1))
    curve["valid"].append(rnd(rmse(yv, va_pred), 1))
best_iter = int(np.argmin(curve["valid"]) + 1)
print(f"\n  early stopping (lr={CURVE_LR}, depth={CURVE_DEPTH}): validation bottoms at tree "
      f"{best_iter} of {len(curve['valid'])}, rmse {min(curve['valid'])}; "
      f"by the last tree it has drifted to {curve['valid'][-1]} "
      f"while training rmse fell to {curve['train'][-1]}")

# ------------------------------------------- learning rate against tree count
lr_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
n_grid = [10, 25, 50, 100, 200, 400]
lr_surface = []
for lr in lr_grid:
    m = GradientBoostingRegressor(n_estimators=max(n_grid), learning_rate=lr, max_depth=3, random_state=0)
    m.fit(Xa, ya)
    preds = list(m.staged_predict(Xv))
    lr_surface.append({"lr": lr, "values": [rnd(rmse(yv, preds[n - 1]), 1) for n in n_grid]})
print("\n  learning rate against number of trees (validation rmse):")
print("      trees:  " + "".join(f"{n:>8d}" for n in n_grid))
for row in lr_surface:
    print(f"    lr {row['lr']:<5}" + "".join(f"{v:>8.0f}" for v in row["values"]))

# ------------------------------------------------------- feature importance
gb_imp = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, ytr)
imp = [{"feature": l, "gain": rnd(float(v), 5)} for l, v in zip(LABELS, gb_imp.feature_importances_)]
imp.sort(key=lambda d: -d["gain"])
print("\n  feature importance by gain (settles the notebook's own contradiction):")
for d in imp[:5]:
    print(f"    {d['feature']:10s} {d['gain']:.4f}")
corr_carat = {l: rnd(float(np.corrcoef(df[f], df["carat"])[0, 1]), 3) for f, l in zip(FEATURES, LABELS)}

# a small, honest sample of the data for the article to draw
samp = df.sample(1400, random_state=3)
out = {
    "n_rows": int(len(df)),
    "dropped_zero_dims": int(dropped),
    "features": LABELS,
    "sample": [{"carat": rnd(a, 2), "price": int(b)} for a, b in zip(samp["carat"], samp["price"])],
    "benchmark": results,
    "speedup": rnd(speedup, 2),
    "curve": curve,
    "best_iter": best_iter,
    "lr_grid": lr_grid,
    "n_grid": n_grid,
    "lr_surface": lr_surface,
    "importance": imp,
    "corr_with_carat": corr_carat,
}
(OUT / "diamonds.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\ndiamonds.json {(OUT / 'diamonds.json').stat().st_size} bytes"
      f"  ({len(df)} stones, {dropped} dropped for having a zero dimension)")
print("correlation of each feature with carat:", corr_carat)
