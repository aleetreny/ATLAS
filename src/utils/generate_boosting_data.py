"""Data for the ATLAS 'Gradient Boosting' web article.

Every number the article prints is produced here, on one machine, with the
versions recorded in the output so the reader knows what was actually run.
Timings are wall-clock means over several repeats with the standard deviation
kept, because a single-run benchmark is a rumour.
"""
import json
import platform
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

import catboost as cb
import lightgbm as lgb
import sklearn
import xgboost as xgb

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


# Every boosted row below is pinned to the SAME capacity: 300 trees, depth 3,
# learning rate 0.1. Comparing libraries at their own defaults measures whose
# defaults are bolder, not whose algorithm is better, and sklearn's histogram
# implementation in particular defaults to 31 leaves rather than depth 3, which
# alone would hand it a win it did not earn.
print("running the benchmark, all boosted rows pinned to 300 trees / depth 3 / lr 0.1 ...")
N_TREES = 300
results = [
    bench("ridge", lambda s: Ridge(alpha=1.0)),
    bench("one deep tree", lambda s: DecisionTreeRegressor(max_depth=12, random_state=s)),
    bench("random forest", lambda s: RandomForestRegressor(n_estimators=100, random_state=s, n_jobs=-1)),
    bench("sklearn, exact splits", lambda s: GradientBoostingRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=3, random_state=s)),
    bench("sklearn, histogram", lambda s: HistGradientBoostingRegressor(max_iter=N_TREES, learning_rate=0.1, max_depth=3, max_leaf_nodes=None, random_state=s)),
    bench("xgboost", lambda s: xgb.XGBRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=3, random_state=s, verbosity=0)),
    bench("lightgbm", lambda s: lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=3, num_leaves=8, random_state=s, verbose=-1)),
    bench("catboost", lambda s: cb.CatBoostRegressor(iterations=N_TREES, learning_rate=0.1, depth=3, random_seed=s, verbose=0, allow_writing_files=False)),
]
for r in results:
    print(f"  {r['name']:24s} rmse {r['rmse']:8.1f} +-{r['rmse_sd']:5.1f}   fit {r['seconds']:6.2f}s +-{r['seconds_sd']:.2f}")

boosted = results[3:]
spread_rmse = max(r["rmse"] for r in boosted) - min(r["rmse"] for r in boosted)
slowest = max(boosted, key=lambda r: r["seconds"])
fastest = min(boosted, key=lambda r: r["seconds"])
speedup = slowest["seconds"] / fastest["seconds"]
print(f"\n  at matched capacity the five boosted rows span only {spread_rmse:.1f} dollars of rmse,")
print(f"  while {fastest['name']} fits {speedup:.0f}x faster than {slowest['name']}")

# And separately: what does letting a modern library use its own capacity buy?
default_lgb = bench("lightgbm, own defaults", lambda s: lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, random_state=s, verbose=-1))
print(f"  letting lightgbm use its default 31 leaves instead: rmse {default_lgb['rmse']:.1f} in {default_lgb['seconds']:.2f}s")
results.append(default_lgb)

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

# ------------------------------------------- native categoricals vs encoding
# The signature feature of CatBoost and LightGBM is handling raw categorical
# columns directly instead of forcing them through an encoder. The companion
# notebook ordinal-encodes everything and therefore never exercises it. Here
# all three treatments are measured on the same split.
CAT = ["cut", "color", "clarity"]
NUM = ["carat", "depth", "table", "x", "y", "z"]
raw = df[NUM + CAT].copy()
for c in CAT:
    raw[c] = raw[c].astype("category")
Rtr, Rte, rytr, ryte = train_test_split(raw, y, test_size=0.2, random_state=42)

one_hot = pd.get_dummies(df[NUM + CAT], columns=CAT)
Otr, Ote, oytr, oyte = train_test_split(one_hot, y, test_size=0.2, random_state=42)

cat_results = []


def cat_bench(name, fit_predict, repeats=3):
    times, scores = [], []
    for s in range(repeats):
        t0 = time.perf_counter()
        pred = fit_predict(s)
        times.append(time.perf_counter() - t0)
        scores.append(rmse(ryte, pred))
    cat_results.append({
        "name": name,
        "rmse": rnd(float(np.mean(scores)), 1),
        "rmse_sd": rnd(float(np.std(scores)), 1),
        "seconds": rnd(float(np.mean(times)), 2),
        "n_features": None,
    })
    return cat_results[-1]


def _ordinal(s):
    m = lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=3, random_state=s, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _native(s):
    m = lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=3, random_state=s, verbose=-1)
    m.fit(Rtr, rytr, categorical_feature=CAT)
    return m.predict(Rte)


def _onehot(s):
    m = lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=3, random_state=s, verbose=-1)
    m.fit(Otr, oytr)
    return m.predict(Ote)


def _cb_native(s):
    m = cb.CatBoostRegressor(iterations=N_TREES, learning_rate=0.1, depth=3, random_seed=s,
                             verbose=0, allow_writing_files=False, cat_features=CAT)
    m.fit(df.loc[Rtr.index, NUM + CAT], rytr)
    return m.predict(df.loc[Rte.index, NUM + CAT])


cat_bench("lightgbm, ordinal codes", _ordinal)
cat_bench("lightgbm, one-hot", _onehot)
cat_bench("lightgbm, native categorical", _native)
cat_bench("catboost, native categorical", _cb_native)
cat_results[0]["n_features"] = int(Xtr.shape[1])
cat_results[1]["n_features"] = int(Otr.shape[1])
cat_results[2]["n_features"] = int(Rtr.shape[1])
cat_results[3]["n_features"] = int(Rtr.shape[1])
print("\n  how you feed categories in (same library, same settings):")
for r in cat_results:
    print(f"    {r['name']:32s} {r['n_features']:3d} features   rmse {r['rmse']:7.1f}   {r['seconds']:5.2f}s")

# a small, honest sample of the data for the article to draw
samp = df.sample(1400, random_state=3)
out = {
    "n_rows": int(len(df)),
    "dropped_zero_dims": int(dropped),
    "features": LABELS,
    "sample": [{"carat": rnd(a, 2), "price": int(b)} for a, b in zip(samp["carat"], samp["price"])],
    "benchmark": results,
    "categoricals": cat_results,
    "speedup": rnd(speedup, 2),
    "versions": {
        "xgboost": xgb.__version__,
        "lightgbm": lgb.__version__,
        "catboost": cb.__version__,
        "scikit-learn": sklearn.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
    },
    "n_trees": N_TREES,
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
