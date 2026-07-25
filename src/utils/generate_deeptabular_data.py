"""Data for the ATLAS 'Deep Learning for Tables' web article.

This is the topic where it is easiest to write something flattering and false,
so the design is built around three experiments that can actually settle the
question rather than around a tour of architectures.

  1. A benchmark on real diamonds: MLP, TabNet, NAM, EBM and boosting, repeated
     with different seeds so the differences can be read against their noise.
  2. The rotation test from Grinsztajn et al. (2022). Rotating the feature space
     is information-preserving for any model that does not care which axis is
     which. Trees care enormously. This is the sharpest single explanation of
     why they win here, and it takes ten lines to run.
  3. The uninformative-feature test: bury the signal among junk columns and
     watch which family degrades.

Plus the artefact that makes NAMs worth the trouble: the learned shape
functions, exported so the article can draw what the network believes about
each feature on its own.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from interpret.glassbox import ExplainableBoostingRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deep-learning-tables" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))

# --------------------------------------------------------------- the diamonds
df = pd.read_csv(ROOT / "datasets" / "diamonds.csv")
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].reset_index(drop=True)
CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = list("JIHGFED")
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df["cut_o"] = df["cut"].map({v: i for i, v in enumerate(CUT)})
df["color_o"] = df["color"].map({v: i for i, v in enumerate(COLOR)})
df["clarity_o"] = df["clarity"].map({v: i for i, v in enumerate(CLARITY)})

FEATURES = ["carat", "cut_o", "color_o", "clarity_o", "depth", "table", "x", "y", "z"]
LABELS = ["carat", "cut", "colour", "clarity", "depth %", "table %", "x mm", "y mm", "z mm"]
X = df[FEATURES].to_numpy(np.float32)
y = df["price"].to_numpy(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

sx = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sx.transform(Xtr).astype(np.float32), sx.transform(Xte).astype(np.float32)
ymu, ysd = ytr.mean(), ytr.std()
ytr_s = ((ytr - ymu) / ysd).astype(np.float32)


# ---------------------------------------------------------------- a real NAM
class ShapeNet(nn.Module):
    """One small network per feature, seeing only that feature. The sum of their
    outputs is the prediction, so each can be plotted on its own."""

    def __init__(self, n_features, hidden=48):
        super().__init__()
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(n_features)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def shapes(self, x):
        return torch.cat([net(x[:, i: i + 1]) for i, net in enumerate(self.nets)], dim=1)

    def forward(self, x):
        return self.shapes(x).sum(1, keepdim=True) + self.bias


def train_nam(Xa, ya, seed, epochs=40, batch=1024):
    torch.manual_seed(seed)
    model = ShapeNet(Xa.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    Xt, yt = torch.tensor(Xa), torch.tensor(ya).unsqueeze(1)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i: i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def nam_predict(model, Xa):
    with torch.no_grad():
        return (model(torch.tensor(Xa)).squeeze(1).numpy() * ysd + ymu)


# ------------------------------------------------------------- the benchmark
def bench(name, fn, repeats=3):
    ts, ss = [], []
    for s in range(repeats):
        t0 = time.perf_counter()
        pred = fn(s)
        ts.append(time.perf_counter() - t0)
        ss.append(rmse(yte, pred))
    r = {"name": name, "rmse": rnd(np.mean(ss), 1), "rmse_sd": rnd(np.std(ss), 1),
         "seconds": rnd(np.mean(ts), 1)}
    print(f"  {name:28s} rmse {r['rmse']:8.1f} +-{r['rmse_sd']:5.1f}   {r['seconds']:6.1f}s")
    return r


def run_ridge(s):
    return Ridge(alpha=1.0).fit(Xtr_s, ytr).predict(Xte_s)


def run_mlp(s):
    m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=120, random_state=s,
                     early_stopping=True, n_iter_no_change=8)
    m.fit(Xtr_s, ytr_s)
    return m.predict(Xte_s) * ysd + ymu


def run_nam(s):
    return nam_predict(train_nam(Xtr_s, ytr_s, s), Xte_s)


def run_tabnet(s):
    m = TabNetRegressor(seed=s, verbose=0, n_d=8, n_a=8, n_steps=3)
    m.fit(Xtr_s, ytr_s.reshape(-1, 1), max_epochs=40, patience=10, batch_size=4096, drop_last=False)
    return m.predict(Xte_s).ravel() * ysd + ymu


def run_ebm(s):
    return ExplainableBoostingRegressor(random_state=s, interactions=0).fit(Xtr, ytr).predict(Xte)


def run_ebm_pairs(s):
    """EBM as it is actually used: additive shape functions PLUS a budget of
    pairwise interaction terms, which is the principled answer to what pure
    additivity gives up."""
    return ExplainableBoostingRegressor(random_state=s).fit(Xtr, ytr).predict(Xte)


def run_gbm(s):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=s).fit(Xtr, ytr).predict(Xte)


print("benchmark on 53,920 diamonds, three seeds each ...")
results = [
    bench("ridge", run_ridge),
    bench("mlp (256-128-64)", run_mlp),
    bench("neural additive model", run_nam),
    bench("tabnet", run_tabnet, repeats=2),
    bench("explainable boosting, additive", run_ebm, repeats=2),
    bench("explainable boosting, +pairs", run_ebm_pairs, repeats=2),
    bench("gradient boosting", run_gbm),
]

# ------------------------------------------------ the rotation test (Grinsztajn)
# Rotating the features destroys nothing an axis-agnostic model relies on. A
# tree can only cut along axes, so it loses the very structure it exploits.
print("\nrotation test: the same data seen through a random orthogonal basis")
rot_rows = []
rng_r = np.random.RandomState(0)
Q, _ = np.linalg.qr(rng_r.standard_normal((X.shape[1], X.shape[1])))
Xtr_r = (Xtr_s @ Q).astype(np.float32)
Xte_r = (Xte_s @ Q).astype(np.float32)
for name, plain, rotated in [
    ("gradient boosting",
     lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr_s, ytr).predict(Xte_s),
     lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr_r, ytr).predict(Xte_r)),
    ("mlp",
     lambda: MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=120, random_state=0, early_stopping=True).fit(Xtr_s, ytr_s).predict(Xte_s) * ysd + ymu,
     lambda: MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=120, random_state=0, early_stopping=True).fit(Xtr_r, ytr_s).predict(Xte_r) * ysd + ymu),
    ("ridge",
     lambda: Ridge(alpha=1.0).fit(Xtr_s, ytr).predict(Xte_s),
     lambda: Ridge(alpha=1.0).fit(Xtr_r, ytr).predict(Xte_r)),
]:
    a, b = rmse(yte, plain()), rmse(yte, rotated())
    rot_rows.append({"name": name, "plain": rnd(a, 1), "rotated": rnd(b, 1), "penalty": rnd((b / a - 1) * 100, 1)})
    print(f"  {name:20s} {a:8.1f} -> {b:8.1f}   {(b/a-1)*100:+6.1f}%")

# ---------------------------------------------- the uninformative-feature test
print("\nburying the signal: how each family copes with junk columns")
junk_rows = []
rng_j = np.random.RandomState(1)
for extra in [0, 10, 30, 60]:
    Ja = np.hstack([Xtr_s, rng_j.standard_normal((len(Xtr_s), extra))]).astype(np.float32) if extra else Xtr_s
    Je = np.hstack([Xte_s, rng_j.standard_normal((len(Xte_s), extra))]).astype(np.float32) if extra else Xte_s
    g = rmse(yte, HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Ja, ytr).predict(Je))
    m = rmse(yte, MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=120, random_state=0, early_stopping=True).fit(Ja, ytr_s).predict(Je) * ysd + ymu)
    junk_rows.append({"extra": extra, "gbm": rnd(g, 1), "mlp": rnd(m, 1)})
    print(f"  +{extra:3d} noise columns   boosting {g:8.1f}   mlp {m:8.1f}")

# ---------------------------------------------------- NAM shape functions
print("\nexporting the learned shape functions ...")
nam = train_nam(Xtr_s, ytr_s, 0)

# Each shape is centred on its mean over the TRAINING DISTRIBUTION, not over the
# plotting grid. That choice is what makes the article's waterfall exact rather
# than merely suggestive: with mu_j = E[f_j(x_j)] and base = E[F(x)], the model's
# own additivity gives F(x) = base + sum_j (f_j(x_j) - mu_j) with no error term.
with torch.no_grad():
    train_shapes = nam.shapes(torch.tensor(Xtr_s)).numpy()  # standardised units
mu = train_shapes.mean(0)
base = float(nam_predict(nam, Xtr_s).mean())

shapes = []
for j, label in enumerate(LABELS):
    lo, hi = np.percentile(Xtr[:, j], [1, 99])
    raw = np.linspace(lo, hi, 60)
    grid = np.zeros((60, X.shape[1]), dtype=np.float32)
    grid[:, j] = ((raw - sx.mean_[j]) / sx.scale_[j]).astype(np.float32)
    with torch.no_grad():
        out = nam.shapes(torch.tensor(grid))[:, j].numpy()
    out = (out - mu[j]) * ysd  # dollars away from the average diamond
    shapes.append({"feature": label, "key": FEATURES[j], "lo": rnd(lo, 3), "hi": rnd(hi, 3),
                   "x": [rnd(v, 3) for v in raw], "y": [rnd(v, 1) for v in out],
                   "range": rnd(float(out.max() - out.min()), 1)})
order = sorted(range(len(shapes)), key=lambda j: -shapes[j]["range"])
for j in order[:4]:
    print(f"  {shapes[j]['feature']:10s} spans ${shapes[j]['range']:.0f} across its range")

# Check the identity actually holds before the article leans on it.
recon = base + sum((train_shapes[:, j] - mu[j]) * ysd for j in range(X.shape[1]))
gap = float(np.abs(recon - nam_predict(nam, Xtr_s)).max())
print(f"  waterfall identity holds to ${gap:.4f}")

# A handful of real stones, so the widget breaks down actual rows rather than
# a slider fantasy: cheapest, median, dearest, plus two in between.
qs = np.percentile(yte, [2, 25, 50, 75, 98])
examples = []
for q in qs:
    i = int(np.argmin(np.abs(yte - q)))
    with torch.no_grad():
        contrib = (nam.shapes(torch.tensor(Xte_s[i: i + 1])).numpy()[0] - mu) * ysd
    examples.append({
        "price": rnd(float(yte[i]), 0),
        "pred": rnd(float(nam_predict(nam, Xte_s[i: i + 1])[0]), 0),
        "values": [rnd(float(v), 3) for v in Xte[i]],
        "contrib": [rnd(float(v), 1) for v in contrib],
        "cut": CUT[int(round(Xte[i][1]))], "colour": COLOR[int(round(Xte[i][2]))],
        "clarity": CLARITY[int(round(Xte[i][3]))],
    })
print(f"  {len(examples)} example stones, ${examples[0]['price']:.0f} to ${examples[-1]['price']:.0f}")

out = {
    "n_rows": int(len(df)),
    "n_train": int(len(Xtr)),
    "n_test": int(len(Xte)),
    "labels": LABELS,
    "benchmark": results,
    "rotation": rot_rows,
    "junk": junk_rows,
    "shapes": shapes,
    "nam_base": rnd(base, 0),
    "examples": examples,
    "price_mean": rnd(float(y.mean()), 0),
    "versions": {"torch": torch.__version__, "sklearn": __import__("sklearn").__version__},
}
(OUT / "tabular.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\ntabular.json {(OUT / 'tabular.json').stat().st_size} bytes")
