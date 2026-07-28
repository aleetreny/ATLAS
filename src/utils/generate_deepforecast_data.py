"""Everything article 4.2d cites, measured.

One model for many series instead of one model per series, which is the change
every deep forecaster of the last decade is built around. What is measured:

 1. The crossover. A global model is trained on 4, 16, 64 and 280 series and
    scored on the same held out ones, against a model fitted to each of those
    series alone. Sharing has to earn its keep and the number of series it
    takes is the answer.
 2. The two N-BEATS stacks. A generic basis (the network chooses the shape) and
    an interpretable one (polynomial trend plus Fourier season), scored for
    accuracy against each other, and then the interpretable one's own trend and
    season scored against the parts the generator wrote down before the series
    existed, which is the only place a decomposition can be scored at all.
 3. Attention is not an explanation. Three covariates, one of which really
    drives the series, one of which is a lagged copy of it and therefore adds
    nothing once the first is present, and one which is noise. The model's own
    variable selection weights are read off, and then each covariate is
    permuted and the error is measured. The two rankings do not agree, and that
    is the point.
 4. Scaling. The series have different levels, and what happens to a shared
    model when they are not normalised is measured rather than asserted.

Runs in about twenty minutes on four cores. Stages cached under
~/.atlas_vision_data/deepforecast_cache/.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "deep-forecast" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "deepforecast_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from src.utils.timeseries_data import (  # noqa: E402
    co2_monthly, digest, synthetic_corpus,
)

SEED = 44
LOOKBACK = 36
HORIZON = 12
PERIOD = 12
LENGTH = 180
N_PER_FAMILY = 40      # 7 families -> 280 series
EPOCHS = 40
HIDDEN = 128
THREADS = min(4, os.cpu_count() or 4)


def cached(name, fn):
    f = CACHE / f"{name}.json"
    if f.is_file():
        return json.loads(f.read_text())
    t0 = time.time()
    out = fn()
    f.write_text(json.dumps(out))
    print(f"  [{name}] {time.time() - t0:.1f}s")
    return out


def r(x, k=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, k) for v in np.asarray(x, dtype=float).tolist()]
    return round(float(x), k)


def torch_setup():
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    return torch


# ---------------------------------------------------------------------------
# The collection, split into series the models may learn from and series they
# may not. The held out ones include the real carbon dioxide record, which no
# model here ever trains on.
# ---------------------------------------------------------------------------
def build_corpus():
    corpus = synthetic_corpus(seed=SEED, n_per_family=N_PER_FAMILY, length=LENGTH,
                              period=PERIOD)
    rng = np.random.default_rng(SEED + 1)
    idx = rng.permutation(len(corpus))
    n_test = 56
    test = [corpus[i] for i in idx[:n_test]]
    train = [corpus[i] for i in idx[n_test:]]
    return train, test


def windows(series_list, lookback, horizon, normalise=True):
    X, Yt, S = [], [], []
    for s in series_list:
        y = np.asarray(s["y"], dtype=float)
        for i in range(lookback, len(y) - horizon + 1):
            win = y[i - lookback:i]
            tgt = y[i:i + horizon]
            if normalise:
                m = win.mean()
                sd = win.std() + 1e-6
            else:
                m, sd = 0.0, 1.0
            X.append((win - m) / sd)
            Yt.append((tgt - m) / sd)
            S.append((m, sd))
    return np.array(X, dtype=np.float32), np.array(Yt, dtype=np.float32), np.array(S)


# ---------------------------------------------------------------------------
# N-BEATS, in the two flavours the paper describes.
# ---------------------------------------------------------------------------
def make_nbeats(torch, kind, blocks=3, stacks=2, hidden=HIDDEN):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, basis):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(LOOKBACK, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.basis = basis
            if basis == "generic":
                self.theta_b = nn.Linear(hidden, LOOKBACK)
                self.theta_f = nn.Linear(hidden, HORIZON)
            elif basis == "trend":
                self.deg = 3
                self.theta_b = nn.Linear(hidden, self.deg + 1)
                self.theta_f = nn.Linear(hidden, self.deg + 1)
                tb = np.stack([(np.arange(LOOKBACK) / LOOKBACK) ** p
                               for p in range(self.deg + 1)])
                tf = np.stack([(np.arange(HORIZON) / HORIZON) ** p
                               for p in range(self.deg + 1)])
                self.register_buffer("Tb", torch.tensor(tb, dtype=torch.float32))
                self.register_buffer("Tf", torch.tensor(tf, dtype=torch.float32))
            else:                                   # seasonal
                self.harm = 4
                self.theta_b = nn.Linear(hidden, 2 * self.harm)
                self.theta_f = nn.Linear(hidden, 2 * self.harm)
                rows_b, rows_f = [], []
                for k in range(1, self.harm + 1):
                    rows_b.append(np.cos(2 * np.pi * k * np.arange(LOOKBACK) / PERIOD))
                    rows_b.append(np.sin(2 * np.pi * k * np.arange(LOOKBACK) / PERIOD))
                    rows_f.append(np.cos(2 * np.pi * k * np.arange(LOOKBACK, LOOKBACK + HORIZON)
                                         / PERIOD))
                    rows_f.append(np.sin(2 * np.pi * k * np.arange(LOOKBACK, LOOKBACK + HORIZON)
                                         / PERIOD))
                self.register_buffer("Tb", torch.tensor(np.stack(rows_b), dtype=torch.float32))
                self.register_buffer("Tf", torch.tensor(np.stack(rows_f), dtype=torch.float32))

        def forward(self, x):
            h = self.body(x)
            if self.basis == "generic":
                return self.theta_b(h), self.theta_f(h)
            return self.theta_b(h) @ self.Tb, self.theta_f(h) @ self.Tf

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "generic":
                kinds = ["generic"] * (blocks * stacks)
            else:
                kinds = ["trend"] * blocks + ["seasonal"] * blocks
            self.blocks = nn.ModuleList([Block(k) for k in kinds])
            self.kinds = kinds

        def forward(self, x, parts=False):
            residual = x
            out = torch.zeros(x.shape[0], HORIZON)
            per = {}
            for blk in self.blocks:
                b, f = blk(residual)
                residual = residual - b
                out = out + f
                if parts:
                    per.setdefault(blk.basis, torch.zeros(x.shape[0], HORIZON))
                    per[blk.basis] = per[blk.basis] + f
            return (out, per) if parts else out

    return Net()


def train_net(torch, net, X, Y, epochs=EPOCHS, batch=256, lr=1e-3, log=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.tensor(X)
    yt = torch.tensor(Y)
    n = len(xt)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = net(xt[idx])
            loss = (pred - yt[idx]).abs().mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if log is not None:
            log.append(round(tot / n, 5))
    return net


EVAL_TAIL = 60          # the last stretch of a held out series, scored by everyone


def evaluate(torch, net, series_list, normalise=True, tail=EVAL_TAIL):
    """Score the same origins for every model: the last `tail` points of each
    held out series.

    The tail matters. A model fitted to one series has to be trained on
    something, and if it is trained on windows that overlap the ones it is
    scored on then it is being asked about data it has seen, which is the
    leakage this branch spent a whole article on. So every arm here forecasts
    the same final stretch, and the local arm may only learn from what comes
    before it.
    """
    errs = []
    with torch.no_grad():
        for s in series_list:
            y = np.asarray(s["y"], dtype=float)
            scale = float(np.mean(np.abs(y[PERIOD:] - y[:-PERIOD])))
            for o in range(len(y) - tail, len(y) - HORIZON + 1, 6):
                win = y[o - LOOKBACK:o]
                m = win.mean() if normalise else 0.0
                sd = (win.std() + 1e-6) if normalise else 1.0
                x = torch.tensor(((win - m) / sd)[None, :].astype(np.float32))
                p = net(x).numpy()[0] * sd + m
                errs.append(np.abs(p - y[o:o + HORIZON]) / scale)
    e = np.array(errs)
    return float(e.mean()), e.mean(axis=0)


def seasonal_naive(series_list, tail=EVAL_TAIL):
    errs = []
    for s in series_list:
        y = np.asarray(s["y"], dtype=float)
        scale = float(np.mean(np.abs(y[PERIOD:] - y[:-PERIOD])))
        for o in range(len(y) - tail, len(y) - HORIZON + 1, 6):
            f = [y[o - PERIOD + (i % PERIOD)] for i in range(HORIZON)]
            errs.append(np.abs(np.array(f) - y[o:o + HORIZON]) / scale)
    e = np.array(errs)
    return float(e.mean()), e.mean(axis=0)


def local_ets(series_list, tail=EVAL_TAIL):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    errs = []
    for s in series_list:
        y = np.asarray(s["y"], dtype=float)
        scale = float(np.mean(np.abs(y[PERIOD:] - y[:-PERIOD])))
        for o in range(len(y) - tail, len(y) - HORIZON + 1, 6):
            try:
                m = ExponentialSmoothing(y[:o], trend="add", damped_trend=True,
                                         seasonal="add", seasonal_periods=PERIOD,
                                         initialization_method="estimated").fit()
                f = np.asarray(m.forecast(HORIZON))
            except Exception:
                f = np.array([y[o - PERIOD + (i % PERIOD)] for i in range(HORIZON)])
            errs.append(np.abs(f - y[o:o + HORIZON]) / scale)
    e = np.array(errs)
    return float(e.mean()), e.mean(axis=0)


# ---------------------------------------------------------------------------
# Stage 1: how many series it takes before sharing wins
# ---------------------------------------------------------------------------
def stage_scaling():
    torch = torch_setup()
    train, test = build_corpus()
    rng = np.random.default_rng(SEED + 2)
    sizes = [4, 16, 64, len(train)]
    rows = []
    for k in sizes:
        pick = [train[i] for i in rng.choice(len(train), size=k, replace=False)] \
            if k < len(train) else train
        X, Y, _ = windows(pick, LOOKBACK, HORIZON)
        net = make_nbeats(torch, "generic")
        t0 = time.time()
        train_net(torch, net, X, Y)
        mase, per_h = evaluate(torch, net, test)
        rows.append({"series": k, "windows": int(len(X)), "mase": r(mase, 4),
                     "per_h": r(per_h.tolist(), 4), "seconds": round(time.time() - t0, 1)})
        print(f"    global on {k} series: {mase:.4f}")

    # One model per test series, trained only on that series and only on the
    # part of it that comes before the stretch everyone is scored on. Without
    # that second restriction the local arm is scored on windows it trained on
    # and comes out at 0.39, which is not a forecast, it is a memory.
    local_errs = []
    for s in test[:20]:
        cut = len(s["y"]) - EVAL_TAIL
        head = {"y": np.asarray(s["y"], dtype=float)[:cut]}
        X, Y, _ = windows([head], LOOKBACK, HORIZON)
        net = make_nbeats(torch, "generic")
        train_net(torch, net, X, Y, epochs=EPOCHS)
        m, _ = evaluate(torch, net, [s])
        local_errs.append(m)
    local = float(np.mean(local_errs))

    sn, sn_h = seasonal_naive(test)
    ets, ets_h = local_ets(test)
    return {"rows": rows, "local_nbeats": r(local, 4), "local_n": 20,
            "seasonal_naive": r(sn, 4), "seasonal_naive_h": r(sn_h.tolist(), 4),
            "ets": r(ets, 4), "ets_h": r(ets_h.tolist(), 4),
            "n_train": len(train), "n_test": len(test), "epochs": EPOCHS,
            "threads": THREADS}


# ---------------------------------------------------------------------------
# Stage 2: the two bases, and scoring the interpretable one against the truth
# ---------------------------------------------------------------------------
def stage_bases():
    torch = torch_setup()
    train, test = build_corpus()
    X, Y, _ = windows(train, LOOKBACK, HORIZON)
    out = {}
    nets = {}
    # Three seeds each, because the difference between the two bases turned out
    # to be smaller than a thousandth and a table with one training run per row
    # cannot tell that from a tie.
    for kind in ["generic", "interpretable"]:
        runs = []
        log = []
        for sd in range(3):
            torch.manual_seed(SEED + sd)
            net = make_nbeats(torch, kind)
            lg = []
            train_net(torch, net, X, Y, log=lg)
            mase, per_h = evaluate(torch, net, test)
            runs.append((mase, per_h))
            if sd == 0:
                nets[kind] = net
                log = lg
            print(f"    {kind} seed {sd}: {mase:.4f}")
        vals = np.array([m for m, _ in runs])
        out[kind] = {"mase": r(float(vals.mean()), 4),
                     "per_h": r(runs[0][1].tolist(), 4),
                     "seeds": r(vals.tolist(), 4),
                     "spread": r(float(vals.max() - vals.min()), 4),
                     "loss": r(log, 5),
                     "params": int(sum(p.numel() for p in nets[kind].parameters()))}

    # the parts, against the parts that were written down
    net = nets["interpretable"]
    rows = {}
    examples = []
    with torch.no_grad():
        for s in test:
            y = np.asarray(s["y"], dtype=float)
            tr = np.asarray(s["trend"], dtype=float)
            se = np.asarray(s["season"], dtype=float)
            sd = float(np.std(y))
            for o in range(LOOKBACK, len(y) - HORIZON + 1, 12):
                win = y[o - LOOKBACK:o]
                m = win.mean()
                sdw = win.std() + 1e-6
                x = torch.tensor(((win - m) / sdw)[None, :].astype(np.float32))
                _, parts = net(x, parts=True)
                trend_hat = parts["trend"].numpy()[0] * sdw + m
                season_hat = parts["seasonal"].numpy()[0] * sdw
                tr_true = tr[o:o + HORIZON]
                se_true = se[o:o + HORIZON]
                # the split between level and trend is not identified, so both
                # sides are centred before they are compared
                e_tr = float(np.sqrt(np.mean(((trend_hat - trend_hat.mean())
                                              - (tr_true - tr_true.mean())) ** 2))) / sd
                e_se = float(np.sqrt(np.mean(((season_hat - season_hat.mean())
                                              - (se_true - se_true.mean())) ** 2))) / sd
                rows.setdefault(s["family"], []).append((e_tr, e_se))
                if len(examples) < 3 and o == LOOKBACK + 12 and s["family"] in (
                        "trend_season", "ar1", "level_shift"):
                    examples.append({
                        "family": s["family"],
                        "context": r(win, 4),
                        "truth": r(y[o:o + HORIZON], 4),
                        "forecast": r((net(x).numpy()[0] * sdw + m), 4),
                        "trend_hat": r(trend_hat, 4), "season_hat": r(season_hat, 4),
                        "trend_true": r(tr_true, 4), "season_true": r(se_true, 4),
                    })
    table = []
    for fam, vals in rows.items():
        a = np.array(vals)
        table.append({"family": fam, "n": len(vals),
                      "trend_err": r(float(a[:, 0].mean()), 4),
                      "season_err": r(float(a[:, 1].mean()), 4)})
    table.sort(key=lambda x: x["trend_err"])
    out["decomposition"] = table
    out["examples"] = examples
    return out


# ---------------------------------------------------------------------------
# Stage 3: attention, and what it is worth
# ---------------------------------------------------------------------------
def covariate_corpus(n_series=240, length=LENGTH, seed=SEED + 5):
    """Series with a covariate that really drives them, one that only echoes
    it, and one that is noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        t = np.arange(length, dtype=float)
        level = 40 + rng.uniform(-8, 8)
        amp = rng.uniform(2, 6)
        phase = rng.uniform(0, 2 * np.pi)
        season = amp * np.sin(2 * np.pi * t / PERIOD + phase)
        driver = (rng.random(length) < 0.12).astype(float)     # a promotion
        effect = rng.uniform(4, 9)
        copy = driver.copy()                                   # the same column twice
        echo = np.concatenate([[0.0], driver[:-1]])            # yesterday's promotion
        noise_cov = rng.normal(0, 1, length)
        y = level + season + effect * driver + rng.normal(0, 0.8, length)
        out.append({"y": y, "covs": np.stack([driver, copy, echo, noise_cov]),
                    "effect": effect, "family": "promotion"})
    return out


def make_attention_net(torch, n_cov=4, hidden=64):
    nn = torch.nn

    class Net(nn.Module):
        """A small model in the shape of a temporal fusion transformer: each
        input gets its own encoder, a softmax over inputs decides how much of
        each to use, and attention over the lookback produces the forecast."""

        def __init__(self):
            super().__init__()
            self.hist = nn.Linear(LOOKBACK, hidden)
            self.cov = nn.ModuleList([nn.Linear(LOOKBACK + HORIZON, hidden)
                                      for _ in range(n_cov)])
            self.select = nn.Linear(hidden * (n_cov + 1), n_cov + 1)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, HORIZON))

        def forward(self, x, covs, weights=False):
            parts = [torch.relu(self.hist(x))]
            for i, layer in enumerate(self.cov):
                parts.append(torch.relu(layer(covs[:, i, :])))
            stacked = torch.stack(parts, dim=1)
            w = torch.softmax(self.select(torch.cat(parts, dim=1)), dim=1)
            fused = (stacked * w[:, :, None]).sum(dim=1)
            out = self.head(fused)
            return (out, w) if weights else out

    return Net()


def stage_attention():
    torch = torch_setup()
    corpus = covariate_corpus()
    train, test = corpus[:200], corpus[200:]

    def make_windows(items):
        X, C, Y = [], [], []
        for s in items:
            y = np.asarray(s["y"], dtype=float)
            covs = np.asarray(s["covs"], dtype=float)
            for i in range(LOOKBACK, len(y) - HORIZON + 1):
                win = y[i - LOOKBACK:i]
                m = win.mean()
                sd = win.std() + 1e-6
                X.append((win - m) / sd)
                C.append(covs[:, i - LOOKBACK:i + HORIZON])
                Y.append((y[i:i + HORIZON] - m) / sd)
        return (np.array(X, dtype=np.float32), np.array(C, dtype=np.float32),
                np.array(Y, dtype=np.float32))

    Xtr, Ctr, Ytr = make_windows(train)
    Xte, Cte, Yte = make_windows(test)
    net = make_attention_net(torch)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, ct, yt = torch.tensor(Xtr), torch.tensor(Ctr), torch.tensor(Ytr)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(30):
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = (net(xt[idx], ct[idx]) - yt[idx]).abs().mean()
            loss.backward()
            opt.step()

    xe, ce, ye = torch.tensor(Xte), torch.tensor(Cte), torch.tensor(Yte)
    with torch.no_grad():
        base_pred, w = net(xe, ce, weights=True)
        base = float((base_pred - ye).abs().mean())
        weights = w.mean(dim=0).numpy()
        names = ["the history", "the promotion", "the same column again",
                 "yesterday's promotion", "noise"]
        rows = []
        rng = np.random.default_rng(SEED + 9)
        for j in range(4):
            ce2 = ce.clone()
            perm = torch.tensor(rng.permutation(len(ce2)))
            ce2[:, j, :] = ce[perm, j, :]
            err = float((net(xe, ce2) - ye).abs().mean())
            rows.append({"name": names[j + 1], "weight": r(float(weights[j + 1]), 4),
                         "loss_when_permuted": r(err, 5),
                         "cost": r(err - base, 5),
                         "ratio": r(err / base, 4)})
        # and the pair together, which is the only way to see what the two
        # copies are worth: permuting either one alone leaves the other one
        # holding the same information
        ce2 = ce.clone()
        perm = torch.tensor(rng.permutation(len(ce2)))
        ce2[:, 0, :] = ce[perm, 0, :]
        ce2[:, 1, :] = ce[perm, 1, :]
        both = float((net(xe, ce2) - ye).abs().mean())
        rows_hist = {"name": names[0], "weight": r(float(weights[0]), 4)}
    return {"base_loss": r(base, 5), "rows": rows, "history": rows_hist,
            "both_copies": {"loss_when_permuted": r(both, 5), "cost": r(both - base, 5),
                            "ratio": r(both / base, 4)},
            "n_train": len(train), "n_test": len(test),
            "n_windows": int(len(Xtr)), "epochs": 30,
            "effect": r(float(np.mean([s["effect"] for s in corpus])), 3)}


# ---------------------------------------------------------------------------
# Stage 4: what normalising is worth
# ---------------------------------------------------------------------------
def stage_scaling_ablation():
    torch = torch_setup()
    train, test = build_corpus()
    out = {}
    fams = sorted({s["family"] for s in test})
    per_family = {}
    for norm in [True, False]:
        key = "normalised" if norm else "raw"
        X, Y, _ = windows(train, LOOKBACK, HORIZON, normalise=norm)
        runs = []
        by_fam = {f: [] for f in fams}
        for sd in range(3):
            torch.manual_seed(SEED + sd)
            net = make_nbeats(torch, "generic")
            train_net(torch, net, X, Y)
            mase, _ = evaluate(torch, net, test, normalise=norm)
            runs.append(mase)
            for f in fams:
                m, _ = evaluate(torch, net, [s for s in test if s["family"] == f],
                                normalise=norm)
                by_fam[f].append(m)
            print(f"    normalise={norm} seed {sd}: {mase:.4f}")
        v = np.array(runs)
        out[key] = r(float(v.mean()), 4)
        out[f"{key}_seeds"] = r(v.tolist(), 4)
        out[f"{key}_spread"] = r(float(v.max() - v.min()), 4)
        per_family[key] = {f: r(float(np.mean(by_fam[f])), 4) for f in fams}
    out["per_family"] = per_family
    out["families"] = fams
    levels = [float(np.mean(s["y"])) for s in train]
    out["level_min"] = r(min(levels), 2)
    out["level_max"] = r(max(levels), 2)
    return out


# ---------------------------------------------------------------------------
# Stage 5: the real series, which nothing here has ever been trained on
# ---------------------------------------------------------------------------
def stage_real():
    torch = torch_setup()
    train, _ = build_corpus()
    X, Y, _ = windows(train, LOOKBACK, HORIZON)
    net = make_nbeats(torch, "generic")
    train_net(torch, net, X, Y)
    y_raw, labels, meta = co2_monthly()
    y = np.round(y_raw, 4)
    scale = float(np.mean(np.abs(y[PERIOD:] - y[:-PERIOD])))
    errs = []
    with torch.no_grad():
        for o in range(len(y) - 120, len(y) - HORIZON + 1):
            win = y[o - LOOKBACK:o]
            m = win.mean()
            sd = win.std() + 1e-6
            x = torch.tensor(((win - m) / sd)[None, :].astype(np.float32))
            p = net(x).numpy()[0] * sd + m
            errs.append(np.abs(p - y[o:o + HORIZON]) / scale)
    e = np.array(errs)
    o = len(y) - 60
    win = y[o - LOOKBACK:o]
    m = win.mean()
    sd = win.std() + 1e-6
    with torch.no_grad():
        x = torch.tensor(((win - m) / sd)[None, :].astype(np.float32))
        demo = net(x).numpy()[0] * sd + m
    return {"mase": r(float(e.mean()), 4), "per_h": r(e.mean(axis=0).tolist(), 4),
            "n_origins": int(len(errs)), "digest": digest(y),
            "demo": {"origin": int(o), "label": labels[o],
                     "context": r(win, 4), "forecast": r(demo, 4),
                     "truth": r(y[o:o + HORIZON], 4)}}


def main():
    import torch

    # the previous articles' numbers on the same real series, so the last row
    # of this page's table can be read against them instead of beside them
    arima = ROOT / "arima" / "data" / "arima.json"
    board = None
    if arima.is_file():
        prev = json.loads(arima.read_text())
        board = {"rows": [{"model": x["model"], "mase_mean": x["mase_mean"]}
                          for x in prev["rolling"]["rows"]],
                 "n_origins": prev["rolling"]["n_origins"],
                 "co2_digest": prev["meta"]["co2_digest"]}

    print("scaling with the number of series")
    scaling = cached("scaling", stage_scaling)
    print("the two bases")
    bases = cached("bases", stage_bases)
    print("attention")
    att = cached("attention", stage_attention)
    print("normalisation")
    norm = cached("normalisation", stage_scaling_ablation)
    print("the real series")
    real = cached("real", stage_real)

    train, test = build_corpus()
    data = {
        "meta": {
            "seed": SEED, "lookback": LOOKBACK, "horizon": HORIZON, "period": PERIOD,
            "length": LENGTH, "n_train": len(train), "n_test": len(test),
            "families": sorted({s["family"] for s in train}),
            "epochs": EPOCHS, "hidden": HIDDEN, "threads": THREADS,
            "torch": torch.__version__, "numpy": np.__version__,
            "corpus_digest": digest(np.concatenate([s["y"] for s in train[:20]])),
        },
        "scaling": scaling, "bases": bases, "attention": att,
        "normalisation": norm, "real": real, "board": board,
        "examples": [{"family": s["family"], "y": r(s["y"], 4)} for s in test[:6]],
    }
    (OUT / "deepforecast.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT / 'deepforecast.json'} "
          f"({(OUT / 'deepforecast.json').stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
