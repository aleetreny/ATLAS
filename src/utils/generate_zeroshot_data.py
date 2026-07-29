"""Everything article 4.2e cites, measured.

A forecaster that is never fitted to the series it forecasts. The recipe is the
one Chronos describes: scale each window, quantise it into a fixed alphabet,
and train a small autoregressive model over that alphabet on a large collection
of series. Then hand it a series it has never seen and ask for the next twelve
values, without fitting anything.

The corpus here is entirely written down by `timeseries_data.synthetic_corpus`,
which is what makes the central claim checkable rather than hopeful: the real
series this model is scored on cannot be in its training data, because its
training data does not contain any real series at all. Every published
foundation forecaster has to argue that point; this one can prove it.

What is measured:

 1. Zero shot accuracy on the same carbon dioxide series, the same origins and
    the same units as articles 4.2a and 4.2c, so the number goes straight into
    their leaderboard.
 2. The floor the alphabet puts under it. An oracle that already knows the
    future and only has to encode it cannot do better than the quantiser, and
    that bound is measured at three alphabet sizes.
 3. What the corpus buys: the same evaluation after pretraining on a quarter,
    a half and all of it.
 4. What it cannot do: an entire family removed from the corpus, and the same
    evaluation on that family, beside a model that did see it.
 5. Intervals, which are free here and were not free anywhere else in this
    branch: a hundred sampled continuations per origin, scored for coverage the
    same way article 4.2a scored its analytic ones.

Runs in about twenty five minutes on four cores. Stages cached under
~/.atlas_vision_data/zeroshot_cache/.
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
OUT = ROOT / "zero-shot" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "zeroshot_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from src.utils.timeseries_data import (  # noqa: E402
    co2_monthly, digest, sunspots_yearly, synthetic_corpus,
)

SEED = 45
CONTEXT = 48
HORIZON = 12
PERIOD = 12
BINS = 128
CLIP = 4.0            # window standard deviations covered by the alphabet
LENGTH = 180
N_PER_FAMILY = 60     # 7 families -> 420 series
EPOCHS = 14
THREADS = min(4, os.cpu_count() or 4)
ARIMA_JSON = ROOT / "arima" / "data" / "arima.json"


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


_y, CO2LAB, CO2META = co2_monthly()
CO2 = np.round(_y, 4)
_s, SUNLAB, SUNMETA = sunspots_yearly()
SUN = np.round(_s, 4)
PREV = json.loads(ARIMA_JSON.read_text()) if ARIMA_JSON.is_file() else None
if PREV is not None:
    assert PREV["meta"]["co2_digest"] == digest(CO2), "not the same series as 4.2a"


# ---------------------------------------------------------------------------
# The alphabet.
#
# Everything the model will ever see is one of BINS symbols. A window is
# centred and scaled by its own mean and standard deviation, clipped to +-CLIP,
# and each value replaced by the index of the bin it falls in. That is the
# whole tokeniser, and it is also a hard floor on accuracy, which is measured
# rather than assumed.
# ---------------------------------------------------------------------------
def edges(bins=BINS, clip=CLIP):
    return np.linspace(-clip, clip, bins + 1)


def centres(bins=BINS, clip=CLIP):
    e = edges(bins, clip)
    return (e[:-1] + e[1:]) / 2


def encode(x, m, sd, bins=BINS, clip=CLIP):
    z = np.clip((np.asarray(x, dtype=float) - m) / sd, -clip, clip - 1e-9)
    idx = np.floor((z + clip) / (2 * clip) * bins).astype(int)
    return np.clip(idx, 0, bins - 1)


def decode(idx, m, sd, bins=BINS, clip=CLIP):
    return centres(bins, clip)[np.asarray(idx, dtype=int)] * sd + m


def make_model(torch, bins=BINS, d=96, layers=2, heads=4, seq=CONTEXT + HORIZON):
    nn = torch.nn

    class Model(nn.Module):
        """A small decoder over the alphabet: embeddings, learned positions,
        two masked self attention blocks, and a linear head back to the
        alphabet. Deliberately the smallest thing that is honestly the same
        shape as the published models."""

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(bins, d)
            self.pos = nn.Parameter(torch.zeros(seq, d))
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=0.0,
                batch_first=True, norm_first=True, activation="gelu")
            self.body = nn.TransformerEncoder(layer, num_layers=layers)
            self.head = nn.Linear(d, bins)
            self.seq = seq

        def forward(self, idx):
            n = idx.shape[1]
            h = self.emb(idx) + self.pos[:n]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            h = self.body(h, mask=mask)
            return self.head(h)

    return Model()


def token_windows(series_list, context=CONTEXT, horizon=HORIZON, stride=1):
    rows = []
    for s in series_list:
        y = np.asarray(s["y"], dtype=float)
        for i in range(context, len(y) - horizon + 1, stride):
            win = y[i - context:i]
            m = float(win.mean())
            sd = float(win.std()) + 1e-6
            rows.append(encode(y[i - context:i + horizon], m, sd))
    return np.array(rows, dtype=np.int64)


def train_model(torch, model, tokens, epochs=EPOCHS, batch=128, lr=2e-3, log=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    xt = torch.tensor(tokens)
    n = len(xt)
    g = torch.Generator().manual_seed(SEED)
    lossf = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            b = xt[idx]
            opt.zero_grad()
            logits = model(b[:, :-1])
            loss = lossf(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        if log is not None:
            log.append(round(tot / n, 5))
    return model


def sample_paths(torch, model, context_vals, n_samples=100, horizon=HORIZON,
                 temperature=1.0, seed=SEED):
    """Continue a window `n_samples` times and decode every path."""
    m = float(np.mean(context_vals))
    sd = float(np.std(context_vals)) + 1e-6
    ctx = encode(context_vals, m, sd)
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor(ctx, dtype=torch.long)[None, :].repeat(n_samples, 1)
    with torch.no_grad():
        for _ in range(horizon):
            logits = model(idx)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)
            idx = torch.cat([idx, nxt], dim=1)
    out = idx[:, -horizon:].numpy()
    return decode(out, m, sd)


def evaluate_series(torch, model, y, origins, n_samples=100, seed=SEED):
    scale = float(np.mean(np.abs(y[PERIOD:] - y[:-PERIOD])))
    errs, cover80, cover95, widths = [], [], [], []
    for o in origins:
        paths = sample_paths(torch, model, y[o - CONTEXT:o], n_samples=n_samples, seed=seed + o)
        med = np.median(paths, axis=0)
        truth = y[o:o + HORIZON]
        errs.append(np.abs(med - truth) / scale)
        lo80, hi80 = np.percentile(paths, [10, 90], axis=0)
        lo95, hi95 = np.percentile(paths, [2.5, 97.5], axis=0)
        cover80.append(((truth >= lo80) & (truth <= hi80)).astype(float))
        cover95.append(((truth >= lo95) & (truth <= hi95)).astype(float))
        widths.append(hi95 - lo95)
    return (np.array(errs), np.array(cover80), np.array(cover95), np.array(widths))


def corpus(n_per_family=N_PER_FAMILY, families=None):
    kw = {} if families is None else {"families": families}
    return synthetic_corpus(seed=SEED, n_per_family=n_per_family, length=LENGTH,
                            period=PERIOD, **kw)


def co2_origins():
    n = len(CO2)
    return list(range(n - 120, n - HORIZON + 1))


# ---------------------------------------------------------------------------
# Stage 1: the floor the alphabet puts under everything
# ---------------------------------------------------------------------------
def stage_quantisation():
    rows = []
    origins = co2_origins()
    scale_co2 = float(np.mean(np.abs(CO2[PERIOD:] - CO2[:-PERIOD])))
    for bins in [32, 64, 128, 512]:
        errs = []
        for o in origins:
            win = CO2[o - CONTEXT:o]
            m = float(win.mean())
            sd = float(win.std()) + 1e-6
            truth = CO2[o:o + HORIZON]
            back = decode(encode(truth, m, sd, bins), m, sd, bins)
            errs.append(np.abs(back - truth) / scale_co2)
        e = np.array(errs)
        rows.append({"bins": bins, "mase": r(float(e.mean()), 5),
                     "step_ppm": r(float(2 * CLIP * np.mean([np.std(CO2[o - CONTEXT:o])
                                                             for o in origins]) / bins), 4)})
    demo_o = origins[len(origins) // 2]
    win = CO2[demo_o - CONTEXT:demo_o]
    m = float(win.mean())
    sd = float(win.std()) + 1e-6
    return {"rows": rows, "clip": CLIP, "bins": BINS, "context": CONTEXT,
            "n_origins": len(origins),
            "demo": {"label": CO2LAB[demo_o], "window": r(win, 4),
                     "tokens": encode(win, m, sd).tolist(),
                     "reconstructed": r(decode(encode(win, m, sd), m, sd), 4),
                     "mean": r(m, 4), "sd": r(sd, 4)}}


# ---------------------------------------------------------------------------
# Stage 2: pretrain, and forecast something it has never seen
# ---------------------------------------------------------------------------
def stage_pretrain():
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)

    items = corpus()
    tokens = token_windows(items)
    model = make_model(torch)
    log = []
    train_model(torch, model, tokens, log=log)
    torch.save(model.state_dict(), CACHE / "model.pt")

    origins = co2_origins()
    errs, c80, c95, widths = evaluate_series(torch, model, CO2, origins)
    sun_origins = list(range(len(SUN) - 80, len(SUN) - HORIZON + 1))
    serrs, sc80, sc95, _ = evaluate_series(torch, model, SUN, sun_origins)

    demo_o = origins[len(origins) // 2]
    paths = sample_paths(torch, model, CO2[demo_o - CONTEXT:demo_o], n_samples=100,
                         seed=SEED + demo_o)
    return {
        "loss": r(log, 5),
        "n_series": len(items), "n_windows": int(len(tokens)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "co2": {"mase": r(float(errs.mean()), 4),
                "per_h": r(errs.mean(axis=0).tolist(), 4),
                "n_origins": len(origins),
                "coverage": [{"h": i + 1, "c80": r(float(c80[:, i].mean()), 4),
                              "c95": r(float(c95[:, i].mean()), 4),
                              "w95": r(float(widths[:, i].mean()), 4)}
                             for i in range(HORIZON)]},
        "sunspots": {"mase": r(float(serrs.mean()), 4),
                     "per_h": r(serrs.mean(axis=0).tolist(), 4),
                     "n_origins": len(sun_origins),
                     "coverage": [{"h": i + 1, "c80": r(float(sc80[:, i].mean()), 4),
                                   "c95": r(float(sc95[:, i].mean()), 4)}
                                  for i in range(HORIZON)]},
        "demo": {"label": CO2LAB[demo_o], "origin": int(demo_o),
                 "history": r(CO2[demo_o - CONTEXT:demo_o], 4),
                 "truth": r(CO2[demo_o:demo_o + HORIZON], 4),
                 "paths": r(paths[:40], 4),
                 "median": r(np.median(paths, axis=0), 4),
                 "lo80": r(np.percentile(paths, 10, axis=0), 4),
                 "hi80": r(np.percentile(paths, 90, axis=0), 4),
                 "lo95": r(np.percentile(paths, 2.5, axis=0), 4),
                 "hi95": r(np.percentile(paths, 97.5, axis=0), 4)},
    }


# ---------------------------------------------------------------------------
# Stage 3: what the corpus is worth
# ---------------------------------------------------------------------------
def stage_corpus_size():
    import torch
    torch.set_num_threads(THREADS)
    rows = []
    origins = co2_origins()[::2]
    for share, n_fam in [(0.25, 15), (0.5, 30), (1.0, N_PER_FAMILY)]:
        torch.manual_seed(SEED)
        items = corpus(n_per_family=n_fam)
        tokens = token_windows(items)
        model = make_model(torch)
        train_model(torch, model, tokens)
        errs, _, _, _ = evaluate_series(torch, model, CO2, origins, n_samples=40)
        rows.append({"share": share, "series": len(items), "windows": int(len(tokens)),
                     "mase": r(float(errs.mean()), 4)})
        print(f"    corpus {len(items)} series: {errs.mean():.4f}")
    return {"rows": rows, "n_origins": len(origins), "n_samples": 40}


# ---------------------------------------------------------------------------
# Stage 4: the family it was never shown
# ---------------------------------------------------------------------------
def stage_unseen_family():
    import torch
    torch.set_num_threads(THREADS)
    held = "trend_season"
    families = ["trend_season", "ar1", "random_walk", "level_shift", "damped_trend",
                "seasonal_only", "noise"]
    without = [f for f in families if f != held]

    test = corpus(n_per_family=8, families=(held,))
    rows = {}
    for name, fams in [("saw it", families), ("never saw it", without)]:
        torch.manual_seed(SEED)
        items = corpus(n_per_family=N_PER_FAMILY, families=tuple(fams))
        tokens = token_windows(items)
        model = make_model(torch)
        train_model(torch, model, tokens)
        errs = []
        for s in test:
            y = np.asarray(s["y"], dtype=float)
            ors = list(range(len(y) - 60, len(y) - HORIZON + 1, 6))
            e, _, _, _ = evaluate_series(torch, model, y, ors, n_samples=40)
            errs.append(e.mean())
        rows[name] = {"mase": r(float(np.mean(errs)), 4), "n_series": len(test),
                      "families": len(fams)}
        print(f"    {name}: {rows[name]['mase']}")
    return {"held_out": held, "rows": rows,
            "gap": r(rows["never saw it"]["mase"] - rows["saw it"]["mase"], 4),
            "ratio": r(rows["never saw it"]["mase"] / rows["saw it"]["mase"], 4)}


# ---------------------------------------------------------------------------
# Stage 5: how much history it actually uses
# ---------------------------------------------------------------------------
def stage_context():
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    model = make_model(torch)
    model.load_state_dict(torch.load(CACHE / "model.pt"))
    origins = co2_origins()[::2]
    scale = float(np.mean(np.abs(CO2[PERIOD:] - CO2[:-PERIOD])))
    rows = []
    for ctx in [12, 24, 36, 48]:
        errs = []
        for o in origins:
            paths = sample_paths(torch, model, CO2[o - ctx:o], n_samples=40, seed=SEED + o)
            errs.append(np.abs(np.median(paths, axis=0) - CO2[o:o + HORIZON]) / scale)
        rows.append({"context": ctx, "mase": r(float(np.array(errs).mean()), 4)})
        print(f"    context {ctx}: {rows[-1]['mase']}")
    return {"rows": rows, "trained_on": CONTEXT, "n_origins": len(origins)}


def main():
    import torch

    print("the alphabet")
    quant = cached("quantisation", stage_quantisation)
    print("pretraining")
    pre = cached("pretrain", stage_pretrain)
    print("corpus size")
    size = cached("corpus_size", stage_corpus_size)
    print("the unseen family")
    unseen = cached("unseen_family", stage_unseen_family)
    print("context length")
    ctx = cached("context", stage_context)

    board = None
    if PREV is not None:
        board = {"rows": [{"model": x["model"], "mase_mean": x["mase_mean"],
                           "mase": x["mase"]} for x in PREV["rolling"]["rows"]],
                 "n_origins": PREV["rolling"]["n_origins"],
                 "coverage": PREV["rolling"]["coverage"]}

    data = {
        "meta": {
            "seed": SEED, "context": CONTEXT, "horizon": HORIZON, "bins": BINS,
            "clip": CLIP, "epochs": EPOCHS, "length": LENGTH,
            "threads": THREADS, "torch": torch.__version__, "numpy": np.__version__,
            "co2_digest": digest(CO2), "sun_digest": digest(SUN),
            "co2": CO2META, "sunspots": SUNMETA,
            "n_series": pre["n_series"], "n_windows": pre["n_windows"],
            "params": pre["params"],
        },
        "series": {"co2": r(CO2, 4), "co2_labels": CO2LAB},
        "quantisation": quant, "pretrain": pre, "corpus_size": size,
        "unseen": unseen, "context": ctx, "board": board,
    }
    (OUT / "zeroshot.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT / 'zeroshot.json'} "
          f"({(OUT / 'zeroshot.json').stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
