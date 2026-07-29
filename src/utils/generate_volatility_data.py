"""Everything article 4.2b cites, measured.

Two questions that look like one. Several series at once (a vector
autoregression on US macro accounts) and the size of the error rather than its
sign (a variance model on twenty years of S&P 500 returns).

What is measured here:

 1. A vector model costs k^2 p parameters, and whether that buys anything is a
    per variable question, so it is asked per variable against the same
    univariate model on the same origins.
 2. Granger causality between two independent random walks, five hundred times,
    which is where the phrase "Granger causes" earns its scare quotes: the
    rejection rate is measured in levels and again in differences.
 3. An impulse response, computed twice with the variables in two orders, so
    the reader can see the size of the thing the ordering decides.
 4. The mean of a return is not forecastable and its square is: both R^2 are
    measured on the same returns with the same lags.
 5. GARCH(1,1), its persistence and the half life that follows from it, then
    scored out of sample against a rolling window, against EWMA, and against
    the VIX, which is a different instrument's forecast of the same quantity.
 6. Value at Risk exceedances against their nominal rate, which is the number
    that decides whether the tail assumption was any good.

Runs in about four minutes on four cores. Stages cached under
~/.atlas_vision_data/volatility_cache/.
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
OUT = ROOT / "volatility" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "volatility_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from src.utils.timeseries_data import (  # noqa: E402
    digest, macro_quarterly, sp500_returns, vix_daily,
)

SEED = 42
N_SPURIOUS = 500
SPUR_N = 200
VAR_TEST = 60          # quarters held out
H = 8                  # quarters ahead
VAR_LAGS = (1, 2, 4)


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


# ---------------------------------------------------------------------------
# The data, rounded before it is measured on.
# ---------------------------------------------------------------------------
MACRO, MLAB, MMETA = macro_quarterly()
# Growth rates in percent: the accounts are trending and a vector
# autoregression on trending levels is the spurious regression this article
# spends a section on.
GROWTH = {k: np.round(100 * np.diff(np.log(v)), 4) for k, v in MACRO.items()
          if k != "unemp"}
GROWTH["unemp"] = np.round(np.diff(MACRO["unemp"]), 4)
GLAB = MLAB[1:]
VARS = ["realgdp", "realcons", "realinv", "unemp"]

RET, RLAB, RMETA = sp500_returns()
RET = np.round(RET, 6)
VIX, VIXLAB, VIXMETA = vix_daily()
VIX = np.round(VIX, 4)


def acf(y, nlags):
    y = np.asarray(y, dtype=float)
    c = y - y.mean()
    d0 = float(c @ c)
    return np.array([1.0] + [float(c[k:] @ c[:-k] / d0) for k in range(1, nlags + 1)])


# ---------------------------------------------------------------------------
# Stage 1: one model for four series, against four models for four series
# ---------------------------------------------------------------------------
def stage_var():
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.arima.model import ARIMA

    Y = np.column_stack([GROWTH[v] for v in VARS])
    n = len(Y)
    origins = list(range(n - VAR_TEST, n - H + 1))
    rows = {}
    for p in VAR_LAGS:
        err = np.zeros((len(origins), H, len(VARS)))
        for i, o in enumerate(origins):
            m = VAR(Y[:o]).fit(p)
            f = m.forecast(Y[o - p:o], H)
            err[i] = np.abs(f - Y[o:o + H])
        rows[f"VAR({p})"] = err
    for p in VAR_LAGS:
        err = np.zeros((len(origins), H, len(VARS)))
        for i, o in enumerate(origins):
            for j, v in enumerate(VARS):
                m = ARIMA(Y[:o, j], order=(p, 0, 0), trend="c").fit()
                err[i, :, j] = np.abs(np.asarray(m.forecast(H)) - Y[o:o + H, j])
        rows[f"AR({p})"] = err
    mean_err = np.zeros((len(origins), H, len(VARS)))
    for i, o in enumerate(origins):
        mean_err[i] = np.abs(Y[:o].mean(axis=0)[None, :] - Y[o:o + H])

    # scale: the in sample mean absolute change of each column, so the four
    # variables can share a table
    scale = np.array([np.mean(np.abs(np.diff(Y[:origins[0], j]))) for j in range(len(VARS))])
    table = []
    per_origin = {}
    for name, err in list(rows.items()) + [("mean", mean_err)]:
        scaled = err / scale[None, None, :]
        per_var = scaled.mean(axis=(0, 1))
        per_origin[name] = scaled.mean(axis=(1, 2))
        table.append({"model": name, "overall": r(float(per_var.mean()), 4),
                      "per_var": {v: r(float(per_var[j]), 4) for j, v in enumerate(VARS)}})
    table.sort(key=lambda x: x["overall"])

    # A ranking without a floor under it is a ranking of nothing. The origins
    # are paired (every model forecasts the same quarters from the same data),
    # so the honest scale for a difference between two rows is the standard
    # error of the per origin difference, not the spread of either column.
    best = table[0]["model"]
    for row in table:
        d = per_origin[row["model"]] - per_origin[best]
        se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        row["gap_vs_best"] = r(float(d.mean()), 4)
        row["se"] = r(se, 4)
        row["t"] = r(float(d.mean() / se) if se > 0 else 0.0, 3)
        row["resolves"] = bool(abs(d.mean()) > 2 * se)

    counts = [{"k": len(VARS), "p": p, "var_params": len(VARS) ** 2 * p + len(VARS),
               "ar_params": len(VARS) * (p + 1)} for p in VAR_LAGS]

    winner_per_var = {}
    for v in VARS:
        best = min(table, key=lambda x: x["per_var"][v])
        winner_per_var[v] = best["model"]

    # the fitted VAR(4) coefficient matrix the page draws, on all the data
    m = VAR(Y).fit(2)
    coefs = np.asarray(m.coefs)          # (p, k, k)
    return {
        "table": table, "counts": counts, "vars": VARS,
        "n_origins": len(origins), "horizon": H,
        "winner_per_var": winner_per_var,
        "coefs": [r(c, 4) for c in coefs.tolist()],
        "coef_labels": VARS,
        "series": {v: r(GROWTH[v], 4) for v in VARS},
        "labels": GLAB,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Stage 2: Granger, on data with nothing in it
# ---------------------------------------------------------------------------
def stage_granger():
    from statsmodels.tsa.stattools import grangercausalitytests

    rng = np.random.default_rng(SEED)
    lag = 4

    def p_of(x, y, lag):
        d = np.column_stack([y, x])
        out = grangercausalitytests(d, maxlag=[lag])
        return float(out[lag][0]["ssr_ftest"][1])

    def classic_t(x, y):
        """The regression the phrase "spurious" was coined about: y on x, no
        lags of anything, and read the t statistic."""
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        s2 = float(res @ res) / (len(x) - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        return float(beta[1] / np.sqrt(cov[1, 1]))

    levels, diffs, ct_lv, ct_df, r2_lv = [], [], [], [], []
    for _ in range(N_SPURIOUS):
        a = np.cumsum(rng.normal(0, 1, SPUR_N))
        b = np.cumsum(rng.normal(0, 1, SPUR_N))
        levels.append(p_of(a, b, lag))
        diffs.append(p_of(np.diff(a), np.diff(b), lag))
        ct_lv.append(abs(classic_t(a, b)))
        ct_df.append(abs(classic_t(np.diff(a), np.diff(b))))
        X = np.column_stack([np.ones(SPUR_N), a])
        beta, *_ = np.linalg.lstsq(X, b, rcond=None)
        res = b - X @ beta
        r2_lv.append(1 - float(res @ res) / float(((b - b.mean()) ** 2).sum()))
    levels = np.array(levels)
    diffs = np.array(diffs)
    ct_lv = np.array(ct_lv)
    ct_df = np.array(ct_df)

    # and the same test on the real accounts, where it is normally reported
    real = []
    for a in VARS:
        for b in VARS:
            if a == b:
                continue
            real.append({"cause": a, "effect": b,
                         "p": r(p_of(GROWTH[a], GROWTH[b], lag), 5)})
    real.sort(key=lambda x: x["p"])

    return {
        "n": N_SPURIOUS, "length": SPUR_N, "lag": lag,
        "levels_rate": r(float((levels < 0.05).mean()), 4),
        "diffs_rate": r(float((diffs < 0.05).mean()), 4),
        "levels_rate01": r(float((levels < 0.01).mean()), 4),
        "diffs_rate01": r(float((diffs < 0.01).mean()), 4),
        "nominal": 0.05,
        "classic_levels_rate": r(float((ct_lv > 1.96).mean()), 4),
        "classic_diffs_rate": r(float((ct_df > 1.96).mean()), 4),
        "classic_median_t": r(float(np.median(ct_lv)), 3),
        "classic_median_r2": r(float(np.median(r2_lv)), 4),
        "levels_hist": r(np.histogram(levels, bins=20, range=(0, 1))[0].tolist(), 0),
        "diffs_hist": r(np.histogram(diffs, bins=20, range=(0, 1))[0].tolist(), 0),
        "real": real,
        "example": {
            "a": r(np.cumsum(np.random.default_rng(SEED + 3).normal(0, 1, SPUR_N)), 3),
            "b": r(np.cumsum(np.random.default_rng(SEED + 4).normal(0, 1, SPUR_N)), 3),
        },
    }


# ---------------------------------------------------------------------------
# Stage 3: the impulse response, and the order nobody mentions
# ---------------------------------------------------------------------------
def stage_irf():
    from statsmodels.tsa.api import VAR

    steps = 12
    orders = {
        "gdp first": ["realgdp", "realcons", "realinv", "unemp"],
        "investment first": ["realinv", "realcons", "realgdp", "unemp"],
    }
    out = {}
    for name, order in orders.items():
        Y = np.column_stack([GROWTH[v] for v in order])
        m = VAR(Y).fit(2)
        irf = m.irf(steps)
        # response of every variable to a one standard deviation shock in
        # investment, orthogonalised by a Cholesky factor of the residual
        # covariance, which is where the ordering enters
        j = order.index("realinv")
        resp = np.asarray(irf.orth_irfs)[:, :, j]
        out[name] = {"order": order,
                     "resp": {v: r(resp[:, order.index(v)], 5) for v in VARS}}
    a = np.array([out["gdp first"]["resp"][v] for v in VARS])
    b = np.array([out["investment first"]["resp"][v] for v in VARS])
    peak = {}
    for i, v in enumerate(VARS):
        peak[v] = {"gdp_first": r(float(np.max(np.abs(a[i]))), 5),
                   "inv_first": r(float(np.max(np.abs(b[i]))), 5),
                   "ratio": r(float(np.max(np.abs(b[i])) / max(1e-12, np.max(np.abs(a[i])))), 4)}
    return {"steps": steps, "orders": out, "peak": peak,
            "max_gap": r(float(np.max(np.abs(a - b))), 5),
            "worst_var": VARS[int(np.argmax(np.max(np.abs(a - b), axis=1)))]}


# ---------------------------------------------------------------------------
# Stage 4: no mean to forecast, and a great deal of variance
# ---------------------------------------------------------------------------
def stage_two_r2():
    lags = 5
    n = len(RET)

    def r2_of(target, source, lags):
        m = len(target) - lags
        X = np.column_stack([np.ones(m)] + [source[lags - j - 1:len(source) - j - 1][:m]
                                            for j in range(lags)])
        Y = target[lags:lags + m]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        return 1 - float(res @ res) / float(((Y - Y.mean()) ** 2).sum())

    sq = RET ** 2
    ab = np.abs(RET)
    return {
        "lags": lags, "n": int(n),
        "r2_return": r(r2_of(RET, RET, lags), 6),
        "r2_square": r(r2_of(sq, sq, lags), 6),
        "r2_abs": r(r2_of(ab, ab, lags), 6),
        "acf_return": r(acf(RET, 40), 5),
        "acf_abs": r(acf(ab, 40), 5),
        "acf_square": r(acf(sq, 40), 5),
        "band": r(1.96 / np.sqrt(n), 5),
        "ljung": None,
    }


# ---------------------------------------------------------------------------
# Stage 5: GARCH, its persistence, and three rivals out of sample
# ---------------------------------------------------------------------------
def garch_filter(r_, omega, alpha, beta, var0):
    """The recursion, written out because the page runs it too."""
    n = len(r_)
    v = np.empty(n)
    prev = var0
    for i in range(n):
        v[i] = prev
        prev = omega + alpha * r_[i] ** 2 + beta * prev
    return v


def garch_nll(r_, omega, alpha, beta, var0):
    v = garch_filter(r_, omega, alpha, beta, var0)
    return float(0.5 * np.sum(np.log(2 * np.pi * v) + r_ ** 2 / v))


def stage_garch():
    from arch import arch_model

    y = RET - RET.mean()
    res = arch_model(y, mean="zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    pers = alpha + beta
    half = float(np.log(0.5) / np.log(pers))
    uncond = omega / (1 - pers)

    rest = arch_model(y, mean="zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    nu = float(rest.params["nu"])

    # The page runs the same filter, so where the recursion starts has to be
    # pinned down rather than guessed. `arch` does not start at the sample
    # variance: it backcasts an exponentially weighted average of the first
    # observations and uses it for BOTH terms of the first step, so the first
    # variance is omega + (alpha + beta) * backcast. Starting at the sample
    # variance instead leaves a difference of 0.36 that decays like beta^t, and
    # a guard calibrated on that would be measuring the initial condition.
    var0 = float(np.var(y))
    v_arch = np.asarray(res.conditional_volatility) ** 2
    backcast = float(res.model._backcast)
    start = omega + (alpha + beta) * backcast
    v_bc = garch_filter(y, omega, alpha, beta, start)
    v_naive = garch_filter(y, omega, alpha, beta, var0)
    agree = float(np.max(np.abs(v_bc - v_arch)))

    return {
        "omega": r(omega, 8), "alpha": r(alpha, 5), "beta": r(beta, 5),
        "persistence": r(pers, 5), "half_life": r(half, 2),
        "uncond_var": r(uncond, 5), "uncond_vol_annual": r(np.sqrt(uncond * 252), 3),
        "nu": r(nu, 3), "nll": r(garch_nll(y, omega, alpha, beta, start), 3),
        "backcast": r(backcast, 6), "sample_var": r(var0, 6),
        "start_var": r(start, 6),
        "agreement": float(f"{agree:.3e}"),
        "start_gap": float(f"{float(np.max(np.abs(v_naive - v_bc))):.3e}"),
        "vol": r(np.sqrt(v_bc), 5),
        "mean": r(float(RET.mean()), 6),
        "n": int(len(y)),
        "worst_day": {"i": int(np.argmin(RET)), "label": RLAB[int(np.argmin(RET))],
                      "ret": r(float(RET.min()), 4)},
        "peak_vol": {"i": int(np.argmax(v_bc)), "label": RLAB[int(np.argmax(v_bc))],
                     "vol_annual": r(float(np.sqrt(v_bc.max() * 252)), 2)},
    }


def stage_forecast_var():
    """Four forecasts of tomorrow's variance, scored on the same days.

    The target is the squared return, which is an unbiased but extremely noisy
    proxy: under a correct model its own R^2 ceiling is bounded, so the page
    reports QLIKE (which a noisy proxy leaves consistent) beside the mean
    squared error, and says which ranking it trusts.
    """
    from arch import arch_model

    y = RET - RET.mean()
    n = len(y)
    start = 1000
    step = 20                       # refit monthly, filter daily in between
    fc = {k: np.full(n, np.nan) for k in ["GARCH", "EWMA", "rolling 60", "constant"]}
    lam = 0.94
    ew = np.var(y[:start])
    ew_path = np.empty(n)
    for i in range(n):
        ew_path[i] = ew
        ew = lam * ew + (1 - lam) * y[i] ** 2
    for o in range(start, n, step):
        res = arch_model(y[:o], mean="zero", vol="GARCH", p=1, q=1,
                         dist="normal").fit(disp="off")
        w = float(res.params["omega"])
        a = float(res.params["alpha[1]"])
        b = float(res.params["beta[1]"])
        v = float(np.asarray(res.conditional_volatility)[-1] ** 2)
        last = y[o - 1] ** 2
        for k in range(o, min(o + step, n)):
            v = w + a * last + b * v
            fc["GARCH"][k] = v
            last = y[k] ** 2
        for k in range(o, min(o + step, n)):
            fc["EWMA"][k] = ew_path[k]
            fc["rolling 60"][k] = np.var(y[max(0, k - 60):k])
            fc["constant"][k] = np.var(y[:o])
    mask = ~np.isnan(fc["GARCH"])
    target = y ** 2
    rows = []
    for k, v in fc.items():
        e = v[mask]
        t = target[mask]
        mse = float(np.mean((t - e) ** 2))
        qlike = float(np.mean(np.log(e) + t / e))
        rows.append({"model": k, "mse": r(mse, 4), "qlike": r(qlike, 5),
                     "corr": r(float(np.corrcoef(np.sqrt(e), np.abs(t) ** 0.5)[0, 1]), 4)})
    rows.sort(key=lambda x: x["qlike"])
    return {"rows": rows, "n_scored": int(mask.sum()), "start": start, "refit_every": step,
            "target": "squared daily return"}


def stage_vix():
    """A second instrument's forecast of the same quantity.

    The VIX is a 30 calendar day implied volatility. The comparable GARCH
    number is its 22 trading day ahead average variance, annualised, so that
    is what is built here, and both are scored against the realised volatility
    of the 22 days that followed.
    """
    from arch import arch_model

    y = RET - RET.mean()
    # align the two files by date
    ridx = {d: i for i, d in enumerate(RLAB)}
    pairs = [(ridx[d], i) for i, d in enumerate(VIXLAB) if d in ridx]
    res = arch_model(y, mean="zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    w = float(res.params["omega"])
    a = float(res.params["alpha[1]"])
    b = float(res.params["beta[1]"])
    v = np.asarray(res.conditional_volatility) ** 2
    pers = a + b
    uncond = w / (1 - pers)
    horizon = 22
    g, vx, realised, labels = [], [], [], []
    for ri, vi in pairs:
        if ri + horizon >= len(y) or ri < 1:
            continue
        # average variance over the next `horizon` days under the fitted model
        v1 = w + a * y[ri - 1] ** 2 + b * v[ri - 1]
        tot = 0.0
        cur = v1
        for _ in range(horizon):
            tot += cur
            cur = uncond + pers * (cur - uncond)
        # returns are already in percent, so the square root of an annualised
        # variance is already an annualised volatility in percent, and the
        # extra factor of a hundred that first went in here put the model at
        # 1367 beside a VIX of 14.7
        g.append(np.sqrt(tot / horizon * 252))
        vx.append(VIX[vi])
        realised.append(np.sqrt(np.mean(y[ri:ri + horizon] ** 2) * 252))
        labels.append(VIXLAB[vi])
    g = np.array(g)
    vx = np.array(vx)
    realised = np.array(realised)
    out = {"n": int(len(g)), "horizon": horizon,
           "corr_garch_vix": r(float(np.corrcoef(g, vx)[0, 1]), 4),
           "corr_garch_real": r(float(np.corrcoef(g, realised)[0, 1]), 4),
           "corr_vix_real": r(float(np.corrcoef(vx, realised)[0, 1]), 4),
           "mean_garch": r(float(g.mean()), 3), "mean_vix": r(float(vx.mean()), 3),
           "mean_real": r(float(realised.mean()), 3),
           "premium": r(float((vx - realised).mean()), 3),
           "start": labels[0], "end": labels[-1]}
    # a thinned scatter for the page
    keep = np.linspace(0, len(g) - 1, 240).astype(int)
    out["scatter"] = [{"g": r(g[i], 2), "v": r(vx[i], 2), "r": r(realised[i], 2)}
                      for i in keep]
    out["path"] = [{"t": labels[i], "g": r(g[i], 2), "v": r(vx[i], 2),
                    "r": r(realised[i], 2)} for i in
                   range(0, len(g), max(1, len(g) // 300))]
    return out


def stage_var_risk():
    """The number that decides whether the tail assumption was any good."""
    from arch import arch_model
    from scipy import stats

    y = RET - RET.mean()
    n = len(y)
    start = 1000
    step = 20
    levels = [0.01, 0.05]
    models = ["GARCH normal", "GARCH t", "historical 500", "normal constant"]
    hits = {m: {lv: [] for lv in levels} for m in models}
    for o in range(start, n, step):
        rn = arch_model(y[:o], mean="zero", vol="GARCH", p=1, q=1,
                        dist="normal").fit(disp="off")
        rt = arch_model(y[:o], mean="zero", vol="GARCH", p=1, q=1,
                        dist="t").fit(disp="off")
        nu = float(rt.params["nu"])
        pn = rn.params
        pt = rt.params
        vn = float(np.asarray(rn.conditional_volatility)[-1] ** 2)
        vt = float(np.asarray(rt.conditional_volatility)[-1] ** 2)
        lastn = y[o - 1] ** 2
        sd_hist = np.sort(y[max(0, o - 500):o])
        sd_const = float(np.std(y[:o]))
        for k in range(o, min(o + step, n)):
            vn = float(pn["omega"]) + float(pn["alpha[1]"]) * lastn + float(pn["beta[1]"]) * vn
            vt = float(pt["omega"]) + float(pt["alpha[1]"]) * lastn + float(pt["beta[1]"]) * vt
            lastn = y[k] ** 2
            for lv in levels:
                zn = stats.norm.ppf(lv)
                zt = stats.t.ppf(lv, nu) / np.sqrt(nu / (nu - 2))
                hits["GARCH normal"][lv].append(y[k] < zn * np.sqrt(vn))
                hits["GARCH t"][lv].append(y[k] < zt * np.sqrt(vt))
                hits["historical 500"][lv].append(
                    y[k] < sd_hist[int(lv * len(sd_hist))])
                hits["normal constant"][lv].append(y[k] < zn * sd_const)
    rows = []
    for m in models:
        row = {"model": m}
        for lv in levels:
            h = np.array(hits[m][lv], dtype=float)
            obs = float(h.mean())
            n_h = len(h)
            # Kupiec unconditional coverage test
            x = h.sum()
            if 0 < x < n_h:
                ll0 = x * np.log(lv) + (n_h - x) * np.log(1 - lv)
                ll1 = x * np.log(x / n_h) + (n_h - x) * np.log(1 - x / n_h)
                lr = 2 * (ll1 - ll0)
                p = float(1 - stats.chi2.cdf(lr, 1))
            else:
                lr, p = float("nan"), float("nan")
            row[f"obs{int(lv * 100)}"] = r(obs, 5)
            row[f"exp{int(lv * 100)}"] = lv
            row[f"ratio{int(lv * 100)}"] = r(obs / lv, 3)
            row[f"kupiec{int(lv * 100)}"] = r(p, 4)
        rows.append(row)
    return {"rows": rows, "n_days": int(len(hits[models[0]][0.01])), "levels": levels,
            "start": start}


def main():
    import arch
    import statsmodels

    print("var")
    var = cached("var", stage_var)
    print("granger")
    gr = cached("granger", stage_granger)
    print("irf")
    irf = cached("irf", stage_irf)
    print("two r2")
    two = cached("two_r2", stage_two_r2)
    print("garch")
    ga = cached("garch", stage_garch)
    print("variance forecast")
    fv = cached("forecast_var", stage_forecast_var)
    print("vix")
    vx = cached("vix", stage_vix)
    print("value at risk")
    vr = cached("var_risk", stage_var_risk)

    thin = max(1, len(RET) // 1400)
    data = {
        "meta": {
            "macro": MMETA, "sp500": RMETA, "vix": VIXMETA,
            "macro_digest": digest(np.column_stack([GROWTH[v] for v in VARS])),
            "ret_digest": digest(RET),
            "seed": SEED, "threads": min(4, os.cpu_count() or 4),
            "statsmodels": statsmodels.__version__, "arch": arch.__version__,
            "numpy": np.__version__,
            "ret_n": int(len(RET)), "macro_n": int(len(GLAB)),
        },
        "returns": {
            "y": r(RET, 6), "labels": RLAB,
            "thin": [{"t": RLAB[i], "r": r(RET[i], 4)} for i in range(0, len(RET), thin)],
        },
        "var": var, "granger": gr, "irf": irf, "two_r2": two,
        "garch": ga, "forecast": fv, "vix": vx, "risk": vr,
    }
    (OUT / "volatility.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT / 'volatility.json'} "
          f"({(OUT / 'volatility.json').stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
