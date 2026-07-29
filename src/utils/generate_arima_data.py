"""Everything article 4.2a cites, measured.

Five things get measured here, and three of them came out differently from the
script they were meant to illustrate:

 1. Stationarity. A unit root test passes a series that anyone can see is still
    seasonal, because a fixed cycle is stationary in the sense the test means.
    The picture and the p-value disagree and both are right.
 2. Order identification. The Box-Jenkins reading of a partial autocorrelation
    plot is a hypothesis test run ten times, so its false positive rate is
    predictable in closed form before any data exists. Predicted against
    measured, on white noise and on real AR processes, beside AIC.
 3. The equivalence. Simple exponential smoothing is ARIMA(0,1,1). The identity
    is exact and this file checks it to machine precision at every smoothing
    rate, and then fits both to the same series and gets two different answers,
    because an identity between models is not an identity between estimators.
 4. Coverage. Prediction intervals are the part of a forecast nobody scores.
    Scored here on a rolling origin, at every horizon, against their nominal
    level.
 5. A leaderboard on two series that disagree about everything, with the
    seasonal naive forecast as the unit.

Runs in about six minutes on four cores. Every stage is cached under
~/.atlas_vision_data/arima_cache/ so an interruption costs nothing.
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
OUT = ROOT / "arima" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "arima_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from src.utils.timeseries_data import (  # noqa: E402
    co2_monthly, digest, sunspots_yearly,
)

SEED = 41
MAX_LAG = 10          # how far the identification rule is allowed to look
N_SIM = 400           # simulated series per (process, length) cell
CO2_TEST = 120        # months held out of CO2 for the rolling evaluation
SUN_TEST = 80         # years held out of sunspots
H = 12                # longest horizon scored


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
    """Round for export, and for measurement: the page only ever has this."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, k) for v in np.asarray(x, dtype=float).tolist()]
    v = float(x)
    return round(v, k)


# ---------------------------------------------------------------------------
# The two series, rounded before anything is measured on them.
#
# Rounding at export instead would make every browser side check disagree with
# this file in the fourth decimal, and in a recursion that runs 526 steps that
# is not a rounding difference any more.
# ---------------------------------------------------------------------------
_y, _lab, CO2META = co2_monthly()
CO2 = np.round(_y, 4)
CO2LAB = _lab
_s, _slab, SUNMETA = sunspots_yearly()
SUN = np.round(_s, 4)
SUNLAB = _slab

CO2_DIGEST = digest(CO2)
SUN_DIGEST = digest(SUN)


# ---------------------------------------------------------------------------
# Autocorrelation, written out rather than imported.
#
# The page recomputes both of these in JavaScript, so the definition has to be
# pinned down on this side too: the divisor is n for every lag (the biased
# estimator statsmodels also uses by default), because the alternative gives a
# sequence that is not positive definite and a partial autocorrelation solved
# from it can leave the unit circle.
# ---------------------------------------------------------------------------
def acf(y, nlags):
    y = np.asarray(y, dtype=float)
    n = len(y)
    c = y - y.mean()
    d0 = float(np.dot(c, c))
    return np.array([1.0] + [float(np.dot(c[k:], c[:-k]) / d0) for k in range(1, nlags + 1)])


def pacf_durbin_levinson(y, nlags):
    """Partial autocorrelation by the Durbin-Levinson recursion.

    Same recursion in the browser, so the guard compares two implementations of
    one definition rather than two definitions.
    """
    rho = acf(y, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    v = 1.0
    for k in range(1, nlags + 1):
        num = rho[k] - sum(phi[k - 1, j] * rho[k - j] for j in range(1, k))
        den = v
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        v = v * (1 - phi[k, k] ** 2)
        out[k] = phi[k, k]
    return out


# ---------------------------------------------------------------------------
# Stage 1: is it stationary, and does the test agree with the picture
# ---------------------------------------------------------------------------
def stage_stationarity():
    from statsmodels.tsa.stattools import adfuller, kpss

    def variants(y, period):
        d1 = np.diff(y)
        dS = y[period:] - y[:-period]
        both = np.diff(dS)
        return [("raw", "as it comes", y),
                ("d1", "one difference", d1),
                ("dS", f"one seasonal difference (lag {period})", dS),
                ("d1dS", "both differences", both)]

    def row(name, label, s, period):
        a = adfuller(s, autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
        ac = acf(s, max(period, MAX_LAG))
        return {
            "name": name, "label": label, "n": int(len(s)),
            "adf_stat": r(a[0], 3), "adf_p": r(a[1], 4),
            "kpss_stat": r(k[0], 3), "kpss_p": r(k[1], 4),
            "acf1": r(ac[1], 4), "acf_season": r(ac[period], 4),
            "sd": r(float(np.std(s)), 4),
        }

    co2rows = [row(n, l, s, 12) for n, l, s in variants(CO2, 12)]
    sunrows = [row(n, l, s, 11) for n, l, s in variants(SUN, 11)]

    # The series the page plots for each state, and the two correlograms it
    # draws, both recomputed in the browser and checked against these.
    curves = {}
    for n, _, s in variants(CO2, 12):
        curves[n] = {
            "y": r(s, 4),
            "acf": r(acf(s, 36), 6),
            "pacf": r(pacf_durbin_levinson(s, 36), 6),
        }
    return {
        "co2": co2rows, "sunspots": sunrows, "curves": curves,
        "band36": r(1.96 / np.sqrt(len(CO2)), 4),
    }


# ---------------------------------------------------------------------------
# Stage 2: what a correlogram can and cannot identify
#
# The rule under test is the one every textbook prints: the partial
# autocorrelation of an AR(p) cuts off after lag p, so read off the last lag
# that pokes outside the +-1.96/sqrt(n) band. That is ten hypothesis tests at
# the 5% level, so on a series with no autocorrelation at all the rule reports
# some order with probability 1 - 0.95^10 before any data is collected. The
# closed form goes into the file next to the measurement.
# ---------------------------------------------------------------------------
def stage_identify():
    rng = np.random.default_rng(SEED)

    def sim_ar(phis, n, rng):
        p = len(phis)
        burn = 200
        e = rng.normal(0, 1, n + burn)
        y = np.zeros(n + burn)
        phis = np.asarray(phis, dtype=float)
        for i in range(p, n + burn):
            y[i] = float(phis @ y[i - p:i][::-1]) + e[i]
        return y[burn:]

    def rule_order(y, nlags=MAX_LAG):
        band = 1.96 / np.sqrt(len(y))
        p = pacf_durbin_levinson(y, nlags)
        sig = [k for k in range(1, nlags + 1) if abs(p[k]) > band]
        return (max(sig) if sig else 0), band

    def aic_order(y, nlags=MAX_LAG):
        """AR(k) by ordinary least squares on k lags, k = 0..nlags, min AIC.

        Same design matrix for every k (the last `nlags` rows are dropped once,
        not k times), or the criterion is comparing likelihoods of different
        samples and the comparison means nothing.
        """
        n = len(y)
        Y = y[nlags:]
        m = len(Y)
        best, bestk = np.inf, 0
        for k in range(0, nlags + 1):
            if k == 0:
                X = np.ones((m, 1))
            else:
                cols = [y[nlags - j - 1:n - j - 1] for j in range(k)]
                X = np.column_stack([np.ones(m)] + cols)
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ beta
            s2 = float(resid @ resid) / m
            aic = m * np.log(s2) + 2 * (k + 1)
            if aic < best:
                best, bestk = aic, k
        return bestk

    processes = [
        ("white noise", []),
        ("AR(1), 0.7", [0.7]),
        ("AR(2), 0.6 / -0.3", [0.6, -0.3]),
        ("AR(3), 0.5 / 0.2 / -0.25", [0.5, 0.2, -0.25]),
    ]
    lengths = [60, 120, 240]
    rows = []
    for pname, phis in processes:
        for n in lengths:
            hits_rule = hits_aic = 0
            over_rule = over_aic = 0
            for _ in range(N_SIM):
                y = sim_ar(phis, n, rng)
                pr, _ = rule_order(y)
                pa = aic_order(y)
                hits_rule += int(pr == len(phis))
                hits_aic += int(pa == len(phis))
                over_rule += int(pr > len(phis))
                over_aic += int(pa > len(phis))
            rows.append({
                "process": pname, "true_p": len(phis), "n": n,
                "rule_acc": r(hits_rule / N_SIM, 4), "aic_acc": r(hits_aic / N_SIM, 4),
                "rule_over": r(over_rule / N_SIM, 4), "aic_over": r(over_aic / N_SIM, 4),
            })

    predicted_fp = 1 - 0.95 ** MAX_LAG
    noise_rows = [x for x in rows if x["true_p"] == 0]
    measured_fp = float(np.mean([x["rule_over"] for x in noise_rows]))

    # Four correlograms the reader gets to identify, with the answer stored.
    examples = []
    for pname, phis in processes:
        y = sim_ar(phis, 200, rng)
        pr, band = rule_order(y)
        examples.append({
            "process": pname, "true_p": len(phis),
            "acf": r(acf(y, MAX_LAG), 5), "pacf": r(pacf_durbin_levinson(y, MAX_LAG), 5),
            "band": r(band, 5), "rule_p": pr, "aic_p": aic_order(y),
            # the whole series, not the tail: the page recomputes both
            # correlograms from it, and a guard that compares a correlogram of
            # 80 points against one of 200 measures nothing but the difference
            # between 80 and 200 (it read 0.15 before this was fixed)
            "y": r(y, 4), "n": int(len(y)),
        })
    rng2 = np.random.default_rng(SEED + 1)
    examples = [dict(e, shuffled=int(i)) for i, e in enumerate(examples)]
    rng2.shuffle(examples)

    return {
        "rows": rows, "examples": examples, "n_sim": N_SIM, "max_lag": MAX_LAG,
        "predicted_fp": r(predicted_fp, 4), "measured_fp": r(measured_fp, 4),
        "lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Stage 3: the identity, and then the two estimators
# ---------------------------------------------------------------------------
def ses_levels(y, alpha, level0=None):
    """The smoothing recursion. l[t] = a*y[t] + (1-a)*l[t-1]."""
    lvl = y[0] if level0 is None else level0
    out = np.empty(len(y))
    for i, v in enumerate(y):
        lvl = alpha * v + (1 - alpha) * lvl
        out[i] = lvl
    return out


def ma1_forecasts(y, theta, level0=None):
    """ARIMA(0,1,1) written in error form: yhat[t+1] = y[t] + theta*e[t].

    Started from the same place as the smoother, which is the whole point: the
    two recursions are the same recursion when theta = alpha - 1, and the only
    thing that can make them differ is where they start.
    """
    prev = y[0] if level0 is None else level0
    out = np.empty(len(y))
    for i, v in enumerate(y):
        e = v - prev
        prev = v + theta * e
        out[i] = prev
    return out


def stage_equivalence():
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    # (a) the identity, over the whole grid the page's slider can reach
    grid = [round(0.02 + 0.02 * i, 2) for i in range(50)]
    worst = 0.0
    for a in grid:
        d = np.max(np.abs(ses_levels(CO2, a) - ma1_forecasts(CO2, a - 1)))
        worst = max(worst, d)

    # (b) two estimators on the same real series
    ses = SimpleExpSmoothing(CO2, initialization_method="heuristic").fit(optimized=True)
    alpha_hat = float(ses.params["smoothing_level"])
    ar = ARIMA(CO2, order=(0, 1, 1), trend="n").fit()
    theta_hat = float(ar.params[0])
    f_ses = float(ses.forecast(1)[0])
    f_ar = float(ar.forecast(1)[0])

    # (c) the same two estimators on data that really is ARIMA(0,1,1), where
    #     the equivalence has somewhere to live: theta inside (-1, 0).
    rng = np.random.default_rng(SEED + 7)
    true_theta = -0.4
    n = 400
    e = rng.normal(0, 1.0, n + 1)
    z = np.cumsum(e[1:] + true_theta * e[:-1]) + 100.0
    z = np.round(z, 4)
    ses2 = SimpleExpSmoothing(z, initialization_method="heuristic").fit(optimized=True)
    ar2 = ARIMA(z, order=(0, 1, 1), trend="n").fit()
    a2 = float(ses2.params["smoothing_level"])
    t2 = float(ar2.params[0])

    return {
        "identity_worst": float(f"{worst:.3e}"),
        "identity_grid": [grid[0], grid[-1], len(grid)],
        "co2": {
            "alpha_hat": r(alpha_hat, 4), "theta_hat": r(theta_hat, 4),
            "implied_alpha": r(1 + theta_hat, 4),
            "f_ses": r(f_ses, 4), "f_arima": r(f_ar, 4),
            "h1_gap": r(abs(f_ses - f_ar), 4),
            "reachable": bool(-1 < theta_hat < 0),
            "sse_ses": r(float(ses.sse), 2),
        },
        "ma1": {
            "true_theta": true_theta, "n": n,
            "alpha_hat": r(a2, 4), "theta_hat": r(t2, 4),
            "implied_alpha": r(1 + t2, 4),
            "gap": r(abs(a2 - (1 + t2)), 4),
            "f_ses": r(float(ses2.forecast(1)[0]), 4),
            "f_arima": r(float(ar2.forecast(1)[0]), 4),
            "h1_gap": r(abs(float(ses2.forecast(1)[0]) - float(ar2.forecast(1)[0])), 4),
            "y": r(z[-120:], 4),
        },
        "sweep": [{"alpha": a,
                   "sse": r(float(np.sum((CO2[1:] - ses_levels(CO2, a)[:-1]) ** 2)), 2)}
                  for a in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]],
    }


# ---------------------------------------------------------------------------
# Stage 4 and 5: the rolling evaluation, coverage and the leaderboard
# ---------------------------------------------------------------------------
def mase_scale(train, period):
    """In sample mean absolute seasonal naive error: the unit of the table."""
    return float(np.mean(np.abs(train[period:] - train[:-period])))


def forecasts_at(train, h, period, models):
    """Every model's h step path from one origin, plus the interval model."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    out = {}
    n = len(train)
    if "naive" in models:
        out["naive"] = np.full(h, train[-1])
    if "seasonal naive" in models:
        idx = [(n - period + (i % period)) for i in range(h)]
        out["seasonal naive"] = train[np.array(idx)]
    if "drift" in models:
        slope = (train[-1] - train[0]) / (n - 1)
        out["drift"] = train[-1] + slope * np.arange(1, h + 1)
    if "mean" in models:
        out["mean"] = np.full(h, float(train.mean()))
    if "ETS" in models:
        e = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                 seasonal="add", seasonal_periods=period,
                                 initialization_method="estimated").fit()
        out["ETS"] = np.asarray(e.forecast(h), dtype=float)
    if "ETS(A,Ad,N)" in models:
        e = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        out["ETS(A,Ad,N)"] = np.asarray(e.forecast(h), dtype=float)
    if "SARIMA" in models:
        m = SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 1, 1, period),
                    trend="n").fit(disp=False)
        f = m.get_forecast(h)
        out["SARIMA"] = np.asarray(f.predicted_mean, dtype=float)
        out["_intervals"] = {
            "80": np.asarray(f.conf_int(alpha=0.2), dtype=float),
            "95": np.asarray(f.conf_int(alpha=0.05), dtype=float),
        }
    if "AR(9)" in models:
        m = ARIMA(train, order=(9, 0, 0), trend="c").fit()
        f = m.get_forecast(h)
        out["AR(9)"] = np.asarray(f.predicted_mean, dtype=float)
        out["_intervals"] = {
            "80": np.asarray(f.conf_int(alpha=0.2), dtype=float),
            "95": np.asarray(f.conf_int(alpha=0.05), dtype=float),
        }
    return out


def rolling(y, period, n_test, models, step=1):
    n = len(y)
    origins = list(range(n - n_test, n - H + 1, step))
    err = {m: np.zeros((len(origins), H)) for m in models}
    scale = np.zeros(len(origins))
    cover = {"80": np.zeros((len(origins), H)), "95": np.zeros((len(origins), H))}
    width = {"80": np.zeros((len(origins), H)), "95": np.zeros((len(origins), H))}
    for i, o in enumerate(origins):
        train, test = y[:o], y[o:o + H]
        scale[i] = mase_scale(train, period)
        f = forecasts_at(train, H, period, models)
        for m in models:
            err[m][i] = np.abs(f[m] - test)
        iv = f.get("_intervals")
        if iv is not None:
            for lvl in ("80", "95"):
                lo, hi = iv[lvl][:, 0], iv[lvl][:, 1]
                cover[lvl][i] = ((test >= lo) & (test <= hi)).astype(float)
                width[lvl][i] = hi - lo
    return origins, err, scale, cover, width


def summarise(origins, err, scale, models):
    rows = []
    for m in models:
        per_h = (err[m] / scale[:, None]).mean(axis=0)
        rows.append({"model": m,
                     "mase": r(per_h.tolist(), 4),
                     "mase_h1": r(float(per_h[0]), 4),
                     "mase_h12": r(float(per_h[-1]), 4),
                     "mase_mean": r(float(per_h.mean()), 4)})
    rows.sort(key=lambda x: x["mase_mean"])
    return rows


def stage_rolling_co2():
    models = ["naive", "seasonal naive", "drift", "ETS", "SARIMA"]
    origins, err, scale, cover, width = rolling(CO2, 12, CO2_TEST, models)
    rows = summarise(origins, err, scale, models)
    cov = [{"h": h + 1,
            "c80": r(float(cover["80"][:, h].mean()), 4),
            "c95": r(float(cover["95"][:, h].mean()), 4),
            "w80": r(float(width["80"][:, h].mean()), 4),
            "w95": r(float(width["95"][:, h].mean()), 4)} for h in range(H)]
    # one origin drawn on the page, with its intervals
    o = origins[len(origins) // 2]
    f = forecasts_at(CO2[:o], H, 12, models)
    iv = f["_intervals"]
    demo = {
        "origin": int(o), "label": CO2LAB[o],
        "history": r(CO2[o - 60:o], 4), "truth": r(CO2[o:o + H], 4),
        "sarima": r(f["SARIMA"], 4), "ets": r(f["ETS"], 4),
        "snaive": r(f["seasonal naive"], 4), "naive": r(f["naive"], 4),
        "lo80": r(iv["80"][:, 0], 4), "hi80": r(iv["80"][:, 1], 4),
        "lo95": r(iv["95"][:, 0], 4), "hi95": r(iv["95"][:, 1], 4),
    }
    return {"rows": rows, "coverage": cov, "n_origins": len(origins), "demo": demo,
            "test": CO2_TEST}


def stage_rolling_sun():
    models = ["naive", "seasonal naive", "mean", "ETS(A,Ad,N)", "AR(9)"]
    origins, err, scale, cover, width = rolling(SUN, 11, SUN_TEST, models)
    rows = summarise(origins, err, scale, models)
    cov = [{"h": h + 1,
            "c80": r(float(cover["80"][:, h].mean()), 4),
            "c95": r(float(cover["95"][:, h].mean()), 4)} for h in range(H)]
    return {"rows": rows, "coverage": cov, "n_origins": len(origins), "test": SUN_TEST}


def stage_refit_control():
    """What the shortcut costs.

    Every number above refits the model at every origin, which is the honest
    protocol and also the expensive one. The usual shortcut is to fit once and
    then filter the new observations through the fitted parameters. Both are
    run here on the same origins so the page can say what the shortcut is
    worth instead of assuming it is free.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    n = len(CO2)
    origins = list(range(n - CO2_TEST, n - H + 1, 4))
    first = origins[0]
    base = SARIMAX(CO2[:first], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                   trend="n").fit(disp=False)
    e_refit, e_fixed, scale = [], [], []
    for o in origins:
        train, test = CO2[:o], CO2[o:o + H]
        scale.append(mase_scale(train, 12))
        m = SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                    trend="n").fit(disp=False)
        e_refit.append(np.abs(np.asarray(m.forecast(H)) - test))
        fixed = base.apply(train, refit=False)
        e_fixed.append(np.abs(np.asarray(fixed.forecast(H)) - test))
    scale = np.array(scale)
    a = (np.array(e_refit) / scale[:, None]).mean()
    b = (np.array(e_fixed) / scale[:, None]).mean()
    return {"n_origins": len(origins), "refit": r(float(a), 4), "fixed": r(float(b), 4),
            "gap": r(float(b - a), 4), "ratio": r(float(b / a), 4),
            "first_origin": CO2LAB[first]}


def stage_seasonality_needs_a_period():
    """Does a seasonal model help a cycle that is not a season?

    Sunspots have a cycle of about eleven years that is not eleven years twice,
    so this fits the seasonal machinery at every period from 9 to 13 and
    reports what the best one is worth against no seasonal term at all.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    n = len(SUN)
    origins = list(range(n - SUN_TEST, n - H + 1, 2))
    rows = []
    for period in [0, 9, 10, 11, 12, 13]:
        errs, scale = [], []
        for o in origins:
            train, test = SUN[:o], SUN[o:o + H]
            scale.append(mase_scale(train, 11))
            if period == 0:
                m = SARIMAX(train, order=(2, 0, 1), trend="c").fit(disp=False)
            else:
                m = SARIMAX(train, order=(2, 0, 1),
                            seasonal_order=(1, 0, 0, period), trend="c").fit(disp=False)
            errs.append(np.abs(np.asarray(m.forecast(H)) - test))
        scale = np.array(scale)
        rows.append({"period": period,
                     "mase": r(float((np.array(errs) / scale[:, None]).mean()), 4)})
    best = min([x for x in rows if x["period"] > 0], key=lambda x: x["mase"])
    none = [x for x in rows if x["period"] == 0][0]
    return {"rows": rows, "best_period": best["period"], "best": best["mase"],
            "none": none["mase"], "helps": bool(best["mase"] < none["mase"]),
            "n_origins": len(origins)}


def main():
    import statsmodels

    threads = min(4, os.cpu_count() or 4)
    print("stationarity")
    stat = cached("stationarity", stage_stationarity)
    print("identification")
    ident = cached("identify", stage_identify)
    print("equivalence")
    equiv = cached("equivalence", stage_equivalence)
    print("rolling co2")
    roll = cached("rolling_co2", stage_rolling_co2)
    print("rolling sunspots")
    rsun = cached("rolling_sun", stage_rolling_sun)
    print("refit control")
    refit = cached("refit_control", stage_refit_control)
    print("period sweep")
    period = cached("period_sweep", stage_seasonality_needs_a_period)

    data = {
        "meta": {
            "co2": CO2META, "sunspots": SUNMETA,
            "co2_digest": CO2_DIGEST, "sun_digest": SUN_DIGEST,
            "co2_n": int(len(CO2)), "sun_n": int(len(SUN)),
            "seed": SEED, "horizon": H, "max_lag": MAX_LAG,
            "statsmodels": statsmodels.__version__,
            "numpy": np.__version__,
            "threads": threads,
        },
        "series": {
            "co2": {"y": r(CO2, 4), "labels": CO2LAB},
            "sunspots": {"y": r(SUN, 4), "labels": SUNLAB},
        },
        "stationarity": stat,
        "identify": ident,
        "equivalence": equiv,
        "rolling": roll,
        "sunspots": rsun,
        "refit": refit,
        "period": period,
    }
    (OUT / "arima.json").write_text(json.dumps(data, separators=(",", ":")))
    kb = (OUT / "arima.json").stat().st_size / 1024
    print(f"wrote {OUT / 'arima.json'} ({kb:.0f} kB)")
    print(f"co2 digest {CO2_DIGEST}  sunspots digest {SUN_DIGEST}")


if __name__ == "__main__":
    main()
