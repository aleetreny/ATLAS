"""The series module 4.2 shares, and the proof that it shares them.

Five articles in this branch forecast the same things, and "the same series"
has to be a fact rather than an intention: the module before this one paid for
that lesson twice (the digits of 3.2 are checked against the encoder article 30
published, and the scenes of 24 are checked against the cache article 23 left).
So every loader here is deterministic, bundled with its library rather than
downloaded, and carries a digest that each generator asserts.

What is in here:

  co2_monthly()      Mauna Loa CO2, weekly flask samples averaged into months.
                     Trend and a fixed twelve month cycle: the series the
                     seasonal machinery was designed for.
  sunspots_yearly()  Yearly sunspot counts since 1700. A cycle with no fixed
                     period, which is the counterexample.
  macro_quarterly()  US quarterly macro accounts. Several series at once, which
                     is what a vector model needs.
  sp500_returns()    Daily log returns of the S&P 500, 1999 to 2018. Almost no
                     mean to forecast and a great deal of variance.
  synthetic_corpus() A collection of series drawn from written down processes,
                     so that a forecaster can be scored against the truth and
                     not only against the next observation.

Nothing here touches the network: `statsmodels` ships the first three inside
the package and `arch` ships the fourth, so a clean clone reproduces every
figure in the branch by running its generator.
"""
import hashlib

import numpy as np


def digest(x, places=6):
    """A stable fingerprint of an array, for asserting reuse across articles.

    Rounded before hashing because the alternative fingerprints the last bit of
    a float, and two machines that agree on a series to six decimals should
    agree on the digest too.
    """
    a = np.asarray(x, dtype=float).ravel()
    return hashlib.sha1(np.round(a, places).tobytes()).hexdigest()[:16]


def co2_monthly():
    """Monthly mean CO2 at Mauna Loa, 1958-03 to 2001-12 (526 months).

    The bundled series is weekly with 59 missing flask samples; averaging into
    calendar months leaves five months with nothing in them, and those five are
    filled by linear interpolation. Both numbers are reported on the page,
    because "we interpolated five points" is part of the data and not part of
    the plumbing.
    """
    import statsmodels.api as sm

    weekly = sm.datasets.co2.load_pandas().data["co2"]
    monthly = weekly.resample("MS").mean()
    gaps = int(monthly.isna().sum())
    filled = monthly.interpolate()
    meta = {
        "name": "Mauna Loa CO2",
        "unit": "parts per million",
        "weekly_n": int(weekly.shape[0]),
        "weekly_missing": int(weekly.isna().sum()),
        "months_interpolated": gaps,
        "period": 12,
        "start": str(filled.index[0].date())[:7],
        "end": str(filled.index[-1].date())[:7],
    }
    return filled.values.astype(float), [str(d.date())[:7] for d in filled.index], meta


def sunspots_yearly():
    """Yearly sunspot activity, 1700 to 2008 (309 years).

    The cycle is about eleven years and is not the same eleven years twice, so
    every seasonal method in the branch has somewhere to fail.
    """
    import statsmodels.api as sm

    d = sm.datasets.sunspots.load_pandas().data
    years = [str(int(y)) for y in d["YEAR"].values]
    meta = {
        "name": "sunspot activity",
        "unit": "yearly count",
        "period": 11,
        "start": years[0],
        "end": years[-1],
    }
    return d["SUNACTIVITY"].values.astype(float), years, meta


def macro_quarterly(columns=("realgdp", "realcons", "realinv", "unemp")):
    """US quarterly macro accounts, 1959Q1 to 2009Q3 (203 quarters)."""
    import statsmodels.api as sm

    d = sm.datasets.macrodata.load_pandas().data
    labels = [f"{int(y)}Q{int(q)}" for y, q in zip(d["year"], d["quarter"])]
    cols = {c: d[c].values.astype(float) for c in columns}
    meta = {
        "name": "US macro accounts",
        "period": 4,
        "start": labels[0],
        "end": labels[-1],
        "columns": list(columns),
    }
    return cols, labels, meta


def sp500_returns():
    """Daily log returns of the S&P 500 close, in percent, 1999 to 2018.

    Bundled with `arch`. Twenty years is long enough to contain 2008, which is
    the whole reason a variance model has anything to do.
    """
    import arch.data.sp500

    d = arch.data.sp500.load()
    close = d["Adj Close"].values.astype(float)
    r = 100.0 * np.diff(np.log(close))
    labels = [str(t.date()) for t in d.index[1:]]
    meta = {
        "name": "S&P 500 daily log return",
        "unit": "percent",
        "start": labels[0],
        "end": labels[-1],
        "n_closes": int(close.shape[0]),
    }
    return r, labels, meta


def vix_daily():
    """Daily VIX close. A second, independent forecast of the same variance."""
    import arch.data.vix

    d = arch.data.vix.load()
    return (d["vix"].values.astype(float),
            [str(t.date()) for t in d.index],
            {"name": "VIX", "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
             "n": int(d.shape[0])})


# ---------------------------------------------------------------------------
# The written down collection.
#
# Everything above is real and therefore has no answer key: the best any method
# can do is unknown, so "how far from the best" cannot be measured on it. These
# series have their components written down before they are observed, which is
# what makes a decomposition scoreable and what lets a pretrained model be
# tested on a family it has never seen.
# ---------------------------------------------------------------------------

FAMILIES = ("trend_season", "ar1", "random_walk", "level_shift", "damped_trend",
            "seasonal_only", "noise")


def _series(kind, rng, n, period):
    """One series and its parts. Returns (y, parts) with parts summing to y."""
    t = np.arange(n, dtype=float)
    zero = np.zeros(n)
    if kind == "trend_season":
        level = 50 + rng.uniform(-10, 10)
        slope = rng.uniform(-0.15, 0.35)
        amp = rng.uniform(2.0, 9.0)
        phase = rng.uniform(0, 2 * np.pi)
        trend = level + slope * t
        season = amp * np.sin(2 * np.pi * t / period + phase) + 0.4 * amp * np.sin(
            4 * np.pi * t / period + phase / 2)
        noise = rng.normal(0, rng.uniform(0.4, 1.6), n)
        return trend + season + noise, {"trend": trend, "season": season, "noise": noise}
    if kind == "ar1":
        phi = rng.uniform(-0.85, 0.92)
        sd = rng.uniform(0.6, 2.2)
        e = rng.normal(0, sd, n)
        y = np.empty(n)
        y[0] = e[0] / np.sqrt(max(1e-6, 1 - phi ** 2))
        for i in range(1, n):
            y[i] = phi * y[i - 1] + e[i]
        base = 20 + rng.uniform(-8, 8)
        return y + base, {"trend": np.full(n, base), "season": zero, "noise": y}
    if kind == "random_walk":
        sd = rng.uniform(0.4, 1.5)
        drift = rng.uniform(-0.06, 0.06)
        y = 30 + np.cumsum(rng.normal(drift, sd, n))
        return y, {"trend": y.copy(), "season": zero, "noise": zero}
    if kind == "level_shift":
        base = 40 + rng.uniform(-10, 10)
        y = np.full(n, base) + rng.normal(0, rng.uniform(0.3, 1.0), n)
        k = rng.integers(n // 4, 3 * n // 4)
        jump = rng.choice([-1.0, 1.0]) * rng.uniform(4, 14)
        step = np.where(t >= k, jump, 0.0)
        return y + step, {"trend": np.full(n, base) + step, "season": zero,
                          "noise": y - base}
    if kind == "damped_trend":
        level = 45 + rng.uniform(-10, 10)
        slope = rng.uniform(0.3, 1.2)
        phi = rng.uniform(0.80, 0.97)
        trend = level + slope * (1 - phi ** (t + 1)) / (1 - phi)
        noise = rng.normal(0, rng.uniform(0.4, 1.4), n)
        return trend + noise, {"trend": trend, "season": zero, "noise": noise}
    if kind == "seasonal_only":
        level = 35 + rng.uniform(-8, 8)
        amp = rng.uniform(3, 10)
        phase = rng.uniform(0, 2 * np.pi)
        season = amp * np.sin(2 * np.pi * t / period + phase)
        noise = rng.normal(0, rng.uniform(0.4, 1.6), n)
        return level + season + noise, {"trend": np.full(n, level), "season": season,
                                        "noise": noise}
    if kind == "noise":
        level = 25 + rng.uniform(-8, 8)
        noise = rng.normal(0, rng.uniform(0.8, 3.0), n)
        return level + noise, {"trend": np.full(n, level), "season": zero, "noise": noise}
    raise ValueError(kind)


def synthetic_corpus(seed=41, n_per_family=40, length=180, period=12,
                     families=FAMILIES):
    """A collection of series whose components are known before they are seen.

    Returns a list of dicts with `y`, `family`, and the three parts that sum to
    `y`. Used three ways in the branch: as an answer key for a decomposition,
    as the training corpus for a model that must then forecast something it has
    never seen, and as the population a leaderboard is averaged over.
    """
    rng = np.random.default_rng(seed)
    out = []
    for fam in families:
        for _ in range(n_per_family):
            y, parts = _series(fam, rng, length, period)
            out.append({"family": fam, "y": y, "period": period, **parts})
    return out


def rolling_origins(n, n_test, horizon, step=1):
    """Origins for a rolling evaluation: fit on [0, o), score [o, o+h).

    The last origin is the one whose whole horizon still fits inside the data,
    so every horizon is scored on the same number of origins and a leaderboard
    column is not quietly shorter than the one beside it.
    """
    first = n - n_test
    last = n - horizon
    return list(range(first, last + 1, step))
