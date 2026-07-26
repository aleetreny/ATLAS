"""Data for the ATLAS 'Calibration' web article.

Same forest, same construction as generate_imbalanced_data.py (seed 17, all
2,747 cottonwood/willow positives, majority thinned to a round 1%, 60/40
split), because this article repairs the probabilities that article broke and
the numbers must be comparable. The only new cut: a quarter of the training
data is held out as a calibration set, because both repairs are themselves
fits and fitting them on the test set would be the oldest sin in the atlas.

What gets measured:
  * four models with four different relationships to the truth: logistic
    (honest), naive Bayes (delusionally overconfident), random forest
    (compressed toward the middle), and the SMOTE logistic from the previous
    article (uniformly inflated four-fold)
  * Platt scaling and isotonic regression fitted on the calibration split,
    applied to each, with reliability bins, ECE, Brier, AUROC and mean p
    before and after
  * the Brier decomposition (reliability - resolution + uncertainty), plus a
    'constant 1%' pseudo-model: perfectly calibrated, perfectly useless
  * how much calibration data each method needs: ECE against calibration-set
    size, eleven seeded draws per size, for both methods on the SMOTE
    logistic (the models with near-binary or nearly-calibrated scores are
    useless for this question: naive Bayes reduces both repairs to estimating
    two frequencies, and the forest has almost nothing left to repair)
  * a 300-point balanced sample for the browser's live PAV widget, scored by
    the crudest ranker imaginable, negated elevation plus distance to water,
    with scikit-learn's isotonic fit on the same points as the reference
    answer. Model scores are too good here: every decent model orders the
    balanced draw almost perfectly and the staircase collapses to two or
    three treads. The hand-made score gives the algorithm real work, and
    makes the deeper point that isotonic calibration needs only an ordering,
    not a model.

Run from anywhere: `python src/utils/generate_calibration_data.py`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.imbalanced_learning import ScratchSMOTE  # noqa: E402

OUT = ROOT / "calibration" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 17
EPS = 1e-6

# ------------------------------------------------ the forest, reconstructed
cov = fetch_covtype()
X_all = cov.data
y_all = (cov.target == 4).astype(int)

rs = np.random.RandomState(SEED)
neg_idx = np.where(y_all == 0)[0]
pos_idx = np.where(y_all == 1)[0]
keep_neg = rs.choice(neg_idx, int(y_all.sum()) * 99, replace=False)
idx = np.concatenate([pos_idx, keep_neg])
rs.shuffle(idx)
X, y = X_all[idx], y_all[idx]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
# The new cut: fit / calibration, stratified, out of the same training rows.
Xfit, Xcal, yfit, ycal = train_test_split(Xtr, ytr, test_size=0.25, random_state=SEED, stratify=ytr)
scaler = StandardScaler().fit(Xfit)
A, C, B = scaler.transform(Xfit), scaler.transform(Xcal), scaler.transform(Xte)
print(f"fit {len(yfit):,} ({int(yfit.sum())} positive), calibration {len(ycal):,} "
      f"({int(ycal.sum())} positive), test {len(yte):,} ({int(yte.sum())} positive)")

base_rate = float(yte.mean())


# ---------------------------------------------------------------- measuring
def bins_of(y_true, p, nbins=10, min_n=10):
    edges = np.linspace(0, 1, nbins + 1)
    out = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < nbins - 1 else (p >= lo) & (p <= hi)
        if m.sum() < min_n:
            continue
        out.append({"lo": rnd(lo, 2), "hi": rnd(hi, 2), "n": int(m.sum()),
                    "predicted": rnd(float(p[m].mean()), 4), "observed": rnd(float(y_true[m].mean()), 4)})
    return out


def ece_of(y_true, p, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    total = 0.0
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < nbins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() / len(p) * abs(p[m].mean() - y_true[m].mean())
    return float(total)


def decomp(y_true, p, nbins=10):
    """Murphy's decomposition of the Brier score over the same 10 bins:
    Brier = reliability - resolution + uncertainty."""
    edges = np.linspace(0, 1, nbins + 1)
    base = y_true.mean()
    rel = res = 0.0
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < nbins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rel += m.sum() / len(p) * (p[m].mean() - y_true[m].mean()) ** 2
        res += m.sum() / len(p) * (y_true[m].mean() - base) ** 2
    return rnd(rel, 5), rnd(res, 5), rnd(float(base * (1 - base)), 5)


def state(y_true, p):
    rel, res, unc = decomp(y_true, p)
    return {
        "ece": rnd(ece_of(y_true, p), 5),
        "brier": rnd(brier_score_loss(y_true, p), 5),
        "auroc": rnd(roc_auc_score(y_true, p), 4),
        "mean_p": rnd(float(p.mean())),
        "rel": rel, "res": res, "unc": unc,
        "bins": bins_of(y_true, p),
    }


# ----------------------------------------------------------------- the models
def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


models = {}
raw_cal = {}
raw_test = {}


def add_model(name, fit_fn):
    t0 = time.perf_counter()
    p_cal, p_test = fit_fn()
    secs = time.perf_counter() - t0
    raw_cal[name], raw_test[name] = p_cal, p_test
    models[name] = {"seconds": rnd(secs, 2), "raw": state(yte, p_test)}
    r = models[name]["raw"]
    print(f"  {name:<16} ece {r['ece']:.5f}  brier {r['brier']:.5f}  auroc {r['auroc']:.4f}  mean p {r['mean_p']:.4f}  ({secs:.1f}s)")


def fit_logistic():
    m = LogisticRegression(max_iter=3000).fit(A, yfit)
    return m.predict_proba(C)[:, 1], m.predict_proba(B)[:, 1]


def fit_nb():
    m = GaussianNB().fit(A, yfit)
    return m.predict_proba(C)[:, 1], m.predict_proba(B)[:, 1]


def fit_rf():
    m = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(A, yfit)
    return m.predict_proba(C)[:, 1], m.predict_proba(B)[:, 1]


def fit_smote_lr():
    Xs, ys = ScratchSMOTE(k_neighbors=5, random_state=SEED).fit_resample(A, yfit)
    m = LogisticRegression(max_iter=3000).fit(Xs, ys)
    return m.predict_proba(C)[:, 1], m.predict_proba(B)[:, 1]


print("\n=== the four models, raw ===")
add_model("logistic", fit_logistic)
add_model("naive_bayes", fit_nb)
add_model("random_forest", fit_rf)
add_model("smote_logistic", fit_smote_lr)

# The perfectly-calibrated, perfectly-useless reference.
p_const = np.full(len(yte), base_rate)
models["constant"] = {"seconds": 0.0, "raw": state(yte, p_const)}
r = models["constant"]["raw"]
print(f"  {'constant 1%':<16} ece {r['ece']:.5f}  brier {r['brier']:.5f}  auroc {r['auroc']:.4f}  mean p {r['mean_p']:.4f}")


# ------------------------------------------------------------- the two repairs
def platt(p_cal, y_cal, p_test):
    m = LogisticRegression(C=1e6, max_iter=3000).fit(logit(p_cal).reshape(-1, 1), y_cal)
    return m.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_test):
    m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_cal, y_cal)
    return m.predict(p_test)


print("\n=== after repair (ECE / Brier) ===")
for name in ["logistic", "naive_bayes", "random_forest", "smote_logistic"]:
    pc, pt = raw_cal[name], raw_test[name]
    models[name]["platt"] = state(yte, platt(pc, ycal, pt))
    models[name]["isotonic"] = state(yte, isotonic(pc, ycal, pt))
    a, b, c = models[name]["raw"], models[name]["platt"], models[name]["isotonic"]
    print(f"  {name:<16} raw {a['ece']:.5f}/{a['brier']:.5f}  platt {b['ece']:.5f}/{b['brier']:.5f}  "
          f"isotonic {c['ece']:.5f}/{c['brier']:.5f}")

# The analytic fix for a known resampling ratio: shift the logit by the log
# odds of the true against the trained prevalence (50% after SMOTE).
p_shift = 1 / (1 + np.exp(-(logit(raw_test["smote_logistic"]) + np.log(base_rate / (1 - base_rate)))))
models["smote_logistic"]["logit_shift"] = state(yte, p_shift)
s = models["smote_logistic"]["logit_shift"]
print(f"  smote + logit shift: ece {s['ece']:.5f}  brier {s['brier']:.5f}  (no calibration data used)")

# ------------------------------------------- how much data does repair need?
print("\n=== calibration-set size against test ECE (SMOTE logistic) ===")
sizes = [100, 300, 1000, 3000, 10000, 30000]
size_curve = {"model": "smote_logistic", "sizes": sizes, "platt": [], "isotonic": []}
pc_full, pt_full = raw_cal["smote_logistic"], raw_test["smote_logistic"]
for size in sizes:
    eces_p, eces_i = [], []
    for rep in range(11):
        rs2 = np.random.RandomState(100 + rep)
        pos = np.where(ycal == 1)[0]
        neg = np.where(ycal == 0)[0]
        n_pos = max(1, round(size * float(ycal.mean())))
        take = np.concatenate([rs2.choice(pos, n_pos, replace=False),
                               rs2.choice(neg, size - n_pos, replace=False)])
        eces_p.append(ece_of(yte, platt(pc_full[take], ycal[take], pt_full)))
        eces_i.append(ece_of(yte, isotonic(pc_full[take], ycal[take], pt_full)))
    size_curve["platt"].append(rnd(float(np.mean(eces_p)), 5))
    size_curve["isotonic"].append(rnd(float(np.mean(eces_i)), 5))
    print(f"  n={size:>6,}  platt {size_curve['platt'][-1]:.5f}  isotonic {size_curve['isotonic'][-1]:.5f}")

# ------------------------------------------------- the PAV widget's raw data
# A balanced 300-point draw from the calibration split, scored by a deliberate
# non-model: elevation plus distance to water, negated (low and wet should
# mean cottonwood) and rescaled to [0, 1]. Real models order this draw almost
# perfectly and give the algorithm nothing to pool; the crude score leaves
# genuine violations, and makes the point that isotonic regression asks only
# for an ordering. The scikit-learn isotonic fit on the same points ships as
# the reference answer.
rs3 = np.random.RandomState(SEED)
pos = np.where(ycal == 1)[0]
neg = np.where(ycal == 0)[0]
take = np.concatenate([rs3.choice(pos, 150, replace=False), rs3.choice(neg, 150, replace=False)])
crude = -(C[:, 0] + C[:, 3])          # standardised units: Elevation, Horizontal_Distance_To_Hydrology
sc = crude[take]
sc = (sc - sc.min()) / (sc.max() - sc.min())
yv = ycal[take]
order = np.argsort(sc, kind="stable")
sc, yv = sc[order], yv[order]
iso_demo = IsotonicRegression(y_min=0, y_max=1).fit(sc, yv)
pav = {
    "scores": [rnd(v, 5) for v in sc],
    "y": [int(v) for v in yv],
    "sk_fit": [rnd(v, 5) for v in iso_demo.predict(sc)],
}
print(f"\nPAV demo: 300 balanced calibration points, sklearn fit has "
      f"{len(np.unique(pav['sk_fit']))} distinct levels")

out = {
    "n_fit": int(len(yfit)),
    "n_calib": int(len(ycal)),
    "n_test": int(len(yte)),
    "n_calib_pos": int(ycal.sum()),
    "base_rate": rnd(base_rate, 4),
    "models": models,
    "size_curve": size_curve,
    "pav": pav,
    "versions": {"seed": SEED},
}
(OUT / "repair.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'repair.json'}  ({(OUT / 'repair.json').stat().st_size / 1024:.1f} kB)")
