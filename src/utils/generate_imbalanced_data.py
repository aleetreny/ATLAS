"""Data for the ATLAS 'Imbalanced Learning' web article.

The rare thing is the whole subject, so the dataset has to have a genuinely
rare thing in it. Forest Covertypes, 581,012 mapped cells of Roosevelt National
Forest, where cover type 4 (cottonwood/willow) is 2,747 cells: 0.47% of the
forest. All 2,747 positives are kept and the majority is thinned to make the
prevalence a round 1%, which changes nothing about the phenomenon and keeps
every model fit under a minute.

What gets measured, because each is a claim the article makes out loud:
  * the accuracy trap: the majority-class constant scores 99% and catches nothing
  * ROC flatters imbalance, precision-recall does not, with both AUCs
  * six interventions on the same logistic model: nothing, class weights,
    undersampling, oversampling, SMOTE, ADASYN
  * the punchline nobody advertises: every resampling method destroys the
    predicted probabilities, reporting a base rate of ~50% in a 1% world, while
    barely moving the RANKING at all. Class weights do the same. The fix is the
    next article.

The scratch samplers come from src/models/imbalanced_learning.py, the repo's
own implementations, and SMOTE is cross-checked against imbalanced-learn here.

Run from anywhere: `python src/utils/generate_imbalanced_data.py`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.imbalanced_learning import (  # noqa: E402
    ScratchRandomOversampler,
    ScratchRandomUndersampler,
    ScratchSMOTE,
)

OUT = ROOT / "imbalanced-learning" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 17

# ----------------------------------------------------------------- the forest
cov = fetch_covtype()
X_all = cov.data
y_all = (cov.target == 4).astype(int)  # cottonwood/willow, the rarest type
FEATURES = [f.replace(" ", "_") for f in cov.feature_names]

n_pos = int(y_all.sum())
print(f"{len(y_all):,} cells, {n_pos:,} cottonwood/willow ({n_pos / len(y_all) * 100:.2f}%)")

rs = np.random.RandomState(SEED)
neg_idx = np.where(y_all == 0)[0]
pos_idx = np.where(y_all == 1)[0]
keep_neg = rs.choice(neg_idx, n_pos * 99, replace=False)  # exactly 1% prevalence
idx = np.concatenate([pos_idx, keep_neg])
rs.shuffle(idx)
X, y = X_all[idx], y_all[idx]
print(f"thinned to {len(y):,} cells at {y.mean() * 100:.2f}% prevalence")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(Xtr)
A, B = scaler.transform(Xtr), scaler.transform(Xte)
print(f"train {len(ytr):,} ({int(ytr.sum())} positive), test {len(yte):,} ({int(yte.sum())} positive)")


def confusion(y_true, pred):
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp, fp, fn, tn = confusion(y_true, pred)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "acc": rnd((tp + tn) / len(y_true)),
        "precision": rnd(prec),
        "recall": rnd(rec),
        "f1": rnd(2 * prec * rec / max(prec + rec, 1e-12)),
        "auroc": rnd(roc_auc_score(y_true, p)),
        "auprc": rnd(average_precision_score(y_true, p)),
        "brier": rnd(brier_score_loss(y_true, p), 5),
        "mean_p": rnd(float(p.mean())),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --------------------------------------------------------- the accuracy trap
base_rate = float(yte.mean())
trap = {
    "n_test": int(len(yte)),
    "n_pos": int(yte.sum()),
    "base_rate": rnd(base_rate, 4),
    "constant_acc": rnd(1 - base_rate),
}
print(f"\nthe constant 'never' answer scores {trap['constant_acc']:.4f} accuracy and finds 0 of {trap['n_pos']}")

# ------------------------------------------------------------- the baseline
t0 = time.perf_counter()
lr = LogisticRegression(max_iter=3000).fit(A, ytr)
print(f"baseline logistic fitted in {time.perf_counter() - t0:.1f}s")
p_base = lr.predict_proba(B)[:, 1]
m_base = metrics(yte, p_base)
print(f"  baseline @0.5: acc {m_base['acc']:.4f}  recall {m_base['recall']:.3f}  precision {m_base['precision']:.3f}")

# The threshold sweep for the confusion-matrix explorer.
sweep = []
for t in np.concatenate([np.linspace(0.005, 0.1, 20), np.linspace(0.12, 0.95, 43)]):
    m = metrics(yte, p_base, t)
    sweep.append({"t": rnd(t, 3), **{k: m[k] for k in ["acc", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]}})
best_f1 = max(sweep, key=lambda s: s["f1"])
print(f"  best F1 on the sweep: {best_f1['f1']:.3f} at threshold {best_f1['t']}")

# ------------------------------------------------- ROC vs precision-recall
fpr, tpr, _ = roc_curve(yte, p_base)
prec_c, rec_c, _ = precision_recall_curve(yte, p_base)


def thin(xs, ys, n=140):
    """Thin a curve to n points, keeping both ends."""
    if len(xs) <= n:
        return [[rnd(a, 4), rnd(b, 4)] for a, b in zip(xs, ys)]
    ii = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
    return [[rnd(xs[i], 4), rnd(ys[i], 4)] for i in ii]


curves = {
    "roc": thin(fpr, tpr),
    "pr": thin(rec_c, prec_c),
    "auroc": rnd(roc_auc_score(yte, p_base)),
    "auprc": rnd(average_precision_score(yte, p_base)),
    "base_rate": rnd(base_rate, 4),
}
print(f"  AUROC {curves['auroc']:.4f} vs AUPRC {curves['auprc']:.4f} on the same predictions")

# ------------------------------------------------------- the six interventions
def fit_on(Xr, yr, name, weight=None):
    t0 = time.perf_counter()
    m = LogisticRegression(max_iter=3000, class_weight=weight).fit(Xr, yr)
    p = m.predict_proba(B)[:, 1]
    secs = time.perf_counter() - t0
    out = metrics(yte, p)
    out["name"] = name
    out["n_train"] = int(len(yr))
    out["train_prevalence"] = rnd(float(np.mean(yr)), 4)
    out["seconds"] = rnd(secs, 2)
    print(f"  {name:<22} n={len(yr):>7,} prev {out['train_prevalence']:.3f}  "
          f"recall {out['recall']:.3f}  precision {out['precision']:.3f}  "
          f"auprc {out['auprc']:.4f}  brier {out['brier']:.5f}  mean p {out['mean_p']:.3f}")
    return out, p


print("\n=== six interventions, same logistic model, same test set ===")
bench = []
probs = {}
r, p = fit_on(A, ytr, "baseline")
bench.append(r); probs["baseline"] = p
r, p = fit_on(A, ytr, "class weights", weight="balanced")
bench.append(r); probs["class weights"] = p

Xu, yu = ScratchRandomUndersampler(random_state=SEED).fit_resample(A, ytr)
r, p = fit_on(Xu, yu, "undersampling")
bench.append(r); probs["undersampling"] = p

Xo, yo = ScratchRandomOversampler(random_state=SEED).fit_resample(A, ytr)
r, p = fit_on(Xo, yo, "oversampling")
bench.append(r); probs["oversampling"] = p

Xs, ys = ScratchSMOTE(k_neighbors=5, random_state=SEED).fit_resample(A, ytr)
r, p = fit_on(Xs, ys, "SMOTE")
bench.append(r); probs["SMOTE"] = p

from imblearn.over_sampling import ADASYN, SMOTE  # noqa: E402

Xa, ya = ADASYN(random_state=SEED).fit_resample(A, ytr)
r, p = fit_on(Xa, ya, "ADASYN")
bench.append(r); probs["ADASYN"] = p

# Cross-check the repo's scratch SMOTE against the library's.
Xs2, ys2 = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(A, ytr)
lr_lib = LogisticRegression(max_iter=3000).fit(Xs2, ys2)
p_lib = lr_lib.predict_proba(B)[:, 1]
gap = abs(average_precision_score(yte, p_lib) - next(b for b in bench if b["name"] == "SMOTE")["auprc"])
print(f"  scratch SMOTE vs imblearn SMOTE: AUPRC differs by {gap:.4f} (different draws, same recipe)")
smote_gap = rnd(gap)  # shipped, so the article's quoted cross-check is anchored in the data file

# --------------------------------------------------- reliability, per method
def reliability(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() < 10:
            continue
        out.append({"lo": rnd(lo, 2), "hi": rnd(hi, 2), "n": int(m.sum()),
                    "predicted": rnd(float(p[m].mean()), 4), "observed": rnd(float(y_true[m].mean()), 4)})
    return out


calib = {k.replace(" ", "_"): reliability(yte, p) for k, p in probs.items()}

# ----------------------------------------------- 2-D panel for SMOTE geometry
# Elevation against distance to water: cottonwood/willow is a riparian tree, so
# the rare class hugs low elevations near hydrology and the picture is honest.
J2 = [FEATURES.index("Elevation"), FEATURES.index("Horizontal_Distance_To_Hydrology")]
rs2 = np.random.RandomState(SEED + 1)
pos2 = np.where(ytr == 1)[0]
neg2 = rs2.choice(np.where(ytr == 0)[0], 4000, replace=False)
pos_keep = rs2.choice(pos2, 700, replace=False)
two_d = {
    "features": ["elevation (m)", "distance to water (m)"],
    "pos": [[rnd(Xtr[i, J2[0]], 1), rnd(Xtr[i, J2[1]], 1)] for i in pos_keep],
    "neg": [[rnd(Xtr[i, J2[0]], 1), rnd(Xtr[i, J2[1]], 1)] for i in neg2],
}
print(f"\n2-D panel: {len(two_d['pos'])} positives drawn of {len(pos2)}, {len(two_d['neg'])} negatives of {len(ytr) - len(pos2)}")

out = {
    "n_forest": int(len(y_all)),
    "n_pos_forest": n_pos,
    "true_prevalence": rnd(n_pos / len(y_all), 5),
    "n": int(len(y)),
    "n_train": int(len(ytr)),
    "n_test": int(len(yte)),
    "trap": trap,
    "baseline": m_base,
    "sweep": sweep,
    "curves": curves,
    "benchmark": bench,
    "smote_gap_auprc": smote_gap,
    "calibration": calib,
    "two_d": two_d,
    "versions": {"seed": SEED},
}
(OUT / "forest.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'forest.json'}  ({(OUT / 'forest.json').stat().st_size / 1024:.1f} kB)")
