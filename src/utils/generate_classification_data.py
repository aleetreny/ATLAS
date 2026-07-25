"""Data for the ATLAS 'Probabilistic Classification' web article.

The target stops being a number and becomes a decision, which changes what a
model owes you. Not just the label: the probability, and a probability is a
promise that can be checked.

Everything the article claims is measured here on the Breast Cancer Wisconsin
diagnostic set, 569 biopsies with 30 features. It was chosen for one specific
reason: three of its features (radius, perimeter, area) are the same quantity
measured three ways and correlate at r > 0.98. Naive Bayes assumes they are
independent, so it counts that single piece of evidence three times. That is not
a hypothetical failure mode, it is arithmetic, and its effect on the probability
the model reports is enormous while its effect on accuracy is almost nil.

Run from anywhere: `python src/utils/generate_classification_data.py`.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "probabilistic-classification" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

SEED = 7

# ----------------------------------------------------------------- the data
raw = load_breast_cancer()
X_all = raw.data
# sklearn ships 0 = malignant, 1 = benign. Flip it: the positive class should be
# the thing you are trying to detect, or every metric below reads backwards.
y_all = 1 - raw.target
FEATURES = [f.replace(" ", "_") for f in raw.feature_names]

Xtr, Xte, ytr, yte = train_test_split(
    X_all, y_all, test_size=0.3, random_state=SEED, stratify=y_all
)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

print(f"{len(y_all)} biopsies, {X_all.shape[1]} features")
print(f"  malignant {int(y_all.sum())}  benign {int((1 - y_all).sum())}")
print(f"  train {len(ytr)}  test {len(yte)}")


# ------------------------------------------------- which single feature wins
# The article opens in one dimension, so it needs the feature that carries the
# most signal on its own rather than one picked because it sounds good.
singles = []
for j, name in enumerate(FEATURES):
    auc = roc_auc_score(ytr, Xtr[:, j])
    singles.append((max(auc, 1 - auc), j, name))
singles.sort(reverse=True)
BEST_J = singles[0][1]
BEST_NAME = singles[0][2]
print(f"\nstrongest single feature: {BEST_NAME} (train auc {singles[0][0]:.3f})")
for a, _, n in singles[:5]:
    print(f"  {n:<26} {a:.3f}")


def metrics(y, p):
    """Everything a probability owes you, in one place.

    accuracy grades the label, auc grades the ranking, and brier and log loss
    grade the number itself. A model can be excellent at the first two and
    useless at the third, which is the entire point of this article.
    """
    return {
        "acc": rnd(((p >= 0.5).astype(int) == y).mean()),
        "auc": rnd(roc_auc_score(y, p)),
        "brier": rnd(brier_score_loss(y, p), 5),
        "logloss": rnd(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)), 4),
    }


# ------------------------------------------------------ 1-D: OLS vs logistic
x1 = Xtr[:, BEST_J]
x1te = Xte[:, BEST_J]
lin1 = LinearRegression().fit(x1.reshape(-1, 1), ytr)
log1 = LogisticRegression(C=1e6, max_iter=10000).fit(x1.reshape(-1, 1), ytr)

grid1 = np.linspace(x1.min() * 0.95, x1.max() * 1.02, 120)
ols_curve = lin1.predict(grid1.reshape(-1, 1))
log_curve = log1.predict_proba(grid1.reshape(-1, 1))[:, 1]

# How often does the straight line promise something impossible?
ols_train = lin1.predict(x1.reshape(-1, 1))
impossible = int(((ols_train < 0) | (ols_train > 1)).sum())
print(f"\nOLS on the 0/1 target: {impossible} of {len(ytr)} training fits fall outside [0,1]")
print(f"  its range is {ols_train.min():.3f} to {ols_train.max():.3f}")

one_d = {
    "feature": BEST_NAME,
    "points": [{"x": rnd(a, 4), "y": int(b)} for a, b in zip(x1, ytr)],
    "grid": [rnd(v, 4) for v in grid1],
    "ols": [rnd(v, 4) for v in ols_curve],
    "logistic": [rnd(v, 4) for v in log_curve],
    "ols_coef": {"intercept": rnd(lin1.intercept_, 5), "slope": rnd(lin1.coef_[0], 5)},
    "log_coef": {"intercept": rnd(log1.intercept_[0], 5), "slope": rnd(log1.coef_[0][0], 5)},
    "out_of_range": impossible,
    "ols_min": rnd(ols_train.min()),
    "ols_max": rnd(ols_train.max()),
    "ols_test": metrics(yte, np.clip(lin1.predict(x1te.reshape(-1, 1)), 0, 1)),
    "log_test": metrics(yte, log1.predict_proba(x1te.reshape(-1, 1))[:, 1]),
    # The scrolly panel shows the TRAINING points, so the identical-ranking
    # claim it makes has to be about the training set, not the held-out one.
    "auc_train_ols": rnd(roc_auc_score(ytr, ols_train)),
    "auc_train_log": rnd(roc_auc_score(ytr, log1.decision_function(x1.reshape(-1, 1)))),
}
print(f"  1-D training AUC: ols {one_d['auc_train_ols']}  logistic {one_d['auc_train_log']}")


# ------------------------------------------------- the three redundant twins
# radius, perimeter and area of a circle are one number wearing three hats.
TRIO = ["mean_radius", "mean_perimeter", "mean_area"]
trio_idx = [FEATURES.index(f) for f in TRIO]
corr = np.corrcoef(Xtr[:, trio_idx], rowvar=False)
print("\nthe three twins, correlation on the training set:")
for a in range(3):
    for b in range(a + 1, 3):
        print(f"  {TRIO[a]:<16} x {TRIO[b]:<16} r = {corr[a][b]:.4f}")


# ---------------------------------------------------- the headline benchmark
QDA_REG = 0.05


def fit_all(cols, label):
    """Logistic, naive Bayes and QDA on the same columns.

    QDA is the control: Gaussian naive Bayes is QDA with the off-diagonal terms
    of each covariance forced to zero, so the gap between the two is the price
    of the naive assumption.

    It is not a perfectly clean control, and the article says so. With 30
    features and 398 rows the unregularised QDA cannot be fitted at all: the
    per-class covariance is singular and sklearn raises LinAlgError. It needs
    reg_param, which shrinks each covariance toward a scaled identity, so the
    comparison is against a slightly-shrunk full covariance rather than a raw
    one. The conclusion is not sensitive to the amount: measured on this split,
    Brier comes out 0.043 at reg 0.01, 0.034 at 0.05 and 0.033 at 0.1, against
    naive Bayes at 0.062 for all of them.
    """
    A, B = Xtr_s[:, cols], Xte_s[:, cols]
    out = {}
    lr = LogisticRegression(max_iter=10000).fit(A, ytr)
    out["logistic"] = metrics(yte, lr.predict_proba(B)[:, 1])
    nb = GaussianNB().fit(A, ytr)
    out["naive_bayes"] = metrics(yte, nb.predict_proba(B)[:, 1])
    try:
        qda = QuadraticDiscriminantAnalysis(reg_param=QDA_REG).fit(A, ytr)
        out["qda"] = metrics(yte, qda.predict_proba(B)[:, 1])
    except Exception as exc:  # singular covariance on tiny column sets
        print(f"  qda skipped on {label}: {exc}")
    return out


ALL = list(range(len(FEATURES)))
bench_all = fit_all(ALL, "all 30")
print("\n=== all 30 features ===")
for k, v in bench_all.items():
    print(f"  {k:<14} acc {v['acc']:.4f}  auc {v['auc']:.4f}  brier {v['brier']:.5f}  logloss {v['logloss']:.4f}")

# Now the controlled experiment: keep one of each redundant trio, drop the rest.
# Any feature correlated above the cut with a feature already kept is dropped.
def prune(threshold):
    C = np.abs(np.corrcoef(Xtr, rowvar=False))
    keep = []
    for j in range(len(FEATURES)):
        if all(C[j][k] < threshold for k in keep):
            keep.append(j)
    return keep


pruned = {}
for thr in [0.99, 0.95, 0.9, 0.8, 0.7]:
    cols = prune(thr)
    res = fit_all(cols, f"r<{thr}")
    pruned[str(thr)] = {"n_features": len(cols), **res}
    nb = res["naive_bayes"]
    lr = res["logistic"]
    print(
        f"  keep r < {thr}: {len(cols):2d} features | "
        f"nb acc {nb['acc']:.4f} brier {nb['brier']:.5f} | lr acc {lr['acc']:.4f} brier {lr['brier']:.5f}"
    )


# --------------------------------------------------------- reliability curve
def reliability(y, p, bins=10):
    """Bin by predicted probability and report what actually happened.

    A perfectly calibrated model puts the dot on the diagonal in every bin.
    Empty bins are dropped rather than drawn at zero, which would invent a
    failure the model never had a chance to commit.
    """
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out.append(
            {
                "lo": rnd(lo, 2),
                "hi": rnd(hi, 2),
                "n": int(m.sum()),
                # Six decimals, not four: naive Bayes' bottom bucket averages
                # well under 0.0001 and the article quotes that number.
                "predicted": rnd(p[m].mean(), 6),
                "observed": rnd(y[m].mean(), 4),
            }
        )
    return out


lr_full = LogisticRegression(max_iter=10000).fit(Xtr_s, ytr)
nb_full = GaussianNB().fit(Xtr_s, ytr)
p_lr = lr_full.predict_proba(Xte_s)[:, 1]
p_nb = nb_full.predict_proba(Xte_s)[:, 1]

print("\n=== how extreme are the probabilities? ===")
for name, p in [("logistic", p_lr), ("naive bayes", p_nb)]:
    frac = float(((p < 0.01) | (p > 0.99)).mean())
    print(f"  {name:<12} {frac * 100:5.1f}% of test predictions are past 99% certainty")

calib = {
    "logistic": reliability(yte, p_lr),
    "naive_bayes": reliability(yte, p_nb),
    "extreme_logistic": rnd(float(((p_lr < 0.01) | (p_lr > 0.99)).mean())),
    "extreme_nb": rnd(float(((p_nb < 0.01) | (p_nb > 0.99)).mean())),
}


# -------------------------------------------------------- the threshold, and
# what it costs. 0.5 is a default, not a law, and on this problem the two kinds
# of mistake are not remotely the same kind of mistake.
sweep = []
for t in np.linspace(0.01, 0.99, 99):
    pred = (p_lr >= t).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    tn = int(((pred == 0) & (yte == 0)).sum())
    sweep.append(
        {
            "t": rnd(t, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": rnd(tp / max(tp + fn, 1)),
            "precision": rnd(tp / max(tp + fp, 1)),
        }
    )

at_half = next(s for s in sweep if s["t"] == 0.5)
print(f"\nat threshold 0.50: {at_half['fn']} missed cancers, {at_half['fp']} false alarms")
# lowest threshold that misses nobody
perfect = [s for s in sweep if s["fn"] == 0]
if perfect:
    best = max(perfect, key=lambda s: s["t"])
    print(f"at threshold {best['t']:.2f}: 0 missed cancers, {best['fp']} false alarms")


# --------------------------------------------- per-feature class conditionals
# What Gaussian naive Bayes actually stores: two bells per feature.
shapes = []
for j in trio_idx + [BEST_J, FEATURES.index("mean_texture"), FEATURES.index("mean_smoothness")]:
    if any(s["j"] == j for s in shapes):
        continue
    col = Xtr[:, j]
    lo, hi = np.percentile(col, 1), np.percentile(col, 99)
    g = np.linspace(lo, hi, 80)
    entry = {"j": int(j), "feature": FEATURES[j], "grid": [rnd(v, 4) for v in g], "curves": {}}
    for cls, tag in [(0, "benign"), (1, "malignant")]:
        c = col[ytr == cls]
        mu, sd = c.mean(), c.std()
        entry["curves"][tag] = {
            "mu": rnd(mu, 4),
            "sd": rnd(sd, 4),
            "pdf": [rnd(np.exp(-0.5 * ((v - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), 6) for v in g],
        }
    shapes.append(entry)


# ------------------------------------------------------------- 2-D boundaries
# Two features chosen to be readable rather than optimal: the strongest single
# predictor, plus texture, which is close to independent of it.
J2 = [BEST_J, FEATURES.index("mean_texture")]
print(f"\n2-D panel: {FEATURES[J2[0]]} and {FEATURES[J2[1]]}, r = {np.corrcoef(Xtr[:, J2], rowvar=False)[0][1]:.3f}")
sc2 = StandardScaler().fit(Xtr[:, J2])
A2, B2 = sc2.transform(Xtr[:, J2]), sc2.transform(Xte[:, J2])
lr2 = LogisticRegression(max_iter=10000).fit(A2, ytr)
nb2 = GaussianNB().fit(A2, ytr)

gx = np.linspace(A2[:, 0].min() - 0.4, A2[:, 0].max() + 0.4, 60)
gy = np.linspace(A2[:, 1].min() - 0.4, A2[:, 1].max() + 0.4, 60)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])

two_d = {
    "features": [FEATURES[J2[0]], FEATURES[J2[1]]],
    "mu": [rnd(v, 5) for v in sc2.mean_],
    "sd": [rnd(v, 5) for v in sc2.scale_],
    "train": [{"x": rnd(a, 3), "y": rnd(b, 3), "c": int(c)} for (a, b), c in zip(A2, ytr)],
    "test": [{"x": rnd(a, 3), "y": rnd(b, 3), "c": int(c)} for (a, b), c in zip(B2, yte)],
    "gx": [rnd(v, 3) for v in gx],
    "gy": [rnd(v, 3) for v in gy],
    "logistic": [rnd(v, 3) for v in lr2.predict_proba(G)[:, 1]],
    "naive_bayes": [rnd(v, 3) for v in nb2.predict_proba(G)[:, 1]],
    "logistic_test": metrics(yte, lr2.predict_proba(B2)[:, 1]),
    "nb_test": metrics(yte, nb2.predict_proba(B2)[:, 1]),
}


# --------------------------------------------------- coefficients as odds
coef = lr_full.coef_[0]
order = np.argsort(-np.abs(coef))[:8]
odds = [
    {
        "feature": FEATURES[j],
        "beta": rnd(coef[j]),
        "odds_ratio": rnd(float(np.exp(coef[j])), 3),
    }
    for j in order
]
print("\n=== strongest coefficients, per standard deviation ===")
for o in odds[:5]:
    print(f"  {o['feature']:<26} beta {o['beta']:+.3f}  odds x{o['odds_ratio']:.2f}")


# ----------------------------------------- cross-validated, so the headline is
# not one lucky split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_res = {"logistic": [], "naive_bayes": [], "qda": []}
for tr, te in cv.split(X_all, y_all):
    s = StandardScaler().fit(X_all[tr])
    A, B = s.transform(X_all[tr]), s.transform(X_all[te])
    yA, yB = y_all[tr], y_all[te]
    for name, model in [
        ("logistic", LogisticRegression(max_iter=10000)),
        ("naive_bayes", GaussianNB()),
        ("qda", QuadraticDiscriminantAnalysis(reg_param=QDA_REG)),
    ]:
        p = model.fit(A, yA).predict_proba(B)[:, 1]
        cv_res[name].append(metrics(yB, p))

cv_summary = {}
print("\n=== five-fold cross validation, mean +- sd ===")
for name, rows in cv_res.items():
    cv_summary[name] = {
        k: {"mean": rnd(np.mean([r[k] for r in rows]), 5), "sd": rnd(np.std([r[k] for r in rows]), 5)}
        for k in ["acc", "auc", "brier", "logloss"]
    }
    s = cv_summary[name]
    print(
        f"  {name:<12} acc {s['acc']['mean']:.4f}+-{s['acc']['sd']:.4f}  "
        f"auc {s['auc']['mean']:.4f}  brier {s['brier']['mean']:.5f}+-{s['brier']['sd']:.5f}"
    )


# ---------------------------------------------------------------- write it out
out = {
    "n": int(len(y_all)),
    "n_train": int(len(ytr)),
    "n_test": int(len(yte)),
    "n_features": int(X_all.shape[1]),
    "n_malignant": int(y_all.sum()),
    "features": FEATURES,
    "one_d": one_d,
    "two_d": two_d,
    "trio": {"names": TRIO, "corr": [[rnd(v, 4) for v in row] for row in corr]},
    "benchmark": bench_all,
    "pruned": pruned,
    "calibration": calib,
    "threshold_sweep": sweep,
    "shapes": shapes,
    "odds": odds,
    "cv": cv_summary,
    "versions": {"seed": SEED},
}
(OUT / "cancer.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
size = (OUT / "cancer.json").stat().st_size
print(f"\nwrote {OUT / 'cancer.json'}  ({size / 1024:.1f} kB)")
