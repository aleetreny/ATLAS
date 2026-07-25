"""Data for the ATLAS 'Geometric Classification' web article.

Three datasets, each doing one job.

  * Two synthetic 2-D sets, because the margin and the kernel trick are claims
    about geometry and the only honest way to argue about geometry is to draw
    it. Both are seeded, so what the article says is on screen is on screen.
  * The 178 wines from the first article, which is where discriminant analysis
    earns its keep: three classes means LDA gets two discriminant axes, and
    putting them next to the PCA and PLS axes the reader already met makes the
    difference between the three a picture rather than a definition.
  * The 569 breast biopsies from the previous article, for the part everybody
    skips: measuring whether the kernel was worth it on real tabular data
    against a linear model that takes a millisecond.

Run from anywhere: `python src/utils/generate_geometric_data.py`.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_breast_cancer, load_wine, make_circles
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "geometric-classification" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 11


# ------------------------------------------------------------ 2-D: the margin
# Two clouds with a gap you can see, so "the widest street" is a claim the
# reader can check by eye before any arithmetic shows up.
rs = np.random.RandomState(SEED)
n_side = 40
A = rs.multivariate_normal([-1.5, -0.7], [[0.55, 0.18], [0.18, 0.42]], n_side)
B = rs.multivariate_normal([1.6, 1.0], [[0.5, -0.12], [-0.12, 0.5]], n_side)

# The article tells this in two acts, so the data comes in two versions. Act one
# is a genuinely separable problem, which is the only setting in which "every
# one of these lines is perfect on the training set" is a true sentence. Act two
# adds ONE point across the gap, and the whole soft-margin argument is about
# what that single observation is allowed to do to the boundary.
# Placed by search rather than by eye: this is the position that keeps the
# problem separable, so a hard margin still exists and the demonstration is not
# a cheat, while squeezing its street from 0.383 down to 0.002. A boundary
# threaded through a gap that tight is a boundary held hostage by one point.
ROGUE = [0.4, 0.6]
Xclean = np.vstack([A, B])
yclean = np.array([0] * len(A) + [1] * len(B))
Xm = np.vstack([A, [ROGUE], B])
ym = np.array([0] * (len(A) + 1) + [1] * len(B))
ROGUE_IDX = len(A)

print(f"margin panel: {len(Xclean)} separable points, {len(Xm)} once the rogue is added")

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]


def sweep(X, y, tag):
    runs = []
    for C in C_GRID:
        svc = SVC(kernel="linear", C=C).fit(X, y)
        wv = svc.coef_[0]
        b = float(svc.intercept_[0])
        half = 1.0 / float(np.linalg.norm(wv))
        runs.append(
            {
                "C": C,
                "w": [rnd(v, 5) for v in wv],
                "b": rnd(b, 5),
                "half_width": rnd(half, 4),
                "n_support": int(len(svc.support_)),
                "support": [int(i) for i in svc.support_],
                "train_acc": rnd(svc.score(X, y)),
            }
        )
        print(f"  [{tag}] C={C:<7} half-width {half:.3f}  support vectors {len(svc.support_):3d}"
              f"  train acc {svc.score(X, y):.3f}")
    return runs


clean_runs = sweep(Xclean, yclean, "separable")
margin_runs = sweep(Xm, ym, "with rogue")


def hard(X, y):
    """The true hard margin, at a C large enough that no violation is affordable.

    C = 1000 is not that C: with the rogue present the solver still prefers to
    misclassify it, so the sweep never actually shows the collapse the article
    is about. Pushing C to a million forces the separable answer, which exists.
    How much narrower it is gets printed below rather than written here, so the
    number cannot go stale if SEED or the rogue's position ever changes."""
    svc = SVC(kernel="linear", C=1e6).fit(X, y)
    return {
        "w": [rnd(v, 5) for v in svc.coef_[0]],
        "b": rnd(float(svc.intercept_[0]), 5),
        "half_width": rnd(1.0 / float(np.linalg.norm(svc.coef_[0])), 4),
        "n_support": int(len(svc.support_)),
        "support": [int(i) for i in svc.support_],
        "train_acc": rnd(svc.score(X, y)),
    }


hard_clean = hard(Xclean, yclean)
hard_rogue = hard(Xm, ym)
print(f"  [hard, separable]  half-width {hard_clean['half_width']}  sv {hard_clean['n_support']}  "
      f"acc {hard_clean['train_acc']}")
print(f"  [hard, with rogue] half-width {hard_rogue['half_width']}  sv {hard_rogue['n_support']}  "
      f"acc {hard_rogue['train_acc']}  "
      f"({hard_clean['half_width'] / max(hard_rogue['half_width'], 1e-9):.0f}x narrower)")

# The five alternative separators the first scrolly step draws. Each is the
# maximum-margin normal rotated a little and re-centred between the two class
# means. Only rotations that STILL separate the clean data perfectly survive,
# because the step's whole claim is that the training error cannot choose
# between them, and a line that gets one wrong would make that claim false.
base_w = np.array(clean_runs[-1]["w"], dtype=float)
mid = (Xclean[yclean == 0].mean(0) + Xclean[yclean == 1].mean(0)) / 2
alts = []
for ang in np.linspace(-0.75, 0.75, 41):
    c, s = np.cos(ang), np.sin(ang)
    wv = np.array([base_w[0] * c - base_w[1] * s, base_w[0] * s + base_w[1] * c])
    b = -float(wv @ mid)
    wrong = int((((Xclean @ wv + b) > 0).astype(int) != yclean).sum())
    if wrong == 0:
        alts.append({"angle": rnd(ang, 4), "w": [rnd(v, 5) for v in wv], "b": rnd(b, 5)})
if len(alts) > 5:
    pick = np.linspace(0, len(alts) - 1, 5).round().astype(int)
    alts = [alts[i] for i in pick]
print(f"  {len(alts)} alternative separators, every one of them perfect on the separable set")

margin_panel = {
    "points": [{"x": rnd(a, 4), "y": rnd(b, 4), "c": int(c)} for (a, b), c in zip(Xm, ym)],
    "rogue_index": ROGUE_IDX,
    "runs": margin_runs,
    "clean_runs": clean_runs,
    "hard_clean": hard_clean,
    "hard_rogue": hard_rogue,
    "alternatives": alts,
}


# ------------------------------------------------------- 2-D: the kernel trick
# Concentric rings: no straight line separates them, and the lift to a third
# dimension that does is one squared radius.
Xc, yc = make_circles(n_samples=220, factor=0.42, noise=0.09, random_state=SEED)
Xc = Xc * 2.2

lin_c = SVC(kernel="linear", C=1).fit(Xc, yc)
rbf_c = SVC(kernel="rbf", C=1, gamma="scale").fit(Xc, yc)
print(f"\nrings: linear svm accuracy {lin_c.score(Xc, yc):.3f}, rbf {rbf_c.score(Xc, yc):.3f}")

gx = np.linspace(Xc[:, 0].min() - 0.5, Xc[:, 0].max() + 0.5, 70)
gy = np.linspace(Xc[:, 1].min() - 0.5, Xc[:, 1].max() + 0.5, 70)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])

# The two whole-data decision surfaces are deliberately NOT exported: the only
# panel that draws a field is the gamma one, and it draws the three tuned
# surfaces below. Shipping 9,800 more numbers the browser never reads would be
# a third of this file for nothing.
rings = {
    "points": [{"x": rnd(a, 4), "y": rnd(b, 4), "c": int(c), "r2": rnd(a * a + b * b, 4)} for (a, b), c in zip(Xc, yc)],
    "gx": [rnd(v, 4) for v in gx],
    "gy": [rnd(v, 4) for v in gy],
    "linear_acc": rnd(lin_c.score(Xc, yc)),
    "rbf_acc": rnd(rbf_c.score(Xc, yc)),
}

# The gamma / C grid on the same rings, held out honestly.
# Split by index, so the browser can tell the two halves apart on the chart.
# The gamma panel is an argument about held-out accuracy and it cannot make it
# while drawing all 220 points in the same colour.
idx_tr, idx_te = train_test_split(
    np.arange(len(yc)), test_size=0.4, random_state=SEED, stratify=yc
)
Xc_tr, Xc_te = Xc[idx_tr], Xc[idx_te]
yc_tr, yc_te = yc[idx_tr], yc[idx_te]
G_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
C_GRID2 = [0.1, 1, 10, 100]
surface = []
for gval in G_GRID:
    row = []
    for C in C_GRID2:
        m = SVC(kernel="rbf", C=C, gamma=gval).fit(Xc_tr, yc_tr)
        row.append({"train": rnd(m.score(Xc_tr, yc_tr)), "test": rnd(m.score(Xc_te, yc_te)), "sv": int(len(m.support_))})
    surface.append({"gamma": gval, "cells": row})
print("\ngamma sweep on the rings (C = 1):")
for r in surface:
    c1 = r["cells"][1]
    print(f"  gamma {r['gamma']:<6} train {c1['train']:.3f}  test {c1['test']:.3f}  support vectors {c1['sv']}")

# Decision surfaces at three chosen gammas, so the article can show under-,
# well- and over-fitted boundaries side by side rather than describing them.
gsurf = []
for gval in [0.03, 1, 100]:
    m = SVC(kernel="rbf", C=1, gamma=gval).fit(Xc_tr, yc_tr)
    gsurf.append(
        {
            "gamma": gval,
            "field": [rnd(v, 3) for v in m.decision_function(G)],
            "train": rnd(m.score(Xc_tr, yc_tr)),
            "test": rnd(m.score(Xc_te, yc_te)),
            "sv": int(len(m.support_)),
        }
    )

rings["train_idx"] = sorted(int(i) for i in idx_tr)
rings["test_idx"] = sorted(int(i) for i in idx_te)
rings["gamma_surfaces"] = gsurf
rings["gamma_grid"] = surface
rings["gamma_values"] = G_GRID
rings["C_values"] = C_GRID2


# ------------------------------------------------------- wine: three axes
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
yw = wine.target

pca = PCA(n_components=2).fit(Xw)
Zpca = pca.transform(Xw)
pls = PLSRegression(n_components=2).fit(Xw, yw)
Zpls = pls.transform(Xw)
lda = LinearDiscriminantAnalysis(n_components=2).fit(Xw, yw)
Zlda = lda.transform(Xw)


def separation(Z, y):
    """Fisher's criterion in the projected plane: between-class scatter over
    within-class scatter. The number LDA is explicitly maximising, so quoting it
    is not a rigged comparison, it is naming the objective and then reporting
    who won on it."""
    overall = Z.mean(0)
    sb = 0.0
    sw = 0.0
    for c in np.unique(y):
        Zc = Z[y == c]
        sb += len(Zc) * float(((Zc.mean(0) - overall) ** 2).sum())
        sw += float(((Zc - Zc.mean(0)) ** 2).sum())
    return sb / sw


axes_out = {}
for name, Z in [("pca", Zpca), ("pls", Zpls), ("lda", Zlda)]:
    # Put every projection on a common scale so the panels are comparable and
    # the winner is not just the one with the largest units.
    Zs = (Z - Z.mean(0)) / Z.std(0)
    axes_out[name] = {
        "points": [{"x": rnd(a, 3), "y": rnd(b, 3), "c": int(c)} for (a, b), c in zip(Zs, yw)],
        "separation": rnd(separation(Zs, yw), 3),
        "cv_acc": rnd(cross_val_score(LogisticRegression(max_iter=5000), Zs, yw, cv=5).mean()),
    }
    print(f"\n{name}: fisher separation {axes_out[name]['separation']:.3f}  "
          f"logistic on the 2 axes, 5-fold {axes_out[name]['cv_acc']:.4f}")


# -------------------------------------------- LDA vs QDA as the data runs out
# QDA estimates a covariance per class, which costs d(d+1)/2 numbers each. The
# crossover is the whole practical difference between the two, so measure it.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
curve = []
# Capped at 40: the smallest wine class has 48 members, and a row that could
# not draw a sample would be a hole in the chart rather than a measurement.
for n_per in [4, 6, 8, 10, 14, 20, 30, 40]:
    accs = {"lda": [], "qda": [], "gnb": []}
    for rep in range(24):
        r = np.random.RandomState(1000 + rep)
        idx = []
        ok = True
        for c in np.unique(yw):
            pool = np.where(yw == c)[0]
            if len(pool) < n_per + 5:
                ok = False
                break
            idx.extend(r.choice(pool, n_per, replace=False))
        if not ok:
            continue
        idx = np.array(idx)
        rest = np.setdiff1d(np.arange(len(yw)), idx)
        for key, model in [
            ("lda", LinearDiscriminantAnalysis()),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0.02)),
            ("gnb", GaussianNB()),
        ]:
            try:
                accs[key].append(model.fit(Xw[idx], yw[idx]).score(Xw[rest], yw[rest]))
            except Exception:
                pass
    # A model that could not be fitted at all at this sample size is recorded as
    # a miss rather than dropped, because "it did not run" is the finding at the
    # small end and silently omitting it would hide exactly what we are
    # measuring.
    row = {"n_per_class": n_per, "reps": {k: len(v) for k, v in accs.items()}}
    for k, v in accs.items():
        row[k] = rnd(np.mean(v)) if v else None
        row[k + "_sd"] = rnd(np.std(v), 4) if v else None
    curve.append(row)

print("\n=== how much data each discriminant needs (13 features, 3 classes) ===")
show = lambda v: "  n/a " if v is None else f"{v:.4f}"
for r in curve:
    print(f"  {r['n_per_class']:2d} per class:  lda {show(r['lda'])}   qda {show(r['qda'])}   gnb {show(r['gnb'])}"
          f"   (fitted {r['reps']['qda']}/24 qda)")


# ------------------------------------------- was the kernel worth it, really
cancer = load_breast_cancer()
Xb = cancer.data
yb = 1 - cancer.target
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=7, stratify=yb)
sb = StandardScaler().fit(Xb_tr)
Ab, Bb = sb.transform(Xb_tr), sb.transform(Xb_te)

MODELS = [
    ("logistic regression", LogisticRegression(max_iter=10000)),
    ("linear SVM", SVC(kernel="linear", C=1)),
    ("RBF SVM", SVC(kernel="rbf", C=1, gamma="scale")),
    ("RBF SVM, tuned", SVC(kernel="rbf", C=10, gamma=0.01)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.05)),
]
bench = []
for name, model in MODELS:
    t0 = time.perf_counter()
    m = model.fit(Ab, yb_tr)
    fit_s = time.perf_counter() - t0
    accs = cross_val_score(model, sb.transform(Xb), yb, cv=cv)
    bench.append(
        {
            "name": name,
            "test_acc": rnd(m.score(Bb, yb_te)),
            "cv_mean": rnd(accs.mean()),
            "cv_sd": rnd(accs.std()),
            "seconds": rnd(fit_s, 4),
            "linear": name in ("logistic regression", "linear SVM", "LDA"),
        }
    )
    print(f"  {name:<22} held out {m.score(Bb, yb_te):.4f}   5-fold {accs.mean():.4f} +- {accs.std():.4f}   {fit_s*1000:6.1f} ms")

print("\n=== was the kernel worth it on 30 real features? ===")
best_lin = max(b["cv_mean"] for b in bench if b["linear"])
best_ker = max(b["cv_mean"] for b in bench if not b["linear"])
print(f"  best linear model  {best_lin:.4f}")
print(f"  best kernel model  {best_ker:.4f}")
print(f"  the kernel is worth {(best_ker - best_lin) * 100:+.2f} accuracy points")


out = {
    "margin": margin_panel,
    "rings": rings,
    "axes": axes_out,
    "wine_classes": [str(c) for c in wine.target_names],
    "discriminant_curve": curve,
    "benchmark": bench,
    "kernel_gain": rnd(best_ker - best_lin),
    "n_cancer": int(len(yb)),
    "n_wine": int(len(yw)),
    "versions": {"seed": SEED},
}
(OUT / "geometry.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'geometry.json'}  ({(OUT / 'geometry.json').stat().st_size / 1024:.1f} kB)")
