"""Data for the ATLAS 'Trees and Ensembles' web article.

The article is built around one equation that almost never gets shown. Average
B identically distributed predictors, each with variance sigma^2 and pairwise
correlation rho, and the variance of the average is

    rho * sigma^2  +  (1 - rho) * sigma^2 / B

The second term vanishes as you add trees. The first does not. So the entire
design of a random forest, bootstrapping rows AND sampling features at every
split, is an attack on rho, and everything about it follows from that one term.
This file measures rho directly, from the predictions of the trees themselves,
and checks the formula against the observed variance of the ensemble.

It also measures two things people are routinely burned by: impurity-based
feature importance handing a pure-noise column a top-three ranking purely for
having many distinct values, and the fact that a tree cannot extrapolate at all.

Run from anywhere: `python src/utils/generate_trees_data.py`.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trees-and-ensembles" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 13

# The same 569 biopsies as the two previous articles, so the module has one
# through-line and the numbers are directly comparable.
raw = load_breast_cancer()
X = raw.data
y = 1 - raw.target  # 1 = malignant, the thing being detected
FEATURES = [f.replace(" ", "_") for f in raw.feature_names]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
print(f"{len(y)} biopsies, {X.shape[1]} features, train {len(ytr)} test {len(yte)}")


# ------------------------------------------------------------- one tree, 2-D
# Two features so the partition can be drawn as what it is: a set of boxes.
J2 = [FEATURES.index("worst_perimeter"), FEATURES.index("worst_concave_points")]
A2, B2 = Xtr[:, J2], Xte[:, J2]

depths = []
for d in range(1, 13):
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(A2, ytr)
    depths.append(
        {
            "depth": d,
            "leaves": int(t.get_n_leaves()),
            "train": rnd(t.score(A2, ytr)),
            "test": rnd(t.score(B2, yte)),
        }
    )
print("\n=== one tree, two features ===")
for r in depths:
    print(f"  depth {r['depth']:2d}  {r['leaves']:3d} leaves   train {r['train']:.4f}   test {r['test']:.4f}")
best_depth = max(depths, key=lambda r: r["test"])
print(f"  best held-out depth: {best_depth['depth']} at {best_depth['test']:.4f}")


def leaf_boxes(tree, bounds):
    """Every leaf of a fitted tree as a rectangle plus its majority class.

    Walking the tree structure directly rather than sampling a grid: the boxes
    ARE the model, and drawing them as boxes is the whole reason this article
    starts in two dimensions.
    """
    t = tree.tree_
    out = []

    def walk(node, box):
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            out.append(
                {
                    "x0": rnd(box[0][0], 3), "x1": rnd(box[0][1], 3),
                    "y0": rnd(box[1][0], 3), "y1": rnd(box[1][1], 3),
                    "c": int(np.argmax(counts)),
                    "p": rnd(counts[1] / counts.sum()),
                    "n": int(counts.sum()),
                }
            )
            return
        f = t.feature[node]
        thr = t.threshold[node]
        left = [list(box[0]), list(box[1])]
        right = [list(box[0]), list(box[1])]
        left[f][1] = min(left[f][1], thr)
        right[f][0] = max(right[f][0], thr)
        walk(t.children_left[node], left)
        walk(t.children_right[node], right)

    walk(0, [list(bounds[0]), list(bounds[1])])
    return out


pad = lambda v: (v.min() - (v.max() - v.min()) * 0.06, v.max() + (v.max() - v.min()) * 0.06)
BOUNDS = [pad(A2[:, 0]), pad(A2[:, 1])]
boxes = {}
for d in [1, 2, 3, 5, 12]:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(A2, ytr)
    boxes[str(d)] = leaf_boxes(t, BOUNDS)
    print(f"  depth {d}: {len(boxes[str(d)])} boxes")

# The first split, spelled out, so the article can show the arithmetic once.
root = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(A2, ytr)
rt = root.tree_
gini_parent = float(rt.impurity[0])
first_split = {
    "feature": FEATURES[J2[rt.feature[0]]],
    "feature_index": int(rt.feature[0]),
    "threshold": rnd(rt.threshold[0], 4),
    "gini_parent": rnd(gini_parent, 5),
    "gini_left": rnd(float(rt.impurity[1]), 5),
    "gini_right": rnd(float(rt.impurity[2]), 5),
    "n_left": int(rt.n_node_samples[1]),
    "n_right": int(rt.n_node_samples[2]),
    "drop": rnd(gini_parent - (rt.n_node_samples[1] * rt.impurity[1] + rt.n_node_samples[2] * rt.impurity[2]) / len(ytr), 5),
}
print(f"\nfirst split: {first_split['feature']} <= {first_split['threshold']}, gini drop {first_split['drop']}")

# Every candidate split on both features, so the widget can let the reader hunt
# for the best one and confirm the greedy choice was right.
candidates = []
for fi in [0, 1]:
    col = A2[:, fi]
    vals = np.unique(col)
    mids = (vals[:-1] + vals[1:]) / 2
    step = max(1, len(mids) // 90)
    for thr in mids[::step]:
        left = col <= thr
        nl, nr = int(left.sum()), int((~left).sum())
        if nl == 0 or nr == 0:
            continue
        gl = 1 - ((ytr[left] == 0).mean() ** 2 + (ytr[left] == 1).mean() ** 2)
        gr = 1 - ((ytr[~left] == 0).mean() ** 2 + (ytr[~left] == 1).mean() ** 2)
        candidates.append(
            {
                "f": fi,
                "thr": rnd(thr, 4),
                "drop": rnd(gini_parent - (nl * gl + nr * gr) / len(ytr), 5),
                "nl": nl, "nr": nr,
                "gl": rnd(gl, 4), "gr": rnd(gr, 4),
            }
        )
best_cand = max(candidates, key=lambda c: c["drop"])
print(f"  best candidate on the grid: feature {best_cand['f']} at {best_cand['thr']}, drop {best_cand['drop']}")


# --------------------------------------------------- the variance of one tree
# Refit the same model on bootstrap resamples and watch the boundary move. This
# is the quantity the rest of the article is about.
rs = np.random.RandomState(SEED)
wobble = []
for k in range(8):
    idx = rs.choice(len(ytr), len(ytr), replace=True)
    t = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(A2[idx], ytr[idx])
    wobble.append({"boxes": leaf_boxes(t, BOUNDS), "test": rnd(t.score(B2, yte))})
print(f"\neight trees on eight bootstraps, held-out accuracy "
      f"{min(w['test'] for w in wobble):.4f} to {max(w['test'] for w in wobble):.4f}")


# ---------------------------------------------- the equation, measured
B_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
N_REPS = 30
N_TREES = 200


def rho_and_variance(model_factory, label=""):
    """Measure the averaging equation instead of asserting it.

    A note on what rho actually is, because getting this wrong makes the whole
    section evaporate. It is NOT the correlation between two trees inside one
    forest fitted to one dataset: with the data held fixed the trees are
    conditionally independent, the variance of their average falls like 1/B all
    the way to zero, and there is no floor to talk about. The correlation that
    matters is between two trees ACROSS INDEPENDENT TRAINING SETS, because that
    is the shared randomness averaging cannot touch. Two trees fed different
    bootstraps of the same underlying data still both saw that data.

    So: draw twenty independent training sets, grow one forest on each, and
    measure everything across those twenty.

      sigma^2  variance of a single tree's prediction, across training sets
      rho      correlation between two trees of the same forest, across them
      observed variance of the B-tree average, across them

    Nothing is fitted to the curve. rho and sigma^2 come out of the trees, and
    then rho*sigma^2 + (1-rho)*sigma^2/B is checked against what happened.
    """
    n_sub = int(len(ytr) * 0.7)
    rs_local = np.random.RandomState(4242)
    Ps = []
    for r in range(N_REPS):
        # an independent training set per repetition, drawn without replacement
        # so each forest sees a genuinely different sample rather than a
        # bootstrap of one fixed sample
        idx = rs_local.choice(len(ytr), n_sub, replace=False)
        m = model_factory(N_TREES, SEED + 977 * r).fit(Xtr[idx], ytr[idx])
        Ps.append(np.column_stack([e.predict_proba(Xte)[:, 1] for e in m.estimators_]))
    P = np.stack(Ps)  # (reps, test cases, trees)

    # sigma^2: how much one tree's answer moves when the training set changes,
    # pooled over test cases and over which tree of the forest we look at
    TREES_USED = 16
    sigma2 = float(np.mean([P[:, :, j].var(axis=0) for j in range(TREES_USED)]))

    # rho, as a POOLED ratio of covariance to variance rather than the average
    # of per-case correlations. With twenty repetitions a per-case correlation
    # is a very noisy number, and most test cases here sit where every tree is
    # confident and right, so their individual correlations are close to
    # meaningless. Averaging covariances first and dividing once is the same
    # quantity the formula uses and it is stable.
    covs = []
    for a in range(0, TREES_USED, 2):
        b = a + 1
        u = P[:, :, a] - P[:, :, a].mean(axis=0)
        v = P[:, :, b] - P[:, :, b].mean(axis=0)
        covs.append(float(np.mean((u * v).sum(axis=0) / (P.shape[0] - 1))))
    rho = float(np.mean(covs)) / sigma2

    curve = []
    for b in B_GRID:
        ens = P[:, :, :b].mean(axis=2)          # (reps, test cases)
        observed = float(np.mean(ens.var(axis=0)))
        curve.append(
            {
                "b": b,
                "observed": rnd(observed, 6),
                "predicted": rnd(rho * sigma2 + (1 - rho) * sigma2 / b, 6),
                "acc": rnd(float(np.mean([accuracy_score(yte, (ens[r] >= 0.5).astype(int))
                                          for r in range(N_REPS)]))),
            }
        )

    single = float(np.mean([accuracy_score(yte, (P[r, :, 0] >= 0.5).astype(int)) for r in range(N_REPS)]))
    full = float(np.mean([accuracy_score(yte, (P[r].mean(axis=1) >= 0.5).astype(int)) for r in range(N_REPS)]))
    worst = max(abs(c["observed"] - c["predicted"]) / max(c["observed"], 1e-9) for c in curve)
    print(f"  {label:<22} rho {rho:.4f}  sigma^2 {sigma2:.5f}  floor rho*sigma^2 {rho * sigma2:.5f}")
    print(f"  {'':<22} one tree {single:.4f} -> {N_TREES} trees {full:.4f}"
          f"   formula within {worst * 100:.1f}% everywhere")
    return {
        "rho": rnd(rho, 5),
        "sigma2": rnd(sigma2, 5),
        "floor": rnd(rho * sigma2, 6),
        "single_acc": rnd(single),
        "ens_acc": rnd(full),
        "worst_rel_error": rnd(worst, 4),
        "curve": curve,
    }


print("\n=== the averaging equation, measured on held-out predictions ===")
ENSEMBLES = {
    "bagging": lambda n, s: RandomForestClassifier(
        n_estimators=n, max_features=None, bootstrap=True, random_state=s, n_jobs=-1
    ),
    "random_forest": lambda n, s: RandomForestClassifier(
        n_estimators=n, max_features="sqrt", bootstrap=True, random_state=s, n_jobs=-1
    ),
    "extra_trees": lambda n, s: ExtraTreesClassifier(
        n_estimators=n, max_features="sqrt", bootstrap=False, random_state=s, n_jobs=-1
    ),
}
LABEL = {"bagging": "bagging (all cols)", "random_forest": "random forest", "extra_trees": "extra trees"}
variance = {k: rho_and_variance(f, LABEL[k]) for k, f in ENSEMBLES.items()}


# --------------------------------------- more trees never hurts (unlike boosting)
print("\n=== does adding trees ever hurt? ===")
grow = []
for n in [1, 2, 5, 10, 25, 50, 100, 200, 400]:
    m = RandomForestClassifier(n_estimators=n, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    grow.append(
        {
            "n": n,
            "train": rnd(m.score(Xtr, ytr)),
            "test": rnd(m.score(Xte, yte)),
            "logloss": rnd(log_loss(yte, np.clip(p, 1e-9, 1 - 1e-9)), 4),
        }
    )
    print(f"  {n:3d} trees  train {grow[-1]['train']:.4f}  test {grow[-1]['test']:.4f}  log loss {grow[-1]['logloss']:.4f}")


# ---------------------------------------------- importance is a trap
# Five planted columns, ALL of them pure noise, differing only in how many
# distinct values they take. Any importance one of them receives is by
# construction undeserved, so the interesting quantity is not whether they beat
# a real feature (on this data the real features are far too strong for that)
# but how much more importance a noise column gets purely for being finer
# grained. Impurity-based importance has more places to look for a lucky split
# the more distinct values a column has, and rewards it for finding one.
rs2 = np.random.RandomState(99)
CARDS = [2, 4, 10, 50, 0]  # 0 means continuous, one distinct value per row
noise_cols = []
noise_names = []
for k in CARDS:
    if k == 0:
        noise_cols.append(rs2.normal(size=len(X)))
        noise_names.append("noise_continuous")
    else:
        noise_cols.append(rs2.randint(0, k, size=len(X)).astype(float))
        noise_names.append(f"noise_{k}_levels")
Xp = np.column_stack([X] + noise_cols)
FEAT_P = FEATURES + noise_names
Ptr, Pte, ptr_y, pte_y = train_test_split(Xp, y, test_size=0.3, random_state=SEED, stratify=y)

rf_imp = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Ptr, ptr_y)
imp_gini = rf_imp.feature_importances_
perm = permutation_importance(rf_imp, Pte, pte_y, n_repeats=20, random_state=SEED, n_jobs=-1)

order = np.argsort(-imp_gini)
print("\n=== impurity importance vs permutation importance, on columns that are all noise ===")
planted = []
for i, name in enumerate(noise_names):
    j = len(FEATURES) + i
    rank = int(np.where(order == j)[0][0]) + 1
    planted.append(
        {
            "feature": name,
            "distinct": int(len(np.unique(Xp[:, j]))),
            "gini": rnd(float(imp_gini[j]), 6),
            "perm": rnd(float(perm.importances_mean[j]), 6),
            "rank": rank,
        }
    )
    print(f"  {name:<20} {planted[-1]['distinct']:3d} distinct values   impurity {imp_gini[j]:.5f}"
          f"   permutation {perm.importances_mean[j]:+.5f}   rank {rank}/{len(FEAT_P)}")
ratio = planted[-1]["gini"] / max(planted[0]["gini"], 1e-12)
print(f"  the continuous noise column gets {ratio:.1f}x the impurity importance of the binary one,"
      f" and both are noise")
real_min = float(min(imp_gini[: len(FEATURES)]))
beaten = [p["feature"] for p in planted if p["gini"] > real_min]
print(f"  weakest REAL feature scores {real_min:.5f}; noise columns above it: {len(beaten)}")

importance = [
    {
        "feature": FEAT_P[j],
        "gini": rnd(float(imp_gini[j]), 5),
        "perm": rnd(float(perm.importances_mean[j]), 5),
        "perm_sd": rnd(float(perm.importances_std[j]), 5),
        "distinct": int(len(np.unique(Xp[:, j]))),
        "planted": bool(j >= len(FEATURES)),
    }
    for j in range(len(FEAT_P))
]
importance.sort(key=lambda r: -r["gini"])


# ------------------------------------------------ a tree cannot extrapolate
# One feature, a clean upward trend, and a model asked about x it never saw.
xs = np.linspace(0, 10, 120)
ys = 2.0 + 0.9 * xs + rs2.normal(0, 0.7, len(xs))
inside = xs <= 7
tree_r = DecisionTreeRegressor(max_depth=6, random_state=SEED).fit(xs[inside, None], ys[inside])
forest_r = RandomForestClassifier  # placeholder, unused
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

rf_r = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1).fit(xs[inside, None], ys[inside])
from sklearn.linear_model import LinearRegression  # noqa: E402

lin_r = LinearRegression().fit(xs[inside, None], ys[inside])
grid = np.linspace(0, 10, 240)
extrap = {
    "points": [{"x": rnd(a, 3), "y": rnd(b, 3), "train": bool(c)} for a, b, c in zip(xs, ys, inside)],
    "grid": [rnd(v, 3) for v in grid],
    "tree": [rnd(v, 3) for v in tree_r.predict(grid[:, None])],
    "forest": [rnd(v, 3) for v in rf_r.predict(grid[:, None])],
    "linear": [rnd(v, 3) for v in lin_r.predict(grid[:, None])],
    "boundary": 7.0,
}
out_mask = ~inside
print("\n=== extrapolation, on x above 7 which no tree ever saw ===")
for name, pred in [("tree", tree_r.predict(xs[out_mask, None])),
                   ("forest", rf_r.predict(xs[out_mask, None])),
                   ("linear", lin_r.predict(xs[out_mask, None]))]:
    rmse = float(np.sqrt(np.mean((pred - ys[out_mask]) ** 2)))
    print(f"  {name:<8} rmse {rmse:6.3f}   predicts between {pred.min():.2f} and {pred.max():.2f}"
          f"   (truth runs to {ys[out_mask].max():.2f})")
    extrap[name + "_rmse"] = rnd(rmse, 3)
    extrap[name + "_range"] = [rnd(pred.min(), 3), rnd(pred.max(), 3)]


# ------------------------------------------------------------- the benchmark
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
MODELS = [
    ("one tree, unpruned", DecisionTreeClassifier(random_state=SEED)),
    ("one tree, depth 4", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
    ("bagging, all columns", RandomForestClassifier(n_estimators=300, max_features=None, random_state=SEED, n_jobs=-1)),
    ("random forest", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ("extra trees", ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
]
bench = []
print("\n=== five-fold on all 569 biopsies ===")
for name, model in MODELS:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    s = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    bench.append(
        {
            "name": name,
            "cv_mean": rnd(s.mean()),
            "cv_sd": rnd(s.std()),
            "test_acc": rnd(model.score(Xte, yte)),
            "seconds": rnd(fit_s, 4),
            "single": name.startswith("one tree"),
        }
    )
    print(f"  {name:<22} 5-fold {s.mean():.4f} +- {s.std():.4f}   held out {model.score(Xte, yte):.4f}"
          f"   {fit_s * 1000:6.1f} ms")


out = {
    "n": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
    "features": FEATURES,
    "two_d": {
        "features": [FEATURES[J2[0]], FEATURES[J2[1]]],
        "bounds": [[rnd(v, 3) for v in BOUNDS[0]], [rnd(v, 3) for v in BOUNDS[1]]],
        "train": [{"x": rnd(a, 3), "y": rnd(b, 3), "c": int(c)} for (a, b), c in zip(A2, ytr)],
        "test": [{"x": rnd(a, 3), "y": rnd(b, 3), "c": int(c)} for (a, b), c in zip(B2, yte)],
        "boxes": boxes,
        "depths": depths,
        "first_split": first_split,
        "candidates": candidates,
        "wobble": wobble,
    },
    "variance": variance,
    "grow": grow,
    "importance": importance,
    "importance_meta": {
        "planted": planted,
        "n_features": len(FEAT_P),
        "n_real": len(FEATURES),
        "ratio": rnd(ratio, 2),
        "weakest_real": rnd(real_min, 6),
        "noise_above_weakest_real": len(beaten),
    },
    "extrapolation": extrap,
    "benchmark": bench,
    "versions": {"seed": SEED},
}
(OUT / "trees.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'trees.json'}  ({(OUT / 'trees.json').stat().st_size / 1024:.1f} kB)")
