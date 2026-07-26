"""Data for the ATLAS 'Kernel Discriminant Analysis' web article: the last
unlit chip of module 1.

The demonstration dataset is the two moons the DBSCAN article clustered
without labels, now WITH their labels: a classification problem where no
straight projection direction works, which the reader will verify personally
with a dial. KDA (kernel Fisher discriminant) is implemented here in numpy,
the classical way: maximise between-class over within-class scatter of the
kernel-space projection. The page re-evaluates every projection and boundary
live from the shipped dual coefficients, and its accuracies must match.

Also measured: the honest benchmark on the 569 biopsies (articles 7 to 9),
where the kernel's worth on real tabular data gets the same sober audit the
SVM article gave it.

Run from anywhere: `python src/utils/generate_kda_data.py`.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kda" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19


def rbf(A, B, gamma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


def kda_fit(X, y, gamma, mu=1e-3):
    """Kernel Fisher discriminant, binary. Returns dual coefs, threshold and
    the sign convention (which side of b is class 1)."""
    K = rbf(X, X, gamma)
    n = len(y)
    i0 = np.where(y == 0)[0]
    i1 = np.where(y == 1)[0]
    m0 = K[:, i0].mean(1)
    m1 = K[:, i1].mean(1)
    N = np.zeros((n, n))
    for idx in (i0, i1):
        Kc = K[:, idx]
        H = np.eye(len(idx)) - 1.0 / len(idx)
        N += Kc @ H @ Kc.T
    alpha = np.linalg.solve(N + mu * np.eye(n), m1 - m0)
    z = K.T @ alpha
    b = (z[i0].mean() + z[i1].mean()) / 2
    return alpha, b


def kda_predict(Xq, X, alpha, b, gamma):
    z = rbf(Xq, X, gamma) @ alpha
    return (z > b).astype(int), z


# ------------------------------------------------- the moons, labels restored
Xm, ym = make_moons(n_samples=300, noise=0.09, random_state=SEED)
Xm = np.round(Xm * 2.0, 3)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=SEED, stratify=ym)
print(f"moons: {len(Xtr)} train / {len(Xte)} test")

# LDA and the best possible 1-D linear projection, for the dial's reference
lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
acc_lda = float(lda.score(Xte, yte))
w = lda.coef_[0] / np.linalg.norm(lda.coef_[0])

best_line = (0.0, 0.0)
for theta in np.linspace(0, np.pi, 721):
    d = np.array([np.cos(theta), np.sin(theta)])
    ztr = Xtr @ d
    zte = Xte @ d
    order = np.sort(np.unique(ztr))
    cuts = (order[:-1] + order[1:]) / 2
    accs = [max(((ztr > c).astype(int) == ytr).mean(),
                ((ztr <= c).astype(int) == ytr).mean()) for c in cuts]
    c = cuts[int(np.argmax(accs))]
    acc = max(((zte > c).astype(int) == yte).mean(), ((zte <= c).astype(int) == yte).mean())
    if acc > best_line[1]:
        best_line = (float(theta), float(acc))
print(f"  LDA test accuracy {acc_lda:.3f}; best 1-D projection over all angles: "
      f"{best_line[1]:.3f} at {np.degrees(best_line[0]):.0f} degrees")

# KDA across gamma: the seven stops that tell the whole arc, chosen by probe.
# At 0.01 the kernel is nearly flat and KDA collapses onto the linear ceiling
# (0.878, LDA's exact score); the plateau holds from 0.1 to 30; past 100 the
# bumps narrow onto individual points and the train/test gap opens wide.
GAMMAS = [0.01, 0.03, 0.3, 3.0, 30.0, 300.0, 1000.0]
kda_out = []
print("  KDA (kernel Fisher discriminant):")
for g in GAMMAS:
    alpha, b = kda_fit(Xtr, ytr, g)
    pr_tr, _ = kda_predict(Xtr, Xtr, alpha, b, g)
    pr_te, _ = kda_predict(Xte, Xtr, alpha, b, g)
    kda_out.append({
        "gamma": g,
        "alpha": [rnd(v, 6) for v in alpha],
        "b": rnd(b, 6),
        "acc_train": rnd((pr_tr == ytr).mean(), 4),
        "acc_test": rnd((pr_te == yte).mean(), 4),
    })
    print(f"    gamma {g:>4}: train {kda_out[-1]['acc_train']:.3f}  test {kda_out[-1]['acc_test']:.3f}")

# ---------------------------------------------- the biopsies, soberly audited
cancer = load_breast_cancer()
Xc = cancer.data
yc = cancer.target
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_acc(fit_predict):
    accs = []
    for tr, te in kf.split(Xc, yc):
        sc = StandardScaler().fit(Xc[tr])
        A, B = sc.transform(Xc[tr]), sc.transform(Xc[te])
        accs.append(fit_predict(A, yc[tr], B, yc[te]))
    return float(np.mean(accs)), float(np.std(accs))


def kda_cv_gamma(A, ytr2):
    """Small inner CV to pick gamma for the cancer benchmark."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    best = (None, -1)
    for g in [0.001, 0.003, 0.01, 0.03, 0.1]:
        accs = []
        for tr, va in inner.split(A, ytr2):
            alpha, b = kda_fit(A[tr], ytr2[tr], g)
            pr, _ = kda_predict(A[va], A[tr], alpha, b, g)
            accs.append((pr == ytr2[va]).mean())
        m = np.mean(accs)
        if m > best[1]:
            best = (g, m)
    return best[0]


print("\ncancer, 569 biopsies, 30 features, 5-fold:")
bench = []


def add(name, fn):
    m, s = cv_acc(fn)
    bench.append({"name": name, "acc": rnd(m, 4), "sd": rnd(s, 4)})
    print(f"  {name:<12} {m:.4f} ± {s:.4f}")


add("LDA", lambda A, ya, B, yb: LinearDiscriminantAnalysis().fit(A, ya).score(B, yb))
add("QDA", lambda A, ya, B, yb: QuadraticDiscriminantAnalysis(reg_param=0.1).fit(A, ya).score(B, yb))


def kda_bench(A, ya, B, yb):
    g = kda_cv_gamma(A, ya)
    alpha, b = kda_fit(A, ya, g)
    pr, _ = kda_predict(B, A, alpha, b, g)
    return (pr == yb).mean()


add("KDA", kda_bench)
add("SVM rbf", lambda A, ya, B, yb: SVC(kernel="rbf", gamma="scale", C=1.0).fit(A, ya).score(B, yb))

out = {
    "x_train": [[rnd(a, 3), rnd(b2, 3)] for a, b2 in Xtr],
    "y_train": [int(v) for v in ytr],
    "x_test": [[rnd(a, 3), rnd(b2, 3)] for a, b2 in Xte],
    "y_test": [int(v) for v in yte],
    "lda": {"w": [rnd(v, 5) for v in w], "acc_test": rnd(acc_lda, 4)},
    "best_line": {"theta": rnd(best_line[0], 4), "acc_test": rnd(best_line[1], 4)},
    "kda": kda_out,
    "bench": bench,
    "versions": {"seed": SEED},
}
(OUT / "discriminants.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'discriminants.json'}  ({(OUT / 'discriminants.json').stat().st_size / 1024:.1f} kB)")
