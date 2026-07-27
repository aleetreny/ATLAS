"""Data for the ATLAS 'Directions' article: PCA, ICA and NMF as one equation.

The spine is that all three factorise the same X into a few building blocks and
the amounts of each, and that what separates them is the constraint: PCA asks
for orthogonal directions of maximum variance, ICA asks for statistically
independent ones, NMF asks for nothing negative. So everything measured here is
measured the same way for all three wherever that is possible, and the places
where it is not possible are themselves the finding.

Four blocks:

1. A 2-D cloud (two real wine measurements) for the angle dial. The reference
   the reader is scored against is swept at the dial's OWN resolution, not at
   the data's, so it is both unbeatable and reachable.
2. The full 13-dimensional wine, standardised and not, because the single most
   expensive PCA mistake is forgetting that variance has units. Ships the raw
   matrix so the browser can do its own eigendecomposition and check itself.
3. Two signals and two microphones for ICA, plus the same experiment with
   gaussian sources, which is the case ICA provably cannot solve. The rotation
   sweep is exported as five fourth-order moments, because the whole 180-degree
   curve is a closed form in those and the widget draws it from them.
4. Atoms: PCA, ICA and NMF factorisations of 2,000 MNIST digits at five ranks,
   quantised to uint8 and tiled into one PNG the browser reads pixels out of.
   Every reference error here is computed from the QUANTISED atoms, because
   those are the numbers the browser will actually have.

Two traps paid for during the writing, both recorded in docs/:

* FastICA run directly on 784 raw pixels does not converge and returns an
  unmixing matrix with condition number 5.1e15, silently. MNIST has 146
  pixels of exactly zero variance, so the whitening step is singular. Running
  ICA on the PCA scores instead is both what every implementation does under
  the hood and well conditioned, and it makes "ICA lives in PCA's subspace"
  true by construction rather than by luck.
* numpy's np.cov defaults to ddof=1 and a browser has no such notion, so the
  whitening and every fourth-order moment here uses the population form.
  Mixing the two put the closed-form rotation sweep 3e-2 away from the honest
  one, which looks exactly like an implementation bug and is not.

Run from anywhere: `python src/utils/generate_directions_data.py`.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces, load_wine
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.vision_data import mnist, subset            # noqa: E402

OUT = ROOT / "directions" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 19
rnd = lambda v, n=4: round(float(v), n)
arr = lambda a, n=4: [round(float(v), n) for v in np.asarray(a).ravel()]

DIAL_STEP = 0.5                       # the reader's dial resolution, in degrees
DEGS = np.arange(0, 180, DIAL_STEP)


# ---------------------------------------------------------------- shared maths
def var0(v):
    """Population variance. A browser has no ddof, so neither does this file."""
    v = np.asarray(v, dtype=float)
    return float(((v - v.mean()) ** 2).mean())


def kurtosis(v):
    v = np.asarray(v, dtype=float)
    c = v - v.mean()
    s = np.sqrt((c ** 2).mean())
    return float(((c / s) ** 4).mean() - 3)


def whiten(X):
    """Centre and decorrelate to identity covariance (population form)."""
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C)
    return Xc @ (V @ np.diag(1 / np.sqrt(w)) @ V.T)


def rot(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def rotation_sweep(Z):
    """kurt(y1)^2 + kurt(y2)^2 for every angle in DEGS, in closed form.

    For whitened Z any rotation keeps unit variance, so the fourth moment of
    y1 = cos(t) z1 + sin(t) z2 is a quartic in (cos, sin) whose five
    coefficients are the mixed fourth moments of Z. The widget draws this
    curve from those five numbers rather than from 600 points per angle.
    """
    z1, z2 = Z[:, 0], Z[:, 1]
    m = np.array([(z1 ** 4).mean(), (z1 ** 3 * z2).mean(), (z1 ** 2 * z2 ** 2).mean(),
                  (z1 * z2 ** 3).mean(), (z2 ** 4).mean()])
    c, s = np.cos(np.radians(DEGS)), np.sin(np.radians(DEGS))
    k1 = (c**4 * m[0] + 4*c**3*s * m[1] + 6*c**2*s**2 * m[2]
          + 4*c*s**3 * m[3] + s**4 * m[4]) - 3
    k2 = (s**4 * m[0] - 4*s**3*c * m[1] + 6*s**2*c**2 * m[2]
          - 4*s*c**3 * m[3] + c**4 * m[4]) - 3
    return m, k1 ** 2 + k2 ** 2


def recovery(rec, src):
    """(how well the worst source is recovered, how much of the other leaks in).

    Both signs and both pairings are allowed, because a factorisation can only
    ever return the sources up to sign and order.
    """
    M = np.abs(np.corrcoef(rec.T, src.T)[:2, 2:])
    if M[0, 0] + M[1, 1] >= M[0, 1] + M[1, 0]:
        return float(min(M[0, 0], M[1, 1])), float(max(M[0, 1], M[1, 0]))
    return float(min(M[0, 1], M[1, 0])), float(max(M[0, 0], M[1, 1]))


def mse(a, b):
    return float(((np.asarray(a) - np.asarray(b)) ** 2).mean())


def hoyer(v):
    """0 when every entry is equal, 1 when one entry carries everything."""
    v = np.asarray(v, dtype=float)
    n = v.size
    denom = np.sqrt((v ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((np.sqrt(n) - np.abs(v).sum() / denom) / (np.sqrt(n) - 1))


def top_mass(v, frac=0.1):
    """Share of an atom's absolute mass carried by its heaviest 10% of pixels.

    Unlike Hoyer this is not inflated by pixels that are dead in every atom,
    which matters on digits where 146 of 784 pixels are always zero.
    """
    a = np.sort(np.abs(np.asarray(v, dtype=float)))[::-1]
    k = max(1, int(round(frac * a.size)))
    return float(a[:k].sum() / a.sum())


print("=" * 72)
print("ATLAS / directions: PCA, ICA, NMF")
print("=" * 72)

# =========================================================== 1. the 2-D cloud
# Two real measurements off the same 178 bottles the clustering articles used.
# Raw units, only centred: the dial's answer has to be allowed to depend on the
# units, because the next section is about exactly that.
wine = load_wine()
NAMES = list(wine.feature_names)
XW = np.round(wine.data, 3)                       # rounded FIRST, then measured
YW = wine.target
PAIR = ("flavanoids", "od280/od315_of_diluted_wines")
pi, pj = NAMES.index(PAIR[0]), NAMES.index(PAIR[1])

RAW2 = XW[:, [pi, pj]] - XW[:, [pi, pj]].mean(0)
STD2 = RAW2 / np.sqrt((RAW2 ** 2).mean(0))
cloud = {}
# Rounded FIRST and measured after, or the browser (which only ever sees the
# rounded coordinates) computes a total that disagrees with the one shipped
# here, and the identity guard fires on the rounding rather than on a bug.
for key, P0 in [("raw", np.round(RAW2, 4)), ("std", np.round(STD2, 4))]:
    # Rounding moves the mean about 1e-5 off zero, and every quantity below
    # assumes a centred cloud: the projection's variance is measured about its
    # own mean and the residual from a line through the origin, so a mean of m
    # leaves captured + thrown away short of the total by exactly m squared.
    # The browser re-centres the rounded coordinates the same way, so the two
    # are computing the same thing and the identity comes out exact at both.
    P = P0 - P0.mean(0)
    total = float((P ** 2).sum(1).mean())         # total variance, both features
    caps = []
    for d in DEGS:
        u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))])
        caps.append(var0(P @ u))
    caps = np.array(caps)
    best_i = int(caps.argmax())
    ev, evec = np.linalg.eigh(P.T @ P / len(P))
    exact = float(np.degrees(np.arctan2(evec[1, -1], evec[0, -1])) % 180)
    cloud[key] = {
        # the rounded coordinates as shipped; the browser centres them itself
        "points": [[rnd(a, 4), rnd(b, 4)] for a, b in P0],
        "total": rnd(total, 8),
        "dial_best_deg": rnd(DEGS[best_i], 3),
        "dial_best_captured": rnd(caps[best_i], 6),
        "dial_best_share": rnd(caps[best_i] / total, 5),
        "exact_deg": rnd(exact, 3),
        "exact_share": rnd(ev[-1] / ev.sum(), 5),
        "worst_deg": rnd(DEGS[int(caps.argmin())], 3),
        "worst_share": rnd(caps.min() / total, 5),
    }
    print(f"\n2-D cloud, {key}: best direction {cloud[key]['dial_best_deg']} deg "
          f"(exact eigenvector {cloud[key]['exact_deg']}), "
          f"{cloud[key]['dial_best_share']*100:.2f}% of the variance; "
          f"worst {cloud[key]['worst_share']*100:.2f}%")

# the identity that makes PCA two algorithms at once, checked before it is claimed
P = np.round(RAW2, 4)
P = P - P.mean(0)
gap = 0.0
for d in DEGS:
    u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))])
    proj = P @ u
    resid = float(((P - np.outer(proj, u)) ** 2).sum(1).mean())
    gap = max(gap, abs(var0(proj) + resid - float((P ** 2).sum(1).mean())))
print(f"  captured + residual = total, worst violation over {len(DEGS)} angles: {gap:.3e}")
cloud["identity_gap"] = float(f"{gap:.3e}")
cloud["features"] = list(PAIR)
cloud["dial_step"] = DIAL_STEP
cloud["classes"] = [int(v) for v in YW]

# ======================================================= 2. the full spectrum
print("\n" + "=" * 72)
print("the 13-dimensional wine: what standardising costs and buys")
spectrum = {
    "features": NAMES,
    "matrix": [[rnd(v, 3) for v in row] for row in XW],   # the browser's own input
    "classes": [int(v) for v in YW],
    "sd": arr(XW.std(0), 4),
}
for key, X in [("raw", XW - XW.mean(0)),
               ("std", (XW - XW.mean(0)) / XW.std(0))]:
    p = PCA().fit(X)
    evr = p.explained_variance_ratio_
    load1 = p.components_[0]
    if load1[np.abs(load1).argmax()] < 0:         # sign is arbitrary; fix it
        load1 = -load1
    k90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    top = int(np.abs(load1).argmax())
    spectrum[key] = {
        "evr": arr(evr, 6),
        "cumulative": arr(np.cumsum(evr), 6),
        "loadings1": arr(load1, 5),
        "k90": k90,
        "top_feature": NAMES[top],
        "top_loading": rnd(abs(load1[top]), 5),
    }
    print(f"  {key:<4} PC1 {evr[0]*100:6.2f}%  PC2 {evr[1]*100:5.2f}%  "
          f"90% at k={k90:<2}  PC1 is mostly {NAMES[top]} ({abs(load1[top]):.4f})")

# variance order is not usefulness order, and the last component proves it
XS = (XW - XW.mean(0)) / XW.std(0)
Z = PCA().fit_transform(XS)
F = []
for k in range(13):
    z = Z[:, k]
    grand = z.mean()
    between = sum(len(z[YW == c]) * (z[YW == c].mean() - grand) ** 2 for c in range(3)) / 2
    within = sum(((z[YW == c] - z[YW == c].mean()) ** 2).sum() for c in range(3)) / (len(z) - 3)
    F.append(between / within)
F = np.array(F)
order = [int(i) + 1 for i in np.argsort(-F)]
beaten = int((F[12] > F[2:12]).sum())
spectrum["separation"] = {
    "f": arr(F, 4),
    "order": order,
    "last_beats": beaten,
    "last_f": rnd(F[12], 4),
    "note": "one-way F of each component's scores against the three cultivars",
}
print(f"  by variance:   {list(range(1, 14))}")
print(f"  by separation: {order}")
print(f"  the LAST component (0.80% of the variance) separates better than "
      f"{beaten} of the ten between it and PC2")

# k-means on the first k components, against article 12's answer on all 13
ari_all = adjusted_rand_score(YW, KMeans(3, n_init=10, random_state=SEED).fit_predict(XS))
rows = []
for k in range(1, 14):
    a = adjusted_rand_score(YW, KMeans(3, n_init=10, random_state=SEED).fit_predict(Z[:, :k]))
    rows.append({"k": k, "ari": rnd(a, 4)})
best_k = max(rows, key=lambda r: r["ari"])
spectrum["kmeans"] = {"all_features": rnd(ari_all, 4), "by_k": rows,
                      "best_k": best_k["k"], "best_ari": best_k["ari"]}
print(f"  k-means k=3: all 13 features ARI {ari_all:.4f}; best on "
      f"{best_k['k']} component(s) at {best_k['ari']:.4f}")

# ================================================================== 3. the ICA
print("\n" + "=" * 72)
print("two voices, two microphones")
N_SIG = 600
t = np.linspace(0, 8, N_SIG)
S = np.column_stack([np.sin(2.2 * t), 2 * (t / 1.7 - np.floor(t / 1.7 + 0.5))])
S = np.round((S - S.mean(0)) / S.std(0), 4)
MIX = np.array([[1.0, 0.2], [1.8, 1.0]])          # chosen by the search below
# Rounded before anything is measured: the browser whitens the rounded
# recordings, so the fourth moments here have to come from those too or the
# closed-form sweep disagrees with the page by a few parts in a million.
Xmix = np.round(S @ MIX.T, 4)
Zw = whiten(Xmix)
moments, obj = rotation_sweep(Zw)
peak = float(DEGS[int(obj.argmax())])
pca_match, pca_leak = recovery(Zw, S)
ica_match, ica_leak = recovery(Zw @ rot(peak), S)
print(f"  mixing desk {MIX.tolist()}, det {np.linalg.det(MIX):+.3f}")
print(f"  kurtosis: voices {kurtosis(S[:,0]):+.3f} / {kurtosis(S[:,1]):+.3f}; "
      f"mics {kurtosis(Xmix[:,0]):+.3f} / {kurtosis(Xmix[:,1]):+.3f}")
print(f"  dial optimum {peak} deg, objective span {obj.max()-obj.min():.4f}")
print(f"  PCA alone: worst voice recovered {pca_match:.4f}, other voice leaking "
      f"{pca_leak:.4f}")
print(f"  at the peak: worst voice {ica_match:.4f}, leakage {ica_leak:.4f}")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fica = FastICA(n_components=2, random_state=SEED, whiten="unit-variance",
                   max_iter=4000, tol=1e-9).fit(Xmix)
    fica_warned = len(caught)
Yf = fica.transform(Xmix)
f_match, f_leak = recovery(Yf, S)
# sklearn's answer expressed as an angle in the same whitened frame as the dial
Rsk = np.linalg.lstsq(Zw, Yf, rcond=None)[0]
f_angle = float(np.degrees(np.arctan2(Rsk[1, 0], Rsk[0, 0])) % 180)
print(f"  sklearn FastICA: worst voice {f_match:.4f}, leakage {f_leak:.4f}, "
      f"{fica.n_iter_} iterations, angle {f_angle:.2f} deg "
      f"(dial peak {peak}, same answer mod 90: {abs((f_angle - peak) % 90):.2f})")

# the gaussian case: the same everything, sources that ICA provably cannot split
rs = np.random.RandomState(SEED)
SG = rs.normal(size=(N_SIG, 2))
SG = np.round((SG - SG.mean(0)) / SG.std(0), 4)
XG = np.round(SG @ MIX.T, 4)
ZG = whiten(XG)
gmoments, gobj = rotation_sweep(ZG)
gpeak = float(DEGS[int(gobj.argmax())])
g_match, g_leak = recovery(ZG @ rot(gpeak), SG)
print(f"\n  gaussian voices: objective span {gobj.max()-gobj.min():.4f} against "
      f"{obj.max()-obj.min():.4f} ({(obj.max()-obj.min())/(gobj.max()-gobj.min()):.0f}x flatter)")

# ...and where the peak lands over independent draws, which is the real proof
def peak_spread(gaussian, draws=40):
    peaks, spans, matches = [], [], []
    for s in range(draws):
        r2 = np.random.RandomState(200 + s)
        if gaussian:
            Sd = r2.normal(size=(N_SIG, 2))
        else:
            Sd = S + r2.normal(0, 0.02, (N_SIG, 2))
        Sd = (Sd - Sd.mean(0)) / Sd.std(0)
        Zd = whiten(Sd @ MIX.T)
        _, o = rotation_sweep(Zd)
        pk = float(DEGS[int(o.argmax())])
        peaks.append(pk % 90)                      # a 90 deg turn only swaps the two
        spans.append(float(o.max() - o.min()))
        matches.append(recovery(Zd @ rot(pk), Sd)[0])
    a = np.radians(4 * np.array(peaks))
    Rbar = float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))
    sd = float(np.degrees(np.sqrt(-2 * np.log(max(Rbar, 1e-12)))) / 4)
    return sd, float(np.mean(spans)), float(np.mean(matches)), float(np.std(matches))


nong_sd, nong_span, nong_m, nong_s = peak_spread(False)
gauss_sd, gauss_span, gauss_m, gauss_s = peak_spread(True)
# what a peak landing uniformly at random would give, with the same 40 draws
uni = []
for s in range(400):
    r3 = np.random.RandomState(9000 + s)
    a = np.radians(4 * r3.uniform(0, 90, 40))
    Rb = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    uni.append(np.degrees(np.sqrt(-2 * np.log(max(Rb, 1e-12)))) / 4)
uni_sd = float(np.mean(uni))
print(f"  peak position over 40 draws (folded mod 90):")
print(f"    real voices    {nong_sd:6.2f} deg spread, worst match {nong_m:.4f}")
print(f"    gaussian       {gauss_sd:6.2f} deg spread, worst match {gauss_m:.4f}")
print(f"    pure chance    {uni_sd:6.2f} deg spread (400 uniform draws of 40)")

ica_out = {
    "n": N_SIG,
    "sources": [[rnd(a, 4), rnd(b, 4)] for a, b in S],
    "mixing": [[rnd(v, 3) for v in row] for row in MIX],
    "mixed": [[rnd(a, 4), rnd(b, 4)] for a, b in Xmix],
    "white": [[rnd(a, 4), rnd(b, 4)] for a, b in Zw],
    "moments": arr(moments, 6),
    "dial_step": DIAL_STEP,
    "peak_deg": peak,
    "objective_max": rnd(obj.max(), 5),
    "objective_min": rnd(obj.min(), 5),
    "objective_span": rnd(obj.max() - obj.min(), 5),
    "pca_match": rnd(pca_match, 4), "pca_leak": rnd(pca_leak, 4),
    "ica_match": rnd(ica_match, 4), "ica_leak": rnd(ica_leak, 4),
    "kurtosis": {"sources": [rnd(kurtosis(S[:, 0]), 4), rnd(kurtosis(S[:, 1]), 4)],
                 "mics": [rnd(kurtosis(Xmix[:, 0]), 4), rnd(kurtosis(Xmix[:, 1]), 4)],
                 "white": [rnd(kurtosis(Zw[:, 0]), 4), rnd(kurtosis(Zw[:, 1]), 4)]},
    "fastica": {"angle": rnd(f_angle, 3), "iters": int(fica.n_iter_),
                "match": rnd(f_match, 4), "leak": rnd(f_leak, 4),
                "warned": int(fica_warned),
                "agrees_mod90": rnd(abs((f_angle - peak) % 90), 3)},
    "gaussian": {
        "sources": [[rnd(a, 4), rnd(b, 4)] for a, b in SG],
        "mixed": [[rnd(a, 4), rnd(b, 4)] for a, b in XG],
        "white": [[rnd(a, 4), rnd(b, 4)] for a, b in ZG],
        "moments": arr(gmoments, 6),
        "peak_deg": gpeak,
        "objective_span": rnd(gobj.max() - gobj.min(), 5),
        "flatter_by": rnd((obj.max() - obj.min()) / (gobj.max() - gobj.min()), 1),
        "match": rnd(g_match, 4), "leak": rnd(g_leak, 4),
    },
    "stability": {
        "draws": 40,
        "real_spread_deg": rnd(nong_sd, 2), "real_span": rnd(nong_span, 4),
        "real_match": rnd(nong_m, 4), "real_match_sd": rnd(nong_s, 4),
        "gaussian_spread_deg": rnd(gauss_sd, 2), "gaussian_span": rnd(gauss_span, 4),
        "gaussian_match": rnd(gauss_m, 4), "gaussian_match_sd": rnd(gauss_s, 4),
        "chance_spread_deg": rnd(uni_sd, 2),
    },
}

# ================================================================ 4. the atoms
print("\n" + "=" * 72)
print("atoms: the same 2,000 digits factorised three ways at five ranks")
Xtr, ytr, _, _ = mnist()
Xd, yd = subset(Xtr, ytr, 2000, seed=SEED)
DIG = Xd.reshape(len(Xd), -1) / 255.0
RANKS = [2, 4, 8, 16, 25]
KMAX = max(RANKS)
print(f"  {DIG.shape[0]} digits, {DIG.shape[1]} pixels, "
      f"{int((DIG.var(0) == 0).sum())} of them dead in every image")

pca_full = PCA(n_components=KMAX, random_state=SEED).fit(DIG)
MEAN_IMG = pca_full.mean_.copy()
PCA_ATOMS = pca_full.components_.copy()

ica_atoms, ica_meta = {}, {}
for r in RANKS:
    p = PCA(n_components=r, random_state=SEED).fit(DIG)
    scores = p.transform(DIG)
    chosen = None
    for seed in (SEED, SEED + 1, SEED + 2, SEED + 3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f = FastICA(n_components=r, random_state=seed, whiten="unit-variance",
                        max_iter=3000, tol=1e-6).fit(scores)
            ok = len(caught) == 0
        if ok:
            chosen = (seed, f)
            break
        chosen = chosen or (seed, f)
    seed, f = chosen
    ica_atoms[r] = f.mixing_.T @ p.components_
    sv = np.linalg.svd(f.components_, compute_uv=False)
    ica_meta[r] = {"seed": int(seed), "iters": int(f.n_iter_),
                   "converged": int(f.n_iter_ < 3000),
                   "cond": rnd(sv.max() / sv.min(), 3),
                   "offset_norm": rnd(np.linalg.norm(f.mean_), 8)}

nmf_atoms, nmf_meta = {}, {}
for r in RANKS:
    m = NMF(n_components=r, init="nndsvda", max_iter=3000, random_state=SEED, tol=1e-6)
    W = m.fit_transform(DIG)
    nmf_atoms[r] = m.components_.copy()
    nmf_meta[r] = {"iters": int(m.n_iter_), "converged": int(m.n_iter_ < 3000)}

# ---- quantise to uint8, because that is what the browser will read out of a PNG
TILE = 28
SHOW = [int(np.where(yd == c)[0][0]) for c in range(10)]
tiles, decode, index = [], [], {}


def add_tile(vec, key, lo=None, hi=None):
    """Quantise one atom to 8 bits and return what the BROWSER will decode.

    The endpoints go into the JSON at six decimals, so the reference has to
    decode with the rounded endpoints too. Decoding here at full precision and
    there at six put the two 1e-5 apart, which is small, invisible, and enough
    to make the per-image guard fire on rounding instead of on a mistake.

    `lo`/`hi` override the per-tile range. The showcase digits use a fixed
    [0, 1] because MNIST pixels are ALREADY bytes: with a per-tile range they
    would be requantised onto a different 256-point grid and come back up to
    0.002 away from themselves, which is invisible on screen and is exactly
    enough to make the browser's fit of a digit disagree with the reference
    fit of the same digit by a tenth of a per cent.
    """
    v = np.asarray(vec, dtype=float)
    lo = float(v.min()) if lo is None else float(lo)
    hi = float(v.max()) if hi is None else float(hi)
    span = hi - lo if hi > lo else 1.0
    q = np.round((v - lo) / span * 255).astype(np.uint8)
    lo_r, hi_r = rnd(lo, 6), rnd(lo + span, 6)
    index[key] = len(tiles)
    tiles.append(q)
    decode.append([lo_r, hi_r])
    return lo_r + q.astype(float) / 255.0 * (hi_r - lo_r)


show_q = [add_tile(DIG[i], f"show{n}", lo=0.0, hi=1.0) for n, i in enumerate(SHOW)]
worst_show = max(float(np.abs(q - DIG[i]).max()) for q, i in zip(show_q, SHOW))
print(f"  showcase digits survive the round trip to {worst_show:.2e}")
mean_q = add_tile(MEAN_IMG, "mean")
pca_q = np.array([add_tile(PCA_ATOMS[j], f"pca{j}") for j in range(KMAX)])
ica_q = {r: np.array([add_tile(ica_atoms[r][j], f"ica{r}_{j}") for j in range(r)])
         for r in RANKS}
nmf_q = {r: np.array([add_tile(nmf_atoms[r][j], f"nmf{r}_{j}") for j in range(r)])
         for r in RANKS}
print(f"  {len(tiles)} tiles of {TILE}x{TILE} quantised to uint8")

# ---- every reference number below comes from the quantised atoms
def fit_linear(x, atoms, offset):
    """Least squares coefficients: what the browser solves for PCA and ICA."""
    G = atoms @ atoms.T
    return np.linalg.solve(G, atoms @ (x - offset))


def fit_nmf(x, atoms, iters=400):
    """Multiplicative updates for the coefficients with the atoms held fixed:
    the browser runs exactly this loop, so the reference has to as well."""
    w = np.full(len(atoms), float(x.mean() / max(atoms.mean(), 1e-9)) / len(atoms) + 1e-3)
    HHt = atoms @ atoms.T
    xHt = atoms @ x
    for _ in range(iters):
        w = w * xHt / np.maximum(w @ HHt, 1e-10)
    return w


# Quantising each atom to uint8 puts about 0.002 of noise on a pixel, which is
# larger than the value of a background pixel, so "how many pixels came out
# below zero" counted at zero measures the rounding and not the algorithm: it
# read 34% where the unrounded fit says 25%. Count ink that is negative by a
# stated, visible amount instead, and check the two agree.
NEG = 0.01
errors, per_image = [], []
for r in RANKS:
    recon_p = np.array([fit_linear(x, pca_q[:r], mean_q) @ pca_q[:r] + mean_q for x in DIG])
    ep = mse(DIG, recon_p)
    recon_i = np.array([fit_linear(x, ica_q[r], mean_q) @ ica_q[r] + mean_q for x in DIG])
    ei = mse(DIG, recon_i)
    Wn = np.array([fit_nmf(x, nmf_q[r]) for x in DIG])
    en = mse(DIG, Wn @ nmf_q[r])
    recon_n = Wn @ nmf_q[r]
    pex = PCA(n_components=r, random_state=SEED).fit(DIG)
    recon_exact = pex.inverse_transform(pex.transform(DIG))
    # Before quantisation ICA is a rotation of PCA's basis, so the two land on
    # the same subspace and reach the SAME error, exactly. After rounding each
    # atom to 8 bits the two sets of perturbed vectors span subspaces that are
    # very slightly different, and per image that shows. Both numbers are
    # exported, because the claim the article makes is about the first and the
    # number the reader's browser can check is the second.
    pex_scores = pex.transform(DIG)
    ica_exact = ica_atoms[r]
    G = ica_exact @ ica_exact.T
    coef_i = np.linalg.solve(G, ica_exact @ (DIG - MEAN_IMG).T).T
    ei_exact = mse(DIG, coef_i @ ica_exact + MEAN_IMG)
    per_img_gap = float(np.abs(((DIG - recon_p) ** 2).mean(1)
                               - ((DIG - recon_i) ** 2).mean(1)).max())
    row = {
        "rank": r,
        "pca": rnd(ep, 8), "ica": rnd(ei, 8), "nmf": rnd(en, 8),
        "pca_unquantised": rnd(mse(DIG, recon_exact), 8),
        "ica_unquantised": rnd(ei_exact, 8),
        "ica_gap": float(f"{abs(ei - ep):.3e}"),
        "ica_gap_unquantised": float(f"{abs(ei_exact - mse(DIG, recon_exact)):.3e}"),
        "ica_gap_worst_image": float(f"{per_img_gap:.3e}"),
        "nmf_ratio": rnd(en / ep, 4),
        "variance_explained": rnd(float(pex.explained_variance_ratio_.sum()), 5),
        "negative_threshold": NEG,
        "pca_negative_pixels": rnd(float((recon_p < -NEG).mean()), 5),
        "pca_negative_unquantised": rnd(float((recon_exact < -NEG).mean()), 5),
        "pca_most_negative": rnd(float(recon_p.min()), 5),
        "nmf_negative_pixels": rnd(float((recon_n < -NEG).mean()), 5),
        "nmf_min_pixel": rnd(float(recon_n.min()), 8),
        "nmf_zero_coeffs": rnd(float((Wn < 1e-6).sum(1).mean()), 3),
        "pca_hoyer": rnd(float(np.mean([hoyer(a) for a in pca_q[:r]])), 4),
        "ica_hoyer": rnd(float(np.mean([hoyer(a) for a in ica_q[r]])), 4),
        "nmf_hoyer": rnd(float(np.mean([hoyer(a) for a in nmf_q[r]])), 4),
        "pca_top10": rnd(float(np.mean([top_mass(a) for a in pca_q[:r]])), 4),
        "ica_top10": rnd(float(np.mean([top_mass(a) for a in ica_q[r]])), 4),
        "nmf_top10": rnd(float(np.mean([top_mass(a) for a in nmf_q[r]])), 4),
        "ica_iters": ica_meta[r]["iters"], "ica_seed": ica_meta[r]["seed"],
        "ica_converged": ica_meta[r]["converged"], "ica_cond": ica_meta[r]["cond"],
        "nmf_iters": nmf_meta[r]["iters"], "nmf_converged": nmf_meta[r]["converged"],
    }
    errors.append(row)
    # The load-time guard: the browser must reproduce these per-image numbers.
    # Measured on show_q, the tiles as the browser DECODES them, not on the
    # originals, so the guard compares two fits of the same vector.
    per_image.append({
        "rank": r,
        "pca": [rnd(mse(q, fit_linear(q, pca_q[:r], mean_q) @ pca_q[:r] + mean_q), 8)
                for q in show_q],
        "ica": [rnd(mse(q, fit_linear(q, ica_q[r], mean_q) @ ica_q[r] + mean_q), 8)
                for q in show_q],
        "nmf": [rnd(mse(q, fit_nmf(q, nmf_q[r]) @ nmf_q[r]), 8) for q in show_q],
    })
    print(f"  rank {r:>2}: PCA {ep:.6f}  ICA {ei:.6f} (gap {abs(ei-ep):.1e})  "
          f"NMF {en:.6f} ({en/ep:.3f}x)   PCA ink below -{NEG} "
          f"{row['pca_negative_pixels']*100:5.2f}% (unrounded "
          f"{row['pca_negative_unquantised']*100:5.2f}%)"
          f"   NMF zero coefficients {row['nmf_zero_coeffs']:.2f}/{r}")

# nesting: PCA's rank-r atoms are the first r of the rank-25 set; nobody else's are
def best_match(A, B):
    Au = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bu = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.abs(Au @ Bu.T).max(1)


nest = []
for r in RANKS[:-1]:
    nxt = RANKS[RANKS.index(r) + 1]
    nest.append({
        "from": r, "to": nxt,
        "pca": rnd(float(best_match(PCA(n_components=r, random_state=SEED).fit(DIG).components_,
                                    PCA(n_components=nxt, random_state=SEED).fit(DIG).components_).mean()), 6),
        "ica": rnd(float(best_match(ica_atoms[r], ica_atoms[nxt]).mean()), 4),
        "nmf": rnd(float(best_match(nmf_atoms[r], nmf_atoms[nxt]).mean()), 4),
        "nmf_worst": rnd(float(best_match(nmf_atoms[r], nmf_atoms[nxt]).min()), 4),
    })
print("  nesting (mean best |cosine| of the smaller set inside the larger):")
for row in nest:
    print(f"    {row['from']:>2} -> {row['to']:<2}  PCA {row['pca']:.6f}  "
          f"ICA {row['ica']:.4f}  NMF {row['nmf']:.4f} (worst atom {row['nmf_worst']:.4f})")

# the same recipe on faces, where the famous parts figure comes from
print("\n  the same measurements on 400 faces, for the folklore check:")
FAC = fetch_olivetti_faces(shuffle=False).images.reshape(400, 32, 2, 32, 2).mean((2, 4))
FAC = np.round(FAC.reshape(400, -1), 4)
KF = 16
pf = PCA(n_components=KF, random_state=SEED).fit(FAC)
ef = mse(FAC, pf.inverse_transform(pf.transform(FAC)))
mf = NMF(n_components=KF, init="nndsvda", max_iter=20000, random_state=SEED, tol=1e-6)
Wf = mf.fit_transform(FAC)
enf = mse(FAC, Wf @ mf.components_)
faces_out = {
    "n": int(FAC.shape[0]), "side": 32, "rank": KF,
    "pca_mse": rnd(ef, 8), "nmf_mse": rnd(enf, 8), "nmf_ratio": rnd(enf / ef, 4),
    "nmf_iters": int(mf.n_iter_), "nmf_converged": int(mf.n_iter_ < 20000),
    "pca_top10": rnd(float(np.mean([top_mass(a) for a in pf.components_])), 4),
    "nmf_top10": rnd(float(np.mean([top_mass(a) for a in mf.components_])), 4),
    "pca_negative_pixels": rnd(float((pf.inverse_transform(pf.transform(FAC)) < -NEG).mean()), 5),
}
digits16 = next(r for r in errors if r["rank"] == KF)
print(f"    faces  rank {KF}: NMF costs {enf/ef:.3f}x PCA, atom top-10% mass "
      f"{faces_out['nmf_top10']:.4f} against PCA's {faces_out['pca_top10']:.4f}")
print(f"    digits rank {KF}: NMF costs {digits16['nmf_ratio']:.3f}x PCA, atom top-10% mass "
      f"{digits16['nmf_top10']:.4f} against PCA's {digits16['pca_top10']:.4f}")

atoms_out = {
    "ranks": RANKS, "tile": TILE, "n_images": int(DIG.shape[0]),
    "dead_pixels": int((DIG.var(0) == 0).sum()),
    "sprite_cols": int(np.ceil(np.sqrt(len(tiles)))),
    "index": index, "decode": decode,
    "showcase": [{"label": int(yd[i]), "tile": index[f"show{n}"]}
                 for n, i in enumerate(SHOW)],
    "errors": errors, "per_image": per_image, "nesting": nest, "faces": faces_out,
    "nmf_fit_iters": 400,
}

# ---- write the sprite
cols = atoms_out["sprite_cols"]
rows_n = int(np.ceil(len(tiles) / cols))
sheet = np.zeros((rows_n * TILE, cols * TILE), dtype=np.uint8)
for n, q in enumerate(tiles):
    r0, c0 = (n // cols) * TILE, (n % cols) * TILE
    sheet[r0:r0 + TILE, c0:c0 + TILE] = q.reshape(TILE, TILE)
try:
    from PIL import Image
    Image.fromarray(sheet, mode="L").save(OUT / "atoms.png", optimize=True)
except ImportError:
    import zlib
    import struct

    def chunk(kind, data):
        c = kind + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    raw = b"".join(b"\x00" + sheet[i].tobytes() for i in range(sheet.shape[0]))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", sheet.shape[1], sheet.shape[0], 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    (OUT / "atoms.png").write_bytes(png)
print(f"  wrote {OUT/'atoms.png'}  ({(OUT/'atoms.png').stat().st_size/1024:.1f} kB, "
      f"{cols}x{rows_n} tiles)")

# ===================================================================== write it
out = {
    "meta": {
        "seed": SEED,
        "wine_rounding": 3,
        "dial_step": DIAL_STEP,
        "pair": list(PAIR),
        "note": "every figure on the page is computed by this file; the browser "
                "recomputes the eigendecompositions, the rotation sweep and the "
                "reconstructions itself and checks against the references here",
    },
    "cloud": cloud,
    "spectrum": spectrum,
    "ica": ica_out,
    "atoms": atoms_out,
}
path = OUT / "directions.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size/1024:.1f} kB)")
