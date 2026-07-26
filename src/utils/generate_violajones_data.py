"""Data for the ATLAS 'Viola-Jones' web article (module 1.3).

Three ideas made face detection run in real time on 2001 hardware, and all
three are measured here rather than described:

  * Haar-like features: differences of rectangle sums, as cheap as a visual
    feature gets.
  * The integral image: any rectangle's sum in four lookups whatever its
    size, so a large feature costs exactly what a small one costs.
  * Boosting plus a cascade: pick the few useful features out of a hundred
    thousand, then order them so that most windows are thrown out after one
    or two tests.

The classifier is trained here from scratch (AdaBoost over decision stumps,
vectorised in numpy). Positives are the 100 face crops that ship with
scikit-image plus their mirror images; negatives are 2,000 patches mined at
random from ten face-free photographs, because the 100 non-face crops in the
same dataset are so unlike faces that a single feature separates them and
there is no story left to tell. Feature values follow scikit-image's sign
convention exactly (verified below), so the browser, which recomputes
everything live from a sprite of the same images, inherits a checked
convention.

The cascade is then run over every 25x25 window of a real photograph, which
is where its economy stops being a claim about arithmetic and becomes a
count.

Run from anywhere: `python src/utils/generate_violajones_data.py`.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image as sk_integral, resize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "viola-jones" / "data"
IMG = ROOT / "viola-jones" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19
N_POOL = 12000
N_ROUNDS = 40
TYPES = ["type-2-x", "type-2-y", "type-3-x", "type-3-y"]
N_NEG = 2000

rs = np.random.RandomState(SEED)

# ----------------------------------------------------------------- the data
lfw = data.lfw_subset()                       # (200, 25, 25), faces first
faces = lfw[:100]
N = faces.shape[1]
positives = np.concatenate([faces, faces[:, :, ::-1]])       # plus mirrors
print(f"positives: {len(positives)} ({len(faces)} crops and their mirrors), {N}x{N}")

BACKDROPS = ["brick", "coins", "grass", "gravel", "moon", "page", "text",
             "rocket", "hubble_deep_field", "checkerboard"]
negatives = []
per = N_NEG // len(BACKDROPS)
for name in BACKDROPS:
    im = getattr(data, name)()
    if im.ndim == 3:
        im = rgb2gray(im)
    im = im.astype(float)
    if im.max() > 1.5:
        im = im / 255.0
    H, W = im.shape
    for _ in range(per):
        r = rs.randint(0, H - N)
        c = rs.randint(0, W - N)
        negatives.append(im[r:r + N, c:c + N])
negatives = np.array(negatives)
print(f"negatives: {len(negatives)} patches mined from {len(BACKDROPS)} face-free photographs")

X_img = np.concatenate([positives, negatives])
y = np.concatenate([np.ones(len(positives), int), np.zeros(len(negatives), int)])
perm = rs.permutation(len(y))
X_img, y = X_img[perm], y[perm]
# Three splits, not two. The boosting learns on the first; the cascade's stage
# thresholds are set on the second, because a threshold is itself something
# fitted and setting it on the data it will be judged on is the same mistake
# the calibration article spent an entire page on; the third is untouched.
c1, c2 = int(0.60 * len(y)), int(0.80 * len(y))
train_i, calib_i, test_i = np.arange(c1), np.arange(c1, c2), np.arange(c2, len(y))
print(f"train {len(train_i)} ({int(y[train_i].sum())} faces) / "
      f"calibration {len(calib_i)} ({int(y[calib_i].sum())} faces) / "
      f"test {len(test_i)} ({int(y[test_i].sum())} faces)")


# ------------------------------------------------- integral images, our own
def integral(img):
    """Summed-area table with a zero row and column, so the four-lookup
    formula needs no special case at the edges."""
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return ii


def rect_sum(ii, r0, c0, r1, c1):
    """Sum over the inclusive rectangle [r0..r1] x [c0..c1] in four lookups."""
    return ii[r1 + 1, c1 + 1] - ii[r0, c1 + 1] - ii[r1 + 1, c0] + ii[r0, c0]


_check = integral(X_img[0])
assert abs(rect_sum(_check, 3, 5, 17, 20) - X_img[0][3:18, 5:21].sum()) < 1e-9
print("four-lookup rectangle sum verified against the direct sum")

# --------------------------------------------------------- the feature pool
coords_all, types_all = haar_like_feature_coord(N, N, TYPES)
pool_idx = np.sort(rs.choice(len(coords_all), N_POOL, replace=False))
coords = coords_all[pool_idx]
ftypes = types_all[pool_idx]
print(f"\nfeature pool: {N_POOL:,} drawn from the {len(coords_all):,} that exist at {N}x{N}")

# Precompute each feature as (rect, sign) so evaluation is a flat loop; the sign
# convention is scikit-image's, where the FIRST rectangle is subtracted.
flat = []
for coord in coords:
    rects = []
    for k, ((r0, c0), (r1, c1)) in enumerate(coord):
        rects.append((int(r0), int(c0), int(r1), int(c1), -1.0 if k % 2 == 0 else 1.0))
    flat.append(rects)


def feature_matrix(images):
    iis = [integral(im) for im in images]
    out = np.zeros((len(images), len(flat)), dtype=np.float32)
    for j, rects in enumerate(flat):
        acc = np.zeros(len(images))
        for (r0, c0, r1, c1, s) in rects:
            acc += s * np.array([rect_sum(ii, r0, c0, r1, c1) for ii in iis])
        out[:, j] = acc
    return out


t0 = time.perf_counter()
X = feature_matrix(X_img)
print(f"feature matrix {X.shape} in {time.perf_counter() - t0:.1f}s")

sk_vals = haar_like_feature(sk_integral(X_img[0]), 0, 0, N, N,
                            feature_type=ftypes, feature_coord=coords)
gap = float(np.abs(sk_vals - X[0]).max())
scale = float(np.abs(sk_vals).max())
rel = gap / scale
print(f"our Haar values vs scikit-image's, worst of {N_POOL:,}: {gap:.2e} absolute, "
      f"{rel:.1e} relative (the matrix is float32, so this is its rounding)")
assert rel < 1e-5, "sign convention drifted from scikit-image"


# ------------------------------------------------ AdaBoost over stumps, ours
Xtr, ytr = X[train_i], y[train_i]
Xca, yca = X[calib_i], y[calib_i]
Xte, yte = X[test_i], y[test_i]
ORDER = np.argsort(Xtr, axis=0, kind="stable").astype(np.int32)   # once, not per round
YS = ytr[ORDER]


def best_stump(w):
    """The weighted-error-minimising (feature, threshold, polarity) over every
    feature at once: with the values pre-sorted, the error of every candidate
    cut is two cumulative sums."""
    ws = w[ORDER]
    pos = np.cumsum(ws * (YS == 1), axis=0)
    neg = np.cumsum(ws * (YS == 0), axis=0)
    tot_pos, tot_neg = pos[-1], neg[-1]
    err_a = pos + (tot_neg - neg)          # predict face when value > cut
    err_b = neg + (tot_pos - pos)          # the opposite polarity
    err = np.minimum(err_a, err_b)
    k, j = np.unravel_index(np.argmin(err), err.shape)
    polarity = 1 if err_a[k, j] <= err_b[k, j] else -1
    col = Xtr[ORDER[:, j], j]
    thr = float(0.5 * (col[k] + col[k + 1])) if k + 1 < len(col) else float(col[k] + 1e-6)
    return int(j), thr, int(polarity), float(err[k, j])


def stump(col, thr, polarity):
    return ((col > thr) if polarity == 1 else (col <= thr)).astype(int)


w = np.where(ytr == 1, 0.5 / max(ytr.sum(), 1), 0.5 / max((1 - ytr).sum(), 1))
learners = []
t0 = time.perf_counter()
for t in range(N_ROUNDS):
    j, thr, pol, err = best_stump(w)
    err = min(max(err, 1e-10), 1 - 1e-10)
    alpha = 0.5 * np.log((1 - err) / err)
    miss = stump(Xtr[:, j], thr, pol) != ytr
    w = w * np.exp(alpha * np.where(miss, 1.0, -1.0))
    w /= w.sum()
    learners.append({"feature": j, "threshold": thr, "polarity": pol, "alpha": float(alpha),
                     "weighted_error": float(err)})
print(f"\nAdaBoost: {N_ROUNDS} rounds in {time.perf_counter() - t0:.1f}s")


def score(Xa, upto):
    s = np.zeros(len(Xa))
    for L in learners[:upto]:
        s += L["alpha"] * np.where(stump(Xa[:, L["feature"]], L["threshold"], L["polarity"]) == 1, 1.0, -1.0)
    return s


def rates(s, yy, thr=0.0):
    pred = s > thr
    det = float(pred[yy == 1].mean())
    fp = float(pred[yy == 0].mean())
    return det, fp


curve = []
for k in [1, 2, 3, 5, 10, 20, 30, 40]:
    d_tr, f_tr = rates(score(Xtr, k), ytr)
    d_te, f_te = rates(score(Xte, k), yte)
    curve.append({"n": k, "det": rnd(d_te, 4), "fp": rnd(f_te, 4),
                  "det_train": rnd(d_tr, 4), "fp_train": rnd(f_tr, 4),
                  "acc": rnd(float(((score(Xte, k) > 0).astype(int) == yte).mean()), 4)})
    print(f"  {k:>2} features: test detection {d_te:.3f}, false positives {f_te:.4f}, "
          f"accuracy {curve[-1]['acc']:.4f}")

acc1 = curve[0]["acc"]

# ------------------------------------------------------------- the cascade
STAGE_DETECTION = 0.98      # what each stage promises to keep, on calibration faces
STAGE_MAX_PASS = 0.5        # and how much of the surviving background it may pass on
stages = []
cum = 0
alive_ca = np.ones(len(yca), dtype=bool)
print(f"\nbuilding the cascade the way the paper does: grow a stage until it keeps "
      f"{STAGE_DETECTION:.0%} of the calibration faces while passing at most "
      f"{STAGE_MAX_PASS:.0%} of the background still standing")
while cum < len(learners):
    size = 0
    while cum < len(learners):
        cum += 1
        size += 1
        s_ca = score(Xca, cum)
        face_scores = np.sort(s_ca[(yca == 1) & alive_ca])
        k = int(np.floor((1 - STAGE_DETECTION) * len(face_scores)))
        # A threshold has to sit BETWEEN two achievable scores, not a hair
        # under one of them: a few stumps only ever sum to a handful of
        # distinct values, and "achievable minus 1e-9" stops being that once
        # the number is written to JSON with six decimals, which silently
        # moves every window sitting exactly on that score to the other side.
        keep = float(face_scores[k])
        below = s_ca[s_ca < keep - 1e-12]
        thr = float((keep + below.max()) / 2) if below.size else keep - 1.0
        negs = alive_ca & (yca == 0)
        fp = float((s_ca[negs] > thr).mean()) if negs.any() else 0.0
        if fp <= STAGE_MAX_PASS:
            break
    # the last stage is the ensemble's own verdict, so the cascade can only
    # ever return a subset of what the full classifier would have said
    if cum >= len(learners):
        thr = 0.0
    det_ca = float((s_ca[(yca == 1) & alive_ca] > thr).mean())
    passed = s_ca > thr
    n_neg_before = int((alive_ca & (yca == 0)).sum())
    alive_ca = alive_ca & passed
    n_neg_after = int((alive_ca & (yca == 0)).sum())
    stages.append({"upto": cum, "threshold": thr, "size": size})
    print(f"  stage {len(stages)}: {size:>2} features (total {cum:>2}), keeps {det_ca:.3f} of its faces, "
          f"background {n_neg_before:>4} -> {n_neg_after:>4}")
    if n_neg_after == 0:
        break

alive = np.ones(len(yte), dtype=bool)
evaluated = np.zeros(len(yte))
cascade_rows = []
prev = 0
for st in stages:
    s = score(Xte, st["upto"])
    evaluated[alive] += st["upto"] - prev
    before_f = int((alive & (yte == 1)).sum())
    before_n = int((alive & (yte == 0)).sum())
    alive = alive & (s > st["threshold"])
    cascade_rows.append({"upto": st["upto"], "size": st["size"],
                         "faces_in": before_f, "faces_out": int((alive & (yte == 1)).sum()),
                         "neg_in": before_n, "neg_out": int((alive & (yte == 0)).sum())})
    print(f"  stage of {st['size']:>2} (total {st['upto']:>2}): "
          f"non-faces {before_n:>4} -> {cascade_rows[-1]['neg_out']:>4}, "
          f"faces {before_f:>3} -> {cascade_rows[-1]['faces_out']:>3}")
    prev = st["upto"]

full_cost = stages[-1]["upto"]
cas_det = float(alive[yte == 1].mean())
cas_fp = float(alive[yte == 0].mean())
mono_det, mono_fp = rates(score(Xte, full_cost), yte)
print(f"\n  cascade:    detection {cas_det:.3f}, false positives {cas_fp:.4f}, "
      f"{evaluated.mean():.2f} features per window ({evaluated[yte == 0].mean():.2f} per non-face)")
print(f"  monolithic: detection {mono_det:.3f}, false positives {mono_fp:.4f}, "
      f"{full_cost} features per window")
print(f"  speedup on background windows: {full_cost / max(evaluated[yte == 0].mean(), 1e-9):.1f}x")

# ------------------------------------------- the cascade over a real photo
# Her face is about 100 px across in the 512 px original, so a quarter-size
# copy puts it at roughly the 25 px the detector was trained on.
scan = resize(rgb2gray(data.astronaut()), (128, 128), anti_aliasing=True)
scan_u8 = np.clip(np.round(scan * 255), 0, 255).astype(np.uint8)
Image.fromarray(scan_u8, mode="L").save(IMG / "scan.png")
scan_exact = scan_u8 / 255.0
S = scan_exact.shape[0]
positions = [(r, c) for r in range(S - N + 1) for c in range(S - N + 1)]
windows = np.array([scan_exact[r:r + N, c:c + N] for r, c in positions])
print(f"\nscanning a {S}x{S} photograph: {len(windows):,} windows of {N}x{N}")

t0 = time.perf_counter()
# Only the features the cascade actually uses: computing all 12,000 for every
# window would take three minutes to answer a question that needs 40 of them.
used = sorted({L["feature"] for L in learners})
used_pos = {j: k for k, j in enumerate(used)}
Xw_small = np.zeros((len(windows), len(used)), dtype=np.float32)
iis_w = [integral(im) for im in windows]
for k, j in enumerate(used):
    acc = np.zeros(len(windows))
    for (r0, c0, r1, c1, s) in flat[j]:
        acc += s * np.array([rect_sum(ii, r0, c0, r1, c1) for ii in iis_w])
    Xw_small[:, k] = acc


def score_small(upto):
    s = np.zeros(len(windows))
    for L in learners[:upto]:
        col = Xw_small[:, used_pos[L["feature"]]]
        s += L["alpha"] * np.where(stump(col, L["threshold"], L["polarity"]) == 1, 1.0, -1.0)
    return s


alive_w = np.ones(len(windows), dtype=bool)
evaluated_w = np.zeros(len(windows))
scan_rows = []
prev = 0
for st in stages:
    s = score_small(st["upto"])
    evaluated_w[alive_w] += st["upto"] - prev
    before = int(alive_w.sum())
    alive_w = alive_w & (s > st["threshold"])
    scan_rows.append({"upto": st["upto"], "size": st["size"],
                      "in": before, "out": int(alive_w.sum())})
    print(f"  stage of {st['size']:>2}: {before:>6,} windows in -> {int(alive_w.sum()):>6,} out "
          f"({100 * (1 - alive_w.sum() / max(before, 1)):.1f}% rejected)")
    prev = st["upto"]
final_scores = score_small(full_cost)
best_w = int(np.argmax(np.where(alive_w, final_scores, -np.inf))) if alive_w.any() else int(np.argmax(final_scores))
print(f"  {int(alive_w.sum())} windows survive; average {evaluated_w.mean():.2f} features per window "
      f"instead of {full_cost} ({time.perf_counter() - t0:.1f}s of feature extraction here)")
print(f"  best-scoring surviving window at row {positions[best_w][0]}, col {positions[best_w][1]}")

# The claim a cascade actually makes: same verdicts, less work. Compare its
# survivors with what the full ensemble says about every window.
mono_pos = final_scores > 0
both = int((mono_pos & alive_w).sum())
only_mono = int((mono_pos & ~alive_w).sum())
only_cas = int((~mono_pos & alive_w).sum())
print(f"  full ensemble on every window: {int(mono_pos.sum())} positives; "
      f"cascade and ensemble agree on {both}, ensemble-only {only_mono}, cascade-only {only_cas}")

survivors = [[int(positions[i][0]), int(positions[i][1]), rnd(final_scores[i], 3)]
             for i in np.where(alive_w)[0]]

# --------------------------------------------------------------- exports
sprite_cols, sprite_rows = 20, 10
gallery = np.concatenate([faces[:100], negatives[:100]])
gal_y = np.array([1] * 100 + [0] * 100)
sprite = np.zeros((sprite_rows * N, sprite_cols * N))
for i in range(len(gallery)):
    r, c = divmod(i, sprite_cols)
    sprite[r * N:(r + 1) * N, c * N:(c + 1) * N] = gallery[i]
Image.fromarray(np.clip(np.round(sprite * 255), 0, 255).astype(np.uint8), mode="L").save(
    IMG / "faces-sprite.png")
print(f"\nsprite: {(IMG / 'faces-sprite.png').stat().st_size / 1024:.0f} kB")

mean_face = faces.mean(axis=0)
Image.fromarray(np.clip(np.round(mean_face * 255), 0, 255).astype(np.uint8), mode="L").save(
    IMG / "mean-face.png")

demo = np.clip(np.round(faces[0] * 255), 0, 255).astype(int)

# Every learner the cascade uses, so the browser can run the real thing: the
# rectangles are in window coordinates, which is all a sliding window needs.
all_learners = []
for L in learners:
    j = L["feature"]
    all_learners.append({
        "rects": [[int(r0), int(c0), int(r1), int(c1)] for (r0, c0), (r1, c1) in coords[j]],
        "type": str(ftypes[j]),
        "threshold": rnd(L["threshold"], 6),
        "polarity": L["polarity"],
        "alpha": rnd(L["alpha"], 4),
        "weighted_error": rnd(L["weighted_error"], 4),
    })
top = all_learners[:10]

out = {
    "n": int(N),
    "sprite": {"image": "faces-sprite.png", "cols": sprite_cols, "rows": sprite_rows,
               "tile": int(N), "labels": [int(v) for v in gal_y]},
    "mean_face": "mean-face.png",
    "pool": {"enumerated": int(len(coords_all)), "sampled": N_POOL, "types": TYPES},
    "train": {"n": int(len(train_i)), "faces": int(ytr.sum()),
              "n_calib": int(len(calib_i)), "faces_calib": int(yca.sum()),
              "n_test": int(len(test_i)), "faces_test": int(yte.sum()),
              "n_negatives_mined": int(len(negatives)), "backdrops": len(BACKDROPS),
              "stage_detection_target": STAGE_DETECTION, "stage_pass_target": STAGE_MAX_PASS},
    "skimage_gap": float(f"{gap:.3g}"),
    "top_features": top,
    "learners": all_learners,
    "curve": curve,
    "single_feature_acc": acc1,
    "cascade": {
        "stages": [{"upto": s["upto"], "size": s["size"], "threshold": rnd(s["threshold"], 6)}
                   for s in stages],
        "rows": cascade_rows,
        "detection": rnd(cas_det, 4), "false_positive": rnd(cas_fp, 5),
        "mono_detection": rnd(mono_det, 4), "mono_false_positive": rnd(mono_fp, 5),
        "mean_evaluated": rnd(evaluated.mean(), 2),
        "mean_evaluated_neg": rnd(evaluated[yte == 0].mean(), 2),
        "full_cost": int(full_cost),
    },
    "scan": {
        "image": "scan.png", "size": int(S), "window": int(N),
        "n_windows": int(len(windows)),
        "rows": scan_rows,
        "mean_evaluated": rnd(evaluated_w.mean(), 2),
        "survivors": survivors,
        "best": [int(positions[best_w][0]), int(positions[best_w][1]), rnd(final_scores[best_w], 3)],
        "agreement": {"ensemble_positive": int(mono_pos.sum()), "both": both,
                      "ensemble_only": only_mono, "cascade_only": only_cas},
    },
    "demo": {"pixels": [[int(v) for v in row] for row in demo]},
    "versions": {"seed": SEED, "rounds": N_ROUNDS},
}
(OUT / "cascade.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"wrote {OUT / 'cascade.json'}  ({(OUT / 'cascade.json').stat().st_size / 1024:.1f} kB)")
