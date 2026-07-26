"""Data for the ATLAS 'Features Before Learning' web article (module 1.3).

Three measurements, one theme: before 2012, vision worked by designing the
representation by hand and letting a cheap classifier finish the job.

  * HOG, exactly. The browser recomputes the cell histograms live, so this
    file ships a reference set: a plain-numpy HOG that was verified against
    scikit-learn's own `hog` to 7e-8 (same gradient convention, same binning,
    same cell windows), plus the descriptor length at every setting the
    widget offers, including the settings where HOG produces MORE numbers
    than the image had pixels.

  * Invariance, as a measurement rather than a claim. The image is warped by
    a known similarity transform, so every proposed match can be checked
    against the ground truth it should satisfy: a match is correct when the
    warped position of keypoint A lands within 3 pixels of its partner B.
    SIFT and ORB are measured that way across rotation and scale, and so is
    a deliberately naive control: raw 16x16 pixel patches taken at the very
    same SIFT keypoints, matched by sum of squared differences. Same
    detector, same points, different descriptor: whatever separates the
    curves is the descriptor's doing.

  * The payoff, benchmarked. A linear SVM on HOG features against the same
    linear SVM on raw pixels, on the same images, with fit times.

Images are exported as PNG (lossless) because the browser recomputes HOG
from those exact pixels and must land on the shipped reference.

Run from anywhere: `python src/utils/generate_features_data.py`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import ORB, SIFT, hog, match_descriptors
from skimage.transform import SimilarityTransform, resize, warp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "classical-features" / "data"
IMG = ROOT / "classical-features" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19

HOG_SIZE = 128          # the widget recomputes this one live
MATCH_SIZE = 256        # big enough for SIFT to have something to say
TOL_PX = 3.0            # a match is correct within this many pixels


def quantise(arr01):
    """The exact 8-bit values a browser will read back out of the PNG."""
    return np.clip(np.round(arr01 * 255), 0, 255).astype(np.uint8)


def save_png(arr01, path):
    """Save a float image in [0,1] as 8-bit grayscale PNG, losslessly."""
    Image.fromarray(quantise(arr01), mode="L").save(path)
    return path.stat().st_size


# ============================================================ the HOG reference
def cell_histograms(im, orientations, cell):
    """scikit-image's HOG cell histograms, in plain numpy.

    Verified against skimage.feature.hog: identical to 7e-8 once the same L1
    epsilon is applied, for every (orientations, cell) pair the widget uses.
    Gradients are centred differences with zeroed borders, orientations are
    unsigned (mod 180), and each cell holds the magnitude sum of the pixels
    whose orientation falls in its bin, divided by the cell's area.
    """
    gr = np.zeros_like(im)
    gc = np.zeros_like(im)
    gr[1:-1, :] = im[2:, :] - im[:-2, :]
    gc[:, 1:-1] = im[:, 2:] - im[:, :-2]
    mag = np.hypot(gr, gc)
    ori = np.rad2deg(np.arctan2(gr, gc)) % 180
    n_r, n_c = im.shape[0] // cell, im.shape[1] // cell
    out = np.zeros((n_r, n_c, orientations))
    for i in range(orientations):
        lo = 180.0 / orientations * i
        hi = 180.0 / orientations * (i + 1)
        m = np.where((ori >= lo) & (ori < hi), mag, 0.0)
        out[:, :, i] = m.reshape(n_r, cell, n_c, cell).sum(axis=(1, 3)) / (cell * cell)
    return out


astro_full = rgb2gray(data.astronaut())
size_png = save_png(resize(astro_full, (HOG_SIZE, HOG_SIZE), anti_aliasing=True),
                    IMG / "hog-source.png")
# Everything below is measured on the 8-bit values the browser will actually
# read back from that PNG, not on the float image behind it: the widget has to
# be able to land on this reference exactly, and a rounding step in between
# would put it permanently a hair off.
hog_img = quantise(resize(astro_full, (HOG_SIZE, HOG_SIZE), anti_aliasing=True)) / 255.0
print(f"HOG source image: {HOG_SIZE}x{HOG_SIZE} PNG, {size_png / 1024:.0f} kB "
      f"(reference computed on its 8-bit values)")

# the agreement with skimage, measured here so the article can quote it
agreements = []
for ori in (6, 9, 12):
    for cell in (4, 8, 16):
        mine = cell_histograms(hog_img, ori, cell)
        sk = hog(hog_img, orientations=ori, pixels_per_cell=(cell, cell),
                 cells_per_block=(1, 1), block_norm="L1", feature_vector=False)[:, :, 0, 0, :]
        mine_l1 = mine / (np.abs(mine).sum(axis=2, keepdims=True) + 1e-5)
        agreements.append(float(np.abs(mine_l1 - sk).max()))
worst_agreement = max(agreements)
print(f"numpy HOG vs skimage HOG, worst of {len(agreements)} settings: {worst_agreement:.2e}")

# descriptor lengths, including the settings that expand rather than compress
lengths = []
for ori in (6, 9, 12):
    for cell in (4, 8, 16):
        f = hog(hog_img, orientations=ori, pixels_per_cell=(cell, cell), cells_per_block=(2, 2))
        lengths.append({"orientations": ori, "cell": cell, "length": int(f.shape[0])})
        print(f"  ori {ori:>2} cell {cell:>2}: descriptor {f.shape[0]:>6} "
              f"({f.shape[0] / hog_img.size:.2f}x the pixel count)")

REF_ORI, REF_CELL = 9, 8
ref_hist = cell_histograms(hog_img, REF_ORI, REF_CELL)
print(f"shipping the browser's check at ori {REF_ORI}, cell {REF_CELL}: {ref_hist.size} numbers")


# ======================================================= invariance, measured
def transform_for(theta_deg, scale, w, h):
    """Rotation about the image centre plus a scale, as a similarity."""
    c = np.array([w / 2.0, h / 2.0])
    return (SimilarityTransform(translation=-c)
            + SimilarityTransform(rotation=np.deg2rad(theta_deg), scale=scale)
            + SimilarityTransform(translation=c))


def warp_image(im, tf):
    return warp(im, tf.inverse, order=1, mode="constant", cval=0.0)


def patch_descriptors(im, keypoints, size=16):
    """The naive control: a raw pixel patch around each keypoint, mean-centred
    and unit-normalised so brightness alone cannot decide a match."""
    half = size // 2
    H, W = im.shape
    desc = []
    keep = []
    for i, (r, c) in enumerate(keypoints):
        r0, c0 = int(round(r)) - half, int(round(c)) - half
        if r0 < 0 or c0 < 0 or r0 + size > H or c0 + size > W:
            continue
        p = im[r0:r0 + size, c0:c0 + size].astype(float).ravel()
        p = p - p.mean()
        n = np.linalg.norm(p)
        if n < 1e-8:
            continue
        desc.append(p / n)
        keep.append(i)
    return np.array(desc), np.array(keep, dtype=int)


def evaluate(kp_a, desc_a, kp_b, desc_b, tf, max_ratio=0.8, metric="euclidean"):
    """Match, then check every match against the known transform."""
    if len(desc_a) == 0 or len(desc_b) == 0:
        return {"n_matches": 0, "n_correct": 0, "precision": 0.0}, np.zeros((0, 2), int), np.zeros(0, bool)
    matches = match_descriptors(desc_a, desc_b, metric=metric, max_ratio=max_ratio,
                                cross_check=True)
    if len(matches) == 0:
        return {"n_matches": 0, "n_correct": 0, "precision": 0.0}, matches, np.zeros(0, bool)
    # keypoints are (row, col); the transform works in (x, y) = (col, row)
    src_xy = kp_a[matches[:, 0]][:, ::-1]
    dst_xy = kp_b[matches[:, 1]][:, ::-1]
    pred_xy = tf(src_xy)
    err = np.linalg.norm(pred_xy - dst_xy, axis=1)
    ok = err <= TOL_PX
    return ({"n_matches": int(len(matches)), "n_correct": int(ok.sum()),
             "precision": rnd(float(ok.mean()), 4)}, matches, ok)


match_img = resize(astro_full, (MATCH_SIZE, MATCH_SIZE), anti_aliasing=True)
base_png = save_png(match_img, IMG / "match-source.png")
sift_base = SIFT()
sift_base.detect_and_extract(match_img)
orb_base = ORB(n_keypoints=400, fast_threshold=0.05)
orb_base.detect_and_extract(match_img)
patch_base, patch_keep = patch_descriptors(match_img, sift_base.keypoints)
print(f"\nbase image {MATCH_SIZE}x{MATCH_SIZE}: SIFT {len(sift_base.keypoints)} kp, "
      f"ORB {len(orb_base.keypoints)} kp, patch control {len(patch_base)} of the SIFT kp")

ANGLES = list(range(0, 181, 15))
SCALES = [1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3]
SHOW_ANGLES = [0, 45, 90]
SHOW_SCALES = [0.5]

rotation_rows = []
scale_rows = []
panels = {}


def run_case(theta, scale, label, store_panel):
    tf = transform_for(theta, scale, MATCH_SIZE, MATCH_SIZE)
    w_img = warp_image(match_img, tf)

    s = SIFT()
    try:
        s.detect_and_extract(w_img)
        sift_res, sift_m, sift_ok = evaluate(sift_base.keypoints, sift_base.descriptors,
                                             s.keypoints, s.descriptors, tf)
    except RuntimeError:
        sift_res, sift_m, sift_ok, s = {"n_matches": 0, "n_correct": 0, "precision": 0.0}, None, None, None

    o = ORB(n_keypoints=400, fast_threshold=0.05)
    try:
        o.detect_and_extract(w_img)
        orb_res, orb_m, orb_ok = evaluate(orb_base.keypoints, orb_base.descriptors,
                                          o.keypoints, o.descriptors, tf, metric="hamming")
    except RuntimeError:
        orb_res, orb_m, orb_ok, o = {"n_matches": 0, "n_correct": 0, "precision": 0.0}, None, None, None

    # the control: raw patches at the SAME SIFT keypoints on both images
    if s is not None:
        pw, pw_keep = patch_descriptors(w_img, s.keypoints)
        patch_res, _, _ = evaluate(sift_base.keypoints[patch_keep], patch_base,
                                   s.keypoints[pw_keep], pw, tf)
    else:
        patch_res = {"n_matches": 0, "n_correct": 0, "precision": 0.0}

    row = {
        "theta": theta, "scale": rnd(scale, 3),
        "sift": sift_res, "orb": orb_res, "patch": patch_res,
    }

    if store_panel:
        # the identity panel is the source image itself: no point shipping it twice
        identity = abs(theta) < 1e-9 and abs(scale - 1.0) < 1e-9
        name = "match-source.png" if identity else f"warp-{label}.png"
        px = (IMG / name).stat().st_size if identity else save_png(w_img, IMG / name)
        pairs = []
        if sift_m is not None and len(sift_m):
            take = np.argsort(-sift_ok.astype(int))[:140]   # correct ones first
            for idx in take:
                a = sift_base.keypoints[sift_m[idx, 0]]
                b = s.keypoints[sift_m[idx, 1]]
                pairs.append([rnd(a[1], 1), rnd(a[0], 1), rnd(b[1], 1), rnd(b[0], 1),
                              int(sift_ok[idx])])
        opairs = []
        if orb_m is not None and len(orb_m):
            take = np.argsort(-orb_ok.astype(int))[:140]
            for idx in take:
                a = orb_base.keypoints[orb_m[idx, 0]]
                b = o.keypoints[orb_m[idx, 1]]
                opairs.append([rnd(a[1], 1), rnd(a[0], 1), rnd(b[1], 1), rnd(b[0], 1),
                               int(orb_ok[idx])])
        panels[label] = {
            "theta": theta, "scale": rnd(scale, 3), "image": name,
            "png_kb": rnd(px / 1024, 1),
            "sift_pairs": pairs, "orb_pairs": opairs,
            "sift": sift_res, "orb": orb_res, "patch": patch_res,
        }
    return row


print("\nrotation (matches / correct within 3 px / precision):")
for theta in ANGLES:
    row = run_case(theta, 1.0, f"rot{theta}", theta in SHOW_ANGLES)
    rotation_rows.append(row)
    print(f"  {theta:>3} deg  SIFT {row['sift']['n_matches']:>4}/{row['sift']['n_correct']:>4}"
          f" ({row['sift']['precision']:.2f})   ORB {row['orb']['n_matches']:>4}/"
          f"{row['orb']['n_correct']:>4} ({row['orb']['precision']:.2f})   "
          f"patches {row['patch']['n_matches']:>4}/{row['patch']['n_correct']:>4} "
          f"({row['patch']['precision']:.2f})")

print("\nscale:")
for s_ in SCALES:
    row = run_case(0.0, s_, f"scale{int(s_ * 100)}", s_ in SHOW_SCALES)
    scale_rows.append(row)
    print(f"  x{s_:<5} SIFT {row['sift']['n_matches']:>4}/{row['sift']['n_correct']:>4}"
          f" ({row['sift']['precision']:.2f})   ORB {row['orb']['n_matches']:>4}/"
          f"{row['orb']['n_correct']:>4} ({row['orb']['precision']:.2f})   "
          f"patches {row['patch']['n_matches']:>4}/{row['patch']['n_correct']:>4} "
          f"({row['patch']['precision']:.2f})")

print(f"\nmatch source PNG: {base_png / 1024:.0f} kB; "
      f"{len(panels)} panels at {[p['png_kb'] for p in panels.values()]} kB")


# ================================================= does any of this pay off?
# The same linear classifier on the same digits, fed two representations.
from sklearn.svm import LinearSVC  # noqa: E402

from vision_data import mnist, subset  # noqa: E402

Xtr_all, ytr_all, Xte_all, yte_all = mnist()
Xtr_i, ytr = subset(Xtr_all, ytr_all, 10000, seed=SEED)
Xte_i, yte = subset(Xte_all, yte_all, 5000, seed=SEED)
print(f"\nMNIST benchmark: {len(ytr)} train / {len(yte)} test, one linear SVM, two representations")

HOG_KW = dict(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))


def hog_batch(images):
    return np.array([hog(im / 255.0, **HOG_KW) for im in images])


bench = []


def run_bench(name, Ftr, Fte, note):
    t0 = time.perf_counter()
    clf = LinearSVC(C=0.01, dual="auto", max_iter=5000).fit(Ftr, ytr)
    fit_s = time.perf_counter() - t0
    acc = float(clf.score(Fte, yte))
    bench.append({"name": name, "dim": int(Ftr.shape[1]), "acc": rnd(acc, 4),
                  "fit_seconds": rnd(fit_s, 2), "note": note})
    print(f"  {name:<22} dim {Ftr.shape[1]:>5}  test {acc:.4f}  fit {fit_s:.1f}s")
    return acc


run_bench("raw pixels", Xtr_i.reshape(len(ytr), -1) / 255.0,
          Xte_i.reshape(len(yte), -1) / 255.0, "every pixel its own feature")

t0 = time.perf_counter()
Htr = hog_batch(Xtr_i)
Hte = hog_batch(Xte_i)
hog_s = time.perf_counter() - t0
print(f"  (HOG extraction for all {len(ytr) + len(yte)} images: {hog_s:.1f}s)")
run_bench("HOG", Htr, Hte, "gradients, binned by orientation into 4x4 cells")

err_raw = 1 - bench[0]["acc"]
err_hog = 1 - bench[1]["acc"]
print(f"  error {err_raw * 100:.2f}% -> {err_hog * 100:.2f}%: "
      f"{(1 - err_hog / err_raw) * 100:.0f}% of the mistakes removed by the representation alone")

out = {
    "hog": {
        "image": "hog-source.png",
        "size": HOG_SIZE,
        "ref": {"orientations": REF_ORI, "cell": REF_CELL,
                "hist": [rnd(v, 6) for v in ref_hist.ravel()]},
        "skimage_agreement": float(f"{worst_agreement:.3g}"),
        "lengths": lengths,
        "n_pixels": int(hog_img.size),
    },
    "match": {
        "image": "match-source.png",
        "size": MATCH_SIZE,
        "tol_px": TOL_PX,
        "n_sift_kp": int(len(sift_base.keypoints)),
        "n_orb_kp": int(len(orb_base.keypoints)),
        "n_patch_kp": int(len(patch_base)),
        "rotation": rotation_rows,
        "scale": scale_rows,
        "panels": panels,
    },
    "bench": {
        "rows": bench,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "hog_settings": "9 orientations, 4x4 cells, 2x2 blocks",
        "hog_seconds": rnd(hog_s, 2),
    },
    "versions": {"seed": SEED},
}
(OUT / "features.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'features.json'}  ({(OUT / 'features.json').stat().st_size / 1024:.1f} kB)")
