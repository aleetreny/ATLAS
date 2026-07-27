"""Data for the ATLAS 'Which Pixels' web article (module 1.3): U-Net, Mask R-CNN, SAM.

The detection article asked WHERE and answered with a box. A box is a coarse
answer: it says an object is somewhere inside a rectangle, and on these
garments a perfect rectangle is mostly not the garment. This article asks
which pixels, and measures what that costs and what it buys.

It runs on the SAME canvases as the detection article, rebuilt here with the
same seeds and asserted byte for byte against that article's cached splits.
That matters for one reason: because the garments are pasted by this file, the
mask of each object is known exactly, for free, in the same way its box was.
Compositing is a pixel-wise maximum, so during compositing an owner map is
recorded (which object won each pixel), and from it every ground truth on this
page is exact: the semantic map, the per-instance masks, and therefore IoU and
Dice, which are counted rather than estimated.

Measured here:
  * S0 the masks: the owner map, the amodal mask against the visible mask, and
    the guard that these are article 23's pictures and article 23's boxes.
  * S1 the metric: IoU and Dice on real masks, with the identity
    Dice = 2 IoU / (1 + IoU) checked on every pair rather than quoted; plus
    what a PERFECT box scores when read as a mask, per class, which is the
    number that says why this article exists.
  * S2 the U-Net: an encoder-decoder with skip connections, ablated by
    ZEROING the skip tensors rather than removing them, so parameter count,
    multiply-accumulates and every layer shape are identical across arms and
    the only thing that changes is whether encoder detail can reach the
    decoder. The error is then decomposed into a boundary band and an
    interior, because that is where the claim about skips lives.
  * S3 semantic against instance: how many pairs of touching same-class
    objects a semantic map merges into one blob, counted on the ground truth
    and on the prediction. This is the thing a per-pixel classifier cannot fix
    with more training.
  * S4 Mask R-CNN: article 23's trained detector, loaded from its own cache,
    plus a mask head on RoI-aligned features. Mask AP against box AP on the
    same detections, and the resolution ceiling declared in advance.
  * S5 the idea behind SAM, at the only scale this page can honestly test: a
    promptable, class-agnostic head trained on the five garment classes and
    evaluated on the five classes it has NEVER seen, which are already in
    these pictures as distractors. This is not SAM and the article says so.
  * S6 the export: the promptable network folded, quantised and shipped, with
    reference outputs computed after quantisation so the browser reproduces
    the generator rather than approximating it.

Run from anywhere: `python src/utils/generate_segmentation_data.py`.
"""
import base64
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize as skresize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_data import fashion_mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "segmentation" / "data"
IMG = ROOT / "segmentation" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

# ------------------------------------------------------------------ constants
# Every one of these is article 23's, because the pictures have to be the same
# pictures. Changing any of them breaks the guard in S0 on purpose.
SEED = 19
CANVAS = 64
INK = 32
MIN_SIDE, MAX_SIDE = 16, 41
N_TRAIN = 6000
N_VAL = 1500
N_TEST_MAX = 2500
MAX_OBJ = 3
BATCH = 64

DET_CLASSES = ["trouser", "dress", "t-shirt", "bag", "sneaker"]
DET_SOURCE = [1, 3, 0, 8, 7]
HELD_CLASSES = ["pullover", "coat", "shirt", "sandal", "ankle boot"]
HELD_SOURCE = [2, 4, 6, 5, 9]
N_CLS = len(DET_CLASSES)
N_SEM = N_CLS + 1                 # background is class 0 in the semantic map

DIST_ID = 1000                    # owner ids at or above this are distractors
EPOCHS = 14
# The four U-Net arms are the expensive part of this file: the decoder runs two
# wide convolutions at full resolution, which costs more than the whole
# detector of the previous article. Half the canvases is still 3,000 scenes
# and about 6,000 garments, and the arms are compared against each other
# rather than against anything trained elsewhere, so the cut is priced
# rather than hidden: it is reported in the export and in the article.
N_UNET_TRAIN = 3000
N_BOOT = 200
PROMPT_EPOCHS = 10

# Capped at the machine's core count: still 12 on the box that produces the
# published figures, so nothing moves here, and no oversubscription elsewhere,
# where 10 threads on 4 cores run 4.7x slower AND change the fifth decimal.
THREADS = min(12, os.cpu_count() or 12)
torch.set_num_threads(THREADS)
torch.manual_seed(SEED)

CACHE = Path.home() / ".atlas_vision_data" / "segmentation_cache"
CACHE.mkdir(parents=True, exist_ok=True)
DET_CACHE = Path.home() / ".atlas_vision_data" / "detection_cache"
DCONFIG = f"fscenes5-c64-tr{N_TRAIN}-va{N_VAL}-te{N_TEST_MAX}-s{SEED}"


def cached(name, fn):
    path = CACHE / f"{name}.json"
    if path.exists():
        print(f"  [cached] {name}", flush=True)
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result))
    return result


def cached_npz(name, fn):
    path = CACHE / f"{name}.npz"
    if path.exists():
        print(f"  [cached] {name}", flush=True)
        d = np.load(path)
        return {k: d[k] for k in d.files}
    out = fn()
    np.savez_compressed(path, **out)
    return out


def cached_net(name, fn):
    jpath = CACHE / f"{name}.json"
    mpath = CACHE / f"{name}.pt"
    if jpath.exists() and mpath.exists():
        print(f"  [cached] {name}", flush=True)
        return json.loads(jpath.read_text()), torch.load(mpath, weights_only=True)
    row, state = fn()
    torch.save(state, mpath)
    jpath.write_text(json.dumps(row))
    return row, state


T0 = time.perf_counter()
elapsed = lambda: time.perf_counter() - T0

Ftr_x, Ftr_y, Fte_x, Fte_y = fashion_mnist()
print(f"Fashion-MNIST source: {len(Ftr_y)} train items, {len(Fte_y)} test items", flush=True)


# ===================================================== S0 the scenes and masks
def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter)


def resize_tile(img_u8, s):
    r = skresize(img_u8.astype(np.float64) / 255.0, (s, s), order=1, anti_aliasing=True)
    return np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)


def ink_box(tile):
    m = tile >= INK
    if not m.any():
        return None
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return (int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1)


def sprite_for(img_u8, target_long):
    native = ink_box(img_u8)
    if native is None:
        return None, None
    long_native = max(native[2] - native[0], native[3] - native[1])
    s = int(round(target_long * img_u8.shape[0] / max(long_native, 1)))
    s = int(np.clip(s, 8, CANVAS))
    tile = resize_tile(img_u8, s)
    box = ink_box(tile)
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        return None, None
    return tile, box


def class_pools(labels, source_ids, lo=0.0, hi=1.0):
    pools = []
    for c in source_ids:
        idx = np.where(labels == c)[0]
        pools.append(idx[int(len(idx) * lo):int(len(idx) * hi)])
    return pools


def synth_split(n, images, pools, dpools, base, cap, crowd_frac, crowd_cap):
    """Article 23's scene builder, with the owner map added.

    Not one random draw is added, moved or removed: the sequence of rng calls
    is identical to the detection article's, which is what lets S0 assert that
    these are the same pictures. Everything new here is arithmetic on tiles
    that were already being pasted.

    The owner map is the whole point. Compositing does `cv = max(cv, tile)`, so
    a pixel belongs to whichever object is brightest there, ties going to
    whoever painted first. Recording that during the paste gives an exact,
    unambiguous label for every pixel, which is the ground truth a segmenter is
    asked for and the reason IoU and Dice on this page are counted rather than
    estimated.
    """
    canvases = np.zeros((n, CANVAS, CANVAS), np.uint8)
    owners = np.full((n, CANVAS, CANVAS), -1, np.int16)
    rows, vis, drows = [], [], []
    crowded = np.zeros(n, np.uint8)
    rejected = 0
    for i in range(n):
        rng = np.random.default_rng(SEED * 100003 + base + i)
        is_crowd = int(rng.random() < crowd_frac)
        cap_i = crowd_cap if is_crowd else cap
        crowded[i] = is_crowd
        cv = rng.integers(0, 21, (CANVAS, CANVAS)).astype(np.uint8)
        n_obj = int(rng.integers(2, MAX_OBJ + 1)) if is_crowd else int(rng.integers(1, MAX_OBJ + 1))
        placed = []
        for _ in range(n_obj):
            c = int(rng.integers(0, N_CLS))
            tile, bx = sprite_for(images[int(rng.choice(pools[c]))],
                                  int(rng.integers(MIN_SIDE, MAX_SIDE)))
            if bx is None:
                continue
            s = tile.shape[0]
            ok = False
            for _ in range(40):
                ox = int(rng.integers(0, CANVAS - s + 1))
                oy = int(rng.integers(0, CANVAS - s + 1))
                box = (ox + bx[0], oy + bx[1], ox + bx[2], oy + bx[3])
                if all(iou_xyxy(box, p[1]) <= cap_i for p in placed):
                    placed.append((c, box, tile, ox, oy))
                    ok = True
                    break
            if not ok:
                rejected += 1
        dist = []
        for _ in range(int(rng.integers(0, 4))):
            tile, bx = sprite_for(images[int(rng.choice(dpools[int(rng.integers(0, 5))]))],
                                  int(rng.integers(MIN_SIDE, MAX_SIDE)))
            if bx is None:
                continue
            s = tile.shape[0]
            for _ in range(40):
                ox = int(rng.integers(0, CANVAS - s + 1))
                oy = int(rng.integers(0, CANVAS - s + 1))
                box = (ox + bx[0], oy + bx[1], ox + bx[2], oy + bx[3])
                if (all(iou_xyxy(box, p[1]) <= 0.1 for p in placed)
                        and all(iou_xyxy(box, d[1]) <= cap_i for d in dist)):
                    dist.append((-1, box, tile, ox, oy))
                    break
        own = np.full((CANVAS, CANVAS), -1, np.int16)
        for k, (_c, _b, tile, ox, oy) in enumerate(placed + dist):
            s = tile.shape[0]
            sub = cv[oy:oy + s, ox:ox + s]
            # strictly brighter wins, so a tie leaves the pixel with whoever
            # painted it first and the map stays a function
            take = (tile >= INK) & (tile > sub)
            osub = own[oy:oy + s, ox:ox + s]
            osub[take] = k if k < len(placed) else DIST_ID + (k - len(placed))
            cv[oy:oy + s, ox:ox + s] = np.maximum(sub, tile)
        canvases[i] = cv
        owners[i] = own
        for k, (c, box, tile, ox, oy) in enumerate(placed):
            s = tile.shape[0]
            m = tile >= INK
            sub = cv[oy:oy + s, ox:ox + s]
            vis.append(float((sub[m] == tile[m]).mean()) if m.any() else 0.0)
            amodal = int(m.sum())
            visible = int((own == k).sum())
            rows.append([i, c, box[0], box[1], box[2], box[3], amodal, visible])
        for (_c, box, _t, _ox, _oy) in dist:
            drows.append([i, box[0], box[1], box[2], box[3]])
    return {"canvas": canvases, "owner": owners,
            "boxes": np.array(rows, np.float32).reshape(-1, 8),
            "vis": np.array(vis, np.float32),
            "dboxes": np.array(drows, np.float32).reshape(-1, 5),
            "crowded": crowded,
            "rejected": np.array([rejected], np.int64)}


print("\nS0 the same pictures as article 23, with the owner map recorded", flush=True)
tr_pool = class_pools(Ftr_y, DET_SOURCE)
tr_dpool = class_pools(Ftr_y, HELD_SOURCE)
va_pool = class_pools(Fte_y, DET_SOURCE, 0.0, 0.5)
va_dpool = class_pools(Fte_y, HELD_SOURCE, 0.0, 0.5)
te_pool = class_pools(Fte_y, DET_SOURCE, 0.5, 1.0)
te_dpool = class_pools(Fte_y, HELD_SOURCE, 0.5, 1.0)

S_TR = cached_npz(f"{DCONFIG}-mtrain", lambda: synth_split(
    N_TRAIN, Ftr_x, tr_pool, tr_dpool, 0, 0.35, 0.0, 0.35))
S_VA = cached_npz(f"{DCONFIG}-mval", lambda: synth_split(
    N_VAL, Fte_x, va_pool, va_dpool, 1000000, 0.35, 0.20, 0.60))
S_TE = cached_npz(f"{DCONFIG}-mtest", lambda: synth_split(
    N_TEST_MAX, Fte_x, te_pool, te_dpool, 2000000, 0.35, 0.20, 0.60))


def guard_same_pictures():
    """The claim 'these are article 23's canvases' is checked, not asserted."""
    out = {"checked": False}
    for split, mine in (("train", S_TR), ("val", S_VA), ("test", S_TE)):
        path = DET_CACHE / f"{DCONFIG}-{split}.npz"
        if not path.exists():
            continue
        theirs = np.load(path)
        same_px = bool(np.array_equal(mine["canvas"], theirs["canvas"]))
        same_box = bool(np.array_equal(mine["boxes"][:, :6], theirs["boxes"]))
        out["checked"] = True
        out[split] = {"canvases_identical": same_px, "boxes_identical": same_box,
                      "n": int(len(mine["canvas"]))}
        assert same_px, f"the {split} canvases differ from the detection article's"
        assert same_box, f"the {split} boxes differ from the detection article's"
    return out


SAME = guard_same_pictures()
if SAME["checked"]:
    checked = ", ".join(f"{k} {v['n']}" for k, v in SAME.items() if isinstance(v, dict))
    print("  every canvas and every box identical to the detection article's cache "
          f"({checked})", flush=True)
else:
    print("  detection cache absent, so the scenes were rebuilt from the same seeds "
          "and cannot be cross-checked against it", flush=True)


def semantic_map(owner, boxes_img):
    """Owner ids to a semantic map: 0 background, 1..5 the garment classes.

    A distractor is background here, because the model is never told the five
    held-out classes exist. How often it labels one as a garment anyway is
    measured in S2, and it is the segmentation version of the false positives
    the detection article counted.
    """
    sem = np.zeros(owner.shape, np.int64)
    for k, cls in enumerate(boxes_img):
        sem[owner == k] = int(cls) + 1
    return sem


def build_semantic(split):
    n = len(split["canvas"])
    per_img = [[] for _ in range(n)]
    for r in split["boxes"]:
        per_img[int(r[0])].append(int(r[1]))
    sem = np.zeros((n, CANVAS, CANVAS), np.int8)
    for i in range(n):
        sem[i] = semantic_map(split["owner"][i], per_img[i])
    return {"sem": sem}


SEM_TR = cached_npz(f"{DCONFIG}-sem-train", lambda: build_semantic(S_TR))["sem"]
SEM_VA = cached_npz(f"{DCONFIG}-sem-val", lambda: build_semantic(S_VA))["sem"]
SEM_TE = cached_npz(f"{DCONFIG}-sem-test", lambda: build_semantic(S_TE))["sem"]

Xtr = torch.tensor(S_TR["canvas"], dtype=torch.float32).unsqueeze(1) / 255.0
Xva = torch.tensor(S_VA["canvas"], dtype=torch.float32).unsqueeze(1) / 255.0
Xte = torch.tensor(S_TE["canvas"], dtype=torch.float32).unsqueeze(1) / 255.0
Ytr = torch.tensor(SEM_TR, dtype=torch.long)
Yva = torch.tensor(SEM_VA, dtype=torch.long)
Yte = torch.tensor(SEM_TE, dtype=torch.long)

pix_tr = int(np.prod(SEM_TR.shape))
fg_share = float((SEM_TR > 0).sum()) / pix_tr
dist_share = float((S_TR["owner"] >= DIST_ID).sum()) / pix_tr
amodal = S_TE["boxes"][:, 6]
visible = S_TE["boxes"][:, 7]
hidden = 1.0 - visible / np.maximum(amodal, 1)
fully_hidden = int((visible == 0).sum())
print(f"  {len(S_TE['boxes']):,} test objects; a garment covers {fg_share * 100:.2f}% of the "
      f"training pixels and the untaught distractors another {dist_share * 100:.2f}%", flush=True)
print(f"  {fully_hidden} objects are hidden ENTIRELY: they have a box and no visible pixel "
      f"at all, which is a thing a detector can report and a segmenter cannot", flush=True)
print(f"  occlusion: {float((hidden > 0).mean()) * 100:.1f}% of objects lose at least one pixel "
      f"to a brighter neighbour, {float(hidden.mean()) * 100:.2f}% of their ink on average, "
      f"worst {float(hidden.max()) * 100:.1f}%", flush=True)

# The owner map is new here, so it is checked against a quantity the detection
# article computed a completely different way: the share of an object's ink that
# survives the composite. That one counts a pixel as visible when the object
# EQUALS the maximum, this one when the object strictly wins it, so the two
# differ by exactly the pixels where two objects tie. Anything larger than that
# would mean the map is wrong.
vis_hidden = 1.0 - S_TE["vis"]
tie_share = float(np.mean(hidden - vis_hidden))
assert tie_share >= -1e-9, "the owner map claims more ink than the composite allows"
assert tie_share < 0.02, f"owner map and visible ink disagree by {tie_share:.4f}"
print(f"  cross-check: the detection article's visible-ink fraction says "
      f"{float(vis_hidden.mean()) * 100:.2f}% hidden, this owner map says "
      f"{float(hidden.mean()) * 100:.2f}%; the {tie_share * 100:.3f} point difference is "
      f"the pixels where two garments are exactly as bright as each other", flush=True)

dataset_block = {
    "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST_MAX, "canvas": CANVAS,
    "classes": DET_CLASSES, "held_out": HELD_CLASSES, "ink_threshold": INK,
    "objects_test": int(len(S_TE["boxes"])),
    "foreground_share": rnd(fg_share), "distractor_share": rnd(dist_share),
    "occluded_frac": rnd(float((hidden > 0).mean())),
    "hidden_mean": rnd(float(hidden.mean())),
    "hidden_max": rnd(float(hidden.max())),
    "fully_hidden": fully_hidden,
    "tie_share": rnd(tie_share, 6),
    "hidden_by_visible_ink": rnd(float(vis_hidden.mean())),
    "same_pictures": SAME,
    "owner_rule": "compositing is a pixel-wise maximum, so a pixel belongs to the "
                  "brightest object over it and ties go to whoever was painted "
                  "first; the map is recorded during the paste, so it is exact",
    "amodal_vs_visible": "the box of an object is amodal, it covers the whole "
                         "garment even where a neighbour is brighter; its mask is "
                         "visible, it covers only the pixels the object won. The "
                         "two disagree by exactly the occlusion measured above.",
    "limits": [
        "the pictures are composited by maximum, so occlusion is a brightness "
        "contest rather than a depth order, and no object is ever cut off by the "
        "edge of the canvas",
        "masks are exact by construction, which removes the annotation noise that "
        "dominates real segmentation benchmarks and makes every number here an "
        "upper bound on how clean a real one would be",
        "at 64 by 64 an average garment is a few hundred pixels, so a boundary "
        "band one pixel wide is a large share of it and the boundary decomposition "
        "in S2 is more severe here than it would be at full resolution",
        "the five held-out classes stand in for 'a thing the model was never "
        "taught', which is a much weaker test than the open world SAM was built for",
    ],
}


# ================================================================ S1 the metric
def mask_iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def mask_dice(a, b):
    inter = float(np.logical_and(a, b).sum())
    tot = float(a.sum() + b.sum())
    return 2.0 * inter / tot if tot > 0 else 0.0


print("\nS1 the metric: IoU and Dice counted on real masks, and what a perfect box scores",
      flush=True)


def metric_geometry():
    """Every pair of (object mask, its own filled box) on a slice of test, plus
    the identity that relates the two metrics, checked pair by pair."""
    rs = np.random.RandomState(SEED)
    per_img = [[] for _ in range(N_TEST_MAX)]
    for j, r in enumerate(S_TE["boxes"]):
        per_img[int(r[0])].append((j, r))
    gap = 0.0
    by_class = [[] for _ in range(N_CLS)]
    for i in range(1200):
        own = S_TE["owner"][i]
        for k, (j, r) in enumerate(per_img[i]):
            m = own == k
            if not m.any():
                continue
            bx = np.zeros_like(m)
            bx[int(r[3]):int(r[5]), int(r[2]):int(r[4])] = True
            iou = mask_iou(m, bx)
            dice = mask_dice(m, bx)
            gap = max(gap, abs(dice - (2 * iou / (1 + iou)) if iou > 0 else dice))
            by_class[int(r[1])].append(iou)
    rows = [{"cls": DET_CLASSES[c], "index": c, "n": len(v),
             "box_as_mask_iou": rnd(float(np.mean(v))),
             "box_as_mask_dice": rnd(float(np.mean([2 * x / (1 + x) for x in v])))}
            for c, v in enumerate(by_class)]
    allv = [x for v in by_class for x in v]
    # the curve the browser draws: Dice against IoU, from the identity
    curve = [{"iou": rnd(t / 40, 3), "dice": rnd(2 * (t / 40) / (1 + t / 40), 6)}
             for t in range(41)]
    # No sample of individual pairs is exported: the widget recomputes them
    # exactly from the owner map it already has, so a shipped sample would be
    # dead weight nobody reads.
    return {"per_class": rows, "curve": curve,
            "identity_max_gap": rnd(gap, 12),
            "box_as_mask_iou_mean": rnd(float(np.mean(allv))),
            "box_as_mask_iou_best": rnd(float(np.max(allv))),
            "n_objects": len(allv),
            "identity": "Dice = 2 IoU / (1 + IoU), so the two metrics order any "
                        "two predictions the same way and only the number differs"}


GEO = cached(f"{DCONFIG}-metric-geometry", metric_geometry)
print(f"  a PERFECT box read as a mask scores IoU {GEO['box_as_mask_iou_mean']:.4f} on average "
      f"over {GEO['n_objects']:,} objects, best {GEO['box_as_mask_iou_best']:.4f}", flush=True)
print("  " + ", ".join(f"{r['cls']} {r['box_as_mask_iou']:.3f}" for r in GEO["per_class"]),
      flush=True)
print(f"  Dice = 2 IoU / (1 + IoU) checked on every pair: max gap "
      f"{GEO['identity_max_gap']:.3e}", flush=True)


# =================================================================== S2 U-Net
class UNet(nn.Module):
    """Encoder, decoder, and skips that can be switched OFF without changing a
    single shape.

    The usual ablation deletes the skip connection, which also deletes the
    channels it contributed and therefore changes the parameter count, the
    multiply-accumulates and the width of the first decoder convolution. Then
    the comparison is not about skips. Here the concatenation always happens
    and the skip tensor is replaced by zeros, so the two arms are the same
    network to the last weight and the only difference is whether the encoder's
    detail can reach the decoder at all.
    """

    CH = [8, 16, 32]

    def __init__(self, skips=True, levels=2, n_out=N_SEM, cin=1):
        super().__init__()
        c1, c2, c3 = self.CH
        self.skips = skips
        self.levels = levels
        self.e1 = nn.Conv2d(cin, c1, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c1)
        self.e2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c2)
        self.e3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(c3)
        self.d2 = nn.Conv2d(c3 + c2, c2, 3, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(c2)
        self.d1 = nn.Conv2d(c2 + c1, c1, 3, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(c1)
        self.out = nn.Conv2d(c1, n_out, 1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.act(self.b1(self.e1(x)))              # 64
        s2 = self.act(self.b2(self.e2(self.pool(s1))))  # 32
        z = self.act(self.b3(self.e3(self.pool(s2))))   # 16
        k2 = s2 if self.skips else torch.zeros_like(s2)
        k1 = s1 if self.skips else torch.zeros_like(s1)
        u = F.interpolate(z, scale_factor=2, mode="nearest")
        u = self.act(self.b4(self.d2(torch.cat([u, k2], 1))))
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.act(self.b5(self.d1(torch.cat([u, k1], 1))))
        return self.out(u)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, shape):
    total = [0]
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total[0] += mod.weight.numel() * out.shape[-1] * out.shape[-2]

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(hook))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape))
    for h in hooks:
        h.remove()
    model.train(was)
    return int(total[0])


def lr_at(ep, epochs, base):
    if ep < 2:
        return base * (ep + 1) / 2
    if ep >= int(epochs * 0.85):
        return base * 0.01
    if ep >= int(epochs * 0.6):
        return base * 0.1
    return base


def boundary_band(sem):
    """Pixels whose four-neighbourhood is not all one label. On these canvases
    that band is where almost all of the disagreement lives, so the error is
    reported on it and off it separately."""
    b = np.zeros(sem.shape, bool)
    b[:, 1:, :] |= sem[:, 1:, :] != sem[:, :-1, :]
    b[:, :-1, :] |= sem[:, 1:, :] != sem[:, :-1, :]
    b[:, :, 1:] |= sem[:, :, 1:] != sem[:, :, :-1]
    b[:, :, :-1] |= sem[:, :, 1:] != sem[:, :, :-1]
    return b


BAND_TE = boundary_band(SEM_TE)
print(f"\nS2 the U-Net: skips ablated by zeroing them, so the two arms are the same network",
      flush=True)
print(f"  the boundary band is {float(BAND_TE.mean()) * 100:.2f}% of every test pixel",
      flush=True)


def fit_unet(skips, epochs, seed, cin=1, n_out=N_SEM, tag=""):
    torch.manual_seed(seed)
    model = UNet(skips=skips, n_out=n_out, cin=cin)
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed)
    hist = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(ep, epochs, 0.08)
        perm = torch.randperm(N_UNET_TRAIN, generator=g)
        model.train()
        run, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            s = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr[s]), Ytr[s])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
        hist.append(rnd(run / max(nb, 1), 5))
        if ep % 6 == 0 or ep == epochs - 1:
            print(f"    {tag} epoch {ep + 1}/{epochs} loss {hist[-1]:.4f} "
                  f"({elapsed() / 60:.1f} min elapsed)", flush=True)
    return model, hist, time.perf_counter() - t0


@torch.no_grad()
def predict_sem(model, X, batch=250):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch]).argmax(1).to(torch.int8))
    return torch.cat(out).numpy()


def score_semantic(pred, truth, band):
    """Per-class IoU and Dice, counted from the confusion of two integer maps.
    Nothing here is sampled or estimated: these are pixel counts."""
    rows = []
    for c in range(N_SEM):
        p = pred == c
        t = truth == c
        inter = int(np.logical_and(p, t).sum())
        union = int(np.logical_or(p, t).sum())
        rows.append({"cls": "background" if c == 0 else DET_CLASSES[c - 1],
                     "index": c,
                     "iou": rnd(inter / union if union else float("nan")),
                     "dice": rnd(2 * inter / (int(p.sum()) + int(t.sum()))
                                 if int(p.sum()) + int(t.sum()) else float("nan")),
                     "pixels": int(t.sum())})
    fg = [r["iou"] for r in rows[1:]]
    acc = float((pred == truth).mean())
    on = float((pred[band] == truth[band]).mean())
    off = float((pred[~band] == truth[~band]).mean())
    return {"per_class": rows, "miou_fg": rnd(float(np.mean(fg))),
            "miou_all": rnd(float(np.mean([r["iou"] for r in rows]))),
            "pixel_acc": rnd(acc), "acc_boundary": rnd(on), "acc_interior": rnd(off),
            "band_share": rnd(float(band.mean()))}


def unet_arm(name, skips, epochs, seed):
    def run():
        model, hist, secs = fit_unet(skips, epochs, seed, tag=name)
        pred = predict_sem(model, Xte)
        sc = score_semantic(pred, SEM_TE, BAND_TE)
        # what it does with the five classes it was never taught
        dist = S_TE["owner"] >= DIST_ID
        row = {"id": name, "skips": bool(skips), "epochs": epochs, "seed": seed,
               "params": n_params(model), "macs": count_macs(model, (1, 1, CANVAS, CANVAS)),
               "seconds": rnd(secs, 1), "history": hist,
               "distractor_claimed": rnd(float((pred[dist] > 0).mean())),
               "distractor_pixels": int(dist.sum()), **sc}
        return row, model.state_dict()
    return cached_net(f"{DCONFIG}-unet-{name}-e{epochs}-s{seed}", run)


UNET_ARMS = []
unet_states = {}
for name, skips, seed in (("skips", True, SEED), ("no-skips", False, SEED),
                          ("skips-seed20", True, 20), ("no-skips-seed20", False, 20)):
    row, state = unet_arm(name, skips, EPOCHS, seed)
    UNET_ARMS.append(row)
    unet_states[name] = state
    print(f"  {name:16s} mIoU(fg) {row['miou_fg']:.4f}  pixel acc {row['pixel_acc']:.4f}  "
          f"boundary {row['acc_boundary']:.4f} interior {row['acc_interior']:.4f}  "
          f"{row['params']:,} params  {row['seconds']:.0f}s", flush=True)

skip_row = next(r for r in UNET_ARMS if r["id"] == "skips")
nosk_row = next(r for r in UNET_ARMS if r["id"] == "no-skips")
skip_row20 = next(r for r in UNET_ARMS if r["id"] == "skips-seed20")
nosk_row20 = next(r for r in UNET_ARMS if r["id"] == "no-skips-seed20")
seed_spread = max(abs(skip_row["miou_fg"] - skip_row20["miou_fg"]),
                  abs(nosk_row["miou_fg"] - nosk_row20["miou_fg"]))
skip_delta = skip_row["miou_fg"] - nosk_row["miou_fg"]
# The headline metric turned out to be dominated by the seed, so the same
# comparison is made region by region, each against its own noise floor. That
# is where the claim about skips actually lives, and it is the only place the
# two seeds agree with each other.
REGIONS = {}
for field in ("acc_boundary", "acc_interior"):
    d19 = skip_row[field] - nosk_row[field]
    d20 = skip_row20[field] - nosk_row20[field]
    spread = max(abs(skip_row[field] - skip_row20[field]),
                 abs(nosk_row[field] - nosk_row20[field]))
    REGIONS[field] = {
        "delta_seed19": rnd(d19), "delta_seed20": rnd(d20),
        "delta_mean": rnd((d19 + d20) / 2), "seed_spread": rnd(spread),
        "same_sign": bool(d19 * d20 > 0),
        "resolved": bool(abs((d19 + d20) / 2) > spread),
    }
assert skip_row["params"] == nosk_row["params"], "the ablation changed the parameter count"
assert skip_row["macs"] == nosk_row["macs"], "the ablation changed the arithmetic"
print(f"  skips are worth {skip_delta:+.4f} mIoU against a seed spread of {seed_spread:.4f}, "
      f"with identical parameters ({skip_row['params']:,}) and identical MACs", flush=True)
print(f"  where they pay: boundary accuracy {nosk_row['acc_boundary']:.4f} to "
      f"{skip_row['acc_boundary']:.4f}, interior {nosk_row['acc_interior']:.4f} to "
      f"{skip_row['acc_interior']:.4f}", flush=True)


# ================================================== S3 semantic against instance
print("\nS3 what a per-pixel classifier cannot do: tell two of the same thing apart",
      flush=True)


def connected(mask):
    """Four-connected components, iterative so a long garment cannot blow the
    recursion limit."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    cur = 0
    stack = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or lab[y, x]:
                continue
            cur += 1
            stack.append((y, x))
            lab[y, x] = cur
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
    return lab, cur


def merge_census():
    """How often does a same-class pair touch, and how often does the semantic
    map turn the pair into one blob? Counted on the GROUND TRUTH first, because
    that number is a property of the pictures and not of any model."""
    per_img = [[] for _ in range(N_TEST_MAX)]
    for r in S_TE["boxes"]:
        per_img[int(r[0])].append(int(r[1]))
    touching, merged, imgs_with = 0, 0, 0
    for i in range(N_TEST_MAX):
        own = S_TE["owner"][i]
        cls = per_img[i]
        found = False
        for a in range(len(cls)):
            for b in range(a + 1, len(cls)):
                if cls[a] != cls[b]:
                    continue
                ma, mb = own == a, own == b
                if not ma.any() or not mb.any():
                    continue
                adj = (np.logical_and(ma[1:, :], mb[:-1, :]).any()
                       or np.logical_and(ma[:-1, :], mb[1:, :]).any()
                       or np.logical_and(ma[:, 1:], mb[:, :-1]).any()
                       or np.logical_and(ma[:, :-1], mb[:, 1:]).any())
                if not adj:
                    continue
                touching += 1
                found = True
                _lab, n = connected(np.logical_or(ma, mb))
                if n == 1:
                    merged += 1
        imgs_with += int(found)
    return {"same_class_pairs_touching": touching, "merged_into_one": merged,
            "images_affected": imgs_with, "n_test": N_TEST_MAX}


MERGE = cached(f"{DCONFIG}-merge-census", merge_census)
print(f"  {MERGE['same_class_pairs_touching']} pairs of same-class objects touch in the "
      f"{MERGE['n_test']:,} test canvases, and the semantic map fuses "
      f"{MERGE['merged_into_one']} of them into a single region", flush=True)


# ============================================ S4 Mask R-CNN: the detector plus
class TinyDet(nn.Module):
    """Article 23's detector, verbatim, so its trained weights load into it."""

    CH = [12, 24, 48, 64]

    def __init__(self, n_anchors, n_classes=N_CLS, stride=8):
        super().__init__()
        self.pool_after = {4: (0, 1), 8: (0, 1, 2), 16: (0, 1, 2, 3)}[stride]
        cin = 1
        convs, bns = [], []
        for cout in self.CH:
            convs.append(nn.Conv2d(cin, cout, 3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(cout))
            cin = cout
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.head = nn.Conv2d(cin, n_anchors * (5 + n_classes), 1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.stride = stride

    def body(self, x):
        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            x = self.act(b(c(x)))
            if i in self.pool_after:
                x = self.pool(x)
        return x

    def forward(self, x):
        return self.head(self.body(x))

    def raw(self, x):
        o = self.forward(x)
        B, _, G, _ = o.shape
        return o.view(B, self.n_anchors, 5 + self.n_classes, G, G).permute(0, 1, 3, 4, 2)


MAIN_ANCHORS = [[12.571, 23.7605], [26.035, 13.3785], [26.0578, 32.3766]]
ARM_KEY = f"{DCONFIG}-e25-b64-arm-slots3-kmeans"
DET_STATE = DET_CACHE / f"{ARM_KEY}.pt"
DET_ROW = DET_CACHE / f"{ARM_KEY}.json"

print("\nS4 Mask R-CNN: article 23's detector, with a mask head bolted on", flush=True)
detector = TinyDet(len(MAIN_ANCHORS), stride=8)
if DET_STATE.exists():
    detector.load_state_dict(torch.load(DET_STATE, weights_only=True))
    det_row = json.loads(DET_ROW.read_text()) if DET_ROW.exists() else {}
    DET_NOTE = ("loaded from the detection article's own cache, so this is literally "
                "the network that article measured, not a retrained copy of it")
    print(f"  loaded {ARM_KEY} (mAP@0.5 {det_row.get('map50')}), "
          f"{n_params(detector):,} parameters", flush=True)
else:
    det_row = {}
    DET_NOTE = ("the detection article's cache was absent, so the trunk is untrained "
                "and the mask numbers below are not comparable to anything")
    print("  WARNING: no cached detector, the mask head has nothing good to sit on", flush=True)
detector.eval()


def anchor_tensor(anchors):
    return torch.tensor(np.asarray(anchors, np.float32)).view(1, -1, 1, 1, 2)


def grid_offsets(G):
    gy, gx = torch.meshgrid(torch.arange(G, dtype=torch.float32),
                            torch.arange(G, dtype=torch.float32), indexing="ij")
    return gx.view(1, 1, G, G), gy.view(1, 1, G, G)


def decode(raw, an_t, stride, gx, gy, clip=False):
    cx = (gx + torch.sigmoid(raw[..., 0])) * stride
    cy = (gy + torch.sigmoid(raw[..., 1])) * stride
    w = an_t[..., 0] * torch.exp(raw[..., 2].clamp(-4, 4))
    h = an_t[..., 1] * torch.exp(raw[..., 3].clamp(-4, 4))
    b = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    return b.clamp(0.0, float(CANVAS)) if clip else b


def roi_align(feat, img_idx, rois, bins, stride=8):
    G = feat.shape[-1]
    u = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
    cx = (rois[:, 0:1] + u[None, :] * (rois[:, 2:3] - rois[:, 0:1])) / stride
    cy = (rois[:, 1:2] + u[None, :] * (rois[:, 3:4] - rois[:, 1:2])) / stride
    gx = cx * 2.0 / G - 1.0
    gy = cy * 2.0 / G - 1.0
    R = len(rois)
    grid = torch.stack([gx[:, None, :].expand(R, bins, bins),
                        gy[:, :, None].expand(R, bins, bins)], -1)
    return F.grid_sample(feat[img_idx], grid, mode="bilinear", align_corners=False,
                         padding_mode="border")


MASK_BINS = 14
TOPK = 32


class MaskHead(nn.Module):
    """RoIAlign at 14x14 on the shared stride-8 map, two convolutions, one mask
    per class. The ceiling is declared before running, as article 23 declared
    its own: the map is 8x8, so a 20 px box covers 2.5 cells and everything
    finer than that is upsampling rather than evidence. A real Mask R-CNN pools
    from a feature pyramid at stride 4 or finer."""

    def __init__(self, cin=64, n_classes=N_CLS, per_class=True):
        super().__init__()
        self.per_class = per_class
        self.c1 = nn.Conv2d(cin, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.out = nn.Conv2d(32, n_classes if per_class else 1, 1)
        self.act = nn.ReLU()

    def forward(self, pooled):
        z = self.act(self.c1(pooled))
        z = self.act(self.c2(z))
        return self.out(z)


@torch.no_grad()
def proposals_from(model, X, anchors, stride, topk=TOPK):
    model.eval()
    an_t = anchor_tensor(anchors)
    G = CANVAS // stride
    gx, gy = grid_offsets(G)
    bx_, im_, ob_, cl_, cp_ = [], [], [], [], []
    for i in range(0, len(X), 250):
        raw = model.raw(X[i:i + 250])
        B = raw.shape[0]
        bx = decode(raw, an_t, stride, gx, gy, clip=True).reshape(B, -1, 4)
        ob = torch.sigmoid(raw[..., 4]).reshape(B, -1)
        cp = torch.softmax(raw[..., 5:], -1).reshape(B, -1, N_CLS)
        cbest, cidx = cp.max(-1)
        k = min(topk, ob.shape[1])
        sc, idx = ob.topk(k, dim=1)
        bx_.append(torch.gather(bx, 1, idx[..., None].expand(B, k, 4)).reshape(-1, 4))
        im_.append((torch.arange(B) + i).repeat_interleave(k))
        ob_.append(sc.reshape(-1))
        cl_.append(torch.gather(cidx, 1, idx).reshape(-1))
        cp_.append(torch.gather(cbest, 1, idx).reshape(-1))
    return (torch.cat(bx_), torch.cat(im_), torch.cat(ob_), torch.cat(cl_), torch.cat(cp_))


def crop_mask_target(owner, k, roi, bins=MASK_BINS):
    """The object's visible mask resampled into the RoI's own frame. Nearest
    neighbour on purpose: the target is a set of pixels, and interpolating a
    label map invents pixels that belong to nobody."""
    x1, y1, x2, y2 = roi
    ys = np.clip(((np.arange(bins) + 0.5) / bins * (y2 - y1) + y1).astype(int), 0, CANVAS - 1)
    xs = np.clip(((np.arange(bins) + 0.5) / bins * (x2 - x1) + x1).astype(int), 0, CANVAS - 1)
    return (owner[np.ix_(ys, xs)] == k).astype(np.float32)


def paste_mask(prob, roi, thr=0.5):
    out = np.zeros((CANVAS, CANVAS), bool)
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(CANVAS, x2), min(CANVAS, y2)
    if x2 <= x1 or y2 <= y1:
        return out
    b = prob.shape[0]
    ys = np.clip(((np.arange(y2 - y1) + 0.5) / (y2 - y1) * b).astype(int), 0, b - 1)
    xs = np.clip(((np.arange(x2 - x1) + 0.5) / (x2 - x1) * b).astype(int), 0, b - 1)
    out[y1:y2, x1:x2] = prob[np.ix_(ys, xs)] >= thr
    return out


def all_point_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def average_precision(records, npos_by_class):
    """Confidence order, each ground-truth object consumed at most once, per
    class, then the all-point envelope. Article 23's rule with the overlap
    function swapped for whichever one the caller matched with."""
    aps = []
    for c in range(N_CLS):
        rows = [r for r in records if r["cls"] == c]
        npos = npos_by_class[c]
        if npos == 0:
            continue
        tp, fp, rec, prec = 0, 0, [], []
        for r in rows:
            if r["hit"]:
                tp += 1
            else:
                fp += 1
            rec.append(tp / npos)
            prec.append(tp / (tp + fp))
        aps.append(all_point_ap(np.array(rec), np.array(prec)) if rec else 0.0)
    return float(np.mean(aps)) if aps else 0.0


def run_mask_rcnn():
    n_tr = min(3000, N_TRAIN)
    with torch.no_grad():
        feat_tr = torch.cat([detector.body(Xtr[i:i + 250]) for i in range(0, n_tr, 250)])
        feat_te = torch.cat([detector.body(Xte[i:i + 250]) for i in range(0, N_TEST_MAX, 250)])
    pb, pi, po, pc, pcp = proposals_from(detector, Xtr[:n_tr], MAIN_ANCHORS, 8)
    pb_np = pb.numpy()

    tr_by_img = [[] for _ in range(N_TRAIN)]
    for r in S_TR["boxes"]:
        tr_by_img[int(r[0])].append(r)
    sel, tgt_mask, tgt_cls = [], [], []
    for i in range(n_tr):
        g = tr_by_img[i]
        if not g:
            continue
        owner = S_TR["owner"][i]
        gb = np.array([r[2:6] for r in g], np.float32)
        for s in range(i * TOPK, (i + 1) * TOPK):
            o = np.array([iou_xyxy(pb_np[s], b) for b in gb])
            k = int(o.argmax())
            if o[k] < 0.5:
                continue
            sel.append(s)
            tgt_mask.append(crop_mask_target(owner, k, pb_np[s]))
            tgt_cls.append(int(g[k][1]))
    sel_t = torch.tensor(sel, dtype=torch.long)
    tm = torch.tensor(np.stack(tgt_mask))
    tc = torch.tensor(np.array(tgt_cls), dtype=torch.long)
    print(f"  {len(sel_t):,} proposals overlap an object enough to carry a mask target",
          flush=True)

    heads = {}
    for per_class in (True, False):
        name = "per-class" if per_class else "class-agnostic"
        torch.manual_seed(SEED)
        head = MaskHead(per_class=per_class)
        opt = torch.optim.SGD(head.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        g = torch.Generator().manual_seed(SEED)
        t0 = time.perf_counter()
        for ep in range(12):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(ep, 12, 0.05)
            perm = torch.randperm(len(sel_t), generator=g)
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]
                s = sel_t[idx]
                pooled = roi_align(feat_tr, pi[s], pb[s], MASK_BINS)
                logit = head(pooled)
                if per_class:
                    logit = logit[torch.arange(len(s)), tc[idx]]
                else:
                    logit = logit[:, 0]
                loss = F.binary_cross_entropy_with_logits(logit, tm[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        heads[name] = (head, time.perf_counter() - t0)
        print(f"    {name} head trained in {heads[name][1]:.0f}s "
              f"({elapsed() / 60:.1f} min elapsed)", flush=True)

    tb, ti, to, tcl, tcp = proposals_from(detector, Xte, MAIN_ANCHORS, 8)
    score_all = (to * tcp).numpy()
    keep = score_all >= 0.05
    tb_np, ti_np, tcl_np = tb.numpy()[keep], ti.numpy()[keep], tcl.numpy()[keep]
    score = score_all[keep]
    slots = np.arange(len(keep))[keep]

    te_by_img = [[] for _ in range(N_TEST_MAX)]
    for r in S_TE["boxes"]:
        te_by_img[int(r[0])].append(r)
    npos = [0] * N_CLS
    for r in S_TE["boxes"]:
        npos[int(r[1])] += 1

    rows = []
    for name, (head, secs) in heads.items():
        with torch.no_grad():
            probs = []
            for i in range(0, len(tb_np), 1024):
                sl = slice(i, i + 1024)
                pooled = roi_align(feat_te, torch.tensor(ti_np[sl]),
                                   torch.tensor(tb_np[sl]), MASK_BINS)
                lg = head(pooled)
                if head.per_class:
                    lg = lg[torch.arange(lg.shape[0]), torch.tensor(tcl_np[sl])]
                else:
                    lg = lg[:, 0]
                probs.append(torch.sigmoid(lg).numpy())
            probs = np.concatenate(probs)
        order = np.lexsort((slots, ti_np, -score))
        used_box, used_mask = {}, {}
        recs_box, recs_mask, ious = [], [], []
        for j in order:
            img = int(ti_np[j])
            cls = int(tcl_np[j])
            idxs = [k for k, r in enumerate(te_by_img[img]) if int(r[1]) == cls]
            owner = S_TE["owner"][img]
            bo = [iou_xyxy(tb_np[j], te_by_img[img][k][2:6]) for k in idxs]
            pm = paste_mask(probs[j], tb_np[j])
            mo = [mask_iou(pm, owner == k) for k in idxs]
            fb = used_box.setdefault(img, set())
            fm = used_mask.setdefault(img, set())
            hb = hm = False
            if bo:
                cand = [v if idxs[q] not in fb else -1.0 for q, v in enumerate(bo)]
                k = int(np.argmax(cand))
                if cand[k] >= 0.5:
                    fb.add(idxs[k])
                    hb = True
            if mo:
                cand = [v if idxs[q] not in fm else -1.0 for q, v in enumerate(mo)]
                k = int(np.argmax(cand))
                if cand[k] >= 0.5:
                    fm.add(idxs[k])
                    hm = True
                    ious.append(cand[k])
            recs_box.append({"cls": cls, "hit": hb})
            recs_mask.append({"cls": cls, "hit": hm})
        rows.append({
            "variant": name, "seconds": rnd(secs, 1), "params": n_params(head),
            "bins": MASK_BINS,
            "box_ap50": rnd(average_precision(recs_box, npos)),
            "mask_ap50": rnd(average_precision(recs_mask, npos)),
            "mean_matched_mask_iou": rnd(float(np.mean(ious)) if ious else 0.0),
            "n_detections": int(len(order)),
        })
        print(f"    {name:15s} box AP@0.5 {rows[-1]['box_ap50']:.4f}  "
              f"mask AP@0.5 {rows[-1]['mask_ap50']:.4f}  "
              f"mean matched mask IoU {rows[-1]['mean_matched_mask_iou']:.4f}", flush=True)
    return {"variants": rows, "topk": TOPK, "score_floor": 0.05,
            "detector": DET_NOTE, "detector_map50": det_row.get("map50"),
            "detector_params": n_params(detector),
            "declared": "the mask is pooled from an 8 by 8 map, so anything finer "
                        "than an eighth of the canvas is upsampling rather than "
                        "evidence; a real Mask R-CNN pools from a pyramid at stride "
                        "4 or finer, and this ceiling was written down before the "
                        "head was trained"}


MASKRCNN = cached(f"{DCONFIG}-maskrcnn-b{MASK_BINS}", run_mask_rcnn)


# =========================================== S5 the one property of SAM we can test
print("\nS5 promptable and class-agnostic: trained on five classes, asked about five others",
      flush=True)
PROMPT_SIGMA = 8.0
_PK = None


def prompt_channel(px, py):
    """The prompt as a smooth bump at the clicked pixel. A single one-hot pixel
    is too sparse a cue for a small network to find, and the bump is computed
    once as a double-width kernel and then cropped, because this runs a few
    hundred thousand times."""
    global _PK
    if _PK is None:
        ys, xs = np.mgrid[-CANVAS + 1:CANVAS, -CANVAS + 1:CANVAS]
        _PK = np.exp(-(xs ** 2 + ys ** 2) / (2 * PROMPT_SIGMA ** 2)).astype(np.float32)
    return _PK[CANVAS - 1 - py:2 * CANVAS - 1 - py, CANVAS - 1 - px:2 * CANVAS - 1 - px]


def sample_prompts(split, n_per_object, rng, held_out=False):
    """One training example is a canvas, a point inside one object, and that
    object's mask. The point is sampled from the object's VISIBLE pixels,
    because a prompt the user could not have clicked is not a prompt."""
    rows = []
    per_img = [[] for _ in range(len(split["canvas"]))]
    if held_out:
        counts = [0] * len(split["canvas"])
        for r in split["dboxes"]:
            i = int(r[0])
            per_img[i].append(DIST_ID + counts[i])
            counts[i] += 1
    else:
        counts = [0] * len(split["canvas"])
        for r in split["boxes"]:
            i = int(r[0])
            per_img[i].append(counts[i])
            counts[i] += 1
    for i, ids in enumerate(per_img):
        for oid in ids:
            m = split["owner"][i] == oid
            pts = np.argwhere(m)
            if len(pts) < 8:
                continue
            for _ in range(n_per_object):
                py, px = pts[rng.integers(0, len(pts))]
                rows.append((i, int(px), int(py), int(oid)))
    return rows


class PromptNet(nn.Module):
    """The same U-Net, with a second input channel that says where the user
    pointed and a single output channel that says which pixels belong to the
    thing under the point. No class ever enters it, which is the whole idea."""

    def __init__(self):
        super().__init__()
        self.net = UNet(skips=True, n_out=1, cin=2)

    def forward(self, x):
        return self.net(x)


def build_prompt_batch(split, rows, idx):
    xs = np.zeros((len(idx), 2, CANVAS, CANVAS), np.float32)
    ys = np.zeros((len(idx), CANVAS, CANVAS), np.float32)
    for n, j in enumerate(idx):
        i, px, py, oid = rows[j]
        xs[n, 0] = split["canvas"][i] / 255.0
        xs[n, 1] = prompt_channel(px, py)
        ys[n] = (split["owner"][i] == oid).astype(np.float32)
    return torch.tensor(xs), torch.tensor(ys)


def run_promptable():
    rng = np.random.default_rng(SEED)
    tr_rows = sample_prompts(S_TR, 1, rng)
    print(f"  {len(tr_rows):,} training prompts, all of them on the five taught classes",
          flush=True)
    torch.manual_seed(SEED)
    model = PromptNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(SEED)
    hist = []
    t0 = time.perf_counter()
    for ep in range(PROMPT_EPOCHS):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(ep, PROMPT_EPOCHS, 0.08)
        perm = torch.randperm(len(tr_rows), generator=g).numpy()
        model.train()
        run, nb = 0.0, 0
        for i in range(0, len(perm), 48):
            xb, yb = build_prompt_batch(S_TR, tr_rows, perm[i:i + 48])
            loss = F.binary_cross_entropy_with_logits(model(xb)[:, 0], yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
        hist.append(rnd(run / max(nb, 1), 5))
        print(f"    epoch {ep + 1}/{PROMPT_EPOCHS} loss {hist[-1]:.4f} "
              f"({elapsed() / 60:.1f} min elapsed)", flush=True)
    secs = time.perf_counter() - t0

    @torch.no_grad()
    def evaluate_rows(rows, split, cap=1500):
        model.eval()
        rng2 = np.random.default_rng(SEED + 1)
        pick = rng2.permutation(len(rows))[:cap]
        ious, dices = [], []
        for i in range(0, len(pick), 64):
            xb, yb = build_prompt_batch(split, rows, pick[i:i + 64])
            pr = torch.sigmoid(model(xb)[:, 0]).numpy() >= 0.5
            gt = yb.numpy() > 0.5
            for a, b in zip(pr, gt):
                ious.append(mask_iou(a, b))
                dices.append(mask_dice(a, b))
        return {"n": len(pick), "iou": rnd(float(np.mean(ious))),
                "dice": rnd(float(np.mean(dices))),
                "iou_median": rnd(float(np.median(ious))),
                "above_50": rnd(float(np.mean([v >= 0.5 for v in ious]))),
                "above_75": rnd(float(np.mean([v >= 0.75 for v in ious])))}

    te_known = sample_prompts(S_TE, 1, np.random.default_rng(SEED + 2))
    te_held = sample_prompts(S_TE, 1, np.random.default_rng(SEED + 3), held_out=True)
    known = evaluate_rows(te_known, S_TE)
    held = evaluate_rows(te_held, S_TE)
    print(f"  taught classes    IoU {known['iou']:.4f}  Dice {known['dice']:.4f}  "
          f"{known['above_50'] * 100:.1f}% above 0.5", flush=True)
    print(f"  NEVER-SEEN classes IoU {held['iou']:.4f}  Dice {held['dice']:.4f}  "
          f"{held['above_50'] * 100:.1f}% above 0.5", flush=True)
    row = {"known": known, "held_out": held, "history": hist, "seconds": rnd(secs, 1),
           "params": n_params(model), "macs": count_macs(model, (1, 2, CANVAS, CANVAS)),
           "epochs": PROMPT_EPOCHS, "sigma": PROMPT_SIGMA,
           "n_train_prompts": len(tr_rows),
           "gap": rnd(known["iou"] - held["iou"]),
           "honesty": "this is not SAM. SAM is a ViT trained on eleven million "
                      "images and a billion masks, and nothing on this page could "
                      "be trained into it. What is testable here is the property "
                      "that makes SAM useful: a model asked for THE THING UNDER "
                      "THIS POINT, never for a class, and measured on categories "
                      "it was never trained on."}
    return row, model.state_dict()


PROMPT, prompt_state = cached_net(f"{DCONFIG}-prompt-e{PROMPT_EPOCHS}", run_promptable)
promptnet = PromptNet()
promptnet.load_state_dict(prompt_state)
promptnet.eval()


# ================================================================= S6 the export
print("\nS6 export: fold, quantise, then compute the reference", flush=True)
LIVE_N = 12


def fold_unet(model):
    """Batch normalisation folded into the convolution before it, so the browser
    runs convolutions and nothing else."""
    layers = []
    net = model.net
    pairs = [(net.e1, net.b1), (net.e2, net.b2), (net.e3, net.b3),
             (net.d2, net.b4), (net.d1, net.b5)]
    for conv, bn in pairs:
        w = conv.weight.detach().double()
        g = bn.weight.detach().double()
        b = bn.bias.detach().double()
        rm = bn.running_mean.detach().double()
        rv = bn.running_var.detach().double()
        s = g / torch.sqrt(rv + bn.eps)
        layers.append((w * s.view(-1, 1, 1, 1), b - rm * s))
    layers.append((net.out.weight.detach().double(), net.out.bias.detach().double()))
    return layers


def folded_forward(layers, x, skips=True):
    def conv(i, t, pad=1):
        w, b = layers[i]
        return F.conv2d(t, w.to(t.dtype), b.to(t.dtype), padding=pad)
    s1 = F.relu(conv(0, x))
    s2 = F.relu(conv(1, F.max_pool2d(s1, 2)))
    z = F.relu(conv(2, F.max_pool2d(s2, 2)))
    u = F.interpolate(z, scale_factor=2, mode="nearest")
    u = F.relu(conv(3, torch.cat([u, s2], 1)))
    u = F.interpolate(u, scale_factor=2, mode="nearest")
    u = F.relu(conv(4, torch.cat([u, s1], 1)))
    return conv(5, u, pad=0)


LAYERS = fold_unet(promptnet)
live_idx = list(range(LIVE_N))
sprite = np.concatenate([S_TE["canvas"][i] for i in live_idx], axis=1)
Image.fromarray(sprite.astype(np.uint8), mode="L").save(IMG / "canvases.png")

# The three semantic layers the U-Net widget flips between, as index images.
best_unet = UNet(skips=True)
best_unet.load_state_dict(unet_states["skips"])
plain_unet = UNet(skips=False)
plain_unet.load_state_dict(unet_states["no-skips"])
pred_skip = predict_sem(best_unet, Xte[:LIVE_N])
pred_plain = predict_sem(plain_unet, Xte[:LIVE_N])
for name, arr in (("truth", SEM_TE[:LIVE_N]), ("skips", pred_skip), ("noskips", pred_plain)):
    strip = np.concatenate([arr[i] for i in live_idx], axis=1).astype(np.uint8)
    Image.fromarray(strip, mode="L").save(IMG / f"sem-{name}.png")

# The owner map itself, one byte per pixel, so the browser can score whatever
# the reader points at instead of only the prompts chosen here: 0 background,
# 1..K the garments, 200 and up the five classes nothing was trained on.
own_strip = np.zeros((CANVAS, LIVE_N * CANVAS), np.uint8)
for n, i in enumerate(live_idx):
    o = S_TE["owner"][i]
    b = np.zeros((CANVAS, CANVAS), np.uint8)
    known = (o >= 0) & (o < DIST_ID)
    b[known] = o[known].astype(np.uint8) + 1
    dist = o >= DIST_ID
    b[dist] = np.clip(o[dist] - DIST_ID, 0, 50).astype(np.uint8) + 200
    own_strip[:, n * CANVAS:(n + 1) * CANVAS] = b
Image.fromarray(own_strip, mode="L").save(IMG / "owner.png")
print(f"  wrote the sprite and three {LIVE_N * CANVAS}x{CANVAS} label strips", flush=True)

# instance ids for the same twelve, so the instance widget can outline objects
inst_rows = []
for i in live_idx:
    own = S_TE["owner"][i]
    objs = [r for r in S_TE["boxes"] if int(r[0]) == i]
    ent = []
    for k, r in enumerate(objs):
        m = own == k
        if not m.any():
            continue
        ys, xs = np.where(m)
        ent.append({"cls": int(r[1]), "box": [int(r[2]), int(r[3]), int(r[4]), int(r[5])],
                    "pixels": int(m.sum()),
                    "mask_box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                 int(ys.max()) + 1]})
    dists = int((own >= DIST_ID).sum())
    inst_rows.append({"index": i, "objects": ent, "distractor_pixels": dists})

flat, meta, off = [], [], 0
for k, (w, b) in enumerate(LAYERS):
    wn = w.numpy().astype(np.float32)
    bn_ = b.numpy().astype(np.float32)
    meta.append({"i": k, "shape": list(w.shape), "w_offset": off, "w_size": int(wn.size),
                 "b_offset": off + int(wn.size), "b_size": int(bn_.size)})
    flat.append(wn.reshape(-1))
    flat.append(bn_.reshape(-1))
    off += int(wn.size) + int(bn_.size)
vals32 = np.concatenate(flat).astype(np.float32)
vals16 = vals32.astype(np.float16)
deq = vals16.astype(np.float64)
QL = []
for m in meta:
    w = torch.tensor(deq[m["w_offset"]:m["w_offset"] + m["w_size"]]).view(*m["shape"])
    b = torch.tensor(deq[m["b_offset"]:m["b_offset"] + m["b_size"]])
    QL.append((w, b))
b64 = base64.b64encode(vals16.tobytes()).decode("ascii")

fold_gap = 0.0
with torch.no_grad():
    for i in range(4):
        x = torch.zeros(1, 2, CANVAS, CANVAS)
        x[0, 0] = torch.tensor(S_TE["canvas"][i] / 255.0, dtype=torch.float32)
        x[0, 1] = torch.tensor(prompt_channel(32, 32))
        a = promptnet(x)
        b = folded_forward(LAYERS, x.double()).float()
        fold_gap = max(fold_gap, float((a - b).abs().max()))
quant_gap = float(np.abs(deq - vals32.astype(np.float64)).max())
print(f"  {len(vals32):,} folded values, float16 base64 is {len(b64) / 1024:.0f} kB; "
      f"fold gap {fold_gap:.2e}, weight quantisation {quant_gap:.2e}", flush=True)

# The reference the browser reproduces: computed in float64 with the QUANTISED
# weights, so quantisation cannot show up in the printed agreement.
live_rows = []
for n, i in enumerate(live_idx):
    own = S_TE["owner"][i]
    objs = [r for r in S_TE["boxes"] if int(r[0]) == i]
    prompts = []
    for k, r in enumerate(objs):
        pts = np.argwhere(own == k)
        if len(pts) == 0:
            continue
        py, px = pts[len(pts) // 2]
        x = torch.zeros(1, 2, CANVAS, CANVAS, dtype=torch.float64)
        x[0, 0] = torch.tensor(S_TE["canvas"][i] / 255.0, dtype=torch.float64)
        x[0, 1] = torch.tensor(prompt_channel(int(px), int(py)), dtype=torch.float64)
        with torch.no_grad():
            lg = folded_forward(QL, x)[0, 0]
        pr = (torch.sigmoid(lg).numpy() >= 0.5)
        gt = own == k
        prompts.append({"x": int(px), "y": int(py), "cls": int(r[1]),
                        "iou": rnd(mask_iou(pr, gt), 6),
                        "pred_pixels": int(pr.sum()), "true_pixels": int(gt.sum()),
                        "logit_checksum": rnd(float(np.abs(lg.numpy()).sum()), 4)})
    live_rows.append({"index": i, "prompts": prompts})

pixel_check = {"index": 0, "row": 32,
               "values": [int(v) for v in S_TE["canvas"][0][32]],
               "note": "the browser must read exactly these integers out of the sprite "
                       "before anything downstream can be believed"}

prompt_json = {
    "format": "folded convolutions, float16 little-endian, base64",
    "dtype": "float16", "n_values": int(len(vals32)), "layers": meta, "weights_b64": b64,
    "arch": {"channels": UNet.CH, "kernel": 3, "pad": 1, "levels": 2,
             "in_channels": 2, "out_channels": 1, "canvas": CANVAS,
             "sigma": PROMPT_SIGMA,
             "input": "channel 0 is the canvas divided by 255, channel 1 is "
                      "exp(-d^2 / (2 sigma^2)) with d the distance in pixels to the "
                      "point the reader clicked",
             "order": "e1 at 64, pool, e2 at 32, pool, e3 at 16, upsample by nearest "
                      "and concatenate e2, d2, upsample and concatenate e1, d1, then "
                      "a 1x1 head; every convolution is 3x3 with padding 1 except the "
                      "head, and every one but the head is followed by ReLU",
             "output": "one logit per pixel; a pixel is in the mask when its "
                       "sigmoid is at least 0.5"},
    "errors": {"fold_abs": rnd(fold_gap, 12), "weight_quantisation_abs": rnd(quant_gap, 12),
               "note": "the reference masks below were computed in float64 with the "
                       "quantised weights, so quantisation cannot appear in the "
                       "browser's printed agreement; what remains is dtype alone"},
    "live": {"sprite": "canvases.png", "tile": CANVAS, "count": LIVE_N,
             "pixel_check": pixel_check, "rows": live_rows},
}
(OUT / "promptnet.json").write_text(json.dumps(prompt_json, separators=(",", ":")),
                                    encoding="utf-8")

out = {
    "meta": {"article": "segmentation", "chip": "U-Net, Mask R-CNN and SAM",
             "seed": SEED, "canvas": CANVAS, "classes": DET_CLASSES,
             "mask_convention": "the visible mask: the pixels an object actually "
                                "won in the composite, not the pixels it would have "
                                "covered alone"},
    "dataset": dataset_block,
    "metric": GEO,
    "unet": {"arms": UNET_ARMS, "seed_spread": rnd(seed_spread),
             "regions": REGIONS,
             "n_train_used": N_UNET_TRAIN, "n_train_available": N_TRAIN,
             "skip_delta": rnd(skip_delta),
             "skip_resolved": bool(abs(skip_delta) > seed_spread),
             "band_share": rnd(float(BAND_TE.mean())),
             "control": "the two arms are the same network to the last weight: the "
                        "skip tensors are replaced by zeros rather than removed, so "
                        "parameters, multiply-accumulates and every layer shape are "
                        "identical and only the information flow differs",
             "epochs": EPOCHS},
    "instances": MERGE,
    "mask_rcnn": MASKRCNN,
    "promptable": PROMPT,
    "images": {"sprite": "canvases.png", "truth": "sem-truth.png",
               "skips": "sem-skips.png", "noskips": "sem-noskips.png",
               "owner": "owner.png",
               "owner_encoding": "0 background, 1 to 5 the garments in class order, "
                                 "200 and up the five held-out classes this page never "
                                 "trained anything on",
               "tile": CANVAS, "count": LIVE_N,
               "encoding": "one byte per pixel, the value IS the class index: 0 "
                           "background and 1 to 5 the garment classes in order"},
    "live_instances": inst_rows,
    "versions": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH,
                 "torch": str(torch.__version__)},
    "runtime": {"total_seconds": rnd(elapsed(), 1), "threads": THREADS,
                "note": "wall clock on one machine with torch.set_num_threads(THREADS); a "
                        "different thread count moves it a lot, so this is provenance rather "
                        "than a portable constant"},
}
path = OUT / "segment.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
print(f"wrote {OUT / 'promptnet.json'}  "
      f"({(OUT / 'promptnet.json').stat().st_size / 1024:.1f} kB)")
print(f"total wall clock {elapsed() / 60:.1f} min")
