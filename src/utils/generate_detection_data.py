"""Data for the ATLAS 'Detection' web article (module 1.3): YOLO, SSD, Faster R-CNN.

The Viola-Jones article asked WHAT is in this window and answered it by trying
every window. This one asks WHERE, and the whole article is the bill for that
question and the tricks that reduce it.

Everything here is measured on a synthetic detection set built in this file:
Fashion-MNIST garments pasted at random positions and scales onto a 64x64
canvas, so every ground-truth box is exact and precision, recall and mAP are
exact rather than approximate.

Measured here:
  * E0 the geometry: IoU against centre shift and against scale error, in
    closed form and cross-checked against the box formula; the ten-class ink
    aspect table that decides which five classes the article uses; the
    crowding, size and visible-ink distributions of the generated set.
  * E1 grid assignment: for strides 4, 8, 16 and 1, 2, 3 slots per cell, how
    many objects collide in a cell, how many get no slot at all, and the
    recall ceiling that follows, computed under the SAME matching rule the
    mAP uses so that prediction and measurement are commensurable.
  * E2 the window bill: every window count derived from an explicit list of
    window shapes and asserted, never quoted; the same in multiply-accumulates
    from a counter that walks the modules; the sharing factor placed next to
    the area redundancy 64 x 784 / 4096, because they are the same number up
    to border effects; article 19's own cascade counts read out of
    viola-jones/data/cascade.json and cross-checked.
  * E3 the anchor factorial: one slot against three identical slots against
    three k-means slots against three deliberately wrong slots, one box
    parameterisation shared by all four, so slot multiplicity, prior diversity
    and prior quality are separated instead of confounded.
  * E4 grid resolution: the same arm at stride 16 and stride 4, moved only by
    relocating a pooling layer, so the layer list and the parameter count are
    genuinely fixed.
  * E5 sliding a classifier, trained the way article 19 trained one, hard
    negative mining included, both rounds reported; plus the measurement that
    a grid cell and a sliding window are the same quantity, done on the window
    classifier (exact) and on the detector (not exact, and the border cells
    are where it is not).
  * E6 non-maximum suppression: what a wrong threshold costs, as exact counts
    of true objects suppressed and duplicates surviving, with the threshold
    chosen on validation and reported on test.
  * E7 what confidence should be: objectness, class probability, and their
    product, plus the empirical precision of each score bin.
  * E8 two stages: proposals from the shared feature map, RoI-aligned on that
    map rather than re-cropped from pixels, with the ablations that separate
    rejection from refinement from simply having had more gradient steps.
  * E9 baselines and a failure taxonomy, including whether a missed object was
    ever present in the raw grid before scoring and suppression threw it away.
  * E10 the export: batch normalisation folded away, weights quantised BEFORE
    the reference boxes are computed, so the browser reproduces the generator
    rather than approximating it, and the residual disagreement is attributable
    to one thing only.

Uncertainty is not optional here: nine arms separated by a hundredth of a
point of mAP mean nothing without it. Every mAP ships with a bootstrap
interval over test images (valid because detection matching is within-image,
so resampling whole images preserves the true-positive labels exactly), and
the two load-bearing arms are replicated across seeds. The generator prints
'not resolved' itself when a difference is smaller than the seed spread.

The runtime budget is measured rather than assumed: the calibration model
named in the brief is trained for one epoch first, and the arm list is trimmed
from a pre-declared ladder if this machine turns out to be slower than the
brief's calibration point. What was dropped is exported.

Run from anywhere: `python src/utils/generate_detection_data.py`.
"""
import base64
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.measure import label as sklabel
from skimage.transform import resize as skresize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_data import fashion_mnist, subset, FASHION_CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "detection" / "data"
IMG = ROOT / "detection" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

# ------------------------------------------------------------------ constants
SEED = 19
CANVAS = 64
INK = 32                     # a pixel counts as ink at or above this value
MIN_SIDE, MAX_SIDE = 16, 41  # the OBJECT's longest side, uniform on [16, 40]
N_TRAIN = 6000
N_VAL = 1500
N_TEST_MAX = 2500
BATCH = 64
BASE_LR = 0.02
SCORE_FLOOR = 0.01
PRE_NMS_TOPK = 40        # candidates kept per image before suppression
SLIDE_TOPK = 100         # the same cap for the sliding-window baseline
MAX_OBJ = 3
N_BOOT = 200

# The five classes, chosen on the aspect table measured in E0 below and not
# quoted from anywhere. Ordered by ink aspect, widest last.
DET_CLASSES = ["trouser", "dress", "t-shirt", "bag", "sneaker"]
DET_SOURCE = [1, 3, 0, 8, 7]
# The five held out, which become the distractors: a false positive in this
# article is a genuinely object-like thing that is not one of the five, which
# is what article 19's mined negatives were.
HELD_CLASSES = ["pullover", "coat", "shirt", "sandal", "ankle boot"]
HELD_SOURCE = [2, 4, 6, 5, 9]
N_CLS = len(DET_CLASSES)

# The sliding-window baseline's shape list. Square windows alone cannot reach
# IoU 0.5 against a trouser (ink aspect 0.42), so the baseline scans three
# aspect ratios, which is also what makes its window bill honest.
SCAN_SIZES = [16, 22, 28, 34, 40]
SCAN_ASPECTS = [0.5, 1.0, 2.0]
WIN = 24                     # the window classifier's input side
WIN_STRIDE = 4               # its network stride, two 2x2 pools

torch.set_num_threads(12)
torch.manual_seed(SEED)

CACHE = Path.home() / ".atlas_vision_data" / "detection_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def cached(name, fn):
    """A finished experiment is written outside the repository and reused.
    The cache stores results, not shortcuts: deleting it reproduces every
    number from scratch."""
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


def cached_arm(name, fn):
    """A trained arm is a json row plus a state dict; both must be present for
    the cache to count as a hit."""
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


def elapsed():
    return time.perf_counter() - T0


# ===================================================================== data in
Ftr_x, Ftr_y, Fte_x, Fte_y = fashion_mnist()
print(f"Fashion-MNIST source: {len(Ftr_y)} train items, {len(Fte_y)} test items, "
      f"{len(FASHION_CLASSES)} classes", flush=True)


# ============================================================= M0 the governor
# The brief's calibration point is 5.0 s/epoch for a 20-layer residual network
# of width 16 on 6000 28x28 images. That number decides how many arms fit in
# the budget, so it is measured on this machine before anything else is spent.
class CalBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        f = self.b2(self.c2(self.act(self.b1(self.c1(x)))))
        return self.act(f + (x if self.short is None else self.short(x)))


class CalNet(nn.Module):
    """The brief's calibration network: resnet-20, width 16, ten classes."""

    def __init__(self, depth=20, width=16):
        super().__init__()
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        blocks, cin = [], width
        for cout, stride in [(width, 1), (width * 2, 2), (width * 4, 2)]:
            for b in range(n):
                blocks.append(CalBlock(cin, cout, stride if b == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(cin, 10))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def measure_calibration():
    Xc_u8, yc_np = subset(Ftr_x, Ftr_y, 6000, seed=SEED)
    Xc = torch.tensor(Xc_u8, dtype=torch.float32).unsqueeze(1) / 255.0
    yc = torch.tensor(yc_np, dtype=torch.long)
    torch.manual_seed(SEED)
    m = CalNet()
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(yc), generator=g)
    m.train()
    t0 = time.perf_counter()
    for i in range(0, len(yc), 128):
        idx = perm[i:i + 128]
        opt.zero_grad()
        loss = lossf(m(Xc[idx]), yc[idx])
        loss.backward()
        opt.step()
    secs = time.perf_counter() - t0
    return {"seconds": rnd(secs, 2), "params": sum(p.numel() for p in m.parameters()),
            "n": 6000, "batch": 128, "threads": 12}


print("\nE-0 runtime governor, part one: the brief's calibration model", flush=True)
calib = cached("calibration-r20w16-n6000-b128-t12", measure_calibration)
CONTENTION = calib["seconds"] / 5.0
print(f"  calibration epoch {calib['seconds']:.1f}s against the brief's 5.0s, "
      f"a contention factor of {CONTENTION:.2f}", flush=True)
# The tier is NOT decided here. Contention against a different architecture
# turned out to be a poor predictor of this article's own arm cost (measured:
# a factor of 1.8 out), so the decision waits until part two below has timed
# the real training step. This is the whole point of measuring rather than
# assuming, applied to the governor itself.
# The test set is never shrunk: evaluating 2500 canvases costs seconds, not
# minutes, so trading it away would buy nothing and would fork the cache
# between runs. The ladder trades epochs and arms instead, which is where the
# time actually is.
N_TEST = N_TEST_MAX
DCONFIG = f"fscenes5-c64-tr{N_TRAIN}-va{N_VAL}-te{N_TEST_MAX}-s{SEED}"


# ============================================================ small box helpers
def iou_xyxy(a, b):
    """IoU of two half-open boxes [x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_iou_np(a, b):
    """(N,4) against (M,4), returning (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float64)
    ix = np.maximum(0.0, np.minimum(a[:, None, 2], b[None, :, 2])
                    - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0.0, np.minimum(a[:, None, 3], b[None, :, 3])
                    - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def shape_iou(wh_a, wh_b):
    """IoU of two box SHAPES aligned at a common centre, which is the quantity
    anchor assignment and IoU-k-means both use."""
    inter = np.minimum(wh_a[..., 0], wh_b[..., 0]) * np.minimum(wh_a[..., 1], wh_b[..., 1])
    union = wh_a[..., 0] * wh_a[..., 1] + wh_b[..., 0] * wh_b[..., 1] - inter
    return inter / np.maximum(union, 1e-12)


def group_index(img, n):
    """Detections grouped by image once, so nothing downstream has to scan the
    whole detection array per image. Returns an ordering and the offsets into
    it, so image i owns order[starts[i]:starts[i + 1]]."""
    order = np.argsort(img, kind="stable")
    counts = np.bincount(img.astype(np.int64), minlength=n)[:n]
    starts = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=starts[1:])
    return order, starts


def topk_per_image(dets, n_img, k):
    """Keep the k highest-scoring candidates per image before suppression.

    Standard practice, and here it also bounds the cost of everything
    downstream: an untrained head passes all 192 candidates through the score
    floor, and the suppression sweep would then be quadratic in a number that
    depends on how well training went. k is far above the two or three objects
    a canvas contains, so nothing that could be matched is discarded.
    """
    order, starts = group_index(dets["img"], n_img)
    keep = []
    for i in range(n_img):
        sel = order[starts[i]:starts[i + 1]]
        if len(sel) > k:
            sel = sel[np.argsort(-dets["score"][sel])[:k]]
        keep.append(sel)
    keep = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    return {kk: v[keep] for kk, v in dets.items()}


def coverage(dets, gts, n_img, thr=0.5):
    """For every ground-truth box, whether ANY detection of its own class
    reaches IoU >= thr with it. Used for recall, for the suppression counts and
    for the failure taxonomy, so all three agree by construction."""
    order, starts = group_index(dets["img"], n_img)
    out = []
    for i in range(n_img):
        sel = order[starts[i]:starts[i + 1]]
        g = gts[i]
        c = np.zeros(len(g), bool)
        for j, r in enumerate(g):
            same = sel[dets["cls"][sel] == int(r[0])]
            if len(same) and box_iou_np(r[None, 1:], dets["box"][same]).max() >= thr:
                c[j] = True
        out.append(c)
    return out


# ===================================================================== M1 data
def resize_tile(img_u8, s):
    r = skresize(img_u8.astype(np.float64) / 255.0, (s, s), order=1, anti_aliasing=True)
    return np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)


def ink_box(tile):
    """Tight box of the ink, half-open [x1, y1, x2, y2). Computed on the
    RESIZED sprite and before compositing, so it is exact by construction."""
    m = tile >= INK
    if not m.any():
        return None
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return (int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1)


def sprite_for(img_u8, target_long):
    """A sprite whose OBJECT, not whose tile, has the requested longest side.

    Scaling the 28x28 tile instead would mean a trouser at 'size 16' has an ink
    box 6 px wide, because a trouser fills a quarter of its tile's width. The
    size sampled in this article is therefore the size of the thing, which is
    also what makes the size terciles in the recall table mean anything.
    """
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
    """Build one split of canvases.

    Compositing is a pixel-wise maximum, so there is NO occlusion: the result
    is the union of the silhouettes with the brighter pixel winning, and the
    boxes stay amodal. The visible ink fraction below is the honest
    measurement of what that costs.
    """
    canvases = np.zeros((n, CANVAS, CANVAS), np.uint8)
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
        for (_c, _b, tile, ox, oy) in placed + dist:
            s = tile.shape[0]
            cv[oy:oy + s, ox:ox + s] = np.maximum(cv[oy:oy + s, ox:ox + s], tile)
        canvases[i] = cv
        for (c, box, tile, ox, oy) in placed:
            s = tile.shape[0]
            m = tile >= INK
            sub = cv[oy:oy + s, ox:ox + s]
            vis.append(float((sub[m] == tile[m]).mean()) if m.any() else 0.0)
            rows.append([i, c, box[0], box[1], box[2], box[3]])
        for (_c, box, _t, _ox, _oy) in dist:
            drows.append([i, box[0], box[1], box[2], box[3]])
    return {"canvas": canvases,
            "boxes": np.array(rows, np.float32).reshape(-1, 6),
            "vis": np.array(vis, np.float32),
            "dboxes": np.array(drows, np.float32).reshape(-1, 5),
            "crowded": crowded,
            "rejected": np.array([rejected], np.int64)}


print("\nE-1 dataset: fashion scenes, 64x64 canvases with exact boxes", flush=True)
tr_pool = class_pools(Ftr_y, DET_SOURCE)
tr_dpool = class_pools(Ftr_y, HELD_SOURCE)
va_pool = class_pools(Fte_y, DET_SOURCE, 0.0, 0.5)
va_dpool = class_pools(Fte_y, HELD_SOURCE, 0.0, 0.5)
te_pool = class_pools(Fte_y, DET_SOURCE, 0.5, 1.0)
te_dpool = class_pools(Fte_y, HELD_SOURCE, 0.5, 1.0)

D_TR = cached_npz(f"{DCONFIG}-train", lambda: synth_split(
    N_TRAIN, Ftr_x, tr_pool, tr_dpool, 0, 0.35, 0.0, 0.35))
D_VA = cached_npz(f"{DCONFIG}-val", lambda: synth_split(
    N_VAL, Fte_x, va_pool, va_dpool, 1000000, 0.35, 0.20, 0.60))
D_TE = cached_npz(f"{DCONFIG}-test", lambda: synth_split(
    N_TEST_MAX, Fte_x, te_pool, te_dpool, 2000000, 0.35, 0.20, 0.60))


def as_tensor(d):
    return torch.tensor(d["canvas"], dtype=torch.float32).unsqueeze(1) / 255.0


Xtr = as_tensor(D_TR)
Xva = as_tensor(D_VA)
Xte = as_tensor(D_TE)[:N_TEST]
Btr, Bva = D_TR["boxes"], D_VA["boxes"]
Bte = D_TE["boxes"][D_TE["boxes"][:, 0] < N_TEST]
Dte = D_TE["dboxes"][D_TE["dboxes"][:, 0] < N_TEST]
print(f"  train {N_TRAIN} canvases / {len(Btr)} boxes, val {N_VAL} / {len(Bva)}, "
      f"test {N_TEST} / {len(Bte)}; {int(D_TR['rejected'][0])} placements rejected in train",
      flush=True)


def boxes_by_image(B, n):
    out = [[] for _ in range(n)]
    for r in B:
        out[int(r[0])].append(r[1:])
    return [np.array(v, np.float32).reshape(-1, 5) for v in out]


GT_TR = boxes_by_image(Btr, N_TRAIN)
GT_VA = boxes_by_image(Bva, N_VAL)
GT_TE = boxes_by_image(Bte, N_TEST)


def pad_gt(gts):
    p = np.zeros((len(gts), MAX_OBJ, 4), np.float32)
    n = np.zeros(len(gts), np.int64)
    for i, g in enumerate(gts):
        k = min(len(g), MAX_OBJ)
        p[i, :k] = g[:k, 1:]
        n[i] = k
    return torch.tensor(p), torch.tensor(n)


GTPAD_TR, GTN_TR = pad_gt(GT_TR)


# =========================================================== E0 the geometry
print("\nE0 geometry: what IoU actually is, and what these canvases contain", flush=True)


def aspect_table():
    """Ink-box aspect (w/h) of every Fashion-MNIST class at its native 28x28.
    This is the evidence the five-class choice rests on, exported in full so a
    reader sees the selection rather than being told about it."""
    rows = []
    for c, name in enumerate(FASHION_CLASSES):
        idx = np.where(Ftr_y == c)[0][:800]
        vals = []
        for i in idx:
            bx = ink_box(Ftr_x[i])
            if bx is not None and bx[3] > bx[1]:
                vals.append((bx[2] - bx[0]) / (bx[3] - bx[1]))
        vals = np.array(vals)
        rows.append({"cls": name, "index": c, "aspect": rnd(vals.mean()),
                     "sd": rnd(vals.std()), "n": len(vals),
                     "used": name in DET_CLASSES})
    return rows


ASPECT = cached("aspect-table-fm800", aspect_table)
used = [r for r in ASPECT if r["used"]]
spread = max(r["aspect"] for r in used) / min(r["aspect"] for r in used)
srt = sorted(r["aspect"] for r in used)
min_gap = min(srt[i + 1] / srt[i] for i in range(len(srt) - 1))
sandal = next(r for r in ASPECT if r["cls"] == "sandal")
print(f"  aspect spread across the five chosen classes {spread:.2f}x, smallest adjacent "
      f"gap {min_gap:.2f}x; sandal excluded, its aspect sd is {sandal['sd']:.2f}", flush=True)

# IoU against a centre shift and against a scale error, in closed form and
# cross-checked against the box formula on rasterised boxes. Nothing quoted.
iou_geom = {"sizes": [16, 28, 40], "shift": [], "scale": [], "max_closed_gap": 0.0}
gap = 0.0
for L in iou_geom["sizes"]:
    srow, crow = [], []
    for d in range(0, L + 1):
        closed_axis = (L - d) / (L + d) if d < L else 0.0
        a = (0.0, 0.0, float(L), float(L))
        b = (float(d), 0.0, float(L + d), float(L))
        gap = max(gap, abs(closed_axis - iou_xyxy(a, b)))
        closed_diag = ((L - d) ** 2) / (2 * L * L - (L - d) ** 2) if d < L else 0.0
        bd = (float(d), float(d), float(L + d), float(L + d))
        gap = max(gap, abs(closed_diag - iou_xyxy(a, bd)))
        srow.append({"d": d, "axis": rnd(closed_axis, 5), "diag": rnd(closed_diag, 5)})
    for f100 in range(50, 201, 5):
        f = f100 / 100.0
        inter = (min(1.0, f) * L) ** 2
        closed = inter / (L * L + (f * L) ** 2 - inter)
        s = f * L
        a = (0.0, 0.0, float(L), float(L))
        b = ((L - s) / 2, (L - s) / 2, (L + s) / 2, (L + s) / 2)
        gap = max(gap, abs(closed - iou_xyxy(a, b)))
        crow.append({"f": rnd(f, 2), "iou": rnd(closed, 5)})
    iou_geom["shift"].append({"L": L, "rows": srow})
    iou_geom["scale"].append({"L": L, "rows": crow})
iou_geom["max_closed_gap"] = rnd(gap, 12)
# The shift that costs exactly half the IoU, solved rather than rounded.
iou_geom["iou50_shift_axis_px"] = rnd(28.0 / 3.0, 4)
iou_geom["iou50_shift_diag_px"] = rnd(28.0 * (1 - math.sqrt(2.0 / 3.0)), 4)
iou_geom["ink_threshold"] = INK
print(f"  closed form against the box formula: max gap {gap:.2e}; a 28 px box loses half "
      f"its IoU at {iou_geom['iou50_shift_axis_px']:.3f} px along one axis, "
      f"{iou_geom['iou50_shift_diag_px']:.3f} px diagonally", flush=True)

# A reference table the browser checks its own IoU against.
rs = np.random.RandomState(SEED)
iou_ref = []
for _ in range(24):
    a = rs.randint(0, 40, 2)
    wa = rs.randint(8, 30, 2)
    b = rs.randint(0, 40, 2)
    wb = rs.randint(8, 30, 2)
    A = (float(a[0]), float(a[1]), float(a[0] + wa[0]), float(a[1] + wa[1]))
    B = (float(b[0]), float(b[1]), float(b[0] + wb[0]), float(b[1] + wb[1]))
    iou_ref.append({"a": list(A), "b": list(B), "iou": rnd(iou_xyxy(A, B), 8)})

# What the canvases contain. The size and shape distributions are not exported
# as histograms: the article draws 900 individual test boxes as a scatter, which
# shows both margins at once and shows them jointly.
PAIR_HIST_MAX = 0.7
per_img = np.array([len(g) for g in GT_TE])
max_pair = np.zeros(N_TEST)
for i, g in enumerate(GT_TE):
    if len(g) > 1:
        m = box_iou_np(g[:, 1:], g[:, 1:])
        np.fill_diagonal(m, 0.0)
        max_pair[i] = m.max()
ISO = np.where(max_pair < 0.05)[0]
CROWD = np.where(max_pair > 0.20)[0]
vis_te = D_TE["vis"][:len(Bte)]
dataset_block = {
    "name": "fashion scenes", "canvas": CANVAS, "classes": DET_CLASSES,
    "held_out": HELD_CLASSES, "ink_threshold": INK,
    "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
    "boxes_train": len(Btr), "boxes_val": len(Bva), "boxes_test": len(Bte),
    "distractors_test": len(Dte),
    "sprite_side": [MIN_SIDE, MAX_SIDE - 1],
    "rejected_train": int(D_TR["rejected"][0]),
    "aspect_table": ASPECT,
    "aspect_spread": rnd(spread), "aspect_min_gap": rnd(min_gap),
    "objects_hist": [int((per_img == k).sum()) for k in range(0, MAX_OBJ + 1)],
    # The upper edge travels with the counts: a histogram whose range the
    # reader has to guess is a histogram drawn on the wrong axis.
    "pair_iou_hist": np.histogram(max_pair, bins=20, range=(0, PAIR_HIST_MAX))[0].tolist(),
    "pair_iou_hist_max": PAIR_HIST_MAX,
    "visible_ink": {"mean": rnd(float(vis_te.mean())),
                    "p05": rnd(float(np.percentile(vis_te, 5))),
                    "p50": rnd(float(np.percentile(vis_te, 50))),
                    "min": rnd(float(vis_te.min()))},
    "isolated_test": len(ISO), "crowded_test": len(CROWD),
    "crowded_stratum_frac": 0.20, "iou_cap": 0.35, "crowd_cap": 0.60,
    "limits": ["compositing is a pixel-wise maximum, so there is no occlusion and "
               "every box is amodal",
               "objects lie fully inside the canvas, so there is no truncation and "
               "the outer ring of cells receives almost no positives",
               "boxes carry no annotation noise, the images are single-channel, "
               "and the splits are drawn from one distribution",
               "the five classes were chosen for aspect diversity, so the anchor "
               "effect measured in E3 is an upper bound for this kind of data"],
}
print(f"  test canvases: {len(ISO)} isolated (max pairwise IoU < 0.05), {len(CROWD)} crowded "
      f"(> 0.20); the most overlapped 5% of objects keep "
      f"{dataset_block['visible_ink']['p05'] * 100:.0f}% of their ink", flush=True)


# ================================================================ IoU k-means
def iou_kmeans(wh, k, seed=SEED, iters=60):
    """YOLO's k-means: assignment by 1 - shape IoU, update by the mean of the
    members. At k = 1 the assignment is trivial, so the centroid IS the mean
    box, which the assertion below states rather than leaves to argument."""
    rs = np.random.RandomState(seed)
    cen = wh[rs.choice(len(wh), k, replace=False)].copy()
    for _ in range(iters):
        d = 1.0 - shape_iou(wh[:, None, :], cen[None, :, :])
        a = d.argmin(1)
        new = cen.copy()
        for j in range(k):
            if (a == j).any():
                new[j] = wh[a == j].mean(0)
        if np.allclose(new, cen):
            cen = new
            break
        cen = new
    return cen[np.argsort(cen[:, 0] * cen[:, 1])]


WH_TR = np.stack([Btr[:, 4] - Btr[:, 2], Btr[:, 5] - Btr[:, 3]], 1)
WH_TE = np.stack([Bte[:, 4] - Bte[:, 2], Bte[:, 5] - Bte[:, 3]], 1)
A1 = iou_kmeans(WH_TR, 1)
assert np.allclose(A1[0], WH_TR.mean(0), atol=1e-4), "at k=1 the centroid must be the mean box"
A3 = iou_kmeans(WH_TR, 3)
ANCHOR_SETS = {
    "slots1": A1,
    "slots3-same": np.repeat(A1, 3, axis=0),
    "slots3-kmeans": A3,
    "slots3-square": np.array([[28.0, 28.0]] * 3, np.float32),
}
kcurve = []
for k in range(1, 9):
    cen = iou_kmeans(WH_TR, k)
    best_tr = shape_iou(WH_TR[:, None, :], cen[None, :, :]).max(1)
    best_te = shape_iou(WH_TE[:, None, :], cen[None, :, :]).max(1)
    kcurve.append({"k": k, "train_iou": rnd(best_tr.mean()), "test_iou": rnd(best_te.mean()),
                   "shapes": [[rnd(v, 2) for v in c] for c in cen]})
print(f"  IoU-k-means fitted on train, evaluated on test: k=1 covers "
      f"{kcurve[0]['test_iou']:.3f}, k=3 covers {kcurve[2]['test_iou']:.3f}, "
      f"k=8 covers {kcurve[7]['test_iou']:.3f}", flush=True)


# =========================================================== E1 grid assignment
def build_targets(gts, anchors, stride, n_img):
    """The one thing a one-stage detector decides before it learns anything:
    which slot of which cell is responsible for which object.

    Centre cell, then the best FREE slot by shape IoU at a common centre, ties
    broken by slot index. Depends only on boxes, anchors and stride, so it is
    computed once per arm and cached rather than recomputed every epoch.
    """
    G = CANVAS // stride
    A = len(anchors)
    t_obj = np.zeros((n_img, A, G, G), np.uint8)
    t_box = np.zeros((n_img, A, G, G, 4), np.float32)
    t_cls = np.zeros((n_img, A, G, G), np.int64)
    per_slot = np.zeros(A, np.int64)
    unassignable = 0
    collisions = 0
    offsets = []
    cell_count = np.zeros((G, G), np.int64)
    an = np.asarray(anchors, np.float64)
    for i, g in enumerate(gts):
        occupied = {}
        for row in g:
            c = int(row[0])
            x1, y1, x2, y2 = row[1:]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = x2 - x1, y2 - y1
            col = min(int(cx // stride), G - 1)
            r = min(int(cy // stride), G - 1)
            key = (r, col)
            occupied[key] = occupied.get(key, 0) + 1
            if occupied[key] > 1:
                collisions += 1
            ious = shape_iou(np.array([[w, h]]), an[None, :, :])[0]
            order = np.lexsort((np.arange(A), -ious))
            slot = None
            for a in order:
                if t_obj[i, a, r, col] == 0:
                    slot = int(a)
                    break
            if slot is None:
                unassignable += 1
                continue
            t_obj[i, slot, r, col] = 1
            per_slot[slot] += 1
            cell_count[r, col] += 1
            tw = float(np.clip(math.log(max(w, 1e-6) / an[slot, 0]), -4, 4))
            th = float(np.clip(math.log(max(h, 1e-6) / an[slot, 1]), -4, 4))
            t_box[i, slot, r, col] = [cx / stride - col, cy / stride - r, tw, th]
            t_cls[i, slot, r, col] = c
            offsets.append(cx / stride - col)
    total = sum(len(g) for g in gts)
    return {"t_obj": t_obj, "t_box": t_box, "t_cls": t_cls,
            "stats": np.array([total, unassignable, collisions], np.int64),
            "per_slot": per_slot, "cell_count": cell_count,
            "offsets": np.array(offsets, np.float32)}


def targets_for(anchors, stride):
    """Keyed by the anchors themselves rather than by the arm, because two arms
    with the same anchors and stride have byte-identical targets and assignment
    is 20% of an epoch."""
    tag = "_".join(f"{a[0]:.2f}x{a[1]:.2f}" for a in np.asarray(anchors))
    return cached_npz(f"{DCONFIG}-targets-{tag}-s{stride}",
                      lambda: build_targets(GT_TR, anchors, stride, N_TRAIN))


print("\nE1 grid assignment: who is responsible for each object", flush=True)
grid_rows = []
for stride in (4, 8, 16):
    for A in (1, 2, 3):
        cen = iou_kmeans(WH_TR, A)
        t = build_targets(GT_TE, cen, stride, N_TEST)
        total, unassign, colls = [int(v) for v in t["stats"]]
        cc = t["cell_count"]
        occ = cc[cc > 0]
        # The ceiling, computed under the SAME rule the mAP uses. With the
        # decode w = p * exp(t), t clamped to +-4, any size is reachable and
        # the centre of an object always lies inside its own cell, so the only
        # ceiling is slot capacity: an unassignable object cannot be produced
        # by any slot. The prior-only ceiling below is the different, weaker
        # thing people usually mean, and both are exported.
        prior_iou = shape_iou(WH_TE[:, None, :], cen[None, :, :]).max(1)
        grid_rows.append({
            "stride": stride, "slots": A, "cells": (CANVAS // stride) ** 2,
            "candidates": (CANVAS // stride) ** 2 * A,
            "boxes": total, "collisions": colls, "unassignable": unassign,
            "capacity_ceiling": rnd(1.0 - unassign / max(total, 1)),
            "prior_only_iou": rnd(float(prior_iou.mean())),
            "prior_only_at50": rnd(float((prior_iou >= 0.5).mean())),
            "mean_per_occupied_cell": rnd(float(occ.mean()) if len(occ) else 0.0),
            "anchors": [[rnd(v, 2) for v in c] for c in cen],
            "per_slot": [int(v) for v in t["per_slot"]],
        })
        if stride == 8:
            cellmap8 = t["cell_count"]
        print(f"  stride {stride:2d}, {A} slot(s): {colls:5d} objects share a cell, "
              f"{unassign:4d} get no slot, capacity ceiling "
              f"{grid_rows[-1]['capacity_ceiling']:.4f}, prior-only mean IoU "
              f"{grid_rows[-1]['prior_only_iou']:.3f}", flush=True)
cellmap = (cellmap8 / max(cellmap8.sum(), 1)).astype(np.float64)
border_share = float(cellmap[0].sum() + cellmap[-1].sum()
                     + cellmap[1:-1, 0].sum() + cellmap[1:-1, -1].sum())
print(f"  the outer ring is 28 of 64 cells and receives {border_share * 100:.1f}% of the "
      f"positives, because no object is ever truncated by the canvas edge", flush=True)


# =============================================================== architectures
class TinyDet(nn.Module):
    """The detector. Four 3x3 convolutions and a 1x1 head, and the grid stride
    is moved ONLY by relocating a pooling layer, so the layer list, the depth
    and the parameter count are identical at strides 4, 8 and 16. Receptive
    field and multiply-accumulates necessarily change, and E2's table says by
    how much."""

    CH = [12, 24, 48, 64]

    def __init__(self, n_anchors, n_classes=N_CLS, stride=8):
        super().__init__()
        assert stride in (4, 8, 16)
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


class WinClf(nn.Module):
    """The sliding-window classifier, deliberately UNPADDED. An unpadded
    network computes the same function on a crop and on the corresponding cell
    of a full map; a padded one does not, and E5 measures both."""

    def __init__(self, n_classes=N_CLS):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3)
        self.c2 = nn.Conv2d(8, 16, 3)
        self.c3 = nn.Conv2d(16, 32, 3)
        self.out = nn.Conv2d(32, n_classes + 1, 2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))
        x = self.pool(self.act(self.c2(x)))
        x = self.act(self.c3(x))
        return self.out(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, shape):
    """Multiply-accumulates, counted by walking the modules during a real
    forward pass rather than by hand."""
    total = [0]
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total[0] += mod.weight.numel() * out.shape[-1] * out.shape[-2]
        elif isinstance(mod, nn.Linear):
            total[0] += mod.in_features * mod.out_features * (out.numel() // out.shape[-1])

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(hook))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape))
    for h in hooks:
        h.remove()
    model.train(was)
    return int(total[0])


# ============================================================ decode and loss
def anchor_tensor(anchors):
    return torch.tensor(np.asarray(anchors, np.float32)).view(1, -1, 1, 1, 2)


def grid_offsets(G):
    ys, xs = torch.meshgrid(torch.arange(G, dtype=torch.float32),
                            torch.arange(G, dtype=torch.float32), indexing="ij")
    return xs.view(1, 1, G, G), ys.view(1, 1, G, G)


def decode(raw, an_t, stride, gx, gy, clip=False):
    """One box parameterisation for every arm, anchor-free included:
        x = (col + sigmoid(tx)) * stride,  w = prior_w * exp(clamp(tw, -4, 4)).
    The anchor-free arm's prior is the k=1 centroid, so it differs from the
    one-slot anchor arm in nothing at all and from the three-slot arm in slot
    count alone. That is what makes E3 a controlled comparison."""
    cx = (gx + torch.sigmoid(raw[..., 0])) * stride
    cy = (gy + torch.sigmoid(raw[..., 1])) * stride
    w = an_t[..., 0] * torch.exp(raw[..., 2].clamp(-4, 4))
    h = an_t[..., 1] * torch.exp(raw[..., 3].clamp(-4, 4))
    box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    # Clipping to the canvas is an INFERENCE step, applied identically to every
    # arm and to the browser. Training leaves it off so the box loss keeps a
    # gradient on predictions that currently hang over the edge.
    return box.clamp(0.0, float(CANVAS)) if clip else box


def iou_batch(pred, gt):
    """(B,P,4) against (B,M,4) giving (B,P,M)."""
    ix = (torch.min(pred[:, :, None, 2], gt[:, None, :, 2])
          - torch.max(pred[:, :, None, 0], gt[:, None, :, 0])).clamp_min(0)
    iy = (torch.min(pred[:, :, None, 3], gt[:, None, :, 3])
          - torch.max(pred[:, :, None, 1], gt[:, None, :, 1])).clamp_min(0)
    inter = ix * iy
    pa = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    ga = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    return inter / (pa[:, :, None] + ga[:, None, :] - inter).clamp_min(1e-9)


LOSS_W = {"box": 5.0, "obj_pos": 1.0, "obj_neg": 0.5, "cls": 1.0}


def det_loss(raw, tobj, tbox, tcls, gtb, gtn, an_t, stride, gx, gy, box_mode="smoothl1"):
    """Four terms, each normalised by its own count so the weights below are
    weights on MEANS and a 100:1 negative-to-positive ratio cannot quietly
    become the whole objective."""
    B, A, G, _, _ = raw.shape
    pos = tobj > 0
    npos = int(pos.sum())
    if npos == 0:
        return raw.sum() * 0.0, {}
    pred = decode(raw, an_t, stride, gx, gy)
    with torch.no_grad():
        ious = iou_batch(pred.reshape(B, -1, 4), gtb)
        valid = (torch.arange(gtb.shape[1])[None, :] < gtn[:, None]).float()
        maxiou = (ious * valid[:, None, :]).amax(-1).view(B, A, G, G)
        ignore = (maxiou > 0.5) & (~pos)          # YOLOv3's ignore rule
    if box_mode == "giou":
        p = pred[pos]
        g = torch.stack([
            (gx.expand(B, A, G, G)[pos] + tbox[..., 0][pos]) * stride
            - an_t[..., 0].expand(B, A, G, G)[pos] * torch.exp(tbox[..., 2][pos]) / 2,
            (gy.expand(B, A, G, G)[pos] + tbox[..., 1][pos]) * stride
            - an_t[..., 1].expand(B, A, G, G)[pos] * torch.exp(tbox[..., 3][pos]) / 2,
            (gx.expand(B, A, G, G)[pos] + tbox[..., 0][pos]) * stride
            + an_t[..., 0].expand(B, A, G, G)[pos] * torch.exp(tbox[..., 2][pos]) / 2,
            (gy.expand(B, A, G, G)[pos] + tbox[..., 1][pos]) * stride
            + an_t[..., 1].expand(B, A, G, G)[pos] * torch.exp(tbox[..., 3][pos]) / 2], -1)
        ix = (torch.min(p[:, 2], g[:, 2]) - torch.max(p[:, 0], g[:, 0])).clamp_min(0)
        iy = (torch.min(p[:, 3], g[:, 3]) - torch.max(p[:, 1], g[:, 1])).clamp_min(0)
        inter = ix * iy
        pa = (p[:, 2] - p[:, 0]).clamp_min(0) * (p[:, 3] - p[:, 1]).clamp_min(0)
        ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        union = (pa + ga - inter).clamp_min(1e-9)
        cw = torch.max(p[:, 2], g[:, 2]) - torch.min(p[:, 0], g[:, 0])
        ch = torch.max(p[:, 3], g[:, 3]) - torch.min(p[:, 1], g[:, 1])
        area_c = (cw * ch).clamp_min(1e-9)
        box_l = (1.0 - (inter / union - (area_c - union) / area_c)).mean()
    else:
        pxy = torch.sigmoid(raw[..., 0:2])
        twh = raw[..., 2:4].clamp(-4, 4)
        box_l = (F.smooth_l1_loss(pxy[pos], tbox[..., 0:2][pos], reduction="mean")
                 + F.smooth_l1_loss(twh[pos], tbox[..., 2:4][pos], reduction="mean"))
    bce = F.binary_cross_entropy_with_logits(raw[..., 4], pos.float(), reduction="none")
    neg = (~pos) & (~ignore)
    obj_p = bce[pos].mean()
    obj_n = bce[neg].mean() if int(neg.sum()) > 0 else bce.sum() * 0.0
    cls_l = F.cross_entropy(raw[..., 5:][pos], tcls[pos], reduction="mean")
    loss = (LOSS_W["box"] * box_l + LOSS_W["obj_pos"] * obj_p
            + LOSS_W["obj_neg"] * obj_n + LOSS_W["cls"] * cls_l)
    parts = {"box": float(box_l.detach()), "obj_pos": float(obj_p.detach()),
             "obj_neg": float(obj_n.detach()), "cls": float(cls_l.detach()),
             "ignored": int(ignore.sum())}
    return loss, parts


def lr_at(ep, epochs, base):
    if ep < 2:
        return base * (ep + 1) / 2.0
    if ep >= int(0.85 * epochs):
        return base * 0.01
    if ep >= int(0.60 * epochs):
        return base * 0.1
    return base


def fit_detector(model, tgt, anchors, stride, epochs, seed, lr, init_state=None,
                 box_mode="smoothl1", lr_mode="schedule"):
    """SGD with momentum, identical in every arm. Gradient norm clipped at 10
    and a finiteness assertion every step, because a probe of exactly these
    settings went to NaN and a silently exported NaN mAP is the failure mode
    this guard exists to prevent."""
    torch.manual_seed(seed)
    if init_state is not None:
        model.load_state_dict(init_state)
    an_t = anchor_tensor(anchors)
    G = CANVAS // stride
    gx, gy = grid_offsets(G)
    t_obj = torch.tensor(tgt["t_obj"])
    t_box = torch.tensor(tgt["t_box"])
    t_cls = torch.tensor(tgt["t_cls"])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed)
    history, t0 = [], time.perf_counter()
    for ep in range(epochs):
        for pg in opt.param_groups:
            pg["lr"] = lr if lr_mode == "const" else lr_at(ep, epochs, lr)
        model.train()
        perm = torch.randperm(N_TRAIN, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            raw = model.raw(Xtr[idx])
            loss, _ = det_loss(raw, t_obj[idx], t_box[idx], t_cls[idx],
                               GTPAD_TR[idx], GTN_TR[idx], an_t, stride, gx, gy, box_mode)
            if not torch.isfinite(loss):
                return None, None
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        history.append(rnd(tot / max(nb, 1), 5))
    return history, time.perf_counter() - t0


def train_with_guard(build, tgt, anchors, stride, epochs, seed, box_mode="smoothl1",
                     init_state=None, lr=BASE_LR, lr_mode="schedule"):
    model = build()
    hist, secs = fit_detector(model, tgt, anchors, stride, epochs, seed, lr,
                              init_state, box_mode, lr_mode)
    fallback = False
    if hist is None:
        print("    loss went non-finite: restarting once at half the learning rate",
              flush=True)
        fallback = True
        model = build()
        hist, secs = fit_detector(model, tgt, anchors, stride, epochs, seed, lr / 2,
                                  init_state, box_mode, lr_mode)
        if hist is None:
            raise RuntimeError("training diverged at half the learning rate as well")
    return model, hist, secs, fallback


# =================================================== inference, NMS, precision
@torch.no_grad()
def infer(model, X, anchors, stride, batch=250):
    """Every candidate above the score floor, with the slot it came from, so
    the NMS experiment can be run afterwards on fixed detections."""
    model.eval()
    an_t = anchor_tensor(anchors)
    G = CANVAS // stride
    gx, gy = grid_offsets(G)
    imgs, clss, scores, boxes, slots, objs, cps = [], [], [], [], [], [], []
    # The slot id is the flattened (anchor, row, col) index, so slot // (G*G)
    # recovers which anchor produced a detection.
    for i in range(0, len(X), batch):
        raw = model.raw(X[i:i + batch])
        B = raw.shape[0]
        bx = decode(raw, an_t, stride, gx, gy, clip=True).reshape(B, -1, 4)
        obj = torch.sigmoid(raw[..., 4]).reshape(B, -1)
        cp = torch.softmax(raw[..., 5:], -1).reshape(B, -1, N_CLS)
        cbest, cidx = cp.max(-1)
        sc = obj * cbest
        keep = sc >= SCORE_FLOOR
        bi, pi = torch.nonzero(keep, as_tuple=True)
        imgs.append((bi + i).numpy())
        clss.append(cidx[bi, pi].numpy())
        scores.append(sc[bi, pi].numpy())
        boxes.append(bx[bi, pi].numpy())
        slots.append(pi.numpy())
        objs.append(obj[bi, pi].numpy())
        cps.append(cbest[bi, pi].numpy())
    out = {"img": np.concatenate(imgs).astype(np.int64),
           "cls": np.concatenate(clss).astype(np.int64),
           "score": np.concatenate(scores).astype(np.float64),
           "box": np.concatenate(boxes).astype(np.float64).reshape(-1, 4),
           "slot": np.concatenate(slots).astype(np.int64),
           "obj": np.concatenate(objs).astype(np.float64),
           "clsp": np.concatenate(cps).astype(np.float64)}
    return topk_per_image(out, len(X), PRE_NMS_TOPK)


def nms_indices(box, score, thr, mode="hard", sigma=0.5):
    order = np.argsort(-score)
    keep, sc = [], score.copy()
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        o = box_iou_np(box[i:i + 1], box[rest])[0]
        if mode == "hard":
            rest = rest[o <= thr]
        elif mode == "linear":
            sc[rest] = np.where(o > thr, sc[rest] * (1 - o), sc[rest])
            rest = rest[sc[rest] >= SCORE_FLOOR]
        else:
            sc[rest] = sc[rest] * np.exp(-(o ** 2) / sigma)
            rest = rest[sc[rest] >= SCORE_FLOOR]
        order = rest[np.argsort(-sc[rest])]
    return np.array(keep, np.int64), sc


def apply_nms(dets, thr, classwise=True, mode="hard", n_img=None):
    """Suppression is post-processing: the detections do not change, only which
    of them survive."""
    n_img = n_img if n_img is not None else int(dets["img"].max()) + 1
    keep_all, pair_tests = [], 0
    order = np.argsort(dets["img"] * 100 + (dets["cls"] if classwise else 0))
    img = dets["img"][order]
    cls = dets["cls"][order] if classwise else np.zeros(len(order), np.int64)
    bounds = np.flatnonzero(np.r_[True, (img[1:] != img[:-1]) | (cls[1:] != cls[:-1]), True])
    new_score = dets["score"].copy()
    for s, e in zip(bounds[:-1], bounds[1:]):
        idx = order[s:e]
        k, sc = nms_indices(dets["box"][idx], dets["score"][idx], thr, mode)
        pair_tests += len(idx) * (len(idx) - 1) // 2
        if mode != "hard":
            new_score[idx] = sc
        keep_all.append(idx[k])
    keep = np.concatenate(keep_all) if keep_all else np.zeros(0, np.int64)
    out = {k: v[keep] for k, v in dets.items()}
    if mode != "hard":
        out["score"] = new_score[keep]
        sel = out["score"] >= SCORE_FLOOR
        out = {k: v[sel] for k, v in out.items()}
    return out, pair_tests


def match(dets, gts, iou_thr, n_img):
    """Sorted by confidence descending, ties broken by image then slot; each
    detection matched to the highest-IoU UNMATCHED ground-truth box of the same
    class in the same image; each ground-truth box consumed at most once."""
    tables = {}
    npos_img = np.zeros((N_CLS, n_img), np.int64)
    for i, g in enumerate(gts):
        for r in g:
            npos_img[int(r[0]), i] += 1
    for c in range(N_CLS):
        sel = np.flatnonzero(dets["cls"] == c)
        order = sel[np.lexsort((dets["slot"][sel], dets["img"][sel], -dets["score"][sel]))]
        used = {}
        tp = np.zeros(len(order))
        best_iou = np.zeros(len(order))
        raw_iou = np.zeros(len(order))
        for j, d in enumerate(order):
            i = int(dets["img"][d])
            g = gts[i]
            rows = np.flatnonzero(g[:, 0] == c) if len(g) else np.zeros(0, np.int64)
            if len(rows) == 0:
                continue
            flags = used.setdefault(i, np.zeros(len(g), bool))
            o = box_iou_np(dets["box"][d:d + 1], g[rows, 1:])[0]
            raw_iou[j] = o.max()
            o = np.where(flags[rows], -1.0, o)
            k = int(np.argmax(o))
            best_iou[j] = max(o[k], 0.0)
            if o[k] >= iou_thr:
                flags[rows[k]] = True
                tp[j] = 1.0
        tables[c] = {"order": order, "img": dets["img"][order], "tp": tp,
                     "iou": best_iou, "raw_iou": raw_iou,
                     "score": dets["score"][order], "npos_img": npos_img[c]}
    return tables


def all_point_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def interp_ap(rec, prec, points):
    if len(rec) == 0:
        return 0.0
    mpre = np.maximum.accumulate(prec[::-1])[::-1]
    out = []
    for r in points:
        i = int(np.searchsorted(rec, r, side="left"))
        out.append(mpre[i] if i < len(mpre) else 0.0)
    return float(np.mean(out))


VOC11 = np.linspace(0, 1, 11)
COCO101 = np.linspace(0, 1, 101)


def ap_from_table(tbl, mult=None, kind="all"):
    img, tp, npos_img = tbl["img"], tbl["tp"], tbl["npos_img"]
    if mult is None:
        w = np.ones(len(tp))
        npos = int(npos_img.sum())
    else:
        w = mult[img].astype(np.float64)
        npos = float((npos_img * mult).sum())
    if npos <= 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp * w)
    cfp = np.cumsum((1 - tp) * w)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    if kind == "all":
        return all_point_ap(rec, prec)
    if kind == "voc11":
        return interp_ap(rec, prec, VOC11)
    return interp_ap(rec, prec, COCO101)


def mean_ap(tables, mult=None, kind="all"):
    vals = [ap_from_table(tables[c], mult, kind) for c in range(N_CLS)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def bootstrap_map(tables, n_img, n_boot=N_BOOT, seed=SEED):
    """Resampling whole IMAGES is what makes this valid: matching is
    within-image, so the true-positive labels survive the resample unchanged
    and only the global ranking moves."""
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        mult = rs.multinomial(n_img, np.full(n_img, 1.0 / n_img))
        vals.append(mean_ap(tables, mult))
    v = np.sort(np.array(vals))
    return rnd(float(v[int(0.025 * len(v))])), rnd(float(v[int(0.975 * len(v)) - 1]))


IOU_SWEEP = [rnd(0.5 + 0.05 * i, 2) for i in range(10)]


def evaluate(dets, gts, n_img, boot=True):
    out = {}
    t50 = match(dets, gts, 0.5, n_img)
    t75 = match(dets, gts, 0.75, n_img)
    out["map50"] = rnd(mean_ap(t50))
    out["map75"] = rnd(mean_ap(t75))
    out["map50_voc11"] = rnd(mean_ap(t50, kind="voc11"))
    out["map50_coco101"] = rnd(mean_ap(t50, kind="coco101"))
    out["ap_per_class"] = [rnd(ap_from_table(t50[c])) for c in range(N_CLS)]
    aps = []
    for thr in IOU_SWEEP:
        t = t50 if thr == 0.5 else (t75 if thr == 0.75 else match(dets, gts, thr, n_img))
        aps.append(mean_ap(t))
    out["map5095"] = rnd(float(np.mean(aps)))
    out["map_by_iou"] = [rnd(v) for v in aps]
    if boot:
        lo, hi = bootstrap_map(t50, n_img)
        out["map50_ci"] = [lo, hi]
    matched = np.concatenate([t50[c]["iou"][t50[c]["tp"] > 0] for c in range(N_CLS)])
    out["mean_matched_iou"] = rnd(float(matched.mean()) if len(matched) else 0.0)
    return out, t50


def recall_by_size(dets, gts, n_img, thr=0.5):
    areas = np.array([(r[3] - r[1]) * (r[4] - r[2]) for g in gts for r in g])
    if len(areas) == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0]
    q = [float(np.percentile(areas, 33.3)), float(np.percentile(areas, 66.7))]
    cov = coverage(dets, gts, n_img, thr)
    hit = [0, 0, 0]
    tot = [0, 0, 0]
    for i, g in enumerate(gts):
        for j, r in enumerate(g):
            a = (r[3] - r[1]) * (r[4] - r[2])
            t = 0 if a < q[0] else (1 if a < q[1] else 2)
            tot[t] += 1
            hit[t] += int(cov[i][j])
    return [rnd(h / max(t, 1)) for h, t in zip(hit, tot)], [rnd(v, 1) for v in q]


def pick_nms_on_val(model, anchors, stride, grid=None):
    """The threshold is chosen on VALIDATION, per arm, and never on the split
    the number is reported on."""
    grid = grid if grid is not None else [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
    dets = infer(model, Xva, anchors, stride)
    best, best_t = -1.0, grid[0]
    curve = []
    for t in grid:
        kept, _ = apply_nms(dets, t, n_img=N_VAL)
        m = mean_ap(match(kept, GT_VA, 0.5, N_VAL))
        curve.append({"t": t, "map50": rnd(m)})
        if m > best:
            best, best_t = m, t
    return best_t, curve


# ============================================================ E2 the window bill
print("\nE2 the window bill: how many windows each approach evaluates", flush=True)
SCAN_SHAPES = []
for s in SCAN_SIZES:
    for r in SCAN_ASPECTS:
        w = int(round(s * math.sqrt(r)))
        h = int(round(s / math.sqrt(r)))
        SCAN_SHAPES.append((w, h))


def positions(w, h, stride=1):
    if w > CANVAS or h > CANVAS:
        return 0
    return ((CANVAS - w) // stride + 1) * ((CANVAS - h) // stride + 1)


det_1 = TinyDet(3, stride=8)
det_16 = TinyDet(3, stride=16)
det_4 = TinyDet(3, stride=4)
win = WinClf()
MAC_DET8 = count_macs(det_1, (1, 1, CANVAS, CANVAS))
MAC_DET16 = count_macs(det_16, (1, 1, CANVAS, CANVAS))
MAC_DET4 = count_macs(det_4, (1, 1, CANVAS, CANVAS))
MAC_WIN = count_macs(win, (1, 1, WIN, WIN))
P_DET, P_WIN = n_params(det_1), n_params(win)
assert count_macs(win, (1, 1, WIN, WIN)) == MAC_WIN

w_one = positions(28, 28, 1)
w_one_s4 = positions(28, 28, WIN_STRIDE)
w_five = sum(positions(s, s, 1) for s in SCAN_SIZES)
w_five3 = sum(positions(w, h, 1) for (w, h) in SCAN_SHAPES)
w_five3_s4 = sum(positions(w, h, WIN_STRIDE) for (w, h) in SCAN_SHAPES)
w_det = (CANVAS // 8) ** 2 * 3
assert w_one == (CANVAS - 28 + 1) ** 2
assert w_five3 == sum((CANVAS - w + 1) * (CANVAS - h + 1) for (w, h) in SCAN_SHAPES)

# Sharing, decomposed, because a single headline factor bundles several causes
# and attributing it to any one of them is the error this article is about.
torch.manual_seed(SEED)
crop_macs = 64 * count_macs(TinyDet(3, stride=8), (1, 1, 28, 28))
full_macs = MAC_DET8
area_redundancy = 64 * 28 * 28 / (CANVAS * CANVAS)
sharing = crop_macs / full_macs

naive_macs = w_five3 * MAC_WIN
conv_scan_macs_ref = w_five3_s4 * MAC_WIN
total_ratio = naive_macs / MAC_DET8
# Three factors whose product IS the total, by construction and not by luck:
# stride, then how many window shapes, then what one shape's scan costs
# against one pass of the detector.
f_stride = w_five3 / w_five3_s4
f_shapes = w_five3_s4 / w_one_s4
f_net = (MAC_WIN * w_one_s4) / MAC_DET8
factor_product = f_stride * f_shapes * f_net
assert abs(factor_product - total_ratio) / total_ratio < 1e-9, \
    "the factors must reconstruct the ratio"

# Article 19's own counts, read from its json and cross-checked.
vj_path = ROOT / "viola-jones" / "data" / "cascade.json"
vj = json.loads(vj_path.read_text()) if vj_path.exists() else None
if vj is not None:
    assert vj["scan"]["n_windows"] == (vj["scan"]["size"] - vj["scan"]["window"] + 1) ** 2
    vj_block = {"n_windows": vj["scan"]["n_windows"], "size": vj["scan"]["size"],
                "window": vj["scan"]["window"],
                "scan_mean_evaluated": vj["scan"]["mean_evaluated"],
                "cascade_mean_evaluated": vj["cascade"]["mean_evaluated"],
                "cascade_full_cost": vj["cascade"]["full_cost"],
                "note": "mean_evaluated appears twice in article 19 with different "
                        "values: over a whole photograph's windows, and over its own "
                        "held-out set. Both are carried here."}
else:
    vj_block = None


def time_scan():
    """Two timings, because the naive scan's cost is dominated by resizing
    crops, not by the classifier, and every plan that skipped this was wrong
    about the budget."""
    Xs = Xte[:100]
    boxes, idx = [], []
    for i in range(100):
        for (w, h) in SCAN_SHAPES:
            for y in range(0, CANVAS - h + 1, WIN_STRIDE):
                for x in range(0, CANVAS - w + 1, WIN_STRIDE):
                    boxes.append([x, y, x + w, y + h])
                    idx.append(i)
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    idx_t = torch.tensor(idx, dtype=torch.long)
    torch.manual_seed(SEED)
    m = WinClf().eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(idx_t), 4096):
            crops = crop_resize(Xs, idx_t[s:s + 4096], boxes_t[s:s + 4096], WIN)
            m(crops)
    naive = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        for (w, h) in SCAN_SHAPES:
            Wp = int(round(CANVAS * WIN / w))
            Hp = int(round(CANVAS * WIN / h))
            m(F.interpolate(Xs, size=(Hp, Wp), mode="bilinear", align_corners=False))
    conv = time.perf_counter() - t0
    d = TinyDet(3, stride=8).eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        d(Xs)
    det = time.perf_counter() - t0
    return {"n_images": 100, "windows_per_image": len(idx) // 100,
            "naive_seconds": rnd(naive, 3), "conv_seconds": rnd(conv, 3),
            "detector_seconds": rnd(det, 3),
            "speedup": rnd(naive / max(conv, 1e-9), 2)}


def crop_resize(X, idx, boxes, out=WIN, chunk=4096):
    """Bilinear crop-and-resize, batched. align_corners=False here matches
    F.interpolate exactly, which is what lets E5 compare the crop path and the
    convolutional path without the resize itself becoming a confound."""
    H, W = X.shape[2], X.shape[3]
    res = []
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        b = boxes[s:s + chunk]
        R = len(ii)
        u = (torch.arange(out, dtype=torch.float32) + 0.5) / out
        gx = (b[:, 0:1] + u[None, :] * (b[:, 2:3] - b[:, 0:1])) * 2.0 / W - 1.0
        gy = (b[:, 1:2] + u[None, :] * (b[:, 3:4] - b[:, 1:2])) * 2.0 / H - 1.0
        grid = torch.stack([gx[:, None, :].expand(R, out, out),
                            gy[:, :, None].expand(R, out, out)], -1)
        res.append(F.grid_sample(X[ii], grid, mode="bilinear",
                                 align_corners=False, padding_mode="border"))
    return torch.cat(res) if res else torch.zeros(0, 1, out, out)


scan_time = cached(f"{DCONFIG}-scan-timing", time_scan)
print(f"  window shapes {len(SCAN_SHAPES)}: {w_five3:,} windows at stride 1, "
      f"{w_five3_s4:,} at stride {WIN_STRIDE}; the detector evaluates {w_det}", flush=True)
print(f"  naive crop scan {scan_time['naive_seconds']:.2f}s / 100 canvases against "
      f"{scan_time['conv_seconds']:.2f}s convolutional, a speedup of "
      f"{scan_time['speedup']:.1f}x", flush=True)
print(f"  64 crops of 28x28 cost {crop_macs / 1e6:.2f} MMAC against {full_macs / 1e6:.2f} "
      f"MMAC for one 64x64 pass: sharing {sharing:.2f}x, area redundancy "
      f"{area_redundancy:.2f}x. They are the same number up to border effects.", flush=True)


# The measurement Plan B's spine claim needs and did not have: the detector is
# padded, so a crop and the corresponding cells of the full map do NOT agree,
# and the border cells are where they do not.
def crop_vs_full():
    torch.manual_seed(SEED)
    m = TinyDet(3, stride=8).eval()
    with torch.no_grad():
        full = m.raw(Xte[:32])[:, :, 2:6, 2:6]
        crop = m.raw(Xte[:32, :, 16:48, 16:48])
    d = (full - crop).abs()
    inner = d[:, :, 1:3, 1:3].max().item()
    border = max(d[:, :, 0, :].max().item(), d[:, :, 3, :].max().item(),
                 d[:, :, :, 0].max().item(), d[:, :, :, 3].max().item())
    mag = full.abs().mean().item()
    torch.manual_seed(SEED + 1)
    w = WinClf().eval()
    with torch.no_grad():
        wf = w(Xte[:32])[:, :, 0:3, 0:3]
        wc = torch.stack([w(Xte[:32, :, 4 * i:4 * i + WIN, 4 * j:4 * j + WIN])[:, :, 0, 0]
                          for i in range(3) for j in range(3)], -1).view(32, N_CLS + 1, 3, 3)
    return {"detector_interior": rnd(inner, 9), "detector_border": rnd(border, 9),
            "detector_magnitude": rnd(mag, 6),
            "winclf_gap": rnd((wf - wc).abs().max().item(), 12),
            "note": "an unpadded network computes the same function on a crop and on a "
                    "cell of the full map; a padded one does not, and the disagreement "
                    "lives on the border cells"}


equiv = cached(f"{DCONFIG}-equivalence", crop_vs_full)
print(f"  window classifier, crop against convolutional: {equiv['winclf_gap']:.3e}. "
      f"Detector: {equiv['detector_interior']:.2e} interior, "
      f"{equiv['detector_border']:.2e} border, on outputs of magnitude "
      f"{equiv['detector_magnitude']:.3f}", flush=True)

windows_block = {
    "shapes": [[w, h] for (w, h) in SCAN_SHAPES],
    "sizes": SCAN_SIZES, "aspects": SCAN_ASPECTS,
    "exhaustive_one_size": w_one, "exhaustive_one_size_stride4": w_one_s4,
    "exhaustive_five_sizes": w_five, "exhaustive_five_by_three": w_five3,
    "exhaustive_five_by_three_stride4": w_five3_s4,
    "detector_candidates": w_det, "detector_cells": (CANVAS // 8) ** 2,
    "macs": {"detector_s8": MAC_DET8, "detector_s16": MAC_DET16, "detector_s4": MAC_DET4,
             "window_classifier": MAC_WIN, "naive_scan": naive_macs,
             "conv_scan_equivalent": conv_scan_macs_ref,
             "s4_over_s8": rnd(MAC_DET4 / MAC_DET8, 3)},
    "params": {"detector": P_DET, "window_classifier": P_WIN},
    "sharing": {"crop_macs": crop_macs, "full_macs": full_macs,
                "factor": rnd(sharing, 3), "area_redundancy": rnd(area_redundancy, 3),
                "residual": rnd(sharing / area_redundancy, 3),
                "note": "the sharing factor IS the overlap factor up to border effects"},
    "factors": [{"name": "stride 1 against stride 4", "value": rnd(f_stride, 3)},
                {"name": f"{len(SCAN_SHAPES)} window shapes against one",
                 "value": rnd(f_shapes, 3)},
                {"name": "one shape scanned against one pass of the detector",
                 "value": rnd(f_net, 3)}],
    "factor_product": rnd(factor_product, 3), "total_ratio": rnd(total_ratio, 3),
    "attribution_note": "the product of the three factors reconstructs the total by "
                        "construction, which is the reason it must not be attributed to "
                        "any one of them",
    "timing": scan_time, "equivalence": equiv, "viola_jones": vj_block,
}


# ============================== E-0 runtime governor, part two: the real thing
# Part one timed a network from a different article. This times THIS article's
# training step, on this article's data, with this article's loss, and then
# picks the largest arm list whose predicted cost fits the budget. Nothing is
# extrapolated across architectures, which is where a MAC-proportional model
# was measured to be a factor of 1.8 out.
BUDGET_SECONDS = 40 * 60
STEPS_PER_EPOCH = (N_TRAIN + BATCH - 1) // BATCH


def measure_step():
    tgt = targets_for(ANCHOR_SETS["slots3-kmeans"], 8)
    torch.manual_seed(SEED)
    m = TinyDet(3, stride=8)
    opt = torch.optim.SGD(m.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)
    an_t = anchor_tensor(ANCHOR_SETS["slots3-kmeans"])
    gx, gy = grid_offsets(8)
    t_obj = torch.tensor(tgt["t_obj"])
    t_box = torch.tensor(tgt["t_box"])
    t_cls = torch.tensor(tgt["t_cls"])
    warm, timed = 4, 10
    t0 = None
    for step in range(warm + timed):
        if step == warm:
            t0 = time.perf_counter()
        sl = slice((step * BATCH) % N_TRAIN, (step * BATCH) % N_TRAIN + BATCH)
        opt.zero_grad()
        loss, _ = det_loss(m.raw(Xtr[sl]), t_obj[sl], t_box[sl], t_cls[sl],
                           GTPAD_TR[sl], GTN_TR[sl], an_t, 8, gx, gy)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
    per_step = (time.perf_counter() - t0) / timed
    return {"seconds_per_step": rnd(per_step, 4),
            "seconds_per_epoch": rnd(per_step * STEPS_PER_EPOCH, 2),
            "steps_per_epoch": STEPS_PER_EPOCH, "warmup_steps": warm, "timed_steps": timed}


print("\nE-0 runtime governor, part two: timing this article's own training step",
      flush=True)
probe = cached("step-probe-tinydet-a3-s8-b64-t12", measure_step)
S_EPOCH = probe["seconds_per_epoch"]
# The grid arms cost what their multiply-accumulates say relative to stride 8.
# That ratio compares like with like, which is the case where the model holds.
COST_S16 = MAC_DET16 / MAC_DET8
COST_S4 = MAC_DET4 / MAC_DET8
NON_TRAINING = 480.0     # dataset, analytics, scans, suppression sweeps, export

# The rungs, in order. Note what the last one keeps: a seed replicate survives
# all the way down, and epochs are spent instead. Without a seed spread every
# comparison in the article has to be reported as unresolved, and fourteen
# epochs of an unresolvable experiment is worth less than twelve of a
# resolvable one. Only the bottom rung, for a machine slower still, gives it up.
TIERS = [
    ("FULL", 25, [], [("slots3-kmeans", (SEED + 1, SEED + 2)),
                      ("slots3-same", (SEED + 1, SEED + 2))]),
    ("STANDARD", 20, ["boxparam"], [("slots3-kmeans", (SEED + 1,))]),
    ("LEAN", 16, ["boxparam"], [("slots3-kmeans", (SEED + 1,))]),
    ("FRUGAL", 12, ["boxparam", "rcnn"], [("slots3-kmeans", (SEED + 1,))]),
    ("MINIMAL", 10, ["boxparam", "seeds", "rcnn"], []),
]


def predict(epochs, dropped, replicates):
    """Four stride-8 arms, two grid arms, the replicates, and the continuation
    control, in epochs of stride-8 cost."""
    units = 4.0 + COST_S16 + COST_S4
    units += sum(len(x[1]) for x in replicates)
    if "boxparam" not in dropped:
        units += 1.0
    train = units * epochs * S_EPOCH
    if "rcnn" not in dropped:
        train += 8 * S_EPOCH
    return train + NON_TRAINING


TIER, EPOCHS, DROPPED, REPLICATES = TIERS[-1]
PREDICTED = predict(EPOCHS, DROPPED, REPLICATES)
for name, ep, dropped, reps in TIERS:
    p = predict(ep, dropped, reps)
    fits = p <= BUDGET_SECONDS
    print(f"  {name:9s} {ep} epochs: predicted {p / 60:5.1f} min "
          f"{'fits' if fits else 'over budget'}", flush=True)
    if fits:
        TIER, EPOCHS, DROPPED, REPLICATES = name, ep, dropped, reps
        PREDICTED = p
        break
CONFIG = f"{DCONFIG}-e{EPOCHS}-b{BATCH}"
TIER_REASON = (f"a real training step measured {probe['seconds_per_step']:.3f}s, so one "
               f"epoch of the stride-8 arm costs {S_EPOCH:.1f}s; tier {TIER} is the "
               f"largest arm list predicted to fit {BUDGET_SECONDS / 60:.0f} minutes "
               f"(the brief's calibration model ran at {CONTENTION:.2f}x its stated 5.0s)")
print(f"  step {probe['seconds_per_step']:.3f}s, epoch {S_EPOCH:.1f}s -> tier {TIER}, "
      f"{EPOCHS} epochs, predicted {PREDICTED / 60:.1f} min", flush=True)
if DROPPED:
    print(f"  dropped from the arm ladder: {', '.join(DROPPED)}", flush=True)


# ========================================================= E3 the anchor factorial
print(f"\nE3 the anchor factorial: four arms at stride 8, {EPOCHS} epochs, "
      f"everything but the slots held fixed", flush=True)
ARM_CONTROL = ("trunk, dataset, seed, epochs, batch, optimiser, schedule, loss weights, "
               "box parameterisation, score floor and evaluation protocol are identical "
               "in every arm; the arm's id names the one thing that changed")
arms = []
arm_states = {}


def run_arm(arm_id, anchors, stride, epochs=None, seed=SEED, box_mode="smoothl1",
            init_from=None, lr=BASE_LR, lr_mode="schedule"):
    epochs = epochs if epochs is not None else EPOCHS
    tgt = targets_for(anchors, stride)
    A = len(anchors)

    def build():
        torch.manual_seed(seed + 7 * stride + A)
        return TinyDet(A, stride=stride)

    def go():
        init = arm_states.get(init_from) if init_from else None
        model, hist, secs, fallback = train_with_guard(
            build, tgt, anchors, stride, epochs, seed, box_mode, init, lr, lr_mode)
        thr, vcurve = pick_nms_on_val(model, anchors, stride)
        dets = infer(model, Xte, anchors, stride)
        kept, _ = apply_nms(dets, thr, n_img=N_TEST)
        ev, _t = evaluate(kept, GT_TE, N_TEST)
        rec, quart = recall_by_size(kept, GT_TE, N_TEST)
        stats = [int(v) for v in tgt["stats"]]
        row = {"id": arm_id, "slots": A, "stride": stride, "epochs": epochs,
               "seed": seed, "box_mode": box_mode, "init_from": init_from,
               "anchors": [[rnd(v, 2) for v in a] for a in np.asarray(anchors)],
               "params": n_params(model),
               "macs": count_macs(model, (1, 1, CANVAS, CANVAS)),
               "nms_threshold": thr, "nms_val_curve": vcurve,
               "history": hist, "seconds": rnd(secs, 1), "lr_fallback": fallback,
               "positives_per_slot": [int(v) for v in tgt["per_slot"]],
               "assignment": {"boxes": stats[0], "unassignable": stats[1],
                              "collisions": stats[2]},
               "recall_by_size": rec, "size_terciles": quart,
               "detections_kept": int(len(kept["img"]))}
        row.update(ev)
        return row, model.state_dict()

    row, state = cached_arm(f"{CONFIG}-arm-{arm_id}", go)
    model = TinyDet(len(anchors), stride=stride)
    model.load_state_dict(state)
    arm_states[arm_id] = state
    arms.append(row)
    ci = row.get("map50_ci", [0, 0])
    print(f"  {arm_id:22s} mAP@0.5 {row['map50']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  "
          f"@0.75 {row['map75']:.4f}  @[.5:.95] {row['map5095']:.4f}  "
          f"slots used {row['positives_per_slot']}  {row['seconds']:.0f}s", flush=True)
    return row, model


E3_ARMS = ["slots1", "slots3-same", "slots3-kmeans", "slots3-square"]
E3_MODELS = {}
for arm_id in E3_ARMS:
    r, m = run_arm(arm_id, ANCHOR_SETS[arm_id], 8)
    E3_MODELS[arm_id] = m
MAIN = "slots3-kmeans"
main_model = E3_MODELS[MAIN]
main_anchors = ANCHOR_SETS[MAIN]
main_row = next(a for a in arms if a["id"] == MAIN)

# A free post-hoc control: keep only slot 0 at test time, which separates
# "multiplicity helped training" from "multiplicity helped inference".
dets_main = infer(main_model, Xte, main_anchors, 8)
G8 = CANVAS // 8
slot0 = dets_main["slot"] < G8 * G8
d0 = {k: v[slot0] for k, v in dets_main.items()}
k0, _ = apply_nms(d0, main_row["nms_threshold"], n_img=N_TEST)
ev0, _ = evaluate(k0, GT_TE, N_TEST, boot=False)
print(f"  {MAIN} evaluated with slot 0 only: mAP@0.5 {ev0['map50']:.4f} against "
      f"{main_row['map50']:.4f} with all three", flush=True)
anchors_block = {"k_curve": kcurve, "sets": {k: [[rnd(x, 2) for x in a] for a in np.asarray(v)]
                                             for k, v in ANCHOR_SETS.items()},
                 "slot0_only": {"map50": ev0["map50"], "map75": ev0["map75"]},
                 "box_shapes": [[rnd(v, 1) for v in wh] for wh in WH_TR[:900]],
                 "box_classes": [int(c) for c in Btr[:900, 1]],
                 "comparisons": [
                     {"pair": ["slots1", "slots3-same"], "isolates": "slot multiplicity, "
                      "prior identical"},
                     {"pair": ["slots3-same", "slots3-kmeans"], "isolates": "prior "
                      "diversity, head parameters identical"},
                     {"pair": ["slots3-same", "slots3-square"], "isolates": "prior "
                      "quality, head parameters identical"}]}


# ============================================================ E4 grid resolution
print("\nE4 grid resolution: the same arm with the pooling moved, nothing else", flush=True)
for stride in (16, 4):
    run_arm(f"stride{stride}", ANCHOR_SETS[MAIN], stride)


# ===================================================== seed replicates and boxparam
seed_spread = []
if REPLICATES:
    print("\nreplication: the load-bearing arms at more than one seed", flush=True)
    for arm_id, extra in REPLICATES:
        vals = [next(a for a in arms if a["id"] == arm_id)["map50"]]
        for s in extra:
            r, _ = run_arm(f"{arm_id}-seed{s}", ANCHOR_SETS[arm_id], 8, seed=s)
            vals.append(r["map50"])
        seed_spread.append({"arm": arm_id, "map50": vals, "n_seeds": len(vals),
                            "spread": rnd(max(vals) - min(vals))})
        print(f"  {arm_id}: {vals} over {len(vals)} seeds, "
              f"spread {max(vals) - min(vals):.4f}", flush=True)
SPREAD = max([s["spread"] for s in seed_spread], default=None)

if "boxparam" not in DROPPED:
    print("\nbox parameterisation: the loss on the box, and nothing else", flush=True)
    run_arm("boxparam-giou", ANCHOR_SETS[MAIN], 8, box_mode="giou")


def verdict(a_id, b_id):
    a = next(x for x in arms if x["id"] == a_id)
    b = next(x for x in arms if x["id"] == b_id)
    d = a["map50"] - b["map50"]
    if SPREAD is None:
        return {"pair": [a_id, b_id], "delta": rnd(d), "resolved": None,
                "note": "no seed replicate was run at this tier, so the difference is "
                        "not resolved by this experiment"}
    n_seeds = max(s.get("n_seeds", 1) for s in seed_spread)
    return {"pair": [a_id, b_id], "delta": rnd(d), "seed_spread": SPREAD,
            "seeds": n_seeds, "resolved": bool(abs(d) > SPREAD),
            "note": ("larger than the seed spread" if abs(d) > SPREAD else
                     "smaller than the seed spread, so not resolved by this "
                     "experiment") + f" (spread measured over {n_seeds} seeds)"}


verdicts = [verdict("slots3-same", "slots1"),
            verdict("slots3-kmeans", "slots3-same"),
            verdict("slots3-square", "slots3-same"),
            verdict("stride4", "slots3-kmeans"),
            verdict("stride16", "slots3-kmeans")]
for v in verdicts:
    print(f"  {v['pair'][0]} minus {v['pair'][1]}: {v['delta']:+.4f}, {v['note']}", flush=True)


# ==================================== E5 sliding a classifier, article 19's way
print("\nE5 sliding a classifier, trained the way article 19 trained one", flush=True)
N_SLIDE_TEST = min(800, N_TEST)
N_MINE = min(1500, N_TRAIN)


def sample_window_crops(gts, n_img, rs, per_img_neg=4, jitter=2):
    """Positives at IoU >= 0.7, background at IoU <= 0.3, and the negatives are
    drawn at ALL FIFTEEN window shapes and resized by the same path the test
    windows use, because a scale mismatch between the two would make the
    baseline lose for a reason that has nothing to do with sliding windows."""
    idx, boxes, labels = [], [], []
    for i in range(n_img):
        g = gts[i]
        for r in g:
            c = int(r[0])
            x1, y1, x2, y2 = r[1:]
            w, h = x2 - x1, y2 - y1
            best = min(SCAN_SHAPES, key=lambda s: -shape_iou(np.array([w, h], np.float64),
                                                             np.array(s, np.float64)))
            for _ in range(jitter):
                dx = rs.uniform(-0.12, 0.12) * best[0]
                dy = rs.uniform(-0.12, 0.12) * best[1]
                cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
                b = [cx - best[0] / 2, cy - best[1] / 2, cx + best[0] / 2, cy + best[1] / 2]
                if iou_xyxy(b, (x1, y1, x2, y2)) >= 0.7:
                    idx.append(i)
                    boxes.append(b)
                    labels.append(c)
        for _ in range(per_img_neg):
            w, h = SCAN_SHAPES[rs.randint(len(SCAN_SHAPES))]
            if w > CANVAS or h > CANVAS:
                continue
            x = rs.randint(0, CANVAS - w + 1)
            y = rs.randint(0, CANVAS - h + 1)
            b = [x, y, x + w, y + h]
            if len(g) == 0 or box_iou_np(np.array([b], np.float64), g[:, 1:]).max() <= 0.3:
                idx.append(i)
                boxes.append(b)
                labels.append(N_CLS)
    return (torch.tensor(idx, dtype=torch.long),
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))


def fit_winclf(model, X, idx, boxes, labels, epochs=8, seed=SEED, lr=0.02):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed)
    lossf = nn.CrossEntropyLoss()
    hist, t0 = [], time.perf_counter()
    for ep in range(epochs):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(ep, epochs, lr)
        perm = torch.randperm(len(idx), generator=g)
        tot, nb = 0.0, 0
        model.train()
        for i in range(0, len(idx), 256):
            sel = perm[i:i + 256]
            crops = crop_resize(X, idx[sel], boxes[sel], WIN)
            opt.zero_grad()
            loss = lossf(model(crops).flatten(1), labels[sel])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist.append(rnd(tot / max(nb, 1), 5))
    return hist, time.perf_counter() - t0


@torch.no_grad()
def winclf_scan(model, X, decim=1):
    """The convolutional scan: the IMAGE is resized once per window shape, not
    the crop, which is the only version of this that fits in the budget."""
    model.eval()
    imgs, clss, scores, boxes = [], [], [], []
    for (w, h) in SCAN_SHAPES:
        Wp = int(round(CANVAS * WIN / w))
        Hp = int(round(CANVAS * WIN / h))
        if Wp < WIN or Hp < WIN:
            continue
        sx, sy = CANVAS / Wp, CANVAS / Hp
        for s in range(0, len(X), 200):
            xr = F.interpolate(X[s:s + 200], size=(Hp, Wp), mode="bilinear",
                               align_corners=False)
            o = model(xr)[:, :, ::decim, ::decim]
            p = torch.softmax(o, 1)
            sc = 1.0 - p[:, N_CLS]
            cl = p[:, :N_CLS].argmax(1)
            keep = sc >= SCORE_FLOOR
            bi, yy, xx = torch.nonzero(keep, as_tuple=True)
            if len(bi) == 0:
                continue
            x1 = xx.float() * WIN_STRIDE * decim * sx
            y1 = yy.float() * WIN_STRIDE * decim * sy
            imgs.append((bi + s).numpy())
            clss.append(cl[bi, yy, xx].numpy())
            scores.append(sc[bi, yy, xx].numpy())
            boxes.append(torch.stack([x1, y1, x1 + WIN * sx, y1 + WIN * sy], -1)
                         .clamp(0.0, float(CANVAS)).numpy())
    if not imgs:
        return {"img": np.zeros(0, np.int64), "cls": np.zeros(0, np.int64),
                "score": np.zeros(0), "box": np.zeros((0, 4)), "slot": np.zeros(0, np.int64)}
    n = sum(len(a) for a in imgs)
    out = {"img": np.concatenate(imgs).astype(np.int64),
           "cls": np.concatenate(clss).astype(np.int64),
           "score": np.concatenate(scores).astype(np.float64),
           "box": np.concatenate(boxes).astype(np.float64).reshape(-1, 4),
           "slot": np.arange(n, dtype=np.int64)}
    return topk_per_image(out, len(X), SLIDE_TOPK)


def run_sliding():
    rs = np.random.RandomState(SEED)
    idx, boxes, labels = sample_window_crops(GT_TR, min(3000, N_TRAIN), rs)
    m = WinClf()
    hist1, s1 = fit_winclf(m, Xtr, idx, boxes, labels)
    d = winclf_scan(m, Xte[:N_SLIDE_TEST])
    kept, _ = apply_nms(d, 0.3, n_img=N_SLIDE_TEST)
    ev1, _ = evaluate(kept, GT_TE[:N_SLIDE_TEST], N_SLIDE_TEST)
    ev1.update({"round": "round1", "seconds": rnd(s1, 1), "history": hist1,
                "n_train_crops": int(len(idx)),
                "n_positives": int((labels < N_CLS).sum()),
                "n_negatives": int((labels == N_CLS).sum())})
    rows = [ev1]
    # One round of hard negative mining, which is article 19's own method.
    t0 = time.perf_counter()
    mined = winclf_scan(m, Xtr[:N_MINE])
    m_order, m_starts = group_index(mined["img"], N_MINE)
    fp = []
    for i in range(N_MINE):
        sel = m_order[m_starts[i]:m_starts[i + 1]]
        if len(sel) == 0:
            continue
        g = GT_TR[i]
        o = box_iou_np(mined["box"][sel], g[:, 1:]).max(1) if len(g) else np.zeros(len(sel))
        bad = sel[o <= 0.3]
        if len(bad):
            bad = bad[np.argsort(-mined["score"][bad])[:4]]
            for b in bad:
                fp.append((i, mined["box"][b]))
    mine_secs = time.perf_counter() - t0
    if fp:
        idx2 = torch.cat([idx, torch.tensor([f[0] for f in fp], dtype=torch.long)])
        boxes2 = torch.cat([boxes, torch.tensor(np.array([f[1] for f in fp]),
                                                dtype=torch.float32)])
        labels2 = torch.cat([labels, torch.full((len(fp),), N_CLS, dtype=torch.long)])
    else:
        idx2, boxes2, labels2 = idx, boxes, labels
    m2 = WinClf()
    hist2, s2 = fit_winclf(m2, Xtr, idx2, boxes2, labels2)
    d2 = winclf_scan(m2, Xte[:N_SLIDE_TEST])
    kept2, _ = apply_nms(d2, 0.3, n_img=N_SLIDE_TEST)
    ev2, _ = evaluate(kept2, GT_TE[:N_SLIDE_TEST], N_SLIDE_TEST)
    ev2.update({"round": "round2-mined", "seconds": rnd(s2, 1), "history": hist2,
                "n_mined": len(fp), "mining_seconds": rnd(mine_secs, 1),
                "n_train_crops": int(len(idx2))})
    rows.append(ev2)
    # Evaluation stride, by decimating the output grid at constant features.
    strides = []
    for decim, lab in ((1, WIN_STRIDE), (2, WIN_STRIDE * 2), (4, WIN_STRIDE * 4)):
        dd = winclf_scan(m2, Xte[:N_SLIDE_TEST], decim=decim)
        kk, _ = apply_nms(dd, 0.3, n_img=N_SLIDE_TEST)
        e, _ = evaluate(kk, GT_TE[:N_SLIDE_TEST], N_SLIDE_TEST, boot=False)
        strides.append({"evaluation_stride": lab, "map50": e["map50"],
                        "windows": int(len(dd["img"]))})
    return {"rounds": rows, "evaluation_stride": strides,
            "n_test": N_SLIDE_TEST, "params": P_WIN, "macs_per_window": MAC_WIN}


sliding = cached(f"{CONFIG}-sliding", run_sliding)
print(f"  round 1 mAP@0.5 {sliding['rounds'][0]['map50']:.4f}; after mining "
      f"{sliding['rounds'][1]['n_mined']} hard negatives "
      f"{sliding['rounds'][1]['map50']:.4f}", flush=True)
# The paired comparison the article needs: the detector on the SAME canvases.
det_slide, _ = apply_nms({k: v[dets_main["img"] < N_SLIDE_TEST] for k, v in dets_main.items()},
                         main_row["nms_threshold"], n_img=N_SLIDE_TEST)
ev_slide, _ = evaluate(det_slide, GT_TE[:N_SLIDE_TEST], N_SLIDE_TEST, boot=False)
sliding["detector_same_canvases"] = {"map50": ev_slide["map50"], "map75": ev_slide["map75"],
                                     "n_test": N_SLIDE_TEST}
print(f"  the detector on the same {N_SLIDE_TEST} canvases: mAP@0.5 "
      f"{ev_slide['map50']:.4f}", flush=True)
for s in sliding["evaluation_stride"]:
    print(f"    evaluation stride {s['evaluation_stride']:2d}: mAP@0.5 {s['map50']:.4f} "
          f"from {s['windows']:,} scored windows", flush=True)


# ============================================================ E6 suppression
print("\nE6 non-maximum suppression: the detections are fixed, the number is not", flush=True)
NMS_GRID = [rnd(0.05 * i + 0.05, 2) for i in range(18)]


def restrict(dets, gts, n_img, subset_idx):
    """A subset of images, renumbered, so a crowding split is a real split and
    not a mask applied halfway through."""
    keepmask = np.isin(dets["img"], subset_idx)
    sub = {k: v[keepmask] for k, v in dets.items()}
    remap = -np.ones(n_img, np.int64)
    remap[subset_idx] = np.arange(len(subset_idx))
    sub["img"] = remap[sub["img"]]
    return sub, [gts[i] for i in subset_idx], len(subset_idx)


def nms_rows(sub, g, n, mode="hard", classwise=True):
    """One row per threshold, with the two failure modes as exact counts:
    a true object that the raw grid DID cover and suppression removed, and a
    duplicate that survived. Returns the matching tables too, so the bootstrap
    over the optimum does not have to redo the suppression."""
    raw_cov = coverage(sub, g, n, 0.5)
    rows, tables = [], []
    for t in NMS_GRID:
        kept, pairs = apply_nms(sub, t, classwise=classwise, mode=mode, n_img=n)
        tbl = match(kept, g, 0.5, n)
        tables.append(tbl)
        tp = sum(float(tbl[c]["tp"].sum()) for c in range(N_CLS))
        nd = sum(len(tbl[c]["tp"]) for c in range(N_CLS))
        npos = sum(int(tbl[c]["npos_img"].sum()) for c in range(N_CLS))
        dup = sum(int(((tbl[c]["tp"] == 0) & (tbl[c]["raw_iou"] >= 0.5)).sum())
                  for c in range(N_CLS))
        cov = coverage(kept, g, n, 0.5)
        sup = sum(int((raw_cov[i] & ~cov[i]).sum()) for i in range(n))
        rows.append({"t": t, "map50": rnd(mean_ap(tbl)), "kept": int(nd),
                     "precision": rnd(tp / max(nd, 1)), "recall": rnd(tp / max(npos, 1)),
                     "duplicates_surviving": dup, "objects_suppressed": sup,
                     "pair_tests": pairs})
    return rows, tables


def nms_curve(dets, gts, n_img, subset_idx=None, mode="hard", classwise=True):
    if subset_idx is not None:
        sub, g, n = restrict(dets, gts, n_img, subset_idx)
    else:
        sub, g, n = dets, gts, n_img
    return nms_rows(sub, g, n, mode, classwise)[0]


dets_val = infer(main_model, Xva, main_anchors, 8)
curve_all = nms_curve(dets_main, GT_TE, N_TEST)
best_all = max(curve_all, key=lambda r: r["map50"])
worst_all = min(curve_all, key=lambda r: r["map50"])
print(f"  best threshold on test would be {best_all['t']} at mAP {best_all['map50']:.4f}; "
      f"the worst on the same detections is {worst_all['t']} at {worst_all['map50']:.4f}",
      flush=True)

# Crowding: the threshold is chosen on the VALIDATION crowding subsets and
# reported on the corresponding TEST subsets, with a bootstrap interval on the
# optimum, because an argmax of a noisy curve is a selection effect and not a
# finding.
max_pair_va = np.zeros(N_VAL)
for i, g in enumerate(GT_VA):
    if len(g) > 1:
        m = box_iou_np(g[:, 1:], g[:, 1:])
        np.fill_diagonal(m, 0.0)
        max_pair_va[i] = m.max()
VA_ISO = np.where(max_pair_va < 0.05)[0]
VA_CROWD = np.where(max_pair_va > 0.20)[0]


def optimum_ci(dets, gts, n_img, subset_idx, n_boot=100):
    """Where the best threshold sits, with an interval on WHERE, because an
    argmax of a noisy curve over a few hundred images is a selection effect
    until somebody puts an interval on it."""
    rs = np.random.RandomState(SEED)
    sub, g, n = restrict(dets, gts, n_img, subset_idx)
    rows, tables = nms_rows(sub, g, n)
    ts = np.array([r["t"] for r in rows])
    picks = []
    for _ in range(n_boot):
        mult = rs.multinomial(n, np.full(n, 1.0 / n))
        vals = [mean_ap(tb, mult) for tb in tables]
        picks.append(ts[int(np.argmax(vals))])
    picks = np.sort(np.array(picks))
    return rows, [float(picks[int(0.025 * len(picks))]),
                  float(picks[int(0.975 * len(picks)) - 1])]


def best_t(rows):
    return max(rows, key=lambda r: r["map50"])["t"]


va_iso_rows = nms_curve(dets_val, GT_VA, N_VAL, VA_ISO)
va_crowd_rows = nms_curve(dets_val, GT_VA, N_VAL, VA_CROWD)
t_iso, t_crowd = best_t(va_iso_rows), best_t(va_crowd_rows)
te_iso_rows, iso_ci = optimum_ci(dets_main, GT_TE, N_TEST, ISO)
te_crowd_rows, crowd_ci = optimum_ci(dets_main, GT_TE, N_TEST, CROWD)
iso_at_val = next(r for r in te_iso_rows if r["t"] == t_iso)
crowd_at_val = next(r for r in te_crowd_rows if r["t"] == t_crowd)
print(f"  chosen on validation: {t_iso} for isolated scenes, {t_crowd} for crowded ones; "
      f"on test they score {iso_at_val['map50']:.4f} and {crowd_at_val['map50']:.4f}",
      flush=True)
print(f"  the test optimum itself has a 95% interval of {iso_ci} isolated and {crowd_ci} "
      f"crowded, which is how flat these curves are", flush=True)

variants = []
for mode in ("hard", "linear", "gaussian"):
    kept, pairs = apply_nms(dets_main, main_row["nms_threshold"], mode=mode, n_img=N_TEST)
    ev, _ = evaluate(kept, GT_TE, N_TEST, boot=False)
    variants.append({"mode": mode, "map50": ev["map50"], "map75": ev["map75"],
                     "kept": int(len(kept["img"])), "pair_tests": pairs})
kept_ag, pairs_ag = apply_nms(dets_main, main_row["nms_threshold"], classwise=False,
                              n_img=N_TEST)
ev_ag, _ = evaluate(kept_ag, GT_TE, N_TEST, boot=False)
variants.append({"mode": "class-agnostic", "map50": ev_ag["map50"], "map75": ev_ag["map75"],
                 "kept": int(len(kept_ag["img"])), "pair_tests": pairs_ag})
nms_block = {"grid": NMS_GRID, "all": curve_all,
             "isolated": {"val": va_iso_rows, "test": te_iso_rows, "chosen": t_iso,
                          "optimum_ci": iso_ci, "n": len(ISO)},
             "crowded": {"val": va_crowd_rows, "test": te_crowd_rows, "chosen": t_crowd,
                         "optimum_ci": crowd_ci, "n": len(CROWD)},
             "variants": variants,
             "cost_note": "pair_tests is the number of pairwise IoU tests suppression "
                          "performed over the whole split, which is its own cost"}


# ================================================ E7 what confidence should be
print("\nE7 what a confidence should be: objectness, class, or the product", flush=True)
score_rows = []
for rule in ("objectness", "class", "product"):
    d = {k: v.copy() for k, v in dets_main.items()}
    d["score"] = (dets_main["obj"] if rule == "objectness" else
                  dets_main["clsp"] if rule == "class" else
                  dets_main["obj"] * dets_main["clsp"])
    sel = d["score"] >= SCORE_FLOOR
    d = {k: v[sel] for k, v in d.items()}
    kept, _ = apply_nms(d, main_row["nms_threshold"], n_img=N_TEST)
    ev, _ = evaluate(kept, GT_TE, N_TEST, boot=False)
    score_rows.append({"rule": rule, "map50": ev["map50"], "map75": ev["map75"]})
    print(f"  {rule:11s} mAP@0.5 {ev['map50']:.4f}", flush=True)

kept_main, _ = apply_nms(dets_main, main_row["nms_threshold"], n_img=N_TEST)
tbl_main = match(kept_main, GT_TE, 0.5, N_TEST)
all_tp = np.concatenate([tbl_main[c]["tp"] for c in range(N_CLS)])
all_sc_sorted = np.concatenate([tbl_main[c]["score"] for c in range(N_CLS)])
bins = np.linspace(0, 1, 11)
rel = []
ece_w, ece_u, tot = 0.0, [], max(len(all_tp), 1)
for b in range(10):
    m = (all_sc_sorted >= bins[b]) & (all_sc_sorted < bins[b + 1] + (1e-9 if b == 9 else 0))
    n = int(m.sum())
    if n == 0:
        rel.append({"lo": rnd(bins[b], 2), "hi": rnd(bins[b + 1], 2), "n": 0,
                    "precision": None, "mean_score": None})
        continue
    p = float(all_tp[m].mean())
    s = float(all_sc_sorted[m].mean())
    rel.append({"lo": rnd(bins[b], 2), "hi": rnd(bins[b + 1], 2), "n": n,
                "precision": rnd(p), "mean_score": rnd(s)})
    ece_w += n / tot * abs(p - s)
    ece_u.append(abs(p - s))
reliability = {"bins": rel, "ece_weighted": rnd(ece_w), "n_detections": int(len(all_tp)),
               "ece_unweighted": rnd(float(np.mean(ece_u)) if ece_u else 0.0),
               "event": "the event is 'this surviving detection matches an unmatched "
                        "ground-truth box of its own class at IoU >= 0.5 under the "
                        "confidence-sorted greedy matching used for mAP'"}
print(f"  weighted calibration error {reliability['ece_weighted']:.4f} over "
      f"{reliability['n_detections']} surviving detections", flush=True)


# ================================================================ E8 two stages
print("\nE8 two stages: proposals from the shared feature map, not from pixels", flush=True)


class RoIHead(nn.Module):
    """3x3 RoIAlign bins on the SHARED 8x8x64 map, then a small MLP. Declared
    in advance: a 20 px box on a stride-8 map covers 2.5 cells, so this second
    stage has very little spatial detail to work with and the expected gain is
    small. That is a property of this toy, not of Faster R-CNN."""

    def __init__(self, cin=64, bins=3, n_classes=N_CLS):
        super().__init__()
        self.bins = bins
        self.fc = nn.Sequential(nn.Linear(cin * bins * bins, 64), nn.ReLU())
        self.cls = nn.Linear(64, n_classes + 1)
        self.box = nn.Linear(64, 4)

    def forward(self, pooled):
        z = self.fc(pooled.flatten(1))
        return self.cls(z), self.box(z)


def roi_align(feat, img_idx, rois, bins=3, stride=8):
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


@torch.no_grad()
def proposals_from(model, X, anchors, stride, topk=32):
    """The top-k slots by OBJECTNESS, plus what stage one thought they were.
    Every image contributes exactly k, in image order, so everything downstream
    can reshape to (n_images, k) instead of searching."""
    model.eval()
    an_t = anchor_tensor(anchors)
    G = CANVAS // stride
    gx, gy = grid_offsets(G)
    ob_, bx_, im_, cl_, cp_ = [], [], [], [], []
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
    return (torch.cat(bx_), torch.cat(im_), torch.cat(ob_),
            torch.cat(cl_), torch.cat(cp_))


TOPK = 32


def run_two_stage():
    n_tr = min(3000, N_TRAIN)
    with torch.no_grad():
        feat_tr = torch.cat([main_model.body(Xtr[i:i + 250]) for i in range(0, n_tr, 250)])
        feat_te = torch.cat([main_model.body(Xte[i:i + 250]) for i in range(0, N_TEST, 250)])
    pb, pi, po, _pc, _pcp = proposals_from(main_model, Xtr[:n_tr], main_anchors, 8, TOPK)
    pb_np = pb.numpy()
    lab = np.full(len(pb_np), N_CLS, np.int64)
    tgt = np.zeros((len(pb_np), 4), np.float32)
    for i in range(n_tr):
        g = GT_TR[i]
        if len(g) == 0:
            continue
        sel = np.arange(i * TOPK, (i + 1) * TOPK)
        o = box_iou_np(pb_np[sel], g[:, 1:])
        best = o.argmax(1)
        bo = o.max(1)
        for j, s in enumerate(sel):
            if bo[j] >= 0.5:
                r = g[best[j]]
                lab[s] = int(r[0])
                pw = max(pb_np[s, 2] - pb_np[s, 0], 1e-6)
                ph = max(pb_np[s, 3] - pb_np[s, 1], 1e-6)
                px = (pb_np[s, 0] + pb_np[s, 2]) / 2
                py = (pb_np[s, 1] + pb_np[s, 3]) / 2
                gw = max(r[3] - r[1], 1e-6)
                gh = max(r[4] - r[2], 1e-6)
                tgt[s] = [((r[1] + r[3]) / 2 - px) / pw, ((r[2] + r[4]) / 2 - py) / ph,
                          math.log(gw / pw), math.log(gh / ph)]
            elif bo[j] >= 0.4:
                lab[s] = -1                     # neither positive nor background
    rs = np.random.RandomState(SEED)
    pos = np.flatnonzero(lab >= 0)
    pos = pos[lab[pos] < N_CLS]
    neg = np.flatnonzero(lab == N_CLS)
    n_neg = min(len(neg), 3 * len(pos))
    neg = rs.choice(neg, n_neg, replace=False) if n_neg > 0 else neg
    sel = np.concatenate([pos, neg])
    # Two heads are trained, and they answer three questions: `full` refines
    # the box, `cls-only` never learns to, and `box-only` reuses `full`'s box
    # with stage ONE's class, which is what isolates refinement from rejection.
    heads = {}
    for variant in ("full", "cls-only"):
        torch.manual_seed(SEED)
        head = RoIHead()
        opt = torch.optim.SGD(head.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        gg = torch.Generator().manual_seed(SEED)
        sel_t = torch.tensor(sel, dtype=torch.long)
        lab_t = torch.tensor(lab, dtype=torch.long)
        tgt_t = torch.tensor(tgt)
        t0 = time.perf_counter()
        for ep in range(15):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(ep, 15, 0.05)
            perm = torch.randperm(len(sel_t), generator=gg)
            for i in range(0, len(sel_t), 256):
                s = sel_t[perm[i:i + 256]]
                pooled = roi_align(feat_tr, pi[s], pb[s])
                cl, bx = head(pooled)
                loss = F.cross_entropy(cl, lab_t[s])
                fg = lab_t[s] < N_CLS
                if variant != "cls-only" and int(fg.sum()) > 0:
                    loss = loss + 5.0 * F.smooth_l1_loss(bx[fg], tgt_t[s][fg])
                opt.zero_grad()
                loss.backward()
                opt.step()
        secs = time.perf_counter() - t0
        heads[variant] = (head, rnd(secs, 1))

    tb, ti, to, tc, tcp = proposals_from(main_model, Xte, main_anchors, 8, TOPK)
    rows = []
    for variant in ("full", "cls-only", "box-only"):
        head, secs = heads["full" if variant == "box-only" else variant]
        with torch.no_grad():
            cl, bx = [], []
            for i in range(0, len(ti), 2048):
                pooled = roi_align(feat_te, ti[i:i + 2048], tb[i:i + 2048])
                c, b = head(pooled)
                cl.append(c)
                bx.append(b)
            cl = torch.cat(cl)
            bx = torch.cat(bx)
        p = torch.softmax(cl, 1)
        sc, cid = p[:, :N_CLS].max(1)
        boxes = tb.clone()
        if variant != "cls-only":
            pw = (tb[:, 2] - tb[:, 0]).clamp_min(1e-6)
            ph = (tb[:, 3] - tb[:, 1]).clamp_min(1e-6)
            px = (tb[:, 0] + tb[:, 2]) / 2
            py = (tb[:, 1] + tb[:, 3]) / 2
            nx = px + bx[:, 0] * pw
            ny = py + bx[:, 1] * ph
            nw = pw * torch.exp(bx[:, 2].clamp(-2, 2))
            nh = ph * torch.exp(bx[:, 3].clamp(-2, 2))
            boxes = torch.stack([nx - nw / 2, ny - nh / 2,
                                 nx + nw / 2, ny + nh / 2], -1).clamp(0.0, float(CANVAS))
        if variant == "box-only":
            # The class and the score come from stage ONE, so the only thing
            # on trial in this row is the box refinement.
            d = {"img": ti.numpy(), "cls": tc.numpy(), "score": (tcp * to).numpy(),
                 "box": boxes.numpy(), "slot": np.arange(len(ti))}
        else:
            d = {"img": ti.numpy(), "cls": cid.numpy(), "score": (sc * to).numpy(),
                 "box": boxes.numpy(), "slot": np.arange(len(ti))}
        keep = d["score"] >= SCORE_FLOOR
        d = {k: v[keep] for k, v in d.items()}
        kept, _ = apply_nms(d, main_row["nms_threshold"], n_img=N_TEST)
        ev, _t50 = evaluate(kept, GT_TE, N_TEST, boot=False)
        rows.append({"variant": variant, "map50": ev["map50"], "map75": ev["map75"],
                     "map5095": ev["map5095"], "mean_matched_iou": ev["mean_matched_iou"],
                     "seconds": secs, "params": n_params(head),
                     "n_proposals_per_image": TOPK})
    # Proposal recall against k, on the full slot list ranked by objectness.
    ab, _ai, _ao, _ac, _acp = proposals_from(main_model, Xte, main_anchors, 8, topk=192)
    all_k = ab.shape[0] // N_TEST
    ab_np = ab.numpy().reshape(N_TEST, all_k, 4)
    prec = []
    for k in (1, 2, 4, 8, 16, 32, 64, 128, all_k):
        hit50, hit70, tot = 0, 0, 0
        for i in range(N_TEST):
            g = GT_TE[i]
            tot += len(g)
            if len(g) == 0:
                continue
            o = box_iou_np(g[:, 1:], ab_np[i, :k]).max(1)
            hit50 += int((o >= 0.5).sum())
            hit70 += int((o >= 0.7).sum())
        prec.append({"k": int(k), "recall50": rnd(hit50 / max(tot, 1)),
                     "recall70": rnd(hit70 / max(tot, 1))})
    return {"ablations": rows, "proposal_recall": prec, "topk": TOPK,
            "roialign": {"bins": 3, "on": "the shared 8x8x64 feature map",
                         "positive_iou": 0.5, "background_iou": 0.4,
                         "pos_neg_ratio": "1:3", "epochs": 15},
            "declared": "a 20 px box on a stride-8 map covers 2.5 cells, so the gain "
                        "from a second stage here is expected to be small"}


two_stage = cached(f"{CONFIG}-twostage", run_two_stage)
for r in two_stage["ablations"]:
    print(f"  {r['variant']:10s} mAP@0.5 {r['map50']:.4f}  @0.75 {r['map75']:.4f}  "
          f"mean matched IoU {r['mean_matched_iou']:.4f}", flush=True)
pr = two_stage["proposal_recall"]
print(f"  proposal recall at IoU 0.5: {pr[0]['recall50']:.3f} with 1 proposal, "
      f"{pr[5]['recall50']:.3f} with 32, {pr[-1]['recall50']:.3f} with all 192", flush=True)

# The control for "you simply gave it more gradient steps".
if "rcnn" not in DROPPED:
    # Eight more epochs of the SAME optimiser from the finished arm's weights,
    # at the rate its schedule ended on, which is the control for "you simply
    # gave the second stage more gradient steps".
    extra_row, _ = run_arm("one-stage-extra", main_anchors, 8, epochs=8,
                           init_from=MAIN, lr=BASE_LR * 0.01, lr_mode="const")
    two_stage["one_stage_extra"] = {"map50": extra_row["map50"],
                                    "map75": extra_row["map75"],
                                    "extra_epochs": 8}


# =================================================== E9 baselines and failures
print("\nE9 baselines and the failure taxonomy", flush=True)


def blob_baseline():
    """Connected components with a SCORE, because a single operating point
    cannot produce a ranked list and its area under a PR curve would be low
    for reasons that have nothing to do with detection."""
    img, box, sc = [], [], []
    for i in range(N_TEST):
        lab = sklabel(D_TE["canvas"][i] >= INK)
        for j in range(1, lab.max() + 1):
            ys, xs = np.nonzero(lab == j)
            if len(ys) < 12:
                continue
            img.append(i)
            box.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            sc.append(len(ys))
    sc = np.array(sc, np.float64)
    sc = sc / max(sc.max(), 1.0)
    d = {"img": np.array(img, np.int64), "cls": np.zeros(len(img), np.int64),
         "score": sc, "box": np.array(box, np.float64).reshape(-1, 4),
         "slot": np.arange(len(img), dtype=np.int64)}
    gts_ag = [np.concatenate([np.zeros((len(g), 1)), g[:, 1:]], 1) if len(g)
              else np.zeros((0, 5)) for g in GT_TE]
    tbl = match(d, gts_ag, 0.5, N_TEST)
    ag = {"blob_ap50": rnd(ap_from_table(tbl[0]))}
    dd = {k: v.copy() for k, v in dets_main.items()}
    dd["cls"] = np.zeros(len(dd["img"]), np.int64)
    kept, _ = apply_nms(dd, main_row["nms_threshold"], n_img=N_TEST)
    ag["detector_ap50_agnostic"] = rnd(ap_from_table(match(kept, gts_ag, 0.5, N_TEST)[0]))
    ag["n_blobs"] = len(img)
    return ag


def failure_taxonomy():
    tbl = match(kept_main, GT_TE, 0.5, N_TEST)
    cats = {"duplicate": 0, "poor localisation": 0, "wrong class": 0,
            "held-out distractor": 0, "background": 0}
    for c in range(N_CLS):
        tp = tbl[c]["tp"]
        for j, d in enumerate(tbl[c]["order"]):
            if tp[j] > 0:
                continue
            i = int(kept_main["img"][d])
            b = kept_main["box"][d:d + 1]
            g = GT_TE[i]
            same = g[g[:, 0] == c] if len(g) else np.zeros((0, 5))
            other = g[g[:, 0] != c] if len(g) else np.zeros((0, 5))
            dboxes = Dte[Dte[:, 0] == i][:, 1:]
            o_same = box_iou_np(b, same[:, 1:]).max() if len(same) else 0.0
            o_other = box_iou_np(b, other[:, 1:]).max() if len(other) else 0.0
            o_dist = box_iou_np(b, dboxes).max() if len(dboxes) else 0.0
            if o_same >= 0.5:
                cats["duplicate"] += 1
            elif o_other >= 0.5:
                cats["wrong class"] += 1
            elif o_same >= 0.1:
                cats["poor localisation"] += 1
            elif o_dist >= 0.3:
                cats["held-out distractor"] += 1
            else:
                cats["background"] += 1
    cov_kept = coverage(kept_main, GT_TE, N_TEST, 0.5)
    cov_raw = coverage(dets_main, GT_TE, N_TEST, 0.5)
    order, starts = group_index(dets_main["img"], N_TEST)
    missed, in_raw_same, in_raw_any = 0, 0, 0
    for i in range(N_TEST):
        raw = order[starts[i]:starts[i + 1]]
        for j, r in enumerate(GT_TE[i]):
            if cov_kept[i][j]:
                continue
            missed += 1
            in_raw_same += int(cov_raw[i][j])
            if len(raw) and box_iou_np(r[None, 1:], dets_main["box"][raw]).max() >= 0.5:
                in_raw_any += 1
    return {"false_positives": cats,
            "order": "duplicate, then wrong class, then poor localisation, then "
                     "held-out distractor, then background",
            "missed": missed,
            "missed_present_in_raw_same_class": in_raw_same,
            "missed_present_in_raw_any_class": in_raw_any,
            "note": "a miss that was present in the raw grid was thrown away by scoring "
                    "or suppression, not by the head"}


baselines = cached(f"{CONFIG}-baselines", blob_baseline)
failures = cached(f"{CONFIG}-failures", failure_taxonomy)
print(f"  blobs with an area score: class-agnostic AP@0.5 {baselines['blob_ap50']:.4f} "
      f"against the detector's {baselines['detector_ap50_agnostic']:.4f}", flush=True)
fp = failures["false_positives"]
print(f"  false positives: {fp['duplicate']} duplicates, {fp['poor localisation']} poorly "
      f"localised, {fp['wrong class']} wrong class, {fp['held-out distractor']} held-out "
      f"distractors, {fp['background']} background", flush=True)
print(f"  {failures['missed']} misses, of which "
      f"{failures['missed_present_in_raw_same_class']} were present in the raw grid before "
      f"scoring and suppression", flush=True)


# ================================================================= E10 export
print("\nE10 export: fold, quantise, then compute the reference", flush=True)


def fold_bn(model):
    """Batch normalisation folded into the preceding convolution, so the
    browser runs convolution, bias, ReLU and pooling and nothing else."""
    layers = []
    for conv, bn in zip(model.convs, model.bns):
        w = conv.weight.detach().double()
        g = bn.weight.detach().double()
        b = bn.bias.detach().double()
        m = bn.running_mean.detach().double()
        v = bn.running_var.detach().double()
        s = g / torch.sqrt(v + bn.eps)
        layers.append((w * s.view(-1, 1, 1, 1), b - m * s))
    layers.append((model.head.weight.detach().double(), model.head.bias.detach().double()))
    return layers


def folded_forward(layers, x, pool_after, double=True):
    x = x.double() if double else x.float()
    for i, (w, b) in enumerate(layers[:-1]):
        ww = w if double else w.float()
        bb = b if double else b.float()
        x = F.relu(F.conv2d(x, ww, bb, padding=1))
        if i in pool_after:
            x = F.max_pool2d(x, 2)
    w, b = layers[-1]
    return F.conv2d(x, w if double else w.float(), b if double else b.float())


main_model.eval()
LAYERS = fold_bn(main_model)
with torch.no_grad():
    ref_torch = main_model(Xte[:64])
    ref_fold32 = folded_forward(LAYERS, Xte[:64], main_model.pool_after, double=False)
    ref_fold64 = folded_forward(LAYERS, Xte[:64], main_model.pool_after, double=True)
fold_gap = float((ref_torch - ref_fold32).abs().max())
fold_rel = fold_gap / float(ref_torch.abs().mean())
dtype_gap = float((ref_fold64.float() - ref_fold32).abs().max())

flat = []
meta = []
off = 0
for k, (w, b) in enumerate(LAYERS):
    wn = w.numpy().astype(np.float32)
    bn_ = b.numpy().astype(np.float32)
    meta.append({"i": k, "shape": list(w.shape), "w_offset": off, "w_size": wn.size,
                 "b_offset": off + wn.size, "b_size": bn_.size})
    flat.append(wn.reshape(-1))
    flat.append(bn_.reshape(-1))
    off += wn.size + bn_.size
vals32 = np.concatenate(flat).astype(np.float32)
vals16 = vals32.astype(np.float16)
deq = vals16.astype(np.float64)
quant_gap = float(np.abs(deq - vals32.astype(np.float64)).max())

QLAYERS = []
for m in meta:
    w = torch.tensor(deq[m["w_offset"]:m["w_offset"] + m["w_size"]]).view(*m["shape"])
    b = torch.tensor(deq[m["b_offset"]:m["b_offset"] + m["b_size"]])
    QLAYERS.append((w, b))
with torch.no_grad():
    ref_quant = folded_forward(QLAYERS, Xte[:64], main_model.pool_after, double=True)
quant_out_gap = float((ref_quant - ref_fold64).abs().max())

b64 = base64.b64encode(vals16.tobytes()).decode("ascii")
print(f"  {len(vals32):,} folded values, float16 base64 is {len(b64) / 1024:.0f} kB "
      f"(float32 would be {len(vals32) * 4 * 4 / 3 / 1024:.0f} kB)", flush=True)
print(f"  fold gap {fold_gap:.3e} ({fold_rel:.2e} relative), dtype gap {dtype_gap:.3e}, "
      f"weight quantisation {quant_gap:.3e} on weights and {quant_out_gap:.3e} on outputs",
      flush=True)


# What the quantisation costs in the only unit that matters.
class QuantDet(nn.Module):
    def __init__(self, layers, pool_after, A, stride):
        super().__init__()
        self.layers = layers
        self.pool_after = pool_after
        self.n_anchors = A
        self.n_classes = N_CLS
        self.stride = stride

    def raw(self, x):
        o = folded_forward(self.layers, x, self.pool_after, double=False).float()
        B, _, G, _ = o.shape
        return o.view(B, self.n_anchors, 5 + N_CLS, G, G).permute(0, 1, 3, 4, 2)

    def eval(self):
        return self


qmodel = QuantDet([(w.float(), b.float()) for w, b in QLAYERS], main_model.pool_after,
                  len(main_anchors), 8)
dets_q = infer(qmodel, Xte, main_anchors, 8)
kept_q, _ = apply_nms(dets_q, main_row["nms_threshold"], n_img=N_TEST)
ev_q, _ = evaluate(kept_q, GT_TE, N_TEST, boot=False)
quant_map_delta = rnd(ev_q["map50"] - main_row["map50"], 5)
if abs(quant_map_delta) > 0.005:
    print(f"  WARNING: float16 weights move mAP@0.5 by {quant_map_delta:+.4f}", flush=True)
else:
    print(f"  float16 weights move mAP@0.5 by {quant_map_delta:+.5f}", flush=True)

# The reference chain, computed in float64 with the EXPORTED weights, so the
# browser reproduces the generator instead of approximating it.
LIVE_N = 12
live_idx = list(range(LIVE_N))
sprite = np.concatenate([D_TE["canvas"][i] for i in live_idx], axis=1)
Image.fromarray(sprite.astype(np.uint8), mode="L").save(IMG / "canvases.png")
grid_tr = np.concatenate([np.concatenate([D_TR["canvas"][r * 8 + c] for c in range(8)], 1)
                          for r in range(4)], 0)
Image.fromarray(grid_tr.astype(np.uint8), mode="L").save(IMG / "trainset.png")
# The cell map is exported as counts in grid.cell_map and drawn as a labelled
# heatmap, so there is no image of it: a greyscale 8x8 png carried the same
# numbers with less precision and nothing read it.

an_t = anchor_tensor(main_anchors)
gx8, gy8 = grid_offsets(G8)
with torch.no_grad():
    raw_live = folded_forward(QLAYERS, Xte[:LIVE_N].double(), main_model.pool_after,
                              double=True)
    raw_live = raw_live.view(LIVE_N, len(main_anchors), 5 + N_CLS, G8, G8).permute(0, 1, 3, 4, 2)
    box_live = decode(raw_live.float(), an_t, 8, gx8, gy8, clip=True)
    obj_live = torch.sigmoid(raw_live[..., 4].float())
    cp_live = torch.softmax(raw_live[..., 5:].float(), -1)
live_rows = []
for i in range(LIVE_N):
    sc = (obj_live[i] * cp_live[i].max(-1).values).reshape(-1).numpy()
    bx = box_live[i].reshape(-1, 4).numpy()
    cl = cp_live[i].max(-1).indices.reshape(-1).numpy()
    # Exactly the inference pipeline the browser is asked to reproduce: score
    # floor, then the same pre-suppression cap, then suppression. Only the
    # first 24 rows of the candidate list are shipped, which is a prefix of
    # what the browser will compute and so still checkable.
    order_live = np.argsort(-sc)
    order_live = order_live[sc[order_live] >= SCORE_FLOOR][:PRE_NMS_TOPK]
    d = {"img": np.zeros(len(order_live), np.int64), "cls": cl[order_live],
         "score": sc[order_live].astype(np.float64),
         "box": bx[order_live].astype(np.float64), "slot": order_live.astype(np.int64)}
    kept, _ = apply_nms(d, main_row["nms_threshold"], n_img=1)
    d = {k: v[:24] for k, v in d.items()}
    live_rows.append({
        "index": i,
        "gt": [[int(r[0])] + [rnd(v, 2) for v in r[1:]] for r in GT_TE[i]],
        "decoded": [{"slot": int(s), "cls": int(c), "score": rnd(v, 6),
                     "box": [rnd(x, 4) for x in b]}
                    for s, c, v, b in zip(d["slot"], d["cls"], d["score"], d["box"])],
        "nms": [{"slot": int(s), "cls": int(c), "score": rnd(v, 6),
                 "box": [rnd(x, 4) for x in b]}
                for s, c, v, b in zip(kept["slot"], kept["cls"], kept["score"], kept["box"])],
        "raw_checksum": rnd(float(raw_live[i].abs().sum()), 6),
    })
pixel_check = {"index": 0, "row": 32,
               "values": [int(v) for v in D_TE["canvas"][0][32]],
               "note": "the browser must read exactly these integers out of the sprite "
                       "before anything downstream can be believed"}

detector_json = {
    "format": "folded convolutions, float16 little-endian, base64",
    "dtype": "float16",
    "n_values": int(len(vals32)),
    "layers": meta,
    "weights_b64": b64,
    "arch": {"channels": TinyDet.CH, "kernel": 3, "pad": 1,
             "pool_after": list(main_model.pool_after), "head_kernel": 1,
             "anchors": [[rnd(v, 4) for v in a] for a in np.asarray(main_anchors)],
             "stride": 8, "grid": G8, "n_classes": N_CLS, "classes": DET_CLASSES,
             "input": "one channel, 64x64, pixel value divided by 255, no mean subtraction",
             "order": "for each layer: conv2d with padding 1 (the head 1x1) then bias, "
                      "then ReLU except on the head, with a 2x2 max pool of stride 2 "
                      "after each layer listed in pool_after",
             "channel_layout": "head channel a*(5+C)+t, t = 0..3 box, 4 objectness, "
                               "5..9 class logits",
             "decode": "x=(col+sigmoid(t0))*stride, y=(row+sigmoid(t1))*stride, "
                       "w=anchor_w*exp(clamp(t2,-4,4)), h=anchor_h*exp(clamp(t3,-4,4)), "
                       "corners x-w/2, y-h/2, x+w/2, y+h/2, then every coordinate "
                       "clamped to [0, 64]",
             "score": "sigmoid(objectness) * max(softmax(class logits))",
             "score_floor": SCORE_FLOOR, "pre_nms_topk": PRE_NMS_TOPK,
             "nms_threshold": main_row["nms_threshold"]},
    "errors": {"fold_abs": rnd(fold_gap, 12), "fold_relative": rnd(fold_rel, 12),
               "dtype_float64_vs_float32": rnd(dtype_gap, 12),
               "weight_quantisation_abs": rnd(quant_gap, 12),
               "output_quantisation_abs": rnd(quant_out_gap, 12),
               "map50_delta_from_quantisation": quant_map_delta,
               "note": "the reference boxes below were computed in float64 with the "
                       "quantised weights, so quantisation cannot appear in the browser's "
                       "printed agreement; what remains is dtype alone"},
    "live": {"sprite": "canvases.png", "tile": CANVAS, "count": LIVE_N,
             "pixel_check": pixel_check, "rows": live_rows},
}
(OUT / "detector.json").write_text(json.dumps(detector_json, separators=(",", ":")),
                                   encoding="utf-8")

# ============================================ ranked detections for the widget
WID_N = 150
sel = dets_main["img"] < WID_N
wd = {k: v[sel] for k, v in dets_main.items()}
rows_w = []
for i in range(WID_N):
    s = np.flatnonzero(wd["img"] == i)
    s = s[np.argsort(-wd["score"][s])[:12]]
    for j in s:
        rows_w.append([i, int(wd["cls"][j]), int(round(wd["score"][j] * 1000)),
                       int(round(wd["box"][j, 0] * 4)), int(round(wd["box"][j, 1] * 4)),
                       int(round(wd["box"][j, 2] * 4)), int(round(wd["box"][j, 3] * 4))])
gt_w = [[i, int(r[0])] + [int(round(v * 4)) for v in r[1:]]
        for i in range(WID_N) for r in GT_TE[i]]

runtime_block = {
    "calibration_seconds": calib["seconds"], "calibration_model": "resnet-20 width 16, "
    "6000 28x28 images, batch 128, 12 threads",
    "brief_calibration_seconds": 5.0, "contention": rnd(CONTENTION, 3),
    "tier": TIER, "epochs": EPOCHS, "test_canvases": N_TEST,
    "arms_dropped": DROPPED, "reason": TIER_REASON,
    "lr_fallbacks": [a["id"] for a in arms if a.get("lr_fallback")],
    "total_seconds": rnd(elapsed(), 1),
    "step_probe": probe, "seconds_per_epoch": S_EPOCH,
    "predicted_seconds": rnd(PREDICTED, 1), "budget_seconds": BUDGET_SECONDS,
    "replicates": [[a, list(s)] for a, s in REPLICATES],
    "assumptions": ["the cost of one training step is MEASURED on this article's own "
                    "network, data and loss, not extrapolated from the calibration "
                    "model, which was a factor of 1.8 out",
                    "the stride-4 and stride-16 arms are costed by their multiply "
                    "accumulates relative to stride 8, which compares like with like",
                    "target assignment is identical every epoch, so it is computed once "
                    "per anchor set and cached",
                    "torch.set_num_threads(12) as the brief specifies"],
    "ladder": ["box parameterisation arms", "seed replicates", "the extra-epochs "
               "control", "epochs 25 to 20 to 16 to 14"],
}

out = {
    "meta": {"article": "detection", "chip": "YOLO, SSD and Faster R-CNN",
             "seed": SEED, "canvas": CANVAS, "classes": DET_CLASSES,
             "box_convention": "half-open [x1, y1, x2, y2) in canvas pixels, "
                               "tight around ink at or above 32, amodal"},
    "runtime": runtime_block,
    "dataset": dataset_block,
    "iou_reference": iou_ref,
    "iou_geometry": iou_geom,
    "grid": {"rows": grid_rows, "cell_map": cellmap8.tolist(),
             "border_share": rnd(border_share),
             "ceiling_note": "with w = prior * exp(clamp(t, -4, 4)) every size is "
                             "reachable and an object's centre always lies in its own "
                             "cell, so the only ceiling is slot capacity; the prior-only "
                             "column is the weaker quantity usually quoted"},
    "anchors": anchors_block,
    "arms": arms,
    "arm_control": ARM_CONTROL,
    "verdicts": verdicts,
    "seed_spread": seed_spread,
    "sliding": sliding,
    "nms": nms_block,
    "score_rule": score_rows,
    "reliability": reliability,
    "two_stage": two_stage,
    "baselines": baselines,
    "failures": failures,
    "windows": windows_block,
    "eval": {"protocol": "detections pooled across the split, sorted by confidence "
                         "descending within class with ties broken by image then slot, "
                         "matched to the highest-IoU unmatched ground-truth box of the "
                         "same class in the same image, each ground-truth box consumed "
                         "at most once",
             "iou_sweep": IOU_SWEEP, "score_floor": SCORE_FLOOR,
             "pre_nms_topk": PRE_NMS_TOPK, "sliding_topk": SLIDE_TOPK,
             "ap_conventions": ["all-point envelope", "VOC2007 11-point", "COCO 101-point"],
             "bootstrap": {"resamples": N_BOOT, "unit": "test image",
                           "why": "matching is within-image, so resampling whole images "
                                  "preserves the true-positive labels exactly"},
             "widget": {"n_images": WID_N, "detections": rows_w, "gt": gt_w,
                        "box_scale": 4, "score_scale": 1000,
                        "columns": ["img", "cls", "score", "x1", "y1", "x2", "y2"]}},
    "loss": {"weights": LOSS_W, "box": "smooth L1 on the sigmoid offsets and on the log "
                                       "size ratios, each term normalised by its own count",
             "ignore_rule": "a non-assigned slot whose decoded box reaches IoU > 0.5 with "
                            "any object contributes no objectness gradient",
             "optimiser": "SGD momentum 0.9, lr 0.02, weight decay 5e-4, 2 warmup epochs, "
                          "x0.1 at 60% and x0.01 at 85%",
             "guard": "gradient norm clipped at 10, loss asserted finite every step, one "
                      "restart at half the learning rate on a non-finite loss"},
    "versions": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "tier": TIER,
                 "torch": str(torch.__version__)},
}
path = OUT / "detect.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
print(f"wrote {OUT / 'detector.json'}  "
      f"({(OUT / 'detector.json').stat().st_size / 1024:.1f} kB)")
print(f"total wall clock {elapsed() / 60:.1f} min at tier {TIER}")
