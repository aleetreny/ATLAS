"""Data for the ATLAS 'Style' article: what a mapping network and per-layer
styles are worth, measured on a dataset whose factors are known exactly.

The previous article left a generator with a latent space that nobody had
asked anything of: a gaussian that got mapped onto digits, smooth but
meaningless. StyleGAN's claim is that where you inject the latent, and what
distribution you allow it to have, decides whether that space has any
structure. Every part of that claim is checkable, but only if the true factors
of the data are known, which they never are on a face dataset. So the data
here is drawn by this file:

    a 32x32 RGB sprite: one of four shapes, at a size, at a position, in a
    colour, on a grey background, drawn deterministically from those.

Five factors, all written down, all recoverable from an image by arithmetic
rather than opinion (area, centroid and mean hue of the foreground; the shape
needs a small classifier, whose held-out accuracy is reported before it is
used). And one deliberate hole in the joint distribution: large sprites are
never blue. That hole is the whole argument. A generator that maps a gaussian
straight into pixels has to bend the space to leave that region empty, and the
bend is measurable; a mapping network is allowed to do the bending itself and
hand the synthesis network something better shaped.

Blocks:

0. The dataset and the hole, plus the measurement functions and their error on
   real sprites, which is the floor under every number that uses them.
1. The shape reader: a small classifier, so that "which shape is this" is a
   measurement of the same kind as the others.
2. Two generators on the same data with the same budget: one where the latent
   goes in at the input, one where it goes in at every layer through adaptive
   instance normalisation after passing through a mapping network. Two seeds
   each, because the previous article measured how much the seed is worth.
3. StyleGAN's own two metrics: perceptual path length and linear separability,
   in z and in w. Plus a third that only exists because this dataset is
   synthetic: the path length measured in TRUE factors rather than in some
   network's features.
4. Style mixing, layer by layer, scored by which factor actually moved.
5. The noise inputs: what they change, and what they leave alone.
6. Truncation: the quality against diversity trade, swept.
7. The grain: why the sprites carry none, which is the one decision in the
   dataset that was made by measurement after the first version failed.

Run from anywhere: `python src/utils/generate_stylegan_data.py`. Stages cache
into ~/.atlas_vision_data/stylegan_cache/.
"""
import base64
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "stylegan" / "data"
IMG = ROOT / "stylegan" / "img"
CACHE = Path.home() / ".atlas_vision_data" / "stylegan_cache"
for d in (OUT, IMG, CACHE):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

SMALL = os.environ.get("ATLAS_STYLE_SMALL") == "1"
THREADS = min(4, os.cpu_count() or 4)
rnd = lambda v, n=4: round(float(v), n)


def cached(key, fn):
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = CACHE / f"{h}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    val = fn()
    with open(path, "wb") as fh:
        pickle.dump(val, fh)
    print(f"    [{key}] computed in {time.time() - t0:.1f}s", flush=True)
    return val


# ----------------------------------------------------------------------------
# 0. The sprites
# ----------------------------------------------------------------------------
RES = 32
SHAPES = ["disc", "square", "triangle", "ring"]
SIZE_LO, SIZE_HI = 4.0, 11.0
POS_LO, POS_HI = 11.0, 21.0
BIG = 8.0                      # above this counts as large
BLUE_LO, BLUE_HI = 0.50, 0.75  # the hue band that large sprites never wear
# No per-pixel grain in the data, and that is a measurement rather than a
# preference: block 7 below trains this exact architecture on three copies of
# these sprites that differ in nothing but the grain, and publishes the result.
# At the budget this file uses, the grainless arm ends 368.88 from the data
# against 847.81 at a grain of 0.015 and 575.93 at 0.04, on a scale where real
# against real is 6.44. The mechanism is a shortcut: one number, how much of an
# image a three pixel blur removes, separates a grainy sprite from the same
# sprite drawn clean 95.5% of the time at 0.015 and 100% at 0.04, so the
# discriminator can win without ever looking at a shape. The two grainy arms
# then fail in different ways, one collapsing onto a single shape and one
# losing the colour, which is in the article.
#
# Deterministic sprites also make the noise-input study below ask a better
# question: what do the per-pixel noise inputs learn to do when the data has no
# stochastic detail in it at all.
GRAIN = 0.0


def hsv_to_rgb(h, s, v):
    """Vectorised over the leading axis, so a whole batch of colours at once."""
    h = np.asarray(h) % 1.0
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    out = np.zeros(h.shape + (3,))
    for k, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                   (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m, 0] = np.broadcast_to(r, h.shape)[m]
        out[m, 1] = np.broadcast_to(g, h.shape)[m]
        out[m, 2] = np.broadcast_to(b, h.shape)[m]
    return out


def rgb_to_hue(rgb):
    """Hue in [0, 1) from an (n, 3) array. Only meaningful where the colour is
    saturated, which is why the sprites are drawn saturated."""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = np.maximum(mx - mn, 1e-9)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, (g - b) / d % 6,
                 np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return (h / 6) % 1.0


def sample_factors(n, rng, hole=True):
    """Five factors: shape, size, and where it is in each direction, and its
    hue. The hole: a large sprite is never in the blue band, which
    the rejection loop below enforces by resampling the hue rather than by
    reshaping the marginal, so both marginals stay uniform and only the JOINT
    has the gap in it."""
    shape = rng.integers(0, len(SHAPES), n)
    size = rng.uniform(SIZE_LO, SIZE_HI, n)
    cx = rng.uniform(POS_LO, POS_HI, n)
    cy = rng.uniform(POS_LO, POS_HI, n)
    hue = rng.uniform(0, 1, n)
    if hole:
        for _ in range(64):
            bad = (size > BIG) & (hue >= BLUE_LO) & (hue < BLUE_HI)
            if not bad.any():
                break
            hue[bad] = rng.uniform(0, 1, int(bad.sum()))
    return dict(shape=shape, size=size, cx=cx, cy=cy, hue=hue)


def _sdfs(dx, dy, r):
    """The four shapes as signed distances, each scaled so that its area is
    exactly pi r squared. Without that scaling "size" means a different thing
    for a square than for a triangle, and every later number that compares a
    measured area against the size that was drawn measures the shape instead.
    The scales are solved for numerically in `_shape_scales`, and asserted."""
    s = SHAPE_SCALE
    k = math.sqrt(3.0)
    disc = np.sqrt(dx ** 2 + dy ** 2) - r * s[0]
    square = np.maximum(np.abs(dx), np.abs(dy)) - r * s[1]
    rt = r * s[2]
    px = np.abs(dx) - rt
    py = -dy + rt / k
    inside = px + k * py > 0
    tx = np.where(inside, (px - k * py) / 2, px)
    ty = np.where(inside, (-k * px - py) / 2, py)
    tx = tx - np.clip(tx, -2 * rt, 0)
    tri = -np.sqrt(tx ** 2 + ty ** 2) * np.sign(ty)
    rr = r * s[3]
    ring = np.abs(np.sqrt(dx ** 2 + dy ** 2) - rr * 0.75) - rr * 0.25
    return [disc, square, tri, ring]


def _shape_scales():
    """Bisect on each shape's scale until the drawn area matches a disc's."""
    r0 = 9.0
    n = 400
    ys, xs = np.mgrid[0:n, 0:n]
    dx = (xs - n / 2) * (RES / n)
    dy = (ys - n / 2) * (RES / n)
    cell = (RES / n) ** 2
    target = math.pi * r0 ** 2
    out = [1.0, 1.0, 1.0, 1.0]
    for i in range(4):
        lo, hi = 0.2, 4.0
        for _ in range(50):
            mid = (lo + hi) / 2
            global SHAPE_SCALE
            probe = list(out)
            probe[i] = mid
            keep = SHAPE_SCALE
            SHAPE_SCALE = probe
            area = float((_sdfs(dx, dy, r0)[i] < 0).sum()) * cell
            SHAPE_SCALE = keep
            if area < target:
                lo = mid
            else:
                hi = mid
        out[i] = (lo + hi) / 2
    return out


SHAPE_SCALE = [1.0, 1.0, 1.0, 1.0]
SHAPE_SCALE = _shape_scales()


def draw(factors, rng=None, grain=GRAIN):
    """Signed distance for each shape, softened over one pixel, so size and
    position are continuous rather than quantised: a generator that gets the
    size slightly wrong produces a slightly wrong picture instead of the same
    picture."""
    n = len(factors["size"])
    ys, xs = np.mgrid[0:RES, 0:RES]
    xs = xs[None] + 0.5
    ys = ys[None] + 0.5
    cx = factors["cx"][:, None, None]
    cy = factors["cy"][:, None, None]
    r = factors["size"][:, None, None]
    dx = xs - cx
    dy = ys - cy
    sdf = np.zeros((n, RES, RES))
    sh = factors["shape"]
    for i, s in enumerate(_sdfs(dx, dy, r)):
        sdf[sh == i] = s[sh == i]
    alpha = np.clip(0.5 - sdf, 0, 1)[..., None]
    fg = hsv_to_rgb(factors["hue"], 0.85, 0.95)[:, None, None, :]
    bg = np.full((n, 1, 1, 3), 0.86)
    img = alpha * fg + (1 - alpha) * bg
    if grain and rng is not None:
        img = img + rng.normal(0, grain, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def measure(img, thresh=0.30):
    """Read the factors back out of an image with arithmetic only.

    The background is the median of the border ring, the foreground is
    everything far enough from it, and size, position and hue are the area,
    centroid and mean hue of that. Reported against the truth on real sprites
    below, so every later use of these carries its own error bar.
    """
    n = img.shape[0]
    border = np.concatenate([img[:, 0, :, :], img[:, -1, :, :],
                             img[:, :, 0, :], img[:, :, -1, :]], axis=1)
    bg = np.median(border, axis=1)[:, None, None, :]
    d = np.abs(img - bg).sum(axis=-1)
    mask = d > thresh
    area = mask.sum(axis=(1, 2)).astype(float)
    ys, xs = np.mgrid[0:RES, 0:RES]
    denom = np.maximum(area, 1)
    cx = (mask * (xs[None] + 0.5)).sum(axis=(1, 2)) / denom
    cy = (mask * (ys[None] + 0.5)).sum(axis=(1, 2)) / denom
    hues = np.zeros(n)
    sat = np.zeros(n)
    for i in range(n):
        px = img[i][mask[i]]
        if len(px) < 4:
            hues[i] = np.nan
            continue
        mean_rgb = px.mean(axis=0)
        hues[i] = rgb_to_hue(mean_rgb)
        sat[i] = (mean_rgb.max() - mean_rgb.min()) / max(mean_rgb.max(), 1e-9)
    return dict(area=area, size=np.sqrt(np.maximum(area, 0) / math.pi),
                cx=cx, cy=cy, hue=hues, sat=sat, coverage=area / (RES * RES))


def hue_dist(a, b):
    """Hue lives on a circle, so the difference between 0.02 and 0.98 is 0.04
    and not 0.96. Getting this wrong makes a colour metric report enormous
    errors on the colours it got exactly right."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1 - d)


def stage_dataset():
    rng = np.random.default_rng(19)
    n = 2000 if SMALL else 24000
    f = sample_factors(n, rng)
    img = draw(f, rng)
    # the measurement floor: how well arithmetic recovers what was drawn
    m = measure(img)
    floor = dict(
        size_r=rnd(float(np.corrcoef(m["size"], f["size"])[0, 1]), 5),
        size_err=rnd(float(np.mean(np.abs(m["size"] - f["size"]))), 4),
        size_bias=rnd(float(np.mean(m["size"] - f["size"])), 4),
        size_by_shape=[rnd(float(np.mean(m["size"][f["shape"] == i] - f["size"][f["shape"] == i])), 4)
                       for i in range(len(SHAPES))],
        cx_err=rnd(float(np.mean(np.abs(m["cx"] - f["cx"]))), 4),
        cy_err=rnd(float(np.mean(np.abs(m["cy"] - f["cy"]))), 4),
        hue_err=rnd(float(np.nanmean(hue_dist(m["hue"], f["hue"]))), 5),
        hue_p99=rnd(float(np.nanpercentile(hue_dist(m["hue"], f["hue"]), 99)), 5),
        shape_scale=[rnd(s, 5) for s in SHAPE_SCALE],
        thresh=0.30,
    )
    assert floor["hue_err"] < 0.02, f"the hue measurement is not usable: {floor}"
    assert floor["cx_err"] < 0.35, f"the position measurement is not usable: {floor}"
    assert floor["size_err"] < 0.6, f"the size measurement is not usable: {floor}"
    assert max(abs(v) for v in floor["size_by_shape"]) < 0.7, \
        f"the four shapes are not the same size at the same size: {floor}"
    # and the hole, confirmed in the drawn data rather than in the sampler
    big = f["size"] > BIG
    blue = (f["hue"] >= BLUE_LO) & (f["hue"] < BLUE_HI)
    assert not (big & blue).any(), "the hole is not empty"
    hole = dict(big_share=rnd(float(big.mean()), 4), blue_share=rnd(float(blue.mean()), 4),
                expected_joint=rnd(float(big.mean() * blue.mean()), 4), observed=0.0,
                big_lo=BIG, blue_lo=BLUE_LO, blue_hi=BLUE_HI)
    return dict(n=n, floor=floor, hole=hole, factors=f, images=img)


# ----------------------------------------------------------------------------
# 1. The shape reader
# ----------------------------------------------------------------------------
def build_reader():
    """The reader's shape in one place, so the cache can rebuild it."""
    import torch
    import torch.nn as nn

    torch.manual_seed(5)
    # Flatten rather than average: a global average over the whole image
    # throws away where things are, and "which of these four outlines is it"
    # is exactly a question about where things are. With the average pool the
    # same classifier sat at chance.
    body = nn.Sequential(
        nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
        nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(True))
    head = nn.Linear(64, len(SHAPES))
    return nn.Sequential(body, head), body, head


def stage_reader(data):
    import torch

    key = f"reader-{data['n']}-{RES}-v2"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = CACHE / f"{h}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            saved = pickle.load(fh)
        net, body, head = build_reader()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in saved["state"].items()})
        net.eval()
        print(f"    the shape reader reaches {saved['acc']*100:.2f}% on held-out sprites",
              flush=True)
        return net, body, head, saved["acc"]
    net, body, head, acc = train_reader(data)
    with open(path, "wb") as fh:
        pickle.dump(dict(acc=acc, state={k: v.detach().numpy()
                                         for k, v in net.state_dict().items()}), fh)
    return net, body, head, acc


def train_reader(data):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    torch.manual_seed(5)
    img = data["images"]
    y = data["factors"]["shape"]
    n_tr = int(len(img) * 0.85)
    x = torch.from_numpy(img.transpose(0, 3, 1, 2).copy())
    yt = torch.from_numpy(y).long()
    net, body, head = build_reader()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(1 if SMALL else 6):
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 128):
            idx = perm[i:i + 128]
            loss = lossf(net(x[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(x[n_tr:]).argmax(1) == yt[n_tr:]).float().mean().item()
    print(f"    the shape reader reaches {acc*100:.2f}% on held-out sprites", flush=True)
    assert SMALL or acc > 0.9, ("the shape reader is not good enough to be used as a "
                                f"measurement: {acc:.4f}")
    return net, body, head, acc


def read_shapes(net, imgs):
    import torch
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
        return net(t).argmax(1).numpy()


def read_features(body, imgs):
    import torch
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
        return body(t).numpy()


# ----------------------------------------------------------------------------
# 2. Two generators
# ----------------------------------------------------------------------------
Z_DIM = 12
W_DIM = 12
CH = [48, 48, 32, 24, 16]     # 4x4, 8x8 (a), 8x8 (b), 16x16, 32x32


def build(kind, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class Mapping(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(Z_DIM, 32), nn.LeakyReLU(0.2, True),
                nn.Linear(32, 32), nn.LeakyReLU(0.2, True),
                nn.Linear(32, 32), nn.LeakyReLU(0.2, True),
                nn.Linear(32, W_DIM))

        def forward(self, z):
            return self.net(z)

    class Block(nn.Module):
        """One synthesis block: upsample, convolve, add per-pixel noise scaled
        by a learned amount, then normalise each channel of each sample and
        put back a mean and a scale that come from the style."""
        def __init__(self, cin, cout, up, styled):
            super().__init__()
            self.up = up
            self.conv = nn.Conv2d(cin, cout, 3, 1, 1)
            self.styled = styled
            self.noise = nn.Parameter(torch.zeros(1, cout, 1, 1))
            if styled:
                self.affine = nn.Linear(W_DIM, cout * 2)
            else:
                self.norm = nn.BatchNorm2d(cout)
            self.act = nn.LeakyReLU(0.2, True)

        def forward(self, x, w, noise=None):
            if self.up:
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = self.conv(x)
            if noise is None:
                noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])
            x = x + self.noise * noise
            if self.styled:
                mu = x.mean(dim=(2, 3), keepdim=True)
                sd = x.std(dim=(2, 3), keepdim=True) + 1e-5
                s = self.affine(w)
                gamma, beta = s[:, :x.shape[1]], s[:, x.shape[1]:]
                x = (x - mu) / sd * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
            else:
                x = self.norm(x)
            return self.act(x)

    class Gen(nn.Module):
        def __init__(self, styled):
            super().__init__()
            self.styled = styled
            self.mapping = Mapping() if styled else None
            if styled:
                self.const = nn.Parameter(torch.randn(1, CH[0], 4, 4) * 0.5)
            else:
                self.fc = nn.Linear(Z_DIM, CH[0] * 4 * 4)
            self.blocks = nn.ModuleList([
                Block(CH[0], CH[0], False, styled),
                Block(CH[0], CH[1], True, styled),
                Block(CH[1], CH[2], False, styled),
                Block(CH[2], CH[3], True, styled),
                Block(CH[3], CH[4], True, styled)])
            self.to_rgb = nn.Conv2d(CH[4], 3, 1)

        def forward(self, z, ws=None, noises=None):
            n = z.shape[0]
            if self.styled:
                if ws is None:
                    w = self.mapping(z)
                    ws = [w] * len(self.blocks)
                x = self.const.expand(n, -1, -1, -1)
            else:
                ws = [None] * len(self.blocks)
                x = self.fc(z).view(n, CH[0], 4, 4)
            for i, b in enumerate(self.blocks):
                x = b(x, ws[i], None if noises is None else noises[i])
            return torch.sigmoid(self.to_rgb(x))

    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 24, 4, 2, 1), nn.GroupNorm(4, 24), nn.LeakyReLU(0.2, True),
                nn.Conv2d(24, 48, 4, 2, 1), nn.GroupNorm(4, 48), nn.LeakyReLU(0.2, True),
                nn.Conv2d(48, 64, 4, 2, 1), nn.GroupNorm(4, 64), nn.LeakyReLU(0.2, True),
                nn.Flatten())
            self.fc = nn.Linear(64 * 4 * 4, 1)

        def forward(self, x):
            return self.fc(self.net(x))

    return Gen(kind == "styled"), Disc()


def train_gan(kind, seed, images, steps, batch):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    X = torch.from_numpy(images.transpose(0, 3, 1, 2).copy())
    G, D = build(kind, seed)
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    curve = []
    for it in range(steps):
        real = X[torch.randint(0, X.shape[0], (batch,))]
        with torch.no_grad():
            fake = G(torch.randn(batch, Z_DIM))
        ld = bce(D(real), torch.ones(batch, 1)) + bce(D(fake), torch.zeros(batch, 1))
        od.zero_grad(); ld.backward(); od.step()
        out = D(G(torch.randn(batch, Z_DIM)))
        lg = bce(out, torch.ones(batch, 1))
        og.zero_grad(); lg.backward(); og.step()
        if (it + 1) % max(1, steps // 10) == 0:
            curve.append(dict(it=it + 1, d=rnd(ld.item(), 4), g=rnd(lg.item(), 4)))
            print(f"      {kind}/{seed} step {it+1}: d {ld.item():.3f} g {lg.item():.3f}",
                  flush=True)
    G.eval()
    return dict(state={k: v.detach().numpy() for k, v in G.state_dict().items()},
                curve=curve, params=sum(p.numel() for p in G.parameters()),
                dparams=sum(p.numel() for p in D.parameters()))


def load_gen(kind, seed, state):
    import torch

    G, _ = build(kind, seed)
    G.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    G.eval()
    return G


def generate(G, n, seed=0, psi=1.0, w_mean=None, batch=256):
    import torch

    torch.manual_seed(seed)
    out = []
    with torch.no_grad():
        for i in range(0, n, batch):
            m = min(batch, n - i)
            z = torch.randn(m, Z_DIM)
            if G.styled and psi != 1.0:
                w = G.mapping(z)
                w = w_mean + psi * (w - w_mean)
                out.append(G(z, ws=[w] * len(G.blocks)).numpy())
            else:
                out.append(G(z).numpy())
    return np.concatenate(out).transpose(0, 2, 3, 1)


# ----------------------------------------------------------------------------
# 3. StyleGAN's own metrics
# ----------------------------------------------------------------------------
def slerp(a, b, t):
    """Spherical interpolation, which is what the paper uses in z because a
    gaussian's mass lives on a shell and a straight line between two points on
    a shell goes through the middle, where the generator has never been."""
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    d = np.clip((an * bn).sum(axis=1, keepdims=True), -1, 1)
    om = np.arccos(d)
    so = np.sin(om)
    flat = so < 1e-6
    out = np.where(flat, a * (1 - t) + b * t,
                   np.sin((1 - t) * om) / np.where(flat, 1, so) * a
                   + np.sin(t * om) / np.where(flat, 1, so) * b)
    return out


def feature_dist(body, a, b):
    fa = read_features(body, a)
    fb = read_features(body, b)
    return ((fa - fb) ** 2).sum(axis=1)


def factor_vector(m, shapes):
    """The true content of an image as a vector in comparable units: position
    in pixels over the image size, size over its range, hue as a distance on
    the circle, shape as a one-hot. Everything on roughly [0, 1] so a step in
    one is worth a step in another."""
    n = len(m["cx"])
    out = np.zeros((n, 4 + len(SHAPES)))
    out[:, 0] = m["cx"] / RES
    out[:, 1] = m["cy"] / RES
    out[:, 2] = (m["size"] - SIZE_LO) / (SIZE_HI - SIZE_LO)
    hue = np.nan_to_num(m["hue"])
    out[:, 3] = hue
    for i in range(len(SHAPES)):
        out[:, 4 + i] = (shapes == i).astype(float)
    return out


def factor_step(ma, mb, sa, sb):
    """Distance between two images in true factors, with hue on a circle."""
    d = ((ma["cx"] - mb["cx"]) / RES) ** 2 + ((ma["cy"] - mb["cy"]) / RES) ** 2
    d = d + ((ma["size"] - mb["size"]) / (SIZE_HI - SIZE_LO)) ** 2
    d = d + hue_dist(np.nan_to_num(ma["hue"]), np.nan_to_num(mb["hue"])) ** 2
    d = d + (sa != sb).astype(float)
    return d


def path_lengths(G, body, reader, kind, n, eps=1e-2, seed=3):
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = {}
    spaces = ["z"] + (["w"] if G.styled else [])
    for space in spaces:
        z1 = rng.normal(size=(n, Z_DIM))
        z2 = rng.normal(size=(n, Z_DIM))
        t = rng.uniform(0, 1 - eps, size=(n, 1))
        with torch.no_grad():
            if space == "z":
                a = slerp(z1, z2, t)
                b = slerp(z1, z2, t + eps)
                ia = G(torch.from_numpy(a).float()).numpy().transpose(0, 2, 3, 1)
                ib = G(torch.from_numpy(b).float()).numpy().transpose(0, 2, 3, 1)
            else:
                w1 = G.mapping(torch.from_numpy(z1).float()).numpy()
                w2 = G.mapping(torch.from_numpy(z2).float()).numpy()
                a = w1 * (1 - t) + w2 * t
                b = w1 * (1 - t - eps) + w2 * (t + eps)
                ta = torch.from_numpy(a).float()
                tb = torch.from_numpy(b).float()
                ia = G(torch.zeros(n, Z_DIM), ws=[ta] * len(G.blocks)).numpy().transpose(0, 2, 3, 1)
                ib = G(torch.zeros(n, Z_DIM), ws=[tb] * len(G.blocks)).numpy().transpose(0, 2, 3, 1)
        perc = feature_dist(body, ia, ib) / eps ** 2
        fac = factor_step(measure(ia), measure(ib), read_shapes(reader, ia),
                          read_shapes(reader, ib)) / eps ** 2
        out[space] = dict(
            perceptual=rnd(float(np.mean(perc)), 4),
            perceptual_p90=rnd(float(np.percentile(perc, 90)), 4),
            factor=rnd(float(np.mean(fac)), 4),
            factor_p90=rnd(float(np.percentile(fac, 90)), 4),
            roughness=rnd(float(np.percentile(perc, 90) / max(np.median(perc), 1e-9)), 4),
            n=n)
    return out


def separability(G, reader, kind, n, seed=4):
    """How linearly the latent codes carry each factor.

    A logistic regression from the latent to a binary attribute, scored on
    held-out samples, against the accuracy of predicting the majority class,
    which is the only baseline that means anything when the attribute is
    lopsided.
    """
    import torch
    from sklearn.linear_model import LogisticRegression

    torch.manual_seed(seed)
    z = torch.randn(n, Z_DIM)
    with torch.no_grad():
        if G.styled:
            w = G.mapping(z)
            imgs = G(z, ws=[w] * len(G.blocks)).numpy().transpose(0, 2, 3, 1)
            spaces = dict(z=z.numpy(), w=w.numpy())
        else:
            imgs = G(z).numpy().transpose(0, 2, 3, 1)
            spaces = dict(z=z.numpy())
    m = measure(imgs)
    sh = read_shapes(reader, imgs)
    attrs = dict(
        large=(m["size"] > BIG * 0.85).astype(int),
        blue=(((np.nan_to_num(m["hue"]) >= BLUE_LO) & (np.nan_to_num(m["hue"]) < BLUE_HI))
              .astype(int)),
        right=(m["cx"] > RES / 2).astype(int),
        disc=(sh == 0).astype(int),
    )
    out = {}
    cut = int(n * 0.7)
    for space, X in spaces.items():
        rows = {}
        for name, y in attrs.items():
            base = max(y[cut:].mean(), 1 - y[cut:].mean())
            if len(np.unique(y[:cut])) < 2:
                rows[name] = dict(acc=None, base=rnd(base, 4))
                continue
            clf = LogisticRegression(max_iter=2000).fit(X[:cut], y[:cut])
            rows[name] = dict(acc=rnd(clf.score(X[cut:], y[cut:]), 4), base=rnd(base, 4))
        out[space] = rows
    return out


def hole_leak(imgs, reader):
    """How much of what a generator produces lands in the region the data
    never occupies. Real data is exactly zero there by construction."""
    m = measure(imgs)
    hue = np.nan_to_num(m["hue"])
    big = m["size"] > BIG
    blue = (hue >= BLUE_LO) & (hue < BLUE_HI)
    return dict(share=rnd(float((big & blue).mean()), 5),
                big=rnd(float(big.mean()), 4), blue=rnd(float(blue.mean()), 4),
                expected_if_independent=rnd(float(big.mean() * blue.mean()), 5))


# ----------------------------------------------------------------------------
# 4. Style mixing
# ----------------------------------------------------------------------------
def style_mixing(G, reader, n=256, seed=6):
    """Which factor each layer's style controls, attributed rather than
    asserted: swap ONE layer's style from another sample and measure how far
    each factor travelled towards that other sample.

    Two decisions that a first version got wrong and the numbers caught.

    The travel is NOT clipped. A clip at 1.5 turned every overshoot into the
    same 1.5 and then the spread across blocks, which is what the page reads
    as "localisation", was reading the ceiling rather than the model. An
    overshoot is a real and interesting outcome: the block that owns a factor
    can push it past the target, which is exactly what "owns" fails to say.

    A pair whose two samples already agree on a factor cannot answer "how far
    did that factor travel", and dividing by their near-zero difference turns
    measurement noise into an enormous ratio. So each factor is scored only on
    the pairs that differ by more than a visible amount, one pixel of geometry
    or a tenth of the hue circle, and the share of pairs that survives is
    exported with the row."""
    import torch

    torch.manual_seed(seed)
    nb = len(G.blocks)
    with torch.no_grad():
        wa = G.mapping(torch.randn(n, Z_DIM))
        wb = G.mapping(torch.randn(n, Z_DIM))
        ia = G(torch.zeros(n, Z_DIM), ws=[wa] * nb).numpy().transpose(0, 2, 3, 1)
        ib = G(torch.zeros(n, Z_DIM), ws=[wb] * nb).numpy().transpose(0, 2, 3, 1)
    ma, mb = measure(ia), measure(ib)
    sa, sb = read_shapes(reader, ia), read_shapes(reader, ib)
    keys = ["cx", "cy", "size", "hue"]
    visible = {"cx": 1.0, "cy": 1.0, "size": 1.0, "hue": 0.1}
    denom = {k: (hue_dist(ma[k], mb[k]) if k == "hue" else np.abs(ma[k] - mb[k])) for k in keys}
    use = {k: denom[k] > visible[k] for k in keys}
    rows = []
    for layer in range(nb):
        ws = [wa] * nb
        ws[layer] = wb
        with torch.no_grad():
            im = G(torch.zeros(n, Z_DIM), ws=ws).numpy().transpose(0, 2, 3, 1)
        mm = measure(im)
        sm = read_shapes(reader, im)
        moved = {}
        for k in keys:
            d = hue_dist(ma[k], mm[k]) if k == "hue" else np.abs(ma[k] - mm[k])
            m = use[k]
            moved[k] = rnd(float(np.median(d[m] / denom[k][m])), 4) if m.sum() >= 30 else None
        agree = (sa != sb)
        moved["shape"] = rnd(float((sm[agree] == sb[agree]).mean()) if agree.any() else 0.0, 4)
        rows.append(dict(layer=layer, res=[4, 8, 8, 16, 32][layer], **moved))
    share = {k: rnd(float(use[k].mean()), 4) for k in keys}
    share["shape"] = rnd(float((sa != sb).mean()), 4)
    return dict(rows=rows, keys=keys + ["shape"], n=n, pairs_used=share,
                visible=visible)


# ----------------------------------------------------------------------------
# 5. The noise inputs
# ----------------------------------------------------------------------------
def noise_study(G, reader, n_w=64, n_noise=24, seed=8):
    import torch

    torch.manual_seed(seed)
    nb = len(G.blocks)
    shapes = [(4, 4), (4, 4), (8, 8), (8, 8), (16, 16)]
    sizes = [(1, 1, 4, 4), (1, 1, 8, 8), (1, 1, 8, 8), (1, 1, 16, 16), (1, 1, 32, 32)]
    with torch.no_grad():
        w = G.mapping(torch.randn(n_w, Z_DIM))
    per_pixel = []
    factors = {k: [] for k in ("cx", "cy", "size", "hue")}
    for i in range(n_w):
        wi = w[i:i + 1].expand(n_noise, -1)
        noises = [torch.randn(n_noise, 1, s[2], s[3]) for s in sizes]
        with torch.no_grad():
            imgs = G(torch.zeros(n_noise, Z_DIM), ws=[wi] * nb,
                     noises=noises).numpy().transpose(0, 2, 3, 1)
        per_pixel.append(imgs.std(axis=0).mean(axis=2))
        m = measure(imgs)
        for k in factors:
            if k == "hue":
                h = np.nan_to_num(m[k])
                factors[k].append(float(np.mean(hue_dist(h, np.median(h)))))
            else:
                factors[k].append(float(np.std(m[k])))
    std_map = np.mean(per_pixel, axis=0)
    # low frequency part of the variation: a 4 pixel box blur of the std map
    k = 4
    pad = np.pad(std_map, k // 2, mode="edge")
    low = np.zeros_like(std_map)
    for dy in range(k):
        for dx in range(k):
            low += pad[dy:dy + RES, dx:dx + RES]
    low /= k * k
    high = std_map - low
    # the same factors, but varying w instead of the noise, as the control
    with torch.no_grad():
        many = G(torch.randn(n_w * 4, Z_DIM)).numpy().transpose(0, 2, 3, 1)
    mm = measure(many)
    across_w = {}
    for kk in factors:
        if kk == "hue":
            h = np.nan_to_num(mm[kk])
            across_w[kk] = rnd(float(np.mean(hue_dist(h, np.median(h)))), 5)
        else:
            across_w[kk] = rnd(float(np.std(mm[kk])), 5)
    # The learned scale on each noise map, which is the model's own answer to
    # "how much of this picture should be left to chance": one number per
    # channel per block, reported as the mean absolute value per block against
    # the typical size of the activations it is added to.
    scales = [rnd(float(np.abs(b.noise.detach().numpy()).mean()), 6) for b in G.blocks]
    return dict(
        learned_scales=scales,
        std_map=np.round(std_map, 5).tolist(),
        mean_std=rnd(float(std_map.mean()), 5),
        high_share=rnd(float(np.abs(high).mean() / max(np.abs(std_map).mean(), 1e-9)), 4),
        across_noise={k: rnd(float(np.mean(v)), 5) for k, v in factors.items()},
        across_w=across_w, n_w=n_w, n_noise=n_noise)


# ----------------------------------------------------------------------------
# 6. Truncation
# ----------------------------------------------------------------------------
def frechet(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    wa, va = np.linalg.eigh(ca)
    sa = va @ np.diag(np.sqrt(np.maximum(wa, 0))) @ va.T
    wm = np.linalg.eigvalsh(sa @ cb @ sa)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb)
                 - 2 * np.sqrt(np.maximum(wm, 0)).sum())


def blur_k(x, k):
    """Box blur over the two spatial axes, edge padded, and the SAME size out
    as in. The first version of this dropped a row and a column, because a
    difference of cumulative sums needs the leading zero that `cumsum` does not
    produce, and nothing complained: the reader that consumed the result is
    convolutional, so it happily scored a 31 by 31 image against 32 by 32 ones
    and the blurred arm of the yardstick came out inflated by a half pixel
    shift on top of the blur it was supposed to measure."""
    assert k % 2 == 1, f"an even box blur has no centre pixel: k = {k}"
    out = x.copy()
    pad = k // 2
    for ax in (1, 2):
        p = np.pad(out, [(0, 0)] + [(pad, pad) if a == ax else (0, 0) for a in (1, 2)]
                   + [(0, 0)], mode="edge")
        sl = lambda s, e: tuple(slice(None) if a != ax else slice(s, e) for a in range(4))
        c = np.cumsum(p, axis=ax)
        c = np.concatenate([np.zeros_like(c[sl(0, 1)]), c], axis=ax)
        out = (c[sl(k, None)] - c[sl(0, -k)]) / k
        assert out.shape == x.shape, f"blur changed the image size: {out.shape} from {x.shape}"
    return out


def hf_energy(imgs):
    """One number per image: how much of it a three pixel blur removes. This is
    the cheapest thing a discriminator can look at, and on a grainy dataset it
    is enough to win with."""
    return np.abs(imgs - blur_k(imgs, 3)).mean(axis=(1, 2, 3))


def auc(pos, neg):
    """Rank based, so it is exactly the probability that a random positive
    scores above a random negative, with ties counted as half."""
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def blob_score(imgs):
    """Is this a sprite: one connected patch of one saturated colour on a flat
    background. Measured with the article's own arithmetic rather than with a
    distance in some network's features, and grain blind by construction,
    because the mask is taken after a three pixel blur.

    Three numbers rather than a verdict, each with the value real sprites give
    printed next to it, because every single one of them can be gamed by a
    failure of a different kind: mush that covers the whole tile is one
    connected region, and so is a sprite. Read the row, not a cell."""
    sm = blur_k(imgs, 3)
    border = np.concatenate([sm[:, 0, :, :], sm[:, -1, :, :],
                             sm[:, :, 0, :], sm[:, :, -1, :]], axis=1)
    bg = np.median(border, axis=1)[:, None, None, :]
    mask = np.abs(sm - bg).sum(axis=-1) > 0.30
    ones, cover, sat, counts = [], [], [], []
    for i in range(len(imgs)):
        m = mask[i]
        cover.append(m.mean())
        # connected components, four neighbours, by flood fill on the mask
        lab = np.zeros_like(m, dtype=np.int32)
        cur = 0
        big = 0
        ys, xs = np.nonzero(m)
        for y0, x0 in zip(ys, xs):
            if lab[y0, x0]:
                continue
            cur += 1
            stack = [(y0, x0)]
            lab[y0, x0] = cur
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = cur
                        stack.append((yy, xx))
            if size >= 4:
                big += 1
        ones.append(1.0 if big == 1 else 0.0)
        counts.append(big)
        px = imgs[i][m]
        if len(px) >= 4:
            mu = px.mean(axis=0)
            sat.append((mu.max() - mu.min()) / max(mu.max(), 1e-9))
    return dict(one_blob=rnd(float(np.mean(ones)), 4),
                blobs=rnd(float(np.mean(counts)), 3),
                coverage=rnd(float(np.mean(cover)), 4),
                sat=rnd(float(np.mean(sat)) if sat else 0.0, 4))


def grain_probe(reader, body, grains, steps, seed=1, n=8000, n_eval=1500):
    """Why this dataset has no per pixel grain, measured instead of asserted.

    The first version of this article drew its sprites with a grain of 0.04 and
    the generator produced mush. Four controlled probes later the cause was the
    data rather than the model: a grain is a texture a generator has to match
    before anything else, and until it does, one number tells every one of its
    samples from every real sprite. So the discriminator can win without ever
    looking at a shape, and a discriminator that is winning hands back nothing
    to learn from.

    The arms are the article's own run and two of its counterfactuals: same
    architecture, same seed, same number of steps, same sprites down to the
    factor, and the only difference is how much grain sits on the pixels. The
    grainless arm is not a re-run of the published generator, it IS the
    published generator, reached through the same cache key, which is why the
    comparison is at the page's budget rather than at a cheap one.

    The shortcut is free to measure: hold the sprites fixed and compare the
    grainy version against the grainless one with the single high frequency
    statistic, which is the probability that one number separates them.

    The outcome is read three ways and they do not all agree, which is
    published rather than resolved. A Frechet distance in the reader's features
    with the grain blurred off both sides ranks the arms one way, and counting
    what a sample IS, one connected patch of one saturated colour, ranks them
    another. Both are printed, with real sprites and a flat grey rectangle as
    the two controls that say what each scale can and cannot see."""
    import torch

    rows = []
    rng0 = np.random.default_rng(19)
    f = sample_factors(n, rng0)
    clean = draw(f, np.random.default_rng(21), grain=0.0)
    ref = read_features(body, blur_k(clean[:n_eval], 3))
    for g in grains:
        imgs = draw(f, np.random.default_rng(21), grain=g)
        shortcut = auc(hf_energy(imgs[:n_eval]), hf_energy(clean[:n_eval]))
        # deliberately the SAME key template the main stage uses, so that the
        # grainless arm of this probe is a cache hit on the published model
        run = cached(f"gan-styled-s{seed}-st{steps}-b64-z{Z_DIM}-ch{CH}-g{g}-v2",
                     lambda g=g, imgs=imgs: train_gan("styled", seed, imgs, steps, 64))
        G = load_gen("styled", seed, run["state"])
        fake = generate(G, n_eval, seed=31)
        fd = frechet(ref, read_features(body, blur_k(fake, 3)))
        # and the same one number at the END of training, now between what this
        # generator draws and what it was trying to copy: if the grain is a
        # shortcut the discriminator never has to give up, it is still there.
        # Reported two sided, because a generator can end up NOISIER than its
        # data as easily as smoother, and either way one number still tells
        # them apart.
        after = auc(hf_energy(imgs[:n_eval]), hf_energy(fake))
        # how easy the game stayed for the discriminator, averaged over the last
        # tenth of training rather than read off one batch
        tail = [c["d"] for c in run["curve"][-3:]]
        blob = blob_score(fake[:400])
        # and whether it still draws more than one thing, because the grainiest
        # arm is not the only way to fail: a run can be saturated, clean and
        # collapsed onto a single shape, and the columns above would all like it
        sh = read_shapes(reader, fake)
        blob["top_shape"] = rnd(float(np.bincount(sh, minlength=len(SHAPES)).max() / len(sh)), 4)
        rows.append(dict(grain=g, shortcut=rnd(shortcut, 4),
                         shortcut_end=rnd(max(after, 1 - after), 4),
                         fake_noisier=bool(after < 0.5),
                         fd_blind=rnd(fd, 4), d_loss=rnd(float(np.mean(tail)), 4),
                         hf_real=rnd(float(hf_energy(imgs[:n_eval]).mean()), 5),
                         hf_fake=rnd(float(hf_energy(fake).mean()), 5), **blob))
        print(f"      grain {g}: one number separates it from the same sprites clean "
              f"{shortcut*100:.1f}% of the time, and from what the generator drew "
              f"{max(after, 1-after)*100:.1f}%; patches {blob['blobs']:.2f}, "
              f"coverage {blob['coverage']*100:.1f}%, saturation {blob['sat']:.3f}, "
              f"commonest shape {blob['top_shape']*100:.1f}%, blind distance {fd:.2f}", flush=True)
    # The two controls that put a top and a bottom on every column above: real
    # sprites, and a flat grey rectangle, which is what "made no effort at all"
    # scores on each of these scales.
    grey = np.full((n_eval, RES, RES, 3), float(np.median(clean)), np.float32)
    real_blob = blob_score(clean[:400])
    real_sh = read_shapes(reader, clean[:400])
    real_blob["top_shape"] = rnd(float(np.bincount(real_sh, minlength=len(SHAPES)).max()
                                       / len(real_sh)), 4)
    ctrl = dict(real=real_blob,
                grey=dict(fd_blind=rnd(frechet(ref, read_features(body, blur_k(grey, 3))), 4),
                          **blob_score(grey[:50])),
                real_fd=rnd(frechet(ref, read_features(body,
                                                       blur_k(clean[n_eval:2 * n_eval], 3))), 4))
    print(f"      controls: real sprites are {ctrl['real']['blobs']:.2f} patches covering "
          f"{ctrl['real']['coverage']*100:.1f}% at saturation {ctrl['real']['sat']:.3f}; a flat "
          f"grey rectangle scores {ctrl['grey']['fd_blind']:.2f} on the blind distance, against "
          f"{ctrl['real_fd']:.2f} for real sprites", flush=True)
    assert ctrl["real"]["one_blob"] > 0.9, \
        f"real sprites do not read as one blob, so the score cannot judge anything: {ctrl['real']}"
    return dict(rows=rows, steps=steps, seed=seed, n=n, n_eval=n_eval, control=ctrl)


def truncation(G, body, reader, real_feats, n=1500, seed=10):
    import torch

    with torch.no_grad():
        w_mean = G.mapping(torch.randn(8192, Z_DIM)).mean(dim=0, keepdim=True)
    rows = []
    for psi in [0.0, 0.25, 0.5, 0.7, 0.85, 1.0, 1.15]:
        imgs = generate(G, n, seed=seed, psi=psi, w_mean=w_mean)
        f = read_features(body, imgs)
        m = measure(imgs)
        leak = hole_leak(imgs, reader)
        rows.append(dict(psi=psi, fd=rnd(frechet(real_feats, f), 4),
                         hole=leak["share"],
                         size_sd=rnd(float(np.std(m["size"])), 4),
                         hue_sd=rnd(float(np.std(np.nan_to_num(m["hue"]))), 4),
                         shapes=int(len(np.unique(read_shapes(reader, imgs))))))
        print(f"      psi {psi}: fd {rows[-1]['fd']:.2f} hole {leak['share']*100:.2f}% "
              f"size sd {rows[-1]['size_sd']:.2f}", flush=True)
    return dict(rows=rows, w_mean=w_mean.numpy()[0].tolist())


def main():
    print(f"threads: {THREADS}{'  (SMALL smoke run)' if SMALL else ''}", flush=True)
    print("0. the sprites", flush=True)
    data = cached(f"sprites-{RES}-{SMALL}-v2", stage_dataset)
    print(f"    {data['n']} sprites, measurement floor {data['floor']}", flush=True)
    print("1. the shape reader", flush=True)
    reader = stage_reader(data)
    print("2. two generators", flush=True)
    steps = 40 if SMALL else 5000
    batch = 64
    runs = {}
    for kind in ("plain", "styled"):
        for seed in ([1] if SMALL else [1, 2]):
            key = f"gan-{kind}-s{seed}-st{steps}-b{batch}-z{Z_DIM}-ch{CH}-g{GRAIN}-v2"
            runs[f"{kind}/{seed}"] = cached(key,
                                            lambda k=kind, s=seed: train_gan(k, s, data["images"],
                                                                             steps, batch))
    print("   trained:", {k: v["params"] for k, v in runs.items()}, flush=True)

    import torch

    net, body, head, reader_acc = reader
    seeds = [1] if SMALL else [1, 2]

    # Quantise first, measure second. The browser gets float16 weights, so
    # every figure this article quotes is measured on the weights the browser
    # will have rather than on the ones the optimiser produced.
    quant = {}
    q_error = {}
    for key, r in runs.items():
        kind = key.split("/")[0]
        seed = int(key.split("/")[1])
        state32 = r["state"]
        state16 = {k: v.astype(np.float16).astype(np.float32) for k, v in state32.items()}
        G32 = load_gen(kind, seed, state32)
        G16 = load_gen(kind, seed, state16)
        torch.manual_seed(1)
        z = torch.randn(128, Z_DIM)
        with torch.no_grad():
            noises = [torch.zeros(128, 1, s, s) for s in (4, 8, 8, 16, 32)]
            ws = None
            if G32.styled:
                ws = [G32.mapping(z)] * len(G32.blocks)
            a = G32(z, ws=ws, noises=noises).numpy()
            ws16 = [G16.mapping(z)] * len(G16.blocks) if G16.styled else None
            b = G16(z, ws=ws16, noises=noises).numpy()
        q_error[key] = rnd(float(np.abs(a - b).mean()), 6)
        quant[key] = dict(state=state16, G=G16)
    print(f"   float16 costs {max(q_error.values()):.2e} per pixel at worst", flush=True)

    n_eval = 400 if SMALL else 3000
    real_feats = read_features(body, data["images"][:n_eval])
    # The yardstick, calibrated before anything is judged by it: two disjoint
    # halves of the real sprites (the floor, what identical looks like), the
    # same sprites blurred, and uniform noise (the ceiling). Without these,
    # "151 against 384" is two numbers with no scale under them.
    other = read_features(body, data["images"][n_eval:2 * n_eval])
    rng_c = np.random.default_rng(77)
    noise_imgs = rng_c.uniform(0, 1, (n_eval, RES, RES, 3)).astype(np.float32)

    scale = dict(
        floor=rnd(frechet(real_feats, other), 4),
        blur3=rnd(frechet(real_feats, read_features(body, blur_k(data["images"][:n_eval], 3))), 4),
        noise=rnd(frechet(real_feats, read_features(body, noise_imgs)), 4),
        n=n_eval)
    print(f"    the yardstick: real against real {scale['floor']:.2f}, blurred "
          f"{scale['blur3']:.2f}, noise {scale['noise']:.2f}", flush=True)

    # And a floor under the hole itself. The rectangle is exactly empty in the
    # factors that were DRAWN, but the numbers this page compares are measured
    # off pixels, and the measured size carries a bias of about a third of a
    # pixel from the antialiased edge. A sprite drawn just under the threshold
    # measures just over it, so a share of the real data lands in the forbidden
    # region by measurement error alone. Every generator's share has to be read
    # against this, not against zero.
    real_hole = hole_leak(data["images"][:n_eval], net)
    scale["hole_floor"] = real_hole["share"]
    scale["hole_floor_expected"] = real_hole["expected_if_independent"]
    print(f"    the hole, measured off real pixels: {real_hole['share']*100:.2f}% "
          f"(exactly 0 in the factors that were drawn)", flush=True)
    assert scale["floor"] < scale["blur3"] < scale["noise"], \
        f"the yardstick is not ordered, so it cannot rank anything: {scale}"

    print("3. path lengths and separability", flush=True)
    metrics = {}
    real_m = measure(data["images"][:900])
    scatter = {"real": [[rnd(real_m["size"][i], 3), rnd(float(np.nan_to_num(real_m["hue"][i])), 4)]
                        for i in range(len(real_m["size"]))]}
    for key in runs:
        kind = key.split("/")[0]
        G = quant[key]["G"]
        n = 60 if SMALL else 600
        pl = cached(f"ppl-{key}-{n}-v3", lambda G=G, kind=kind, n=n:
                    path_lengths(G, body, net, kind, n))
        sep = cached(f"sep-{key}-v3", lambda G=G, kind=kind:
                     separability(G, net, kind, 400 if SMALL else 4000))
        imgs = generate(G, n_eval, seed=21)
        leak = hole_leak(imgs, net)
        fd = frechet(real_feats, read_features(body, imgs))
        mm = measure(imgs[:900])
        scatter[kind if key.endswith(f"/{seeds[0]}") else key] = [
            [rnd(mm["size"][i], 3), rnd(float(np.nan_to_num(mm["hue"][i])), 4)]
            for i in range(len(mm["size"]))]
        metrics[key] = dict(path=pl, sep=sep, hole=leak, fd=rnd(fd, 4),
                            params=runs[key]["params"], dparams=runs[key]["dparams"],
                            q_error=q_error[key])
        print(f"    {key}: fd {fd:.2f}, hole {leak['share']*100:.3f}% "
              f"(independent would be {leak['expected_if_independent']*100:.2f}%), "
              f"ppl z {pl['z']['perceptual']:.1f}"
              + (f" w {pl['w']['perceptual']:.1f}" if 'w' in pl else ''), flush=True)

    print("4. style mixing", flush=True)
    styled_key = f"styled/{seeds[0]}"
    Gs = quant[styled_key]["G"]
    mixing = cached(f"mix-{styled_key}-v4", lambda: style_mixing(Gs, net,
                                                                 n=64 if SMALL else 400))
    print("    pairs that differ visibly: "
          + ", ".join(f"{k} {v*100:.0f}%" for k, v in mixing["pairs_used"].items()), flush=True)
    for r in mixing["rows"]:
        print(f"    layer {r['layer']} ({r['res']}x{r['res']}): "
              + ", ".join(f"{k} " + ("--" if r[k] is None else f"{r[k]:.2f}")
                          for k in mixing["keys"]), flush=True)

    print("5. the noise inputs", flush=True)
    noise = cached(f"noise-{styled_key}-v3",
                   lambda: noise_study(Gs, net, n_w=8 if SMALL else 48,
                                       n_noise=6 if SMALL else 24))
    print(f"    per pixel std {noise['mean_std']:.4f}, "
          f"{noise['high_share']*100:.1f}% of it high frequency", flush=True)

    print("6. truncation", flush=True)
    trunc = cached(f"trunc-{styled_key}-v3",
                   lambda: truncation(Gs, body, net, real_feats, n=300 if SMALL else 1500))

    print("7. the grain, which is why this dataset has none", flush=True)
    grain = grain_probe(net, body, grains=[GRAIN, 0.015, 0.04],
                        steps=steps, seed=seeds[0],
                        n=data["n"], n_eval=300 if SMALL else 1500)
    assert grain["rows"][0]["shortcut"] == 0.5, \
        "the grainless arm has to be indistinguishable from itself, or the probe is broken"

    # ------------------------------------------------------------------ write
    print("writing", flush=True)
    rng = np.random.default_rng(31)
    real_tiles = data["images"][rng.permutation(len(data["images"]))[:32]]
    plain_tiles = generate(quant[f"plain/{seeds[0]}"]["G"], 32, seed=77)
    styled_tiles = generate(Gs, 32, seed=77)
    png = {}
    png["real"] = save_png(sheet(real_tiles, 16), IMG / "real.png")
    png["plain"] = save_png(sheet(plain_tiles, 16), IMG / "plain.png")
    png["styled"] = save_png(sheet(styled_tiles, 16), IMG / "styled.png")

    # the export the browser runs
    state = quant[styled_key]["state"]
    order = [k for k in state]
    vals = np.concatenate([state[k].ravel().astype(np.float32) for k in order])
    layers = {}
    off = 0
    for k in order:
        layers[k] = dict(offset=off, shape=list(state[k].shape))
        off += state[k].size
    b64 = base64.b64encode(vals.astype(np.float16).tobytes()).decode("ascii")
    torch.manual_seed(404)
    zc = torch.randn(4, Z_DIM)
    with torch.no_grad():
        wc = Gs.mapping(zc)
        noises = [torch.zeros(4, 1, s, s) for s in (4, 8, 8, 16, 32)]
        ref = Gs(zc, ws=[wc] * len(Gs.blocks), noises=noises).numpy().transpose(0, 2, 3, 1)
    idx = np.random.default_rng(5).permutation(RES * RES)[:96]
    check = dict(
        z=np.round(zc.numpy(), 6).tolist(),
        w=np.round(wc.numpy(), 6).tolist(),
        idx=idx.tolist(),
        rgb=[[[rnd(ref[i].reshape(-1, 3)[p][c], 6) for c in range(3)] for p in idx]
             for i in range(4)],
        note="rendered with every noise input set to zero, so both sides compute the same thing",
    )

    out = dict(
        meta=dict(res=RES, shapes=SHAPES, z=Z_DIM, w=W_DIM, ch=CH, threads=THREADS,
                  steps=steps, batch=batch, seeds=seeds, n_train=data["n"],
                  size_lo=SIZE_LO, size_hi=SIZE_HI, pos_lo=POS_LO, pos_hi=POS_HI,
                  big=BIG, blue=[BLUE_LO, BLUE_HI], grain=GRAIN,
                  factors=list(data["factors"].keys()),
                  reader_acc=rnd(reader_acc, 4), png_bytes=png, tile=RES, sheet_cols=16,
                  blocks=[dict(res=r, ch=c) for r, c in zip([4, 8, 8, 16, 32], CH)]),
        dataset=dict(floor=data["floor"], hole=data["hole"]),
        metrics=metrics, mixing=mixing, noise=noise, truncation=trunc, scatter=scatter,
        scale=scale, grain=grain,
        net=dict(weights_b64=b64, count=int(vals.size), layers=layers,
                 w_mean=[rnd(v, 6) for v in trunc["w_mean"]],
                 format="float16 little-endian, base64, concatenated in the order of `layers`"),
        check=check,
    )
    (OUT / "style.json").write_text(json.dumps(out), "utf-8")
    print(f"wrote {OUT}/style.json ({len(json.dumps(out))/1024:.0f} kB) and "
          f"{len(png)} sheets", flush=True)


def save_png(arr, path):
    from PIL import Image

    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path, optimize=True)
    return path.stat().st_size


def sheet(tiles, cols):
    t = np.asarray(tiles)
    n, h, w = t.shape[:3]
    rows = math.ceil(n / cols)
    out = np.zeros((rows * h, cols * w, 3))
    for k in range(n):
        r, c = divmod(k, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t[k]
    return out


if __name__ == "__main__":
    main()
