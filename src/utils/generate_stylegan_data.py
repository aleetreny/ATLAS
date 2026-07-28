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
    colour, on a background, with a grain of per-pixel noise on top.

Six factors, all written down, all recoverable from an image by arithmetic
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
GRAIN = 0.04


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
    """Six factors. The hole: a large sprite is never in the blue band, which
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
def stage_reader(data):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    torch.manual_seed(5)
    img = data["images"]
    y = data["factors"]["shape"]
    n_tr = int(len(img) * 0.85)
    x = torch.from_numpy(img.transpose(0, 3, 1, 2).copy())
    yt = torch.from_numpy(y).long()
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
    net = nn.Sequential(body, head)
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
    each factor travelled towards that other sample."""
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
    denom = {k: np.maximum(hue_dist(ma[k], mb[k]) if k == "hue" else np.abs(ma[k] - mb[k]), 1e-6)
             for k in keys}
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
            moved[k] = rnd(float(np.median(np.clip(d / denom[k], 0, 1.5))), 4)
        agree = (sa != sb)
        moved["shape"] = rnd(float((sm[agree] == sb[agree]).mean()) if agree.any() else 0.0, 4)
        rows.append(dict(layer=layer, res=[4, 8, 8, 16, 32][layer], **moved))
    return dict(rows=rows, keys=keys + ["shape"], n=n)


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
    return dict(
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
    data = cached(f"sprites-{RES}-{SMALL}-v1", stage_dataset)
    print(f"    {data['n']} sprites, measurement floor {data['floor']}", flush=True)
    print("1. the shape reader", flush=True)
    reader = stage_reader(data)
    print("2. two generators", flush=True)
    steps = 40 if SMALL else 4000
    batch = 64
    runs = {}
    for kind in ("plain", "styled"):
        for seed in ([1] if SMALL else [1, 2]):
            key = f"gan-{kind}-s{seed}-st{steps}-b{batch}-z{Z_DIM}-ch{CH}-v1"
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

    print("3. path lengths and separability", flush=True)
    metrics = {}
    real_m = measure(data["images"][:900])
    scatter = {"real": [[rnd(real_m["size"][i], 3), rnd(float(np.nan_to_num(real_m["hue"][i])), 4)]
                        for i in range(len(real_m["size"]))]}
    for key in runs:
        kind = key.split("/")[0]
        G = quant[key]["G"]
        n = 60 if SMALL else 600
        pl = cached(f"ppl-{key}-{n}-v2", lambda G=G, kind=kind, n=n:
                    path_lengths(G, body, net, kind, n))
        sep = cached(f"sep-{key}-v2", lambda G=G, kind=kind:
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
    mixing = cached(f"mix-{styled_key}-v2", lambda: style_mixing(Gs, net,
                                                                 n=64 if SMALL else 400))
    for r in mixing["rows"]:
        print(f"    layer {r['layer']} ({r['res']}x{r['res']}): "
              + ", ".join(f"{k} {r[k]:.2f}" for k in mixing["keys"]), flush=True)

    print("5. the noise inputs", flush=True)
    noise = cached(f"noise-{styled_key}-v2",
                   lambda: noise_study(Gs, net, n_w=8 if SMALL else 48,
                                       n_noise=6 if SMALL else 24))
    print(f"    per pixel std {noise['mean_std']:.4f}, "
          f"{noise['high_share']*100:.1f}% of it high frequency", flush=True)

    print("6. truncation", flush=True)
    trunc = cached(f"trunc-{styled_key}-v2",
                   lambda: truncation(Gs, body, net, real_feats, n=300 if SMALL else 1500))

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
                  reader_acc=rnd(reader_acc, 4), png_bytes=png, tile=RES, sheet_cols=16,
                  blocks=[dict(res=r, ch=c) for r, c in zip([4, 8, 8, 16, 32], CH)]),
        dataset=dict(floor=data["floor"], hole=data["hole"]),
        metrics=metrics, mixing=mixing, noise=noise, truncation=trunc, scatter=scatter,
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
