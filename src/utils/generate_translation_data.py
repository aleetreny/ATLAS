"""Data for the ATLAS 'Translation' article: pix2pix and CycleGAN.

The two previous articles asked a generator for a picture out of noise. This
one starts from a picture: a flat plan of a scene, to be turned into a render
of that scene. Which changes the problem in a way that is easy to state and
easy to get wrong. There is now a right answer, so an ordinary regression loss
is available; and there is still not a unique right answer, because the plan
does not say which windows are lit or how the grass falls, so the regression
loss is being asked a question it will answer with an average.

That is the whole first half of the article, and here it is not an anecdote:
the scene generator can render the SAME plan many times, so the conditional
distribution of renders given a plan can be sampled, its pixelwise median
computed exactly, and the blur that an L1 model produces compared against the
blur that L1 is mathematically obliged to produce. The blur is not a bug in
the model. It is the correct answer to the question that was asked.

Blocks:

0. The scenes. A plan and a render, paired, where the render has two genuinely
   random ingredients (the grain of the ground and which windows are lit) and
   everything else is a function of the plan. Plus the conditional statistics:
   for a handful of plans, many renders, their pixelwise median and mean, and
   the high frequency energy of each.

1. pix2pix: the same U-Net trained five ways. L1 alone, adversarial alone, the
   two together, and the adversarial one with two other discriminator sizes,
   because "how big is a patch" is the one hyperparameter of that paper that
   is genuinely about the loss rather than about the optimiser. Scored on
   fidelity (distance to the true render), on sharpness (high frequency energy
   against the real renders'), and on whether the windows come out committed or
   averaged.

2. The receptive field of each discriminator, measured by autodiff rather than
   computed from the strides, because that is the number the patch size is
   supposed to be.

3. CycleGAN, on the same two domains with the pairing thrown away. The plan
   carries a building type that decides the roof in the render, and the two
   types are equally common, so nothing in an unpaired objective distinguishes
   the right correspondence from the mirrored one. Several seeds, and a count
   of how many found which.

4. The cycle, and what it hides. A translation that has to be undoable can
   satisfy that by encoding the original in amplitudes nobody looks at, which
   is measurable: perturb the intermediate image slightly and watch the
   reconstruction fall apart, next to a control that perturbs a real image of
   the same domain by the same amount.

Run from anywhere: `python src/utils/generate_translation_data.py`. Stages
cache into ~/.atlas_vision_data/translation_cache/.
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
OUT = ROOT / "translation" / "data"
IMG = ROOT / "translation" / "img"
CACHE = Path.home() / ".atlas_vision_data" / "translation_cache"
for d in (OUT, IMG, CACHE):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

SMALL = os.environ.get("ATLAS_TRANS_SMALL") == "1"
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
# 0. The scenes
# ----------------------------------------------------------------------------
RES = 32
SKY = (0.72, 0.84, 0.94)
GROUND = (0.52, 0.72, 0.42)
ROAD = (0.54, 0.54, 0.57)
BODY = [(0.82, 0.44, 0.34), (0.44, 0.48, 0.78)]     # the two building types
# The plan's label colours are deliberately CROSSED against the render's: the
# type that draws blue on the plan renders red and the other way round. It
# costs the paired model nothing, since it is told the answer, and it makes the
# unpaired experiment mean something: the easy correspondence, the one that
# matches colour to colour, is the wrong one, and nothing in an unpaired
# objective knows that.
PLAN_BODY = [(0.35, 0.45, 0.85), (0.85, 0.40, 0.35)]
LIT = (0.98, 0.90, 0.42)
DARK = (0.20, 0.22, 0.28)
LIT_P = 0.45


def sample_plans(n, rng):
    horizon = rng.integers(9, 15, n)
    road_y = horizon + rng.integers(4, 9, n)
    road_h = rng.integers(4, 7, n)
    bx = rng.integers(2, 17, n)
    bw = rng.integers(7, 13, n)
    bh = rng.integers(7, 14, n)
    btype = rng.integers(0, 2, n)
    return dict(horizon=horizon, road_y=road_y, road_h=road_h, bx=bx, bw=bw, bh=bh, btype=btype)


def _windows(p, i):
    """Where the windows are, as a list of (y0, y1, x0, x1). Deterministic
    given the plan; only whether each is lit is random."""
    out = []
    top = p["horizon"][i] - p["bh"][i]
    x0, w = p["bx"][i], p["bw"][i]
    for wy in range(top + 3, p["horizon"][i] - 2, 4):
        for wx in range(x0 + 2, x0 + w - 2, 4):
            if wy + 2 <= RES and wx + 2 <= x0 + w:
                out.append((wy, wy + 2, wx, wx + 2))
    return out


def render(plans, i, rng=None, lit_mask=None):
    """One render of one plan. `rng` supplies the two random ingredients; pass
    `lit_mask` to fix the windows and vary only the ground."""
    img = np.zeros((RES, RES, 3), np.float32)
    hz = plans["horizon"][i]
    grad = np.linspace(0.0, 0.13, max(hz, 1))[:, None]
    img[:hz] = np.array(SKY) - grad[..., None].reshape(hz, 1, 1) * 0
    for y in range(hz):
        img[y, :, :] = np.array(SKY) - 0.10 * (1 - y / max(hz - 1, 1))
    img[hz:] = GROUND
    if rng is not None:
        img[hz:] += rng.normal(0, 0.055, (RES - hz, RES, 3))
        speck = rng.random((RES - hz, RES)) < 0.06
        img[hz:][speck] = np.array(GROUND) * 0.72
    ry, rh = plans["road_y"][i], plans["road_h"][i]
    ry = min(ry, RES - 2)
    rh = min(rh, RES - ry)
    img[ry:ry + rh] = ROAD
    mid = ry + rh // 2
    for x in range(1, RES - 1, 4):
        img[mid, x:x + 2] = (0.93, 0.93, 0.88)
    # the building, drawn after the ground so it sits in front of it
    top = hz - plans["bh"][i]
    x0, w = plans["bx"][i], plans["bw"][i]
    x1 = min(x0 + w, RES)
    body = np.array(BODY[plans["btype"][i]])
    img[max(top, 0):hz, x0:x1] = body
    if plans["btype"][i] == 1:
        # a pitched roof: three rows of triangle above the body
        for k in range(3):
            y = top - 3 + k
            if y < 0:
                continue
            inset = (2 - k) * max(1, w // 6)
            img[y, min(x0 + inset, RES - 1):max(x1 - inset, 0)] = body * 0.8
    else:
        y = max(top - 1, 0)
        img[y, x0:x1] = body * 0.8
    wins = _windows(plans, i)
    for k, (y0, y1, wx0, wx1) in enumerate(wins):
        if lit_mask is not None:
            on = lit_mask[k]
        elif rng is not None:
            on = rng.random() < LIT_P
        else:
            on = k % 2 == 0
        img[y0:y1, wx0:wx1] = LIT if on else DARK
    return np.clip(img, 0, 1)


def plan_image(plans, i):
    """The flat plan: the same layout in label colours, with no texture, no
    windows and no shading. This is the network's input."""
    img = np.zeros((RES, RES, 3), np.float32)
    hz = plans["horizon"][i]
    img[:hz] = SKY
    img[hz:] = GROUND
    ry, rh = plans["road_y"][i], plans["road_h"][i]
    ry = min(ry, RES - 2)
    rh = min(rh, RES - ry)
    img[ry:ry + rh] = ROAD
    top = hz - plans["bh"][i]
    x0 = plans["bx"][i]
    x1 = min(x0 + plans["bw"][i], RES)
    img[max(top, 0):hz, x0:x1] = PLAN_BODY[plans["btype"][i]]
    return img


def building_box(plans, i):
    top = max(plans["horizon"][i] - plans["bh"][i], 0)
    x0 = plans["bx"][i]
    x1 = min(x0 + plans["bw"][i], RES)
    return top, plans["horizon"][i], x0, x1


def read_type(plans, i, img):
    """Which building type a rendered image shows, by the colour of the body:
    the nearest of the two render colours to the mean of the building box."""
    y0, y1, x0, x1 = building_box(plans, i)
    if y1 <= y0 or x1 <= x0:
        return -1
    mean = np.asarray(img)[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    d = [float(((mean - np.array(c)) ** 2).sum()) for c in BODY]
    return int(np.argmin(d))


def make_pairs(n, seed):
    rng = np.random.default_rng(seed)
    plans = sample_plans(n, rng)
    A = np.stack([plan_image(plans, i) for i in range(n)])
    B = np.stack([render(plans, i, rng) for i in range(n)])
    return plans, A, B


def high_freq(img):
    """Mean absolute detail left after a 3x3 box blur: the energy an L1 loss
    is least able to keep, measured the same way for every image on the page."""
    x = np.asarray(img)
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge") if x.ndim == 4 \
        else np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            acc += pad[..., dy:dy + RES, dx:dx + RES, :] if x.ndim == 4 \
                else pad[dy:dy + RES, dx:dx + RES, :]
    return float(np.abs(x - acc / 9).mean())


def window_stats(plans, i, imgs):
    """How committed the windows are: the mean distance of the window pixels
    from the midpoint between lit and dark, over the maximum possible. One
    means every window picked a side, zero means every window is the average
    of both."""
    wins = _windows(plans, i)
    if not wins:
        return None
    mid = (np.array(LIT) + np.array(DARK)) / 2
    span = np.abs(np.array(LIT) - np.array(DARK)).mean() / 2
    vals = []
    for (y0, y1, x0, x1) in wins:
        patch = imgs[..., y0:y1, x0:x1, :]
        vals.append(np.abs(patch - mid).mean())
    return float(np.mean(vals) / span)


def stage_scenes():
    n = 800 if SMALL else 6000
    plans, A, B = make_pairs(n, 19)
    n_te = 120 if SMALL else 600
    tp, tA, tB = make_pairs(n_te, 23)
    # the conditional distribution, sampled: the same plan rendered many times
    m = 16 if SMALL else 128
    cond = []
    rng = np.random.default_rng(101)
    for i in range(4):
        many = np.stack([render(tp, i, rng) for _ in range(m)])
        med = np.median(many, axis=0)
        avg = many.mean(axis=0)
        # one window pixel across every render of this plan: the distribution
        # an averaging loss is being asked to summarise with a single number
        wins = _windows(tp, i)
        hist = []
        if wins:
            y0, _, x0, _ = wins[0]
            vals = many[:, y0, x0, :].mean(axis=1)
            edges = np.linspace(0, 1, 21)
            counts, _ = np.histogram(vals, bins=edges)
            hist = [dict(lo=rnd(edges[k], 3), hi=rnd(edges[k + 1], 3), n=int(counts[k]))
                    for k in range(len(counts))]
        cond.append(dict(
            i=i, m=m, window_hist=hist,
            window_median=rnd(float(np.median(many[:, wins[0][0], wins[0][2], :].mean(axis=1)))
                              if wins else 0, 4),
            window_mean=rnd(float(np.mean(many[:, wins[0][0], wins[0][2], :].mean(axis=1)))
                            if wins else 0, 4),
            median_hf=rnd(high_freq(med), 5), mean_hf=rnd(high_freq(avg), 5),
            sample_hf=rnd(float(np.mean([high_freq(many[k]) for k in range(m)])), 5),
            median_commit=rnd(window_stats(tp, i, med) or 0, 4),
            sample_commit=rnd(window_stats(tp, i, many[0]) or 0, 4),
            l1_of_median=rnd(float(np.abs(many - med).mean()), 5),
            l1_of_sample=rnd(float(np.abs(many - many[0]).mean()), 5),
            median=np.round(med, 4), sample=np.round(many[0], 4), mean=np.round(avg, 4),
        ))
    # a control that the median really is the L1 optimum on this data
    worst = max(c["l1_of_median"] - min(c["l1_of_median"], c["l1_of_sample"]) for c in cond)
    assert all(c["l1_of_median"] <= c["l1_of_sample"] for c in cond), \
        "the pixelwise median is not the L1 optimum, which would mean the sampler is wrong"
    return dict(plans=plans, A=A, B=B, test=dict(plans=tp, A=tA, B=tB), cond=cond,
                n=n, n_test=n_te, worst=worst,
                real_hf=rnd(high_freq(B[:200]), 5))


# ----------------------------------------------------------------------------
# 1. pix2pix
# ----------------------------------------------------------------------------
def build_unet(seed, ch=(16, 32, 48)):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2, c3 = ch
            self.e1 = nn.Sequential(nn.Conv2d(3, c1, 4, 2, 1), nn.LeakyReLU(0.2, True))
            self.e2 = nn.Sequential(nn.Conv2d(c1, c2, 4, 2, 1), nn.GroupNorm(4, c2),
                                    nn.LeakyReLU(0.2, True))
            self.e3 = nn.Sequential(nn.Conv2d(c2, c3, 4, 2, 1), nn.GroupNorm(4, c3),
                                    nn.ReLU(True))
            self.d3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c3, c2, 3, 1, 1),
                                    nn.GroupNorm(4, c2), nn.ReLU(True))
            self.d2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c2 * 2, c1, 3, 1, 1),
                                    nn.GroupNorm(4, c1), nn.ReLU(True))
            self.d1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c1 * 2, 3, 3, 1, 1))

        def forward(self, x):
            h1 = self.e1(x)
            h2 = self.e2(h1)
            h3 = self.e3(h2)
            u3 = self.d3(h3)
            u2 = self.d2(torch.cat([u3, h2], 1))
            return torch.sigmoid(self.d1(torch.cat([u2, h1], 1)))

    return UNet()


# Four discriminators that differ only in how much of the image one of their
# outputs can see. The sizes are given as (kernel, stride) per layer and the
# resulting window is MEASURED by autodiff below rather than derived, because
# the derivation is exactly the sort of arithmetic that is wrong in half the
# blog posts about this paper.
PATCH_SPECS = {
    "pixel": [(1, 1), (1, 1), (1, 1)],
    "small": [(3, 1), (3, 1), (3, 1)],
    "wide": [(4, 2), (3, 1), (4, 1), (3, 1)],
    "full": [(4, 2), (4, 2), (4, 1), (3, 1)],
}
ARM_PATCH = {"l1": "small", "gan": "small", "l1gan": "small",
             "patch-pixel": "pixel", "patch-full": "full"}


def build_patch_disc(seed, kind, norm=False):
    """No normalisation by default, and that is not a detail.

    A group or instance normalisation inside the discriminator pools its
    statistics over the whole image, so every output of a normalised patch
    discriminator depends on every input pixel and the window it is named after
    is only the window of its convolutions. Measured below, both ways. These
    are small enough not to need the normalisation, so the ladder of patch
    sizes is a ladder of patch sizes.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed + 100)
    spec = PATCH_SPECS[kind]
    layers = []
    c = 6
    widths = [24, 48, 48, 48]
    for i, (k, s) in enumerate(spec):
        last = i == len(spec) - 1
        out = 1 if last else widths[i]
        layers.append(nn.Conv2d(c, out, k, s, k // 2))
        if not last:
            if norm:
                layers.append(nn.GroupNorm(4, out))
            layers.append(nn.LeakyReLU(0.2, True))
        c = out
    return nn.Sequential(*layers)


def pix2pix_run(arm, seed, scenes, steps, batch, lam=100.0):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    A = torch.from_numpy(scenes["A"].transpose(0, 3, 1, 2).copy())
    B = torch.from_numpy(scenes["B"].transpose(0, 3, 1, 2).copy())
    G = build_unet(seed)
    kind = ARM_PATCH[arm]
    D = build_patch_disc(seed, kind)
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    use_gan = arm != "l1"
    use_l1 = arm != "gan"
    for it in range(steps):
        idx = torch.randint(0, A.shape[0], (batch,))
        a, b = A[idx], B[idx]
        fake = G(a)
        if use_gan:
            dr = D(torch.cat([a, b], 1))
            df = D(torch.cat([a, fake.detach()], 1))
            ld = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
            od.zero_grad(); ld.backward(); od.step()
        lg = 0.0
        if use_gan:
            out = D(torch.cat([a, fake], 1))
            lg = bce(out, torch.ones_like(out))
        if use_l1:
            lg = lg + lam * (fake - b).abs().mean()
        og.zero_grad(); lg.backward(); og.step()
        if (it + 1) % max(1, steps // 6) == 0:
            print(f"      {arm}/{seed} step {it+1}: loss {float(lg):.4f}", flush=True)
    G.eval()
    return dict(state={k: v.detach().numpy() for k, v in G.state_dict().items()},
                dstate={k: v.detach().numpy() for k, v in D.state_dict().items()},
                patch=kind,
                params=sum(p.numel() for p in G.parameters()),
                dparams=sum(p.numel() for p in D.parameters()))


def translate(state, seed, A, batch=256):
    import torch

    G = build_unet(seed)
    G.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    G.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(A), batch):
            t = torch.from_numpy(A[i:i + batch].transpose(0, 3, 1, 2).copy())
            out.append(G(t).numpy().transpose(0, 2, 3, 1))
    return np.concatenate(out)


def receptive_field(dstate, seed, kind, norm=False):
    """How wide the discriminator's window really is: put a gradient on one
    output and see how many input pixels it reaches. This is the number the
    patch size is supposed to be, and it is cheaper to measure than to derive."""
    import torch

    D = build_patch_disc(seed, kind, norm=norm)
    if dstate is not None:
        D.load_state_dict({k: torch.from_numpy(v) for k, v in dstate.items()})
    x = torch.zeros(1, 6, RES, RES, requires_grad=True)
    out = D(x)
    cy, cx = out.shape[2] // 2, out.shape[3] // 2
    g = torch.autograd.grad(out[0, 0, cy, cx], x)[0][0].abs().sum(0).numpy()
    ys, xs = np.nonzero(g > 0)
    if not len(ys):
        return dict(field=0, outputs=list(out.shape[2:]))
    return dict(field=int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1),
                outputs=list(out.shape[2:]))


# ----------------------------------------------------------------------------
# 3. CycleGAN, with the pairing thrown away
# ----------------------------------------------------------------------------
def cycle_run(seed, scenes, steps, batch, lam=10.0):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    A = torch.from_numpy(scenes["A"].transpose(0, 3, 1, 2).copy())
    B = torch.from_numpy(scenes["B"].transpose(0, 3, 1, 2).copy())
    half = A.shape[0] // 2
    # The two halves come from different scenes, so no plan in the A pile has
    # its render in the B pile. Shuffling indices would not be enough: the
    # pairing has to be genuinely absent, not merely unused.
    A = A[:half]
    B = B[half:]
    Gab = build_unet(seed)
    Gba = build_unet(seed + 7)
    Da = build_patch_disc(seed + 1, "wide")
    Db = build_patch_disc(seed + 2, "wide")
    og = torch.optim.Adam(list(Gab.parameters()) + list(Gba.parameters()), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(list(Da.parameters()) + list(Db.parameters()), 2e-4, betas=(0.5, 0.999))

    def d_in(x):
        return torch.cat([x, x], 1)          # the discriminators take six channels

    bce = nn.BCEWithLogitsLoss()
    for it in range(steps):
        a = A[torch.randint(0, A.shape[0], (batch,))]
        b = B[torch.randint(0, B.shape[0], (batch,))]
        fb = Gab(a)
        fa = Gba(b)
        dr = Db(d_in(b))
        df = Db(d_in(fb.detach()))
        ld = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
        dr2 = Da(d_in(a))
        df2 = Da(d_in(fa.detach()))
        ld = ld + bce(dr2, torch.ones_like(dr2)) + bce(df2, torch.zeros_like(df2))
        od.zero_grad(); ld.backward(); od.step()

        ob = Db(d_in(fb))
        oa = Da(d_in(fa))
        lg = bce(ob, torch.ones_like(ob)) + bce(oa, torch.ones_like(oa))
        cyc = (Gba(fb) - a).abs().mean() + (Gab(fa) - b).abs().mean()
        lg = lg + lam * cyc
        og.zero_grad(); lg.backward(); og.step()
        if (it + 1) % max(1, steps // 5) == 0:
            print(f"      cycle/{seed} step {it+1}: adv {float(lg - lam*cyc):.3f} "
                  f"cyc {float(cyc):.4f}", flush=True)
    Gab.eval()
    Gba.eval()
    return dict(ab={k: v.detach().numpy() for k, v in Gab.state_dict().items()},
                ba={k: v.detach().numpy() for k, v in Gba.state_dict().items()},
                params=sum(p.numel() for p in Gab.parameters()) * 2)


def cycle_scores(run, seed, scenes):
    """Did it find the correspondence, and what is the cycle made of."""
    import torch

    tp = scenes["test"]["plans"]
    tA = scenes["test"]["A"]
    tB = scenes["test"]["B"]
    fake_b = translate(run["ab"], seed, tA)
    got = np.array([read_type(tp, i, fake_b[i]) for i in range(len(tA))])
    truth = np.asarray(tp["btype"])
    agree = float((got == truth).mean())

    Gba = build_unet(seed + 7)
    Gba.load_state_dict({k: torch.from_numpy(v) for k, v in run["ba"].items()})
    Gba.eval()
    rng = np.random.default_rng(77)

    def back(x):
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
            return Gba(t).numpy().transpose(0, 2, 3, 1)

    rec = back(fake_b)
    base = float(np.abs(rec - tA).mean())
    rows = []
    for sigma in (0.005, 0.02, 0.05):
        noise = rng.normal(0, sigma, fake_b.shape).astype(np.float32)
        pert = np.clip(fake_b + noise, 0, 1)
        rec_p = back(pert)
        real_p = np.clip(tB + noise, 0, 1)
        moved_fake = float(np.abs(rec_p - rec).mean())
        moved_real = float(np.abs(back(real_p) - back(tB)).mean())
        rows.append(dict(sigma=sigma, cycle=rnd(float(np.abs(rec_p - tA).mean()), 5),
                         moved_fake=rnd(moved_fake, 5), moved_real=rnd(moved_real, 5),
                         ratio=rnd(moved_fake / max(moved_real, 1e-9), 3)))
    quant = []
    for bits in (8, 6, 4):
        q = np.round(fake_b * (2 ** bits - 1)) / (2 ** bits - 1)
        quant.append(dict(bits=bits, cycle=rnd(float(np.abs(back(q) - tA).mean()), 5)))
    return dict(agree=rnd(agree, 4), base_cycle=rnd(base, 5), noise=rows, quant=quant,
                fake=np.round(fake_b[:8], 4), rec=np.round(rec[:8], 4))


def main():
    print(f"threads: {THREADS}{'  (SMALL smoke run)' if SMALL else ''}", flush=True)
    print("0. the scenes", flush=True)
    scenes = cached(f"scenes-{RES}-{SMALL}-v3", stage_scenes)
    c = scenes["cond"][0]
    print(f"    a plan rendered {c['m']} times: sample detail {c['sample_hf']:.4f}, "
          f"median {c['median_hf']:.4f}, commitment {c['sample_commit']:.3f} against "
          f"{c['median_commit']:.3f}", flush=True)

    print("1. pix2pix", flush=True)
    steps = 30 if SMALL else 2200
    batch = 32
    arms = ["l1", "gan", "l1gan", "patch-pixel", "patch-full"]
    seeds = [1] if SMALL else [1, 2]
    runs = {}
    for arm in arms:
        for sd in seeds:
            key = f"p2p-{arm}-s{sd}-st{steps}-b{batch}-v2"
            runs[f"{arm}/{sd}"] = cached(key, lambda a=arm, s=sd: pix2pix_run(a, s, scenes,
                                                                             steps, batch))

    print("2. what each arm produces", flush=True)
    tp = scenes["test"]["plans"]
    tA = scenes["test"]["A"]
    tB = scenes["test"]["B"]
    # what a normalisation layer does to the window the patch is named after
    norm_field = {k: receptive_field(None, 1, k, norm=True)["field"] for k in PATCH_SPECS}
    plain_field = {k: receptive_field(None, 1, k, norm=False)["field"] for k in PATCH_SPECS}
    print(f"    patch windows without normalisation {plain_field}, with it {norm_field}",
          flush=True)
    real_hf = high_freq(tB)
    # a plan whose building is too small to hold a window has no windows, and
    # averaging a zero in for it would report the metric as lower than it is
    with_windows = [i for i in range(48) if _windows(tp, i)][:24]
    commit_of = lambda imgs: float(np.mean([window_stats(tp, i, imgs[i]) for i in with_windows]))
    real_commit = commit_of(tB)
    scores = {}
    for key, r in runs.items():
        arm, sd = key.split("/")
        out = translate(r["state"], int(sd), tA)
        commit = commit_of(out)
        field = receptive_field(r["dstate"], int(sd), r["patch"])
        scores[key] = dict(l1=rnd(float(np.abs(out - tB).mean()), 5),
                           hf=rnd(high_freq(out), 5),
                           hf_share=rnd(high_freq(out) / real_hf, 4),
                           commit=rnd(commit, 4), commit_share=rnd(commit / real_commit, 4),
                           field=field["field"], outputs=field["outputs"],
                           patch=r["patch"],
                           params=r["params"], dparams=r["dparams"])
        print(f"    {key}: l1 {scores[key]['l1']:.4f}, detail {scores[key]['hf_share']*100:.1f}% "
              f"of real, windows {scores[key]['commit_share']*100:.1f}%, "
              f"patch {field['field']}px", flush=True)

    print("3. CycleGAN", flush=True)
    csteps = 30 if SMALL else 2500
    cseeds = [1] if SMALL else [1, 2, 3]
    cyc = {}
    for sd in cseeds:
        key = f"cyc-s{sd}-st{csteps}-b16-v2"
        run = cached(key, lambda s=sd: cycle_run(s, scenes, csteps, 16))
        cyc[sd] = cached(f"{key}-scores", lambda r=run, s=sd: cycle_scores(r, s, scenes))
        print(f"    seed {sd}: correspondence right {cyc[sd]['agree']*100:.1f}% of the time, "
              f"cycle error {cyc[sd]['base_cycle']:.4f}", flush=True)

    # ------------------------------------------------------------------ write
    print("writing", flush=True)
    tiles = []
    order = []
    n_show = 6
    for i in range(n_show):
        tiles.append(tA[i])
    order.append("plan")
    for i in range(n_show):
        tiles.append(tB[i])
    order.append("render")
    for arm in arms:
        out = translate(runs[f"{arm}/{seeds[0]}"]["state"], seeds[0], tA[:n_show])
        tiles.extend(out)
        order.append(arm)
    tiles.extend([scenes["cond"][k]["median"] for k in range(4)])
    tiles.extend([scenes["cond"][k]["sample"] for k in range(min(2, n_show))])
    order.append("conditional")
    best = cyc[cseeds[0]]
    tiles.extend(best["fake"][:n_show])
    order.append("cycle")
    png = dict(strip=save_png(sheet(tiles, n_show), IMG / "strip.png"))

    state = runs[f"l1gan/{seeds[0]}"]["state"]
    state_l1 = runs[f"l1/{seeds[0]}"]["state"]
    nets = {}
    for name, st in (("l1gan", state), ("l1", state_l1)):
        keys = list(st)
        vals = np.concatenate([st[k].ravel().astype(np.float32) for k in keys])
        layers = {}
        off = 0
        for k in keys:
            layers[k] = dict(offset=off, shape=list(st[k].shape))
            off += st[k].size
        nets[name] = dict(weights_b64=base64.b64encode(vals.astype(np.float16).tobytes())
                          .decode("ascii"), count=int(vals.size), layers=layers)

    # a reference the browser has to reproduce, on the quantised weights
    import torch
    checks = {}
    for name, st in (("l1gan", state), ("l1", state_l1)):
        st16 = {k: v.astype(np.float16).astype(np.float32) for k, v in st.items()}
        # The browser reads its plans out of a PNG, so the reference is
        # computed on the same 8 bit values rather than on the floats this
        # file has in memory: otherwise the guard measures the quantisation of
        # the input and reports it as an implementation difference.
        plan8 = (np.round(tA[:2] * 255) / 255).astype(np.float32)
        out16 = translate(st16, seeds[0], plan8)
        idx = np.random.default_rng(3).permutation(RES * RES)[:96]
        checks[name] = dict(idx=idx.tolist(),
                            rgb=[[[rnd(out16[i].reshape(-1, 3)[p][c], 6) for c in range(3)]
                                  for p in idx] for i in range(2)])

    out = dict(
        meta=dict(res=RES, threads=THREADS, steps=steps, batch=batch, arms=arms, seeds=seeds,
                  n_train=scenes["n"], n_test=scenes["n_test"], lam=100.0,
                  cycle_steps=csteps, cycle_seeds=cseeds, cycle_lam=10.0,
                  tile=RES, sheet_cols=n_show, sheet_order=order, png_bytes=png,
                  plan_colours=dict(sky=list(SKY), ground=list(GROUND), road=list(ROAD),
                                    body=[list(c) for c in PLAN_BODY]),
                  render_colours=dict(body=[list(c) for c in BODY], lit=list(LIT),
                                      dark=list(DARK), lit_p=LIT_P)),
        conditional=[{k: v for k, v in c.items()
                      if k not in ("median", "sample", "mean")} for c in scenes["cond"]],
        real=dict(hf=rnd(real_hf, 5), commit=rnd(real_commit, 4)),
        fields=dict(plain=plain_field, normalised=norm_field, specs=PATCH_SPECS),
        scores=scores,
        cycle={str(k): {kk: vv for kk, vv in v.items() if kk not in ("fake", "rec")}
               for k, v in cyc.items()},
        plans=dict(test=[{k: int(np.asarray(tp[k])[i]) for k in tp} for i in range(n_show)]),
        nets=nets, check=checks,
    )
    (OUT / "translation.json").write_text(json.dumps(out), "utf-8")
    print(f"wrote {OUT}/translation.json ({len(json.dumps(out))/1024:.0f} kB) and the strip",
          flush=True)


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
        r, cc = divmod(k, cols)
        out[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t[k]
    return out


if __name__ == "__main__":
    main()
