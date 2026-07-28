"""Data for the ATLAS Glow article: how many bits a picture costs, and which
pictures a likelihood turns out to prefer.

A flow on two dimensions is a demonstration. On images it becomes a claim about
compression, because an exact log likelihood in nats is a number of bits per
pixel and there are real compressors to put beside it. So this file measures:

A. The three pieces, ablated one at a time on the same data with the same
   budget: the invertible 1x1 convolution against a fixed permutation and
   against plain channel reversal, actnorm against nothing, affine coupling
   against additive. Two seeds, so every verdict has a floor under it.
B. The dequantisation trap, which is the lesson nobody prints. Pixels are
   discrete. Fit a continuous density straight to them and the likelihood is
   unbounded: the model puts spikes on the atoms and reports bits per dimension
   below zero and falling. Uniform dequantisation fixes it, and the resulting
   number is a genuine upper bound on the code length of the discrete image.
C. Against real compressors. gzip and PNG on the same bytes, the entropy of the
   pixel histogram, and an independent per pixel model, so the reader can see
   where the flow sits between "no model at all" and "a program someone ships".
D. Which dataset the likelihood prefers. The same architecture trained on
   digits and on Fashion-MNIST, then both models evaluated on both held out
   sets: the full two by two table, plus the compressibility of each image,
   which is the competing explanation and is testable.
E. Temperature. Sampling with the latent scaled, scored by the module's shared
   judge and by the same two sample statistic the variational article used.
F. The model the browser runs, in float16, with the round trip a flow can do
   and nothing else in this module can.
G. The module's scoreboard, read off the other four articles' own files.

Run from anywhere: `python src/utils/generate_glow_data.py`.
"""
import gzip
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.generative_common import (          # noqa: E402
    DIGITS, ROOT as REPO, SEED, arr, banner, cached, digits14, export_flat,
    fashion14, judge, mat, rnd, sci, threads, write_json)

OUT = REPO / "glow" / "data"
THREADS = threads()

# ---------------------------------------------------------------- configuration
SIDE = DIGITS["side"]          # 14
DIM = SIDE * SIDE              # 196 dimensions, so bits per dimension is per pixel
K = 6                          # flow steps
HID = 28                       # width inside a coupling
EPOCHS = 30
BATCH = 64
LR = 1e-3
SEEDS = [19, 23]
N_FASHION = 4000
TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LEVELS = 256                   # the images are 8 bit before anything happens

banner("ATLAS / Glow: how many bits a picture costs")
print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED}")


def npy(t):
    return t.detach().cpu().numpy().astype(np.float64)


# ==================================================================== the data
def as_bytes(X):
    """The images as 8 bit integers, which is what a compressor gets and what
    the bits per dimension of a discrete model is about."""
    return np.clip(np.round(np.asarray(X, dtype=float) * (LEVELS - 1)), 0, LEVELS - 1
                   ).astype(np.uint8)


Xd, yd, TR, TE = digits14()
DIG_TR, DIG_TE = as_bytes(Xd[TR]), as_bytes(Xd[TE])
Xf, yf = fashion14(N_FASHION)
n_ftr = int(len(Xf) * 0.75)
FAS_TR, FAS_TE = as_bytes(Xf[:n_ftr]), as_bytes(Xf[n_ftr:])
print(f"  {len(DIG_TR)} digits to fit on and {len(DIG_TE)} held out, "
      f"{len(FAS_TR)} Fashion images to fit on and {len(FAS_TE)} held out, "
      f"{SIDE} by {SIDE} and 8 bits each")
JUDGE = judge()
print(f"  the module's shared judge: {JUDGE['accuracy'] * 100:.2f}% on the held out digits "
      f"and {JUDGE['real_confidence']:.4f} confident on them")


def dequantise(U, seed, on=True):
    """u + uniform noise, divided by the number of levels.

    With the noise, the continuous density's expected log likelihood is a LOWER
    bound on the discrete model's, so the bits per dimension it reports is an
    honest upper bound on a code length. Without it, the target has atoms and
    nothing stops a density from putting a spike on each one, which is block B.
    """
    rs = np.random.RandomState(seed)
    U = np.asarray(U, dtype=np.float64)
    return ((U + (rs.rand(*U.shape) if on else 0.0)) / LEVELS).astype(np.float32)


# =============================================================== the architecture
class ActNorm(nn.Module):
    """A per channel scale and shift, initialised from the first batch it sees
    so that the activations start standardised. Its log determinant is the sum
    of the log scales, once per spatial position."""

    def __init__(self, c):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.register_buffer("ready", torch.zeros(1))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x, torch.zeros(len(x))
        if self.ready.item() == 0 and self.training:
            with torch.no_grad():
                m = x.mean((0, 2, 3), keepdim=True)
                s = x.std((0, 2, 3), keepdim=True) + 1e-6
                self.b.copy_(-m / s)
                self.log_s.copy_(-torch.log(s))
                self.ready.fill_(1)
        y = x * torch.exp(self.log_s) + self.b
        ld = self.log_s.sum() * x.shape[2] * x.shape[3]
        return y, ld.expand(len(x))

    def inverse(self, y):
        if not self.enabled:
            return y
        return (y - self.b) * torch.exp(-self.log_s)


class Invertible1x1(nn.Module):
    """The piece Glow adds. A learned mixing of the channels, which generalises
    the fixed permutation earlier flows used: with four channels its
    determinant is a four by four determinant and costs nothing."""

    def __init__(self, c, seed=0, mode="learned"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(c, c, generator=g))
        self.mode = mode
        if mode == "learned":
            self.W = nn.Parameter(q)
        elif mode == "permute":
            perm = torch.randperm(c, generator=g)
            self.register_buffer("W", torch.eye(c)[perm])
        else:                                   # plain reversal
            self.register_buffer("W", torch.eye(c).flip(0))

    def forward(self, x):
        W = self.W
        y = F.conv2d(x, W.unsqueeze(-1).unsqueeze(-1))
        ld = torch.slogdet(W)[1] * x.shape[2] * x.shape[3]
        return y, ld.expand(len(x))

    def inverse(self, y):
        return F.conv2d(y, torch.inverse(self.W).unsqueeze(-1).unsqueeze(-1))


class Coupling(nn.Module):
    def __init__(self, c, hid=HID, additive=False):
        super().__init__()
        self.half = c // 2
        self.additive = additive
        self.net = nn.Sequential(
            nn.Conv2d(self.half, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 1), nn.ReLU(),
            nn.Conv2d(hid, c - self.half if additive else 2 * (c - self.half), 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, xa):
        out = self.net(xa)
        if self.additive:
            return torch.zeros_like(out), out
        s, t = out.chunk(2, 1)
        return torch.tanh(s) * 2.0, t

    def forward(self, x):
        xa, xb = x[:, :self.half], x[:, self.half:]
        s, t = self.st(xa)
        yb = xb * torch.exp(s) + t
        return torch.cat([xa, yb], 1), s.flatten(1).sum(1)

    def inverse(self, y):
        ya, yb = y[:, :self.half], y[:, self.half:]
        s, t = self.st(ya)
        return torch.cat([ya, (yb - t) * torch.exp(-s)], 1)


def squeeze(x):
    n, c, h, w = x.shape
    return (x.view(n, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
            .reshape(n, c * 4, h // 2, w // 2))


def unsqueeze(x):
    n, c, h, w = x.shape
    return (x.view(n, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
            .reshape(n, c // 4, h * 2, w * 2))


class Glow(nn.Module):
    """Squeeze once, then K steps of actnorm, mixing and coupling.

    One level and no split, which is smaller than the paper's model and is said
    so on the page: the browser has to run this, and every ablation below is
    against the same shape, so the comparisons are fair even though the
    absolute bits per dimension is not a record.
    """

    def __init__(self, seed=SEED, mix="learned", actnorm=True, additive=False,
                 k=K, hid=HID):
        super().__init__()
        torch.manual_seed(seed)
        c = 4
        self.k = k
        self.act = nn.ModuleList([ActNorm(c) for _ in range(k)])
        self.mix = nn.ModuleList([Invertible1x1(c, seed=seed + i, mode=mix)
                                  for i in range(k)])
        self.cpl = nn.ModuleList([Coupling(c, hid=hid, additive=additive) for _ in range(k)])
        for a in self.act:
            a.enabled = actnorm

    def forward(self, x):
        x = squeeze(x)
        ld = torch.zeros(len(x))
        for i in range(self.k):
            x, l1 = self.act[i](x)
            x, l2 = self.mix[i](x)
            x, l3 = self.cpl[i](x)
            ld = ld + l1 + l2 + l3
        return x, ld

    def inverse(self, z):
        for i in reversed(range(self.k)):
            z = self.cpl[i].inverse(z)
            z = self.mix[i].inverse(z)
            z = self.act[i].inverse(z)
        return unsqueeze(z)

    def log_prob(self, x):
        z, ld = self(x)
        lp = (-0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)).flatten(1).sum(1)
        return lp + ld


def to_t(U, seed, deq=True):
    x = dequantise(U, seed, on=deq)
    return torch.tensor(x, dtype=torch.float32).reshape(-1, 1, SIDE, SIDE)


def bits_per_dim(logp, deq=True):
    """A continuous log density becomes a code length for the discrete image:
    subtract the log of the number of levels per dimension and convert."""
    base = np.log2(LEVELS) if deq else 0.0
    return float(-np.mean(logp) / (DIM * np.log(2.0)) + base)


def train_glow(U, seed, mix="learned", actnorm=True, additive=False, deq=True,
               epochs=EPOCHS, tag="digits"):
    def build():
        net = Glow(seed=seed, mix=mix, actnorm=actnorm, additive=additive)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        g = torch.Generator().manual_seed(seed)
        net.train()
        for e in range(epochs):
            X = to_t(U, seed=1000 * seed + e, deq=deq)       # fresh noise each epoch
            order = torch.randperm(len(X), generator=g)
            for s in range(0, len(X), BATCH):
                loss = -net.log_prob(X[order[s:s + BATCH]]).mean() / DIM
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 50.0)
                opt.step()
        net.eval()
        return {k_: v.detach().clone() for k_, v in net.state_dict().items()}

    key = (f"glow-{tag}-s{seed}-mix{mix}-an{int(actnorm)}-add{int(additive)}"
           f"-deq{int(deq)}-k{K}-h{HID}-e{epochs}-b{BATCH}-lr{LR}-n{len(U)}-v1")
    st = cached(key, build)
    net = Glow(seed=seed, mix=mix, actnorm=actnorm, additive=additive)
    net.load_state_dict(st)
    net.eval()
    return net


def evaluate(net, U, seed=7, deq=True, chunk=500):
    X = to_t(U, seed=seed, deq=deq)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            out.append(npy(net.log_prob(X[s:s + chunk])))
    lp = np.concatenate(out)
    return bits_per_dim(lp, deq=deq), lp


def spread(vals):
    v = [float(x) for x in vals]
    return {"mean": rnd(np.mean(v), 4), "min": rnd(min(v), 4), "max": rnd(max(v), 4),
            "spread": rnd(max(v) - min(v), 4), "seeds": len(v)}


# ========================================================== A. the three pieces
banner("A. the three pieces, one at a time, same data and same budget")
# Two names per arm: the sentence goes on the chart, where there is room for it,
# and the short one goes in the readout table, which has five columns to fit
# inside the article's width and cannot afford a sentence in the first of them.
ARMS = [
    ("full", {}, "actnorm, a learned 1x1 mixing, affine coupling", "the full model"),
    ("permute", {"mix": "permute"},
     "the mixing replaced by a fixed random permutation", "fixed permutation"),
    ("reverse", {"mix": "reverse"},
     "the mixing replaced by reversing the channels", "channels reversed"),
    ("no_actnorm", {"actnorm": False}, "actnorm removed", "no actnorm"),
    ("additive", {"additive": True},
     "additive coupling, so every step preserves volume", "additive coupling"),
]
arm_rows = []
for name, kw, label, short in ARMS:
    vals = []
    for s in SEEDS:
        net = train_glow(DIG_TR, s, tag="digits", **kw)
        bpd, _ = evaluate(net, DIG_TE)
        vals.append(bpd)
    sp = spread(vals)
    arm_rows.append({"name": name, "label": label, "short": short, **sp,
                     "params": sum(p.numel() for p in train_glow(
                         DIG_TR, SEEDS[0], tag="digits", **kw).parameters())})
    print(f"  {label:<56}: {sp['mean']:.4f} bits per dimension "
          f"(seeds {sp['min']:.4f} to {sp['max']:.4f})")
# The floor a comparison is read against belongs to the TWO cells being
# compared, not to the widest range anywhere in the table. Here one arm
# (permute) is far noisier than the rest, and a single worst case floor lets it
# set the bar for every other verdict: the additive arm, which is the one real
# difference in this table, was being scored against a floor almost ten times
# its own. The worst case number is kept as well, because it is the right one
# for a claim about the table as a whole.
FLOOR_WORST = max(r["spread"] for r in arm_rows)
full = next(r for r in arm_rows if r["name"] == "full")
for r in arm_rows[1:]:
    d = r["mean"] - full["mean"]
    floor = max(r["spread"], full["spread"])
    r["gap"] = rnd(d, 4)
    r["floor"] = rnd(floor, 4)
    r["resolves"] = bool(abs(d) > floor)
    r["times_floor"] = rnd(abs(d) / floor, 2) if floor > 0 else None
    print(f"    {r['name']:<12} costs {d:+.4f} bits against the full model, against the floor of "
          f"that pair ({floor:.4f}): "
          f"{'bigger than it' if abs(d) > floor else 'inside it, so this page does not call it a difference'}")
FLOOR = FLOOR_WORST

# ================================================== B. the dequantisation trap
banner("B. what happens when a continuous density is fitted to discrete pixels")
raw_rows = []
for ep in (2, 6, 12, 20, EPOCHS):
    net = train_glow(DIG_TR, SEED, deq=False, epochs=ep, tag="digits-raw")
    bpd_raw, lp_raw = evaluate(net, DIG_TE, deq=False)
    raw_rows.append({"epochs": ep, "bpd": rnd(bpd_raw, 4),
                     "logp": rnd(float(lp_raw.mean()), 2)})
    print(f"  no dequantisation, {ep:>2} epochs: {bpd_raw:+.4f} bits per dimension "
          f"(mean log density {lp_raw.mean():+.1f} nats)")
deq_net = train_glow(DIG_TR, SEED, tag="digits")
bpd_deq, lp_deq = evaluate(deq_net, DIG_TE)
trap = {
    "rows": raw_rows, "honest": rnd(bpd_deq, 4),
    "worst": rnd(min(r["bpd"] for r in raw_rows), 4),
    "note": "with uniform dequantisation the continuous model's expected log density is a "
            "lower bound on the discrete model's log likelihood, so the bits per dimension "
            "above is an upper bound on a real code length; without it there is no bound at "
            "all and nothing stops the density diverging on the atoms",
}
print(f"  the same architecture with uniform dequantisation reports {bpd_deq:.4f}, and that "
      f"number is a bound on a code length; the one above is not a number at all")

# ==================================================== C. against real compressors
banner("C. against compressors people actually ship")


def png_bytes(img):
    """A real PNG of a 14 by 14 greyscale image, written here rather than
    imported, so the byte count is the file a viewer would open."""
    raw = b"".join(b"\x00" + row.tobytes() for row in img)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    head = struct.pack(">IIBBBBB", img.shape[1], img.shape[0], 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


gz = np.mean([len(gzip.compress(im.tobytes(), 9)) * 8 / DIM for im in DIG_TE])
pn = np.mean([len(png_bytes(im.reshape(SIDE, SIDE))) * 8 / DIM for im in DIG_TE])
hist = np.bincount(DIG_TR.ravel(), minlength=LEVELS).astype(float)
p_hist = hist / hist.sum()
ent_marginal = float(-(p_hist[p_hist > 0] * np.log2(p_hist[p_hist > 0])).sum())
per_pixel = np.zeros(DIM)
for j in range(DIM):
    c = np.bincount(DIG_TR[:, j], minlength=LEVELS).astype(float) + 1.0
    q = c / c.sum()
    per_pixel[j] = -np.mean(np.log2(q[DIG_TE[:, j]]))
compress = {
    "flow": rnd(bpd_deq, 4),
    "gzip": rnd(float(gz), 4), "png": rnd(float(pn), 4),
    "histogram": rnd(ent_marginal, 4),
    "per_pixel": rnd(float(per_pixel.mean()), 4),
    "raw": 8.0,
}
print(f"  the flow {compress['flow']:.4f}, an independent model per pixel "
      f"{compress['per_pixel']:.4f}, the pixel histogram alone {compress['histogram']:.4f}, "
      f"gzip {compress['gzip']:.4f}, PNG {compress['png']:.4f}, and the raw file 8.0000")

# ============================================= D. which dataset it prefers
banner("D. the same architecture on two datasets, evaluated on both")
cross = {}
for tag, U_tr in (("digits", DIG_TR), ("fashion", FAS_TR)):
    for s in SEEDS:
        net = train_glow(U_tr, s, tag=tag, epochs=EPOCHS)
        for on, U_te in (("digits", DIG_TE), ("fashion", FAS_TE)):
            b, lp = evaluate(net, U_te)
            cross.setdefault(f"{tag}->{on}", []).append(b)
            if tag == "fashion" and s == SEEDS[0]:
                cross.setdefault("_lp_fashion_model", {})[on] = lp
table = {k: spread(v) for k, v in cross.items() if not k.startswith("_")}
for k, v in table.items():
    print(f"  trained on {k.split('->')[0]:<8} evaluated on {k.split('->')[1]:<8}: "
          f"{v['mean']:.4f} bits per dimension (seeds {v['min']:.4f} to {v['max']:.4f})")
ood_gap = table["fashion->fashion"]["mean"] - table["fashion->digits"]["mean"]
rev_gap = table["digits->digits"]["mean"] - table["digits->fashion"]["mean"]
# The floor of a comparison is the spread of the two cells being compared, not
# the widest spread anywhere in the table: the digit model's numbers on Fashion
# wobble by 0.79 between seeds, and charging that wobble to a comparison between
# two other cells would hide an effect eighty times its own noise.
ood_floor = max(table["fashion->fashion"]["spread"], table["fashion->digits"]["spread"])
rev_floor = max(table["digits->digits"]["spread"], table["digits->fashion"]["spread"])
ood_floor_worst = max(v["spread"] for v in table.values())
# A direction is a figure and gets its own branch: printing a signed number
# next to the word "more" is how a page ends up saying the opposite of what it
# measured.
if ood_gap > 0:
    print(f"  the Fashion model spends {abs(ood_gap):.4f} bits per dimension FEWER on the "
          f"digits than on its own held out images, which is the anomaly, against a seed "
          f"floor of {ood_floor:.4f}")
else:
    print(f"  the Fashion model spends {abs(ood_gap):.4f} bits per dimension MORE on the "
          f"digits than on its own held out images, so the anomaly does not reproduce here, "
          f"against a seed floor of {ood_floor:.4f}")
if rev_gap > 0:
    print(f"  the other way round, the digit model spends {abs(rev_gap):.4f} FEWER on Fashion "
          f"than on digits")
else:
    print(f"  the other way round, the digit model spends {abs(rev_gap):.4f} MORE on Fashion "
          f"than on digits")
print(f"  so the two directions {'disagree, which is the asymmetry the literature reports' if ood_gap * rev_gap < 0 else 'agree, so whatever this is, it is not one sided'}")

# the competing explanation: the model prefers whatever compresses
gz_dig = np.array([len(gzip.compress(im.tobytes(), 9)) * 8 / DIM for im in DIG_TE])
gz_fas = np.array([len(gzip.compress(im.tobytes(), 9)) * 8 / DIM for im in FAS_TE])
lp_d = cross["_lp_fashion_model"]["digits"]
lp_f = cross["_lp_fashion_model"]["fashion"]
bpd_d = -lp_d / (DIM * np.log(2)) + 8.0
bpd_f = -lp_f / (DIM * np.log(2)) + 8.0
allb = np.r_[bpd_d, bpd_f]
allg = np.r_[gz_dig, gz_fas]
corr = float(np.corrcoef(allb, allg)[0, 1])
ood = {
    "table": table, "gap": rnd(ood_gap, 4), "reverse_gap": rnd(rev_gap, 4),
    "floor": rnd(ood_floor, 4), "reverse_floor": rnd(rev_floor, 4),
    "floor_worst_cell": rnd(ood_floor_worst, 4),
    "gzip_digits": rnd(float(gz_dig.mean()), 4), "gzip_fashion": rnd(float(gz_fas.mean()), 4),
    "correlation": rnd(corr, 4),
    "resolves": bool(abs(ood_gap) > ood_floor),
}
print(f"  the competing explanation is compressibility, and it is testable: gzip needs "
      f"{gz_dig.mean():.4f} bits per pixel for a digit and {gz_fas.mean():.4f} for a Fashion "
      f"image, and across all {len(allb)} held out images the model's own bits and gzip's "
      f"correlate at {corr:+.4f}")

# ================================================================ E. temperature
banner("E. sampling with the latent scaled")
show = train_glow(DIG_TR, SEED, tag="digits")
temp_rows = []
for t in TEMPS:
    g = torch.Generator().manual_seed(4242)
    with torch.no_grad():
        z = torch.randn(300, 4, SIDE // 2, SIDE // 2, generator=g) * t
        x = npy(show.inverse(z)).reshape(300, DIM)
    pix = np.clip(x, 0, 1)
    P = JUDGE["proba"](pix)
    q = P.mean(0)
    temp_rows.append({
        "t": t, "confidence": rnd(float(P.max(1).mean()), 4),
        "class_entropy": rnd(float(-(q * np.log(np.maximum(q, 1e-12))).sum()), 4),
        "top_class_share": rnd(float(q.max()), 4),
        "outside": rnd(float(((x < 0) | (x > 1)).mean()), 4),
        # the confound the energy article already measured on THIS judge
        "ink": rnd(float(pix.mean()), 4),
    })
    print(f"  temperature {t}: judge confidence {temp_rows[-1]['confidence']}, class entropy "
          f"{temp_rows[-1]['class_entropy']} (uniform over ten is {np.log(10):.4f}), "
          f"{temp_rows[-1]['outside'] * 100:.1f}% of pixels outside the range a picture has, "
          f"ink {temp_rows[-1]['ink']}")
best_t = max(temp_rows, key=lambda r: r["confidence"])
print(f"  the judge likes {best_t['t']} best at {best_t['confidence']}, against "
      f"{next(r for r in temp_rows if r['t'] == 1.0)['confidence']} at temperature one and "
      f"{JUDGE['real_confidence']:.4f} on real held out digits")

# Whether that ranking means anything at all. The energy article measured that
# this same shared judge scores ink rather than digits (its energy correlates
# -0.68 with ink, and uniform noise rescaled to a digit's ink jumps from 0.1396
# to 0.9153). A metric with a known confound has to be re-tested every time it
# is reused, so here is the same test on this axis: if confidence is monotone in
# ink across the sweep, and every temperature is fainter than a real digit, then
# the judge cannot separate "better digits" from "more ink" anywhere in the
# range, and the page must not read its ranking as an answer.
real_ink = float(DIG_TE.astype(np.float64).mean() / 255.0)
ink_corr = float(np.corrcoef([r["ink"] for r in temp_rows],
                             [r["confidence"] for r in temp_rows])[0, 1])
faintest = max(r["ink"] for r in temp_rows)
temp_confounded = bool(abs(ink_corr) > 0.9 and faintest < real_ink)
temperature = {
    "rows": temp_rows, "best": best_t["t"],
    "uniform_entropy": rnd(float(np.log(10)), 4),
    "real_ink": rnd(real_ink, 4), "ink_corr": rnd(ink_corr, 4),
    "all_fainter_than_real": bool(faintest < real_ink),
    "confounded": temp_confounded,
    "ink_ratio": rnd(faintest / real_ink, 3),
}
print(f"  ink runs {temp_rows[0]['ink']} to {temp_rows[-1]['ink']} against {real_ink:.4f} on "
      f"real held out digits, and correlates {ink_corr:+.4f} with the judge's confidence")
if temp_confounded:
    print("  so the ranking is CONFOUNDED: confidence is monotone in ink and every temperature "
          "is fainter than a real digit, so this judge cannot rank temperatures here")
else:
    print("  the ranking survives the ink check, so the page may read it as an answer")

# =========================================================== F. what the browser runs
banner("F. exporting a flow the page can invert")
flat = []
spec = []
for i in range(K):
    for nm, ten in (("act_logs", show.act[i].log_s), ("act_b", show.act[i].b),
                    ("mix_w", show.mix[i].W)):
        spec.append({"name": f"{nm}{i}", "offset": len(flat),
                     "shape": list(ten.shape), "size": int(ten.numel())})
        flat.extend(npy(ten).ravel().tolist())
    for j, conv in enumerate([show.cpl[i].net[0], show.cpl[i].net[2], show.cpl[i].net[4]]):
        for nm, ten in ((f"cpl{j}_w", conv.weight), (f"cpl{j}_b", conv.bias)):
            spec.append({"name": f"{nm}_{i}", "offset": len(flat),
                         "shape": list(ten.shape), "size": int(ten.numel())})
            flat.extend(npy(ten).ravel().tolist())
b64, v16 = export_flat(flat)
print(f"  {len(v16):,} numbers, float16 base64 is {len(b64) / 1024:.0f} kB")

with torch.no_grad():
    probe = to_t(DIG_TE[:16], seed=11)
    zp, ldp = show(probe)
    back = show.inverse(zp)
    round_trip = float(np.abs(npy(back) - npy(probe)).max())
    lp_probe = npy(show.log_prob(probe))
print(f"  encoding sixteen held out digits and inverting them returns the pixels to "
      f"{round_trip:.3e}, which is the thing a flow can do and no other model in this module "
      f"can")

# ===================================================== G. the module's scoreboard
banner("G. the four articles, side by side")
board = {}
for key, path, pull in (
        ("vae", "vae/data/vae.json", lambda d: {
            "toy_bound": d["toy"]["elbo"], "toy_exact": d["toy"]["exact"],
            "toy_kl": d["toy"]["kl_to_truth"], "digit_mse": d["compare"]["vae_test_mse"]}),
        ("vqvae", "vq-vae/data/vqvae.json", lambda d: {"present": True}),
        ("ebm", "ebm/data/ebm.json", lambda d: {"present": True}),
        ("flows", "flows/data/flows.json", lambda d: {
            "toy_exact": d["board"]["flow"]["exact"], "toy_kl": d["board"]["flow"]["kl"],
            "hole": d["board"]["flow"]["hole"], "truth_hole": d["board"]["truth_hole"]})):
    p = REPO / path
    if not p.is_file():
        board[key] = None
        print(f"  {path} is not there, so the scoreboard exports null for it")
        continue
    try:
        board[key] = pull(json.loads(p.read_text("utf-8")))
        print(f"  {path}: {board[key]}")
    except (KeyError, TypeError) as err:
        board[key] = None
        print(f"  {path} is there but its shape is not what was expected ({err})")
board["glow"] = {"bpd": rnd(bpd_deq, 4), "gzip": compress["gzip"], "png": compress["png"]}

SHOW_N = 12
payload = {
    "meta": {
        "seed": SEED, "seeds": SEEDS, "threads": THREADS, "torch": torch.__version__,
        "side": SIDE, "dim": DIM, "k": K, "hidden": HID, "epochs": EPOCHS, "batch": BATCH,
        "lr": LR, "levels": LEVELS, "temps": TEMPS,
        "n_digits_train": int(len(DIG_TR)), "n_digits_test": int(len(DIG_TE)),
        "n_fashion_train": int(len(FAS_TR)), "n_fashion_test": int(len(FAS_TE)),
        "judge_accuracy": rnd(JUDGE["accuracy"], 4),
        "judge_confidence": rnd(JUDGE["real_confidence"], 4),
        "note": "one level, no split, six flow steps: smaller than the paper's model, and "
                "every ablation is against the same shape so the comparisons are fair",
    },
    "arms": {"rows": arm_rows, "floor": rnd(FLOOR, 4),
             "floor_note": "every verdict is read against the floor of ITS OWN pair of cells; "
                           "`floor` here is the widest range anywhere in the table, which is "
                           "the right number for a claim about the table as a whole"},
    "trap": trap,
    "compress": compress,
    "ood": ood,
    "temperature": temperature,
    "board": board,
    "net": {
        "format": "float16 little-endian, base64; the tensors in the order of the spec below, "
                  "each row major in its own shape",
        "layers": spec, "count": int(len(v16)), "weights_b64": b64, "k": K, "hidden": HID,
        "round_trip": sci(round_trip),
        "probe": {"pixels": mat(npy(probe).reshape(-1, DIM)[:4], 5),
                  "logp": arr(lp_probe[:4], 4),
                  "logdet": arr(npy(ldp)[:4], 4)},
        "showcase": mat(dequantise(DIG_TE[:SHOW_N], seed=11), 5),
    },
}
write_json(OUT / "glow.json", payload)
