"""Data for the ATLAS 'EfficientNet and ConvNeXt' web article (module 1.3).

The previous article stopped at the point where a shortcut made depth safe.
This one asks what happened to convolutional networks afterwards, and it asks
it in the only way this repository allows: by training the variants and
measuring, never by quoting a paper.

Two threads, plus the arithmetic that sits under both.

  * Compound scaling. Given a fixed compute budget, is it better spent on
    width, on depth, or on input resolution? Seven allocations of the same
    x3 budget are trained from one base network, including the allocation
    EfficientNet's own exponents (alpha 1.2, beta 1.1, gamma 1.15) generate.
    The exponents produce the allocation and the budget lands where it lands;
    the charts are drawn against ACHIEVED multiply-accumulates, and the spread
    of achieved budgets inside the group is exported so a reader can see the
    residual compute confound rather than be told there is none.

  * The modernisation ladder. Start from a ResNet basic block and apply the
    ConvNeXt modernisations one at a time: patchify stem, depthwise
    convolution, inverted bottleneck, a larger kernel, GELU, fewer
    activations and norms, LayerNorm. Every rung changes exactly one thing,
    holds the rest fixed, uses the same seeds, epochs and optimiser, and
    carries a verdict against a noise floor that is measured at three points
    across the capacity range rather than assumed. Several rungs will not
    help at this scale. Reporting that is the point.

  * Depthwise separable arithmetic against the clock. The closed form
    (k*k*Cin + Cin*Cout against k*k*Cin*Cout) says the ratio tends to 1/k^2,
    so a 3x3 separable convolution can never be more than nine times cheaper
    than a dense one however wide the network grows. The saving is capped by
    the kernel, not by the width. The measured wall clock says something
    else, and the size of the disagreement is a result in its own right.

WHAT THIS GENERATOR ASSUMES, AND WHY IT MEASURES THE ASSUMPTION FIRST
---------------------------------------------------------------------
The brief's calibration point ("a 20-layer resnet-style CNN, 16 base
channels, on 6000 28x28 images, takes about 5 seconds per epoch") does not
survive contact with this machine. The repository's own depth_cache records
that network at 17.3 s per epoch, and a direct re-measurement gave 28.9. So
nothing here is priced off a constant taken on trust. The cost model used is

    seconds per epoch  =  MMAC x (N / 6000) x f

with f near 1.0 for dense 3x3 networks and near 3.0 for depthwise-plus-
pointwise networks, and a further penalty for channel-last LayerNorm. Those
three factors are MEASURED at startup by training two epochs of the ladder's
first rung, of its penultimate rung and of its last rung, before any
experiment is scheduled. The projection is printed. If it exceeds the time
budget the generator works down a cut order that is declared here, in the
docstring, before a single number has been seen. Above 45 minutes the
optional tier goes; only above 60 minutes does anything a chart depends on:

    1. the optional tier (k=11, leave-one-out, a x6 budget tier)
    2. the k=9 point of the kernel sweep
    3. the second matched re-widening (equal parameters)
    4. the second seed on the allocation simplex
    5. the simplex from seven allocations to four

Honest estimate, priced off the registry this file actually builds: at the
rate measured on a contended machine the core plan is about 74 minutes, and
the cut order takes it to about 58; at the rate this repository's own cache
implies it is about 40 and nothing is cut. The gate decides which world it is
in and prints what it did, and if the whole cut order still leaves it over
budget it says that too rather than pretending. The cache in
~/.atlas_vision_data/convnext_cache/ holds one JSON per trained run keyed by
everything that can change a result, so an interrupted run resumes and a
second run is instant.

A hybrid-core CPU (12 physical cores across performance, efficient and
low-power islands, 14 logical threads) means wall clock for an identical
operation moves by about a factor of two depending on thread count and on
what the Windows scheduler decides. The micro-benchmark is therefore run at
three thread counts and the spread is exported. The rule, enforced in the
JSON as ops.precision_note and in the article's prose: no timing claim is
made that is smaller than the measured spread of the same operation across
thread counts.

Fashion-MNIST throughout, 6000 training images frozen across every run, the
full 10000-image test split, resized identically to the training images
wherever a resolution changes, area-antialiased downwards so that a
low-resolution point measures resolution rather than aliasing.

Run from anywhere: `python src/utils/generate_convnext_data.py`.
"""
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_data import fashion_mnist, subset, FASHION_CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "convnext" / "data"
IMG = ROOT / "convnext" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

SEED = 19
SEEDS_CORE = (19, 20)                 # every headline comparison gets both
# Three seeds, not five. The depth article measured a noise floor from three
# and it separated a 20 point effect from a 1 point spread without trouble;
# the two extra seeds here cost more than every isolated ablation put together.
SEEDS_NOISE = (19, 20, 21)            # the noise-floor anchors get more
N_TRAIN = 6000
N_TEST = 10000                        # the full official test split
EPOCHS = 6
EPOCHS_LONG = 12                      # the epochs control
BATCH = 128
# Capped at the machine's core count: still 12 on the box that produced the
# published figures, so nothing moves here, and no oversubscription elsewhere.
THREADS = min(12, os.cpu_count() or 12)
NATIVE = 28
LOSS_THRESHOLD = 0.60                 # the lenet idiom: epochs to reach it
BUDGET_TARGET = 3.0                   # the scaling thread's budget multiplier
ALPHA, BETA, GAMMA = 1.2, 1.1, 1.15   # EfficientNet's own exponents
RUN_OPTIONAL = False                  # the optional tier, off by default
# Both budgets sit where they have to sit for the gate to do anything. Set at
# 45 and 60 they never fired, because the schedule they were guarding costs
# about 190 minutes at the constants this file measures for itself.
TIME_BUDGET_S = 30 * 60               # soft: the optional tier goes above this
HARD_BUDGET_S = 42 * 60               # hard: the rest of the cut order applies
PILOT_RATES = {"sgd": (0.02, 0.05, 0.1), "adamw": (5e-4, 1.5e-3, 4e-3)}
PILOT_EPOCHS = 4
PILOT_N = 3000

# The base network for both threads. C=12, three stages of two blocks, and a
# stem that halves the input, so the scaling thread at R=14 with a stride-1
# stem and the ladder at R=28 with a stride-2 stem see exactly the same
# 14/7/4 stage maps and cost exactly the same. That is what makes a number
# from one thread comparable with a number from the other.
BASE_WIDTH = 12
BASE_BLOCKS = (2, 2, 2)
T1_RES = 14
LADDER_RES = NATIVE

ASSUMED_F_DENSE = 1.0                 # replaced by measurement in M0
ASSUMED_F_MODERN = 3.0
ASSUMED_LN_PENALTY = 1.3

# Ordered by how little the article loses, and it has to contain cuts that
# are worth real minutes: an order made only of cosmetic entries cannot bring
# a three hour schedule down and the gate becomes decoration. Everything cut
# is exported under "cuts_applied" so the page can say what was dropped.
CUT_ORDER = [
    ("optional-tier", "the optional tier (k=11, leave-one-out, a x6 budget tier)"),
    ("epochs-control", "the twelve-epoch control on the two ladder ends"),
    ("kernel-k9", "the k=9 point of the kernel sweep"),
    ("resolution-thin", "the resolution sweep from seven points to five"),
    ("param-matched", "the second matched re-widening (equal parameters)"),
    ("ladder-seed2", "the second seed on the ladder rungs that are not anchors"),
    ("simplex-seed2", "the second seed on the allocation simplex"),
    ("simplex-to-4", "the simplex from seven allocations to four"),
]

torch.set_num_threads(THREADS)
torch.manual_seed(SEED)

CACHE = Path.home() / ".atlas_vision_data" / "convnext_cache"
CACHE.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "v1"


# ============================================================== architecture
# One configuration dictionary drives every network in the file. A ladder rung
# is that dictionary with one key changed, which is the only way an ablation
# column can honestly claim to change one thing.
def cfg(block="basic", width=BASE_WIDTH, blocks=BASE_BLOCKS, res=LADDER_RES,
        stem="conv3s2", kernel=3, act="relu", norm="bn", dw_pos="mid",
        expand=4, lean_act=False, lean_norm=False, res_via=None):
    """block   basic | sep | invert
    stem    conv3s1 | conv3s2 | patchify
    dw_pos  mid (expand, then depthwise at the wide width, MobileNetV2's
            order) | up (depthwise first at the narrow width, ConvNeXt's)
    lean_act   one activation per block instead of the usual two or three
    lean_norm  one normalisation per block, placed after the first operation
    res_via    resolution the image passes THROUGH before reaching `res`,
               which is how the information control removes information
               without changing the tensor shape the network sees
    """
    return {"block": block, "width": int(width), "blocks": tuple(int(b) for b in blocks),
            "res": int(res), "stem": stem, "kernel": int(kernel), "act": act,
            "norm": norm, "dw_pos": dw_pos, "expand": int(expand),
            "lean_act": bool(lean_act), "lean_norm": bool(lean_norm),
            "res_via": None if res_via is None else int(res_via)}


def cfg_hash(c):
    return hashlib.md5(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:8]


class ChanLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an (N, C, H, W) tensor.

    ConvNeXt normalises each position across its channels, which is not what
    BatchNorm2d does and not what nn.LayerNorm does to a channels-first
    tensor without help. The permutation is the honest cost of the swap and
    it is left in the timing, because the article's claim is about the block
    as it would actually be run, not about an idealised version of it.
    """

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
        self.num_features = int(c)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.num_features,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


def make_norm(c, ch):
    return nn.BatchNorm2d(ch) if c["norm"] == "bn" else ChanLayerNorm(ch)


def make_act(c):
    return nn.ReLU() if c["act"] == "relu" else nn.GELU()


def make_shortcut(cin, cout, stride, c):
    """A 1x1 projection only where the shape changes, the residual paper's
    option B. It keeps its normalisation in every rung of the ladder, so a
    rung that thins out the norms is changing the block body and nothing
    else."""
    if stride == 1 and cin == cout:
        return None
    return nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                         nn.BatchNorm2d(cout) if c["norm"] == "bn" else ChanLayerNorm(cout))


class BasicBlock(nn.Module):
    """ResNet's basic block: two dense k x k convolutions and a shortcut."""

    def __init__(self, cin, cout, stride, c):
        super().__init__()
        k, p = c["kernel"], c["kernel"] // 2
        self.c1 = nn.Conv2d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.n1 = make_norm(c, cout)
        self.a1 = make_act(c)
        self.c2 = nn.Conv2d(cout, cout, k, padding=p, bias=False)
        self.n2 = None if c["lean_norm"] else make_norm(c, cout)
        self.a2 = None if c["lean_act"] else make_act(c)
        self.short = make_shortcut(cin, cout, stride, c)

    def forward(self, x):
        f = self.a1(self.n1(self.c1(x)))
        f = self.c2(f)
        if self.n2 is not None:
            f = self.n2(f)
        s = x if self.short is None else self.short(x)
        out = f + s
        return out if self.a2 is None else self.a2(out)


class SepBlock(nn.Module):
    """The same block with each dense k x k replaced by a depthwise k x k and
    a pointwise 1x1. Normalisation and activation stay exactly where the
    basic block put them, so this rung changes the convolution and nothing
    else about the block's shape."""

    def __init__(self, cin, cout, stride, c):
        super().__init__()
        k, p = c["kernel"], c["kernel"] // 2
        self.d1 = nn.Conv2d(cin, cin, k, stride=stride, padding=p, groups=cin, bias=False)
        self.p1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.n1 = make_norm(c, cout)
        self.a1 = make_act(c)
        self.d2 = nn.Conv2d(cout, cout, k, padding=p, groups=cout, bias=False)
        self.p2 = nn.Conv2d(cout, cout, 1, bias=False)
        self.n2 = None if c["lean_norm"] else make_norm(c, cout)
        self.a2 = None if c["lean_act"] else make_act(c)
        self.short = make_shortcut(cin, cout, stride, c)

    def forward(self, x):
        f = self.a1(self.n1(self.p1(self.d1(x))))
        f = self.p2(self.d2(f))
        if self.n2 is not None:
            f = self.n2(f)
        s = x if self.short is None else self.short(x)
        out = f + s
        return out if self.a2 is None else self.a2(out)


class InvertBlock(nn.Module):
    """The inverted bottleneck: expand by `expand`, do the spatial work
    depthwise, project back.

    dw_pos 'mid' is MobileNetV2's order, with the depthwise convolution at the
    expanded width, so it costs k*k*(e*C) per position. dw_pos 'up' is
    ConvNeXt's, with the depthwise convolution moved above the expansion where
    it costs k*k*C, a quarter as much at e=4. That is the move that makes a
    7x7 kernel affordable, and it is measured here rather than asserted.
    """

    def __init__(self, cin, cout, stride, c):
        super().__init__()
        k, p, e = c["kernel"], c["kernel"] // 2, c["expand"]
        mid = e * cout
        self.up = c["dw_pos"] == "up"
        if self.up:
            self.dw = nn.Conv2d(cin, cin, k, stride=stride, padding=p, groups=cin, bias=False)
            self.n_dw = make_norm(c, cin)                                   # the surviving norm
            self.pe = nn.Conv2d(cin, mid, 1, bias=False)
            self.n_e = None if c["lean_norm"] else make_norm(c, mid)
        else:
            self.pe = nn.Conv2d(cin, mid, 1, bias=False)
            self.n_e = make_norm(c, mid)                                    # the surviving norm
            self.dw = nn.Conv2d(mid, mid, k, stride=stride, padding=p, groups=mid, bias=False)
            self.n_dw = None if c["lean_norm"] else make_norm(c, mid)
        self.a_e = make_act(c)                                              # the surviving act
        self.a_dw = None if c["lean_act"] else make_act(c)
        self.pp = nn.Conv2d(mid, cout, 1, bias=False)
        self.n_p = None if c["lean_norm"] else make_norm(c, cout)
        self.a_out = None if c["lean_act"] else make_act(c)
        self.short = make_shortcut(cin, cout, stride, c)

    def forward(self, x):
        if self.up:
            f = self.dw(x)
            f = self.n_dw(f)
            if self.a_dw is not None:
                f = self.a_dw(f)
            f = self.pe(f)
            if self.n_e is not None:
                f = self.n_e(f)
            f = self.a_e(f)
        else:
            f = self.a_e(self.n_e(self.pe(x)))
            f = self.dw(f)
            if self.n_dw is not None:
                f = self.n_dw(f)
            if self.a_dw is not None:
                f = self.a_dw(f)
        f = self.pp(f)
        if self.n_p is not None:
            f = self.n_p(f)
        s = x if self.short is None else self.short(x)
        out = f + s
        return out if self.a_out is None else self.a_out(out)


BLOCKS = {"basic": BasicBlock, "sep": SepBlock, "invert": InvertBlock}


def make_stem(c):
    """Every stem keeps the same normalisation and activation, so the patchify
    rung swaps one convolution for another and changes nothing else. The
    3x3 stride-2 stem and the 2x2 stride-2 patchify stem produce the same
    output shape, which is what lets their costs be compared directly."""
    ch = c["width"]
    if c["stem"] == "patchify":
        conv = nn.Conv2d(1, ch, 2, stride=2, bias=False)
    elif c["stem"] == "conv3s2":
        conv = nn.Conv2d(1, ch, 3, stride=2, padding=1, bias=False)
    else:
        conv = nn.Conv2d(1, ch, 3, stride=1, padding=1, bias=False)
    return nn.Sequential(conv, make_norm(c, ch), make_act(c))


class Net(nn.Module):
    """Stem, three stages, global average pool, linear classifier. The head is
    identical in every configuration in this file."""

    def __init__(self, c):
        super().__init__()
        self.cfg = dict(c)
        cls = BLOCKS[c["block"]]
        self.stem = make_stem(c)
        blocks = []
        cin = c["width"]
        for s, n in enumerate(c["blocks"]):
            cout = c["width"] * (2 ** s)
            for b in range(n):
                blocks.append(cls(cin, cout, 2 if (s > 0 and b == 0) else 1, c))
                cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, 10)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.fc(torch.flatten(self.pool(x), 1))


# ======================================================= counting the cost
# Every multiply-accumulate quoted in the article is counted from the layers
# torch actually built, by watching a single forward pass. The browser is
# given the same layer list and sums it to check the totals it is handed.
def profile(model, res):
    """(ops, macs, params). `ops` is the executed layer list, in order."""
    ops, handles = [], []

    def hook(mod, inp, out, name=""):
        params = sum(p.numel() for p in mod.parameters(recurse=False))
        rec = {"name": name, "params": int(params), "macs": 0,
               "shortcut": ".short" in name}
        if isinstance(mod, nn.Conv2d):
            oh, ow = int(out.shape[-2]), int(out.shape[-1])
            macs = (mod.in_channels // mod.groups) * mod.out_channels * \
                mod.kernel_size[0] * mod.kernel_size[1] * oh * ow
            rec.update(type="conv", cin=int(mod.in_channels), cout=int(mod.out_channels),
                       k=int(mod.kernel_size[0]), stride=int(mod.stride[0]),
                       groups=int(mod.groups), out=[oh, ow], macs=int(macs),
                       depthwise=bool(mod.groups == mod.in_channels and mod.groups > 1),
                       pointwise=bool(mod.kernel_size[0] == 1))
        elif isinstance(mod, nn.Linear):
            rec.update(type="linear", cin=int(mod.in_features), cout=int(mod.out_features),
                       macs=int(mod.in_features * mod.out_features))
        elif isinstance(mod, nn.BatchNorm2d):
            rec.update(type="norm-bn", cin=int(mod.num_features), cout=int(mod.num_features))
        elif isinstance(mod, ChanLayerNorm):
            rec.update(type="norm-ln", cin=mod.num_features, cout=mod.num_features)
        elif isinstance(mod, nn.ReLU):
            rec.update(type="act-relu")
        elif isinstance(mod, nn.GELU):
            rec.update(type="act-gelu")
        else:
            return
        ops.append(rec)

    watched = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, ChanLayerNorm, nn.ReLU, nn.GELU)
    for name, mod in model.named_modules():
        if isinstance(mod, watched):
            handles.append(mod.register_forward_hook(
                lambda m, i, o, n=name: hook(m, i, o, n)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, res, res))
    for h in handles:
        h.remove()
    model.train(was_training)
    macs = sum(o["macs"] for o in ops)
    params = sum(p.numel() for p in model.parameters())
    return ops, int(macs), int(params)


def conv_ops(ops):
    """The rows the browser sums to recheck a total: convolutions and the
    classifier, without the zero-cost normalisations and activations. Flags
    are carried only where they are true and layer names only in the fuller
    `layer_stack`, because this list is exported for every run in the file and
    its size is most of the size of the file."""
    keep = ("type", "cin", "cout", "k", "stride", "groups", "out", "macs", "params")
    out = []
    for o in ops:
        if o["type"] not in ("conv", "linear"):
            continue
        row = {k: o[k] for k in keep if k in o}
        for flag in ("depthwise", "pointwise", "shortcut"):
            if o.get(flag):
                row[flag] = True
        out.append(row)
    return out


def receptive_field(ops):
    """Walked along the main path only: a 1x1 shortcut is a branch, not a
    place where the network looks further out."""
    rf, jump = 1, 1
    for o in ops:
        if o["type"] != "conv" or o.get("shortcut"):
            continue
        rf += (o["k"] - 1) * jump
        jump *= o["stride"]
    return int(rf), int(jump)


def conv_out(size, k, stride, pad):
    return (size + 2 * pad - k) // stride + 1


def stage_maps(c, ideal):
    """The spatial size entering each stage, either the idealised R/2^s the
    textbook cost formula assumes or the size torch actually produces."""
    if c["stem"] == "patchify":
        ks, st, pad = 2, 2, 0
    elif c["stem"] == "conv3s2":
        ks, st, pad = 3, 2, 1
    else:
        ks, st, pad = 3, 1, 1
    if ideal:
        m = [c["res"] / st]
        for _ in range(1, len(c["blocks"])):
            m.append(m[-1] / 2)
    else:
        m = [conv_out(c["res"], ks, st, pad)]
        for _ in range(1, len(c["blocks"])):
            m.append(conv_out(m[-1], c["kernel"], 2, c["kernel"] // 2))
    return m, (ks, st, pad)


def analytic_macs(c, ideal=True):
    """The cost formula as it is normally written, for the dense family only.

    Exact in the channel counts, including the fact that a downsampling block
    costs 1.5 k^2 C^2 rather than 2 k^2 C^2 because its first convolution
    reads half as many channels as it writes. Not exact in space: with
    `ideal` it assumes R/2 and R/4 are whole numbers, which at R=28 they are
    not, and the difference between the two settings is the spatial rounding
    term the article quotes.
    """
    if c["block"] != "basic":
        return None
    k, C, nb = c["kernel"], c["width"], c["blocks"]
    m, (ks, _, _) = stage_maps(c, ideal)
    total = ks * ks * 1 * C * m[0] ** 2
    for s, n in enumerate(nb):
        cs = C * (2 ** s)
        area = m[s] ** 2
        if s == 0:
            total += n * 2 * k * k * cs * cs * area
        else:
            total += (1.5 * k * k * cs * cs + (n - 1) * 2 * k * k * cs * cs) * area
    total += C * (2 ** (len(nb) - 1)) * 10
    return float(total)


def cost_check(c, ops, counted, label):
    """Analytic against counted, with the residual attributed by measurement
    rather than by assertion: the spatial term is computed by evaluating the
    same formula twice, and the shortcut term is read off the counted rows."""
    ideal = analytic_macs(c, ideal=True)
    if ideal is None:
        return None
    real = analytic_macs(c, ideal=False)
    shortcuts = sum(o["macs"] for o in ops if o.get("shortcut"))
    resid = counted - real - shortcuts
    return {"id": label, "analytic": int(round(ideal)), "counted": int(counted),
            "rel_err": rnd((counted - ideal) / ideal, 5),
            "analytic_real_spatial": int(round(real)),
            "spatial_rounding_term": int(round(real - ideal)),
            "spatial_rounding_share": rnd((real - ideal) / max(1.0, counted - ideal), 4),
            "shortcut_term": int(shortcuts),
            "unexplained_term": int(round(resid)),
            "stage_maps_ideal": [rnd(v, 3) for v in stage_maps(c, True)[0]],
            "stage_maps_real": [int(v) for v in stage_maps(c, False)[0]],
            "why": "the idealised formula divides the resolution by two at every "
                   "stage; torch rounds the map up instead, and at this resolution "
                   "that overcount on the last and most expensive stage is most of "
                   "the gap. The 1x1 shortcut convolutions are the rest."}


# ================================================================== the data
Xtr_u8, ytr_np_full, Xte_u8_full, yte_np = fashion_mnist()
Xtr_u8, ytr_np = subset(Xtr_u8, ytr_np_full, N_TRAIN, seed=SEED)
Xtr_native = torch.tensor(Xtr_u8, dtype=torch.float32).unsqueeze(1) / 255.0
Xte_native = torch.tensor(Xte_u8_full, dtype=torch.float32).unsqueeze(1) / 255.0
ytr = torch.tensor(ytr_np, dtype=torch.long)
yte = torch.tensor(yte_np, dtype=torch.long)
print(f"Fashion-MNIST: {len(ytr)} train (frozen, class-stratified, seed {SEED}) / "
      f"{len(yte)} test (the full official split), {NATIVE}x{NATIVE}, "
      f"{len(FASHION_CLASSES)} classes")
print(f"torch {torch.__version__}, {THREADS} threads requested")


def _interp(X, r):
    """Downwards with antialiasing, upwards without. Plain bilinear
    downsampling aliases, and every low-resolution point in the resolution
    ladder would then be measuring the aliasing rather than the resolution."""
    cur = int(X.shape[-1])
    if r == cur:
        return X
    if r < cur:
        return F.interpolate(X, size=(r, r), mode="bilinear",
                             align_corners=False, antialias=True)
    return F.interpolate(X, size=(r, r), mode="bilinear", align_corners=False)


def resize_label(res, via):
    if res == NATIVE and via is None:
        return "none (native 28x28)"
    if via is None:
        d = "bilinear-antialias" if res < NATIVE else "bilinear"
        return f"{d} {NATIVE}->{res}"
    return (f"bilinear-antialias {NATIVE}->{via}, then "
            f"{'bilinear' if res > via else 'bilinear-antialias'} {via}->{res}")


_DATA = {}


def data_at(res, via=None):
    """Train and test resized by exactly the same operation. Nothing in this
    file ever resizes one and not the other."""
    key = (res, via)
    if key not in _DATA:
        def go(X):
            out = X if via is None else _interp(X, via)
            return _interp(out, res).contiguous()
        _DATA[key] = (go(Xtr_native), go(Xte_native))
    return _DATA[key]


# ================================================================= training
def train(model, X, y, epochs, seed, recipe, lr, log=False):
    """Two recipes, both tuned at both ends of the ladder by the pilot.

    `sgd` is the recipe the residual-network article tuned, kept unchanged so
    that a number from this article can be put next to a number from that
    one. `adamw` is the recipe the modern papers use. Comparing a tuned
    recipe against an untuned one is the confound that would make every
    'the modernised block wins' claim in this file unfalsifiable, so neither
    is allowed to run on a rate borrowed from a paper.
    """
    torch.manual_seed(seed)
    steps_per_epoch = max(1, math.ceil(len(y) / BATCH))
    if recipe == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        sched_epoch = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(epochs * 0.5), int(epochs * 0.8)], gamma=0.1)
        sched_step = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        total = epochs * steps_per_epoch
        warm = steps_per_epoch

        def lam(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

        sched_step = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        sched_epoch = None
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    history, epoch_seconds = [], []
    t_all = time.perf_counter()
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(y), generator=g)
        tot = 0.0
        for i in range(0, len(y), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            if sched_step is not None:
                sched_step.step()
            tot += float(loss.detach()) * len(idx)
        if sched_epoch is not None:
            sched_epoch.step()
        epoch_seconds.append(time.perf_counter() - t0)
        history.append(rnd(tot / len(y), 5))
        if log:
            print(f"      epoch {ep + 1}: train loss {history[-1]:.4f} "
                  f"({epoch_seconds[-1]:.1f}s)", flush=True)
    return history, time.perf_counter() - t_all, epoch_seconds


@torch.no_grad()
def evaluate(model, X, y):
    """Overall accuracy and the ten per-class accuracies, which cost nothing
    extra and are what the resolution thread needs: resolution should help
    the classes that are separated by fine texture, not the ones separated by
    silhouette."""
    model.eval()
    preds = []
    for i in range(0, len(y), 500):
        preds.append(model(X[i:i + 500]).argmax(1))
    p = torch.cat(preds)
    acc = float((p == y).float().mean())
    per = []
    for cls in range(10):
        m = y == cls
        per.append(float((p[m] == cls).float().mean()) if int(m.sum()) else float("nan"))
    return acc, per


def epochs_to_threshold(history, thresh=LOSS_THRESHOLD):
    return next((i + 1 for i, v in enumerate(history) if v <= thresh), None)


# ==================================================================== cache
def cached(key, fn):
    """One JSON per trained run, keyed by everything that can change the
    result, so two experiments cannot collide and an interrupted run resumes
    exactly where it stopped."""
    path = CACHE / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")), True
        except Exception:
            pass
    result = fn()
    path.write_text(json.dumps(result), encoding="utf-8")
    return result, False


def run_key(run_id, c, seed, recipe, lr, epochs, n_train):
    return (f"fashion-n{n_train}-e{epochs}-b{BATCH}-{recipe}-lr{lr:g}-"
            f"{run_id}-s{seed}-{cfg_hash(c)}-{CACHE_VERSION}")


def run_once(run_id, c, seed, recipe, lr, epochs, Xa, ya, Xb, yb, extract=None):
    def go():
        torch.manual_seed(seed)
        model = Net(c)
        hist, secs, eps = train(model, Xa, ya, epochs, seed, recipe, lr)
        te_acc, per_class = evaluate(model, Xb, yb)
        tr_acc, _ = evaluate(model, Xa, ya)
        rec = {"seed": int(seed), "test_acc": rnd(te_acc, 4), "train_acc": rnd(tr_acc, 4),
               "per_class": [rnd(v, 4) for v in per_class], "history": hist,
               "seconds": rnd(secs, 1), "epoch_seconds": [rnd(v, 2) for v in eps]}
        if extract is not None:
            rec["extra"] = extract(model)
        return rec

    rec, hit = cached(run_key(run_id, c, seed, recipe, lr, epochs, len(ya)), go)
    if extract is not None and "extra" not in rec:
        # A cache written before the extraction existed: recompute just the
        # extraction rather than throwing away a trained run's numbers.
        extra, _ = cached(run_key(run_id + "-extract", c, seed, recipe, lr, epochs, len(ya)),
                          lambda: go()["extra"])
        rec["extra"] = extra
    return rec, hit


def mean_sd(values):
    v = [float(x) for x in values]
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m, s


# ======================================================= the ladder's rungs
# Eleven rungs, each one key of the configuration dictionary away from the
# rung below it. Rung 9 is the one place two changes travel together, because
# 'fewer activations' and 'fewer normalisations' are one modernisation in the
# paper; both are also measured separately in the isolated ablations, which
# is the only reason the bundling is tolerable here.
RUNG_META = [
    (0, "ResNet basic block, SGD", "control"),
    (1, "AdamW plus cosine, architecture byte-identical", "optimisation"),
    (2, "patchify stem, 2x2 stride 2", "regime"),
    (3, "depthwise separable convolutions", "cost"),
    (4, "widen 12 to 18 channels", "cost"),
    (5, "inverted bottleneck, e=4", "design"),
    (6, "depthwise moved above the expansion", "design"),
    (7, "kernel 3 to 7", "design"),
    (8, "GELU instead of ReLU", "optimisation"),
    (9, "one activation and one norm per block", "regime"),
    (10, "LayerNorm over channels", "optimisation"),
]
LADDER_TOP = 10
LADDER_BN_TOP = 9          # the modernised block still on BatchNorm


def ladder_cfg(n):
    """The configuration at rung n, and the recipe it is trained with."""
    c = cfg(res=LADDER_RES, stem="conv3s2")
    recipe = "sgd"
    if n >= 1:
        recipe = "adamw"
    if n >= 2:
        c["stem"] = "patchify"
    if n >= 3:
        c["block"] = "sep"
    if n >= 4:
        c["width"] = 18                 # 1.5C, so that an e=4 inverted
    if n >= 5:                          # bottleneck matches the basic block's
        c["block"] = "invert"           # dense cost: 8(1.5C)^2 = 18C^2
    if n >= 6:
        c["dw_pos"] = "up"
    if n >= 7:
        c["kernel"] = 7
    if n >= 8:
        c["act"] = "gelu"
    if n >= 9:
        c["lean_act"] = True
        c["lean_norm"] = True
    if n >= 10:
        c["norm"] = "ln"
    return c, recipe


def is_dense(c):
    return c["block"] == "basic"


# ===================================================== M0, the measure gate
# Nothing downstream trusts a constant. The micro-benchmark measures the
# operations the article talks about, WITH gradients flowing back to the
# input, which is what training does and what a benchmark that only asks for
# weight gradients leaves out. It runs at three thread counts because on a
# hybrid-core part the ratio between two operations is not a property of the
# operations alone, and the spread is what licenses the article's precision.
MICRO_SHAPES = [("stage0", 12, 14), ("stage1", 24, 7), ("stage2", 48, 4)]
MICRO_BATCH = 128
MICRO_THREADS = (4, 8, 12)
MICRO_REPS = 15
MICRO_WARM = 4
MICRO_ROUNDS = 3


def micro_ops(ch):
    e = 4 * ch
    return [
        ("dense3", nn.Conv2d(ch, ch, 3, padding=1, bias=False), 3, ch, ch, 1),
        ("dense7", nn.Conv2d(ch, ch, 7, padding=3, bias=False), 7, ch, ch, 1),
        ("dw3", nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False), 3, ch, ch, ch),
        ("dw5", nn.Conv2d(ch, ch, 5, padding=2, groups=ch, bias=False), 5, ch, ch, ch),
        ("dw7", nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False), 7, ch, ch, ch),
        ("dw9", nn.Conv2d(ch, ch, 9, padding=4, groups=ch, bias=False), 9, ch, ch, ch),
        ("pw_same", nn.Conv2d(ch, ch, 1, bias=False), 1, ch, ch, 1),
        ("pw_expand", nn.Conv2d(ch, e, 1, bias=False), 1, ch, e, 1),
        ("pw_project", nn.Conv2d(e, ch, 1, bias=False), 1, e, ch, 1),
    ]


def bench_one(mod, x):
    for _ in range(MICRO_WARM):
        mod(x).sum().backward()
        mod.zero_grad(set_to_none=True)
        x.grad = None
    ts = []
    for _ in range(MICRO_REPS):
        t0 = time.perf_counter()
        mod(x).sum().backward()
        ts.append(time.perf_counter() - t0)
        mod.zero_grad(set_to_none=True)
        x.grad = None
    return float(np.median(ts))


def run_micro():
    rows = {}
    for threads in MICRO_THREADS:
        try:
            torch.set_num_threads(threads)
        except Exception as exc:      # noqa: BLE001
            print(f"  could not request {threads} threads: {exc}", flush=True)
        for rounds in range(MICRO_ROUNDS):
            for shape, ch, s in MICRO_SHAPES:
                torch.manual_seed(SEED)
                for name, mod, k, cin, cout, groups in micro_ops(ch):
                    src = ch if name != "pw_project" else 4 * ch
                    x = torch.randn(MICRO_BATCH, src, s, s, requires_grad=True)
                    ms = bench_one(mod, x) * 1000.0
                    macs = (cin // groups) * cout * k * k * s * s * MICRO_BATCH
                    key = (threads, shape, name)
                    prev = rows.get(key)
                    if prev is None or ms < prev["ms"]:
                        rows[key] = {"threads": threads, "shape": shape, "op": name,
                                     "channels": ch, "spatial": s, "batch": MICRO_BATCH,
                                     "k": k, "groups": groups, "macs": int(macs),
                                     "ms": rnd(ms, 4),
                                     "gmac_s": rnd(macs / (ms / 1000.0) / 1e9, 3)}
    torch.set_num_threads(THREADS)
    return [rows[k] for k in sorted(rows)]


print("\nM0: measuring before planning. Micro-benchmark, forward and backward, "
      "input gradients on, three thread counts:")
micro_rows, micro_hit = cached(
    "micro-" + CACHE_VERSION + "-t" + "_".join(str(t) for t in MICRO_THREADS) +
    f"-r{MICRO_REPS}x{MICRO_ROUNDS}-b{MICRO_BATCH}", run_micro)
print(f"  {len(micro_rows)} rows{' [cached]' if micro_hit else ''}")


def micro_get(threads, shape, op):
    for r in micro_rows:
        if r["threads"] == threads and r["shape"] == shape and r["op"] == op:
            return r
    return None


thread_spread = {}
for shape, ch, s in MICRO_SHAPES:
    for name, *_ in micro_ops(ch):
        vals = [micro_get(t, shape, name)["ms"] for t in MICRO_THREADS
                if micro_get(t, shape, name)]
        if len(vals) > 1 and min(vals) > 0:
            thread_spread[f"{shape}/{name}"] = rnd(max(vals) / min(vals), 3)
worst_spread = max(thread_spread.values()) if thread_spread else 1.0
PRECISION_NOTE = (
    f"On this hybrid-core CPU the same operation, measured three times at each of "
    f"{list(MICRO_THREADS)} threads and reported at its fastest, still moves by up to a "
    f"factor of {worst_spread:.2f} across thread counts. No claim about wall clock "
    f"smaller than that factor is made anywhere in this article. Multiply-accumulate "
    f"counts, which are exact, carry the precise claims instead.")
print(f"  worst spread across thread counts: x{worst_spread:.2f}")
print(f"  {PRECISION_NOTE}")

# The throughput factors, measured on the real training loop rather than on a
# micro-benchmark, because a benchmark of isolated layers is not a training
# step. Two epochs, and the second one is the one that counts, so that the
# first call's warm-up does not become the article's cost model.
print("\nM0: calibrating the cost model on real epochs "
      "(the brief's 5 s/epoch anchor is not assumed):")
CAL_EPOCHS = 2
calibration = {}
for tag, n in (("dense", 0), ("modern-bn", LADDER_BN_TOP), ("modern-ln", LADDER_TOP)):
    c_cal, rec_cal = ladder_cfg(n)
    _, macs_cal, _ = profile(Net(c_cal), c_cal["res"])
    Xa, _ = data_at(c_cal["res"], c_cal["res_via"])

    def go(c_cal=c_cal, rec_cal=rec_cal, Xa=Xa):
        torch.manual_seed(SEED)
        m = Net(c_cal)
        _, _, eps = train(m, Xa, ytr, CAL_EPOCHS, SEED, rec_cal,
                          0.05 if rec_cal == "sgd" else 1e-3)
        return {"epoch_seconds": [rnd(v, 3) for v in eps]}

    res_cal, hit = cached(f"calib-{tag}-{cfg_hash(c_cal)}-{CACHE_VERSION}", go)
    sec = res_cal["epoch_seconds"][-1]
    f = sec / ((macs_cal / 1e6) * (N_TRAIN / 6000.0))
    calibration[tag] = {"rung": n, "macs": macs_cal, "seconds_per_epoch": rnd(sec, 2),
                        "f": rnd(f, 4)}
    print(f"  {tag:10s} rung {n:2d}  {macs_cal / 1e6:6.2f} MMAC  "
          f"{sec:6.2f} s/epoch  f = {f:.2f}{'  [cached]' if hit else ''}", flush=True)

F_DENSE = calibration["dense"]["f"]
F_MODERN = calibration["modern-bn"]["f"]
LN_PENALTY = calibration["modern-ln"]["f"] / F_MODERN
print(f"  measured f_dense {F_DENSE:.2f} (assumed {ASSUMED_F_DENSE:.2f}), "
      f"f_modern {F_MODERN:.2f} (assumed {ASSUMED_F_MODERN:.2f}), "
      f"LayerNorm penalty x{LN_PENALTY:.2f} (assumed x{ASSUMED_LN_PENALTY:.2f})")


def estimate_seconds(c, macs, epochs, n_train, n_seeds):
    f = F_DENSE if is_dense(c) else F_MODERN
    if c["norm"] == "ln":
        f *= LN_PENALTY
    per_epoch = (macs / 1e6) * (n_train / 6000.0) * f
    # one pass over the training subset and the whole test split, per run
    evals = (macs / 1e6) * ((n_train + N_TEST) / 6000.0) * f * 0.35
    return n_seeds * (per_epoch * epochs + evals)


# ============================================== the registry of planned runs
# Every run is declared, profiled and priced before any of them is trained, so
# that the projection is a projection of what will actually happen and the cut
# order is applied to a real schedule rather than to a guess.
PLANNED = []
RESULTS = {}


def plan(group, run_id, c, seeds, recipe, lr_key, epochs=EPOCHS, n_train=N_TRAIN,
         claim_scope="design", label="", cut_tag=None, extract=None,
         layer_stack=False, note=""):
    ops, macs, params = profile(Net(c), c["res"])
    rf, jump = receptive_field(ops)
    entry = {"group": group, "id": run_id, "cfg": c, "seeds": list(seeds),
             "recipe": recipe, "lr_key": lr_key, "epochs": epochs, "n_train": n_train,
             "claim_scope": claim_scope, "label": label or run_id, "cut_tag": cut_tag,
             "extract": extract, "layer_stack": layer_stack, "note": note,
             "ops": ops, "macs": macs, "params": params, "rf": rf, "total_stride": jump}
    PLANNED.append(entry)
    return entry


def find(run_id):
    for e in PLANNED:
        if e["id"] == run_id:
            return e
    return None


def scale_blocks(base, factor):
    """Scale the total number of blocks and hand them out by largest
    remainder, so the stage proportions of the base survive the rounding."""
    exact = [n * factor for n in base]
    out = [max(1, int(math.floor(v))) for v in exact]
    want = max(len(base), int(round(sum(exact))))
    order = sorted(range(len(base)), key=lambda i: exact[i] - math.floor(exact[i]),
                   reverse=True)
    i = 0
    while sum(out) < want:
        out[order[i % len(order)]] += 1
        i += 1
    return tuple(out)


COMPOUND = ALPHA * BETA ** 2 * GAMMA ** 2
PHI = math.log(BUDGET_TARGET) / math.log(COMPOUND)
EFF_D, EFF_W, EFF_R = ALPHA ** PHI, BETA ** PHI, GAMMA ** PHI
SQ3 = BUDGET_TARGET ** 0.5
QT3 = BUDGET_TARGET ** 0.25
SIMPLEX = [
    ("depth", BUDGET_TARGET, 1.0, 1.0),
    ("width", 1.0, SQ3, 1.0),
    ("res", 1.0, 1.0, SQ3),
    ("depth+width", SQ3, QT3, 1.0),
    ("depth+res", SQ3, 1.0, QT3),
    ("width+res", 1.0, QT3, QT3),
    ("efficientnet", EFF_D, EFF_W, EFF_R),
]
SIMPLEX_KEEP4 = {"depth", "width", "res", "efficientnet"}
RES_LADDER = (10, 12, 14, 18, 20, 24, 28)


def t1_cfg(df=1.0, wf=1.0, rf=1.0):
    return cfg(block="basic", width=max(4, int(round(BASE_WIDTH * wf))),
               blocks=scale_blocks(BASE_BLOCKS, df),
               res=min(NATIVE, max(8, int(round(T1_RES * rf)))), stem="conv3s1")


# ---- thread 1, scaling ------------------------------------------------
S0 = t1_cfg()
plan("scaling", "s0-base", S0, SEEDS_CORE, "sgd", ("sgd", "dense"),
     claim_scope="design", label="base network, C=12, blocks (2,2,2), R=14",
     layer_stack=True, note="the base both scaling arms are grown from")
for name, df, wf, rf in SIMPLEX:
    plan("scaling", f"scale-{name}", t1_cfg(df, wf, rf), SEEDS_CORE, "sgd",
         ("sgd", "dense"), claim_scope="design",
         label=f"x{BUDGET_TARGET:g} budget spent on {name}",
         cut_tag=("simplex-seed2" if name in SIMPLEX_KEEP4 else "simplex-to-4"))
for r in RES_LADDER:
    if r == T1_RES:
        continue      # this point is the base network itself
    plan("resolution", f"res-{r}", cfg(block="basic", width=BASE_WIDTH,
                                       blocks=BASE_BLOCKS, res=r, stem="conv3s1"),
         (SEED,), "sgd", ("sgd", "dense"), claim_scope="design",
         label=f"input resolution {r}x{r}")
plan("resolution", "res-24-restride",
     cfg(block="basic", width=BASE_WIDTH, blocks=BASE_BLOCKS, res=24, stem="conv3s2"),
     (SEED,), "sgd", ("sgd", "dense"), claim_scope="design",
     label="24x24 with a stride-2 stem, restoring the receptive-field fraction",
     note="the control none of the usual resolution ladders carry: raising the "
          "input while holding the stride schedule shrinks the receptive field "
          "as a fraction of the image, so part of what a resolution ladder "
          "measures is the network looking at relatively less")
plan("resolution", "res-24-from-12",
     cfg(block="basic", width=BASE_WIDTH, blocks=BASE_BLOCKS, res=24, stem="conv3s1",
         res_via=12),
     (SEED,), "sgd", ("sgd", "dense"), claim_scope="design",
     label="24x24 rebuilt from a 12-pixel downsample",
     note="identical tensor shape, identical architecture, identical compute, "
          "strictly less information than res-24")


# ---- thread 2, the modernisation ladder --------------------------------
GARMENT_IDX = int(np.where(yte_np == 0)[0][0])


def extract_dw(model):
    """The trained 7x7 depthwise kernels of the first block, together with the
    exact tensor that entered them for one garment and the exact tensor that
    left, so the browser can convolve them itself and print its agreement.

    A depthwise convolution touches one channel at a time, so eight channels
    are a complete check of those eight and the file does not need all
    eighteen to make the point.
    """
    blk = model.blocks[0]
    dw = getattr(blk, "dw", None)
    if dw is None or dw.groups <= 1:
        return {}
    store = {}
    h = dw.register_forward_hook(
        lambda m, i, o: store.update(inp=i[0].detach().clone(), out=o.detach().clone()))
    model.eval()
    with torch.no_grad():
        _, Xt = data_at(model.cfg["res"], model.cfg["res_via"])
        model(Xt[GARMENT_IDX:GARMENT_IDX + 1])
    h.remove()
    w = dw.weight.detach()[:, 0].numpy()
    inp = store["inp"][0].numpy()
    out = store["out"][0].numpy()
    n = int(min(8, w.shape[0]))
    return {"k": int(w.shape[-1]), "channels": int(w.shape[0]), "channels_shown": n,
            "padding": int(dw.padding[0]), "stride": int(dw.stride[0]),
            "test_index": GARMENT_IDX, "spatial": int(inp.shape[-1]),
            "weights": [[[rnd(v, 6) for v in row] for row in f] for f in w],
            "input": [[[rnd(v, 5) for v in row] for row in ch] for ch in inp[:n]],
            "output": [[[rnd(v, 5) for v in row] for row in ch] for ch in out[:n]]}


LADDER_SEEDS = {0: SEEDS_NOISE, 4: (19, 20, 21, 22), LADDER_TOP: (19, 20, 21, 22)}
for n, change, scope in RUNG_META:
    c_r, rec_r = ladder_cfg(n)
    end = "dense" if is_dense(c_r) else "modern"
    plan("ladder", f"rung{n:02d}", c_r, LADDER_SEEDS.get(n, SEEDS_CORE), rec_r,
         (rec_r, end), claim_scope=scope, label=change, layer_stack=True,
         extract=extract_dw if n == LADDER_TOP else None,
         note=("two changes travel together on this rung because the paper treats "
               "them as one modernisation; both are measured separately in the "
               "isolated ablations" if n == 9 else ""))

# The controls that make the ladder's column mean something.
ISO_BASE, ISO_RECIPE = ladder_cfg(1)
ISOLATED = [
    ("iso-patchify", {"stem": "patchify"}, "regime", "patchify stem alone"),
    ("iso-depthwise", {"block": "sep"}, "cost", "depthwise separable alone"),
    ("iso-k7", {"kernel": 7}, "design", "kernel 7 on a DENSE block"),
    ("iso-gelu", {"act": "gelu"}, "optimisation", "GELU alone"),
    ("iso-ln", {"norm": "ln"}, "optimisation", "LayerNorm alone"),
    ("iso-lean-act", {"lean_act": True}, "regime", "one activation per block alone"),
    ("iso-lean-norm", {"lean_norm": True}, "regime", "one normalisation per block alone"),
]
for iid, patch, scope, label in ISOLATED:
    c_i = dict(ISO_BASE)
    c_i.update(patch)
    plan("isolated", iid, c_i, (SEED,), ISO_RECIPE, (ISO_RECIPE, "dense"),
         claim_scope=scope, label=label,
         note="applied on its own to rung 1, so that the cumulative column can be "
              "checked against the sum of the parts")

NOT_ISOLABLE = [
    {"rung": 4, "change": "widen 12 to 18 channels",
     "why": "isolable in principle, but widening a dense block answers a different "
            "question from widening a depthwise one, and the budget went to the "
            "matched-cost pair instead"},
    {"rung": 5, "change": "inverted bottleneck, e=4",
     "why": "an inverted bottleneck is defined by expanding and projecting around a "
            "depthwise convolution. Applied to a dense basic block there is nothing "
            "to invert, so an isolated form would be a different change wearing the "
            "same name"},
    {"rung": 6, "change": "depthwise moved above the expansion",
     "why": "the move only exists once there is an expansion to move across, which "
            "rung 5 is what introduces"},
]

# The kernel sweep sits on rung 6's block, where rung 7 is the k=7 point and
# rung 6 itself is the k=3 point, so only two new runs are needed.
K_BASE, K_RECIPE = ladder_cfg(6)
for k in (5, 9):
    c_k = dict(K_BASE)
    c_k["kernel"] = k
    plan("kernels", f"k{k}", c_k, SEEDS_CORE, K_RECIPE, (K_RECIPE, "modern"),
         claim_scope="design", label=f"depthwise kernel {k}x{k}",
         cut_tag="kernel-k9" if k == 9 else None)


def rewiden(target, quantity):
    """Solve the top rung's width so that it matches the base rung on one
    budget. Equal multiply-accumulates and equal parameters are different
    budgets and they do not give the same width, so the file reports both."""
    best = None
    top, _ = ladder_cfg(LADDER_TOP)
    for w in range(4, 41):
        c_w = dict(top)
        c_w["width"] = w
        _, macs_w, params_w = profile(Net(c_w), c_w["res"])
        got = macs_w if quantity == "macs" else params_w
        err = abs(got / target - 1.0)
        if best is None or err < best[0]:
            best = (err, w, macs_w, params_w)
    return best


rung0 = find("rung00")
MATCH = {}
for quantity, tag, cut in (("macs", "flop-matched", None),
                           ("params", "param-matched", "param-matched")):
    err, w, macs_w, params_w = rewiden(rung0[quantity if quantity == "macs" else "params"],
                                       quantity)
    c_m, rec_m = ladder_cfg(LADDER_TOP)
    c_m["width"] = w
    MATCH[quantity] = {"width": w, "achieved_ratio": rnd(
        (macs_w if quantity == "macs" else params_w) /
        (rung0["macs"] if quantity == "macs" else rung0["params"]), 4), "err": rnd(err, 4)}
    plan("matched", tag, c_m, SEEDS_CORE, rec_m, (rec_m, "modern"),
         claim_scope="design", layer_stack=(quantity == "macs"), cut_tag=cut,
         label=f"modernised block re-widened to C={w}, matched on "
               f"{'multiply-accumulates' if quantity == 'macs' else 'parameters'}",
         note="this matched pair, not the raw ladder endpoint, is the article's "
              "single headline comparison")

# The 2x2 that asks whether the ladder was measuring an optimiser interaction
# all along. Three of its four cells are already on the ladder.
c_ms, _ = ladder_cfg(LADDER_TOP)
plan("recipe", "modern-sgd", c_ms, SEEDS_CORE, "sgd", ("sgd", "modern"),
     claim_scope="optimisation", label="the modernised block trained with SGD",
     note="the fourth cell of the recipe 2x2; the other three are ladder rungs")

# The epochs control. Six epochs favours whatever optimises quickly, and this
# is the run that says whether the ladder's verdict is a stopping artefact.
for n in (0, LADDER_TOP):
    c_e, rec_e = ladder_cfg(n)
    plan("epochs", f"long-rung{n:02d}", c_e, (SEED,), rec_e,
         (rec_e, "dense" if is_dense(c_e) else "modern"), epochs=EPOCHS_LONG,
         claim_scope="control", label=f"rung {n} at {EPOCHS_LONG} epochs")

# The pilot's runner-up rates, run at full settings, so that 'the pilot picked
# the right rate' is a checked statement rather than a hope.
for tag, n, rec_c, key in (("pilot-confirm-dense", 0, "sgd", ("sgd", "dense-runnerup")),
                           ("pilot-confirm-modern", LADDER_TOP, "adamw",
                            ("adamw", "modern-runnerup"))):
    c_p, _ = ladder_cfg(n)
    plan("confirm", tag, c_p, (SEED,), rec_c, key, claim_scope="control",
         label=f"rung {n} at the pilot's runner-up rate")

if RUN_OPTIONAL:
    c_k11 = dict(K_BASE)
    c_k11["kernel"] = 11
    plan("kernels", "k11", c_k11, SEEDS_CORE, K_RECIPE, (K_RECIPE, "modern"),
         claim_scope="design", label="depthwise kernel 11x11", cut_tag="optional-tier")
    for name, df, wf, rf in SIMPLEX[:3]:
        f6 = 6.0
        d6 = f6 if name == "depth" else 1.0
        w6 = f6 ** 0.5 if name == "width" else 1.0
        r6 = f6 ** 0.5 if name == "res" else 1.0
        plan("scaling", f"scale6-{name}", t1_cfg(d6, w6, r6), (SEED,), "sgd",
             ("sgd", "dense"), claim_scope="design",
             label=f"x6 budget spent on {name}", cut_tag="optional-tier")


# ============================================ the projection and the cut order
def pilot_estimate():
    total = 0.0
    for end, n in (("dense", 0), ("modern", LADDER_TOP)):
        c_p, _ = ladder_cfg(n)
        _, macs_p, _ = profile(Net(c_p), c_p["res"])
        for recipe in PILOT_RATES:
            total += estimate_seconds(c_p, macs_p, PILOT_EPOCHS, PILOT_N,
                                      len(PILOT_RATES[recipe]))
    return total


def project():
    by_group, total = {}, 0.0
    for e in PLANNED:
        s = estimate_seconds(e["cfg"], e["macs"], e["epochs"], e["n_train"],
                             len(e["seeds"]))
        e["est_seconds"] = s
        g = by_group.setdefault(e["group"], {"runs": 0, "seconds": 0.0})
        g["runs"] += len(e["seeds"])
        g["seconds"] += s
        total += s
    return by_group, total


def apply_cut(tag):
    global PLANNED
    before = len(PLANNED)
    if tag == "optional-tier":
        PLANNED = [e for e in PLANNED if e["cut_tag"] != "optional-tier"]
    elif tag == "epochs-control":
        PLANNED = [e for e in PLANNED if e["group"] != "epochs"]
    elif tag == "kernel-k9":
        PLANNED = [e for e in PLANNED if e["id"] != "k9"]
    elif tag == "resolution-thin":
        # keep the ends, the native size and the two controls; drop the rest
        # res-14 is the base network and is never in this group; keep the two
        # ends, the native size, the biggest point and both controls
        keep = {"res-10", "res-18", "res-24", "res-28",
                "res-24-restride", "res-24-from-12"}
        PLANNED = [e for e in PLANNED if e["group"] != "resolution" or e["id"] in keep]
    elif tag == "ladder-seed2":
        for e in PLANNED:
            if e["group"] == "ladder" and len(e["seeds"]) == 2:
                e["seeds"] = e["seeds"][:1]
    elif tag == "param-matched":
        PLANNED = [e for e in PLANNED if e["id"] != "param-matched"]
    elif tag == "simplex-seed2":
        for e in PLANNED:
            if e["id"].startswith("scale-"):
                e["seeds"] = e["seeds"][:1]
    elif tag == "simplex-to-4":
        PLANNED = [e for e in PLANNED if e["cut_tag"] != "simplex-to-4"]
    return before - len(PLANNED)


pilot_secs = pilot_estimate()
groups, projected = project()
print("\nM0: the projection, at the constants just measured "
      f"(f_dense {F_DENSE:.2f}, f_modern {F_MODERN:.2f}, LayerNorm x{LN_PENALTY:.2f}):")
for g, v in groups.items():
    print(f"  {g:12s} {v['runs']:3d} runs  {v['seconds'] / 60:6.1f} min")
print(f"  {'lr pilot':12s} {sum(len(v) for v in PILOT_RATES.values()) * 2:3d} runs  "
      f"{pilot_secs / 60:6.1f} min")
print(f"  {'TOTAL':12s} {sum(v['runs'] for v in groups.values()):3d} runs  "
      f"{(projected + pilot_secs) / 60:6.1f} min")

cuts_applied = []
for tag, description in CUT_ORDER:
    total_now = projected + pilot_secs
    # The optional tier goes as soon as the soft budget is passed. Anything a
    # chart depends on is only cut once the hard budget is passed as well.
    threshold = TIME_BUDGET_S if tag == "optional-tier" else HARD_BUDGET_S
    if total_now <= threshold:
        continue
    removed = apply_cut(tag)
    groups, projected = project()
    cuts_applied.append({"cut": tag, "description": description,
                         "runs_removed": removed,
                         "projection_after_min": rnd((projected + pilot_secs) / 60, 1)})
    print(f"  over the {threshold / 60:.0f} minute mark: cutting {description} "
          f"-> {(projected + pilot_secs) / 60:.1f} min")
if not cuts_applied:
    print(f"  inside the {TIME_BUDGET_S / 60:.0f} minute budget, nothing cut")
elif (projected + pilot_secs) > HARD_BUDGET_S:
    print(f"  the whole cut order is spent and the projection is still "
          f"{(projected + pilot_secs) / 60:.1f} min. Running it anyway: the cache makes "
          f"an overrun resumable, and cutting further would start removing evidence "
          f"the article's claims rest on.")
PROJECTION = {"minutes": rnd((projected + pilot_secs) / 60, 1),
              "budget_minutes": TIME_BUDGET_S / 60,
              "hard_budget_minutes": HARD_BUDGET_S / 60,
              "still_over_budget_after_cuts": bool((projected + pilot_secs) > HARD_BUDGET_S),
              "by_group": {g: {"runs": v["runs"], "minutes": rnd(v["seconds"] / 60, 1)}
                           for g, v in groups.items()},
              "pilot_minutes": rnd(pilot_secs / 60, 1),
              "cut_order": [{"cut": t, "description": d} for t, d in CUT_ORDER],
              "cuts_applied": cuts_applied}


# ============================================== M2, the learning-rate pilot
# Three rates per recipe, at BOTH ends of the ladder. Tuning one recipe at one
# end and applying the winner everywhere is the confound that would make
# 'AdamW beats SGD' and 'the modernised block wins' unfalsifiable, so the
# sweep is run four times and the whole thing is exported, not just the
# winner. Judged on a held-out half of the frozen training set, never on the
# test split the article reports.
_rs = np.random.RandomState(SEED)
_pi, _vi = [], []
for _cls in range(10):
    _pool = np.where(ytr_np == _cls)[0]
    _perm = _rs.permutation(_pool)
    _h = len(_pool) // 2
    _pi.append(_perm[:_h])
    _vi.append(_perm[_h:])
pilot_idx = np.sort(np.concatenate(_pi))
val_idx = np.sort(np.concatenate(_vi))
Xp, yp = Xtr_native[pilot_idx], ytr[pilot_idx]
Xv, yv = Xtr_native[val_idx], ytr[val_idx]
print(f"\nM2: the learning-rate pilot, {len(yp)} train / {len(yv)} validation, "
      f"{PILOT_EPOCHS} epochs, both ends of the ladder, both recipes:")

pilot_rows = []
LR = {}
for end, rung in (("dense", 0), ("modern", LADDER_TOP)):
    c_p, _ = ladder_cfg(rung)
    for recipe, rates in PILOT_RATES.items():
        scored = []
        for lr in rates:
            rec, hit = run_once(f"pilot-{end}-{recipe}", c_p, SEED, recipe, lr,
                                PILOT_EPOCHS, Xp, yp, Xv, yv)
            row = {"end": end, "rung": rung, "recipe": recipe, "lr": lr,
                   "val_acc": rec["test_acc"], "train_acc": rec["train_acc"],
                   "final_loss": rec["history"][-1], "seconds": rec["seconds"]}
            pilot_rows.append(row)
            scored.append(row)
            print(f"  {end:6s} {recipe:5s} lr {lr:<7g} validation {row['val_acc']:.4f}  "
                  f"loss {row['final_loss']:.4f}{'  [cached]' if hit else ''}", flush=True)
        scored.sort(key=lambda r: -r["val_acc"])
        LR[(recipe, end)] = scored[0]["lr"]
        LR[(recipe, end + "-runnerup")] = scored[1]["lr"]
        print(f"    -> {recipe} at the {end} end: {scored[0]['lr']:g} "
              f"(runner-up {scored[1]['lr']:g}, margin "
              f"{(scored[0]['val_acc'] - scored[1]['val_acc']) * 100:.2f} points)")

PILOT_CHOSEN = {f"{r}/{e}": v for (r, e), v in LR.items()}

# ==================================================== running the schedule
print(f"\nrunning {sum(len(e['seeds']) for e in PLANNED)} trained models "
      f"({len(PLANNED)} configurations):")
WALL0 = time.perf_counter()


def build_run(e):
    lr = LR[e["lr_key"]]
    c = e["cfg"]
    Xa, Xb = data_at(c["res"], c["res_via"])
    recs, cached_n = [], 0
    for seed in e["seeds"]:
        rec, hit = run_once(e["id"], c, seed, e["recipe"], lr, e["epochs"],
                            Xa, ytr, Xb, yte,
                            extract=e["extract"] if seed == SEED else None)
        recs.append(rec)
        cached_n += int(hit)
    te = [r["test_acc"] for r in recs]
    tr = [r["train_acc"] for r in recs]
    te_m, te_sd = mean_sd(te)
    tr_m, _ = mean_sd(tr)
    per_class = np.nanmean(np.array([r["per_class"] for r in recs], dtype=float), axis=0)
    last_two = [rnd(r["history"][-3] - r["history"][-1], 5) for r in recs
                if len(r["history"]) >= 3]
    to_thresh = [epochs_to_threshold(r["history"]) for r in recs]
    c_channels = [c["width"] * (2 ** s) for s in range(len(c["blocks"]))]
    run = {
        "id": e["id"], "group": e["group"], "label": e["label"], "note": e["note"],
        "family": c["block"], "block": c["block"], "blocks": list(c["blocks"]),
        "width": c["width"], "channels": c_channels, "res": c["res"],
        "res_via": c["res_via"], "resize_filter": resize_label(c["res"], c["res_via"]),
        "stem": c["stem"], "kernel": c["kernel"], "act": c["act"], "norm": c["norm"],
        "dw_pos": c["dw_pos"] if c["block"] == "invert" else None,
        "expand": c["expand"] if c["block"] == "invert" else None,
        "lean_act": c["lean_act"], "lean_norm": c["lean_norm"],
        "block_ops": conv_ops(e["ops"]),
        "layer_stack": ([{k: v for k, v in o.items() if k != "shortcut"} for o in e["ops"]]
                        if e["layer_stack"] else None),
        "params": e["params"], "macs": e["macs"],
        "macs_analytic": (int(round(analytic_macs(c))) if analytic_macs(c) else None),
        "macs_rel_err": (rnd((e["macs"] - analytic_macs(c)) / analytic_macs(c), 5)
                         if analytic_macs(c) else None),
        "receptive_field": e["rf"], "total_stride": e["total_stride"],
        "recipe": e["recipe"], "lr": lr, "epochs": e["epochs"],
        "n_train": e["n_train"], "seeds": list(e["seeds"]),
        "test_acc": te, "test_acc_mean": rnd(te_m, 4), "test_acc_sd": rnd(te_sd, 4),
        "train_acc": tr, "train_acc_mean": rnd(tr_m, 4),
        "per_class_acc": [rnd(v, 4) for v in per_class],
        "history": [r["history"] for r in recs],
        "last_two_epochs": rnd(float(np.mean(last_two)), 5) if last_two else None,
        "epochs_to_loss_threshold": ([None if v is None else int(v) for v in to_thresh]),
        "loss_threshold": LOSS_THRESHOLD,
        "seconds": [r["seconds"] for r in recs],
        "seconds_mean": rnd(float(np.mean([r["seconds"] for r in recs])), 1),
        "claim_scope": e["claim_scope"],
    }
    if recs and "extra" in recs[0]:
        run["extra"] = recs[0]["extra"]
    return run, cached_n


for e in PLANNED:
    run, cached_n = build_run(e)
    RESULTS[e["id"]] = run
    flag = "  [cached]" if cached_n == len(e["seeds"]) else ""
    print(f"  [{e['group']:10s}] {e['id']:18s} {run['macs'] / 1e6:6.2f} MMAC  "
          f"{run['params']:7,} par  test {run['test_acc_mean']:.4f}"
          f" +/- {run['test_acc_sd']:.4f}  train {run['train_acc_mean']:.4f}  "
          f"{run['seconds_mean']:5.0f}s{flag}", flush=True)
WALL = time.perf_counter() - WALL0
print(f"  {WALL / 60:.1f} minutes of training in this session "
      f"(projected {PROJECTION['minutes']:.1f})")


def R(run_id):
    return RESULTS[run_id]


# ================================== M3, the noise floor and the decision rule
# Pre-registered before any result above was looked at: sigma is measured at
# three points spanning the capacity range rather than at one, and it is
# modelled as a function of model size rather than pooled, because seed spread
# grows as networks shrink and a single band drawn across a sevenfold capacity
# range is too narrow exactly where the article most wants to claim
# 'within noise'.
sigma_points = []
for rid, run in RESULTS.items():
    if len(run["seeds"]) >= 2:
        sigma_points.append({"id": rid, "macs": run["macs"], "n": len(run["seeds"]),
                             "sd": run["test_acc_sd"], "df": len(run["seeds"]) - 1})
anchors = [p for p in sigma_points if p["id"] in ("rung00", "rung04",
                                                  f"rung{LADDER_TOP:02d}")]
X_fit = np.array([[1.0, 1.0 / math.sqrt(p["macs"] / 1e6)] for p in sigma_points])
y_fit = np.array([p["sd"] for p in sigma_points])
w_fit = np.sqrt(np.array([p["df"] for p in sigma_points], dtype=float))
coef, *_ = np.linalg.lstsq(X_fit * w_fit[:, None], y_fit * w_fit, rcond=None)
pred = X_fit @ coef
flat = float(np.average(y_fit, weights=w_fit ** 2))
sse_fit = float(np.sum((w_fit * (y_fit - pred)) ** 2))
sse_flat = float(np.sum((w_fit * (y_fit - flat)) ** 2))
r2 = 1.0 - sse_fit / sse_flat if sse_flat > 0 else 0.0
USE_SIZE_MODEL = bool(r2 > 0.05 and coef[0] >= 0 and coef[1] >= 0)
DF_POOLED = int(sum(p["df"] for p in sigma_points))

T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
           15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
           21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
           27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def tcrit(df):
    """Two-sided 95% point of Student's t. Two seeds cannot estimate a
    standard deviation, so the threshold is never 2 sigma by default."""
    if df <= 0:
        return float("inf")
    return T_TABLE.get(int(df), 1.96 + 2.4 / max(1.0, float(df)))


def sigma_at(macs):
    if not USE_SIZE_MODEL:
        return flat
    return float(max(1e-4, coef[0] + coef[1] / math.sqrt(macs / 1e6)))


def verdict(a, b):
    """a minus b, with a threshold that depends on both models' sizes and on
    how many seeds each of them actually got."""
    ra, rb = R(a), R(b)
    sa, sb = sigma_at(ra["macs"]), sigma_at(rb["macs"])
    sd = math.sqrt(sa ** 2 / len(ra["seeds"]) + sb ** 2 / len(rb["seeds"]))
    delta = ra["test_acc_mean"] - rb["test_acc_mean"]
    thr = tcrit(DF_POOLED) * sd
    return {"delta": rnd(delta, 4), "sigma_delta": rnd(sd, 5),
            "threshold_used": rnd(thr, 5), "df": DF_POOLED,
            "verdict": "real gain" if delta > thr else
                       ("real loss" if delta < -thr else "within noise")}


NOISE_FLOOR = {
    "anchors": anchors, "all_points": sigma_points,
    "model": "sd = a + b / sqrt(MMAC)",
    "a": rnd(coef[0], 5), "b": rnd(coef[1], 5), "r2_against_flat_pool": rnd(r2, 4),
    "flat_pool_sd": rnd(flat, 5), "used": "size model" if USE_SIZE_MODEL else "flat pool",
    "pooled_df": DF_POOLED, "t_critical": rnd(tcrit(DF_POOLED), 4),
    "residuals": [{"id": p["id"], "macs": p["macs"], "sd": p["sd"],
                   "fitted": rnd(float(pred[i]), 5)} for i, p in enumerate(sigma_points)],
    # Composed from the seed counts that were actually used, because this note
    # named specific numbers and the seed lists above are a tuning knob: the
    # sentence went stale the first time the schedule was trimmed.
    "note": ("three capacity anchors, with "
             + ", ".join(f"{len(LADDER_SEEDS[k])} seeds at rung {k}"
                         for k in sorted(LADDER_SEEDS))
             + f", plus the {len(SEEDS_CORE)}-seed spread of every other "
               "configuration. If the size model does not beat a flat pool the flat "
               "pool is used and this field says so."),
}
print(f"\nM3: noise floor, {NOISE_FLOOR['used']}, r2 against a flat pool "
      f"{r2:.3f}, pooled df {DF_POOLED}, t {tcrit(DF_POOLED):.3f}")
for p in anchors:
    print(f"  anchor {p['id']:8s} {p['macs'] / 1e6:5.2f} MMAC  {p['n']} seeds  "
          f"sd {p['sd']:.4f}  fitted {sigma_at(p['macs']):.4f}")


# ================================================ thread 1, what scaling buys
base = R("s0-base")
arms = []
for name, df, wf, rf in SIMPLEX:
    rid = f"scale-{name}"
    if rid not in RESULTS:
        continue
    run = R(rid)
    lg = math.log(BUDGET_TARGET)
    share_d = math.log(sum(run["blocks"]) / sum(base["blocks"])) / lg
    share_w = 2 * math.log(run["width"] / base["width"]) / lg
    share_r = 2 * math.log(run["res"] / base["res"]) / lg
    tot = share_d + share_w + share_r
    arms.append({"axis": name, "id": rid,
                 "requested": {"depth": rnd(df, 4), "width": rnd(wf, 4), "res": rnd(rf, 4)},
                 "blocks": run["blocks"], "width": run["width"], "res": run["res"],
                 "macs": run["macs"], "params": run["params"],
                 "achieved_budget": rnd(run["macs"] / base["macs"], 4),
                 "ternary": [rnd(share_d / tot, 4), rnd(share_w / tot, 4),
                             rnd(share_r / tot, 4)] if tot > 0 else [0, 0, 0],
                 "test_acc_mean": run["test_acc_mean"], "test_acc_sd": run["test_acc_sd"],
                 "seconds_mean": run["seconds_mean"],
                 "against_base": verdict(rid, "s0-base")})
budget_spread = (max(a["achieved_budget"] for a in arms) /
                 min(a["achieved_budget"] for a in arms)) if arms else 1.0
best_arm = max(arms, key=lambda a: a["test_acc_mean"]) if arms else None
print(f"\nT1: the allocation simplex at a x{BUDGET_TARGET:g} budget "
      f"(achieved budgets span x{budget_spread:.2f}):")
for a in sorted(arms, key=lambda a: -a["test_acc_mean"]):
    print(f"  {a['axis']:14s} blocks {str(a['blocks']):11s} C={a['width']:2d} R={a['res']:2d}  "
          f"{a['macs'] / 1e6:5.2f} MMAC (x{a['achieved_budget']:.2f})  "
          f"test {a['test_acc_mean']:.4f}  {a['against_base']['verdict']}")

pool = ["s0-base"] + [a["id"] for a in arms]


def ranking(key):
    def score(rid):
        run = R(rid)
        gain = run["test_acc_mean"] - 0.10          # chance-corrected
        if key == "accuracy":
            return run["test_acc_mean"]
        if key == "mmac":
            return gain / (run["macs"] / 1e6)
        if key == "kparams":
            return gain / (run["params"] / 1000.0)
        return gain / max(1e-6, run["seconds_mean"])
    return [{"id": rid, "score": rnd(score(rid), 6), "test_acc": R(rid)["test_acc_mean"]}
            for rid in sorted(pool, key=lambda r: -score(r))]


BUDGET_RANKINGS = {k: ranking(k) for k in ("accuracy", "mmac", "kparams", "seconds")}
print("  three budgets, three winners: best raw "
      f"{BUDGET_RANKINGS['accuracy'][0]['id']}, best per MMAC "
      f"{BUDGET_RANKINGS['mmac'][0]['id']}, best per parameter "
      f"{BUDGET_RANKINGS['kparams'][0]['id']}, best per second "
      f"{BUDGET_RANKINGS['seconds'][0]['id']}")

res_rows = []
for r in RES_LADDER:
    rid = "s0-base" if r == T1_RES else f"res-{r}"
    if rid not in RESULTS:
        continue
    run = R(rid)
    res_rows.append({"res": r, "id": rid, "macs": run["macs"], "params": run["params"],
                     "test_acc_mean": run["test_acc_mean"],
                     "test_acc_sd": run["test_acc_sd"],
                     "train_acc_mean": run["train_acc_mean"],
                     "per_class_acc": run["per_class_acc"],
                     "receptive_field": run["receptive_field"],
                     "rf_over_image": rnd(run["receptive_field"] / r, 4),
                     "rf_covers_whole_image": bool(run["receptive_field"] >= r),
                     "resize_filter": run["resize_filter"],
                     "seconds_mean": run["seconds_mean"]})
print("\nT1: the resolution ladder (train and test resized by the same operation, "
      "antialiased downwards):")
for row in res_rows:
    print(f"  {row['res']:2d}x{row['res']:<2d} {row['macs'] / 1e6:5.2f} MMAC  "
          f"test {row['test_acc_mean']:.4f}  receptive field {row['receptive_field']} px, "
          f"x{row['rf_over_image']:.1f} the image")

HARD4 = [0, 2, 4, 6]      # t-shirt, pullover, coat, shirt
EASY6 = [1, 3, 5, 7, 8, 9]
per_class_trend = []
for row in res_rows:
    pc = row["per_class_acc"]
    per_class_trend.append({"res": row["res"],
                            "hard4": rnd(float(np.mean([pc[i] for i in HARD4])), 4),
                            "easy6": rnd(float(np.mean([pc[i] for i in EASY6])), 4)})

INFORMATION = None
if "res-24-from-12" in RESULTS and "res-24" in RESULTS:
    INFORMATION = {
        "arms": [{"id": "res-24", "label": "24 pixels, 24 pixels of information",
                  "test_acc_mean": R("res-24")["test_acc_mean"]},
                 {"id": "res-24-from-12", "label": "24 pixels, 12 pixels of information",
                  "test_acc_mean": R("res-24-from-12")["test_acc_mean"]}],
        "contrast": verdict("res-24", "res-24-from-12"),
        "why_no_third_arm":
            "a network trained on true 12x12 images would differ from these two in "
            "architecture as well as in information: its final stage map is 3x3 "
            "rather than 6x6 and it costs a quarter as much. Adding it as a third "
            "arm would make the split between 'more information' and 'more spatial "
            "budget' unidentified, so the file reports only the contrast that is "
            "clean, and that contrast answers only the information question.",
    }
    print(f"  information control: {R('res-24')['test_acc_mean']:.4f} at 24 pixels "
          f"against {R('res-24-from-12')['test_acc_mean']:.4f} at 24 pixels rebuilt "
          f"from 12, {INFORMATION['contrast']['verdict']}")
RESTRIDE = None
if "res-24-restride" in RESULTS and "res-24" in RESULTS:
    RESTRIDE = {"id": "res-24-restride", "against": "res-24",
                "contrast": verdict("res-24-restride", "res-24"),
                "macs_ratio": rnd(R("res-24-restride")["macs"] / R("res-24")["macs"], 4),
                "rf_over_image_restride": rnd(R("res-24-restride")["receptive_field"] / 24, 4),
                "rf_over_image_plain": rnd(R("res-24")["receptive_field"] / 24, 4),
                "why": "raising the input resolution while holding the stride schedule "
                       "shrinks the receptive field as a fraction of the image. If the "
                       "restrided network matches or beats the plain one at a fraction "
                       "of the cost, resolution scaling as usually performed is partly "
                       "sabotaging itself."}
    print(f"  receptive-field control: restrided 24 scores "
          f"{R('res-24-restride')['test_acc_mean']:.4f} at "
          f"{RESTRIDE['macs_ratio'] * 100:.0f}% of the cost, "
          f"{RESTRIDE['contrast']['verdict']}")

# =========================================== thread 2, the modernisation ladder
print("\nT2: the modernisation ladder, one change per rung:")
rungs = []
for n, change, scope in RUNG_META:
    rid = f"rung{n:02d}"
    run = R(rid)
    row = {"n": n, "id": rid, "change": change, "claim_scope": scope,
           "macs": run["macs"], "params": run["params"],
           "receptive_field": run["receptive_field"],
           "test_acc_mean": run["test_acc_mean"], "test_acc_sd": run["test_acc_sd"],
           "train_acc_mean": run["train_acc_mean"], "seeds": run["seeds"],
           "recipe": run["recipe"], "lr": run["lr"],
           "seconds_mean": run["seconds_mean"],
           "last_two_epochs": run["last_two_epochs"],
           "epochs_to_loss_threshold": run["epochs_to_loss_threshold"],
           "bundled": bool(n == 9), "note": run["note"]}
    if n > 0:
        row["against_previous"] = verdict(rid, f"rung{n - 1:02d}")
    row["against_rung0"] = verdict(rid, "rung00")
    rungs.append(row)
    step = row.get("against_previous")
    print(f"  rung {n:2d} {change:44s} {run['macs'] / 1e6:5.2f} MMAC  "
          f"test {run['test_acc_mean']:.4f}" +
          (f"  step {step['delta'] * 100:+.2f} pts, {step['verdict']}" if step else ""))

cleared = [r for r in rungs if r.get("against_previous", {}).get("verdict") == "real gain"]
lost = [r for r in rungs if r.get("against_previous", {}).get("verdict") == "real loss"]
end_to_end = verdict(f"rung{LADDER_TOP:02d}", "rung00")
MULTIPLICITY = {
    "n_step_tests": len(rungs) - 1,
    "alpha": 0.05,
    "expected_false_positives": rnd(0.05 * (len(rungs) - 1), 3),
    "independent": False,
    "note": "the rungs are cumulative, so rung k's mean enters two of these tests and "
            "consecutive deltas are negatively correlated. The tests are not "
            "independent and the expected count above is not a Bonferroni budget; it "
            "is the reminder that " + str(len(rungs)) + " rungs judged at a 5% "
            "threshold produce roughly " + f"{0.05 * (len(rungs) - 1):.1f}" +
            " false verdicts by chance alone.",
}
print(f"  end to end: {end_to_end['delta'] * 100:+.2f} points, {end_to_end['verdict']}; "
      f"{len(cleared)} rungs cleared their own threshold, {len(lost)} lost ground, "
      f"expected false positives {MULTIPLICITY['expected_false_positives']}")

MATCHED = {}
for quantity, tag in (("macs", "flop-matched"), ("params", "param-matched")):
    if tag not in RESULTS:
        continue
    run = R(tag)
    MATCHED[tag] = {"id": tag, "width": run["width"], "against": "rung00",
                    "macs": run["macs"], "params": run["params"],
                    "macs_ratio": rnd(run["macs"] / R("rung00")["macs"], 4),
                    "params_ratio": rnd(run["params"] / R("rung00")["params"], 4),
                    "test_acc_mean": run["test_acc_mean"],
                    "contrast": verdict(tag, "rung00"),
                    "solved_for": MATCH[quantity]}
    print(f"  {tag:14s} C={run['width']:2d}  {run['macs'] / 1e6:5.2f} MMAC "
          f"(x{MATCHED[tag]['macs_ratio']:.2f} of rung 0), {run['params']:,} params "
          f"(x{MATCHED[tag]['params_ratio']:.2f})  test {run['test_acc_mean']:.4f}  "
          f"{MATCHED[tag]['contrast']['verdict']}")

isolated = []
for iid, patch, scope, label in ISOLATED:
    if iid not in RESULTS:
        continue
    run = R(iid)
    isolated.append({"id": iid, "label": label, "claim_scope": scope,
                     "changed": list(patch.keys())[0], "macs": run["macs"],
                     "params": run["params"], "test_acc_mean": run["test_acc_mean"],
                     "against_rung1": verdict(iid, "rung01")})
sum_iso = sum(i["against_rung1"]["delta"] for i in isolated)
cumulative_total = R(f"rung{LADDER_TOP:02d}")["test_acc_mean"] - R("rung01")["test_acc_mean"]
iso_sigma = math.sqrt(sum(i["against_rung1"]["sigma_delta"] ** 2 for i in isolated))
ISOLATED_BLOCK = {
    "runs": isolated,
    "sum_isolated_deltas": rnd(sum_iso, 4),
    "cumulative_total_delta": rnd(cumulative_total, 4),
    "disagreement": rnd(cumulative_total - sum_iso, 4),
    "disagreement_sigma": rnd(iso_sigma, 5),
    "disagreement_exceeds_noise": bool(abs(cumulative_total - sum_iso) >
                                       tcrit(DF_POOLED) * iso_sigma),
    "note": f"{len(isolated)} of the ladder's changes applied singly to rung 1. The gap "
            "between the sum of the parts and the whole is the interaction the "
            "cumulative column hides. The changes with no isolated form at all are "
            "listed with their reasons rather than invented around.",
}
print(f"  isolated ablations: parts sum to {sum_iso * 100:+.2f} points, the cumulative "
      f"ladder moved {cumulative_total * 100:+.2f}, disagreement "
      f"{(cumulative_total - sum_iso) * 100:+.2f} points"
      f" ({'outside' if ISOLATED_BLOCK['disagreement_exceeds_noise'] else 'inside'} noise)")

# ------------------------------------------------- the kernel, as arithmetic
kernel_rows = []
for k in (3, 5, 7, 9, 11):
    rid = {3: "rung06", 7: "rung07"}.get(k, f"k{k}")
    if rid not in RESULTS:
        continue
    run = R(rid)
    dwm = sum(o["macs"] for o in run["block_ops"] if o.get("depthwise"))
    kernel_rows.append({"k": k, "id": rid, "macs": run["macs"],
                        "depthwise_macs": int(dwm),
                        "depthwise_share": rnd(dwm / run["macs"], 4),
                        "receptive_field": run["receptive_field"],
                        "test_acc_mean": run["test_acc_mean"],
                        "test_acc_sd": run["test_acc_sd"],
                        "against_k3": verdict(rid, "rung06") if rid != "rung06" else None,
                        "first_stage_map": LADDER_RES // 2,
                        "kernel_covers_first_stage": rnd(
                            min(1.0, (k * k) / (LADDER_RES // 2) ** 2), 4)})
KERNELS = {
    "rows": kernel_rows,
    "parity_k": 5,
    "parity_derivation": "the basic block this one replaces stacks two 3x3 convolutions, "
                         "so it grows the receptive field by 2 x (3 - 1) = 4 per block. "
                         "A single depthwise kernel matches that growth when k - 1 = 4, "
                         "that is at k = 5.",
    "parity_is_not_an_optimum_claim": True,
    "why": "the arithmetic licenses one statement and only one: k = 5 restores "
           "receptive-field parity with the two-3x3 block. It does not predict that "
           "accuracy peaks at 5. The block being replaced also carries two weight "
           "matrices and two nonlinearities that a single depthwise kernel does not, "
           "so parity on one proxy is not parity on the others. A measured peak away "
           "from 5 refutes nothing, because parity was the only claim.",
}
print("\nT2: the kernel, reported as receptive-field arithmetic and not as a prediction "
      f"(parity sits at k={KERNELS['parity_k']}):")
for row in kernel_rows:
    print(f"  k={row['k']:2d}  {row['macs'] / 1e6:5.2f} MMAC, depthwise is "
          f"{row['depthwise_share'] * 100:4.1f}% of it  receptive field "
          f"{row['receptive_field']:3d}  test {row['test_acc_mean']:.4f}")

# -------------------------------------------- block against recipe, a full 2x2
RECIPE_2x2 = None
if "modern-sgd" in RESULTS:
    cells = [{"block": "ResNet basic", "recipe": "sgd", "id": "rung00"},
             {"block": "ResNet basic", "recipe": "adamw", "id": "rung01"},
             {"block": "modernised", "recipe": "sgd", "id": "modern-sgd"},
             {"block": "modernised", "recipe": "adamw", "id": f"rung{LADDER_TOP:02d}"}]
    for cell in cells:
        run = R(cell["id"])
        cell.update(lr=run["lr"], test_acc_mean=run["test_acc_mean"],
                    test_acc_sd=run["test_acc_sd"], macs=run["macs"])
    gap_adamw = verdict(f"rung{LADDER_TOP:02d}", "rung01")
    gap_sgd = verdict("modern-sgd", "rung00")
    RECIPE_2x2 = {"cells": cells, "modern_gain_under_adamw": gap_adamw,
                  "modern_gain_under_sgd": gap_sgd,
                  "interaction": rnd(gap_adamw["delta"] - gap_sgd["delta"], 4),
                  "note": "if the modernised block's advantage is present under AdamW and "
                          "absent under SGD, the ladder was measuring an optimiser "
                          "interaction throughout, and every rung above the first is "
                          "reporting that interaction rather than a block design."}
    print(f"\nT2: recipe 2x2, the modernised block gains "
          f"{gap_adamw['delta'] * 100:+.2f} points under AdamW and "
          f"{gap_sgd['delta'] * 100:+.2f} under SGD "
          f"(interaction {RECIPE_2x2['interaction'] * 100:+.2f})")

EPOCHS_CONTROL = None
if "long-rung00" in RESULTS and f"long-rung{LADDER_TOP:02d}" in RESULTS:
    short0, shortT = R("rung00")["test_acc"][0], R(f"rung{LADDER_TOP:02d}")["test_acc"][0]
    long0, longT = R("long-rung00")["test_acc"][0], R(f"long-rung{LADDER_TOP:02d}")["test_acc"][0]
    EPOCHS_CONTROL = {
        "epochs_short": EPOCHS, "epochs_long": EPOCHS_LONG, "seed": SEED,
        "rows": [{"rung": 0, "short": short0, "long": long0,
                  "id_long": "long-rung00"},
                 {"rung": LADDER_TOP, "short": shortT, "long": longT,
                  "id_long": f"long-rung{LADDER_TOP:02d}"}],
        "ranking_flipped": bool((shortT > short0) != (longT > long0)),
        "note": "six epochs favours whatever optimises quickly. Every run in this file "
                "also exports the improvement over its last two epochs and the epoch at "
                "which it first reached a training loss of "
                f"{LOSS_THRESHOLD}, so a reader can see which runs were still moving.",
    }
    print(f"T2: epochs control, rung 0 {short0:.4f} -> {long0:.4f} and rung "
          f"{LADDER_TOP} {shortT:.4f} -> {longT:.4f} at {EPOCHS_LONG} epochs; "
          f"ranking {'FLIPPED' if EPOCHS_CONTROL['ranking_flipped'] else 'held'}")

PILOT_CONFIRM = []
for tag, rung, key in (("pilot-confirm-dense", 0, ("sgd", "dense")),
                       ("pilot-confirm-modern", LADDER_TOP, ("adamw", "modern"))):
    if tag not in RESULTS:
        continue
    chosen = R(f"rung{rung:02d}")
    runner = R(tag)
    PILOT_CONFIRM.append({"id": tag, "rung": rung, "chosen_lr": chosen["lr"],
                          "runnerup_lr": runner["lr"],
                          "chosen_acc": chosen["test_acc"][0],
                          "runnerup_acc": runner["test_acc"][0],
                          "pilot_ranking_held": bool(chosen["test_acc"][0] >=
                                                     runner["test_acc"][0])})
PILOT = {
    "rates": {k: list(v) for k, v in PILOT_RATES.items()},
    "epochs": PILOT_EPOCHS, "n_train": int(len(yp)), "n_val": int(len(yv)),
    "judged_on": "a held-out, class-stratified half of the frozen training set, never "
                 "the test split the article reports",
    "runs": pilot_rows, "chosen": PILOT_CHOSEN, "confirm": PILOT_CONFIRM,
    "disagreement": {
        "same_lr_at_both_ends": {r: bool(LR[(r, "dense")] == LR[(r, "modern")])
                                 for r in PILOT_RATES},
        "any_confirm_failed": bool(any(not c["pilot_ranking_held"] for c in PILOT_CONFIRM)),
        "note": "the pilot ranks rates after four epochs on half the data; the ladder "
                "runs six epochs on all of it. The confirm rows re-run the runner-up "
                "rate at full settings at both ends, so the claim that the pilot picked "
                "the right rate is checked rather than assumed.",
    },
}


# ==================================== thread 3, separable arithmetic and time
# The closed form first, because it is exact and free. A dense convolution
# costs k*k*Cin*Cout per position; a depthwise one followed by a pointwise one
# costs k*k*Cin + Cin*Cout. The ratio is 1/Cout + 1/k^2, so as the network
# gets wider the ratio does not go to zero: it goes to 1/k^2. A 3x3 separable
# convolution can never be more than nine times cheaper than a dense one,
# however many channels are thrown at it. The saving is capped by the kernel.
SEP_K = (1, 3, 5, 7, 9, 11)
SEP_CH = (16, 32, 64, 128, 256, 512)
sep_rows = []
for k in SEP_K:
    for ch in SEP_CH:
        dense = k * k * ch * ch
        sep = k * k * ch + ch * ch
        sep_rows.append({"k": k, "cin": ch, "cout": ch, "dense": dense, "separable": sep,
                         "ratio": rnd(sep / dense, 6), "limit_1_over_k2": rnd(1.0 / (k * k), 6),
                         "one_over_cout": rnd(1.0 / ch, 6),
                         "gap_to_limit": rnd(sep / dense - 1.0 / (k * k), 6)})
SEPARABLE_CLOSED = {
    "dense": "k*k*Cin*Cout per output position",
    "separable": "k*k*Cin + Cin*Cout per output position",
    "ratio": "1/Cout + 1/k^2",
    "limit": "1/k^2 as Cout grows; 1/Cout + 1/k^2 is the exact ratio, 1/k^2 is only "
             "its limit, and confusing the two is what produces the claim that a "
             "separable convolution gets arbitrarily cheap",
    "headline": "a 3x3 separable convolution can never be more than nine times cheaper "
                "than a dense one, no matter how wide the network gets",
    "rows": sep_rows,
}

stopwatch = []
for threads in MICRO_THREADS:
    for shape, ch, s in MICRO_SHAPES:
        d = micro_get(threads, shape, "dense3")
        dw = micro_get(threads, shape, "dw3")
        pw = micro_get(threads, shape, "pw_same")
        if not (d and dw and pw):
            continue
        stopwatch.append({
            "threads": threads, "shape": shape, "channels": ch, "spatial": s,
            "dense_macs": d["macs"], "separable_macs": dw["macs"] + pw["macs"],
            "mac_ratio": rnd((dw["macs"] + pw["macs"]) / d["macs"], 4),
            "dense_ms": d["ms"], "dw_ms": dw["ms"], "pw_ms": pw["ms"],
            "time_ratio": rnd((dw["ms"] + pw["ms"]) / d["ms"], 4),
            "dense_gmac_s": d["gmac_s"],
            "separable_gmac_s": rnd((dw["macs"] + pw["macs"]) /
                                    ((dw["ms"] + pw["ms"]) / 1000.0) / 1e9, 3),
            "arithmetic_intensity_ratio": rnd(
                ((dw["macs"] + pw["macs"]) / d["macs"]) /
                max(1e-9, (dw["ms"] + pw["ms"]) / d["ms"]), 4)})
tr_by_shape = {}
for row in stopwatch:
    tr_by_shape.setdefault(row["shape"], []).append(row["time_ratio"])
time_ratio_spread = {k: rnd(max(v) / min(v), 3) for k, v in tr_by_shape.items() if min(v) > 0}
SEPARABLE = {
    "closed_form": SEPARABLE_CLOSED,
    "stopwatch": stopwatch,
    "time_ratio_spread_across_thread_counts": time_ratio_spread,
    "finding": "the multiply-accumulate saving and the wall-clock saving disagree by a "
               "large factor. How large, this machine cannot pin down: the same "
               "separable-against-dense time ratio moves by up to "
               f"x{max(time_ratio_spread.values()) if time_ratio_spread else 1:.2f} "
               "purely by changing the thread count. The article may state that the "
               "two disagree by a lot; it may not state a specific slowdown multiple.",
    "precision_note": PRECISION_NOTE,
}
print("\nT3: separable arithmetic against the clock "
      f"(the time ratio itself moves by up to x"
      f"{max(time_ratio_spread.values()) if time_ratio_spread else 1:.2f} across thread counts):")
for row in stopwatch:
    if row["threads"] == THREADS:
        print(f"  {row['shape']} C={row['channels']:2d} S={row['spatial']:2d}  "
              f"MACs x{row['mac_ratio']:.3f}  wall clock x{row['time_ratio']:.2f}")

OPS = {"rows": micro_rows, "thread_counts": list(MICRO_THREADS),
       "reps": MICRO_REPS, "warmups": MICRO_WARM, "rounds": MICRO_ROUNDS,
       "batch": MICRO_BATCH, "input_gradients": True,
       "spread_by_op": thread_spread, "worst_spread": rnd(worst_spread, 3),
       "precision_note": PRECISION_NOTE,
       "method": "median of 15 timed repetitions after 4 warm-ups, three interleaved "
                 "rounds, each operation reported at its fastest round, forward and "
                 "backward with requires_grad on the input so that the gradient with "
                 "respect to the activations is computed the way training computes it. "
                 "A benchmark that only asks for weight gradients leaves out the part "
                 "whose cost profile differs most between grouped and dense "
                 "convolutions."}

# ================================ what the same ideas cost at the paper's scale
# Transcribed layer specifications, counted by our own pass. No accuracies, no
# training: the counter validates our transcription, not the architecture, and
# the article must say so wherever these numbers appear.
def count_resnet50(res=224):
    ops, params = [], 0
    macs = 49 * 3 * 64 * (res // 2) ** 2
    params += 7 * 7 * 3 * 64
    ops.append({"name": "stem 7x7/2", "macs": int(macs)})
    s, cin = res // 4, 64
    for n, w, stride in ((3, 64, 1), (4, 128, 2), (6, 256, 2), (3, 512, 2)):
        for b in range(n):
            st = stride if b == 0 else 1
            s_out = s // st
            layers = [(1, cin, w, s), (3, w, w, s_out), (1, w, 4 * w, s_out)]
            if b == 0:
                layers.append((1, cin, 4 * w, s_out))
            for k, ci, co, sp in layers:
                macs += k * k * ci * co * sp * sp
                params += k * k * ci * co
            cin, s = 4 * w, s_out
    macs += cin * 1000
    params += cin * 1000 + 1000
    return {"name": "ResNet-50", "res": res, "macs": int(macs), "params": int(params)}


def count_convnext_t(res=224):
    dims, depths = (96, 192, 384, 768), (3, 3, 9, 3)
    s = res // 4
    macs = 4 * 4 * 3 * dims[0] * s * s
    params = 4 * 4 * 3 * dims[0]
    for i, (d, n) in enumerate(zip(dims, depths)):
        if i > 0:
            s //= 2
            macs += 2 * 2 * dims[i - 1] * d * s * s
            params += 2 * 2 * dims[i - 1] * d
        for _ in range(n):
            macs += 7 * 7 * d * s * s + d * (4 * d) * s * s + (4 * d) * d * s * s
            params += 7 * 7 * d + d * 4 * d + 4 * d * d
    macs += dims[-1] * 1000
    params += dims[-1] * 1000 + 1000
    return {"name": "ConvNeXt-T", "res": res, "macs": int(macs), "params": int(params)}


REFERENCE = {
    "source": "transcribed specification, counted by us, never trained, no accuracies",
    "caveat": "these two rows are the only numbers in this file that were not produced "
              "by training something. The layer tables were typed in from the papers "
              "and then counted by the same arithmetic used everywhere else, so the "
              "counter is validating our transcription and not the architecture. "
              "Normalisation and bias parameters are excluded from the parameter "
              "counts, which are convolution and classifier weights only.",
    "rows": [count_resnet50(), count_convnext_t()],
    "our_scale": {"id": f"rung{LADDER_TOP:02d}", "macs": R(f"rung{LADDER_TOP:02d}")["macs"],
                  "params": R(f"rung{LADDER_TOP:02d}")["params"], "res": LADDER_RES,
                  "n_train": N_TRAIN},
}
_ratio = count_convnext_t()["macs"] / R(f"rung{LADDER_TOP:02d}")["macs"]
print(f"\nscale: ConvNeXt-T is {count_convnext_t()['macs'] / 1e9:.2f} GMAC per image at "
      f"224x224 against {R(f'rung{LADDER_TOP:02d}')['macs'] / 1e6:.2f} MMAC here, a "
      f"factor of {_ratio:,.0f}")

# ======================================= the cost model, checked against itself
CFGS = {e["id"]: e["cfg"] for e in PLANNED}
OPSETS = {e["id"]: e["ops"] for e in PLANNED}
checks = []
for rid in ("rung00", "rung02", "s0-base", "scale-depth", "res-28", "iso-k7"):
    if rid not in RESULTS:
        continue
    ck = cost_check(CFGS[rid], OPSETS[rid], RESULTS[rid]["macs"], rid)
    if ck:
        checks.append(ck)
COST_MODEL = {
    "formula": "sum over stages of blocks x 2 k^2 C_s^2 R_s^2, with a downsampling "
               "block costing 1.5 k^2 C_s^2 R_s^2 because its first convolution reads "
               "half as many channels as it writes",
    "checks": checks,
    "separable_identity": {
        "claim": "an inverted bottleneck at width 1.5C with expansion 4 matches a basic "
                 "block at width C on the DENSE part of the arithmetic",
        "dense_part": "2 x (1.5C) x (4 x 1.5C) = 18 C^2 against 2 k^2 C^2 = 18 C^2 at k=3",
        "depthwise_part_not_included": "k^2 x 1.5C with the depthwise above the "
                                       "expansion, or k^2 x 6C below it. The identity is "
                                       "exact for the pointwise convolutions only, and "
                                       "calling the whole block iso-cost overstates it",
        "at_C12_k7": {"basic_block": 2 * 9 * 12 * 12,
                      "inverted_pointwise": 8 * 18 * 18,
                      "depthwise_above": 49 * 18,
                      "depthwise_below": 49 * 4 * 18},
    },
    "note": "the exported macs_rel_err is dominated by SPATIAL ROUNDING rather than by "
            "the shortcut convolutions, and the decomposition in `checks` shows which "
            "is which by evaluating the same formula twice rather than by asserting a "
            "cause. The pair to look at is rung00 against res-28: identical "
            "architecture, and at the resolution where every halving is exact the whole "
            "error collapses to the shortcut term.",
}
if checks:
    c0 = checks[0]
    print(f"\ncost model: {c0['id']} analytic {c0['analytic'] / 1e6:.3f} MMAC against "
          f"counted {c0['counted'] / 1e6:.3f}, {c0['rel_err'] * 100:+.2f}%; spatial "
          f"rounding accounts for {c0['spatial_rounding_share'] * 100:.0f}% of the gap, "
          f"shortcuts {c0['shortcut_term'] / 1e6:.3f} MMAC")

# ============================== reference tensors the browser checks itself on
torch.manual_seed(SEED)
_t = torch.randn(4, 4, 4, 4)
_bn = nn.BatchNorm2d(4)
_bn.train()
with torch.no_grad():
    _bn_out = _bn(_t)
    _ln_out = ChanLayerNorm(4)(_t)
    _gelu = F.gelu(_t)
    _gelu_tanh = F.gelu(_t, approximate="tanh")
    _relu = F.relu(_t)


def nest(t):
    return [[[[rnd(v, 6) for v in row] for row in ch] for ch in img] for img in t.numpy()]


RECOMPUTE = {
    "norm_ref": {
        "shape": [4, 4, 4, 4], "seed": SEED,
        "input": nest(_t),
        "batchnorm2d": nest(_bn_out), "batchnorm_eps": 1e-5,
        "batchnorm_note": "training-mode statistics: mean and variance over the batch "
                          "and both spatial axes, one pair per channel, biased variance",
        "layernorm_channels": nest(_ln_out), "layernorm_eps": 1e-6,
        "layernorm_note": "one mean and variance per spatial position, taken across the "
                          "channels, which is what ConvNeXt does and what BatchNorm2d "
                          "does not",
        "gelu_exact": nest(_gelu), "gelu_tanh": nest(_gelu_tanh), "relu": nest(_relu),
    },
    "dw_ref": R(f"rung{LADDER_TOP:02d}").get("extra", {}),
    "dw_ref_note": "the trained depthwise kernels of the top rung's first block, the "
                   "exact tensor that entered them for one test garment and the exact "
                   "tensor that left. A depthwise convolution touches one channel at a "
                   "time, so the eight exported channels are a complete check of those "
                   "eight.",
}

# ==================================================== images for the article
pick = [int(np.where(yte_np == c)[0][0]) for c in range(10)]
sprite = np.concatenate(list(Xte_u8_full[pick]), axis=1)
Image.fromarray(sprite.astype(np.uint8), mode="L").save(IMG / "garments.png")
res_sprites = []
for r in RES_LADDER:
    _, Xt = data_at(r, None)
    tiles = (Xt[pick, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.concatenate(list(tiles), axis=1), mode="L").save(
        IMG / f"garments-r{r}.png")
    res_sprites.append({"res": r, "image": f"garments-r{r}.png", "tile": r})
print(f"\nsprites: garments.png plus {len(res_sprites)} resolution rows in {IMG}")

# ========================================================================= json
LIMITATIONS = [
    "28x28 images and 6000 of them is three orders of magnitude away from where either "
    "paper's claims were established. Every result labelled `regime` is a small-scale "
    "observation and not a refutation.",
    "Wall clock on this hybrid-core CPU moves by about a factor of two for an identical "
    "operation depending on thread count and scheduler state. No timing claim below "
    "that factor is made anywhere.",
    f"{EPOCHS} epochs favours architectures that optimise quickly. The epochs control "
    "is the check, and every run exports its last-two-epoch improvement and the epoch "
    "at which it first reached the loss threshold so a reader can see which runs were "
    "still moving.",
    "Eleven rungs judged at a t-based threshold still yields expected false positives. "
    "The count is exported, and because the rungs are cumulative the tests are not "
    "independent.",
    "A single learning rate across a threefold width range is a residual confound "
    "inside the scaling thread. The pilot bounds it at the ladder's two ends; it does "
    "not remove it from the simplex.",
    "The textbook cost formula is exact only where R/2 and R/4 are whole numbers. The "
    "ladder's stages run 14, 7 and 4 rather than 14, 7 and 3.5, so the most expensive "
    "stage is overcounted by about thirty per cent and the exported macs_rel_err is "
    "dominated by that rather than by the shortcut convolutions. One consequence the "
    "article has to respect: moving a block between stages is not free in "
    "multiply-accumulates at this resolution, so no stage-placement comparison here is "
    "compute-neutral.",
]

out = {
    "meta": {
        "article": "EfficientNet and ConvNeXt",
        "generator": "src/utils/generate_convnext_data.py",
        "epochs": EPOCHS, "batch": BATCH, "n_train": N_TRAIN, "n_test": N_TEST,
        "seeds_core": list(SEEDS_CORE), "seeds_noise": list(SEEDS_NOISE),
        "threads_requested": THREADS,
        "recipes": {
            "sgd": "SGD momentum 0.9, weight decay 1e-4, x0.1 at 50% and 80%; the "
                   "residual-network article's recipe, kept so that numbers from the "
                   "two articles can be put side by side",
            "adamw": "AdamW, weight decay 0.05, one epoch of linear warm-up then cosine "
                     "to zero",
        },
        "deliberately_excluded": "LayerScale, stochastic depth, label smoothing and "
                                 "heavy augmentation. They matter at 300 epochs and "
                                 "would confound a block ablation at six.",
        "throughput": {
            "assumed_dense": ASSUMED_F_DENSE, "assumed_modern": ASSUMED_F_MODERN,
            "assumed_ln_penalty": ASSUMED_LN_PENALTY,
            "measured_dense": rnd(F_DENSE, 3), "measured_modern": rnd(F_MODERN, 3),
            "measured_ln_penalty": rnd(LN_PENALTY, 3),
            "law": "seconds per epoch = MMAC x (N/6000) x f",
            "calibration": calibration,
            "thread_spread": thread_spread, "worst_thread_spread": rnd(worst_spread, 3),
        },
        "precision_note": PRECISION_NOTE,
        "projection": PROJECTION,
        "wall_clock_minutes_this_session": rnd(WALL / 60, 2),
    },
    "data": {"name": "Fashion-MNIST", "n_train": int(len(ytr)), "n_test": int(len(yte)),
             "classes": FASHION_CLASSES, "native_res": NATIVE, "seed": SEED,
             "train_subset": "class-stratified, seeded, frozen across every run in this "
                             "file, so no comparison anywhere carries a data-draw "
                             "difference",
             "test_split": "the full official 10000-image test split",
             "resize_policy": "train and test are resized by exactly the same operation; "
                              "downwards is bilinear with antialiasing because plain "
                              "bilinear downsampling aliases and every low-resolution "
                              "point would then be measuring the aliasing"},
    "cost_model": COST_MODEL,
    "pilot": PILOT,
    "scaling": {
        "base": "s0-base", "budget_target": BUDGET_TARGET,
        "arms": arms, "budget_spread": rnd(budget_spread, 4),
        "budget_spread_note": "the design holds the budget fixed algebraically; integer "
                              "block counts, integer channels and integer resolutions "
                              "mean the achieved budgets still span this factor. Every "
                              "chart is drawn against achieved multiply-accumulates and "
                              "the spread is shown, because it is larger than several "
                              "of the accuracy effects on the same chart.",
        "efficientnet": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
                         "compound_per_phi": rnd(COMPOUND, 4), "phi": rnd(PHI, 4),
                         "depth_factor": rnd(EFF_D, 4), "width_factor": rnd(EFF_W, 4),
                         "res_factor": rnd(EFF_R, 4),
                         "note": "alpha x beta^2 x gamma^2 is 1.92, not 2, so a unit "
                                 "step of phi multiplies the budget by 1.92 and the "
                                 "usual 2x, 4x, 8x reading of the compound rule is "
                                 "wrong. Here the exponents generate the allocation and "
                                 "the budget lands where it lands."},
        "budgets": BUDGET_RANKINGS,
        "budgets_note": "scores are chance-corrected accuracy divided by the budget in "
                        "question. Which allocation wins has no answer until the unit "
                        "of the budget is named.",
    },
    "resolution": {"ladder": res_rows,
                   "receptive_field_note":
                       "the architecture is fixed across this ladder, so the receptive "
                       "field is the same number of pixels at every rung and only its "
                       "size RELATIVE to the image changes. At these resolutions that "
                       "ratio is above one everywhere: a three-stage network already "
                       "sees past the edge of a 28-pixel garment. So the "
                       "receptive-field-fraction worry that the scaling literature "
                       "raises does not bite here, and the restride control measures "
                       "what is left of it rather than assuming it matters. This is a "
                       "small-scale observation and it would not survive at 224 pixels.",
                   "per_class_trend": per_class_trend,
                   "hard_four": [FASHION_CLASSES[i] for i in HARD4],
                   "easy_six": [FASHION_CLASSES[i] for i in EASY6],
                   "restride": RESTRIDE, "information_control": INFORMATION},
    "ladder": {
        "rungs": rungs, "top": LADDER_TOP, "end_to_end": end_to_end,
        "noise_floor": NOISE_FLOOR, "multiplicity": MULTIPLICITY,
        "flop_matched": MATCHED.get("flop-matched"),
        "param_matched": MATCHED.get("param-matched"),
        "matched_note": "equal multiply-accumulates and equal parameters are different "
                        "budgets and they are solved for separately. The matched pair, "
                        "not the raw ladder endpoint, is this article's headline "
                        "comparison, because every other rung moves cost and design "
                        "together.",
        "receptive_field_note":
            "each rung exports the receptive field of its main path, and it does not "
            "move smoothly. Replacing a two-convolution block with a single depthwise "
            "convolution roughly halves it, and the 7x7 kernel more than restores it. "
            "So rungs 5 and 6 change how far the network can see as well as what it is "
            "made of, and the honest reading of a loss there is 'this rung changed two "
            "things that the configuration dictionary counts as one'. The kernel sweep "
            "is where that is measured directly.",
        "isolated": ISOLATED_BLOCK, "not_isolable": NOT_ISOLABLE,
        "kernels": KERNELS, "recipe_2x2": RECIPE_2x2, "epochs_control": EPOCHS_CONTROL,
    },
    "separable": SEPARABLE,
    "ops": OPS,
    "reference_architectures": REFERENCE,
    "recompute": RECOMPUTE,
    "garments": {"image": "garments.png", "tile": NATIVE, "count": len(pick),
                 "labels": [int(yte_np[i]) for i in pick],
                 "test_indices": [int(i) for i in pick],
                 "resolutions": res_sprites},
    "runs": RESULTS,
    "limitations": LIMITATIONS,
    "versions": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "torch": torch.__version__,
                 "n_train": N_TRAIN, "n_test": N_TEST, "threads": THREADS,
                 "loss_threshold": LOSS_THRESHOLD,
                 "cache": str(CACHE), "cache_version": CACHE_VERSION},
}
path = OUT / "convnext.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB), "
      f"{len(RESULTS)} configurations, "
      f"{sum(len(r['seeds']) for r in RESULTS.values())} trained models")
