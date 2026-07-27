"""Data for the ATLAS 'Depth' web article (module 1.3): VGG and ResNet.

The previous article ended on the one decision a convolutional network still
leaves to a person: its shape. This one takes the most obvious knob on that
shape, how many layers, and pushes it until it breaks.

Measured here:
  * VGG's arithmetic, and whether it survives contact with a training run:
    two stacked 3x3 convolutions see the same 5x5 window as one 5x5
    convolution, with fewer parameters and an extra nonlinearity;
  * the degradation problem itself, on a ladder of plain networks from 8 to
    56 layers, reporting TRAINING accuracy, because the claim that made
    residual connections necessary is that deeper networks fit their own
    training data worse, which is not overfitting;
  * the explanation everybody reaches for first, vanishing gradients, tested
    directly by measuring the gradient reaching the first layer at every
    depth, with batch normalisation in place exactly as in the paper;
  * what the same ladder does once each pair of layers is given a shortcut;
  * how large the residuals actually are in a trained residual network, which
    is the evidence that the shortcut works by making 'change nothing' the
    cheap default.

Fashion-MNIST rather than MNIST on purpose: an optimisation effect needs a
task where optimisation is the bottleneck, and handwritten digits are not
one. Run from anywhere: `python src/utils/generate_resnet_data.py`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_data import fashion_mnist, subset, FASHION_CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "resnet" / "data"
IMG = ROOT / "resnet" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)

SEED = 19
N_TRAIN = 6000
N_TEST = 4000
EPOCHS = 12
BATCH = 128
WIDTH = 16
DEPTHS = [8, 14, 20, 32, 44, 56]        # 6n+2, the ladder of the CIFAR paper

torch.set_num_threads(12)
torch.manual_seed(SEED)

Xtr_u8, ytr_np, Xte_u8, yte_np = fashion_mnist()
Xtr_u8, ytr_np = subset(Xtr_u8, ytr_np, N_TRAIN, seed=SEED)
Xte_u8, yte_np = subset(Xte_u8, yte_np, N_TEST, seed=SEED)
Xtr = torch.tensor(Xtr_u8, dtype=torch.float32).unsqueeze(1) / 255.0
Xte = torch.tensor(Xte_u8, dtype=torch.float32).unsqueeze(1) / 255.0
ytr = torch.tensor(ytr_np, dtype=torch.long)
yte = torch.tensor(yte_np, dtype=torch.long)
print(f"Fashion-MNIST: {len(ytr)} train / {len(yte)} test, 28x28, "
      f"{len(FASHION_CLASSES)} classes")


# ============================================================ architectures
class Block(nn.Module):
    """Two 3x3 convolutions, with an optional shortcut around them.

    Everything except the shortcut is identical between the two families, so
    the comparison later is about that one addition and nothing else. Batch
    normalisation is present in BOTH, deliberately: the paper's point is that
    degradation survives it, and leaving it out would let the article blame
    vanishing gradients without ever testing the claim.
    """

    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()
        self.residual = residual
        self.short = None
        if residual and (stride != 1 or cin != cout):
            # A 1x1 convolution only where the shape changes, which is the
            # paper's option B and the only place a shortcut carries weights.
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def residual_part(self, x):
        return self.b2(self.c2(self.act(self.b1(self.c1(x)))))

    def forward(self, x):
        f = self.residual_part(x)
        if not self.residual:
            return self.act(f)
        skip = x if self.short is None else self.short(x)
        return self.act(f + skip)


class Net(nn.Module):
    """The CIFAR-paper family: a stem, three stages of n blocks, a global
    average pool and a linear classifier. depth = 6n + 2."""

    def __init__(self, depth, residual, width=WIDTH, act=nn.ReLU):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), act())
        blocks = []
        cin = width
        for stage, (cout, stride) in enumerate([(width, 1), (width * 2, 2), (width * 4, 2)]):
            for b in range(n):
                blk = Block(cin, cout, stride if b == 0 else 1, residual)
                if act is not nn.ReLU:
                    blk.act = act()
                blocks.append(blk)
                cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(width * 4, 10))
        self.depth = depth
        self.residual = residual

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class Stack(nn.Module):
    """A network for the VGG comparison: three stages of `per_stage`
    convolutions of size `kernel`, pooling between stages.

    The control that makes the comparison mean anything is that the spatial
    schedule never changes. Three stages, two pools, the same channel counts;
    the only difference between one 5x5 layer and two 3x3 layers per stage is
    how the same 5x5 window is built. Matching the number of LAYERS instead
    would compare different receptive fields, which is a different experiment
    and not the one VGG's argument is about.
    """

    def __init__(self, kernel, per_stage, width=24):
        super().__init__()
        pad = kernel // 2
        seq = []
        cin = 1
        for stage in range(3):
            cout = width * (2 ** stage)
            for _ in range(per_stage):
                seq += [nn.Conv2d(cin, cout, kernel, padding=pad, bias=False),
                        nn.BatchNorm2d(cout), nn.ReLU()]
                cin = cout
            if stage < 2:
                seq.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*seq)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 10))
        self.stage_field = 1 + per_stage * (kernel - 1)

    def forward(self, x):
        return self.head(self.body(x))


# A run of this generator is half an hour of CPU, so each finished experiment
# is written to a cache outside the repository and reused on the next run.
# Nothing is approximated: the cache stores results, not shortcuts, and
# deleting the directory reproduces everything from scratch.
CACHE = Path.home() / ".atlas_vision_data" / "depth_cache"
CACHE.mkdir(parents=True, exist_ok=True)
# v2: the first version measured accuracy after a gradient probe that had
# quietly moved every batch-norm layer's running statistics, which put the
# reported numbers 0.14 points off the network they claimed to describe. The
# probe restores the buffers now, and the old cache is left behind rather
# than mixed in.
CONFIG = f"fashion-n{N_TRAIN}-e{EPOCHS}-b{BATCH}-w{WIDTH}-s{SEED}-v2"


def cached(name, fn):
    path = CACHE / f"{CONFIG}-{name}.json"
    if path.exists():
        print(f"  [cached] {name}", flush=True)
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result))
    return result


def count(m):
    return sum(p.numel() for p in m.parameters())


def conv_params(m):
    """Only the convolutions, which is what the VGG argument is about."""
    return sum(p.numel() for mod in m.modules() if isinstance(mod, nn.Conv2d)
               for p in mod.parameters())


# =================================================================== training
def train(model, X, y, epochs=EPOCHS, seed=SEED, lr=0.05, log=False):
    """SGD with momentum and a step schedule, the way the paper trained, not
    Adam: an adaptive optimiser is exactly the thing that would paper over an
    optimisation difficulty this article is trying to measure."""
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(epochs * 0.5), int(epochs * 0.8)], gamma=0.1)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    history = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g)
        tot = 0.0
        for i in range(0, len(y), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        history.append(rnd(tot / len(y), 5))
        if log:
            print(f"      epoch {ep + 1}: train loss {history[-1]:.4f}", flush=True)
    return history, time.perf_counter() - t0


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    out = []
    for i in range(0, len(y), 500):
        out.append(model(X[i:i + 500]).argmax(1))
    return float((torch.cat(out) == y).float().mean())


@torch.no_grad()
def mean_loss(model, X, y):
    model.eval()
    lossf = nn.CrossEntropyLoss(reduction="sum")
    tot = 0.0
    for i in range(0, len(y), 500):
        tot += float(lossf(model(X[i:i + 500]), y[i:i + 500]))
    return tot / len(y)


def first_layer_gradient(model, X, y, seed=SEED, batches=8):
    """The gradient actually arriving at the very first convolution, averaged
    over a few batches. This is the quantity the vanishing-gradient story
    predicts should collapse as depth grows.

    Measured twice per network, at initialisation and after training, because
    they answer different questions: whether the signal ever reaches the first
    layer at all, and whether it is still there once the run has stalled. The
    vanishing-gradient hypothesis is a claim about the first one.
    """
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    stem_conv = model.stem[0]
    norms = []
    per_layer = []

    # The probe has to run in train mode, because batch statistics are what
    # the gradient being measured actually sees. But train mode also UPDATES
    # every batch-norm layer's running statistics, and those are what eval
    # mode uses, so a probe taken after training would quietly move every
    # accuracy measured afterwards. Snapshot the buffers and put them back.
    saved = [(m, m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone())
             for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    model.train()
    for b in range(batches):
        idx = perm[b * BATCH:(b + 1) * BATCH]
        model.zero_grad()
        lossf(model(X[idx]), y[idx]).backward()
        norms.append(float(stem_conv.weight.grad.norm()))
        if b == 0:
            for mod in model.modules():
                if isinstance(mod, nn.Conv2d) and mod.weight.grad is not None:
                    per_layer.append(float(mod.weight.grad.norm()))
    model.zero_grad()

    for m, mean, var, n in saved:
        m.running_mean.copy_(mean)
        m.running_var.copy_(var)
        m.num_batches_tracked.copy_(n)

    return float(np.mean(norms)), per_layer


@torch.no_grad()
def residual_sizes(model, X, n=500):
    """For each block of a trained residual network, how big the change it
    makes is next to what came in: mean ||F(x)|| / ||x||. The paper's
    Figure 7 argument, and the evidence that a shortcut works by making
    'leave it alone' the cheapest thing a block can do."""
    model.eval()
    x = model.stem(X[:n])
    out = []
    for blk in model.blocks:
        f = blk.residual_part(x)
        skip = x if blk.short is None else blk.short(x)
        ratio = (f.flatten(1).norm(dim=1) / skip.flatten(1).norm(dim=1).clamp_min(1e-9))
        out.append(rnd(ratio.mean(), 4))
        x = blk.act(f + skip) if blk.residual else blk.act(f)
    return out


# ================================================== VGG: one 5x5 or two 3x3
# Paired by RECEPTIVE FIELD, not by layer count: a 5x5 layer and two stacked
# 3x3 layers see the same window, and that is the only thing being swapped.
# Everything else (three stages, two pools, 24/48/96 channels) is held fixed.
print("\nVGG's arithmetic, then the same thing measured:")
VGG_RUNS = [(5, 1, "one 5x5 per stage"), (3, 2, "two 3x3 per stage"),
            (7, 1, "one 7x7 per stage"), (3, 3, "three 3x3 per stage")]
vgg_rows = []
for kernel, per_stage, label in VGG_RUNS:
    def run_vgg(kernel=kernel, per_stage=per_stage, label=label):
        torch.manual_seed(SEED)
        m = Stack(kernel, per_stage)
        _, secs = train(m, Xtr, ytr, epochs=EPOCHS)
        return {"label": label, "kernel": kernel, "per_stage": per_stage,
                "layers": per_stage * 3, "conv_params": conv_params(m),
                "params": count(m), "stage_field": m.stage_field,
                "train_acc": rnd(accuracy(m, Xtr, ytr)),
                "test_acc": rnd(accuracy(m, Xte, yte)), "seconds": rnd(secs, 1)}

    row = cached(f"vgg-k{kernel}x{per_stage}", run_vgg)
    vgg_rows.append(row)
    print(f"  {label:22s} k={kernel} x{per_stage}/stage  window {row['stage_field']}  "
          f"{row['conv_params']:7,} conv params  train {row['train_acc']:.4f}  "
          f"test {row['test_acc']:.4f}  {row['seconds']:.0f}s", flush=True)

# The arithmetic on its own, at a fixed channel count, which is the form the
# argument is usually quoted in and the form the widget lets a reader play with.
C = 64
kernel_math = [{"k": k, "params": k * k * C * C,
                "stacked_3x3": ((k - 1) // 2) * 9 * C * C,
                "layers_of_3x3": (k - 1) // 2} for k in (3, 5, 7, 9, 11)]
print(f"  at {C} channels in and out: one 5x5 is {5*5*C*C:,} weights, "
      f"two 3x3 are {2*9*C*C:,}")

# ======================================================= the depth ladder
print("\nthe depth ladder, plain against residual "
      f"({EPOCHS} epochs, SGD + momentum, identical seeds):")
ladder = []
grad_profiles = {}
for depth in DEPTHS:
    row = {"depth": depth}
    for residual in (False, True):
        key = "residual" if residual else "plain"

        def run(depth=depth, residual=residual, key=key):
            # Seed the construction as well as the training, so the numbers
            # do not depend on how many models were built before this one.
            torch.manual_seed(SEED + depth + (1000 if residual else 0))
            model = Net(depth, residual)
            # At initialisation, before a single step: this is the measurement
            # the vanishing-gradient hypothesis is actually about.
            g_init, per_layer_init = first_layer_gradient(model, Xtr, ytr)
            hist, secs = train(model, Xtr, ytr)
            gnorm, per_layer = first_layer_gradient(model, Xtr, ytr)
            out = {"train_acc": rnd(accuracy(model, Xtr, ytr)),
                   "test_acc": rnd(accuracy(model, Xte, yte)),
                   "train_loss": rnd(mean_loss(model, Xtr, ytr), 5),
                   "params": count(model), "seconds": rnd(secs, 1),
                   "grad_first": rnd(gnorm, 6), "grad_first_init": rnd(g_init, 6),
                   "history": hist,
                   "grad_layers_init": [rnd(v, 6) for v in per_layer_init],
                   "grad_layers_trained": [rnd(v, 6) for v in per_layer]}
            if residual:
                out["block_residuals"] = residual_sizes(model, Xtr)
            return out

        res = cached(f"ladder-d{depth}-{key}", run)
        row[key] = res
        if depth in (20, DEPTHS[-1]):
            grad_profiles[f"{key}{depth}"] = res["grad_layers_init"]
            grad_profiles[f"{key}{depth}_trained"] = res["grad_layers_trained"]
        print(f"  depth {depth:2d} {key:9s}  train {res['train_acc']:.4f} "
              f"(loss {res['train_loss']:.4f})  test {res['test_acc']:.4f}  "
              f"grad at layer 1: {res['grad_first_init']:.2e} at init, "
              f"{res['grad_first']:.2e} trained  {res['seconds']:.0f}s", flush=True)
    ladder.append(row)

# The per-layer profiles are only needed for the two depths the chart draws;
# carrying them for every rung would triple the file for nothing.
for r in ladder:
    for key in ("plain", "residual"):
        r[key].pop("grad_layers_init", None)
        r[key].pop("grad_layers_trained", None)

# Did the deep networks actually converge, or were they merely stopped early?
# Without this check "deeper trains worse" could just mean "deeper is slower",
# and the whole article would be measuring patience.
for r in ladder:
    for key in ("plain", "residual"):
        h = r[key]["history"]
        r[key]["last_two_epochs"] = rnd(h[-3] - h[-1], 5)
tail = ladder[-1]["plain"]
print(f"\nconvergence check: the {DEPTHS[-1]}-layer plain network improved its training "
      f"loss by {tail['last_two_epochs']:.5f} over its last two epochs, from "
      f"{tail['history'][-3]:.4f} to {tail['history'][-1]:.4f}")

# ================================================== what the ladder means
best_plain = max(ladder, key=lambda r: r["plain"]["train_acc"])
worst_deep_plain = ladder[-1]["plain"]
deg = best_plain["plain"]["train_acc"] - worst_deep_plain["train_acc"]
print(f"\nplain: best training accuracy {best_plain['plain']['train_acc']:.4f} at depth "
      f"{best_plain['depth']}, and {worst_deep_plain['train_acc']:.4f} at depth "
      f"{ladder[-1]['depth']}: a drop of {deg:.4f} in TRAINING accuracy")
res_best = max(ladder, key=lambda r: r["residual"]["train_acc"])
print(f"residual: best training accuracy {res_best['residual']['train_acc']:.4f} at depth "
      f"{res_best['depth']}, deepest {ladder[-1]['residual']['train_acc']:.4f}")
for when in ("grad_first_init", "grad_first"):
    g_shallow = ladder[0]["plain"][when]
    g_deep = ladder[-1]["plain"][when]
    label = "at initialisation" if when.endswith("init") else "after training"
    # Say which way it went rather than printing a ratio that reads as zero
    # when the gradient grew by three orders of magnitude.
    direction = "smaller" if g_deep < g_shallow else "LARGER"
    factor = g_shallow / g_deep if g_deep < g_shallow else g_deep / g_shallow
    print(f"gradient at the first layer {label}, plain: {g_shallow:.2e} at depth "
          f"{DEPTHS[0]} and {g_deep:.2e} at depth {DEPTHS[-1]}: "
          f"{factor:,.1f} times {direction}")

# ====================================== a promise from the previous article
# On a two-layer network in the LeNet article, tanh beat ReLU, and that
# article said the advantage of ReLU is a story about depth and that this one
# would test it. Same networks, same seeds, only the activation swapped.
print("\ndoes ReLU's advantage appear once the network is deep? (promised in the "
      "previous article)")
act_rows = []
for depth in (20, DEPTHS[-1]):
    for name, act in (("relu", nn.ReLU), ("tanh", nn.Tanh)):
        if name == "relu":
            # The ReLU half of this comparison is a rung of the ladder, built
            # with the same seed and the same everything: reuse it rather than
            # spending a quarter of an hour reproducing it.
            rung = next(r for r in ladder if r["depth"] == depth)["plain"]
            act_rows.append({"depth": depth, "act": name,
                             "train_acc": rung["train_acc"], "test_acc": rung["test_acc"],
                             "train_loss": rung["train_loss"], "seconds": rung["seconds"],
                             "history": rung["history"]})
            print(f"  depth {depth:2d} {name:5s} train {rung['train_acc']:.4f}  "
                  f"test {rung['test_acc']:.4f}  [the ladder's own rung]", flush=True)
            continue

        def run_act(depth=depth, act=act):
            torch.manual_seed(SEED + depth)
            m = Net(depth, residual=False, act=act)
            hist, secs = train(m, Xtr, ytr)
            return {"train_acc": rnd(accuracy(m, Xtr, ytr)),
                    "test_acc": rnd(accuracy(m, Xte, yte)),
                    "train_loss": rnd(mean_loss(m, Xtr, ytr), 5),
                    "seconds": rnd(secs, 1), "history": hist}

        row = cached(f"act-d{depth}-plain-{name}", run_act)
        row.update({"depth": depth, "act": name})
        act_rows.append(row)
        print(f"  depth {depth:2d} {name:5s} train {row['train_acc']:.4f}  "
              f"test {row['test_acc']:.4f}", flush=True)

# ============================================ how much does the seed move it?
# One run per rung is only worth reporting if a rung's height is stable. Three
# seeds at a single middle depth give the noise floor for the whole ladder,
# which is the number a reader needs before believing any gap on it.
print("\nthe noise floor: the same 20-layer plain network at three seeds:")
seed_spread = []
ladder20 = next(r for r in ladder if r["depth"] == 20)["plain"]
for s in (SEED, SEED + 1, SEED + 2):
    if s == SEED:
        # Seed 19 at depth 20 is a rung of the ladder already: retraining it
        # to fill a column would be ten minutes spent reproducing a number
        # that is sitting in the cache.
        seed_spread.append({"seed": s, "train_acc": ladder20["train_acc"],
                            "test_acc": ladder20["test_acc"],
                            "train_loss": ladder20["train_loss"],
                            "seconds": ladder20["seconds"]})
        print(f"  seed {s}: train {ladder20['train_acc']:.4f}  "
              f"test {ladder20['test_acc']:.4f}  [the ladder's own rung]", flush=True)
        continue

    def run_seed(s=s):
        torch.manual_seed(s + 20)
        m = Net(20, residual=False)
        _, secs = train(m, Xtr, ytr, seed=s)
        return {"seed": s, "train_acc": rnd(accuracy(m, Xtr, ytr)),
                "test_acc": rnd(accuracy(m, Xte, yte)),
                "train_loss": rnd(mean_loss(m, Xtr, ytr), 5), "seconds": rnd(secs, 1)}

    row = cached(f"seed-d20-plain-{s}", run_seed)
    seed_spread.append(row)
    print(f"  seed {row['seed']}: train {row['train_acc']:.4f}  test {row['test_acc']:.4f}", flush=True)
spread_train = max(r["train_acc"] for r in seed_spread) - min(r["train_acc"] for r in seed_spread)
spread_test = max(r["test_acc"] for r in seed_spread) - min(r["test_acc"] for r in seed_spread)
print(f"  spread across three seeds: {spread_train * 100:.2f} points of training accuracy, "
      f"{spread_test * 100:.2f} of test")

# ================================== the control the degradation claim needs
# "The deepest plain network fits its training data worse" invites one obvious
# objection: that the learning rate was simply wrong for it. So train the same
# network at a third of the rate and at double it. If it recovers, the honest
# finding is about hyperparameters rather than about depth, and the article
# has to say so.
print("\nis the deepest plain network just badly tuned? same network, other rates:")
lr_control = []
deep_plain = ladder[-1]["plain"]
for lr in (0.02, 0.05, 0.1):
    if lr == 0.05:
        # The ladder already trained exactly this network at exactly this rate.
        lr_control.append({"lr": lr, "train_acc": deep_plain["train_acc"],
                           "test_acc": deep_plain["test_acc"],
                           "train_loss": deep_plain["train_loss"],
                           "seconds": deep_plain["seconds"],
                           "history": deep_plain["history"]})
        print(f"  lr {lr:<5} train {deep_plain['train_acc']:.4f} "
              f"(loss {deep_plain['train_loss']:.4f})  test {deep_plain['test_acc']:.4f}  "
              f"[the ladder's own deepest rung]", flush=True)
        continue

    def run_lr(lr=lr):
        torch.manual_seed(SEED + DEPTHS[-1])
        m = Net(DEPTHS[-1], residual=False)
        hist, secs = train(m, Xtr, ytr, lr=lr)
        return {"lr": lr, "train_acc": rnd(accuracy(m, Xtr, ytr)),
                "test_acc": rnd(accuracy(m, Xte, yte)),
                "train_loss": rnd(mean_loss(m, Xtr, ytr), 5),
                "seconds": rnd(secs, 1), "history": hist}

    row = cached(f"lr-d{DEPTHS[-1]}-plain-{lr}", run_lr)
    lr_control.append(row)
    print(f"  lr {lr:<5} train {row['train_acc']:.4f} (loss {row['train_loss']:.4f})  "
          f"test {row['test_acc']:.4f}", flush=True)
# Is any of this reproducible at all? Train the cheapest rung of the ladder a
# second time from scratch and see whether it lands in the same place. Done on
# the 8-layer network on purpose: the check costs a minute there and a quarter
# of an hour at the deep end, and it is the pipeline being tested, not a depth.
def run_repeat():
    torch.manual_seed(SEED + DEPTHS[0])
    m = Net(DEPTHS[0], residual=False)
    train(m, Xtr, ytr)
    return {"train_acc": rnd(accuracy(m, Xtr, ytr)), "test_acc": rnd(accuracy(m, Xte, yte))}


repeat = cached(f"repeat-d{DEPTHS[0]}-plain", run_repeat)
ref = ladder[0]["plain"]
determinism = rnd(abs(repeat["train_acc"] - ref["train_acc"]), 6)
if determinism < 1e-9:
    print(f"\ndeterminism: the {DEPTHS[0]}-layer network trained twice from scratch gave "
          f"{repeat['train_acc']:.4f} both times")
else:
    # A warning rather than an assertion: failing here would throw away an
    # hour of training without ever writing the json.
    print(f"\nWARNING: not deterministic. The {DEPTHS[0]}-layer network trained twice gave "
          f"{repeat['train_acc']} and {ref['train_acc']}")

best_lr = max(lr_control, key=lambda r: r["train_acc"])
best_shallow = max(ladder, key=lambda r: r["plain"]["train_acc"])
print(f"  best rate for {DEPTHS[-1]} plain layers: {best_lr['lr']} at "
      f"{best_lr['train_acc']:.4f} training accuracy, against "
      f"{best_shallow['plain']['train_acc']:.4f} for the best shallower plain network "
      f"({best_shallow['depth']} layers)")

# ============================================ a sprite for the article's images
pick = [int(np.where(yte_np == c)[0][0]) for c in range(10)]
sprite = np.concatenate(list(Xte_u8[pick]), axis=1)
Image.fromarray(sprite.astype(np.uint8), mode="L").save(IMG / "garments.png")

out = {
    "data": {"name": "Fashion-MNIST", "n_train": int(len(ytr)), "n_test": int(len(yte)),
             "classes": FASHION_CLASSES},
    "vgg": {"rows": vgg_rows, "kernel_math": kernel_math, "channels": C},
    "ladder": ladder,
    "seed_spread": {"depth": 20, "family": "plain", "runs": seed_spread,
                    "train_range": rnd(spread_train), "test_range": rnd(spread_test)},
    "lr_control": lr_control,
    "activations": act_rows,
    "determinism_gap": determinism,
    "grad_profiles": grad_profiles,
    "garments": {"image": "garments.png", "tile": 28, "count": len(pick),
                 "labels": [int(yte_np[i]) for i in pick]},
    "versions": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "width": WIDTH,
                 "optimiser": "SGD momentum 0.9, lr 0.05, weight decay 1e-4, "
                              "x0.1 at 50% and 80% of training"},
}
path = OUT / "depth.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
