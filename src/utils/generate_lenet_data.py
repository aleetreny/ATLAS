"""Data for the ATLAS 'Convolutions and LeNet' web article (module 1.3).

The previous two articles designed the representation by hand. This one lets
the model design it, and measures what that buys, on the same handwritten
digits the classical-features article benchmarked HOG on.

Measured here:
  * a convolutional network against a fully connected one with the SAME
    parameter count, trained identically, so the comparison is about
    structure rather than capacity;
  * what happens to both when the test digits are shifted a few pixels, which
    is the property convolution is supposed to buy and the one a table of
    accuracies never shows;
  * AlexNet's list of ingredients, each switched on and off on our own small
    network: ReLU against tanh, dropout, and training-time shifts;
  * the first-layer filters the network ended up with, exported so the
    browser can convolve with them live and check itself against the feature
    maps torch produced.

Run from anywhere: `python src/utils/generate_lenet_data.py`.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_data import mnist, subset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lenet" / "data"
IMG = ROOT / "lenet" / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19
N_TRAIN = 12000
N_TEST = 5000
EPOCHS = 8
BATCH = 128

# Capped at the machine's core count. On the 14-core box that produced the
# published figures this is still 10, so the reduction order and every number
# are unchanged; on a smaller machine it avoids oversubscribing, which costs
# 4.7x in wall clock and quietly moves the fifth decimal. See docs/verificacion.md.
torch.set_num_threads(min(10, os.cpu_count() or 10))
torch.manual_seed(SEED)

Xtr_u8, ytr_np, Xte_u8, yte_np = mnist()
Xtr_u8, ytr_np = subset(Xtr_u8, ytr_np, N_TRAIN, seed=SEED)
Xte_u8, yte_np = subset(Xte_u8, yte_np, N_TEST, seed=SEED)
Xtr = torch.tensor(Xtr_u8, dtype=torch.float32).unsqueeze(1) / 255.0
Xte = torch.tensor(Xte_u8, dtype=torch.float32).unsqueeze(1) / 255.0
ytr = torch.tensor(ytr_np, dtype=torch.long)
yte = torch.tensor(yte_np, dtype=torch.long)
print(f"MNIST: {len(ytr)} train / {len(yte)} test, 28x28")


class LeNet(nn.Module):
    """LeCun's 1998 shape, with today's ReLU and max pooling."""

    def __init__(self, act=nn.ReLU, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.act = act()
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), act(), nn.Dropout(dropout),
            nn.Linear(120, 84), act(), nn.Dropout(dropout), nn.Linear(84, 10),
        )

    def features(self, x):
        return self.act(self.conv1(x))

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        return self.head(x)


class MLP(nn.Module):
    """Every pixel wired to every hidden unit: no notion of neighbours."""

    def __init__(self, hidden, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), act(), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


def count(m):
    return sum(p.numel() for p in m.parameters())


def shifted(X, dr, dc):
    """Translate every image, filling the vacated strip with background, the
    way a camera moving would, rather than the way torch.roll wraps content
    around from the opposite edge."""
    out = torch.zeros_like(X)
    H, W = X.shape[2], X.shape[3]
    r0, r1 = max(0, dr), min(H, H + dr)
    c0, c1 = max(0, dc), min(W, W + dc)
    out[:, :, r0:r1, c0:c1] = X[:, :, r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def train(model, X, y, epochs=EPOCHS, seed=SEED, augment=False, log=False):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            xb = X[idx]
            if augment:
                # A shift of at most two pixels, the cheapest augmentation there
                # is, and done with exactly the same operation the test uses:
                # training on wrapped digits and testing on shifted ones would
                # be measuring two different transformations.
                sr = int(torch.randint(-2, 3, (1,), generator=g))
                sc = int(torch.randint(-2, 3, (1,), generator=g))
                xb = shifted(xb, sr, sc)
            opt.zero_grad()
            loss = lossf(model(xb), y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        history.append(rnd(tot / len(y), 5))
        if log:
            print(f"    epoch {ep + 1}: train loss {history[-1]:.4f}")
    return history, time.perf_counter() - t0


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    out = []
    for i in range(0, len(y), 1000):
        out.append(model(X[i:i + 1000]).argmax(1))
    return float((torch.cat(out) == y).float().mean())


# ============================== structure against capacity, same parameters
lenet = LeNet()
n_lenet = count(lenet)
hidden = round((n_lenet - 10) / 795)
mlp = MLP(hidden)
print(f"\nLeNet has {n_lenet:,} parameters; an MLP with {hidden} hidden units has {count(mlp):,}")

print("training LeNet")
hist_cnn, t_cnn = train(lenet, Xtr, ytr, log=True)
acc_cnn = accuracy(lenet, Xte, yte)
print(f"  test accuracy {acc_cnn:.4f} in {t_cnn:.1f}s")

print("training the MLP")
hist_mlp, t_mlp = train(mlp, Xtr, ytr)
acc_mlp = accuracy(mlp, Xte, yte)
print(f"  test accuracy {acc_mlp:.4f} in {t_mlp:.1f}s")

# ============================================ what a few pixels of shift cost
print("\nshifting the test digits (never the training ones):")
shift_rows = []
for d in range(0, 7):
    Xs = shifted(Xte, d, d)
    a_cnn = accuracy(lenet, Xs, yte)
    a_mlp = accuracy(mlp, Xs, yte)
    shift_rows.append({"shift": d, "cnn": rnd(a_cnn, 4), "mlp": rnd(a_mlp, 4)})
    print(f"  {d} px: convolutional {a_cnn:.4f}   fully connected {a_mlp:.4f}")

# ================================= AlexNet's ingredients, one at a time, ours
print("\nAlexNet's ingredients, measured on this small network:")
ingredients = []

# ReLU against tanh: the same net, the same seed, only the nonlinearity changed
hist_relu = hist_cnn
tanh_net = LeNet(act=nn.Tanh)
hist_tanh, _ = train(tanh_net, Xtr, ytr)
acc_tanh = accuracy(tanh_net, Xte, yte)
target = max(hist_relu[-1], hist_tanh[-1]) * 1.5
ep_relu = next((i + 1 for i, v in enumerate(hist_relu) if v <= target), None)
ep_tanh = next((i + 1 for i, v in enumerate(hist_tanh) if v <= target), None)
print(f"  ReLU   final loss {hist_relu[-1]:.4f}, test {acc_cnn:.4f}, reached {target:.4f} at epoch {ep_relu}")
print(f"  tanh   final loss {hist_tanh[-1]:.4f}, test {acc_tanh:.4f}, reached {target:.4f} at epoch {ep_tanh}")
ingredients.append({"name": "ReLU instead of tanh", "acc": rnd(acc_cnn, 4), "baseline": rnd(acc_tanh, 4),
                    "note": f"same net, same seed; epochs to reach a training loss of {target:.3f}: "
                            f"{ep_relu} against {ep_tanh}"})

# dropout, where it can actually matter: a small training set
Xsm, ysm = Xtr[:2000], ytr[:2000]
small_plain = LeNet()
train(small_plain, Xsm, ysm, epochs=20)
a_plain = accuracy(small_plain, Xte, yte)
small_drop = LeNet(dropout=0.5)
train(small_drop, Xsm, ysm, epochs=20)
a_drop = accuracy(small_drop, Xte, yte)
print(f"  dropout 0.5 on 2,000 digits: {a_drop:.4f} against {a_plain:.4f} without")
ingredients.append({"name": "dropout on the classifier head", "acc": rnd(a_drop, 4), "baseline": rnd(a_plain, 4),
                    "note": "trained on 2,000 digits for 20 epochs, where a 62k-parameter net can overfit"})

# augmentation: shifts at training time, judged on shifted test digits
aug_net = LeNet()
train(aug_net, Xtr, ytr, augment=True)
a_aug_clean = accuracy(aug_net, Xte, yte)
a_aug_shift = accuracy(aug_net, shifted(Xte, 4, 4), yte)
a_cnn_shift4 = shift_rows[4]["cnn"]
print(f"  training-time shifts: clean {a_aug_clean:.4f} (plain {acc_cnn:.4f}), "
      f"4 px shifted {a_aug_shift:.4f} (plain {a_cnn_shift4:.4f})")
ingredients.append({"name": "shifts during training", "acc": rnd(a_aug_shift, 4), "baseline": a_cnn_shift4,
                    "note": f"judged on test digits shifted 4 px; on clean digits it scores "
                            f"{a_aug_clean:.4f} against {acc_cnn:.4f}"})

# ============================================== what the first layer learned
w = lenet.conv1.weight.detach().numpy()[:, 0]        # (6, 5, 5)
b = lenet.conv1.bias.detach().numpy()
print(f"\nfirst layer: {w.shape[0]} filters of {w.shape[1]}x{w.shape[2]}")

# eight test digits, one per row of a sprite, plus their reference maps
rs = np.random.RandomState(SEED)
pick = []
for c in range(8):
    pick.append(int(np.where(yte_np == c)[0][0]))
digits_u8 = Xte_u8[pick]
sprite = np.concatenate(list(digits_u8), axis=1)      # 28 x (8*28)
Image.fromarray(sprite.astype(np.uint8), mode="L").save(IMG / "digits.png")

with torch.no_grad():
    ref_maps = lenet.features(Xte[pick]).numpy()      # (8, 6, 28, 28) after ReLU
print(f"reference feature maps {ref_maps.shape}, "
      f"range {ref_maps.min():.3f} to {ref_maps.max():.3f}")

# the photograph the kernel editor works on
from skimage import data as skdata  # noqa: E402
from skimage.transform import resize as skresize  # noqa: E402

cam = skresize(skdata.camera() / 255.0, (160, 160), anti_aliasing=True)
cam_u8 = np.clip(np.round(cam * 255), 0, 255).astype(np.uint8)
Image.fromarray(cam_u8, mode="L").save(IMG / "kernel-source.png")
print(f"kernel editor image: 160x160, {(IMG / 'kernel-source.png').stat().st_size / 1024:.0f} kB")

out = {
    "data": {"n_train": int(len(ytr)), "n_test": int(len(yte))},
    "models": {
        "cnn": {"name": "LeNet", "params": int(n_lenet), "acc": rnd(acc_cnn, 4),
                "seconds": rnd(t_cnn, 1), "history": hist_cnn},
        "mlp": {"name": "fully connected", "params": int(count(mlp)), "hidden": int(hidden),
                "acc": rnd(acc_mlp, 4), "seconds": rnd(t_mlp, 1), "history": hist_mlp},
        "tanh_history": hist_tanh,
    },
    "shift": shift_rows,
    "ingredients": ingredients,
    "conv1": {
        # Seven decimals, not five: the browser recomputes these maps and
        # prints its agreement with the reference, so the exported precision
        # is what that number ends up measuring.
        "weights": [[[rnd(v, 7) for v in row] for row in f] for f in w],
        "bias": [rnd(v, 7) for v in b],
        "n_filters": int(w.shape[0]), "k": int(w.shape[1]),
    },
    "digits": {"image": "digits.png", "tile": 28, "count": len(pick),
               "labels": [int(yte_np[i]) for i in pick]},
    "ref_maps": {"digit": 0, "maps": [[[rnd(v, 6) for v in row] for row in m] for m in ref_maps[0]]},
    "kernel_image": {"image": "kernel-source.png", "size": 160},
    "versions": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH},
}
(OUT / "conv.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {OUT / 'conv.json'}  ({(OUT / 'conv.json').stat().st_size / 1024:.1f} kB)")
