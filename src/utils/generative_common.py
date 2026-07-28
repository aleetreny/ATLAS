"""Shared ground for module 3.2: the variational autoencoder, the discrete
codebook, the energy based model, normalising flows and Glow.

Five articles that all answer the same question (what is the distribution these
rows came from) with five different objects, so they are only comparable if
they are scored on the same thing. Two datasets do that here:

1. **A two dimensional density written in closed form.** Two Gaussian blobs and
   a ring, with weights that sum to one, so the truth is known exactly. The
   reason it is two dimensional is not simplicity: it is that in two dimensions
   the normalising constant is a sum. Every quantity the rest of the module has
   to approximate (the partition function of an energy model, the marginal
   likelihood a variational bound only bounds, the KL between a fit and the
   truth) is computed here by quadrature on a grid, to a precision this file
   measures rather than assumes. The ring matters too: it is the piece a
   diffeomorphism cannot produce from a Gaussian, and the flow article measures
   what that costs.
2. **The 2,000 MNIST digits of the autoencoders article**, mean pooled to
   14 by 14, rebuilt here from the same seed and then *asserted* to be the same
   ones: not by comparing a checksum of the file, but by running the float16
   network that article shipped on the digits this file just built and checking
   that its codes come back. If the digits had moved, the codes would not.

Everything expensive is cached under ~/.atlas_vision_data/gen32_cache/ with a
key that contains the whole configuration, so a container that gets recycled in
the middle of a run costs nothing.

Nothing in here publishes a wall clock time. Thread counts change the order of
floating point reductions, so the number of threads is recorded in every
article's JSON and the number of seconds is never a headline.
"""
import base64
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = Path.home() / ".atlas_vision_data" / "gen32_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 19

rnd = lambda v, n=4: round(float(v), n)
arr = lambda a, n=4: [round(float(v), n) for v in np.asarray(a).ravel()]
mat = lambda A, n=4: [[round(float(v), n) for v in row] for row in np.asarray(A)]
sci = lambda v, n=3: float(f"%.{n}e" % float(v))


# ---------------------------------------------------------------- threads
def threads():
    """min(N, cores), never a bare number.

    Oversubscribing is not just slow: it changes the order of the floating
    point reductions, so the same network gives a different fifth decimal. The
    module records the number it used in every JSON it writes.
    """
    n = min(4, os.cpu_count() or 4)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    return n


# ---------------------------------------------------------------- caching
def cached(key, build, quiet=False):
    """Run `build()` once and keep the answer on disk under `key`.

    The key is expected to carry the entire configuration of the experiment, so
    that changing a hyperparameter invalidates it by itself and no one has to
    remember to clear anything.
    """
    tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"{tag}.pkl"
    if path.exists():
        if not quiet:
            print(f"    [cache] {key}")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    out = build()
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


# ======================================================== the closed form truth
#
# A mixture of three parts, each of which integrates to one on its own, so the
# weights are the only thing that has to add up and the density below is
# normalised by construction rather than by a fitted constant.
#
#   * a ring: radius Gaussian around r0, angle uniform. Its 2-D density is
#     phi((r - r0) / s) / (2 pi r s), which integrates to one up to the mass
#     the radial Gaussian puts below zero: with r0 / s above eleven that is
#     smaller than 1e-27 and the quadrature check below confirms it.
#   * a round blob.
#   * a thin blob at an angle, which is the piece that makes a flow work for
#     its living: it has to rotate as well as scale.
TOY = {
    "ring": {"weight": 0.5, "r0": 2.0, "sigma": 0.16},
    "blob_a": {"weight": 0.25, "mean": [-2.55, 2.55], "sd": [0.35, 0.35], "angle": 0.0},
    "blob_b": {"weight": 0.25, "mean": [2.55, -2.55], "sd": [0.62, 0.17], "angle": 35.0},
    # The square the quadrature runs over. It is wider than anything anyone
    # would draw on purpose: at 5.0 the long axis of the thin blob still had
    # 2.7e-07 of its mass outside, and that truncation, not the cell width,
    # was the error floor of every integral in the module.
    "domain": 6.5,
    # What the pictures show. Nothing is measured on this box.
    "view": 4.2,
}


def _rot(deg):
    """The rotation that takes a blob's own axes to the plane's."""
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def toy_pdf(X):
    """The exact density of the mixture at the rows of X, shape (n, 2)."""
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    R = TOY["ring"]
    r = np.sqrt((X ** 2).sum(1))
    r = np.maximum(r, 1e-12)
    ring = (np.exp(-0.5 * ((r - R["r0"]) / R["sigma"]) ** 2)
            / (2.0 * np.pi * r * R["sigma"] * np.sqrt(2.0 * np.pi)))
    p = R["weight"] * ring
    for name in ("blob_a", "blob_b"):
        B = TOY[name]
        Q = _rot(B["angle"])
        s = np.asarray(B["sd"], dtype=float)
        d = (X - np.asarray(B["mean"], dtype=float)) @ Q          # into blob axes
        q = ((d / s) ** 2).sum(1)
        p = p + B["weight"] * np.exp(-0.5 * q) / (2.0 * np.pi * s[0] * s[1])
    return p


def toy_logpdf(X):
    return np.log(np.maximum(toy_pdf(X), 1e-300))


def toy_sample(n, seed=SEED):
    """Exact samples: pick a component, then sample it in closed form."""
    rs = np.random.RandomState(seed)
    w = np.array([TOY["ring"]["weight"], TOY["blob_a"]["weight"], TOY["blob_b"]["weight"]])
    which = rs.choice(3, size=n, p=w)
    out = np.empty((n, 2))
    m = which == 0
    R = TOY["ring"]
    rad = R["r0"] + R["sigma"] * rs.randn(int(m.sum()))
    ang = rs.uniform(0, 2 * np.pi, int(m.sum()))
    out[m] = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    for i, name in ((1, "blob_a"), (2, "blob_b")):
        B = TOY[name]
        k = which == i
        z = rs.randn(int(k.sum()), 2) * np.asarray(B["sd"], dtype=float)
        out[k] = z @ _rot(B["angle"]).T + np.asarray(B["mean"], dtype=float)
    return out


def toy_grid(m=1024, half=None):
    """Midpoint grid over the square, and the area of one cell.

    Midpoint (equivalently trapezoid) quadrature on a smooth density that
    decays at the edges converges exponentially in the cell width rather than
    quadratically, which is why a grid this coarse can carry a normalising
    constant to twelve decimal places. `toy_quadrature_report` measures it
    rather than claiming it.
    """
    half = TOY["domain"] if half is None else half
    e = np.linspace(-half, half, m + 1)
    c = 0.5 * (e[1:] + e[:-1])
    gx, gy = np.meshgrid(c, c, indexing="ij")
    X = np.c_[gx.ravel(), gy.ravel()]
    return X, float((2.0 * half / m) ** 2), (m, m)


def toy_quadrature_report(ms=(512, 1024, 2048)):
    """What the grid is worth, measured: the mixture integrates to one in
    closed form, so the distance from one is the error of everything else this
    module computes on the same grid."""
    rows = []
    for m in ms:
        X, cell, _ = toy_grid(m)
        z = float(toy_pdf(X).sum() * cell)
        rows.append({"m": m, "mass": float(f"{z:.15f}"), "error": sci(abs(z - 1.0))})
    return rows


def toy_entropy(m=1024):
    """-E_p[log p] under the truth, by quadrature: the ceiling every fit in the
    module is measured against, since no model can have a higher expected log
    density under p than p itself."""
    X, cell, _ = toy_grid(m)
    p = toy_pdf(X)
    return float(-(p * np.log(np.maximum(p, 1e-300))).sum() * cell)


def toy_kl(log_q_grid, m=1024):
    """KL(truth || model), exactly, from the model's log density on the grid.

    Not an estimate from samples: the integral is the sum, and the error is the
    one `toy_quadrature_report` prints.
    """
    X, cell, _ = toy_grid(m)
    p = toy_pdf(X)
    lq = np.asarray(log_q_grid, dtype=float).ravel()
    return float((p * (np.log(np.maximum(p, 1e-300)) - lq)).sum() * cell)


def toy_mass(region, m=1024):
    """The truth's mass inside a region given as a boolean function of X."""
    X, cell, _ = toy_grid(m)
    return float((toy_pdf(X) * region(X)).sum() * cell)


# ============================================================ the shared digits
DIGITS = {"n": 2000, "side": 14, "seed": SEED, "train": 1400}


def digits14(assert_against_autoencoders=True):
    """The 2,000 digits of the autoencoders article, rebuilt and then proved.

    Returns (FLAT, labels, train_idx, test_idx) with FLAT of shape (2000, 196)
    in [0, 1], rounded to four decimals exactly as that article rounded them,
    because those are the numbers its network was fitted on.

    The proof is the interesting part. Reusing another article's data is a
    claim, and a claim gets asserted: this runs the float16 encoder that
    article shipped in its JSON on the digits built here and checks the codes
    against the ones it published. Pixels that had moved by a thousandth would
    show up there.
    """
    from src.utils.vision_data import mnist, subset

    n, side = DIGITS["n"], DIGITS["side"]
    p = side * side
    Xtr, ytr, _, _ = mnist()
    Xd, yd = subset(Xtr, ytr, n, seed=DIGITS["seed"])
    img = Xd.reshape(n, side, 2, side, 2).mean((2, 4)) / 255.0
    flat = np.round(img.reshape(n, p), 4)
    rs = np.random.RandomState(DIGITS["seed"])
    perm = rs.permutation(n)
    tr, te = perm[:DIGITS["train"]], perm[DIGITS["train"]:]

    if assert_against_autoencoders:
        ae = json.loads((ROOT / "autoencoders" / "data" / "autoencoders.json")
                        .read_text("utf-8"))
        assert list(ae["net"]["labels"]) == [int(v) for v in yd], \
            "the labels of the shared digits moved"
        for s in ae["net"]["showcase"]:
            assert np.allclose(np.array(s["pixels"]), flat[s["index"]], atol=0), \
                "a showcase digit's pixels moved"
        w = decode_half(ae["net"]["weights_b64"], ae["net"]["count"])
        e1, e2 = ae["net"]["layers"][0], ae["net"]["layers"][1]
        h = np.tanh(dense(w, e1, flat))
        z = dense(w, e2, h)
        worst = float(np.abs(z - np.array(ae["net"]["latent"], dtype=float)).max())
        # The tolerance is not a guess: that article measured what quantising
        # its own weights to float16 does to its own codes (`code_shift`), and
        # this encoder is the quantised one while the codes it published came
        # from the full precision network. So the floor of this comparison is
        # exactly that number, and anything above it is a real difference in
        # the pixels. Digits that were not these would miss by units, not by
        # thousandths.
        floor = float(ae["net"]["quant"]["code_shift"])
        assert worst <= 1.5 * floor, (
            f"the autoencoder article's own encoder does not reproduce its own "
            f"codes on these digits (worst {worst:.2e} against a quantisation "
            f"floor of {floor:.2e}): they are not the same digits")
        DIGITS["assert_worst"] = sci(worst)
        DIGITS["assert_floor"] = floor
    return flat, np.asarray(yd, dtype=int), tr, te


def fashion14(n=6000, seed=SEED):
    """Fashion-MNIST through the same 2x2 mean pooling, for the article that
    needs a second dataset to ask a likelihood which one it prefers."""
    from src.utils.vision_data import fashion_mnist, subset

    side = DIGITS["side"]
    Xtr, ytr, Xte, yte = fashion_mnist()
    Xd, yd = subset(Xtr, ytr, n, seed=seed)
    img = Xd.reshape(len(Xd), side, 2, side, 2).mean((2, 4)) / 255.0
    return np.round(img.reshape(len(Xd), side * side), 4), np.asarray(yd, dtype=int)


# ====================================================== the float16 export
def export_half(blocks):
    """The export contract the detection article introduced, unchanged.

    `blocks` is a list of (name, W, b) with W of shape (inputs, outputs); the
    weights are written row major over OUTPUT units so the browser reads a
    layer with a stride and no transpose. Returns (spec, base64, values16).
    """
    vals, spec = [], []
    for name, W, b in blocks:
        W = np.asarray(W, dtype=np.float64)
        spec.append({"name": name, "shape": [int(W.shape[0]), int(W.shape[1])],
                     "w_offset": len(vals), "w_size": int(W.size)})
        vals.extend(W.T.ravel().tolist())
        if b is None:
            spec[-1]["b_offset"] = len(vals)
            spec[-1]["b_size"] = 0
        else:
            b = np.asarray(b, dtype=np.float64).ravel()
            spec[-1]["b_offset"] = len(vals)
            spec[-1]["b_size"] = int(b.size)
            vals.extend(b.tolist())
    v16 = np.array(vals, dtype=np.float32).astype(np.float16)
    return spec, base64.b64encode(v16.tobytes()).decode("ascii"), v16


def export_flat(values):
    """A bare float16 payload for tensors that are not a dense layer."""
    v16 = np.array(np.asarray(values, dtype=np.float64).ravel(),
                   dtype=np.float32).astype(np.float16)
    return base64.b64encode(v16.tobytes()).decode("ascii"), v16


def decode_half(b64, n):
    """float16 base64 back to float64, the way the browser will read it."""
    buf = base64.b64decode(b64)
    return np.frombuffer(buf, dtype=np.float16, count=n).astype(np.float64)


def dense(w, spec, X):
    """One dense layer out of a decoded weight vector, matching the browser."""
    nin, nout = spec["shape"]
    W = w[spec["w_offset"]:spec["w_offset"] + spec["w_size"]].reshape(nout, nin).T
    out = np.asarray(X, dtype=float).reshape(-1, nin) @ W
    if spec["b_size"]:
        out = out + w[spec["b_offset"]:spec["b_offset"] + spec["b_size"]]
    return out


# ============================================================ the shared judge
JUDGE = {"hidden": 96, "epochs": 60, "seed": 7}


def judge():
    """One classifier, trained once, used by every article in the module that
    needs an opinion about whether a generated image looks like a digit.

    It is deliberately small and deliberately shared: a sample score is only
    comparable across three articles if the thing doing the scoring is the same
    object. Returns a dict with a `predict_proba`-style function, its held out
    accuracy, and the confidence it assigns to real held out digits, which is
    the reference any generated sample is read against.
    """
    def build():
        import torch
        import torch.nn as nn

        threads()
        X, y, tr, te = digits14()
        g = torch.Generator().manual_seed(JUDGE["seed"])
        torch.manual_seed(JUDGE["seed"])
        net = nn.Sequential(nn.Linear(X.shape[1], JUDGE["hidden"]), nn.ReLU(),
                            nn.Linear(JUDGE["hidden"], 10))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        Xt = torch.tensor(X[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr])
        for _ in range(JUDGE["epochs"]):
            perm = torch.randperm(len(Xt), generator=g)
            for s in range(0, len(Xt), 128):
                idx = perm[s:s + 128]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        with torch.no_grad():
            logits = net(torch.tensor(X[te], dtype=torch.float32))
            prob = torch.softmax(logits, 1)
            acc = float((logits.argmax(1) == torch.tensor(y[te])).double().mean())
            conf = float(prob.max(1).values.mean())
        W = [p.detach().numpy().copy() for p in net.parameters()]
        return {"W": W, "accuracy": acc, "real_confidence": conf}

    out = cached(f"judge-{JUDGE['hidden']}-{JUDGE['epochs']}-{JUDGE['seed']}-v1", build)

    W1, b1, W2, b2 = out["W"]

    def proba(X):
        h = np.maximum(np.asarray(X, dtype=float).reshape(-1, W1.shape[1]) @ W1.T + b1, 0)
        z = h @ W2.T + b2
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    return {"proba": proba, "accuracy": out["accuracy"],
            "real_confidence": out["real_confidence"],
            "hidden": JUDGE["hidden"], "epochs": JUDGE["epochs"]}


# ============================================================ small utilities
def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
    return path


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)
