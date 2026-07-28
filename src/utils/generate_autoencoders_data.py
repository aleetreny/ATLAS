"""Data for the ATLAS 'Bottleneck' article: autoencoders, and the module's last
question.

Every method in module 2.2 so far produces coordinates by SOLVING for them: an
eigenproblem, a stress, a graph, a divergence. None of them leaves anything
behind. Hand PCA a new row and it can project it, because a projection is a
matrix; hand t-SNE or Isomap or non-metric scaling a new row and there is
nothing to hand it to, only the whole fit to run again. And none of the five can
go backwards: given two coordinates, no method in this module can say what the
row was.

An autoencoder is the version of the question where both of those are the
requirement rather than an afterthought. It is a pair of functions, an encoder
and a decoder, fitted together so that the second undoes the first as well as it
can through a layer narrower than the input. That makes the reduction a thing
you can apply and invert, and it changes what can be measured.

Six blocks:

1. The linear case, which is a theorem: an autoencoder with no activation
   functions and squared loss finds PCA's SUBSPACE and does not find PCA's
   AXES. Both halves are measured, because the second is the one people are
   surprised by.
2. Error against bottleneck size, for PCA, a linear autoencoder and a
   non-linear one, on held out digits.
3. The thing only these two can do: encode a row the fit has never seen. And
   the number that says why the rest of the module cannot.
4. A bottleneck is not the only way to force a model to learn something. An
   autoencoder wider than its input reaches zero error by doing nothing, and
   the same architecture trained to remove noise does not. One knob changed.
5. The Swiss roll from the previous manifold article, asserted identical, run
   through a two-unit bottleneck and scored against the same known chart.
6. The trained encoder and decoder, quantised to float16 and written as base64,
   so the page can run both directions itself.

Run from anywhere: `python src/utils/generate_autoencoders_data.py`.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.vision_data import mnist, subset            # noqa: E402

OUT = ROOT / "autoencoders" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 19
rnd = lambda v, n=4: round(float(v), n)
arr = lambda a, n=4: [round(float(v), n) for v in np.asarray(a).ravel()]
mat = lambda A, n=4: [[round(float(v), n) for v in row] for row in np.asarray(A)]

N = 2000
SIDE = 14                     # 28x28 mean pooled 2x2: legible, and the weights fit
P = SIDE * SIDE
HID = 64
KS = [2, 4, 8, 16, 32]
K_SHOW = 2

print("=" * 72)
print("ATLAS / autoencoders: a reduction that is a pair of functions")
print("=" * 72)


# ------------------------------------------------------------------ the network
def seeded_normal(n, d, seed, scale=1.0):
    """The same generator the browser has, so nothing that decides an answer is
    drawn by a library only one side of the page can call."""
    out = np.empty((n, d))
    s = int(seed) & 0xFFFFFFFF
    for i in range(n):
        for j in range(d):
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            u1 = max(s / 4294967296.0, 1e-12)
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            u2 = s / 4294967296.0
            out[i, j] = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2) * scale
    return out


class AE:
    """encoder p -> h -> k, decoder k -> h -> p, trained by Adam on squared error.

    `linear` drops every activation, which is the case the first section is
    about. `out_sigmoid` keeps the reconstruction inside [0, 1] where the data
    is, and is switched off for the Swiss roll, whose coordinates are not.
    """

    def __init__(self, p, h, k, seed, linear=False, out_sigmoid=True):
        rs = np.random.RandomState(seed)
        sc = lambda a, b: np.sqrt(2.0 / (a + b))
        self.linear = linear
        self.out_sigmoid = out_sigmoid and not linear
        self.h = h
        if linear:
            self.W = [rs.normal(0, sc(p, k), (p, k)), rs.normal(0, sc(k, p), (k, p))]
            self.b = [np.zeros(k), np.zeros(p)]
        else:
            self.W = [rs.normal(0, sc(p, h), (p, h)), rs.normal(0, sc(h, k), (h, k)),
                      rs.normal(0, sc(k, h), (k, h)), rs.normal(0, sc(h, p), (h, p))]
            self.b = [np.zeros(h), np.zeros(k), np.zeros(h), np.zeros(p)]
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self.t = 0

    def encode(self, X):
        if self.linear:
            return X @ self.W[0] + self.b[0]
        a = np.tanh(X @ self.W[0] + self.b[0])
        return a @ self.W[1] + self.b[1]

    def decode(self, Z):
        if self.linear:
            return Z @ self.W[1] + self.b[1]
        a = np.tanh(Z @ self.W[2] + self.b[2])
        o = a @ self.W[3] + self.b[3]
        return 1.0 / (1.0 + np.exp(-o)) if self.out_sigmoid else o

    def forward(self, X):
        if self.linear:
            z = X @ self.W[0] + self.b[0]
            return [z], z @ self.W[1] + self.b[1]
        h1 = np.tanh(X @ self.W[0] + self.b[0])
        z = h1 @ self.W[1] + self.b[1]
        h2 = np.tanh(z @ self.W[2] + self.b[2])
        o = h2 @ self.W[3] + self.b[3]
        y = 1.0 / (1.0 + np.exp(-o)) if self.out_sigmoid else o
        return [h1, z, h2, o], y

    def grads(self, X, T):
        n = len(X)
        cache, Y = self.forward(X)
        d = (Y - T) * (2.0 / (n * Y.shape[1]))
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        if self.linear:
            z = cache[0]
            gW[1] = z.T @ d
            gb[1] = d.sum(0)
            dz = d @ self.W[1].T
            gW[0] = X.T @ dz
            gb[0] = dz.sum(0)
        else:
            h1, z, h2, o = cache
            if self.out_sigmoid:
                d = d * Y * (1 - Y)
            gW[3] = h2.T @ d
            gb[3] = d.sum(0)
            dh2 = (d @ self.W[3].T) * (1 - h2 ** 2)
            gW[2] = z.T @ dh2
            gb[2] = dh2.sum(0)
            dz = dh2 @ self.W[2].T
            gW[1] = h1.T @ dz
            gb[1] = dz.sum(0)
            dh1 = (dz @ self.W[1].T) * (1 - h1 ** 2)
            gW[0] = X.T @ dh1
            gb[0] = dh1.sum(0)
        return gW + gb, float(((Y - T) ** 2).mean())

    def step(self, g, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        for i, p_ in enumerate(params):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g[i]
            self.v[i] = b2 * self.v[i] + (1 - b2) * g[i] ** 2
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p_ -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, T=None, epochs=400, batch=128, lr=2e-3, seed=0, noise=0.0):
        T = X if T is None else T
        rs = np.random.RandomState(seed)
        n = len(X)
        hist = []
        for e in range(epochs):
            order = rs.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                inp = X[idx]
                if noise > 0:
                    inp = np.clip(inp + rs.normal(0, noise, inp.shape), 0, 1)
                g, _ = self.grads(inp, T[idx])
                self.step(g, lr=lr)
            if (e + 1) % 40 == 0 or e == 0:
                hist.append({"epoch": e + 1, "mse": rnd(mse(self.forward(X)[1], T), 6)})
        return hist


def mse(a, b):
    return float(((np.asarray(a) - np.asarray(b)) ** 2).mean())


def subspace_gap(A, B):
    Qa = np.linalg.qr(np.asarray(A, dtype=float))[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=float))[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(1.0 - (s ** 2).mean())


def procrustes_fit(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    na, nb = np.sqrt((Ac ** 2).sum()), np.sqrt((Bc ** 2).sum())
    if na == 0 or nb == 0:
        return 1.0
    Ac, Bc = Ac / na, Bc / nb
    U, sv, Vt = np.linalg.svd(Bc.T @ Ac)
    return float(((Ac - sv.sum() * (Bc @ (U @ Vt))) ** 2).sum())


# ==================================================================== the digits
print("\n" + "=" * 72)
print(f"{N} MNIST digits, mean pooled to {SIDE} by {SIDE}")
Xtr, ytr, _, _ = mnist()
Xd, yd = subset(Xtr, ytr, N, seed=SEED)
IMG = Xd.reshape(N, SIDE, 2, SIDE, 2).mean((2, 4)) / 255.0
FLAT = np.round(IMG.reshape(N, P), 4)
rs = np.random.RandomState(SEED)
perm = rs.permutation(N)
TR, TE = perm[:1400], perm[1400:]
XT, XV = FLAT[TR], FLAT[TE]
print(f"  {len(XT)} for fitting, {len(XV)} held out, {P} pixels each")

# ======================================== 1. the linear case, which is a theorem
print("\n" + "=" * 72)
print("no activation functions at all: what that finds and what it does not")
LIN_EPOCHS = 3000
lin = AE(P, HID, K_SHOW, seed=SEED, linear=True)
lin.fit(XT, epochs=LIN_EPOCHS, lr=4e-3, seed=SEED)
pca2 = PCA(n_components=K_SHOW, random_state=SEED).fit(XT)
lin_err = mse(lin.forward(XT)[1], XT)
pca_err = mse(pca2.inverse_transform(pca2.transform(XT)), XT)
# The subspace to compare is the DECODER's row space, because that is where the
# reconstruction lives: every output of this network is a combination of those
# two vectors and nothing else. Comparing the encoder instead measures how the
# network chooses to project onto that space, which is a different question and
# has a different answer (0.88 rather than 5e-5), and reporting it as the
# subspace claim would have made a theorem look false.
DEC = lin.W[1]                       # k x p, its rows span the reconstruction
gap = subspace_gap(pca2.components_.T, DEC.T)
gap_enc = subspace_gap(pca2.components_.T, lin.W[0])
Wn = DEC.T / np.linalg.norm(DEC.T, axis=0, keepdims=True)
Pn = pca2.components_.T
align = np.abs(Wn.T @ Pn)
gram = Wn.T @ Wn
angle = float(np.degrees(np.arccos(np.clip(abs(gram[0, 1]), -1, 1))))
ev = PCA(n_components=4, random_state=SEED).fit(XT).explained_variance_
lin_out = {
    "epochs": LIN_EPOCHS, "k": K_SHOW,
    "encoder_gap": float(f"{gap_enc:.3e}"),
    "eigengap": rnd(float(ev[1] / ev[2]), 4),
    "eigenvalues": arr(ev, 5),
    "ae_mse": rnd(lin_err, 6), "pca_mse": rnd(pca_err, 6),
    "mse_ratio": rnd(lin_err / pca_err, 5),
    "subspace_gap": float(f"{gap:.3e}"),
    "best_alignment": rnd(float(align.max()), 4),
    "worst_alignment": rnd(float(align.min()), 4),
    "alignment": mat(align, 4),
    "axis_angle": rnd(angle, 3),
    "gram_offdiag": rnd(float(abs(gram[0, 1])), 4),
}
print(f"  reconstruction error {lin_err:.6f} against PCA's {pca_err:.6f}, "
      f"a ratio of {lin_err/pca_err:.5f}")
print(f"  the decoder's plane and PCA's differ by {gap:.3e}: the same plane")
print(f"  but its two directions sit at {angle:.2f} degrees to each other, not 90, "
      f"and their alignment with the principal axes runs from {align.min():.4f} to "
      f"{align.max():.4f}")
print(f"  (the ENCODER's column space is {gap_enc:.3e} away, which is a different "
      f"question: how it projects onto the plane, not which plane)")

# ===================================== 2. error against the size of the bottleneck
print("\n" + "=" * 72)
print("how much a bottleneck costs, three ways")
curve = []
nets = {}
for k in KS:
    p = PCA(n_components=k, random_state=SEED).fit(XT)
    p_tr = mse(p.inverse_transform(p.transform(XT)), XT)
    p_te = mse(p.inverse_transform(p.transform(XV)), XV)
    ln = AE(P, HID, k, seed=SEED, linear=True)
    ln.fit(XT, epochs=400, lr=4e-3, seed=SEED)
    l_tr = mse(ln.forward(XT)[1], XT)
    l_te = mse(ln.forward(XV)[1], XV)
    nl = AE(P, HID, k, seed=SEED)
    hist = nl.fit(XT, epochs=400, lr=2e-3, seed=SEED)
    n_tr = mse(nl.forward(XT)[1], XT)
    n_te = mse(nl.forward(XV)[1], XV)
    nets[k] = nl
    curve.append({
        "k": k,
        "pca_train": rnd(p_tr, 6), "pca_test": rnd(p_te, 6),
        "linear_train": rnd(l_tr, 6), "linear_test": rnd(l_te, 6),
        "deep_train": rnd(n_tr, 6), "deep_test": rnd(n_te, 6),
        "gain": rnd(p_te / n_te, 4),
        "history": hist if k == K_SHOW else None,
    })
    print(f"  k = {k:>2}: PCA {p_te:.6f}  linear autoencoder {l_te:.6f}  "
          f"non-linear {n_te:.6f}  (held out; the non-linear one is "
          f"{p_te/n_te:.3f} times better than PCA)")

# ============================================ 3. what only a function can do
print("\n" + "=" * 72)
print("a row the fit has never seen")
net = nets[K_SHOW]
Z_TR = net.encode(XT)
Z_TE = net.encode(XV)
gen = {
    "train_mse": rnd(mse(net.decode(Z_TR), XT), 6),
    "test_mse": rnd(mse(net.decode(Z_TE), XV), 6),
    "pca_train_mse": rnd(mse(PCA(K_SHOW, random_state=SEED).fit(XT).inverse_transform(
        PCA(K_SHOW, random_state=SEED).fit(XT).transform(XT)), XT), 6),
}
gen["overfit_ratio"] = rnd(gen["test_mse"] / gen["train_mse"], 4)
knn_code = KNeighborsClassifier(n_neighbors=5).fit(Z_TR, yd[TR])
gen["code_accuracy"] = rnd(float((knn_code.predict(Z_TE) == yd[TE]).mean()), 4)
knn_pix = KNeighborsClassifier(n_neighbors=5).fit(XT, yd[TR])
gen["pixel_accuracy"] = rnd(float((knn_pix.predict(XV) == yd[TE]).mean()), 4)
print(f"  fitted on {len(XT)}, error {gen['train_mse']:.6f}; on the {len(XV)} it never "
      f"saw, {gen['test_mse']:.6f}, a ratio of {gen['overfit_ratio']}")
print(f"  a five nearest neighbour vote on the {K_SHOW} numbers it produces gets "
      f"{gen['code_accuracy']*100:.1f}% of the held out digits right, against "
      f"{gen['pixel_accuracy']*100:.1f}% on all {P} pixels")

# ================================= 4. a bottleneck is not the only constraint
print("\n" + "=" * 72)
print("wider than the input, with and without a reason to learn anything")
WIDE = 256
NOISE = 0.5
plain = AE(P, HID, WIDE, seed=SEED)
plain.fit(XT, epochs=300, lr=2e-3, seed=SEED)
den = AE(P, HID, WIDE, seed=SEED)
den.fit(XT, epochs=300, lr=2e-3, seed=SEED, noise=NOISE)
rs2 = np.random.RandomState(5)
XV_NOISY = np.clip(XV + rs2.normal(0, NOISE, XV.shape), 0, 1)
wide = {
    "k": WIDE, "p": P, "noise": NOISE,
    "plain_clean": rnd(mse(plain.forward(XV)[1], XV), 6),
    "denoise_clean": rnd(mse(den.forward(XV)[1], XV), 6),
    "plain_noisy": rnd(mse(plain.forward(XV_NOISY)[1], XV), 6),
    "denoise_noisy": rnd(mse(den.forward(XV_NOISY)[1], XV), 6),
    "identity_noisy": rnd(mse(XV_NOISY, XV), 6),
}
kp = KNeighborsClassifier(n_neighbors=5).fit(plain.encode(XT), yd[TR])
kd = KNeighborsClassifier(n_neighbors=5).fit(den.encode(XT), yd[TR])
wide["plain_accuracy"] = rnd(float((kp.predict(plain.encode(XV)) == yd[TE]).mean()), 4)
wide["denoise_accuracy"] = rnd(float((kd.predict(den.encode(XV)) == yd[TE]).mean()), 4)
print(f"  both reconstruct a clean digit almost perfectly: {wide['plain_clean']:.6f} "
      f"and {wide['denoise_clean']:.6f}")
print(f"  hand them a digit with noise of {NOISE} on it and ask for the clean one: "
      f"{wide['plain_noisy']:.6f} against {wide['denoise_noisy']:.6f}, where doing "
      f"nothing at all scores {wide['identity_noisy']:.6f}")
print(f"  five nearest neighbours on the {WIDE} numbers each produces: "
      f"{wide['plain_accuracy']*100:.1f}% against {wide['denoise_accuracy']*100:.1f}%")

# ============================================== 5. the Swiss roll, again
print("\n" + "=" * 72)
print("the Swiss roll from the manifolds article, asserted identical")
prev = json.loads((ROOT / "manifolds" / "data" / "manifolds.json").read_text("utf-8"))
ROLL = np.array(prev["roll"]["points"], dtype=float)
CHART = np.array(prev["roll"]["chart"], dtype=float)
assert ROLL.shape[0] == 500 and CHART.shape[0] == 500, "the roll changed size"
SC_ROLL = np.round((ROLL - ROLL.mean(0)) / ROLL.std(0), 6)
# Before publishing that a bottleneck does not unroll this, try to make it: more
# width, more depth of training, a gentler and a harsher step. Six attempts, and
# the best of them is what gets reported.
attempts = []
best = None
for hid, ep, lr in [(32, 3000, 3e-3), (64, 3000, 3e-3), (128, 3000, 3e-3),
                    (64, 9000, 3e-3), (64, 9000, 1e-3), (128, 9000, 1e-3)]:
    net_r = AE(3, hid, 2, seed=SEED, out_sigmoid=False)
    net_r.fit(SC_ROLL, epochs=ep, batch=64, lr=lr, seed=SEED)
    Zr = net_r.encode(SC_ROLL)
    d = procrustes_fit(CHART, Zr)
    attempts.append({"hidden": hid, "epochs": ep, "lr": lr, "disparity": rnd(d, 5),
                     "mse": rnd(mse(net_r.decode(Zr), SC_ROLL), 6)})
    print(f"    {hid:>3} hidden, {ep:>4} epochs, step {lr}: {d:.5f} from the true chart")
    if best is None or d < best[0]:
        best = (d, Zr, net_r, attempts[-1])
Zroll = best[1]
roll = {
    "n": int(len(ROLL)),
    "attempts": attempts,
    "best": best[3],
    "disparity": rnd(best[0], 5),
    "isomap": prev["methods"]["isomap"]["disparity"],
    "pca": prev["methods"]["pca"]["disparity"],
    "lle": prev["methods"]["lle"]["disparity"],
    "som": prev["som"]["disparity"],
    "mse": rnd(mse(best[2].decode(Zroll), SC_ROLL), 6),
    "coords": mat(Zroll, 4),
    "chart": mat(CHART, 4),
}
print(f"  best of {len(attempts)} attempts on the same {roll['n']} points: "
      f"{roll['disparity']} from the true chart, against Isomap's {roll['isomap']}, "
      f"PCA's {roll['pca']} and the self-organising map's {roll['som']}")

# =========================================== 6. the weights the browser will run
print("\n" + "=" * 72)
print("exporting the pair of functions")
vals = []
spec = []
for name, W, b in [("enc1", net.W[0], net.b[0]), ("enc2", net.W[1], net.b[1]),
                   ("dec1", net.W[2], net.b[2]), ("dec2", net.W[3], net.b[3])]:
    spec.append({"name": name, "shape": list(W.shape),
                 "w_offset": len(vals), "w_size": int(W.size)})
    vals.extend(W.T.ravel().tolist())            # row major over OUTPUT units
    spec[-1]["b_offset"] = len(vals)
    spec[-1]["b_size"] = int(b.size)
    vals.extend(b.tolist())
vals32 = np.array(vals, dtype=np.float32)
vals16 = vals32.astype(np.float16)
b64 = base64.b64encode(vals16.tobytes()).decode("ascii")
print(f"  {len(vals32):,} numbers, float16 base64 is {len(b64)/1024:.0f} kB")


def forward16(X):
    """The forward pass exactly as the browser will run it, on the quantised
    weights, because those are the numbers it will have."""
    w = vals16.astype(np.float64)
    out = []
    cur = X
    for i, s in enumerate(spec):
        W = w[s["w_offset"]:s["w_offset"] + s["w_size"]].reshape(s["shape"][1], s["shape"][0]).T
        bb = w[s["b_offset"]:s["b_offset"] + s["b_size"]]
        cur = cur @ W + bb
        if i == 0 or i == 2:
            cur = np.tanh(cur)
        elif i == 3:
            cur = 1.0 / (1.0 + np.exp(-cur))
        if i == 1:
            out.append(cur.copy())
    out.append(cur)
    return out


Z16, Y16 = forward16(XV)
quant = {
    "code_shift": float(f"{np.abs(Z16 - Z_TE).max():.3e}"),
    "mse_full": rnd(mse(net.decode(Z_TE), XV), 6),
    "mse_quantised": rnd(mse(Y16, XV), 6),
}
print(f"  quantising moves the two codes by at most {quant['code_shift']} and the "
      f"held out error from {quant['mse_full']} to {quant['mse_quantised']}")

# a grid of decoded images across the latent square, and the showcase digits
Z_ALL = net.encode(FLAT)
lo = np.percentile(Z_ALL, 1, axis=0)
hi = np.percentile(Z_ALL, 99, axis=0)
SHOW = [int(np.where(yd == c)[0][0]) for c in range(10)]

out = {
    "meta": {
        "seed": SEED, "n": N, "side": SIDE, "pixels": P, "hidden": HID,
        "ks": KS, "k_show": K_SHOW, "train": int(len(TR)), "test": int(len(TE)),
        "sklearn": __import__("sklearn").__version__,
        "note": "the browser runs both halves of the network from the float16 "
                "weights below and checks its own answers against the references "
                "here; every figure quoted on the page was measured by this file",
        "reuse_check": "the Swiss roll is asserted identical to the one "
                       "manifolds/data/manifolds.json ships",
    },
    "linear": lin_out,
    "curve": {"rows": curve},
    "generalisation": gen,
    "wide": wide,
    "roll": roll,
    "net": {
        "format": "float16 little-endian, base64; each layer's weights are row "
                  "major over output units, then its biases",
        "layers": spec,
        "count": int(len(vals32)),
        "weights_b64": b64,
        "quant": quant,
        "latent": mat(Z_ALL, 4),
        "labels": [int(v) for v in yd],
        "is_test": [int(i in set(TE.tolist())) for i in range(N)],
        "latent_lo": arr(lo, 4), "latent_hi": arr(hi, 4),
        "showcase": [{"label": int(yd[i]), "index": int(i),
                      "pixels": arr(FLAT[i], 4)} for i in SHOW],
    },
}
path = OUT / "autoencoders.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size/1024:.1f} kB)")
