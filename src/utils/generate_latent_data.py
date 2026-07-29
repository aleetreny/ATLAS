"""Data for the ATLAS latent diffusion article: the same process, somewhere smaller.

The previous article measured the price of diffusion in pixel space and it was
steep: on 14 by 14 digits, reaching a sample statistic worth quoting took
743,108 weights, seventy seven times the network that runs the same process in
two dimensions. The move everybody makes next is to stop working in pixels: put
an autoencoder in front, run the diffusion in the small space it learned, and
let the decoder put the detail back.

This file measures what that buys and what it costs, and the second half is the
part that usually goes unsaid. A latent model cannot produce anything its own
decoder cannot produce, so its quality has a CEILING that has nothing to do with
diffusion at all, and that ceiling is measurable before any diffusion is
trained: encode real held out digits, decode them again, and score what comes
back. Every latent diffusion result in the world sits under its own version of
that number.

A. The space is the variational autoencoder article's own, and that is proved
   rather than declared: its shipped float16 encoder is run here and has to
   reproduce the codes it published, to its own quantisation floor.
B. The ceiling. Round trips through each autoencoder, scored against the same
   floor the previous article used, before diffusion enters.
C. The ladder, against the previous article's. Diffusion in the latent at
   several widths, counted in total weights (the score network AND the decoder
   it needs), against the pixel ladder measured next door.
D. How big the latent has to be. Autoencoders at several bottleneck sizes: the
   ceiling moves, the diffusion gets harder, and the two move in opposite
   directions, which is the whole design problem in one chart.
E. What the browser runs.

Run from anywhere: `python src/utils/generate_latent_data.py`.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.generative_common import (          # noqa: E402
    ROOT as REPO, SEED, arr, banner, c2st, cached, digits14, export_half,
    judge, judge_ink_check, mat, rbf_mmd2, real_vs_real_floor, rnd, sci,
    threads, write_json)

OUT = REPO / "latent-diffusion" / "data"
THREADS = threads()

KS = [2, 4, 8, 16]
K_SHOW = 8
HID = 64                      # the autoencoders article's width, kept
AE_EPOCHS = 600
DIFF_STEPS = 200
DIFF_WIDTHS = [(48, 2, 1500), (96, 3, 3000), (160, 3, 6000)]
SEED_MAIN = SEED

print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED_MAIN}")

X_DIG, Y_DIG, TR, TE = digits14()
J = judge()
print(f"  the module's shared judge: {J['accuracy'] * 100:.2f}% on held out digits and "
      f"{J['real_confidence']:.4f} confident on them")


# ============================== A. the space is the previous article's, proved
banner("A. the latent space is the variational autoencoder's, and here is the proof")
VJ = json.loads((REPO / "vae" / "data" / "vae.json").read_text("utf-8"))
VN = VJ["net"]
VW = np.frombuffer(base64.b64decode(VN["weights_b64"]),
                   dtype=np.float16, count=VN["count"]).astype(np.float64)
VSPEC = {l["name"]: l for l in VN["layers"]}


def vae_dense(x, name, act=None):
    s = VSPEC[name]
    nin, nout = s["shape"]
    w = VW[s["w_offset"]:s["w_offset"] + s["w_size"]].reshape(nout, nin)
    b = VW[s["b_offset"]:s["b_offset"] + s["b_size"]] if s["b_size"] else 0.0
    y = x @ w.T + b
    return np.tanh(y) if act == "tanh" else y


vae_mu = vae_dense(vae_dense(X_DIG, "enc1", "tanh"), "emu")
published = np.asarray(VN["mu"], dtype=float)
worst_code = float(np.abs(vae_mu - published).max())
floor_code = float(VN["quant"]["code_shift"])
print(f"  running that article's own float16 encoder on these digits reproduces its published "
      f"codes to {worst_code:.7f}, against the quantisation floor of {floor_code} it measured")
assert worst_code <= 1.5 * floor_code, (
    f"the variational autoencoder article's own encoder does not reproduce its own codes "
    f"({worst_code:.2e} against a floor of {floor_code:.2e}): this is not its latent space")
SHARED = {"worst": sci(worst_code), "floor": sci(floor_code),
          "k": int(VSPEC["emu"]["shape"][1]), "count": int(VN["count"])}


# ==================================================== the autoencoders to test
class AE(nn.Module):
    """The bottleneck article's shape, unchanged: 196 to 64 to k and back, tanh
    inside and a logistic output, so the ceiling measured here is comparable to
    the numbers that article published."""

    def __init__(self, k, hid=HID, dim=196):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Linear(hid, k))
        self.dec = nn.Sequential(nn.Linear(k, hid), nn.Tanh(),
                                 nn.Linear(hid, dim), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(k, seed=SEED_MAIN, epochs=AE_EPOCHS):
    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        Xt = torch.tensor(X_DIG[TR], dtype=torch.float32)
        net = AE(k)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for s in range(0, len(Xt), 128):
                xb = Xt[perm[s:s + 128]]
                opt.zero_grad()
                loss = ((net(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
            sch.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()],
                "loss": float(loss.detach())}

    out = cached(f"ld-ae-k{k}-h{HID}-e{epochs}-s{seed}-v1", build)
    net = AE(k)
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out


REALPIX = X_DIG[TE]
GAMMA = 1.0 / (2.0 * float(np.median(
    np.linalg.norm(REALPIX[:300, None, :] - REALPIX[None, :300, :], axis=2)) ** 2))
FLOOR = real_vs_real_floor(REALPIX, GAMMA, splits=5, n=250, seed=17)
print(f"  the floor, real digits against real digits: two sample {FLOOR['mmd2']} "
      f"(largest {FLOOR['mmd2_max']}), separable at {FLOOR['c2st']}")


# ================================================ B. the ceiling, before diffusion
banner("B. the ceiling a latent model cannot pass, measured before any diffusion")
ceiling_rows = {}
aes = {}
for k in KS:
    net, meta = train_ae(k)
    aes[k] = net
    with torch.no_grad():
        Z = net.enc(torch.tensor(REALPIX, dtype=torch.float32))
        back = net.dec(Z).numpy()
    m = rbf_mmd2(back[:250], REALPIX[:500], GAMMA)
    c = c2st(back[:250], REALPIX[:250], seed=3)
    mse = float(((back - REALPIX) ** 2).mean())
    npar = sum(p.numel() for p in net.parameters())
    ndec = sum(p.numel() for p in net.dec.parameters())
    ceiling_rows[k] = {"k": k, "mse": rnd(mse, 6), "mmd2": sci(m), "c2st": rnd(c, 4),
                       "params": npar, "decoder_params": ndec,
                       "resolves": bool(abs(m) > FLOOR["mmd2_max"])}
    print(f"  k={k:>2}: round trip mse {mse:.6f}, two sample {m:+.4e}, separable at {c:.4f}"
          f"{'' if abs(m) > FLOOR['mmd2_max'] else '  <- inside the floor'}")
print("  Nothing that samples in these spaces can score better than the row it sits in: the "
      "decoder is the last thing that touches the picture.")


# ================================================== the diffusion, in a latent
def schedule_cos(steps):
    s = 0.008
    t = np.linspace(0, 1, steps + 1)
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return np.cumprod(1 - np.clip(1 - ab[1:] / ab[:-1], 0, 0.999))


DABAR = schedule_cos(DIFF_STEPS)


class LatentScore(nn.Module):
    def __init__(self, k, hid, depth):
        super().__init__()
        layers = [nn.Linear(k + 32, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        layers += [nn.Linear(hid, k)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def embed(t):
        f = torch.exp(torch.linspace(0.0, float(np.log(200.0)), 16, device=t.device))
        a = t[:, None].float() / DIFF_STEPS * f[None, :] * np.pi
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.embed(t)], 1))


def latent_codes(k):
    """The training set in this autoencoder's latent, standardised.

    Standardising is not cosmetic. The forward process pushes towards a standard
    normal, so a latent whose own scale is far from one spends most of the
    schedule undoing an offset instead of learning the shape, and the same
    network then looks worse at k=16 than at k=2 for a reason that has nothing
    to do with the dimension. The scale is stored and undone before decoding.
    """
    with torch.no_grad():
        Z = aes[k].enc(torch.tensor(X_DIG[TR], dtype=torch.float32)).numpy()
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    return (Z - mu) / sd, mu, sd


def train_latent(k, hid, depth, epochs, seed=SEED_MAIN):
    Zs, mu, sd = latent_codes(k)

    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        Zt = torch.tensor(Zs, dtype=torch.float32)
        abt = torch.tensor(DABAR, dtype=torch.float32)
        net = LatentScore(k, hid, depth)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(len(Zt), generator=g)
            for s in range(0, len(Zt), 128):
                zb = Zt[perm[s:s + 128]]
                t = torch.randint(0, DIFF_STEPS, (len(zb),), generator=g)
                eps = torch.randn(zb.shape, generator=g)
                a = abt[t][:, None]
                zt = a.sqrt() * zb + (1 - a).sqrt() * eps
                opt.zero_grad()
                loss = ((net(zt, t) - eps) ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.detach())
            sch.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()], "loss": last}

    out = cached(f"ld-diff-k{k}-h{hid}-d{depth}-e{epochs}-s{seed}-v1", build)
    net = LatentScore(k, hid, depth)
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out, mu, sd


def sample_latent(k, net, mu, sd, n, steps_used, seed):
    """Sample in the latent, then decode. The clamp and its consequence are the
    previous article's: after clipping the predicted clean sample, the noise has
    to be recomputed from it, or the update stops satisfying the identity it was
    derived from and the chain leaves the data."""
    idx = np.linspace(DIFF_STEPS - 1, 0, steps_used).round().astype(int)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g)
    with torch.no_grad():
        for j, t in enumerate(idx):
            a_t = float(DABAR[t])
            a_prev = float(DABAR[idx[j + 1]]) if j + 1 < len(idx) else 1.0
            eps = net(z, torch.full((n,), int(t), dtype=torch.long))
            z0 = ((z - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)).clamp(-4, 4)
            eps = (z - np.sqrt(a_t) * z0) / np.sqrt(max(1 - a_t, 1e-12))
            z = np.sqrt(a_prev) * z0 + np.sqrt(max(1 - a_prev, 0.0)) * eps
        Z = z.numpy() * sd + mu
        pix = aes[k].dec(torch.tensor(Z, dtype=torch.float32)).numpy()
    return pix.clip(0, 1), Z


def score_pool(pix, seed=4):
    return (sci(rbf_mmd2(pix[:250], REALPIX[:500], GAMMA)),
            rnd(c2st(pix[:250], REALPIX[:250], seed=seed), 4))


# ============================================== C. the ladder, against the last one
banner("C. the same ladder as the pixel article, in a latent instead")
PIX = json.loads((REPO / "ddpm" / "data" / "ddpm.json").read_text("utf-8"))["pixels"]
print("  the pixel article's ladder, for comparison:")
for r in PIX["ladder"]:
    print(f"    {r['params']:>8,} weights: two sample {r['mmd2']:+.4f}, separable at {r['c2st']}")

ladder = []
for hid, depth, ep in DIFF_WIDTHS:
    net, meta, mu, sd = train_latent(K_SHOW, hid, depth, ep)
    pix, _ = sample_latent(K_SHOW, net, mu, sd, 250, 100, 7)
    m, c = score_pool(pix)
    ndiff = sum(p.numel() for p in net.parameters())
    ndec = ceiling_rows[K_SHOW]["decoder_params"]
    ladder.append({"hidden": hid, "depth": depth, "epochs": ep,
                   "diffusion_params": ndiff, "decoder_params": ndec,
                   "params": ndiff + ndec, "kb_half": rnd((ndiff + ndec) * 2 / 1024, 1),
                   "mmd2": m, "c2st": c, "ink": sci(float(pix.mean())),
                   "resolves": bool(abs(m) > FLOOR["mmd2_max"])})
    print(f"  k={K_SHOW} diffusion {ndiff:,} + decoder {ndec:,} = {ndiff + ndec:,} weights: "
          f"two sample {m:+.4e}, separable at {c:.4f}")
best_lat = min(ladder, key=lambda r: abs(r["mmd2"]))
best_pix = min(PIX["ladder"], key=lambda r: abs(r["mmd2"]))
print(f"\n  best in a latent: {best_lat['mmd2']} with {best_lat['params']:,} weights")
print(f"  best in pixels:   {best_pix['mmd2']} with {best_pix['params']:,} weights")


# ==================================== D. the trade nobody draws: ceiling against cost
banner("D. the bottleneck size, where the two costs pull in opposite directions")
trade = []
HIDD, DEPTHD, EPD = DIFF_WIDTHS[1]
for k in KS:
    net, meta, mu, sd = train_latent(k, HIDD, DEPTHD, EPD)
    pix, _ = sample_latent(k, net, mu, sd, 250, 100, 7)
    m, c = score_pool(pix)
    ndiff = sum(p.numel() for p in net.parameters())
    ceil = ceiling_rows[k]
    trade.append({"k": k, "ceiling_mmd2": ceil["mmd2"], "ceiling_mse": ceil["mse"],
                  "mmd2": m, "c2st": c,
                  "diffusion_params": ndiff, "decoder_params": ceil["decoder_params"],
                  "params": ndiff + ceil["decoder_params"],
                  "gap": sci(abs(m) - abs(float(ceil["mmd2"]))),
                  "at_ceiling": bool(abs(m) <= 2.0 * abs(float(ceil["mmd2"]))),
                  "resolves": bool(abs(m) > FLOOR["mmd2_max"])})
    print(f"  k={k:>2}: ceiling {ceil['mmd2']:+.4e}, reached {m:+.4e}, "
          f"{ndiff + ceil['decoder_params']:,} weights"
          f"{'  <- at its ceiling' if trade[-1]['at_ceiling'] else '  <- short of its ceiling'}")

best_trade = min(trade, key=lambda r: abs(r["mmd2"]))
at_ceiling = [r["k"] for r in trade if r["at_ceiling"]]
print(f"\n  the best of these is k={best_trade['k']} at {best_trade['mmd2']}")
print(f"  the ones sitting at their own ceiling: {at_ceiling if at_ceiling else 'none'}")
print("  A latent that is too small caps the result before diffusion starts; one that is too "
      "large hands the diffusion a harder problem for a ceiling it cannot reach anyway.")

# the judge, and the confound it travels with
pools = []
for r in trade:
    net, _, mu, sd = train_latent(r["k"], HIDD, DEPTHD, EPD)
    pix, _ = sample_latent(r["k"], net, mu, sd, 250, 100, 7)
    pools.append((f"k={r['k']}", pix))
INK = judge_ink_check(pools, REALPIX, J)
print(f"\n  the ink guard on the shared judge: confidence correlates {INK['corr']} with ink; "
      f"{'CONFOUNDED, so no ranking by the judge goes on the page' if INK['confounded'] else 'not confounded'}")


# ============================================== E. what the browser runs
banner("E. exporting the latent process and its decoder")
NET_SHOW, _, MU_SHOW, SD_SHOW = train_latent(K_SHOW, DIFF_WIDTHS[0][0], DIFF_WIDTHS[0][1],
                                             DIFF_WIDTHS[0][2])
blocks = []
lin = [m for m in NET_SHOW.net if isinstance(m, nn.Linear)]
for i, m in enumerate(lin):
    blocks.append((f"d{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
dec = [m for m in aes[K_SHOW].dec if isinstance(m, nn.Linear)]
for i, m in enumerate(dec):
    blocks.append((f"dec{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
enc = [m for m in aes[K_SHOW].enc if isinstance(m, nn.Linear)]
for i, m in enumerate(enc):
    blocks.append((f"enc{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
spec, b64, v16 = export_half(blocks)
print(f"  {len(v16):,} numbers, float16 base64 is {len(b64) / 1024:.0f} kB")

rs = np.random.RandomState(808)
probe_z = rs.randn(24, K_SHOW).astype(np.float64)
with torch.no_grad():
    probe_pix = aes[K_SHOW].dec(torch.tensor(probe_z * SD_SHOW + MU_SHOW,
                                             dtype=torch.float32)).numpy()
    probe_eps = NET_SHOW(torch.tensor(probe_z, dtype=torch.float32),
                         torch.full((24,), 40, dtype=torch.long)).numpy()
probe = {"z": mat(probe_z, 4), "t": 40, "eps": mat(probe_eps, 4),
         "pixels": mat(probe_pix[:8], 4)}
print(f"  a {len(probe_z)} point probe for the page to check its decoder and its score network")

show_pix, show_z = sample_latent(K_SHOW, NET_SHOW, MU_SHOW, SD_SHOW, 24, 100, 21)
strip_b64 = base64.b64encode(
    np.asarray(show_pix.ravel(), dtype=np.float32).astype(np.float16).tobytes()).decode("ascii")
real_b64 = base64.b64encode(
    np.asarray(REALPIX[:24].ravel(), dtype=np.float32).astype(np.float16).tobytes()).decode("ascii")

payload = {
    "meta": {"seed": SEED_MAIN, "threads": THREADS, "torch": torch.__version__,
             "ks": KS, "k_show": K_SHOW, "hidden": HID, "ae_epochs": AE_EPOCHS,
             "diff_steps": DIFF_STEPS, "side": 14, "pixels": 196, "strip": 24,
             "n_train": int(len(TR)), "n_test": int(len(TE)),
             "judge_accuracy": rnd(J["accuracy"], 4),
             "judge_confidence": rnd(J["real_confidence"], 4),
             "real_ink": rnd(float(REALPIX.mean()), 4),
             "note": "every score here is the same two sample statistic on the same held out "
                     "digits as the pixel article, against the same floor, so the two ladders "
                     "are directly comparable"},
    "shared": SHARED,
    "floor": FLOOR, "gamma": sci(GAMMA),
    "ceiling": [ceiling_rows[k] for k in KS],
    "ladder": ladder, "pixel_ladder": PIX["ladder"], "pixel_floor": PIX["floor"],
    "trade": trade, "best_k": best_trade["k"], "at_ceiling": at_ceiling,
    "ink": INK,
    "net": {"format": "float16 little-endian, base64; the score network, then the decoder, then "
                      "the encoder, each row major over output units",
            "b64": b64, "count": len(v16), "spec": spec,
            "k": K_SHOW, "hidden": DIFF_WIDTHS[0][0], "depth": DIFF_WIDTHS[0][1],
            "steps": DIFF_STEPS, "abar": arr(DABAR, 8),
            "mu": arr(MU_SHOW, 5), "sd": arr(SD_SHOW, 5), "probe": probe},
    "strip": {"samples_b64": strip_b64, "real_b64": real_b64, "n": 24},
}
write_json(OUT / "latent.json", payload)
