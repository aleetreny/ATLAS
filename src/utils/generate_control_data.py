"""Data for the ATLAS conditioning article: telling the model what to make.

Every model in this module so far has been asked for something plausible and
nothing else. Nobody has told one what to produce. Conditioning adds a second
input the sampler has to obey, and the standard way to make it obey harder is
classifier free guidance: train one network that has seen both the label and no
label at all, then at sampling time push it away from what it would have done
unconditioned, by a factor everybody calls the guidance scale.

That factor is a dial, and the article is what happens when you turn it,
because the two things you want move in opposite directions and both are
measurable here:

  does it make the digit you asked for   -> the shared judge's LABEL, against
                                            the label that was requested
  does it look like a real digit         -> the same two sample statistic
                                            every article in this module uses,
                                            against the same floor

A. The pipeline is the previous article's, proved: the same autoencoder, and
   its ceiling reproduced.
B. Guidance, swept. Obedience against realism, and the crossing.
C. What obedience costs in variety, because a model that always draws the same
   perfect four obeys perfectly and is useless. Measured as the spread of what
   comes out at each setting.
D. The condition the model was never given: labels held out of training
   entirely, to separate "it learned this label" from "it learned to obey".
E. What the browser runs.

Run from anywhere: `python src/utils/generate_control_data.py`.
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

OUT = REPO / "controlnet" / "data"
THREADS = threads()

K = 8                          # the bottleneck the previous article settled on
HID = 64
AE_EPOCHS = 600
DIFF_STEPS = 200
DHID, DDEPTH, DEPOCHS = 128, 3, 4000
DROP = 0.15                    # how often the label is hidden during training
SCALES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
HELD_OUT_LABELS = [3, 8]
SEED_MAIN = SEED

print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED_MAIN}")
X_DIG, Y_DIG, TR, TE = digits14()
J = judge()
print(f"  the module's shared judge: {J['accuracy'] * 100:.2f}% on held out digits")


# ================================================ A. the pipeline, and its proof
class AE(nn.Module):
    def __init__(self, k=K, hid=HID, dim=196):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Linear(hid, k))
        self.dec = nn.Sequential(nn.Linear(k, hid), nn.Tanh(),
                                 nn.Linear(hid, dim), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(seed=SEED_MAIN, epochs=AE_EPOCHS):
    """The same cache key the previous article used, so this is literally the
    same trained autoencoder rather than an equivalent one."""
    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        Xt = torch.tensor(X_DIG[TR], dtype=torch.float32)
        net = AE()
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

    out = cached(f"ld-ae-k{K}-h{HID}-e{epochs}-s{seed}-v1", build)
    net = AE()
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net


banner("A. the pipeline is the previous article's, and here is the proof")
AENET = train_ae()
REALPIX = X_DIG[TE]
GAMMA = 1.0 / (2.0 * float(np.median(
    np.linalg.norm(REALPIX[:300, None, :] - REALPIX[None, :300, :], axis=2)) ** 2))
FLOOR = real_vs_real_floor(REALPIX, GAMMA, splits=5, n=250, seed=17)
with torch.no_grad():
    trip = AENET(torch.tensor(REALPIX, dtype=torch.float32)).numpy()
ceil_mmd = rbf_mmd2(trip[:250], REALPIX[:500], GAMMA)
LD = json.loads((REPO / "latent-diffusion" / "data" / "latent.json").read_text("utf-8"))
published_ceiling = float(next(r for r in LD["ceiling"] if r["k"] == K)["mmd2"])
print(f"  the round trip through this autoencoder scores {ceil_mmd:.6e}; the previous article "
      f"published {published_ceiling:.6e} for the same bottleneck")
assert abs(ceil_mmd - published_ceiling) < 0.1 * abs(published_ceiling), (
    f"this is not the autoencoder the latent diffusion article measured "
    f"({ceil_mmd:.3e} against its published {published_ceiling:.3e})")
CEILING = {"mmd2": sci(ceil_mmd), "published": sci(published_ceiling),
           "floor": FLOOR["mmd2_max"], "k": K}
print(f"  the floor, real against real: {FLOOR['mmd2_max']}")


# ============================================ the conditional model
def schedule_cos(steps):
    s = 0.008
    t = np.linspace(0, 1, steps + 1)
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return np.cumprod(1 - np.clip(1 - ab[1:] / ab[:-1], 0, 0.999))


DABAR = schedule_cos(DIFF_STEPS)
with torch.no_grad():
    ZTR = AENET.enc(torch.tensor(X_DIG[TR], dtype=torch.float32)).numpy()
ZMU, ZSD = ZTR.mean(0), ZTR.std(0) + 1e-8
ZS = (ZTR - ZMU) / ZSD


class CondScore(nn.Module):
    """One network, two jobs. The label arrives as an embedding, and index ten
    is the "no label" token: training hides the real label some of the time so
    the SAME weights can answer both questions, which is what makes classifier
    free guidance possible without a second model."""

    def __init__(self, k=K, hid=DHID, depth=DDEPTH, classes=10):
        super().__init__()
        self.emb = nn.Embedding(classes + 1, 32)
        layers = [nn.Linear(k + 32 + 32, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        layers += [nn.Linear(hid, k)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def tembed(t):
        f = torch.exp(torch.linspace(0.0, float(np.log(200.0)), 16, device=t.device))
        a = t[:, None].float() / DIFF_STEPS * f[None, :] * np.pi
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, z, t, y):
        return self.net(torch.cat([z, self.tembed(t), self.emb(y)], 1))


def train_cond(seed=SEED_MAIN, epochs=DEPOCHS, drop=DROP, exclude=()):
    tag = "-".join(str(v) for v in exclude) if exclude else "none"

    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        keep = np.array([i for i in range(len(TR)) if Y_DIG[TR][i] not in exclude])
        Zt = torch.tensor(ZS[keep], dtype=torch.float32)
        Yt = torch.tensor(Y_DIG[TR][keep], dtype=torch.long)
        abt = torch.tensor(DABAR, dtype=torch.float32)
        net = CondScore()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(len(Zt), generator=g)
            for s in range(0, len(Zt), 128):
                idx = perm[s:s + 128]
                zb, yb = Zt[idx], Yt[idx].clone()
                # hide the label some of the time, which is the whole trick
                hide = torch.rand(len(yb), generator=g) < drop
                yb[hide] = 10
                t = torch.randint(0, DIFF_STEPS, (len(zb),), generator=g)
                eps = torch.randn(zb.shape, generator=g)
                a = abt[t][:, None]
                zt = a.sqrt() * zb + (1 - a).sqrt() * eps
                opt.zero_grad()
                loss = ((net(zt, t, yb) - eps) ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.detach())
            sch.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()], "loss": last,
                "n": int(len(keep))}

    out = cached(f"cn-k{K}-h{DHID}-d{DDEPTH}-e{epochs}-drop{drop}-x{tag}-s{seed}-v1", build)
    net = CondScore()
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out


def sample_cond(net, labels, scale, steps_used=100, seed=7):
    """Classifier free guidance: two forward passes, the conditioned one pushed
    away from the unconditioned one by `scale`. At zero this is the plain
    unconditional model and the label does nothing."""
    n = len(labels)
    idx = np.linspace(DIFF_STEPS - 1, 0, steps_used).round().astype(int)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, K, generator=g)
    y = torch.tensor(labels, dtype=torch.long)
    ynull = torch.full((n,), 10, dtype=torch.long)
    with torch.no_grad():
        for j, t in enumerate(idx):
            a_t = float(DABAR[t])
            a_prev = float(DABAR[idx[j + 1]]) if j + 1 < len(idx) else 1.0
            tt = torch.full((n,), int(t), dtype=torch.long)
            e_un = net(z, tt, ynull)
            eps = e_un if scale == 0.0 else e_un + scale * (net(z, tt, y) - e_un)
            z0 = ((z - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)).clamp(-4, 4)
            eps = (z - np.sqrt(a_t) * z0) / np.sqrt(max(1 - a_t, 1e-12))
            z = np.sqrt(a_prev) * z0 + np.sqrt(max(1 - a_prev, 0.0)) * eps
        Z = z.numpy() * ZSD + ZMU
        pix = AENET.dec(torch.tensor(Z, dtype=torch.float32)).numpy()
    return pix.clip(0, 1), Z


# ==================================================== B. the dial, and what it costs
banner("B. obedience against realism, as the guidance scale is turned")
CNET, CMETA = train_cond()
print(f"  trained on {CMETA['n']} digits with the label hidden {DROP:.0%} of the time, "
      f"final batch loss {CMETA['loss']:.5f}")

rs = np.random.RandomState(5)
ASK = rs.randint(0, 10, 500)
sweep = []
pools = []
for w in SCALES:
    pix, Z = sample_cond(CNET, ASK, w, seed=31)
    pred = J["proba"](pix).argmax(1)
    obey = float((pred == ASK).mean())
    m = rbf_mmd2(pix[:250], REALPIX[:500], GAMMA)
    c = c2st(pix[:250], REALPIX[:250], seed=5)
    # variety: the mean pairwise distance of what came out, per requested class,
    # because a model that draws one perfect four obeys perfectly and is useless
    spreads = []
    for lab in range(10):
        sel = pix[ASK == lab]
        if len(sel) > 3:
            d = np.linalg.norm(sel[:, None, :] - sel[None, :, :], axis=2)
            spreads.append(float(d[np.triu_indices(len(sel), 1)].mean()))
    sweep.append({"scale": w, "obeys": rnd(obey, 4), "mmd2": sci(m), "c2st": rnd(c, 4),
                  "variety": rnd(float(np.mean(spreads)), 4),
                  "resolves": bool(abs(m) > FLOOR["mmd2_max"])})
    pools.append((f"w={w}", pix))
    print(f"  scale {w:>4}: makes the digit asked for {obey * 100:5.1f}% of the time, "
          f"two sample {m:+.4e}, variety {np.mean(spreads):.4f}")

# the same two measures on real data, which is the only scale either of them has
real_pred = J["proba"](REALPIX).argmax(1)
real_obey = float((real_pred == Y_DIG[TE]).mean())
rspread = []
for lab in range(10):
    sel = REALPIX[Y_DIG[TE] == lab]
    if len(sel) > 3:
        d = np.linalg.norm(sel[:60, None, :] - sel[None, :60, :], axis=2)
        rspread.append(float(d[np.triu_indices(min(60, len(sel)), 1)].mean()))
REAL = {"obeys": rnd(real_obey, 4), "variety": rnd(float(np.mean(rspread)), 4)}
print(f"  for comparison, on REAL held out digits the judge names the right one "
      f"{real_obey * 100:.1f}% of the time and their variety is {np.mean(rspread):.4f}")

best_obey = max(sweep, key=lambda r: r["obeys"])
best_real = min(sweep, key=lambda r: abs(r["mmd2"]))
print(f"\n  most obedient: scale {best_obey['scale']} at {best_obey['obeys']:.1%}")
print(f"  most realistic: scale {best_real['scale']} at {best_real['mmd2']}")
if best_obey["scale"] != best_real["scale"]:
    print("  and they are NOT the same setting, which is the article")
else:
    print("  and they are the same setting, so on this model there is no trade to draw")


# ================================ C. a label the model was never trained on
banner("C. asking for something it never saw, which separates obeying from knowing")
HNET, HMETA = train_cond(exclude=tuple(HELD_OUT_LABELS))
print(f"  a second model trained on {HMETA['n']} digits, with every "
      f"{' and '.join(str(v) for v in HELD_OUT_LABELS)} removed from training entirely")
held = []
for w in (1.0, 2.0, 5.0):
    for lab in HELD_OUT_LABELS + [l for l in range(10) if l not in HELD_OUT_LABELS][:2]:
        ask = np.full(150, lab)
        pix, _ = sample_cond(HNET, ask, w, seed=77)
        pred = J["proba"](pix).argmax(1)
        obey = float((pred == lab).mean())
        held.append({"scale": w, "label": lab, "seen": lab not in HELD_OUT_LABELS,
                     "obeys": rnd(obey, 4)})
    row = [r for r in held if r["scale"] == w]
    seen = np.mean([r["obeys"] for r in row if r["seen"]])
    unseen = np.mean([r["obeys"] for r in row if not r["seen"]])
    print(f"  scale {w:>4}: labels it was trained on {seen:.1%}, labels removed from training "
          f"{unseen:.1%}")
seen_all = float(np.mean([r["obeys"] for r in held if r["seen"]]))
unseen_all = float(np.mean([r["obeys"] for r in held if not r["seen"]]))
HELD = {"rows": held, "labels": HELD_OUT_LABELS, "seen": rnd(seen_all, 4),
        "unseen": rnd(unseen_all, 4), "n_train": HMETA["n"]}
print(f"\n  averaged: {seen_all:.1%} on labels it saw against {unseen_all:.1%} on labels it did "
      f"not, so the dial makes it obey a token it was given, not conjure a digit it never met")

# the judge is being used two ways here and only one of them is the shaky one
INK = judge_ink_check(pools, REALPIX, J)
print(f"\n  the ink guard: the judge's CONFIDENCE correlates {INK['corr']} with ink across these "
      f"pools, which is why this article scores obedience with its LABEL instead")


# ============================================================ D. what the browser runs
banner("D. exporting the conditional model, its decoder and its encoder")
blocks = []
lin = [m for m in CNET.net if isinstance(m, nn.Linear)]
for i, m in enumerate(lin):
    blocks.append((f"c{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
blocks.append(("emb", CNET.emb.weight.detach().numpy(), None))
for i, m in enumerate([m for m in AENET.dec if isinstance(m, nn.Linear)]):
    blocks.append((f"dec{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
spec, b64, v16 = export_half(blocks)
print(f"  {len(v16):,} numbers, float16 base64 is {len(b64) / 1024:.0f} kB")

rs2 = np.random.RandomState(909)
pz = rs2.randn(20, K).astype(np.float64)
plab = rs2.randint(0, 10, 20)
with torch.no_grad():
    peps = CNET(torch.tensor(pz, dtype=torch.float32),
                torch.full((20,), 40, dtype=torch.long),
                torch.tensor(plab, dtype=torch.long)).numpy()
    pdec = AENET.dec(torch.tensor(pz * ZSD + ZMU, dtype=torch.float32)).numpy()
probe = {"z": mat(pz, 4), "labels": [int(v) for v in plab], "t": 40,
         "eps": mat(peps, 4), "pixels": mat(pdec[:6], 4)}
print(f"  a {len(pz)} point probe for the page to check its score network and its decoder")

SHOW_SCALES = [0.0, 1.0, 2.0, 5.0]
strips = {}
for w in SHOW_SCALES:
    ask = np.repeat(np.arange(10), 2)
    pix, _ = sample_cond(CNET, ask, w, seed=1234)
    strips[str(w)] = base64.b64encode(
        np.asarray(pix.ravel(), dtype=np.float32).astype(np.float16).tobytes()).decode("ascii")
real_by_label = np.concatenate([REALPIX[Y_DIG[TE] == lab][:2] for lab in range(10)])
real_b64 = base64.b64encode(
    np.asarray(real_by_label.ravel(), dtype=np.float32).astype(np.float16).tobytes()).decode("ascii")

payload = {
    "meta": {"seed": SEED_MAIN, "threads": THREADS, "torch": torch.__version__,
             "k": K, "hidden": DHID, "depth": DDEPTH, "epochs": DEPOCHS, "drop": DROP,
             "diff_steps": DIFF_STEPS, "side": 14, "pixels": 196,
             "scales": SCALES, "show_scales": SHOW_SCALES, "asked": 500,
             "n_train": int(len(TR)), "n_test": int(len(TE)),
             "judge_accuracy": rnd(J["accuracy"], 4),
             "real_ink": rnd(float(REALPIX.mean()), 4),
             "note": "obedience is the judge's LABEL against the label that was requested, not "
                     "its confidence: the confidence is the measure this module has already "
                     "shown to be a measure of ink"},
    "ceiling": CEILING, "floor": FLOOR, "gamma": sci(GAMMA),
    "sweep": sweep, "real": REAL,
    "best_obey": best_obey["scale"], "best_real": best_real["scale"],
    "held": HELD, "ink": INK,
    "net": {"format": "float16 little-endian, base64; the conditional score network, its label "
                      "embedding, then the decoder, each row major over output units",
            "b64": b64, "count": len(v16), "spec": spec,
            "k": K, "steps": DIFF_STEPS, "abar": arr(DABAR, 8),
            "mu": arr(ZMU, 5), "sd": arr(ZSD, 5), "null_token": 10, "probe": probe},
    "strips": {"by_scale": strips, "real_b64": real_b64, "per_label": 2},
}
write_json(OUT / "control.json", payload)
