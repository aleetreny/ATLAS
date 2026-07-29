"""Data for the ATLAS few step sampling article: collapsing the chain.

Everything in this module has paid the same tax. Sampling is a chain, each link
is a pass through a network, and the conditioning article doubled the cost of
every link. The obvious question is whether the chain can be shortened, and the
obvious answer, "just take bigger steps", does not work: the trajectory a
diffusion sampler follows is CURVED, and a big step along a curve misses.

So the field went at the curvature itself, and this file measures what that
buys, in the same eight number latent the last two articles used:

A. The setup is the previous articles', proved by reproducing their round trip.
B. Rectified flow. Train on straight lines between noise and data instead of on
   a noise schedule, and the paths come out straighter. Straightness is not a
   metaphor here: it is the average distance between the path a sampler
   actually walks and the straight line joining its two ends, and it is a
   number.
C. Reflow. Retrain on the pairs the model itself produced, which is supposed to
   straighten the paths further. Measured before and against after.
D. Consistency distillation. A student trained to jump from anywhere on a
   trajectory to its endpoint, so that one step is the whole sampler.
E. Steps against quality for all four, on the same axes and the same floor, and
   the honest price of a single step.
F. What the browser runs: the trajectories, drawn, and a sampler with the step
   budget as a dial.

Run from anywhere: `python src/utils/generate_consistency_data.py`.
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

OUT = REPO / "consistency" / "data"
THREADS = threads()

K = 8
HID = 64
AE_EPOCHS = 600
STEPS = 200                    # the diffusion schedule, kept from article 59
FHID, FDEPTH, FEPOCHS = 128, 3, 4000
BUDGETS = [1, 2, 4, 8, 16, 32, 100]
SEED_MAIN = SEED

print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED_MAIN}")
X_DIG, Y_DIG, TR, TE = digits14()
J = judge()


class AE(nn.Module):
    def __init__(self, k=K, hid=HID, dim=196):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Linear(hid, k))
        self.dec = nn.Sequential(nn.Linear(k, hid), nn.Tanh(),
                                 nn.Linear(hid, dim), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(seed=SEED_MAIN, epochs=AE_EPOCHS):
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


banner("A. the same latent as the last two articles, and the proof")
AENET = train_ae()
REALPIX = X_DIG[TE]
GAMMA = 1.0 / (2.0 * float(np.median(
    np.linalg.norm(REALPIX[:300, None, :] - REALPIX[None, :300, :], axis=2)) ** 2))
FLOOR = real_vs_real_floor(REALPIX, GAMMA, splits=5, n=250, seed=17)
with torch.no_grad():
    trip = AENET(torch.tensor(REALPIX, dtype=torch.float32)).numpy()
CEIL = rbf_mmd2(trip[:250], REALPIX[:500], GAMMA)
LD = json.loads((REPO / "latent-diffusion" / "data" / "latent.json").read_text("utf-8"))
pub = float(next(r for r in LD["ceiling"] if r["k"] == K)["mmd2"])
print(f"  the round trip scores {CEIL:.6e}; the latent article published {pub:.6e}")
assert abs(CEIL - pub) < 0.1 * abs(pub), "this is not the autoencoder those articles measured"
print(f"  the floor, real against real: {FLOOR['mmd2_max']}")

with torch.no_grad():
    ZTR = AENET.enc(torch.tensor(X_DIG[TR], dtype=torch.float32)).numpy()
ZMU, ZSD = ZTR.mean(0), ZTR.std(0) + 1e-8
ZS = (ZTR - ZMU) / ZSD


def epochs_for(n, batch, target):
    """How many epochs over `n` items in batches of `batch` come to `target`
    gradient steps.

    This exists because the arms of this article do not train on the same
    amount of data. The base flow sees the real latent codes, of which there
    are a few thousand; reflow and distillation see twenty thousand pairs the
    model made up, because the model can make up as many as anyone wants. Give
    them all the same number of EPOCHS and the two later arms quietly get an
    order of magnitude more gradient steps, and then any improvement is bought
    with compute rather than with the method, which is exactly the claim under
    test. So the epoch counts are derived, not chosen.
    """
    per = int(np.ceil(n / batch))
    return max(1, int(round(target / per)))


FBATCH, CBATCH = 128, 512
BASE_STEPS = int(np.ceil(len(ZS) / FBATCH)) * FEPOCHS
print(f"  the compute budget every arm gets: {BASE_STEPS:,} gradient steps "
      f"({len(ZS):,} codes, batches of {FBATCH}, {FEPOCHS} epochs)")


# ============================================================== the two networks
class Field(nn.Module):
    """One shape for both jobs. A diffusion model predicts the noise at a level
    t; a rectified flow predicts the velocity at a time t. The inputs and the
    output have the same shape, and keeping the architecture identical is what
    makes the comparison a comparison."""

    def __init__(self, k=K, hid=FHID, depth=FDEPTH):
        super().__init__()
        layers = [nn.Linear(k + 32, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        layers += [nn.Linear(hid, k)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def embed(t):
        """t arrives in [0, 1] for both kinds of model, so the embedding is the
        same function and no rescaling hides between the two arms."""
        f = torch.exp(torch.linspace(0.0, float(np.log(200.0)), 16, device=t.device))
        a = t[:, None].float() * f[None, :] * np.pi
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, z, t):
        return self.net(torch.cat([z, self.embed(t)], 1))


def train_flow(pairs=None, tag="base", seed=SEED_MAIN, epochs=FEPOCHS):
    """Rectified flow: sample a pair (noise, data), walk the straight line
    between them, and learn the velocity, which for a straight line is just the
    difference. With `pairs` supplied the noise is not independent of the data:
    it is the noise that actually produced it, which is what reflow means."""
    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        net = Field()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Z1 = torch.tensor(ZS if pairs is None else pairs[1], dtype=torch.float32)
        Z0 = None if pairs is None else torch.tensor(pairs[0], dtype=torch.float32)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(len(Z1), generator=g)
            for s in range(0, len(Z1), FBATCH):
                idx = perm[s:s + FBATCH]
                z1 = Z1[idx]
                z0 = torch.randn(z1.shape, generator=g) if Z0 is None else Z0[idx]
                t = torch.rand(len(z1), generator=g)
                zt = (1 - t)[:, None] * z0 + t[:, None] * z1
                opt.zero_grad()
                loss = ((net(zt, t) - (z1 - z0)) ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.detach())
            sch.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()], "loss": last}

    out = cached(f"cm-flow-{tag}-k{K}-h{FHID}-d{FDEPTH}-e{epochs}-s{seed}-v1", build)
    net = Field()
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out


def walk_flow(net, z0, steps_used, keep_path=False):
    """Euler from t=0 to t=1. With one step this is z0 + v(z0, 0), which is the
    straight line guess, and the whole question is how wrong that is."""
    z = torch.tensor(z0, dtype=torch.float32)
    path = [z.numpy().copy()]
    dt = 1.0 / steps_used
    with torch.no_grad():
        for i in range(steps_used):
            t = torch.full((len(z),), i * dt)
            z = z + dt * net(z, t)
            if keep_path:
                path.append(z.numpy().copy())
    return z.numpy(), (np.stack(path) if keep_path else None)


def straightness(net, n=400, steps_used=100, seed=3):
    """How far the walked path strays from the straight line joining its ends.

    Zero is a straight line. The number is divided by the length of that line,
    so it is a shape rather than a scale, and two models whose samples land in
    different places can still be compared.
    """
    rs = np.random.RandomState(seed)
    z0 = rs.randn(n, K)
    z1, path = walk_flow(net, z0, steps_used, keep_path=True)
    ts = np.linspace(0, 1, len(path))[:, None, None]
    chord = path[0][None, :, :] + ts * (path[-1] - path[0])[None, :, :]
    dev = np.linalg.norm(path - chord, axis=2).mean(0)
    length = np.linalg.norm(path[-1] - path[0], axis=1)
    return float(np.mean(dev / np.maximum(length, 1e-9))), z0, z1


def decode(Z):
    with torch.no_grad():
        return AENET.dec(torch.tensor(Z * ZSD + ZMU, dtype=torch.float32)).numpy().clip(0, 1)


def score(pix, seed=4):
    return (sci(rbf_mmd2(pix[:250], REALPIX[:500], GAMMA)),
            rnd(c2st(pix[:250], REALPIX[:250], seed=seed), 4))


# ============================================== B and C. straightness, and reflow
banner("B. rectified flow, and how straight its paths actually are")
FLOW, FMETA = train_flow()
s_flow, z0_probe, z1_probe = straightness(FLOW)
print(f"  trained, final batch loss {FMETA['loss']:.5f}")
print(f"  its paths stray {s_flow:.5f} of their own length from the straight line: "
      f"{'nearly straight already' if s_flow < 0.05 else 'still visibly curved'}")

banner("C. reflow: retrain on the pairs the model itself produced")
rs = np.random.RandomState(11)
BIG = 20000
pair0 = rs.randn(BIG, K)
pair1, _ = walk_flow(FLOW, pair0, 100)
REP = epochs_for(BIG, FBATCH, BASE_STEPS)
REFLOW, RMETA = train_flow(pairs=(pair0, pair1), tag="reflow", epochs=REP)
s_reflow, _, _ = straightness(REFLOW)
print(f"  retrained on {BIG:,} of its own (noise, sample) pairs for {REP} epochs, "
      f"which is the same {BASE_STEPS:,} gradient steps the base flow got")
print(f"  loss {RMETA['loss']:.5f}")
print(f"  its paths stray {s_reflow:.5f}, against {s_flow:.5f} before")
straighter = s_reflow < s_flow
print(f"  so reflow {'DID' if straighter else 'did NOT'} straighten them here, by a factor of "
      f"{s_flow / max(s_reflow, 1e-12):.2f}")


# ================================================ D. one step, by distillation
banner("D. a student trained to jump straight to the end")


def train_consistency(teacher, tag="cd", seed=SEED_MAIN, epochs=FEPOCHS):
    """The student sees a point on a teacher trajectory and has to name the
    endpoint, from anywhere on it. If it can, then one evaluation is the whole
    sampler, because the endpoint is what a sampler is for."""
    def build():
        threads()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        rs2 = np.random.RandomState(seed)
        z0 = rs2.randn(BIG, K)
        z1, path = walk_flow(teacher, z0, 32, keep_path=True)
        P = torch.tensor(path, dtype=torch.float32)      # (33, BIG, K)
        T = torch.tensor(np.linspace(0, 1, P.shape[0]), dtype=torch.float32)
        E = torch.tensor(z1, dtype=torch.float32)
        net = Field()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(BIG, generator=g)
            for s in range(0, BIG, CBATCH):
                idx = perm[s:s + CBATCH]
                j = torch.randint(0, P.shape[0], (len(idx),), generator=g)
                zt = P[j, idx]
                t = T[j]
                opt.zero_grad()
                # the student names the endpoint directly, not a velocity
                loss = ((net(zt, t) - E[idx]) ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.detach())
            sch.step()
        return {"W": [p.detach().numpy().copy() for p in net.parameters()], "loss": last}

    out = cached(f"cm-{tag}-k{K}-h{FHID}-d{FDEPTH}-e{epochs}-s{seed}-v1", build)
    net = Field()
    with torch.no_grad():
        for p, w in zip(net.parameters(), out["W"]):
            p.copy_(torch.tensor(w))
    return net, out


CEP = epochs_for(BIG, CBATCH, BASE_STEPS)
STUDENT, SMETA = train_consistency(REFLOW, epochs=CEP)
print(f"  distilled from the reflowed teacher for {CEP} epochs, again {BASE_STEPS:,} "
      f"gradient steps, final batch loss {SMETA['loss']:.5f}")


def sample_student(net, z0):
    with torch.no_grad():
        z = torch.tensor(z0, dtype=torch.float32)
        return net(z, torch.zeros(len(z))).numpy()


# ================================================== E. steps against quality
banner("E. what each of them costs in steps")
rs3 = np.random.RandomState(909)
Z0 = rs3.randn(250, K)

# Where a coarse walk ends up against where the fine walk ends up. This is the
# curvature translated into the only currency the reader cares about: taking
# fewer steps moves the answer, and this says by how much, before any of it
# reaches a picture.
gaps = []
REF = {name: walk_flow(net, Z0, 100)[0] for name, net in
       (("rectified flow", FLOW), ("after reflow", REFLOW))}
for name, net in (("rectified flow", FLOW), ("after reflow", REFLOW)):
    ref = REF[name]
    scale = float(np.linalg.norm(ref - Z0, axis=1).mean())
    for b in BUDGETS:
        z = walk_flow(net, Z0, b)[0]
        gaps.append({"arm": name, "steps": b,
                     "gap": rnd(float(np.linalg.norm(z - ref, axis=1).mean()) / scale, 5)})
for name in ("rectified flow", "after reflow"):
    g1 = next(r for r in gaps if r["arm"] == name and r["steps"] == 1)
    print(f"  {name:<18} one step lands {g1['gap'] * 100:.1f}% of the trip away from "
          f"where a hundred steps land")

arms = []
for name, net, kind in (("rectified flow", FLOW, "walk"),
                        ("after reflow", REFLOW, "walk"),
                        ("distilled student", STUDENT, "jump")):
    for b in BUDGETS:
        if kind == "jump" and b > 1:
            continue
        Z = sample_student(net, Z0) if kind == "jump" else walk_flow(net, Z0, b)[0]
        m, c = score(decode(Z))
        arms.append({"arm": name, "steps": b, "mmd2": m, "c2st": c,
                     "resolves": bool(abs(m) > FLOOR["mmd2_max"])})
        print(f"  {name:<18} {b:>3} step{'s' if b != 1 else ' '}: two sample {m:+.4e}, "
              f"separable at {c:.4f}")

one = {r["arm"]: r for r in arms if r["steps"] == 1}
best = {}
for name in ("rectified flow", "after reflow"):
    rows = [r for r in arms if r["arm"] == name]
    best[name] = min(rows, key=lambda r: abs(r["mmd2"]))
print(f"\n  at ONE step: " + ", ".join(f"{k} {v['mmd2']}" for k, v in one.items()))
print(f"  at their best: " + ", ".join(f"{k} {v['mmd2']} at {v['steps']}" for k, v in best.items()))
cost = abs(one["distilled student"]["mmd2"]) / abs(best["after reflow"]["mmd2"])
print(f"  so one distilled step costs {cost:.2f} times the reflowed model's best, for "
      f"{best['after reflow']['steps']} times fewer network evaluations")

pools = [(f"{r['arm']} x{r['steps']}", decode(
    sample_student(STUDENT, Z0) if r["arm"] == "distilled student"
    else walk_flow({"rectified flow": FLOW, "after reflow": REFLOW}[r["arm"]], Z0, r["steps"])[0]))
    for r in (one["rectified flow"], one["after reflow"], one["distilled student"],
              best["after reflow"])]
INK = judge_ink_check(pools, REALPIX, J)
print(f"  the ink guard: confidence correlates {INK['corr']} with ink; "
      f"{'CONFOUNDED' if INK['confounded'] else 'not confounded'}")


# ======================================= E2. the only yardstick for these differences
banner("E2. how far the statistic moves when nothing but the noise changes")
# The real against real floor says which pools are distinguishable from real
# data. It does NOT say which of two pools is closer to it, and after reflow the
# three fast arms land within one percent of each other, which is exactly the
# regime where a floor gets misread as a resolution. So: fix the arm, change the
# noise, and measure how much the number moves on its own. Anything smaller than
# that is not a difference, whatever the chart looks like.
REDRAWS = 6
spread = []
for name, net, kind, b in (("rectified flow", FLOW, "walk", 1),
                           ("rectified flow", FLOW, "walk", 100),
                           ("after reflow", REFLOW, "walk", 1),
                           ("after reflow", REFLOW, "walk", 100),
                           ("distilled student", STUDENT, "jump", 1)):
    vals = []
    for s in range(REDRAWS):
        zz = np.random.RandomState(3000 + s).randn(250, K)
        Z = sample_student(net, zz) if kind == "jump" else walk_flow(net, zz, b)[0]
        vals.append(abs(rbf_mmd2(decode(Z), REALPIX[:500], GAMMA)))
    spread.append({"arm": name, "steps": b, "draws": REDRAWS,
                   "mean": sci(float(np.mean(vals))), "sd": sci(float(np.std(vals))),
                   "lo": sci(float(np.min(vals))), "hi": sci(float(np.max(vals))),
                   "range": sci(float(np.max(vals) - np.min(vals)))})
    print(f"  {name:<18} {b:>3}: {np.mean(vals):.4e} give or take {np.std(vals):.1e}, "
          f"across {REDRAWS} redraws spanning {np.max(vals) - np.min(vals):.2e}")
# One band for the whole chart would be wrong, and wrong in the direction that
# flatters this article: the spread of a statistic scales with the statistic,
# and the collapsed arm's value is thirty times the others', so its spread would
# swallow every comparison worth making. Each pair gets the band its own two
# cells set, which is the lesson the flows article paid for.
pairs = []
CELLS = [(r["arm"], r["steps"]) for r in spread]
for i in range(len(CELLS)):
    for j in range(i + 1, len(CELLS)):
        a = next(r for r in spread if (r["arm"], r["steps"]) == CELLS[i])
        c = next(r for r in spread if (r["arm"], r["steps"]) == CELLS[j])
        d = abs(float(a["mean"]) - float(c["mean"]))
        combined = float(np.hypot(float(a["sd"]), float(c["sd"])))
        # the strictest reading anyone can argue with: two arms are ranked only
        # when six redraws of one never reach six redraws of the other
        overlap = not (float(a["lo"]) > float(c["hi"]) or float(c["lo"]) > float(a["hi"]))
        pairs.append({"a": f"{a['arm']} x{a['steps']}", "b": f"{c['arm']} x{c['steps']}",
                      "diff": sci(d), "combined_sd": sci(combined),
                      "sigmas": rnd(d / max(combined, 1e-12), 2),
                      "resolved": bool(not overlap)})
        print(f"  {a['arm']} x{a['steps']} against {c['arm']} x{c['steps']}: "
              f"{d:.2e} apart, {d / max(combined, 1e-12):.2f} combined sd, "
              f"{'RANKED' if not overlap else 'not separable from redraw noise'}")
RESOLVED = [p for p in pairs if p["resolved"]]
print(f"  {len(RESOLVED)} of {len(pairs)} pairs are rankable at all")


# ============================================================ F. what the browser runs
banner("F. exporting the three samplers and the decoder")
blocks = []
for tag, net in (("f", FLOW), ("r", REFLOW), ("s", STUDENT)):
    for i, m in enumerate([m for m in net.net if isinstance(m, nn.Linear)]):
        blocks.append((f"{tag}{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
for i, m in enumerate([m for m in AENET.dec if isinstance(m, nn.Linear)]):
    blocks.append((f"dec{i}", m.weight.detach().numpy().T, m.bias.detach().numpy()))
spec, b64, v16 = export_half(blocks)
print(f"  {len(v16):,} numbers, float16 base64 is {len(b64) / 1024:.0f} kB")

# The eight number latent has to be drawn on a flat page, so it is projected
# onto its own two strongest directions. The matrix is shipped rather than
# recomputed in the browser, so a path drawn from stored data and a path drawn
# from a live sample land in the same frame.
_, _, VT = np.linalg.svd(ZS - ZS.mean(0), full_matrices=False)
PROJ = VT[:2].T                                    # (K, 2)
print(f"  the two strongest directions of the latent carry the projection")

rsP = np.random.RandomState(77)
PATH_N = 10
PZ0 = rsP.randn(PATH_N, K)
paths = {}
for name, tag, net in (("rectified flow", "flow", FLOW), ("after reflow", "reflow", REFLOW)):
    entry = {"ref": None, "coarse": {}}
    _, ref = walk_flow(net, PZ0, 100, keep_path=True)
    entry["ref"] = [mat(ref[:, i, :] @ PROJ, 4) for i in range(PATH_N)]
    for b in BUDGETS:
        _, p = walk_flow(net, PZ0, b, keep_path=True)
        entry["coarse"][str(b)] = [mat(p[:, i, :] @ PROJ, 4) for i in range(PATH_N)]
    paths[tag] = entry
paths["student"] = {"jump": mat(sample_student(STUDENT, PZ0) @ PROJ, 4),
                    "start": mat(PZ0 @ PROJ, 4)}
print(f"  {PATH_N} trajectories per arm, projected, at every budget the page offers")

# A grid of the truth in the same frame, so the paths have somewhere to land:
# where the real codes are, as a two dimensional histogram.
HG = 44
proj_real = ZS @ PROJ
lim = float(np.abs(proj_real).max()) * 1.08
Hh, _, _ = np.histogram2d(proj_real[:, 0], proj_real[:, 1], bins=HG,
                          range=[[-lim, lim], [-lim, lim]])
cloud = {"limit": rnd(lim, 4), "bins": HG,
         "counts": [[int(v) for v in row] for row in Hh.T]}

pz = np.random.RandomState(505).randn(20, K)
probe = {"z": mat(pz, 4), "t": 0.37, "dec_pixels": None}
with torch.no_grad():
    tt = torch.full((len(pz),), 0.37)
    zz = torch.tensor(pz, dtype=torch.float32)
    probe["flow"] = mat(FLOW(zz, tt).numpy(), 4)
    probe["reflow"] = mat(REFLOW(zz, tt).numpy(), 4)
    probe["student"] = mat(STUDENT(zz, torch.zeros(len(pz))).numpy(), 4)
    probe["dec_pixels"] = mat(AENET.dec(
        torch.tensor(pz * ZSD + ZMU, dtype=torch.float32)).numpy()[:6], 4)
# One forward pass being right does not make a WALK right: the step size, the
# time the field is asked at, and the direction of travel are all decisions the
# loop makes and the single pass never touches. So the probe carries endpoints
# too, and the page has to reproduce them by actually walking.
probe["walk_steps"] = 8
with torch.no_grad():
    probe["walk_flow"] = mat(walk_flow(FLOW, pz, 8)[0], 4)
    probe["walk_reflow"] = mat(walk_flow(REFLOW, pz, 8)[0], 4)
probe["proj"] = mat(pz @ PROJ, 4)
print(f"  a {len(pz)} point probe covering all three networks, an eight step walk, "
      f"the projection and the decoder")

SHOW = [("rectified flow", 1), ("rectified flow", 4), ("after reflow", 1),
        ("distilled student", 1)]
strips = {}
rsS = np.random.RandomState(4242)
SZ0 = rsS.randn(12, K)
for name, b in SHOW:
    Z = (sample_student(STUDENT, SZ0) if name == "distilled student"
         else walk_flow({"rectified flow": FLOW, "after reflow": REFLOW}[name], SZ0, b)[0])
    strips[f"{name}|{b}"] = base64.b64encode(np.asarray(
        decode(Z).ravel(), dtype=np.float32).astype(np.float16).tobytes()).decode("ascii")
real_b64 = base64.b64encode(np.asarray(
    REALPIX[:12].ravel(), dtype=np.float32).astype(np.float16).tobytes()).decode("ascii")

payload = {
    "meta": {"seed": SEED_MAIN, "threads": THREADS, "torch": torch.__version__,
             "k": K, "hidden": FHID, "depth": FDEPTH, "epochs": FEPOCHS,
             "base_steps": BASE_STEPS, "reflow_epochs": REP, "student_epochs": CEP,
             "flow_batch": FBATCH, "student_batch": CBATCH,
             "pairs": BIG, "budgets": BUDGETS, "side": 14, "pixels": 196,
             "n_train": int(len(TR)), "n_test": int(len(TE)),
             "judge_accuracy": rnd(J["accuracy"], 4),
             "note": "every arm gets the same number of gradient steps, not the same number of "
                     "epochs: the two later arms train on twenty thousand made up pairs against "
                     "the first arm's few thousand real codes, so equal epochs would be an "
                     "order of magnitude more compute and the comparison would measure that"},
    "setup": {"ceiling": sci(CEIL), "published": sci(pub), "floor": FLOOR, "gamma": sci(GAMMA)},
    "straight": {"flow": rnd(s_flow, 5), "reflow": rnd(s_reflow, 5),
                 "straighter": bool(straighter),
                 "factor": rnd(s_flow / max(s_reflow, 1e-12), 3),
                 "note": "mean distance from the walked path to the straight line joining its "
                         "ends, over the length of that line; zero is a straight line"},
    "gaps": gaps, "arms": arms, "ink": INK,
    "spread": {"rows": spread, "draws": REDRAWS, "pairs": pairs, "resolved": len(RESOLVED),
               "note": "fix the sampler, change only the noise, and this is how far the two "
                       "sample statistic moves by itself; a pair is ranked only when six "
                       "redraws of one arm never reach six redraws of the other, and the band "
                       "comes from the two cells compared rather than from the whole chart"},
    "one_step": {k: v["mmd2"] for k, v in one.items()},
    "best": {k: {"steps": v["steps"], "mmd2": v["mmd2"]} for k, v in best.items()},
    "cost": rnd(cost, 3),
    "loss": {"flow": rnd(FMETA["loss"], 5), "reflow": rnd(RMETA["loss"], 5),
             "student": rnd(SMETA["loss"], 5)},
    "paths": paths, "cloud": cloud, "proj": mat(PROJ, 5),
    "net": {"format": "float16 little-endian, base64; the base flow, the reflowed flow, the "
                      "distilled student, then the decoder, each row major over output units",
            "b64": b64, "count": len(v16), "spec": spec, "k": K,
            "mu": arr(ZMU, 5), "sd": arr(ZSD, 5), "probe": probe},
    "strips": {"pools": strips, "real_b64": real_b64, "per_pool": 12,
               "show": [{"arm": n, "steps": b} for n, b in SHOW]},
}
write_json(OUT / "consistency.json", payload)
