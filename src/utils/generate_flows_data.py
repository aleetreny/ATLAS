"""Data for the ATLAS normalising flows article: the model that refuses to
approximate, and the one thing it cannot do.

Every other model in this module hands you something that is not the log
likelihood. A variational autoencoder hands you a bound and the gap is
unknown; an energy model hands you an unnormalised score and the constant is
unknown. A flow hands you the number itself, exactly, because a change of
variables is an identity rather than an approximation:

    log p(x) = log N(f(x)) + log |det J_f(x)|

The price is that f has to be invertible with a Jacobian determinant you can
compute, and this file measures both halves of that bargain:

A. The identity, verified rather than asserted. The model's density integrates
   to one over the quadrature square; the analytic log determinant matches a
   central difference Jacobian; and pushing points forward and back returns
   them. Three guards, all of which can fire.
B. Depth. Coupling layers, one to sixteen, three seeds, scored by the EXACT KL
   to the truth rather than by a held out average, because in two dimensions
   the integral is a sum.
C. The mask has to alternate. Without it the flow can only ever touch one
   coordinate, and the other one stays exactly standard normal, which is a
   number.
D. Additive against affine: volume preserving against not, at equal depth.
E. The wall. A flow is a diffeomorphism from a gaussian, so it cannot open a
   hole. The truth's ring has one, and what the flow does instead of opening
   it is measurable three ways: the mass it puts inside, how hard it has to
   stretch, and where in the plane its error lives.
F. The scoreboard the module has been building towards: the variational
   autoencoder's bound, the exact number that bound was bounding, this flow's
   exact number, and the energy model's, all on the same density, all in nats.

Run from anywhere: `python src/utils/generate_flows_data.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.generative_common import (          # noqa: E402
    ROOT as REPO, SEED, TOY, arr, banner, cached, decode_half, dense,
    export_flat, export_half, mat, rnd, sci, threads, toy_entropy, toy_grid,
    toy_kl, toy_logpdf, toy_pdf, toy_quadrature_report, toy_sample, write_json)

OUT = REPO / "flows" / "data"
THREADS = threads()

# ---------------------------------------------------------------- configuration
HID = 32                      # width of the two little networks inside a coupling
S_MAX = 2.5                   # the scale is bounded, or the first steps diverge
DEPTHS = [1, 2, 4, 6, 8, 12, 16]
DEPTH_SHOW = 8                # the depth the page runs and draws
SEEDS = [19, 23, 29]
N_TRAIN, N_TEST = 20000, 5000
EPOCHS, BATCH, LR = 240, 512, 3e-3
GRID = 1024                   # the quadrature the whole module is scored on
VIEW_N = 150                  # the picture the page draws
HOLES = [1.0, 1.2]
RING = TOY["ring"]

banner("ATLAS / normalising flows: the exact number, and the hole it cannot open")
print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED}")
CEILING = -toy_entropy(GRID)
print(f"  the truth's own mean log density is {CEILING:.6f} nats, and the grid this is all "
      f"measured on reproduces the closed form mass to "
      f"{toy_quadrature_report((GRID,))[0]['error']}")


def npy(t):
    return t.detach().cpu().numpy().astype(np.float64)


# =============================================================== the flow itself
class Coupling(nn.Module):
    """One affine coupling layer.

    Half the coordinates pass through untouched and condition the other half,
    which is scaled and shifted. That is the entire trick: the Jacobian is
    triangular, so its determinant is the product of the scales and costs
    nothing, and the inverse is one subtraction and one division.

    `parity` says which coordinate passes through. `additive` drops the scale,
    which is the NICE layer: invertible, trivially, and volume preserving,
    which turns out to cost something measurable.
    """

    def __init__(self, parity, additive=False, hid=HID):
        super().__init__()
        self.parity = parity
        self.additive = additive
        self.net = nn.Sequential(nn.Linear(1, hid), nn.Tanh(), nn.Linear(hid, hid),
                                 nn.Tanh(), nn.Linear(hid, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, xa):
        out = self.net(xa)
        s = torch.tanh(out[:, :1]) * S_MAX
        if self.additive:
            s = torch.zeros_like(s)
        return s, out[:, 1:]

    def forward(self, x):
        """data -> latent, and the log determinant of that step."""
        a, b = (0, 1) if self.parity == 0 else (1, 0)
        xa = x[:, a:a + 1]
        xb = x[:, b:b + 1]
        s, t = self.st(xa)
        yb = xb * torch.exp(s) + t
        y = torch.zeros_like(x)
        y[:, a:a + 1] = xa
        y[:, b:b + 1] = yb
        return y, s[:, 0]

    def inverse(self, y):
        a, b = (0, 1) if self.parity == 0 else (1, 0)
        ya = y[:, a:a + 1]
        s, t = self.st(ya)
        xb = (y[:, b:b + 1] - t) * torch.exp(-s)
        x = torch.zeros_like(y)
        x[:, a:a + 1] = ya
        x[:, b:b + 1] = xb
        return x


class Flow(nn.Module):
    def __init__(self, depth, seed=SEED, additive=False, alternate=True, hid=HID):
        super().__init__()
        torch.manual_seed(seed)
        self.depth = depth
        self.layers = nn.ModuleList([
            Coupling(i % 2 if alternate else 0, additive=additive, hid=hid)
            for i in range(depth)])

    def forward(self, x):
        ld = torch.zeros(len(x))
        for layer in self.layers:
            x, s = layer(x)
            ld = ld + s
        return x, ld

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def log_prob(self, x):
        z, ld = self(x)
        return (-0.5 * (z ** 2).sum(1) - np.log(2 * np.pi)) + ld


def fit_flow(depth, seed, additive=False, alternate=True, hid=HID,
             epochs=EPOCHS, n=N_TRAIN):
    """Exact maximum likelihood. There is no bound and no sampler in the loop:
    the objective IS the log likelihood of the training rows."""
    def build():
        X = torch.tensor(toy_sample(n, seed=5000 + seed), dtype=torch.float32)
        net = Flow(depth, seed=seed, additive=additive, alternate=alternate, hid=hid)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            order = torch.randperm(len(X), generator=g)
            for s in range(0, len(X), BATCH):
                loss = -net.log_prob(X[order[s:s + BATCH]]).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
        return {k: v.detach().clone() for k, v in net.state_dict().items()}

    key = (f"flow-d{depth}-s{seed}-add{int(additive)}-alt{int(alternate)}-h{hid}"
           f"-e{epochs}-n{n}-b{BATCH}-lr{LR}-smax{S_MAX}-v1")
    st = cached(key, build)
    net = Flow(depth, seed=seed, additive=additive, alternate=alternate, hid=hid)
    net.load_state_dict(st)
    net.eval()
    return net


def grid_logq(net, m=GRID, chunk=200000):
    X, cell, _ = toy_grid(m)
    out = np.empty(len(X))
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            out[s:s + chunk] = npy(net.log_prob(
                torch.tensor(X[s:s + chunk], dtype=torch.float32)))
    return out, X, cell


def score(net, Xte, m=GRID):
    lq, X, cell = grid_logq(net, m)
    with torch.no_grad():
        held = float(npy(net.log_prob(torch.tensor(Xte, dtype=torch.float32))).mean())
    return {
        "kl": toy_kl(lq, m=m),
        "held": held,
        "mass": float(np.exp(lq).sum() * cell),
    }, lq, X, cell


Xte = toy_sample(N_TEST, seed=SEED + 777)

# ======================================== A. the identity, verified three ways
banner("A. a change of variables is an identity, so it can be checked")
show = fit_flow(DEPTH_SHOW, SEED)
sc_show, lq_show, Xg, CELL = score(show, Xte)

# The analytic log determinant against a central difference jacobian. In
# float32 a central difference at h = 1e-4 carries about 1e-3 of noise from the
# subtraction alone, so the check would be measuring the arithmetic rather than
# the formula: it runs on a float64 copy of the same weights, where the floor
# is nine orders of magnitude lower and the guard can actually fire.
probe = torch.tensor(toy_sample(200, seed=4242), dtype=torch.float32)
showd = fit_flow(DEPTH_SHOW, SEED).double()
probed = probe.double()
with torch.no_grad():
    _, ld_analytic = show(probe)
    _, ld_analytic_d = showd(probed)
h = 1e-5
jac = np.empty((len(probe), 2, 2))
with torch.no_grad():
    for j in range(2):
        e = torch.zeros_like(probed)
        e[:, j] = h
        jac[:, :, j] = (npy(showd(probed + e)[0]) - npy(showd(probed - e)[0])) / (2 * h)
ld_fd = np.log(np.abs(np.linalg.det(jac)))
jac_gap = float(np.abs(ld_fd - npy(ld_analytic_d)).max())

# invertibility
with torch.no_grad():
    z_probe, _ = show(probe)
    back = show.inverse(z_probe)
round_trip = float(np.abs(npy(back) - npy(probe)).max())

sv = np.linalg.svd(jac, compute_uv=False)
guards = {
    "mass": float(f"{sc_show['mass']:.10f}"),
    "leak": sci(abs(1.0 - sc_show["mass"])),
    "quadrature": toy_quadrature_report((512, GRID, 2048)),
    "jacobian_gap": sci(jac_gap),
    "jacobian_step": h,
    "round_trip": sci(round_trip),
    "probe": int(len(probe)),
}
print(f"  the model's density integrates to {guards['mass']} over the same square on which the "
      f"closed form truth integrates to one, so {guards['leak']} of it sits outside the box: a "
      f"flow puts mass wherever the gaussian it starts from reaches, and the truth does not")
print(f"  its analytic log determinant matches a central difference jacobian to "
      f"{guards['jacobian_gap']} on {len(probe)} points")
print(f"  and pushing those points forward and back returns them to {guards['round_trip']}")

# ============================================================ B. depth
banner("B. how much depth buys, scored by the exact KL")
depth_rows = []
per_depth_nets = {}
for d in DEPTHS:
    kls, helds, params = [], [], None
    for s in SEEDS:
        net = fit_flow(d, s)
        sc, _, _, _ = score(net, Xte)
        kls.append(sc["kl"])
        helds.append(sc["held"])
        params = sum(p.numel() for p in net.parameters())
        if s == SEED:
            per_depth_nets[d] = net
    depth_rows.append({
        "depth": d, "params": int(params),
        "kl": rnd(float(np.mean(kls)), 5), "kl_sd": rnd(float(np.std(kls)), 5),
        "kl_min": rnd(float(np.min(kls)), 5), "kl_max": rnd(float(np.max(kls)), 5),
        "held": rnd(float(np.mean(helds)), 5), "held_sd": rnd(float(np.std(helds)), 5),
        "short_of_ceiling": rnd(CEILING - float(np.mean(helds)), 5),
    })
    print(f"  {d:>2} coupling layers ({params:>5} weights): KL {np.mean(kls):.5f} "
          f"(seeds {np.min(kls):.5f} to {np.max(kls):.5f}), held out {np.mean(helds):.5f} "
          f"against a ceiling of {CEILING:.5f}")

FLOOR = float(np.median([r["kl_sd"] for r in depth_rows]))
best_row = min(depth_rows, key=lambda r: r["kl"])
# "flat from here" is a claim about every later row, so it is checked against
# every later row rather than against the first one that happens to qualify.
flat_from = None
for i, r in enumerate(depth_rows):
    if all(depth_rows[j]["kl"] - best_row["kl"] <= 2 * FLOOR for j in range(i, len(depth_rows))):
        flat_from = r["depth"]
        break
backwards = [(depth_rows[i - 1]["depth"], depth_rows[i]["depth"])
             for i in range(1, len(depth_rows))
             if depth_rows[i]["kl"] > depth_rows[i - 1]["kl"] + 2 * FLOOR]
print(f"  the seed floor is {FLOOR:.5f} nats; the best is {best_row['kl']} at depth "
      f"{best_row['depth']}, and the staircase is flat from depth "
      f"{flat_from if flat_from else 'nowhere inside this sweep'} onwards"
      + (f", going backwards at {backwards}" if backwards else ", never going backwards"))

# ================================================ C. the mask has to alternate
banner("C. what a mask that never alternates leaves untouched")
stuck = fit_flow(DEPTH_SHOW, SEED, alternate=False)
sc_stuck, lq_stuck, _, _ = score(stuck, Xte)
# the coordinate no layer ever transforms: its model marginal is exactly the
# standard normal, and the truth's is not
xs = np.linspace(-TOY["domain"], TOY["domain"], GRID)
P2 = toy_pdf(np.c_[np.repeat(xs, GRID), np.tile(xs, GRID)]).reshape(GRID, GRID)
dx = float(xs[1] - xs[0])
truth_marg = P2.sum(1) * dx                       # marginal of the first coordinate
normal_marg = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)
model_marg = np.exp(lq_stuck.reshape(GRID, GRID)).sum(1) * dx
kl_marg_normal = float((truth_marg * (np.log(np.maximum(truth_marg, 1e-300))
                                      - np.log(normal_marg))).sum() * dx)
kl_marg_model = float((truth_marg * (np.log(np.maximum(truth_marg, 1e-300))
                                     - np.log(np.maximum(model_marg, 1e-300)))).sum() * dx)
mask = {
    "kl": rnd(sc_stuck["kl"], 5),
    "kl_alternating": rnd(depth_rows[DEPTHS.index(DEPTH_SHOW)]["kl"], 5),
    "marginal_vs_normal": rnd(kl_marg_normal, 6),
    "marginal_vs_model": rnd(kl_marg_model, 6),
    "gap": sci(float(np.abs(model_marg - normal_marg).max())),
    "depth": DEPTH_SHOW,
}
print(f"  with every layer conditioning on the same half, the first coordinate is never "
      f"touched: its model marginal differs from the standard normal by at most "
      f"{mask['gap']} across the whole grid")
print(f"  so the KL of that marginal from the truth's is {kl_marg_model:.6f}, which is the "
      f"{kl_marg_normal:.6f} of the untouched prior, and the total KL is {sc_stuck['kl']:.5f} "
      f"against {mask['kl_alternating']} with the mask alternating")

# ==================================================== D. additive against affine
banner("D. volume preserving, and what that costs")
add_rows = []
for d in DEPTHS:
    kls = []
    for s in SEEDS:
        net = fit_flow(d, s, additive=True)
        sc, _, _, _ = score(net, Xte)
        kls.append(sc["kl"])
    aff = next(r for r in depth_rows if r["depth"] == d)
    add_rows.append({"depth": d, "kl": rnd(float(np.mean(kls)), 5),
                     "kl_sd": rnd(float(np.std(kls)), 5),
                     "affine": aff["kl"], "gap": rnd(float(np.mean(kls)) - aff["kl"], 5)})
    print(f"  {d:>2} layers: additive {np.mean(kls):.5f} against affine {aff['kl']:.5f}, "
          f"a gap of {np.mean(kls) - aff['kl']:+.5f}")
add_net = fit_flow(DEPTH_SHOW, SEED, additive=True)
with torch.no_grad():
    _, ld_add = add_net(probe)
add_logdet = float(np.abs(npy(ld_add)).max())
print(f"  and the log determinant of the additive flow is {add_logdet:.3e} at its worst over "
      f"{len(probe)} points, which is the guard that it really is volume preserving")

# ===================================================== E. the topology wall
banner("E. the hole a diffeomorphism cannot open")
r_grid = np.sqrt((Xg ** 2).sum(1))
truth_p = toy_pdf(Xg)
hole_rows = []
for d in DEPTHS:
    lq, _, _ = grid_logq(per_depth_nets[d])
    q = np.exp(lq)
    row = {"depth": d}
    for rr in HOLES:
        inside = r_grid < rr
        row[f"model_{rr}"] = sci(float(q[inside].sum() * CELL))
        row[f"truth_{rr}"] = sci(float(truth_p[inside].sum() * CELL))
    # where the error lives: the ring's own annulus against the two blobs
    ann = (r_grid > RING["r0"] - 4 * RING["sigma"]) & (r_grid < RING["r0"] + 4 * RING["sigma"])
    row["kl_ring"] = rnd(float((truth_p * (np.log(np.maximum(truth_p, 1e-300)) - lq)
                                * ann).sum() * CELL), 5)
    row["kl_rest"] = rnd(float((truth_p * (np.log(np.maximum(truth_p, 1e-300)) - lq)
                                * ~ann).sum() * CELL), 5)
    # how hard it has to stretch, at the data
    pr = torch.tensor(toy_sample(1500, seed=99), dtype=torch.float32)
    jj = np.empty((len(pr), 2, 2))
    with torch.no_grad():
        for j in range(2):
            e = torch.zeros_like(pr)
            e[:, j] = h
            jj[:, :, j] = (npy(per_depth_nets[d](pr + e)[0])
                           - npy(per_depth_nets[d](pr - e)[0])) / (2 * h)
    s_ = np.linalg.svd(jj, compute_uv=False)
    cond = s_[:, 0] / np.maximum(s_[:, 1], 1e-30)
    row["cond_median"] = rnd(float(np.median(cond)), 3)
    row["cond_p999"] = rnd(float(np.quantile(cond, 0.999)), 3)
    row["cond_max"] = rnd(float(cond.max()), 3)
    row["logdet_max"] = rnd(float(np.abs(np.log(np.abs(np.linalg.det(jj)))).max()), 4)
    hole_rows.append(row)
    print(f"  depth {d:>2}: inside r < {HOLES[1]} the truth puts {row[f'truth_{HOLES[1]}']} and "
          f"this flow puts {row[f'model_{HOLES[1]}']}; KL on the ring {row['kl_ring']} against "
          f"{row['kl_rest']} everywhere else; the jacobian's condition number reaches "
          f"{row['cond_max']}")
deep = hole_rows[-1]
ratio = float(deep[f"model_{HOLES[1]}"]) / float(deep[f"truth_{HOLES[1]}"])
col = [float(r[f"model_{HOLES[1]}"]) for r in hole_rows]
tail = col[1:]
print(f"  at the deepest flow measured the hole still holds {ratio:,.0f} times the mass the "
      f"truth puts there. From {DEPTHS[0]} layer to {DEPTHS[1]} it falls by a factor of "
      f"{col[0] / col[1]:,.0f}, and after that it stops: the rest of the column runs from "
      f"{min(tail):.3e} to {max(tail):.3e} with no trend, against a truth of "
      f"{deep[f'truth_{HOLES[1]}']}")
hole_stats = {"first_drop": rnd(col[0] / col[1], 1), "tail_min": sci(min(tail)),
              "tail_max": sci(max(tail)), "column": [sci(v) for v in col]}

# =============================================== F. the scoreboard of the module
banner("F. the same density, three models, three kinds of number")
board = {"ceiling": rnd(CEILING, 6), "flow": {"exact": rnd(sc_show["held"], 5),
                                              "kl": rnd(sc_show["kl"], 5),
                                              "depth": DEPTH_SHOW}}


def read_sibling(name, path):
    p = REPO / path
    if not p.is_file():
        print(f"  {name}: {path} is not there yet, so the scoreboard exports null for it")
        return None
    return json.loads(p.read_text("utf-8"))


vae = read_sibling("the variational autoencoder", "vae/data/vae.json")
if vae:
    t = vae["toy"]
    assert abs(t["ceiling"] - board["ceiling"]) < 1e-5, (
        f"the variational article measured a different truth: {t['ceiling']} against "
        f"{board['ceiling']}")
    board["vae"] = {"bound": t["elbo"], "exact": t["exact"], "gap": t["gap"],
                    "kl": t["kl_to_truth"], "hole": t["picture"]["hole"]["model"]}
    print(f"  the variational autoencoder: bound {t['elbo']}, exact {t['exact']}, so its own "
          f"objective was {t['gap']} nats away from the number it was bounding")
ebm = read_sibling("the energy model", "ebm/data/ebm.json")
if ebm:
    try:
        rows = ebm["arms"]["rows"]
        exact_arm = next(r for r in rows if "exact" in str(r.get("arm", r.get("name", ""))))
        board["ebm"] = {"kl": exact_arm.get("kl"), "held": exact_arm.get("heldout")}
        print(f"  the energy model, trained with an exact gradient: KL {board['ebm']['kl']}")
    except (KeyError, StopIteration) as err:
        board["ebm"] = None
        print(f"  the energy model's file is there but its shape is not what was expected "
              f"({err}), so the scoreboard exports null rather than a guess")
board["flow"]["hole"] = deep[f"model_{HOLES[1]}"]
board["truth_hole"] = deep[f"truth_{HOLES[1]}"]
print(f"  and this flow: exact {board['flow']['exact']}, KL {board['flow']['kl']}, with no "
      f"bound and no constant left over")

# ====================================================== G. what the browser runs
banner("G. exporting a flow small enough to run in a page")
blocks = []
for i, layer in enumerate(show.layers):
    for j, lin in enumerate([layer.net[0], layer.net[2], layer.net[4]]):
        blocks.append((f"l{i}_{j}", npy(lin.weight).T, npy(lin.bias)))
spec, b64, v16 = export_half(blocks)
w16 = v16.astype(np.float64)


def q_forward(X):
    """The forward pass exactly as the browser will run it, on the quantised
    weights, because those are the numbers the page will have."""
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    ld = np.zeros(len(X))
    for i, layer in enumerate(show.layers):
        a, b = (0, 1) if layer.parity == 0 else (1, 0)
        xa = X[:, a:a + 1]
        hh = np.tanh(dense(w16, spec[3 * i], xa))
        hh = np.tanh(dense(w16, spec[3 * i + 1], hh))
        o = dense(w16, spec[3 * i + 2], hh)
        s = np.tanh(o[:, :1]) * S_MAX
        Y = np.empty_like(X)
        Y[:, a] = X[:, a]
        Y[:, b] = X[:, b] * np.exp(s[:, 0]) + o[:, 1]
        X = Y
        ld = ld + s[:, 0]
    return X, ld


zq_probe, ldq_probe = q_forward(npy(probe))
zq, ldq = q_forward(Xte[:2000])
logq_quant = -0.5 * (zq ** 2).sum(1) - np.log(2 * np.pi) + ldq
with torch.no_grad():
    logq_full = npy(show.log_prob(torch.tensor(Xte[:2000], dtype=torch.float32)))
quant = {
    "count": int(len(v16)),
    "logp_shift": sci(float(np.abs(logq_quant - logq_full).max())),
    "held_full": rnd(float(logq_full.mean()), 5),
    "held_quantised": rnd(float(logq_quant.mean()), 5),
}
print(f"  {len(v16):,} numbers, float16 base64 is {len(b64) / 1024:.0f} kB; quantising moves "
      f"the log density by at most {quant['logp_shift']} and the held out average from "
      f"{quant['held_full']} to {quant['held_quantised']}")

VIEW = TOY["view"]
gc_ = np.linspace(-VIEW, VIEW, VIEW_N)
GX = np.c_[np.repeat(gc_, VIEW_N), np.tile(gc_, VIEW_N)]
with torch.no_grad():
    view_model = np.exp(npy(show.log_prob(torch.tensor(GX, dtype=torch.float32))))
view_truth = toy_pdf(GX)
picture = {
    "n": VIEW_N, "lo": rnd(-VIEW, 4), "hi": rnd(VIEW, 4),
    "truth_b64": export_flat(view_truth)[0],
    "model_b64": export_flat(view_model)[0],
    "peak": rnd(float(max(view_truth.max(), view_model.max())), 4),
    "sample": mat(Xte[:700], 3),
}

payload = {
    "meta": {
        "seed": SEED, "seeds": SEEDS, "threads": THREADS, "torch": torch.__version__,
        "hidden": HID, "s_max": S_MAX, "depths": DEPTHS, "depth_show": DEPTH_SHOW,
        "n_train": N_TRAIN, "n_test": N_TEST, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "grid": GRID, "ceiling": rnd(CEILING, 6), "ring": RING, "holes": HOLES,
        "note": "every KL on this page is the exact integral against a density written in "
                "closed form, computed on the same grid the module uses everywhere",
    },
    "guards": guards,
    "depth": {"rows": depth_rows, "floor": rnd(FLOOR, 5), "best": best_row["depth"],
              "flat_from": flat_from, "backwards": backwards},
    "mask": mask,
    "additive": {"rows": add_rows, "logdet_max": sci(add_logdet)},
    "topology": {"rows": hole_rows, "radii": HOLES,
                 "ratio_deepest": rnd(ratio, 1), **hole_stats},
    "board": board,
    "net": {
        "format": "float16 little-endian, base64; three dense layers per coupling, row major "
                  "over output units, then biases",
        "layers": spec, "count": int(len(v16)), "weights_b64": b64,
        "parity": [int(l.parity) for l in show.layers],
        "s_max": S_MAX, "quant": quant,
        # Measured on the QUANTISED weights, because those are the numbers the
        # browser has: against the full precision ones the same comparison would
        # be measuring float16, which moves a log density by 0.05 here.
        "probe": {"x": mat(npy(probe[:24]), 5),
                  "z": mat(zq_probe[:24], 5),
                  "logdet": arr(ldq_probe[:24], 5),
                  "note": "z and logdet are the quantised network's, not the trained one's"},
    },
    "picture": picture,
}
write_json(OUT / "flows.json", payload)
