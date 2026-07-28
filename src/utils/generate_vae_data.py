"""Data for the ATLAS variational autoencoder article: the promise the
autoencoders article ended on, kept and then measured.

That article closed by pointing at a hole in its own argument. Its decoder is a
function from two numbers to an image and it answers at points where no digit
ever was, so it is most of a way to generate digits: sample a point, decode it,
look. What was missing was any reason to believe a sampled point lands somewhere
the decoder handles well, which is a question about the DISTRIBUTION of the code
and not about reconstruction at all.

So the first thing this file does is take that article's own shipped float16
network, off its own JSON, and try exactly that. The result is the article's
opening: the codes of a plain autoencoder are not a distribution anyone can
sample from, and the failure is measurable three different ways.

Then six blocks:

1. The hole, measured on the previous article's weights, with three different
   ways of proposing a point so the failure is not a straw man.
2. The same architecture trained as a variational autoencoder: what the extra
   term costs in reconstruction and what it buys in sampling.
3. The reparameterisation trick against the score function estimator. Both are
   unbiased and one of them is usable; the ratio of their variances is the whole
   reason the model exists in this form, and it is a number.
4. The rate, the distortion, and the units that die. A sweep over beta, with the
   count of latent dimensions the model still uses at each setting.
5. The two dimensional truth of the module, where the bound can be checked
   against the thing it bounds, because in two dimensions the marginal is an
   integral you can actually do. Two independent quadratures agree, and the gap
   between the bound and the truth is measured rather than assumed small.
6. The weights the page runs, in float16, with every figure re-measured on the
   quantised numbers.

Run from anywhere: `python src/utils/generate_vae_data.py`.
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
    DIGITS, ROOT as REPO, SEED, TOY, arr, banner, cached, decode_half, dense,
    digits14, export_flat, export_half, judge, mat, rnd, sci, threads,
    toy_entropy, toy_grid,
    toy_kl, toy_quadrature_report, toy_logpdf, toy_pdf, toy_sample, write_json)

OUT = REPO / "vae" / "data"
THREADS = threads()
torch.manual_seed(SEED)

HID = 64                  # the hidden width the autoencoders article used
K_SHOW = 2                # the latent the page draws
K_COLLAPSE = 16           # wide enough that units can die and be counted
BETAS = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEEDS = [19, 23, 29]
EPOCHS = 400
BATCH = 128
N_EXACT = 400          # rows the exact marginal is computed on, both ways

banner("ATLAS / variational autoencoder: somewhere to sample from")
print(f"torch {torch.__version__} on {THREADS} threads, seed {SEED}")


# ============================================================ shared machinery
def npy(t):
    return t.detach().cpu().numpy().astype(np.float64)


class VAE(nn.Module):
    """The autoencoders article's network with one change: the encoder emits a
    distribution instead of a point.

    Same widths, same tanh hidden layers, same logistic output, so the beta = 0
    arm of the sweep below is that article's model with the noise switched off,
    and every comparison in this file changes one thing.

    The decoder's likelihood is Gaussian with a single learned scale, which
    keeps the objective in nats and makes the bound comparable with the exact
    log likelihood computed in block 5. A Bernoulli likelihood would have been
    cheaper to write and would have made every number here incomparable with
    the rest of the module.
    """

    def __init__(self, p, h, k, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.p, self.k = p, k
        self.e1 = nn.Linear(p, h)
        self.emu = nn.Linear(h, k)
        self.elv = nn.Linear(h, k)
        self.d1 = nn.Linear(k, h)
        self.d2 = nn.Linear(h, p)
        self.log_sigma = nn.Parameter(torch.tensor(-1.5))

    def encode(self, x):
        h = torch.tanh(self.e1(x))
        return self.emu(h), self.elv(h).clamp(-9.0, 4.0)

    def decode(self, z):
        return torch.sigmoid(self.d2(torch.tanh(self.d1(z))))

    def terms(self, x, sample=True, generator=None):
        """(reconstruction log likelihood, KL to the prior), both per row."""
        mu, lv = self.encode(x)
        if sample:
            eps = torch.randn(mu.shape, generator=generator)
            z = mu + torch.exp(0.5 * lv) * eps
        else:
            z = mu
        xh = self.decode(z)
        s = torch.exp(self.log_sigma)
        ll = (-0.5 * ((x - xh) / s) ** 2 - self.log_sigma
              - 0.5 * np.log(2.0 * np.pi)).sum(1)
        kl = 0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv).sum(1)
        return ll, kl, mu, lv, xh


def fit_vae(X, k, beta, seed, epochs=EPOCHS, hid=HID, lr=2e-3, batch=BATCH):
    net = VAE(X.shape[1], hid, k, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    n = len(Xt)
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch):
            xb = Xt[order[s:s + batch]]
            ll, kl, _, _, _ = net.terms(xb, generator=g)
            loss = -(ll - beta * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def gauss_fit(Z):
    return Z.mean(0), np.cov(Z.T, bias=True) + 1e-9 * np.eye(Z.shape[1])


def gauss_draw(mean, cov, n, seed):
    rs = np.random.RandomState(seed)
    return rs.multivariate_normal(mean, cov, size=n)


def nn_distance(A, B, chunk=256):
    """For every row of A, the distance to the closest row of B."""
    out = np.empty(len(A))
    for s in range(0, len(A), chunk):
        d = ((A[s:s + chunk, None, :] - B[None, :, :]) ** 2).sum(-1)
        out[s:s + chunk] = np.sqrt(d.min(1))
    return out


def rbf_mmd2(A, B, gamma):
    """The unbiased two sample statistic, which is zero in expectation when the
    two pools come from the same distribution and positive when they do not."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    def k(U, V):
        d = ((U ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2.0 * U @ V.T)
        return np.exp(-gamma * np.maximum(d, 0.0))

    Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B)
    na, nb = len(A), len(B)
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    return float(Kaa.sum() / (na * (na - 1)) + Kbb.sum() / (nb * (nb - 1))
                 - 2.0 * Kab.mean())


def c2st(A, B, seed=0):
    """Classifier two sample test: how well anything can tell the two pools
    apart. Half is perfect (indistinguishable), one is trivial."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.r_[np.asarray(A, dtype=float), np.asarray(B, dtype=float)]
    yy = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    return float(cross_val_score(clf, X, yy, cv=5, scoring="accuracy").mean())


def pool_scores(pixels, J, real_ref, gamma, full=True):
    """One pool of images, scored four ways that fail differently.

    The classifier says whether a picture looks like SOME digit, and it is the
    weakest of the four on purpose: a classifier trained only on digits is
    confident about things that are not digits, which is a fact this module
    measures directly in the energy based article. The distance to the nearest
    training digit catches a confident blur. The two sample statistics compare
    the whole pool against real held out digits and have a floor measured from
    real data split in two, which is the only scale any of these numbers has.
    """
    pixels = np.asarray(pixels, dtype=float)
    Pp = J["proba"](pixels)
    q = Pp.mean(0)
    out = {
        "confidence": rnd(float(Pp.max(1).mean()), 4),
        "class_entropy": rnd(float(-(q * np.log(np.maximum(q, 1e-12))).sum()), 4),
        "nn_distance": rnd(float(nn_distance(pixels, real_ref).mean()), 4),
    }
    if full:
        out["mmd2"] = sci(rbf_mmd2(pixels, real_ref[:len(pixels) * 2], gamma))
        out["c2st"] = rnd(c2st(pixels, real_ref[:len(pixels)]), 4)
    return out


# ================================================== 1. the hole, on their weights
banner("1. sampling from the previous article's latent space")
X, y, TR, TE = digits14()
P = X.shape[1]
XT, XV = X[TR], X[TE]
J = judge()
print(f"  the shared judge gets {J['accuracy'] * 100:.2f}% of the {len(TE)} held out digits "
      f"right and is {J['real_confidence']:.4f} confident on them")
print(f"  the digits are asserted identical to the autoencoders article's by running its own "
      f"float16 encoder on them: worst code difference {DIGITS['assert_worst']}, against a "
      f"quantisation floor of {DIGITS['assert_floor']}")

AE = json.loads((REPO / "autoencoders" / "data" / "autoencoders.json").read_text("utf-8"))
ae_w = decode_half(AE["net"]["weights_b64"], AE["net"]["count"])
ae_layers = AE["net"]["layers"]


def ae_decode(Z):
    h = np.tanh(dense(ae_w, ae_layers[2], Z))
    return 1.0 / (1.0 + np.exp(-dense(ae_w, ae_layers[3], h)))


# The kernel width for the two sample statistic, and the floor both statistics
# have. A pool of generated images is compared against real held out digits,
# and "how far apart" only means something against how far apart two pools of
# REAL digits are, which is what the floor measures.
N_DRAW = 600
sub = XT[np.random.RandomState(1).choice(len(XT), 400, replace=False)]
d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
GAMMA = float(1.0 / np.median(d2[d2 > 0]))
floor_mmd, floor_c2st = [], []
for s in range(5):
    idx = np.random.RandomState(50 + s).permutation(len(XT))
    A, B = XT[idx[:N_DRAW]], XT[idx[N_DRAW:2 * N_DRAW]]
    floor_mmd.append(rbf_mmd2(A, B, GAMMA))
    floor_c2st.append(c2st(A, B, seed=s))
FLOOR = {
    "gamma": sci(GAMMA), "splits": 5, "pool": N_DRAW,
    "mmd2": sci(float(np.mean(floor_mmd))), "mmd2_sd": sci(float(np.std(floor_mmd))),
    "mmd2_max": sci(float(np.max(np.abs(floor_mmd)))),
    "c2st": rnd(float(np.mean(floor_c2st)), 4), "c2st_sd": rnd(float(np.std(floor_c2st)), 4),
}
print(f"  the floor, from real digits split in two {FLOOR['splits']} times: the two sample "
      f"statistic is {FLOOR['mmd2']} (largest {FLOOR['mmd2_max']}) and a classifier told to "
      f"separate the halves gets {FLOOR['c2st']}, which is chance")

AE_Z = np.array(AE["net"]["latent"], dtype=float)
AE_ZT = AE_Z[TR]
real_hold = pool_scores(XV, J, XT, GAMMA)
recon_hold = pool_scores(ae_decode(AE_Z[TE]), J, XT, GAMMA)
print(f"  real held out digits: confidence {real_hold['confidence']}, two sample "
      f"{real_hold['mmd2']}, separable at {real_hold['c2st']}")
print(f"  the same digits through their autoencoder: confidence {recon_hold['confidence']}, "
      f"two sample {recon_hold['mmd2']}, separable at {recon_hold['c2st']}")

mu_ae, cov_ae = gauss_fit(AE_ZT)
proposals = {
    "gaussian": gauss_draw(mu_ae, cov_ae, N_DRAW, 101),
    "unit": np.random.RandomState(102).randn(N_DRAW, K_SHOW),
    "jitter": (AE_ZT[np.random.RandomState(103).choice(len(AE_ZT), N_DRAW)]
               + np.random.RandomState(104).randn(N_DRAW, K_SHOW)
               * 0.25 * np.sqrt(np.diag(cov_ae))),
}
hole = {"real": real_hold, "reconstruction": recon_hold, "draws": N_DRAW,
        "floor": FLOOR, "arms": {}}
for name, Z in proposals.items():
    sc = pool_scores(ae_decode(Z), J, XT, GAMMA)
    sc["code_distance"] = rnd(float(nn_distance(Z, AE_ZT).mean()), 4)
    hole["arms"][name] = sc
    print(f"  autoencoder, {name:>8} proposal: confidence {sc['confidence']}, two sample "
          f"{sc['mmd2']}, separable at {sc['c2st']}, nearest training digit at "
          f"{sc['nn_distance']}")
# how far a proposed point sits from the codes, in the units the codes themselves set
d_self = np.sqrt(((AE_ZT[:, None, :] - AE_ZT[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(d_self, np.inf)
hole["code_spacing"] = rnd(float(np.median(d_self.min(1))), 4)
hole["gaussian_ratio"] = rnd(hole["arms"]["gaussian"]["code_distance"] / hole["code_spacing"], 3)
print(f"  a training code's own nearest neighbour is {hole['code_spacing']} away, so a "
      f"gaussian proposal lands {hole['gaussian_ratio']} times that far from anything the "
      f"decoder was ever trained on")


# ============================================== 2. the same network, trained as a VAE
banner("2. the same architecture with a distribution in the middle")


def digit_arm(k, beta, seed, epochs=EPOCHS):
    def build():
        net = fit_vae(XT, k, beta, seed, epochs=epochs)
        with torch.no_grad():
            Xv = torch.tensor(XV, dtype=torch.float32)
            Xt = torch.tensor(XT, dtype=torch.float32)
            ll, kl, mu, lv, _ = net.terms(Xv, sample=False)
            xh = net.decode(mu)
            mu_t, lv_t = net.encode(Xt)
            per_dim = 0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv).mean(0)
        return {
            "state": {kk: v.detach().clone() for kk, v in net.state_dict().items()},
            "mse": float(((npy(xh) - XV) ** 2).mean()),
            "kl": float(npy(kl).mean()),
            "elbo": float(npy(ll - kl).mean()),
            "sigma": float(np.exp(npy(net.log_sigma))),
            "per_dim_kl": npy(per_dim).tolist(),
            "mu_test": npy(mu), "lv_test": npy(lv), "mu_train": npy(mu_t),
            "lv_train": npy(lv_t),
        }

    key = (f"vae-digits-k{k}-b{beta}-s{seed}-e{epochs}-h{HID}-n{len(XT)}"
           f"-p{P}-batch{BATCH}-v1")
    return cached(key, build)


def rebuild(k, blob, seed=SEED):
    net = VAE(P, HID, k, seed=seed)
    net.load_state_dict(blob["state"])
    return net


base = digit_arm(K_SHOW, 1.0, SEED)
net2 = rebuild(K_SHOW, base)
with torch.no_grad():
    vae_sample_z = torch.randn(N_DRAW, K_SHOW, generator=torch.Generator().manual_seed(105))
    vae_pool = npy(net2.decode(vae_sample_z))
vae_hole = pool_scores(vae_pool, J, XT, GAMMA)
vae_hole["code_distance"] = rnd(float(nn_distance(npy(vae_sample_z), base["mu_train"]).mean()), 4)
recon_vae = pool_scores(np.clip(npy(net2.decode(torch.tensor(base["mu_test"],
                                                             dtype=torch.float32))), 0, 1),
                        J, XT, GAMMA)
compare = {
    "ae_test_mse": AE["generalisation"]["test_mse"],
    "vae_test_mse": rnd(base["mse"], 6),
    "mse_ratio": rnd(base["mse"] / AE["generalisation"]["test_mse"], 4),
    "kl_nats": rnd(base["kl"], 4),
    "elbo": rnd(base["elbo"], 4),
    "sigma": rnd(base["sigma"], 5),
    "prior_sample": vae_hole,
    "reconstruction": recon_vae,
}
print(f"  reconstruction on the same held out digits: {compare['vae_test_mse']} against the "
      f"autoencoder's {compare['ae_test_mse']}, a ratio of {compare['mse_ratio']}")
print(f"  and sampling from the prior: two sample {vae_hole['mmd2']} against the "
      f"autoencoder's best proposal "
      f"{min(a['mmd2'] for a in hole['arms'].values())}, with a floor of {FLOOR['mmd2']}")

# The controlled version of the same question, with exactly one thing changed.
# The beta = 0 arm IS an autoencoder: same widths, same activations, same
# optimiser, same batches, and no second term. So the pair below differs in one
# number and in nothing else, and it can be asked at several bottleneck sizes,
# which turns out to matter more than anything else on this page.
print("\n  the same comparison at four bottleneck sizes, one thing changed:")


def gauss_kl_to_unit(mean, cov):
    """KL from a fitted gaussian to the standard one, in closed form: how far
    the codes a model actually produces are from the place it samples."""
    d = len(mean)
    sign, logdet = np.linalg.slogdet(cov)
    return float(0.5 * (np.trace(cov) + mean @ mean - d - logdet))


ladder = []
for k in (2, 4, 8, 16):
    row = {"k": k}
    for name, beta in (("ae", 0.0), ("vae", 1.0)):
        a = digit_arm(k, beta, SEED)
        net_k = rebuild(k, a)
        Zt = a["mu_train"]
        m_, c_ = gauss_fit(Zt)
        # An autoencoder has no prior, so the only proposal it can be given is a
        # gaussian fitted to the codes it produced. Handing the variational arm
        # its own N(0, I) and nothing else would be comparing a fitted proposal
        # against an unfitted one, so both get both: the fitted gaussian, and,
        # for the variational arm, the prior it was actually trained towards.
        Zs = gauss_draw(m_, c_, N_DRAW, 300 + k)
        with torch.no_grad():
            pool_fit = npy(net_k.decode(torch.tensor(Zs, dtype=torch.float32)))
            recon = npy(net_k.decode(torch.tensor(a["mu_test"], dtype=torch.float32)))
        sc = pool_scores(pool_fit, J, XT, GAMMA)
        sc["recon_mmd2"] = sci(rbf_mmd2(recon, XT[:2 * len(recon)], GAMMA))
        sc["recon_c2st"] = rnd(c2st(recon, XT[:len(recon)]), 4)
        sc["aggregate_kl"] = rnd(gauss_kl_to_unit(m_, c_), 4)
        if beta > 0.0:
            Zp = np.random.RandomState(400 + k).randn(N_DRAW, k)
            with torch.no_grad():
                pool_prior = npy(net_k.decode(torch.tensor(Zp, dtype=torch.float32)))
            pr = pool_scores(pool_prior, J, XT, GAMMA)
            sc["prior_mmd2"] = pr["mmd2"]
            sc["prior_c2st"] = pr["c2st"]
        row[name] = {"mse": rnd(a["mse"], 6), "kl": rnd(a["kl"], 3), **sc}
    # The two sample statistic is the unbiased estimator, so it can come out
    # negative when two pools really do match, which is why a ratio between two
    # of them is not a quantity: everything is read against the floor measured
    # on real digits instead.
    fl = float(FLOOR["mmd2_max"])
    row["ae_floors"] = rnd(row["ae"]["mmd2"] / fl, 2)
    row["vae_floors"] = rnd(row["vae"]["mmd2"] / fl, 2)
    ladder.append(row)
    print(f"    k = {k:>2}: two sample statistic {row['ae']['mmd2']} for the autoencoder "
          f"against {row['vae']['mmd2']} for the variational one, which is "
          f"{row['ae_floors']} and {row['vae_floors']} times the floor; separable at "
          f"{row['ae']['c2st']} against {row['vae']['c2st']}, floor {FLOOR['c2st']}")
    print(f"            rebuilding a held out digit: {row['ae']['recon_mmd2']} against "
          f"{row['vae']['recon_mmd2']}; the codes sit {row['ae']['aggregate_kl']} and "
          f"{row['vae']['aggregate_kl']} nats from the standard gaussian; the variational "
          f"arm from its own prior instead of the fitted one: {row['vae']['prior_mmd2']}")
compare["ladder"] = ladder
# Where the difference comes from. If the autoencoder's samples are closer to
# real digits, the first question is whether its RECONSTRUCTIONS are closer too,
# because then the sampling result is inherited from a sharper decoder and says
# nothing about which prior is better. That decomposition is the point of the
# two extra columns above.
w2 = ladder[0]
compare["decomposition"] = {
    "sample_gap": rnd(w2["ae"]["mmd2"] - w2["vae"]["mmd2"], 5),
    "recon_gap": rnd(w2["ae"]["recon_mmd2"] - w2["vae"]["recon_mmd2"], 5),
}


# ================================================ 3. the two gradient estimators
banner("3. the trick, and what it is worth")


def estimator_variance(k, seed=SEED, draws=400, rows=64):
    """Both estimators of the same gradient, on the same model and the same batch.

    The pathwise estimator differentiates through the sample; the score function
    estimator differentiates the log density of the sampling distribution and
    treats the objective as a black box. Both are unbiased for the same
    quantity, which is checked here rather than assumed: their means agree.
    The comparison is otherwise identical, including the number of samples, and
    the score function arm is also run WITH a moving average baseline, because
    comparing it without one is comparing against a straw man.
    """
    def build():
        net = rebuild(k, digit_arm(k, 1.0, seed), seed=seed)
        xb = torch.tensor(XT[:rows], dtype=torch.float32)
        g = torch.Generator().manual_seed(seed + 7)
        # The ENCODER's parameters, and only those. For the decoder the two
        # estimators are the same expression, so including it would add an
        # identical variance to both arms and quietly shrink the ratio the
        # article is about.
        params = [net.e1.weight, net.emu.weight, net.elv.weight]
        path, score, score_b = [], [], []
        base_run = None
        for _ in range(draws):
            mu, lv = net.encode(xb)
            eps = torch.randn(mu.shape, generator=g)
            std = torch.exp(0.5 * lv)

            z = mu + std * eps                       # pathwise
            xh = net.decode(z)
            s = torch.exp(net.log_sigma)
            ll = (-0.5 * ((xb - xh) / s) ** 2 - net.log_sigma
                  - 0.5 * np.log(2.0 * np.pi)).sum(1)
            kl = 0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv).sum(1)
            gp = torch.autograd.grad((ll - kl).mean(), params)
            path.append(torch.cat([v.reshape(-1) for v in gp]).detach())

            mu, lv = net.encode(xb)
            std = torch.exp(0.5 * lv)
            zd = (mu + std * eps).detach()           # same noise, no path through it
            xh = net.decode(zd)
            ll = (-0.5 * ((xb - xh) / torch.exp(net.log_sigma)) ** 2 - net.log_sigma
                  - 0.5 * np.log(2.0 * np.pi)).sum(1).detach()
            kl = 0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv).sum(1)
            logq = (-0.5 * ((zd - mu) / std) ** 2 - 0.5 * lv
                    - 0.5 * np.log(2.0 * np.pi)).sum(1)
            # The KL is analytic, so it is outside the expectation over z and
            # differentiates directly; only the reconstruction term needs the
            # log derivative trick.
            base_run = ll.mean() if base_run is None else 0.9 * base_run + 0.1 * ll.mean()
            gs = torch.autograd.grad((ll * logq).mean() - kl.mean(), params,
                                     retain_graph=True)
            score.append(torch.cat([v.reshape(-1) for v in gs]).detach())
            gsb = torch.autograd.grad(((ll - base_run.detach()) * logq).mean() - kl.mean(),
                                      params)
            score_b.append(torch.cat([v.reshape(-1) for v in gsb]).detach())

        Pm = torch.stack(path)
        Sm = torch.stack(score)
        Bm = torch.stack(score_b)
        out = {"k": k, "params": int(Pm.shape[1]), "draws": draws, "rows": rows}
        for name, M in (("pathwise", Pm), ("score", Sm), ("score_baseline", Bm)):
            out[name] = {"variance": float(M.var(0, unbiased=True).sum()),
                         "mean_norm": float(M.mean(0).norm())}
        # unbiasedness: the two means must agree to within their own error
        se = float((Pm.var(0, unbiased=True).sum() / draws
                    + Sm.var(0, unbiased=True).sum() / draws).sqrt())
        out["mean_gap"] = float((Pm.mean(0) - Sm.mean(0)).norm())
        out["mean_gap_se"] = se
        return out

    return cached(f"vae-estimator-k{k}-s{seed}-d{draws}-r{rows}-v1", build)


est_rows = []
for k in (2, 4, 8, 16):
    e = estimator_variance(k)
    ratio = e["score"]["variance"] / e["pathwise"]["variance"]
    ratio_b = e["score_baseline"]["variance"] / e["pathwise"]["variance"]
    est_rows.append({
        "k": k, "params": e["params"], "draws": e["draws"], "rows": e["rows"],
        "pathwise": sci(e["pathwise"]["variance"]),
        "score": sci(e["score"]["variance"]),
        "score_baseline": sci(e["score_baseline"]["variance"]),
        "ratio": rnd(ratio, 1), "ratio_baseline": rnd(ratio_b, 1),
        "mean_gap": sci(e["mean_gap"]), "mean_gap_se": sci(e["mean_gap_se"]),
        "unbiased_ok": bool(e["mean_gap"] < 3.0 * e["mean_gap_se"]),
    })
    print(f"  k = {k:>2}: total gradient variance {e['pathwise']['variance']:.3e} pathwise "
          f"against {e['score']['variance']:.3e} score function, a factor of {ratio:,.0f} "
          f"({ratio_b:,.0f} with a baseline); the two means differ by "
          f"{e['mean_gap']:.2e} against a standard error of {e['mean_gap_se']:.2e}")

# The same comparison small enough to have a closed form, so the page can run
# the two estimators itself and be checked against algebra rather than against
# another simulation. With q = N(mu, s^2) and f(z) = -(z - c)^2, both estimators
# are unbiased for -2(mu - c), and their variances are 4 s^2 and
# 15 s^2 + 14 a^2 + a^4 / s^2 with a = mu - c.
DEMO = {"mu": 0.0, "sigma": 1.0, "c": 6.0}
_a = DEMO["mu"] - DEMO["c"]
_s = DEMO["sigma"]
DEMO["true_gradient"] = rnd(-2.0 * _a, 6)
DEMO["var_pathwise"] = rnd(4.0 * _s ** 2, 6)
DEMO["var_score"] = rnd(15.0 * _s ** 2 + 14.0 * _a ** 2 + _a ** 4 / _s ** 2, 6)
DEMO["ratio"] = rnd(DEMO["var_score"] / DEMO["var_pathwise"], 3)
_rs = np.random.RandomState(31)
_z = DEMO["mu"] + _s * _rs.randn(400000)
_p = -2.0 * (_z - DEMO["c"])
_q = -((_z - DEMO["c"]) ** 2) * (_z - DEMO["mu"]) / _s ** 2
DEMO["draws"] = int(len(_z))
DEMO["measured_pathwise"] = rnd(float(_p.var(ddof=1)), 5)
DEMO["measured_score"] = rnd(float(_q.var(ddof=1)), 5)
DEMO["measured_mean_pathwise"] = rnd(float(_p.mean()), 5)
DEMO["measured_mean_score"] = rnd(float(_q.mean()), 5)
print(f"  the closed form version the page runs itself: both estimators are unbiased for "
      f"{DEMO['true_gradient']} (measured {DEMO['measured_mean_pathwise']} and "
      f"{DEMO['measured_mean_score']} over {DEMO['draws']:,} draws) and their variances are "
      f"{DEMO['var_pathwise']} against {DEMO['var_score']}, a factor of {DEMO['ratio']} "
      f"(measured {DEMO['measured_pathwise']} and {DEMO['measured_score']})")


# ==================================== 4. rate, distortion, and the units that die
banner("4. the beta sweep: what the second term costs and what dies")
ACTIVE = 0.01                     # nats per dimension, the usual threshold
sweep = []
for beta in BETAS:
    per_seed = []
    for sd in SEEDS:
        a = digit_arm(K_COLLAPSE, beta, sd)
        net = rebuild(K_COLLAPSE, a, seed=sd)
        with torch.no_grad():
            zs = torch.randn(300, K_COLLAPSE, generator=torch.Generator().manual_seed(200 + sd))
            pool = npy(net.decode(zs))
        sc = pool_scores(pool, J, XT, GAMMA)
        per_seed.append({
            "mse": a["mse"], "kl": a["kl"], "elbo": a["elbo"],
            "active": int(sum(v > ACTIVE for v in a["per_dim_kl"])),
            "confidence": sc["confidence"], "nn_distance": sc["nn_distance"],
            "mmd2": sc["mmd2"], "c2st": sc["c2st"],
        })
    agg = {kk: float(np.mean([r[kk] for r in per_seed])) for kk in per_seed[0]}
    spread = {kk: float(np.std([r[kk] for r in per_seed])) for kk in per_seed[0]}
    sweep.append({
        "beta": beta, "mse": rnd(agg["mse"], 6), "kl": rnd(agg["kl"], 4),
        "elbo": rnd(agg["elbo"], 3), "active": rnd(agg["active"], 2),
        "confidence": rnd(agg["confidence"], 4), "nn_distance": rnd(agg["nn_distance"], 4),
        "mmd2": sci(agg["mmd2"]), "c2st": rnd(agg["c2st"], 4),
        "mse_sd": sci(spread["mse"]), "kl_sd": rnd(spread["kl"], 4),
        "confidence_sd": rnd(spread["confidence"], 4),
        "mmd2_sd": sci(spread["mmd2"]), "c2st_sd": rnd(spread["c2st"], 4),
        "active_sd": rnd(spread["active"], 2),
        "per_dim_kl": arr(sorted(digit_arm(K_COLLAPSE, beta, SEED)["per_dim_kl"],
                                 reverse=True), 4),
    })
    print(f"  beta {beta:>5}: error {agg['mse']:.6f}, rate {agg['kl']:7.3f} nats, "
          f"{agg['active']:5.2f} of {K_COLLAPSE} dimensions alive, samples at "
          f"{agg['mmd2']:.3e} from real digits (spread over {len(SEEDS)} seeds: "
          f"{spread['mmd2']:.1e}, floor {FLOOR['mmd2']:.3e})")
best_beta = min(sweep, key=lambda r: r["mmd2"])
print(f"  the closest samples come from beta = {best_beta['beta']} at {best_beta['mmd2']}, "
      f"against {sweep[0]['mmd2']} at beta = 0, which is the previous article's objective "
      f"exactly and the same code with one term removed")


# ============================== 5. the two dimensional truth, where the bound is checkable
banner("5. two dimensions, where the marginal is an integral you can do")
TOY_N, TOY_HOLD = 8000, 2000
TOY_HID, TOY_K = 64, 2
TOY_EPOCHS = 240
Dtr = toy_sample(TOY_N, seed=SEED)
Dte = toy_sample(TOY_HOLD, seed=SEED + 101)
CEILING = -toy_entropy()
print(f"  the truth's own mean log density is {CEILING:.6f} nats, which no fit can beat")


class ToyVAE(nn.Module):
    def __init__(self, h=TOY_HID, k=TOY_K, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.k = k
        self.enc = nn.Sequential(nn.Linear(2, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.emu = nn.Linear(h, k)
        self.elv = nn.Linear(h, k)
        self.dec = nn.Sequential(nn.Linear(k, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, 2))
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))

    def encode(self, x):
        h = self.enc(x)
        return self.emu(h), self.elv(h).clamp(-9.0, 4.0)


def toy_terms(net, x, generator=None):
    mu, lv = net.encode(x)
    eps = torch.randn(mu.shape, generator=generator)
    z = mu + torch.exp(0.5 * lv) * eps
    xh = net.dec(z)
    s = torch.exp(net.log_sigma)
    ll = (-0.5 * ((x - xh) / s) ** 2 - net.log_sigma - 0.5 * np.log(2 * np.pi)).sum(1)
    kl = 0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv).sum(1)
    return ll, kl


def fit_toy(seed=SEED):
    def build():
        net = ToyVAE(seed=seed)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        Xt = torch.tensor(Dtr, dtype=torch.float32)
        g = torch.Generator().manual_seed(seed)
        for _ in range(TOY_EPOCHS):
            order = torch.randperm(len(Xt), generator=g)
            for s in range(0, len(Xt), 256):
                ll, kl = toy_terms(net, Xt[order[s:s + 256]], generator=g)
                loss = -(ll - kl).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {kk: v.detach().clone() for kk, v in net.state_dict().items()}

    key = (f"vae-toy-h{TOY_HID}-k{TOY_K}-e{TOY_EPOCHS}-n{TOY_N}-s{seed}"
           f"-r{TOY.get('ring', {}).get('r0')}-v1")
    st = cached(key, build)
    net = ToyVAE(seed=seed)
    net.load_state_dict(st)
    return net


toy_net = fit_toy()
with torch.no_grad():
    SIG = float(np.exp(npy(toy_net.log_sigma)))
    Xte_t = torch.tensor(Dte, dtype=torch.float32)
    g_eval = torch.Generator().manual_seed(555)
    ll, kl = toy_terms(toy_net, Xte_t, generator=g_eval)
    ELBO = float(npy(ll - kl).mean())
print(f"  the fitted decoder's noise scale is {SIG:.5f}")
print(f"  its one sample bound on the held out set is {ELBO:.5f} nats")


def toy_decode(Z, chunk=200000):
    out = np.empty((len(Z), 2))
    with torch.no_grad():
        for s in range(0, len(Z), chunk):
            out[s:s + chunk] = npy(toy_net.dec(torch.tensor(Z[s:s + chunk],
                                                            dtype=torch.float32)))
    return out


def exact_logp_local(Xq, m=129, span=12.0, chunk=48):
    """log p(x) by quadrature over the latent, one adaptive grid per row.

    p(x) is an integral over a two dimensional latent, so it is a sum. The
    integrand is the prior times the decoder's likelihood, and it is sharply
    peaked wherever the decoder is steep, so a global grid would under-resolve
    it. The encoder already knows where the peak is, so each row gets its own
    grid centred on the encoder's mean and `span` posterior standard deviations
    wide. Whether that box is wide enough is not assumed: the mass on its
    boundary is returned and checked.
    """
    with torch.no_grad():
        mu, lv = toy_net.encode(torch.tensor(Xq, dtype=torch.float32))
    mu, sd = npy(mu), np.exp(0.5 * npy(lv))
    sd = np.maximum(sd, 1e-3)
    t = (np.arange(m) + 0.5) / m * 2.0 - 1.0            # midpoints of [-1, 1]
    out = np.empty(len(Xq))
    edge = np.zeros(len(Xq))
    for s in range(0, len(Xq), chunk):
        sl = slice(s, min(s + chunk, len(Xq)))
        b = mu[sl].shape[0]
        z0 = mu[sl][:, None, None, 0] + span * sd[sl][:, None, None, 0] * t[None, :, None]
        z1 = mu[sl][:, None, None, 1] + span * sd[sl][:, None, None, 1] * t[None, None, :]
        Z = np.stack([np.broadcast_to(z0, (b, m, m)),
                      np.broadcast_to(z1, (b, m, m))], -1).reshape(-1, 2)
        G = toy_decode(Z).reshape(b, m, m, 2)
        lp = (-0.5 * (Z ** 2).sum(1) - np.log(2 * np.pi)).reshape(b, m, m)
        d2 = ((Xq[sl][:, None, None, :] - G) ** 2).sum(-1)
        lk = -0.5 * d2 / SIG ** 2 - 2.0 * np.log(SIG) - np.log(2 * np.pi)
        cell = (2.0 * span * sd[sl][:, 0] / m) * (2.0 * span * sd[sl][:, 1] / m)
        w = lp + lk
        mx = w.reshape(b, -1).max(1)
        e = np.exp(w - mx[:, None, None])
        tot = e.reshape(b, -1).sum(1)
        out[sl] = mx + np.log(tot) + np.log(cell)
        rim = (e[:, 0, :].sum(1) + e[:, -1, :].sum(1)
               + e[:, :, 0].sum(1) + e[:, :, -1].sum(1))
        edge[sl] = rim / tot
    return out, edge


def exact_logp_grid(m=1024, mz=2400, half=None):
    """The same density on the whole plane, by a different route.

    p is the pushforward of the prior through the decoder, blurred by the
    decoder's own gaussian noise. So: splat the prior's mass onto a grid at the
    decoded positions with bilinear weights, then convolve with the gaussian by
    FFT. That gives log p everywhere at once, which is what the exact KL needs,
    and it shares no code with the per row quadrature above, so the two
    agreeing is a real check.
    """
    half = TOY["domain"] if half is None else half
    e = np.linspace(-half, half, m + 1)
    c = 0.5 * (e[1:] + e[:-1])
    h = float(c[1] - c[0])
    zt = (np.arange(mz) + 0.5) / mz * 12.0 - 6.0        # prior grid, +-6 sd
    zx, zy = np.meshgrid(zt, zt, indexing="ij")
    Zg = np.c_[zx.ravel(), zy.ravel()]
    w = np.exp(-0.5 * (Zg ** 2).sum(1)) / (2 * np.pi) * (12.0 / mz) ** 2
    G = toy_decode(Zg)
    field = np.zeros((m, m))
    fx = (G[:, 0] - c[0]) / h
    fy = (G[:, 1] - c[0]) / h
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    ax, ay = fx - i0, fy - j0
    keep = (i0 >= 0) & (i0 < m - 1) & (j0 >= 0) & (j0 < m - 1)
    ik, jk = i0[keep], j0[keep]
    for di, dj, wt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)),
                       (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        field += np.bincount((ik + di) * m + (jk + dj), weights=(w * wt)[keep],
                             minlength=m * m).reshape(m, m)
    field /= h * h                                       # mass to density
    # Splatting a point mass bilinearly onto a grid of spacing h is itself a
    # blur: the two weights put variance a(1-a)h^2 on each axis, which averages
    # to h^2/6 over the offsets. Convolving afterwards with the decoder's own
    # sigma would therefore land at sigma^2 + h^2/6 rather than sigma^2, which
    # at this grid is a 0.13% widening and about 0.003 nats in the log density,
    # which is right where the guard against the other quadrature sits. So the
    # kernel is narrowed by exactly that much and the guard measures what is
    # left.
    var = SIG ** 2 - h * h / 6.0
    assert var > 0, f"the grid at {h:.5f} is too coarse for a decoder noise of {SIG:.5f}"
    # fftfreq already puts the offsets in wrapped order (zero first, then the
    # positive ones, then the negative ones), so the kernel is centred where the
    # transform expects it. Calling ifftshift on top of that translates the
    # density by half the grid and wraps it, which the guard against the other
    # quadrature caught as a disagreement of 688 nats.
    off = np.fft.fftfreq(m, d=1.0 / m) * h
    k1 = np.exp(-0.5 * off ** 2 / var)
    k1 = k1 / k1.sum() / h
    K = np.outer(k1, k1)
    dens = np.real(np.fft.ifft2(np.fft.fft2(field) * np.fft.fft2(K))) * h * h
    return np.maximum(dens, 1e-300), c, float(w.sum()), float((1.0 - keep.mean()))


exact_te, edge_te = exact_logp_local(Dte[:N_EXACT])
dens_grid, gc, prior_mass, spilled = exact_logp_grid()
# Reading the grid at the nearest cell would compare the two methods through
# half a cell of position error, which at this density is worth about 0.06 nats
# and is exactly what the guard was reporting before. Interpolating the log
# density bilinearly takes that out and leaves the guard measuring the two
# quadratures against each other.
step = float(gc[1] - gc[0])
fu = np.clip((Dte[:N_EXACT, 0] - gc[0]) / step, 0, len(gc) - 1.001)
fv = np.clip((Dte[:N_EXACT, 1] - gc[0]) / step, 0, len(gc) - 1.001)
iu, iv = fu.astype(int), fv.astype(int)
au, av = fu - iu, fv - iv
lg = np.log(dens_grid)
interp = ((1 - au) * (1 - av) * lg[iu, iv] + au * (1 - av) * lg[iu + 1, iv]
          + (1 - au) * av * lg[iu, iv + 1] + au * av * lg[iu + 1, iv + 1])
cross = float(np.abs(interp - exact_te).max())
print(f"  two independent quadratures for the same log density on {N_EXACT} rows agree to "
      f"{cross:.3e}; the per row boxes leak {edge_te.max():.2e} of their mass at the rim and "
      f"the splat keeps {1 - spilled:.6f} of the prior")

Xg, cell, shape = toy_grid(len(gc))
log_q_grid = np.log(dens_grid).ravel()
model_mass = float(dens_grid.sum() * cell)
KL_TOY = toy_kl(log_q_grid, m=len(gc))


def iwae(Xq, K, seed=777, chunk=250):
    out = np.empty(len(Xq))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for s in range(0, len(Xq), chunk):
            xb = torch.tensor(Xq[s:s + chunk], dtype=torch.float32)
            mu, lv = toy_net.encode(xb)
            sd = torch.exp(0.5 * lv)
            eps = torch.randn((K, xb.shape[0], toy_net.k), generator=g)
            z = mu[None] + sd[None] * eps
            xh = toy_net.dec(z.reshape(-1, toy_net.k)).reshape(K, xb.shape[0], 2)
            sg = torch.exp(toy_net.log_sigma)
            ll = (-0.5 * ((xb[None] - xh) / sg) ** 2 - toy_net.log_sigma
                  - 0.5 * np.log(2 * np.pi)).sum(-1)
            lpz = (-0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)).sum(-1)
            lqz = (-0.5 * eps ** 2 - 0.5 * lv[None] - 0.5 * np.log(2 * np.pi)).sum(-1)
            w = ll + lpz - lqz
            out[s:s + chunk] = npy(torch.logsumexp(w, 0) - np.log(K))
    return out


exact_mean = float(exact_te.mean())
bound_rows = []
for K in (1, 2, 8, 64, 512, 4096):
    v = float(iwae(Dte[:N_EXACT], K).mean())
    bound_rows.append({"k": K, "bound": rnd(v, 5), "gap": rnd(exact_mean - v, 5)})
    print(f"  {K:>5} importance samples: {v:.5f} nats, still {exact_mean - v:.5f} below the "
          f"exact {exact_mean:.5f}")
"""The two numbers that look like they should be equal are not, and the
reason is which one is a sample and which one is an integral. The mean log
density over the 400 held out rows is a Monte Carlo estimate with a standard
error; the expected log density under the truth is the quadrature, exact.
Both are exported, labelled, and the gap against the bound uses the sample
one because the bound is a sample mean over the same rows."""
exact_se = float(exact_te.std(ddof=1) / np.sqrt(len(exact_te)))
toy_block = {
    "n_train": TOY_N, "n_test": TOY_HOLD, "hidden": TOY_HID, "k": TOY_K,
    "epochs": TOY_EPOCHS, "sigma": rnd(SIG, 5), "rows": N_EXACT,
    "ceiling": rnd(CEILING, 6),
    "elbo": rnd(ELBO, 5),
    "exact": rnd(exact_mean, 5),
    "exact_se": rnd(exact_se, 5),
    "exact_grid": rnd(CEILING - KL_TOY, 5),
    "gap": rnd(exact_mean - ELBO, 5),
    "kl_to_truth": rnd(KL_TOY, 5),
    "kl_from_ceiling": rnd(CEILING - exact_mean, 5),
    "quadrature": toy_quadrature_report(),
    "bounds": bound_rows,
    "grid": len(gc),
    "guards": {
        "quadrature_gap": sci(cross),
        "edge_mass": sci(float(edge_te.max())),
        "model_mass": float(f"{model_mass:.10f}"),
        "prior_mass": float(f"{prior_mass:.10f}"),
        "splat_spill": sci(spilled),
    },
}
print(f"  the exact held out log density is {exact_mean:.5f} against a bound of {ELBO:.5f}: "
      f"the bound is loose by {exact_mean - ELBO:.5f} nats")
print(f"  the model integrates to {model_mass:.10f} on the same grid the truth integrates to "
      f"one on, and its exact KL from the truth is {KL_TOY:.5f} nats")
print(f"  so its expected log density under the truth is exactly {CEILING - KL_TOY:.5f}, which "
      f"the {N_EXACT} held out rows estimate as {exact_mean:.5f} plus or minus {exact_se:.5f}")


# ================================================= 6. the weights the page runs
banner("6. exporting the pair the browser will run")
spec, b64, v16 = export_half([
    ("enc1", npy(net2.e1.weight).T, npy(net2.e1.bias)),
    ("emu", npy(net2.emu.weight).T, npy(net2.emu.bias)),
    ("elv", npy(net2.elv.weight).T, npy(net2.elv.bias)),
    ("dec1", npy(net2.d1.weight).T, npy(net2.d1.bias)),
    ("dec2", npy(net2.d2.weight).T, npy(net2.d2.bias)),
])
w16 = v16.astype(np.float64)


def q_encode(Xq):
    h = np.tanh(dense(w16, spec[0], Xq))
    return dense(w16, spec[1], h), np.clip(dense(w16, spec[2], h), -9, 4)


def q_decode(Z):
    return 1.0 / (1.0 + np.exp(-dense(w16, spec[4], np.tanh(dense(w16, spec[3], Z)))))


mu_q, lv_q = q_encode(XV)
quant = {
    "code_shift": sci(float(np.abs(mu_q - base["mu_test"]).max())),
    "mse_full": rnd(base["mse"], 6),
    "mse_quantised": rnd(float(((q_decode(mu_q) - XV) ** 2).mean()), 6),
}
print(f"  {len(v16):,} numbers, float16 base64 is {len(b64) / 1024:.0f} kB; quantising moves "
      f"the codes by at most {quant['code_shift']} and the held out error from "
      f"{quant['mse_full']} to {quant['mse_quantised']}")

# Eight samples from every beta in the sweep, in float16 rather than as JSON
# decimals: the strip is 9 by 8 images of 196 pixels and writing them as text
# would be a quarter of the file.
STRIP = 8
strip_vals = []
for b in BETAS:
    nb = rebuild(K_COLLAPSE, digit_arm(K_COLLAPSE, b, SEED))
    with torch.no_grad():
        zz = torch.randn(STRIP, K_COLLAPSE, generator=torch.Generator().manual_seed(909))
        strip_vals.append(npy(nb.decode(zz)))
strips_b64, strips16 = export_flat(np.concatenate(strip_vals))

# The two dimensional pictures. The model density was already computed on the
# whole quadrature square, so the drawing box is a slice of it rather than a
# second calculation that could disagree with the first.
VIEW = TOY["view"]
inside = np.where(np.abs(gc) <= VIEW)[0]
stride = max(1, len(inside) // 150)
sel = inside[::stride]
truth_sub = toy_pdf(np.c_[np.repeat(gc[sel], len(sel)), np.tile(gc[sel], len(sel))])
model_sub = dens_grid[np.ix_(sel, sel)].ravel()
truth_b64, _ = export_flat(truth_sub)
model_b64, _ = export_flat(model_sub)
toy_block["picture"] = {
    "n": int(len(sel)), "lo": rnd(float(gc[sel][0]), 4), "hi": rnd(float(gc[sel][-1]), 4),
    "truth_b64": truth_b64, "model_b64": model_b64,
    "peak": rnd(float(max(truth_sub.max(), model_sub.max())), 4),
    "sample": mat(Dte[:700], 3),
    "hole": {
        "radius": 1.2,
        "truth": sci(float((toy_pdf(np.c_[np.repeat(gc, len(gc)), np.tile(gc, len(gc))])
                            * (np.hypot(np.repeat(gc, len(gc)), np.tile(gc, len(gc))) < 1.2)
                            ).sum() * cell)),
        "model": sci(float((dens_grid.ravel()
                            * (np.hypot(np.repeat(gc, len(gc)), np.tile(gc, len(gc))) < 1.2)
                            ).sum() * cell)),
    },
}
print(f"  inside the hole of the ring the truth puts {toy_block['picture']['hole']['truth']} "
      f"and this fit puts {toy_block['picture']['hole']['model']}")

SHOW = [int(np.where(y == c)[0][0]) for c in range(10)]
payload = {
    "meta": {
        "seed": SEED, "threads": THREADS, "torch": torch.__version__,
        "n": DIGITS["n"], "side": DIGITS["side"], "pixels": P, "hidden": HID,
        "train": int(len(TR)), "test": int(len(TE)),
        "k_show": K_SHOW, "k_collapse": K_COLLAPSE, "epochs": EPOCHS,
        "batch": BATCH,
        "betas": BETAS, "seeds": SEEDS, "active_threshold": ACTIVE,
        "judge_accuracy": rnd(J["accuracy"], 4),
        "judge_confidence": rnd(J["real_confidence"], 4),
        "judge_hidden": J["hidden"],
        "reuse": {
            "note": "the digits are the autoencoders article's, asserted by running that "
                    "article's own float16 encoder on the ones rebuilt here",
            "worst_code": DIGITS["assert_worst"],
            "floor": DIGITS["assert_floor"],
            "ae_count": AE["net"]["count"],
        },
    },
    "hole": hole,
    "compare": compare,
    "estimators": {"rows": est_rows, "demo": DEMO},
    "sweep": {"rows": sweep, "betas": BETAS, "strip": STRIP,
              "strips_b64": strips_b64, "strip_count": int(len(strips16))},
    "toy": toy_block,
    "net": {
        "format": "float16 little-endian, base64; each layer row major over output units",
        "layers": spec, "count": int(len(v16)), "weights_b64": b64, "quant": quant,
        "mu": mat(np.r_[base["mu_train"], base["mu_test"]][np.argsort(np.r_[TR, TE])], 4),
        "lv": mat(np.r_[base["lv_train"], base["lv_test"]][np.argsort(np.r_[TR, TE])], 4),
        "labels": [int(v) for v in y],
        "is_test": [int(i in set(TE.tolist())) for i in range(len(y))],
        "ae_latent": mat(AE_Z, 4),
        "showcase": [{"label": int(y[i]), "index": int(i), "pixels": arr(X[i], 4)}
                     for i in SHOW],
    },
}
write_json(OUT / "vae.json", payload)
