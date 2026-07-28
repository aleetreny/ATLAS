"""Data for the ATLAS 'Where You Weren't' article: NeRF and Gaussian splatting.

Every other article in module 1.3 works on a flat image. This one works on a
scene, and the only thing either method is ever shown is a set of photographs
of it. So the scene here is defined analytically, in closed form, and the
photographs are rendered from that definition: which means the ground truth is
not an annotation anyone drew, it is the function itself, and every error on
this page is measured against it rather than against another model's opinion.

Five blocks:

0. The scene and its photographs. Four soft solids (a sphere, a torus, a
   rotated cube, a floor slab) written as signed distance functions with a
   compactly supported shell, so the density is exactly zero outside them and
   nothing depends on where a bounding box was drawn. Colour is Blinn-Phong
   with a real specular lobe, which is what makes the emitted radiance depend
   on the viewing direction, which is the part of the rendering equation that
   most explanations of it skip. Cameras sit on three rings; the test cameras
   sit between the rings, and six more sit outside the photographed band
   entirely, which is the only honest way to ask what a fitted scene knows
   about somewhere nobody stood.

1. Quadrature. A pixel is an integral, and a renderer is a quadrature rule for
   it. Before any learning happens, the same rendering equation is evaluated
   at many step counts against a reference at 1024 steps, so the article can
   say what discretising the integral costs before it says what fitting it
   costs.

2. NeRF. A small MLP fit to the photographs by gradient descent through the
   volume rendering integral. The ablations are the ones that decide whether
   the thing works at all: the number of Fourier bands in the positional
   encoding (with a parameter-matched control, because more bands is also more
   weights), the view direction input, and the number of photographs.

3. Gaussian splatting. The same rendering equation, the same photographs, the
   same loss, and a completely different answer to "what is a scene made of":
   a pile of explicit anisotropic blobs, projected and sorted and alpha
   composited. Ablations: anisotropy (at equal count AND at equal parameter
   count, because a full covariance is six numbers where a sphere is one), the
   view-dependent colour term, and how many blobs.

4. Cost, in exact arithmetic. Multiply-accumulates per pixel for both, derived
   from the architectures rather than timed, so the number means the same
   thing on every machine. The wall clock lives in the reader's browser, where
   both renderers run for real.

Guards, all of which can fire:

* Every analytic gradient in this file is checked against central finite
  differences, and the splatting forward pass is written twice, once in numpy
  and once in torch, with an assertion that they agree to 1e-9. The numpy one
  is the reference the browser has to reproduce.
* The browser has to rebuild the ground truth scene from the same closed form,
  so a handful of exact pixel values travel in the JSON for it to check itself
  against.
* Empty space skipping is an optimisation, so the error it introduces is
  measured and exported rather than assumed to be zero.

Run from anywhere: `python src/utils/generate_nerf_data.py`. Stages cache into
~/.atlas_vision_data/nerf_cache/, keyed by their full configuration, so an
interrupted run costs nothing.
"""
import base64
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "nerf" / "data"
IMG = ROOT / "nerf" / "img"
CACHE = Path.home() / ".atlas_vision_data" / "nerf_cache"
for d in (OUT, IMG, CACHE):
    d.mkdir(parents=True, exist_ok=True)

# A small run for smoke tests: same code path, tiny sizes.
SMALL = os.environ.get("ATLAS_NERF_SMALL") == "1"

rnd = lambda v, n=4: round(float(v), n)


# ----------------------------------------------------------------------------
# 0. The scene
# ----------------------------------------------------------------------------
# Everything below is the definition of the world. The browser reimplements it
# from the same constants, so anything that moves here has to move there.

RES = 32 if SMALL else 64          # photographs are RES x RES
FOV_DEG = 44.0
CAM_RADIUS = 3.2
TARGET = np.array([0.0, -0.05, 0.0])
BG = np.array([0.945, 0.953, 0.953])   # the page's own paper colour
LIGHT = np.array([0.40, 0.82, 0.41])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
SHELL = 0.045          # half-width of the soft boundary, in world units
SIGMA_MAX = 55.0       # density inside a solid
AMBIENT = 0.32
DIFFUSE = 0.78

# The axis aligned box every ray is clipped to. Tight around the solids plus
# their shell, so no sample is ever spent on provably empty space.
AABB_LO = np.array([-0.95, -0.69, -0.92])
AABB_HI = np.array([0.98, 0.50, 0.92])


def _rot(ax, ay, az):
    """R = Rz Ry Rx, degrees. Written out so the browser can copy it."""
    a, b, c = np.radians([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


SHAPES = [
    dict(name="sphere", kind="sphere", c=[-0.52, -0.10, 0.02], r=0.34,
         rot=[0, 0, 0], albedo=[0.86, 0.25, 0.21], ks=0.55, shine=48.0),
    dict(name="torus", kind="torus", c=[0.46, 0.00, -0.26], R=0.31, r=0.115,
         rot=[74, 0, 22], albedo=[0.08, 0.52, 0.58], ks=0.22, shine=20.0),
    dict(name="cube", kind="box", c=[0.06, -0.15, 0.55], h=[0.19, 0.19, 0.19],
         rot=[16, 34, 8], albedo=[0.93, 0.72, 0.15], ks=0.09, shine=12.0),
    dict(name="floor", kind="box", c=[0.0, -0.55, 0.0], h=[0.82, 0.05, 0.82],
         rot=[0, 0, 0], albedo=[0.70, 0.71, 0.77], ks=0.03, shine=8.0),
]
for s in SHAPES:
    s["Rm"] = _rot(*s["rot"])


def sdf_one(shape, p):
    """Signed distance to one solid. p is (..., 3)."""
    q = (p - np.asarray(shape["c"])) @ shape["Rm"]     # R^T (p - c)
    if shape["kind"] == "sphere":
        return np.linalg.norm(q, axis=-1) - shape["r"]
    if shape["kind"] == "torus":
        # ring in the xz plane, axis along y
        a = np.hypot(q[..., 0], q[..., 2]) - shape["R"]
        return np.hypot(a, q[..., 1]) - shape["r"]
    # box
    d = np.abs(q) - np.asarray(shape["h"])
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


def shell(s):
    """Occupancy of a solid at signed distance s: 1 inside, 0 outside, and a
    smoothstep across a shell of half-width SHELL. Compact support, so density
    is exactly zero outside the solids and no cutoff has to be argued for."""
    t = np.clip((SHELL - s) / (2.0 * SHELL), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def scene_density(p):
    """(sigma, occupancy per shape) at points p (..., 3)."""
    a = np.stack([shell(sdf_one(s, p)) for s in SHAPES], axis=-1)
    return SIGMA_MAX * a.max(axis=-1), a


NORMAL_EPS = 1.5e-3


def _normal(shape, p):
    """Central differences of one solid's SDF. The same epsilon is written into
    the JSON so the browser gets the same normals, not merely similar ones."""
    n = np.empty(p.shape, dtype=np.float64)
    for k in range(3):
        d = np.zeros(3)
        d[k] = NORMAL_EPS
        n[..., k] = sdf_one(shape, p + d) - sdf_one(shape, p - d)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def scene_colour(p, view, occ, spec=True):
    """Emitted radiance at p seen from direction `view` (unit, pointing from the
    camera into the scene). Blinn-Phong: the specular lobe is the whole reason
    this scene's radiance depends on where you stand."""
    out = np.zeros(p.shape, dtype=np.float64)
    owner = occ.argmax(axis=-1)
    for i, s in enumerate(SHAPES):
        m = (owner == i) & (occ[..., i] > 0.0)
        if not m.any():
            continue
        n = _normal(s, p[m])
        v = -view[m] if view.ndim == p.ndim else -view
        lam = np.maximum((n * LIGHT).sum(-1), 0.0)[..., None]
        h = LIGHT + v
        h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        lobe = np.maximum((n * h).sum(-1), 0.0)[..., None] ** s["shine"]
        out[m] = np.asarray(s["albedo"]) * (AMBIENT + DIFFUSE * lam) + (s["ks"] if spec else 0.0) * lobe
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------

def look_at(pos, target=TARGET):
    """Camera basis (right, down, forward) as rows. Down rather than up, so a
    pixel's row index and its y offset run the same way and the projection used
    by the splatting rasteriser needs no sign flips."""
    f = target - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


def camera(az_deg, el_deg, radius=CAM_RADIUS):
    az, el = math.radians(az_deg), math.radians(el_deg)
    pos = np.array([radius * math.cos(el) * math.sin(az),
                    radius * math.sin(el),
                    radius * math.cos(el) * math.cos(az)]) + TARGET
    return dict(az=az_deg, el=el_deg, pos=pos, basis=look_at(pos))


def rays(cam, res=RES):
    """Origins (one) and unit directions (res*res, 3), row-major from the top
    left, matching the order the browser writes pixels in."""
    f = 0.5 * res / math.tan(math.radians(FOV_DEG) / 2.0)
    j, i = np.meshgrid(np.arange(res), np.arange(res))
    x = (j + 0.5 - 0.5 * res) / f
    y = (i + 0.5 - 0.5 * res) / f
    r, d, fw = cam["basis"]
    dirs = fw + x[..., None] * r + y[..., None] * d
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    return cam["pos"], dirs.reshape(-1, 3), f


def ray_box(o, d, lo=AABB_LO, hi=AABB_HI):
    """Slab test. Returns (t_near, t_far, hit)."""
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t0 = (lo - o) * inv
    t1 = (hi - o) * inv
    tn = np.minimum(t0, t1).max(axis=-1)
    tf = np.maximum(t0, t1).min(axis=-1)
    tn = np.maximum(tn, 0.0)
    return tn, tf, tf > tn


# ----------------------------------------------------------------------------
# The rendering equation, on the true scene
# ----------------------------------------------------------------------------

def render_truth(cam, res=RES, steps=256, chunk=4096, spec=True):
    """Volume render the analytic scene. Midpoint quadrature on `steps` equal
    intervals between where the ray enters and leaves the bounding box."""
    o, dirs, _ = rays(cam, res)
    out = np.tile(BG, (dirs.shape[0], 1))
    alpha = np.zeros(dirs.shape[0])
    depth = np.zeros(dirs.shape[0])
    tn, tf, hit = ray_box(o, dirs)
    idx = np.flatnonzero(hit)
    for s in range(0, idx.size, chunk):
        sel = idx[s:s + chunk]
        d = dirs[sel]
        lo, hi = tn[sel], tf[sel]
        dt = (hi - lo) / steps
        u = (np.arange(steps) + 0.5)[None, :]
        t = lo[:, None] + u * dt[:, None]
        p = o + t[..., None] * d[:, None, :]
        sig, occ = scene_density(p)
        col = scene_colour(p, np.broadcast_to(d[:, None, :], p.shape), occ, spec=spec)
        a = 1.0 - np.exp(-sig * dt[:, None])
        T = np.cumprod(np.concatenate([np.ones((sel.size, 1)), 1.0 - a + 1e-12], axis=1), axis=1)[:, :-1]
        w = T * a
        out[sel] = (w[..., None] * col).sum(1) + (1.0 - w.sum(1))[:, None] * BG
        alpha[sel] = w.sum(1)
        depth[sel] = (w * t).sum(1)
    return (out.reshape(res, res, 3), alpha.reshape(res, res), depth.reshape(res, res))


def psnr(a, b):
    mse = float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


# ----------------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------------

def cached(key, fn):
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = CACHE / f"{h}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    val = fn()
    with open(path, "wb") as fh:
        pickle.dump(val, fh)
    print(f"    [{key}] computed in {time.time() - t0:.1f}s", flush=True)
    return val


# ----------------------------------------------------------------------------
# The camera rig
# ----------------------------------------------------------------------------
RING_EL = [10.0, 28.0, 46.0]
RING_N = 8
TEST_EL = [19.0, 37.0]
OUTSIDE_EL = [-14.0, 68.0]


def build_cameras():
    train = []
    for k, el in enumerate(RING_EL):
        for i in range(RING_N):
            az = 360.0 * i / RING_N + (15.0 * k)
            train.append(camera(az, el))
    inside = []
    for k, el in enumerate(TEST_EL):
        for i in range(4):
            az = 360.0 * i / 4 + 22.5 + 30.0 * k
            inside.append(camera(az, el))
    outside = []
    for k, el in enumerate(OUTSIDE_EL):
        for i in range(3):
            az = 360.0 * i / 3 + 40.0 * k
            outside.append(camera(az, el))
    return train, inside, outside


# Subsets used by the "how many photographs" ablation: the same three
# elevations every time, so the comparison is about density and not coverage.
VIEW_SUBSETS = {
    6: [0, 4, 8, 12, 16, 20],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    24: list(range(24)),
}


# ----------------------------------------------------------------------------
# 2. NeRF
# ----------------------------------------------------------------------------
import torch                                                   # noqa: E402
import torch.nn as nn                                          # noqa: E402
import torch.nn.functional as F                                # noqa: E402

# Never a bare number: oversubscribing is not merely slow, it changes the order
# of the floating point reductions and therefore the result.
THREADS = min(int(os.environ.get("ATLAS_NERF_THREADS", "4")), os.cpu_count() or 4)
torch.set_num_threads(THREADS)

CENTER = (AABB_LO + AABB_HI) / 2.0
HALF = (AABB_HI - AABB_LO) / 2.0
SIG_SCALE = 25.0        # sigma = SIG_SCALE * softplus(raw); the true field peaks at 55


def encode(x, L):
    """gamma(x) = [x, sin(2^0 pi x), cos(2^0 pi x), ...]. The raw input is kept,
    which is what the reference implementation does and what makes L = 0 an
    ordinary coordinate MLP rather than an empty tensor."""
    if L == 0:
        return x
    out = [x]
    for i in range(L):
        f = (2.0 ** i) * math.pi
        out.append(torch.sin(f * x))
        out.append(torch.cos(f * x))
    return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self, Lx=6, Ld=2, W=64, D=4, viewdep=True):
        super().__init__()
        self.Lx, self.Ld, self.W, self.D, self.viewdep = Lx, Ld, W, D, viewdep
        in_x = 3 + 6 * Lx
        self.trunk = nn.ModuleList([nn.Linear(in_x, W)] + [nn.Linear(W, W) for _ in range(D - 1)])
        self.sigma = nn.Linear(W, 1)
        self.feat = nn.Linear(W, W // 2)
        in_c = W // 2 + (3 + 6 * Ld if viewdep else 0)
        self.col1 = nn.Linear(in_c, W // 2)
        self.col2 = nn.Linear(W // 2, 3)

    def forward(self, p, d):
        """p: (..., 3) world points, d: (..., 3) unit view directions."""
        c0 = torch.as_tensor(CENTER, dtype=p.dtype)
        hf = torch.as_tensor(HALF, dtype=p.dtype)
        h = encode((p - c0) / hf, self.Lx)
        for lin in self.trunk:
            h = F.relu(lin(h))
        sig = SIG_SCALE * F.softplus(self.sigma(h)[..., 0])
        f = self.feat(h)
        c = torch.cat([f, encode(d, self.Ld)], dim=-1) if self.viewdep else f
        c = F.relu(self.col1(c))
        return sig, torch.sigmoid(self.col2(c))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def volume_render(model, o, d, tn, tf, steps, jitter=None, chunk=None):
    """Composite the model's field along rays. o (N,3), d (N,3), tn/tf (N,)."""
    N = d.shape[0]
    dt_ = d.dtype
    u = (torch.arange(steps, dtype=dt_) + 0.5) / steps
    u = u.expand(N, steps).clone()
    if jitter is not None:
        u = u + (jitter.to(dt_) - 0.5) / steps
    dt = ((tf - tn) / steps)[:, None]
    t = tn[:, None] + u * (tf - tn)[:, None]
    p = o[:, None, :] + t[..., None] * d[:, None, :]
    dd = d[:, None, :].expand(N, steps, 3)
    sig, col = model(p, dd)
    a = 1.0 - torch.exp(-sig * dt)
    T = torch.cumprod(torch.cat([torch.ones(N, 1, dtype=dt_), 1.0 - a + 1e-10], 1), 1)[:, :-1]
    w = T * a
    rgb = (w[..., None] * col).sum(1) + (1.0 - w.sum(1))[:, None] * torch.as_tensor(BG, dtype=dt_)
    return rgb, w, t


def _train_rays(cams, images):
    """Flatten a set of photographs into (origins, dirs, tn, tf, target) for
    every pixel whose ray meets the scene box. A ray that misses it renders the
    background exactly, so it carries no gradient and is not stored; the bound
    is prior knowledge every implementation of this uses, and the reported PSNR
    is still over the whole frame."""
    O, D, TN, TF, C = [], [], [], [], []
    for cam, im in zip(cams, images):
        o, dirs, _ = rays(cam, RES)
        tn, tf, hit = ray_box(o, dirs)
        sel = np.flatnonzero(hit)
        O.append(np.tile(o, (sel.size, 1)))
        D.append(dirs[sel])
        TN.append(tn[sel])
        TF.append(tf[sel])
        C.append(im.reshape(-1, 3)[sel])
    return [torch.as_tensor(np.concatenate(v)) for v in (O, D, TN, TF, C)]


def render_model(model, cam, steps, res=RES, chunk=2048, grid=None):
    """Deterministic midpoint render of a fitted field, whole frame."""
    dt = next(model.parameters()).dtype
    o, dirs, _ = rays(cam, res)
    out = np.tile(BG, (dirs.shape[0], 1))
    tn, tf, hit = ray_box(o, dirs)
    idx = np.flatnonzero(hit)
    with torch.no_grad():
        for s in range(0, idx.size, chunk):
            sel = idx[s:s + chunk]
            rgb, _, _ = volume_render(
                model,
                torch.as_tensor(np.tile(o, (sel.size, 1)), dtype=dt),
                torch.as_tensor(dirs[sel], dtype=dt),
                torch.as_tensor(tn[sel], dtype=dt),
                torch.as_tensor(tf[sel], dtype=dt),
                steps)
            out[sel] = rgb.numpy()
    return out.reshape(res, res, 3)


def train_nerf(cams, images, *, Lx=6, Ld=2, W=64, D=4, viewdep=True,
               iters=3000, batch=1024, steps=48, seed=19, lr=5e-3, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NeRF(Lx=Lx, Ld=Ld, W=W, D=D, viewdep=viewdep).float()
    O, Dr, TN, TF, C = [t.float() for t in _train_rays(cams, images)]
    n = O.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(0.05) ** (1.0 / iters))
    g = torch.Generator().manual_seed(seed)
    hist = []
    t0 = time.time()
    for it in range(iters):
        idx = torch.randint(0, n, (batch,), generator=g)
        jit = torch.rand(batch, steps, generator=g)
        rgb, _, _ = volume_render(model, O[idx], Dr[idx], TN[idx], TF[idx], steps, jitter=jit)
        loss = ((rgb - C[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == iters - 1:
            hist.append((it, round(float(loss.detach()), 6)))
        if log and (it % log == 0 or it == iters - 1):
            print(f"      it {it:5d}  loss {float(loss.detach()):.5f}  "
                  f"psnr {-10 * math.log10(float(loss.detach())):.2f}  {time.time() - t0:.0f}s", flush=True)
    return model, hist, time.time() - t0


def match_width(target_params, Lx, Ld, D, viewdep):
    """The width that puts an encoding-free MLP on the same parameter budget."""
    best, bw = None, None
    for W in range(8, 400, 2):
        p = NeRF(Lx=Lx, Ld=Ld, W=W, D=D, viewdep=viewdep).n_params()
        if best is None or abs(p - target_params) < abs(best - target_params):
            best, bw = p, W
    return bw, best


# ----------------------------------------------------------------------------
# 3. Gaussian splatting
# ----------------------------------------------------------------------------
# Same rendering equation, same photographs, same loss. What changes is the
# answer to "what is a scene made of": here it is an explicit pile of blobs
# that get projected, sorted and alpha composited, and the only reason that is
# fast is that a blob is cut off at three standard deviations and therefore
# touches a bounded patch of the screen instead of all of it.

CUTOFF2 = 9.0      # Mahalanobis distance squared at which a blob stops counting
DILATE = 0.30      # pixels^2 added to the projected covariance, as in the paper
Z_MIN = 0.30
ALPHA_MAX = 0.995


def quat_to_R(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


class Splats(nn.Module):
    """A cloud of 3D gaussians. `iso` ties the three scales together and drops
    the rotation, which is the ablation the article needs: a sphere has one
    number where an ellipsoid has six."""

    def __init__(self, n, iso=False, viewdep=True, seed=19, dtype=torch.float32):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        lo = torch.as_tensor(AABB_LO, dtype=dtype)
        hi = torch.as_tensor(AABB_HI, dtype=dtype)
        self.iso, self.viewdep, self.n = iso, viewdep, n
        self.mu = nn.Parameter(lo + (hi - lo) * torch.rand(n, 3, generator=g, dtype=dtype))
        self.logs = nn.Parameter(torch.full((n, 1 if iso else 3), math.log(0.075), dtype=dtype))
        self.q = nn.Parameter(torch.cat(
            [torch.ones(n, 1, dtype=dtype), 0.01 * torch.randn(n, 3, generator=g, dtype=dtype)], 1))
        if iso:
            self.q.requires_grad_(False)
        self.op = nn.Parameter(torch.full((n,), -2.0, dtype=dtype))
        self.c0 = nn.Parameter(torch.zeros(n, 3, dtype=dtype))
        self.c1 = nn.Parameter(torch.zeros(n, 3, 3, dtype=dtype))
        if not viewdep:
            self.c1.requires_grad_(False)

    def dof(self):
        """Degrees of freedom per blob, which is what a fair comparison counts:
        a quaternion is four numbers holding three."""
        return 3 + (1 if self.iso else 6) + 1 + 3 + (9 if self.viewdep else 0)

    def sigma3(self):
        s = torch.exp(self.logs)
        if self.iso:
            s = s.expand(-1, 3)
        R = quat_to_R(self.q)
        M = R * s[:, None, :]
        return M @ M.transpose(1, 2)

    def colours(self, campos):
        d = self.mu - torch.as_tensor(campos, dtype=self.mu.dtype)
        d = d / d.norm(dim=-1, keepdim=True)
        raw = (self.c0 + (self.c1 @ d[..., None])[..., 0]) if self.viewdep else self.c0
        return torch.sigmoid(raw)


def project(model, cam, res=RES):
    """World blobs to screen blobs: centre, inverse 2D covariance, depth, and
    the screen radius the browser uses to bound its pixel loop."""
    dt = model.mu.dtype
    Wv = torch.as_tensor(cam["basis"], dtype=dt)
    pos = torch.as_tensor(cam["pos"], dtype=dt)
    f = 0.5 * res / math.tan(math.radians(FOV_DEG) / 2.0)
    pc = (model.mu - pos) @ Wv.T
    z = pc[:, 2].clamp(min=Z_MIN)
    mu2 = torch.stack([0.5 * res + f * pc[:, 0] / z, 0.5 * res + f * pc[:, 1] / z], -1)
    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([f / z, zero, -f * pc[:, 0] / z ** 2], -1),
        torch.stack([zero, f / z, -f * pc[:, 1] / z ** 2], -1),
    ], dim=-2)
    M = J @ Wv
    S2 = M @ model.sigma3() @ M.transpose(1, 2)
    S2 = S2 + DILATE * torch.eye(2, dtype=dt)
    a, b, c = S2[:, 0, 0], S2[:, 0, 1], S2[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-8)
    inv = torch.stack([c / det, -b / det, a / det], -1)      # [i00, i01, i11]
    tr = 0.5 * (a + c)
    root = torch.sqrt((tr * tr - det).clamp(min=0.0))
    radius = 3.0 * torch.sqrt((tr + root).clamp(min=1e-8))
    return mu2, inv, pc[:, 2], radius


def splat_render(model, cam, res=RES, pixels=None):
    """Alpha composite the blobs front to back. `pixels` is (P,2) in pixel
    centres; None means the whole frame in row-major order."""
    dt = model.mu.dtype
    mu2, inv, z, _ = project(model, cam, res)
    keep = torch.nonzero(z > Z_MIN, as_tuple=True)[0]
    order = keep[torch.argsort(z[keep])]
    if pixels is None:
        j, i = np.meshgrid(np.arange(res), np.arange(res))
        pixels = torch.as_tensor(np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1), dtype=dt)
    col = model.colours(cam["pos"])[order]
    op = torch.sigmoid(model.op)[order]
    d = pixels[:, None, :] - mu2[order][None, :, :]
    iv = inv[order]
    m = (iv[:, 0] * d[..., 0] ** 2 + 2 * iv[:, 1] * d[..., 0] * d[..., 1] + iv[:, 2] * d[..., 1] ** 2)
    gv = torch.exp(-0.5 * m) * (m <= CUTOFF2)
    alpha = (op * gv).clamp(max=ALPHA_MAX)
    T = torch.cumprod(torch.cat([torch.ones(d.shape[0], 1, dtype=dt), 1.0 - alpha + 1e-10], 1), 1)[:, :-1]
    w = T * alpha
    return (w[..., None] * col).sum(1) + (1.0 - w.sum(1))[:, None] * torch.as_tensor(BG, dtype=dt)


def render_splat_image(model, cam, res=RES, chunk=2048):
    with torch.no_grad():
        j, i = np.meshgrid(np.arange(res), np.arange(res))
        px = torch.as_tensor(np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1), dtype=model.mu.dtype)
        out = []
        for s in range(0, px.shape[0], chunk):
            out.append(splat_render(model, cam, res, px[s:s + chunk]).numpy())
    return np.concatenate(out).reshape(res, res, 3)


def train_splats(cams, images, *, n=1200, iso=False, viewdep=True, iters=2500,
                 batch=1024, seed=19, recycle=300, log=None):
    """Adam on every blob, with a fixed-budget stand-in for the paper's
    clone-and-split: every `recycle` steps the blobs that have gone transparent
    are moved next to the blobs whose centres are pulling hardest. Setting
    recycle to 0 turns it off, which is one of the ablations."""
    torch.manual_seed(seed)
    model = Splats(n, iso=iso, viewdep=viewdep, seed=seed)
    targets = [torch.as_tensor(im.reshape(-1, 3), dtype=torch.float32) for im in images]
    j, i = np.meshgrid(np.arange(RES), np.arange(RES))
    allpx = torch.as_tensor(np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1), dtype=torch.float32)
    groups = [
        {"params": [model.mu], "lr": 4e-3},
        {"params": [model.logs], "lr": 8e-3},
        {"params": [model.q], "lr": 8e-3},
        {"params": [model.op], "lr": 4e-2},
        {"params": [model.c0], "lr": 2e-2},
        {"params": [model.c1], "lr": 8e-3},
    ]
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.05 ** (1.0 / iters))
    g = torch.Generator().manual_seed(seed)
    accum = torch.zeros(n)
    hits = torch.zeros(n)
    t0 = time.time()
    for it in range(iters):
        v = int(torch.randint(0, len(cams), (1,), generator=g))
        pix = torch.randint(0, RES * RES, (batch,), generator=g)
        rgb = splat_render(model, cams[v], RES, allpx[pix])
        loss = ((rgb - targets[v][pix]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            accum += model.mu.grad.norm(dim=-1)
            hits += 1.0
        opt.step()
        sched.step()
        if recycle and it and it % recycle == 0 and it < 0.7 * iters:
            with torch.no_grad():
                score = accum / hits.clamp(min=1)
                op = torch.sigmoid(model.op)
                dead = torch.nonzero(op < 0.02, as_tuple=True)[0]
                if dead.numel():
                    live = torch.nonzero(op >= 0.02, as_tuple=True)[0]
                    pick = live[torch.argsort(score[live], descending=True)][:dead.numel()]
                    k = pick.numel()
                    dead = dead[:k]
                    model.mu[dead] = model.mu[pick] + 0.35 * torch.exp(model.logs[pick]).mean(
                        dim=-1, keepdim=True) * torch.randn(k, 3, generator=g)
                    model.logs[dead] = model.logs[pick] - math.log(1.6)
                    model.q[dead] = model.q[pick]
                    model.op[dead] = model.op[pick]
                    model.c0[dead] = model.c0[pick]
                    model.c1[dead] = model.c1[pick]
                    for p in (model.mu, model.logs, model.q, model.op, model.c0, model.c1):
                        st = opt.state.get(p)
                        if st:
                            st["exp_avg"][dead] = 0
                            st["exp_avg_sq"][dead] = 0
                accum.zero_()
                hits.zero_()
        with torch.no_grad():
            model.logs.clamp_(math.log(0.004), math.log(0.55))
        if log and (it % log == 0 or it == iters - 1):
            print(f"      it {it:5d}  loss {loss.item():.5f}  {time.time() - t0:.0f}s", flush=True)
    return model, time.time() - t0


# ----------------------------------------------------------------------------
# 4. Export helpers
# ----------------------------------------------------------------------------
# Half precision is what the browser is going to hold, so every model is
# quantised BEFORE anything is measured about it. Otherwise the page would be
# advertising the accuracy of a model nobody runs.

def f16(arr):
    """(base64 of float16, the dequantised values). One function, so a weight
    and the reference computed from it can never come from different bits."""
    a = np.asarray(arr, dtype=np.float32).astype(np.float16)
    return base64.b64encode(a.tobytes()).decode("ascii"), a.astype(np.float32)


def quantise_(module):
    """Round every parameter of a torch module to float16, in place."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(p.detach().float().half().float().to(p.dtype))


def save_png(arr, path, scale=1):
    from PIL import Image
    a = (np.clip(np.asarray(arr), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    im = Image.fromarray(a)
    if scale != 1:
        im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(path)
    return path.stat().st_size


def sheet(tiles, cols, gap=0):
    """Tile equal-sized images into one contact sheet, row major."""
    h, w = tiles[0].shape[:2]
    rows = int(math.ceil(len(tiles) / cols))
    out = np.ones((rows * h + gap * (rows - 1), cols * w + gap * (cols - 1), 3))
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = t
    return out


def photograph(cam, steps=512):
    """A photograph: the integral, then quantised to bytes, because that is what
    a photograph is and because it is what the browser will read back out of a
    PNG. Everything downstream trains and scores against these bytes."""
    im, alpha, depth = render_truth(cam, RES, steps)
    return np.round(np.clip(im, 0, 1) * 255.0) / 255.0, alpha, depth


def nerf_macs(model):
    """Multiply-accumulates for one point through the field, from the layer
    shapes rather than from a stopwatch."""
    m = 0
    for lin in list(model.trunk) + [model.sigma, model.feat, model.col1, model.col2]:
        m += lin.in_features * lin.out_features
    return m


def occupancy(model, res=40, thresh=0.05, sub=3):
    """A conservative binary grid over the scene box: a cell is empty only if
    the fitted density is below `thresh` everywhere in it. Dilated by one cell,
    so a sample can never fall in an empty cell whose neighbour is occupied."""
    edges = [np.linspace(AABB_LO[k], AABB_HI[k], res + 1) for k in range(3)]
    cell = (AABB_HI - AABB_LO) / res
    offs = (np.arange(sub) + 0.5) / sub
    grid = np.zeros((res, res, res), dtype=bool)
    with torch.no_grad():
        for a in range(sub):
            for b in range(sub):
                for c in range(sub):
                    x = edges[0][:-1] + offs[a] * cell[0]
                    y = edges[1][:-1] + offs[b] * cell[1]
                    z = edges[2][:-1] + offs[c] * cell[2]
                    P = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)
                    sig, _ = model(torch.as_tensor(P, dtype=torch.float32),
                                   torch.zeros(P.shape[0], 3))
                    grid |= (sig.numpy() > thresh).reshape(res, res, res)
    d = grid.copy()
    for ax in range(3):
        d |= np.roll(grid, 1, axis=ax) | np.roll(grid, -1, axis=ax)
    return d


def pack_bits(grid):
    flat = grid.reshape(-1).astype(np.uint8)
    return base64.b64encode(np.packbits(flat).tobytes()).decode("ascii")


def render_model_skipped(model, cam, steps, grid, res=RES, chunk=2048):
    """The render the browser performs: identical quadrature, but a sample in an
    empty cell never reaches the network. Returns the image and the fraction of
    samples that survived."""
    o, dirs, _ = rays(cam, res)
    out = np.tile(BG, (dirs.shape[0], 1))
    tn, tf, hit = ray_box(o, dirs)
    idx = np.flatnonzero(hit)
    gres = grid.shape[0]
    kept = tot = 0
    with torch.no_grad():
        for s in range(0, idx.size, chunk):
            sel = idx[s:s + chunk]
            d = dirs[sel]
            lo, hi = tn[sel], tf[sel]
            u = (np.arange(steps) + 0.5) / steps
            t = lo[:, None] + u[None, :] * (hi - lo)[:, None]
            dt = ((hi - lo) / steps)[:, None]
            p = o + t[..., None] * d[:, None, :]
            c = np.floor((p - AABB_LO) / (AABB_HI - AABB_LO) * gres).astype(int)
            c = np.clip(c, 0, gres - 1)
            live = grid[c[..., 0], c[..., 1], c[..., 2]]
            tot += live.size
            kept += int(live.sum())
            sig = np.zeros(p.shape[:2], dtype=np.float32)
            col = np.zeros(p.shape, dtype=np.float32)
            if live.any():
                pp = torch.as_tensor(p[live], dtype=torch.float32)
                dd = torch.as_tensor(np.broadcast_to(d[:, None, :], p.shape)[live].copy(),
                                     dtype=torch.float32)
                s_, c_ = model(pp, dd)
                sig[live] = s_.numpy()
                col[live] = c_.numpy()
            a = 1.0 - np.exp(-sig * dt)
            T = np.cumprod(np.concatenate([np.ones((sel.size, 1)), 1.0 - a + 1e-10], 1), 1)[:, :-1]
            w = T * a
            out[sel] = (w[..., None] * col).sum(1) + (1.0 - w.sum(1))[:, None] * BG
    return out.reshape(res, res, 3), kept / max(tot, 1)


def mean_psnr(model_render, cams, truths):
    return float(np.mean([psnr(model_render(c), t) for c, t in zip(cams, truths)]))


# ----------------------------------------------------------------------------
# Geometry, which the loss never asked for
# ----------------------------------------------------------------------------
# Nothing in either fit was ever told where the surfaces are: both were shown
# photographs. A wrong shape that happens to produce the right photographs from
# the places somebody stood would have scored exactly as well, so the honest
# question is not only "what colour" but "how far away", and that is the
# expected depth of the same ray, sum_i T_i alpha_i t_i.

_TRUTH_DEPTH = {}


def truth_depth(cam, res=RES, steps=GT_STEPS):
    """Memoised: four comparisons ask the same fourteen cameras for it."""
    key = (round(cam["az"], 4), round(cam["el"], 4), res, steps)
    if key not in _TRUTH_DEPTH:
        _, alpha, depth = render_truth(cam, res, steps)
        _TRUTH_DEPTH[key] = (np.where(alpha > 1e-9, depth / np.maximum(alpha, 1e-12), 0.0), alpha)
    return _TRUTH_DEPTH[key]


def field_depth(model, cam, steps, res=RES, chunk=2048):
    dt = next(model.parameters()).dtype
    o, dirs, _ = rays(cam, res)
    depth = np.zeros(dirs.shape[0])
    alpha = np.zeros(dirs.shape[0])
    tn, tf, hit = ray_box(o, dirs)
    idx = np.flatnonzero(hit)
    with torch.no_grad():
        for s in range(0, idx.size, chunk):
            sel = idx[s:s + chunk]
            _, w, t = volume_render(
                model, torch.as_tensor(np.tile(o, (sel.size, 1)), dtype=dt),
                torch.as_tensor(dirs[sel], dtype=dt), torch.as_tensor(tn[sel], dtype=dt),
                torch.as_tensor(tf[sel], dtype=dt), steps)
            a = w.sum(1).numpy()
            alpha[sel] = a
            depth[sel] = np.where(a > 1e-9, (w * t).sum(1).numpy() / np.maximum(a, 1e-12), 0.0)
    return depth.reshape(res, res), alpha.reshape(res, res)


def splat_depth(model, cam, res=RES, chunk=2048):
    """Same accumulation over the sorted blobs, with each blob's view depth."""
    dt = model.mu.dtype
    with torch.no_grad():
        mu2, inv, z, _ = project(model, cam, res)
        keep = torch.nonzero(z > Z_MIN, as_tuple=True)[0]
        order = keep[torch.argsort(z[keep])]
        op = torch.sigmoid(model.op)[order]
        zz = z[order]
        j, i = np.meshgrid(np.arange(res), np.arange(res))
        px = torch.as_tensor(np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1), dtype=dt)
        depth = np.zeros(res * res)
        alpha = np.zeros(res * res)
        for s in range(0, px.shape[0], chunk):
            p = px[s:s + chunk]
            d = p[:, None, :] - mu2[order][None, :, :]
            iv = inv[order]
            m = (iv[:, 0] * d[..., 0] ** 2 + 2 * iv[:, 1] * d[..., 0] * d[..., 1]
                 + iv[:, 2] * d[..., 1] ** 2)
            gv = torch.exp(-0.5 * m) * (m <= CUTOFF2)
            al = (op * gv).clamp(max=ALPHA_MAX)
            T = torch.cumprod(torch.cat([torch.ones(p.shape[0], 1, dtype=dt), 1.0 - al + 1e-10], 1), 1)[:, :-1]
            w = T * al
            a = w.sum(1).numpy()
            alpha[s:s + chunk] = a
            depth[s:s + chunk] = np.where(a > 1e-9, (w * zz).sum(1).numpy() / np.maximum(a, 1e-12), 0.0)
    return depth.reshape(res, res), alpha.reshape(res, res)


def _geometry(nerf, splat, in_c, out_c):
    """How far each fit puts the surface from where it is, over the pixels where
    both it and the scene agree there is a surface at all, and how often they do
    not agree about that."""
    def compare(get, cams):
        errs, fracs, worsts = [], [], []
        for c in cams:
            td, ta = truth_depth(c)
            md, ma = get(c)
            inT, inM = ta > 0.5, ma > 0.5
            both = inT & inM
            errs.append(float(np.abs(td[both] - md[both]).mean()) if both.any() else float("nan"))
            worsts.append(float(np.abs(td[both] - md[both]).max()) if both.any() else float("nan"))
            fracs.append(float((inT != inM).mean()))
        return dict(depth_error=rnd(float(np.mean(errs)), 5),
                    depth_worst=rnd(float(np.max(worsts)), 4),
                    silhouette=rnd(float(np.mean(fracs)), 5))
    return dict(
        pixel=rnd(2 * CAM_RADIUS * math.tan(math.radians(FOV_DEG) / 2) / RES, 5),
        field_inside=compare(lambda c: field_depth(nerf, c, EVAL_STEPS), in_c),
        field_outside=compare(lambda c: field_depth(nerf, c, EVAL_STEPS), out_c),
        splat_inside=compare(lambda c: splat_depth(splat, c), in_c),
        splat_outside=compare(lambda c: splat_depth(splat, c), out_c))


# ----------------------------------------------------------------------------
# 5. The run
# ----------------------------------------------------------------------------
GT_STEPS = 512
EVAL_STEPS = 64
TRAIN_STEPS = 48
ITERS = 60 if SMALL else 4000
SP_ITERS = 60 if SMALL else 3000
N_BLOBS = 60 if SMALL else 1200
SEED = 19
LX_LADDER = [0, 2, 4, 6, 8]
BLOB_LADDER = [300, 600, 1200, 2400] if not SMALL else [30, 60]


def nerf_key(tag, kw):
    return f"nerf-v1-{tag}-r{RES}-i{kw.get('iters', ITERS)}-" + json.dumps(kw, sort_keys=True)


def get_nerf(tag, cams, imgs, **kw):
    kw.setdefault("iters", ITERS)
    kw.setdefault("steps", TRAIN_STEPS)
    kw.setdefault("seed", SEED)

    def run():
        m, hist, el = train_nerf(cams, imgs, log=500, **kw)
        raw = {k: v.detach().numpy().copy() for k, v in m.state_dict().items()}
        quantise_(m)
        return dict(sd={k: v.detach().numpy() for k, v in m.state_dict().items()},
                    raw=raw, hist=hist, seconds=el)

    got = cached(nerf_key(tag, kw), run)
    kinds = {k: kw[k] for k in ("Lx", "Ld", "W", "D", "viewdep") if k in kw}
    m = NeRF(**kinds).float()
    m.load_state_dict({k: torch.as_tensor(v) for k, v in got["sd"].items()})
    m.eval()
    raw = NeRF(**kinds).float()
    raw.load_state_dict({k: torch.as_tensor(v) for k, v in got["raw"].items()})
    raw.eval()
    return m, raw, got


def get_splats(tag, cams, imgs, **kw):
    kw.setdefault("iters", SP_ITERS)
    kw.setdefault("n", N_BLOBS)
    kw.setdefault("seed", SEED)

    def run():
        m, el = train_splats(cams, imgs, log=500, **kw)
        raw = {k: v.detach().numpy().copy() for k, v in m.state_dict().items()}
        quantise_(m)
        return dict(sd={k: v.detach().numpy() for k, v in m.state_dict().items()},
                    raw=raw, seconds=el)

    got = cached(f"splat-v1-{tag}-r{RES}-" + json.dumps(kw, sort_keys=True), run)
    kinds = dict(n=kw["n"], iso=kw.get("iso", False), viewdep=kw.get("viewdep", True),
                 seed=kw["seed"])
    m = Splats(**kinds)
    m.load_state_dict({k: torch.as_tensor(v) for k, v in got["sd"].items()})
    raw = Splats(**kinds)
    raw.load_state_dict({k: torch.as_tensor(v) for k, v in got["raw"].items()})
    return m, raw, got


def main():
    out = {}
    t_all = time.time()
    print(f"threads {THREADS}, resolution {RES}, {'SMALL' if SMALL else 'full'} run", flush=True)

    # ---- the scene box actually contains the scene ---------------------------
    gp = np.stack(np.meshgrid(*[np.linspace(-1.4, 1.4, 141)] * 3, indexing="ij"), -1)
    sig, _ = scene_density(gp.reshape(-1, 3))
    live = gp.reshape(-1, 3)[sig > 0]
    step = 2.8 / 140.0
    assert (live.min(0) - step >= AABB_LO).all() and (live.max(0) + step <= AABB_HI).all(), \
        f"scene box too tight: {live.min(0)} .. {live.max(0)}"
    out["box"] = dict(lo=[rnd(v, 4) for v in AABB_LO], hi=[rnd(v, 4) for v in AABB_HI],
                      solid_fraction=rnd(float((sig > 0).mean() * (2.8 ** 3)
                                               / np.prod(AABB_HI - AABB_LO)), 4))

    # ---- 0. photographs ------------------------------------------------------
    train_c, in_c, out_c = build_cameras()
    print("0. photographs", flush=True)
    photos = cached(f"photos-v1-r{RES}-s{GT_STEPS}",
                    lambda: [photograph(c, GT_STEPS) for c in train_c + in_c + out_c])
    n_tr, n_in = len(train_c), len(in_c)
    tr_img = [p[0] for p in photos[:n_tr]]
    in_img = [p[0] for p in photos[n_tr:n_tr + n_in]]
    out_img = [p[0] for p in photos[n_tr + n_in:]]
    coverage = float(np.mean([p[1].mean() for p in photos]))

    # ---- 1. quadrature -------------------------------------------------------
    print("1. quadrature", flush=True)
    QSTEPS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    quad = cached(f"quad-v1-r{RES}-{QSTEPS}", lambda: _quadrature(train_c, QSTEPS))
    out["quadrature"] = quad

    # ---- 2. NeRF -------------------------------------------------------------
    print("2. NeRF", flush=True)
    main_kw = dict(Lx=6, Ld=2, W=64, D=4, viewdep=True)
    nerf, nerf_raw, nerf_got = get_nerf("main", train_c, tr_img, **main_kw)

    grid = cached(f"occ-v1-{nerf_key('main', dict(main_kw, iters=ITERS, steps=TRAIN_STEPS, seed=SEED))}",
                  lambda: occupancy(nerf))

    nerf_stats = cached("nerfstats-v1-" + nerf_key("main", dict(main_kw, iters=ITERS, steps=TRAIN_STEPS, seed=SEED)),
                        lambda: _nerf_stats(nerf, nerf_raw, grid, train_c, tr_img, in_c, in_img, out_c, out_img))
    out["nerf"] = nerf_stats
    out["nerf"]["params"] = nerf.n_params()
    out["nerf"]["macs_per_point"] = nerf_macs(nerf)
    out["nerf"]["seconds"] = round(nerf_got["seconds"], 1)
    out["nerf"]["loss_curve"] = nerf_got["hist"]

    # encoding ladder, plus the control that spends the same weights on width
    ladder = []
    for L in LX_LADDER:
        kw = dict(main_kw, Lx=L)
        if L == main_kw["Lx"]:
            m, got = nerf, nerf_got          # the same model, not another copy of it
        else:
            m, _, got = get_nerf(f"lx{L}", train_c, tr_img, **kw)
        ladder.append(dict(L=L, params=m.n_params(), model=m,
                           inside=mean_psnr(lambda c: render_model(m, c, EVAL_STEPS), in_c, in_img),
                           train=mean_psnr(lambda c: render_model(m, c, EVAL_STEPS), train_c, tr_img),
                           seconds=round(got["seconds"], 1)))
        print(f"    Lx={L}: {ladder[-1]['inside']:.2f} dB inside, {m.n_params()} params", flush=True)
    W0, p0 = match_width(nerf.n_params(), 0, main_kw["Ld"], main_kw["D"], True)
    kw = dict(main_kw, Lx=0, W=W0)
    mw, _, gotw = get_nerf(f"lx0w{W0}", train_c, tr_img, **kw)
    wide = dict(L=0, W=W0, params=mw.n_params(),
                inside=mean_psnr(lambda c: render_model(mw, c, EVAL_STEPS), in_c, in_img))
    print(f"    Lx=0 widened to W={W0} ({mw.n_params()} params): {wide['inside']:.2f} dB", flush=True)
    out["nerf"]["encoding"] = dict(
        ladder=[{k: (rnd(v, 3) if isinstance(v, float) else v)
                 for k, v in e.items() if k != "model"} for e in ladder],
        wide={k: (rnd(v, 3) if isinstance(v, float) else v) for k, v in wide.items()},
        matched_params=p0)

    # view dependence, measured where it lives rather than over the whole frame
    mnv, _, _ = get_nerf("noview", train_c, tr_img, **dict(main_kw, viewdep=False))
    out["viewdep"] = _viewdep(nerf, mnv, in_c, in_img)

    # photographs
    vcurve = []
    for k in sorted(VIEW_SUBSETS):
        sub = [train_c[i] for i in VIEW_SUBSETS[k]]
        subimg = [tr_img[i] for i in VIEW_SUBSETS[k]]
        m = nerf if k == 24 else get_nerf(f"views{k}", sub, subimg, **main_kw)[0]
        vcurve.append(dict(
            views=k,
            inside=rnd(mean_psnr(lambda c: render_model(m, c, EVAL_STEPS), in_c, in_img), 3),
            outside=rnd(mean_psnr(lambda c: render_model(m, c, EVAL_STEPS), out_c, out_img), 3),
            train=rnd(mean_psnr(lambda c: render_model(m, c, EVAL_STEPS), sub, subimg), 3),
            per_inside=[rnd(psnr(render_model(m, c, EVAL_STEPS), t), 3)
                        for c, t in zip(in_c, in_img)],
            per_outside=[rnd(psnr(render_model(m, c, EVAL_STEPS), t), 3)
                         for c, t in zip(out_c, out_img)]))
        print(f"    {k} views: inside {vcurve[-1]['inside']:.2f}, outside {vcurve[-1]['outside']:.2f}", flush=True)
    out["views"] = vcurve

    # a noise floor for every claim above: the same recipe, other seeds
    seeds = []
    for s in ([SEED] if SMALL else [SEED, 7, 41]):
        m = nerf if s == SEED else get_nerf(f"seed{s}", train_c, tr_img, **dict(main_kw, seed=s))[0]
        seeds.append(mean_psnr(lambda c: render_model(m, c, EVAL_STEPS), in_c, in_img))
    out["nerf"]["seed_psnr"] = [rnd(v, 3) for v in seeds]
    out["nerf"]["seed_spread"] = rnd(max(seeds) - min(seeds), 3)

    # ---- 3. splats -----------------------------------------------------------
    print("3. gaussian splatting", flush=True)
    splat, splat_raw, sp_got = get_splats("main", train_c, tr_img)
    out["splat"] = _splat_stats(splat, splat_raw, train_c, tr_img, in_c, in_img, out_c, out_img)
    out["splat"]["seconds"] = round(sp_got["seconds"], 1)

    counts = []
    for n in BLOB_LADDER:
        m = splat if n == N_BLOBS else get_splats(f"n{n}", train_c, tr_img, n=n)[0]
        counts.append(dict(n=n, dof=m.dof() * n,
                           inside=mean_psnr(lambda c: render_splat_image(m, c), in_c, in_img)))
        print(f"    {n} blobs: {counts[-1]['inside']:.2f} dB", flush=True)
    out["splat"]["counts"] = counts

    iso, _, _ = get_splats("iso", train_c, tr_img, iso=True)
    n_match = int(round(N_BLOBS * splat.dof() / iso.dof()))
    isom, _, _ = get_splats(f"isomatch{n_match}", train_c, tr_img, iso=True, n=n_match)
    nov, _, _ = get_splats("noview", train_c, tr_img, viewdep=False)
    norc, _, _ = get_splats("norecycle", train_c, tr_img, recycle=0)
    out["splat"]["ablations"] = [
        dict(key="aniso", n=N_BLOBS, dof=splat.dof() * N_BLOBS,
             inside=mean_psnr(lambda c: render_splat_image(splat, c), in_c, in_img)),
        dict(key="iso", n=N_BLOBS, dof=iso.dof() * N_BLOBS,
             inside=mean_psnr(lambda c: render_splat_image(iso, c), in_c, in_img)),
        dict(key="iso_matched", n=n_match, dof=isom.dof() * n_match,
             inside=mean_psnr(lambda c: render_splat_image(isom, c), in_c, in_img)),
        dict(key="no_viewdep", n=N_BLOBS, dof=nov.dof() * N_BLOBS,
             inside=mean_psnr(lambda c: render_splat_image(nov, c), in_c, in_img)),
        dict(key="no_recycle", n=N_BLOBS, dof=splat.dof() * N_BLOBS,
             inside=mean_psnr(lambda c: render_splat_image(norc, c), in_c, in_img)),
    ]
    for a in out["splat"]["ablations"]:
        print(f"    {a['key']}: {a['inside']:.2f} dB ({a['dof']} numbers)", flush=True)
    sp_seeds = []
    for s in ([SEED] if SMALL else [SEED, 7, 41]):
        m = splat if s == SEED else get_splats(f"seed{s}", train_c, tr_img, seed=s)[0]
        sp_seeds.append(mean_psnr(lambda c: render_splat_image(m, c), in_c, in_img))
    out["splat"]["seed_psnr"] = [rnd(v, 3) for v in sp_seeds]
    out["splat"]["seed_spread"] = rnd(max(sp_seeds) - min(sp_seeds), 3)

    # ---- 4. geometry, which neither loss ever asked for ---------------------
    print("4. geometry", flush=True)
    out["geometry"] = cached(
        "geom-v1-" + nerf_key("main", dict(main_kw, iters=ITERS, steps=TRAIN_STEPS, seed=SEED))
        + f"-sp{N_BLOBS}-{SP_ITERS}",
        lambda: _geometry(nerf, splat, in_c, out_c))
    for k, v in out["geometry"].items():
        if isinstance(v, dict):
            print(f"    {k}: {v['depth_error']} units off, silhouette {v['silhouette']}", flush=True)

    # ---- 5. what a frame costs, in arithmetic rather than in seconds --------
    out["cost"] = _cost(nerf, splat, out)

    out["_models"] = dict(nerf=nerf, splat=splat, grid=grid, ladder=ladder, wide=wide,
                          wide_model=mw, iso=iso, isom=isom, nov=nov, norc=norc,
                          views={k: (nerf if k == 24 else get_nerf(
                              f"views{k}", [train_c[i] for i in VIEW_SUBSETS[k]],
                              [tr_img[i] for i in VIEW_SUBSETS[k]], **main_kw)[0])
                              for k in sorted(VIEW_SUBSETS)},
                          counts={n: (splat if n == N_BLOBS else
                                      get_splats(f"n{n}", train_c, tr_img, n=n)[0])
                                  for n in BLOB_LADDER},
                          cams=(train_c, in_c, out_c),
                          photos=(tr_img, in_img, out_img),
                          coverage=coverage, mnv=mnv)
    print(f"all stages done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return out


# Multiply-accumulates, counted from the arithmetic each renderer performs.
# Written out term by term because the whole point of quoting arithmetic
# instead of milliseconds is that the reader can check it.
SPLAT_PROJECT_MACS = (
    9        # world point into camera coordinates, a 3x3 by 3
    + 4      # the two divisions by depth and the two image coordinates
    + 18     # M = J W, a 2x3 by a 3x3
    + 27     # M Sigma M', a 2x3 by 3x3 then by 3x2
    + 6      # invert the 2x2 and its determinant
    + 6      # eigenvalues, for the screen radius
    + 12     # the direction dependent colour, a 3x3 by 3
)
SPLAT_TOUCH_MACS = (
    6        # the quadratic form at one pixel
    + 1      # opacity times the gaussian
    + 5      # three colour channels, the accumulated weight, the transmittance
)


def _cost(nerf, splat, out):
    live = out["nerf"]["skip_kept"]
    per_point = nerf_macs(nerf)
    pix = RES * RES
    nerf_frame = pix * EVAL_STEPS * live * per_point
    touch = out["splat"]["touched_per_pixel"]
    splat_frame = splat.n * SPLAT_PROJECT_MACS + pix * touch * SPLAT_TOUCH_MACS
    return dict(
        res=RES, steps=EVAL_STEPS, pixels=pix,
        nerf=dict(macs_per_point=per_point, live_fraction=live,
                  points_per_frame=int(round(pix * EVAL_STEPS * live)),
                  samples_per_frame=pix * EVAL_STEPS,
                  macs_per_pixel=int(round(nerf_frame / pix)),
                  macs_per_frame=int(round(nerf_frame)),
                  params=nerf.n_params()),
        splat=dict(blobs=splat.n, touched_per_pixel=touch,
                   project_macs=SPLAT_PROJECT_MACS, touch_macs=SPLAT_TOUCH_MACS,
                   sort_comparisons=int(round(splat.n * math.log2(max(splat.n, 2)))),
                   macs_per_pixel=int(round(splat_frame / pix)),
                   macs_per_frame=int(round(splat_frame)),
                   numbers=splat.dof() * splat.n),
        ratio=rnd(nerf_frame / splat_frame, 2))


# ----------------------------------------------------------------------------
# 6. Writing it out
# ----------------------------------------------------------------------------

def _layer_export(model):
    """The contract the browser reads: every weight in one float16 blob, plus
    where each layer's slice starts. Names are what nerfkit looks for."""
    flat = []
    layers = []
    def add(name, lin):
        W = lin.weight.detach().numpy().astype(np.float32)
        b = lin.bias.detach().numpy().astype(np.float32)
        layers.append(dict(name=name, shape=[int(W.shape[0]), int(W.shape[1])],
                           w_offset=sum(len(f) for f in flat), w_size=int(W.size)))
        flat.append(W.reshape(-1))
        layers[-1]["b_offset"] = sum(len(f) for f in flat)
        layers[-1]["b_size"] = int(b.size)
        flat.append(b.reshape(-1))
    for i, lin in enumerate(model.trunk):
        add(f"trunk{i}", lin)
    add("sigma", model.sigma)
    add("feat", model.feat)
    add("col1", model.col1)
    add("col2", model.col2)
    blob = np.concatenate(flat)
    b64, _ = f16(blob)
    return layers, b64, int(blob.size)


def _refs(image, n=150, seed=3):
    """A stride of pixels for the browser to check itself against, spread over
    the frame rather than clustered, and written with enough decimals that the
    comparison measures the implementation and not the rounding."""
    flat = image.reshape(-1, 3)
    step = max(1, flat.shape[0] // n)
    idx = list(range(0, flat.shape[0], step))[:n]
    return dict(idx=idx, rgb=[[rnd(v, 6) for v in flat[i]] for i in idx])


def _errmap(a, b, gain=6.0):
    e = np.clip(np.abs(np.asarray(a) - np.asarray(b)).mean(-1) * gain, 0, 1)
    return np.stack([np.ones_like(e), 1 - e, 1 - e * 0.92], -1)


def export(R):
    print("6. writing", flush=True)
    M = R.pop("_models")
    nerf, splat, grid = M["nerf"], M["splat"], M["grid"]
    train_c, in_c, out_c = M["cams"]
    tr_img, in_img, out_img = M["photos"]

    meta = dict(
        res=RES, fov_deg=FOV_DEG, cam_radius=CAM_RADIUS, target=[rnd(v, 6) for v in TARGET],
        bg=[rnd(v, 6) for v in BG], light=[rnd(v, 8) for v in LIGHT],
        shell=SHELL, sigma_max=SIGMA_MAX, ambient=AMBIENT, diffuse=DIFFUSE,
        normal_eps=NORMAL_EPS,
        aabb=dict(lo=[rnd(v, 6) for v in AABB_LO], hi=[rnd(v, 6) for v in AABB_HI]),
        gt_steps=GT_STEPS, eval_steps=EVAL_STEPS, train_steps=TRAIN_STEPS,
        iters=ITERS, splat_iters=SP_ITERS, splat_recycle=300, batch=1024, seed=SEED, threads=THREADS,
        rings=RING_EL, ring_n=RING_N, test_el=TEST_EL, outside_el=OUTSIDE_EL,
        n_train=len(train_c), n_inside=len(in_c), n_outside=len(out_c),
        coverage=rnd(M["coverage"], 4),
        view_subsets={str(k): v for k, v in VIEW_SUBSETS.items()},
    )
    shapes = []
    for s in SHAPES:
        d = dict(name=s["name"], kind=s["kind"], c=[rnd(v, 6) for v in s["c"]],
                 rot=s["rot"], R=[[rnd(v, 8) for v in row] for row in s["Rm"]],
                 albedo=[rnd(v, 6) for v in s["albedo"]], ks=s["ks"], shine=s["shine"])
        if s["kind"] == "sphere":
            d["r"] = s["r"]
        elif s["kind"] == "torus":
            d["R_major"] = s["R"]
            d["r"] = s["r"]
        else:
            d["h"] = s["h"]
        shapes.append(d)

    def cam_json(c):
        return dict(az=rnd(c["az"], 4), el=rnd(c["el"], 4),
                    pos=[rnd(v, 6) for v in c["pos"]],
                    basis=[[rnd(v, 8) for v in row] for row in c["basis"]])

    scene = dict(meta=meta, shapes=shapes,
                 cameras=dict(train=[cam_json(c) for c in train_c],
                              inside=[cam_json(c) for c in in_c],
                              outside=[cam_json(c) for c in out_c]))
    for k in ("box", "quadrature", "nerf", "views", "viewdep", "splat", "geometry", "cost"):
        scene[k] = R[k]
    scene["nerf"]["arch"] = dict(Lx=nerf.Lx, Ld=nerf.Ld, W=nerf.W, D=nerf.D, viewdep=nerf.viewdep)

    # the browser rebuilds the scene from its closed form: these are the pixels
    # it has to land on
    gcam = in_c[0]
    scene["check"] = dict(
        res=32, truth_steps=64, field_steps=32,
        cam=cam_json(gcam),
        truth=_refs(render_truth(gcam, 32, 64)[0]),
        field=_refs(render_model(nerf, gcam, 32, res=32)),
        field_skipped=_refs(render_model_skipped(nerf, gcam, 32, grid, res=32)[0]),
        splat=_refs(render_splat_image(splat, gcam, 32)),
    )

    # ---- the field --------------------------------------------------------
    layers, wb64, n_w = _layer_export(nerf)
    field = dict(
        arch=dict(Lx=nerf.Lx, Ld=nerf.Ld, W=nerf.W, D=nerf.D, viewdep=nerf.viewdep,
                  sig_scale=SIG_SCALE,
                  centre=[rnd(v, 6) for v in CENTER], half=[rnd(v, 6) for v in HALF],
                  aabb=dict(lo=[rnd(v, 6) for v in AABB_LO], hi=[rnd(v, 6) for v in AABB_HI])),
        layers=layers, weights=wb64, n_weights=n_w, params=nerf.n_params(),
        grid=dict(res=int(grid.shape[0]), bits=pack_bits(grid),
                  occupied=rnd(float(grid.mean()), 4)))
    (OUT / "field.json").write_text(json.dumps(field), encoding="utf-8")

    # ---- the blobs --------------------------------------------------------
    # The scale and the rotation travel, not the covariance they build. Every
    # parameter here has already been rounded to float16, so writing them out
    # is exact; writing Sigma instead would round a second time, and then the
    # browser would be running a model the references were never computed from.
    with torch.no_grad():
        logs = splat.logs
        if splat.iso:
            logs = logs.expand(-1, 3)
        blobs = dict(
            n=splat.n, viewdep=splat.viewdep, iso=splat.iso, dof_each=splat.dof(),
            cfg=dict(cutoff2=CUTOFF2, dilate=DILATE, z_min=Z_MIN, alpha_max=ALPHA_MAX,
                     fov_deg=FOV_DEG),
            mu=f16(splat.mu.numpy())[0], logs=f16(logs.numpy().copy())[0],
            quat=f16(splat.q.numpy())[0],
            opacity=f16(splat.op.numpy())[0], c0=f16(splat.c0.numpy())[0],
            c1=f16(splat.c1.numpy().reshape(splat.n, 9))[0])
    (OUT / "blobs.json").write_text(json.dumps(blobs), encoding="utf-8")

    # ---- contact sheets ---------------------------------------------------
    sizes = {}
    sizes["truth-train"] = save_png(sheet(tr_img, 6), IMG / "truth-train.png")
    sizes["truth-test"] = save_png(sheet(in_img + out_img, 7), IMG / "truth-test.png")

    lad = [render_model(e["model"], in_c[0], EVAL_STEPS) for e in M["ladder"]]
    lad.append(render_model(M["wide_model"], in_c[0], EVAL_STEPS))
    lad.append(in_img[0])
    sizes["encoding"] = save_png(sheet(lad, len(lad)), IMG / "encoding.png")
    scene["sheets"] = dict(
        train=dict(cols=6, tile=RES),
        encoding=dict(cols=len(lad), tile=RES,
                      labels=[f"L={e['L']}" for e in M["ladder"]]
                             + [f"L=0, W={M['wide']['W']}", "photograph"]),
    )

    vv = []
    for k in sorted(M["views"]):
        vv.append(render_model(M["views"][k], in_c[0], EVAL_STEPS))
    for k in sorted(M["views"]):
        vv.append(render_model(M["views"][k], out_c[0], EVAL_STEPS))
    vv = [in_img[0]] + vv[:len(M["views"])] + [out_img[0]] + vv[len(M["views"]):]
    sizes["views"] = save_png(sheet(vv, len(M["views"]) + 1), IMG / "views.png")
    scene["sheets"]["views"] = dict(cols=len(M["views"]) + 1, tile=RES,
                                    labels=["photograph"] + [f"{k} photos" for k in sorted(M["views"])])

    vd_on = render_model(nerf, in_c[0], EVAL_STEPS)
    vd_off = render_model(M["mnv"], in_c[0], EVAL_STEPS)
    sizes["viewdep"] = save_png(sheet(
        [in_img[0], vd_on, vd_off, _errmap(vd_on, in_img[0]), _errmap(vd_off, in_img[0])], 5),
        IMG / "viewdep.png")
    scene["sheets"]["viewdep"] = dict(cols=5, tile=RES, labels=[
        "photograph", "with direction", "without", "error, with", "error, without"])

    bl = [render_splat_image(M["counts"][n], in_c[0]) for n in sorted(M["counts"])]
    bl += [render_splat_image(M["iso"], in_c[0]), render_splat_image(M["isom"], in_c[0]),
           in_img[0]]
    sizes["blobs"] = save_png(sheet(bl, len(bl)), IMG / "blobs.png")
    scene["sheets"]["blobs"] = dict(cols=len(bl), tile=RES, labels=(
        [f"{n} blobs" for n in sorted(M["counts"])]
        + ["spheres", f"spheres, {M['isom'].n}", "photograph"]))

    cmp_tiles = []
    for cam, truth in [(in_c[0], in_img[0]), (out_c[0], out_img[0])]:
        a = render_model(nerf, cam, EVAL_STEPS)
        b = render_splat_image(splat, cam)
        cmp_tiles += [truth, a, b, _errmap(a, truth), _errmap(b, truth)]
    sizes["compare"] = save_png(sheet(cmp_tiles, 5), IMG / "compare.png")
    scene["sheets"]["compare"] = dict(cols=5, tile=RES, labels=[
        "photograph", "field", "blobs", "field error", "blob error"],
        rows=["inside the band", "below the band"])

    scene["meta"]["png_bytes"] = sizes
    scene["meta"]["json_bytes"] = dict(
        field=(OUT / "field.json").stat().st_size, blobs=(OUT / "blobs.json").stat().st_size)
    (OUT / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    scene["meta"]["json_bytes"]["scene"] = (OUT / "scene.json").stat().st_size
    (OUT / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    print(f"   scene.json {(OUT / 'scene.json').stat().st_size / 1024:.0f} kB, "
          f"field.json {(OUT / 'field.json').stat().st_size / 1024:.0f} kB, "
          f"blobs.json {(OUT / 'blobs.json').stat().st_size / 1024:.0f} kB, "
          f"png {sum(sizes.values()) / 1024:.0f} kB", flush=True)
    return scene


def _quadrature(cams, steps_list):
    """What discretising the integral costs, before anything is fitted."""
    ref = [render_truth(c, RES, 1024)[0] for c in cams[:4]]
    rows = []
    for s in steps_list:
        ims = [render_truth(c, RES, s)[0] for c in cams[:4]]
        rows.append(dict(steps=s,
                         psnr=rnd(float(np.mean([psnr(a, b) for a, b in zip(ims, ref)])), 3),
                         worst=rnd(float(np.max([np.abs(a - b).max() for a, b in zip(ims, ref)])), 4)))
    return dict(ref_steps=1024, cams=4, rows=rows)


def _nerf_stats(model, raw, grid, tr_c, tr_i, in_c, in_i, out_c, out_i):
    ev = lambda c: render_model(model, c, EVAL_STEPS)
    st = dict(
        train=rnd(mean_psnr(ev, tr_c, tr_i), 3),
        inside=rnd(mean_psnr(ev, in_c, in_i), 3),
        outside=rnd(mean_psnr(ev, out_c, out_i), 3),
        per_inside=[rnd(psnr(ev(c), t), 3) for c, t in zip(in_c, in_i)],
        per_outside=[rnd(psnr(ev(c), t), 3) for c, t in zip(out_c, out_i)],
    )
    # what half precision costs, measured on the model the reader runs
    st["fp16_psnr"] = rnd(float(np.mean([psnr(ev(c), render_model(raw, c, EVAL_STEPS)) for c in in_c])), 2)
    st["fp32_inside"] = rnd(mean_psnr(lambda c: render_model(raw, c, EVAL_STEPS), in_c, in_i), 3)
    # what skipping empty space costs, and what it saves
    sk = [render_model_skipped(model, c, EVAL_STEPS, grid) for c in in_c]
    st["skip_psnr"] = rnd(float(np.mean([psnr(a, ev(c)) for (a, _), c in zip(sk, in_c)])), 2)
    st["skip_kept"] = rnd(float(np.mean([f for _, f in sk])), 4)
    st["skip_inside"] = rnd(float(np.mean([psnr(a, t) for (a, _), t in zip(sk, in_i)])), 3)
    st["grid_res"] = int(grid.shape[0])
    st["grid_occupied"] = rnd(float(grid.mean()), 4)
    # quadrature of the FITTED field: the browser's sample slider lives here
    hi = [render_model(model, c, 256) for c in in_c]
    st["steps_curve"] = [dict(steps=s,
                              self=rnd(float(np.mean([psnr(render_model(model, c, s), h)
                                                      for c, h in zip(in_c, hi)])), 3),
                              truth=rnd(float(np.mean([psnr(render_model(model, c, s), t)
                                                       for c, t in zip(in_c, in_i)])), 3))
                         for s in ([16, 64] if SMALL else [8, 16, 24, 32, 48, 64, 96, 128])]
    return st


def _viewdep(model, noview, cams, imgs):
    """The specular lobe is 2% of the frame, so the frame average cannot see it.
    Split the pixels by how much the true radiance actually moves with the
    camera and the effect stops hiding."""
    on = [render_model(model, c, EVAL_STEPS) for c in cams]
    off = [render_model(noview, c, EVAL_STEPS) for c in cams]
    masks = []
    for c in cams:
        # the same scene with the specular lobe switched off in the closed form:
        # whatever moves is exactly the part of the radiance that depends on
        # where the camera is, and it is a small part of the frame
        a = render_truth(c, RES, GT_STEPS)[0]
        b = render_truth(c, RES, GT_STEPS, spec=False)[0]
        masks.append(np.abs(a - b).max(-1) > 0.02)
    def split(pred):
        hot = cold = 0.0
        for p, t, m in zip(pred, imgs, masks):
            hot += float(np.mean((p[m] - t[m]) ** 2)) if m.any() else 0.0
            cold += float(np.mean((p[~m] - t[~m]) ** 2))
        n = len(pred)
        return -10 * math.log10(hot / n), -10 * math.log10(cold / n)
    on_hot, on_cold = split(on)
    off_hot, off_cold = split(off)
    return dict(
        frac=rnd(float(np.mean([m.mean() for m in masks])), 4),
        on_all=rnd(float(np.mean([psnr(p, t) for p, t in zip(on, imgs)])), 3),
        off_all=rnd(float(np.mean([psnr(p, t) for p, t in zip(off, imgs)])), 3),
        on_hot=rnd(on_hot, 3), off_hot=rnd(off_hot, 3),
        on_cold=rnd(on_cold, 3), off_cold=rnd(off_cold, 3))


def _splat_stats(model, raw, tr_c, tr_i, in_c, in_i, out_c, out_i):
    ev = lambda c: render_splat_image(model, c)
    st = dict(
        n=model.n, dof_each=model.dof(), dof=model.dof() * model.n,
        train=rnd(mean_psnr(ev, tr_c, tr_i), 3),
        inside=rnd(mean_psnr(ev, in_c, in_i), 3),
        outside=rnd(mean_psnr(ev, out_c, out_i), 3),
        per_inside=[rnd(psnr(ev(c), t), 3) for c, t in zip(in_c, in_i)],
        per_outside=[rnd(psnr(ev(c), t), 3) for c, t in zip(out_c, out_i)],
        fp16_psnr=rnd(float(np.mean([psnr(ev(c), render_splat_image(raw, c)) for c in in_c])), 2),
    )
    # how many blobs a pixel actually has to look at, which is the whole
    # reason this is fast
    touch, opaque = [], []
    with torch.no_grad():
        for c in in_c:
            mu2, inv, z, rad = project(model, c, RES)
            keep = z > Z_MIN
            j, i = np.meshgrid(np.arange(RES), np.arange(RES))
            px = torch.as_tensor(np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1), dtype=torch.float32)
            d = px[:, None, :] - mu2[None, keep, :]
            iv = inv[keep]
            m = (iv[:, 0] * d[..., 0] ** 2 + 2 * iv[:, 1] * d[..., 0] * d[..., 1]
                 + iv[:, 2] * d[..., 1] ** 2)
            touch.append(float((m <= CUTOFF2).float().sum(1).mean()))
            opaque.append(float((torch.sigmoid(model.op)[keep] > 0.5).float().mean()))
    st["touched_per_pixel"] = rnd(float(np.mean(touch)), 2)
    st["visible"] = rnd(float(np.mean(opaque)), 4)
    with torch.no_grad():
        s = torch.exp(model.logs)
        if model.iso:
            s = s.expand(-1, 3)
        s, _ = torch.sort(s, dim=-1, descending=True)
        st["axis_ratio_median"] = rnd(float(torch.median(s[:, 0] / s[:, 2])), 3)
        st["scale_median"] = rnd(float(torch.median(s.mean(-1))), 4)
    return st


if __name__ == "__main__":
    export(main())



