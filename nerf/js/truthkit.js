/* The scene itself, rebuilt in the browser from its closed form.
 *
 * Every other picture on this page is a fit to photographs. This file is the
 * thing being photographed: four solids written as signed distance functions,
 * a compactly supported shell so the density is exactly zero outside them, and
 * Blinn-Phong radiance whose specular lobe is the reason the colour of a point
 * depends on where you are standing.
 *
 * The constants all arrive from scene.json, so there is one definition of the
 * world and not two, and the generator ships a handful of exact pixels for
 * this file to check itself against at load time.
 */

/* ---------------------------------------------------------------- geometry */

export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function scale(a, k) { return [a[0] * k, a[1] * k, a[2] * k]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function norm(a) { return Math.sqrt(dot(a, a)); }
export function unit(a) { const n = norm(a) || 1e-12; return [a[0] / n, a[1] / n, a[2] / n]; }

/* R^T v, with R stored row-major as the generator wrote it. */
function rotT(R, v) {
  return [
    R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
    R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
    R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
  ];
}

/* Signed distance to one solid. Same three primitives as the generator. */
export function sdf(shape, p) {
  const q = rotT(shape.R, sub(p, shape.c));
  if (shape.kind === 'sphere') return norm(q) - shape.r;
  if (shape.kind === 'torus') {
    const a = Math.hypot(q[0], q[2]) - shape.R_major;
    return Math.hypot(a, q[1]) - shape.r;
  }
  const dx = Math.abs(q[0]) - shape.h[0];
  const dy = Math.abs(q[1]) - shape.h[1];
  const dz = Math.abs(q[2]) - shape.h[2];
  const out = Math.hypot(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0));
  return out + Math.min(Math.max(dx, dy, dz), 0);
}

export class Scene {
  constructor(meta, shapes) {
    this.m = meta;
    this.shapes = shapes;
    this.lo = meta.aabb.lo;
    this.hi = meta.aabb.hi;
    this.light = meta.light;
  }

  /* Occupancy of one solid: 1 inside, 0 outside, smoothstep across the shell. */
  shell(s) {
    const w = this.m.shell;
    let t = (w - s) / (2 * w);
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return t * t * (3 - 2 * t);
  }

  /* Density and which solid owns the point. Disjoint solids, so the maximum is
     the union and the owner is unambiguous wherever the density is not zero. */
  densityAt(p) {
    let best = 0;
    let who = -1;
    for (let i = 0; i < this.shapes.length; i++) {
      const a = this.shell(sdf(this.shapes[i], p));
      if (a > best) { best = a; who = i; }
    }
    return { sigma: this.m.sigma_max * best, owner: who };
  }

  normal(shape, p) {
    const e = this.m.normal_eps;
    const g = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const a = p.slice();
      const b = p.slice();
      a[k] += e;
      b[k] -= e;
      g[k] = sdf(shape, a) - sdf(shape, b);
    }
    return unit(g);
  }

  /* Emitted radiance at p seen from direction `view` (camera into the scene). */
  radiance(p, view, owner, spec = true) {
    const s = this.shapes[owner];
    const n = this.normal(s, p);
    const v = [-view[0], -view[1], -view[2]];
    const lam = Math.max(dot(n, this.light), 0);
    const h = unit(add(this.light, v));
    const lobe = Math.pow(Math.max(dot(n, h), 0), s.shine);
    const k = (this.m.ambient + this.m.diffuse * lam);
    const e = spec ? s.ks * lobe : 0;
    return [
      Math.min(1, Math.max(0, s.albedo[0] * k + e)),
      Math.min(1, Math.max(0, s.albedo[1] * k + e)),
      Math.min(1, Math.max(0, s.albedo[2] * k + e)),
    ];
  }

  /* Slab test against the scene box. */
  rayBox(o, d) {
    let tn = -Infinity;
    let tf = Infinity;
    for (let k = 0; k < 3; k++) {
      const inv = 1 / (Math.abs(d[k]) < 1e-9 ? 1e-9 : d[k]);
      let a = (this.lo[k] - o[k]) * inv;
      let b = (this.hi[k] - o[k]) * inv;
      if (a > b) { const t = a; a = b; b = t; }
      if (a > tn) tn = a;
      if (b < tf) tf = b;
    }
    return [Math.max(tn, 0), tf];
  }

  /* The rendering equation along one ray, at `steps` midpoints. Returns the
     colour plus the whole profile, which is what the first widget draws. */
  march(o, d, steps, { profile = false, spec = true, stopAt = 1e-5 } = {}) {
    const [tn, tf] = this.rayBox(o, d);
    const bg = this.m.bg;
    if (tf <= tn) {
      return { rgb: bg.slice(), alpha: 0, depth: 0, tn: 0, tf: 0, prof: null };
    }
    const dt = (tf - tn) / steps;
    /* A profile is a picture of the whole ray, so it never stops early: giving
       up once the transmittance is spent would leave the density beyond the
       first surface drawn as zero, which is not what it is. */
    const stop = profile ? 0 : stopAt;
    let T = 1;
    const rgb = [0, 0, 0];
    let acc = 0;
    let depth = 0;
    const prof = profile
      ? { t: new Float32Array(steps), sigma: new Float32Array(steps), w: new Float32Array(steps),
          T: new Float32Array(steps), col: new Float32Array(steps * 3) }
      : null;
    for (let i = 0; i < steps; i++) {
      const t = tn + (i + 0.5) * dt;
      const p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
      const { sigma, owner } = this.densityAt(p);
      let a = 0;
      let c = null;
      if (sigma > 0) {
        a = 1 - Math.exp(-sigma * dt);
        c = this.radiance(p, d, owner, spec);
        const w = T * a;
        rgb[0] += w * c[0];
        rgb[1] += w * c[1];
        rgb[2] += w * c[2];
        acc += w;
        depth += w * t;
      }
      if (prof) {
        prof.t[i] = t;
        prof.sigma[i] = sigma;
        prof.T[i] = T;
        prof.w[i] = T * a;
        if (c) { prof.col[i * 3] = c[0]; prof.col[i * 3 + 1] = c[1]; prof.col[i * 3 + 2] = c[2]; }
      }
      if (sigma > 0) T *= (1 - a + 1e-12);
      if (T < stop) break;
    }
    return {
      rgb: [rgb[0] + (1 - acc) * bg[0], rgb[1] + (1 - acc) * bg[1], rgb[2] + (1 - acc) * bg[2]],
      alpha: acc, depth, tn, tf, prof,
    };
  }
}

/* ----------------------------------------------------------------- cameras */

/* Camera basis (right, down, forward) as rows, matching the generator. */
export function lookAt(pos, target) {
  const f = unit(sub(target, pos));
  /* cross(f, up) with up = (0, 1, 0) is (-f_z, 0, f_x) before normalising */
  const r = unit([-f[2], 0, f[0]]);
  const d = [
    f[1] * r[2] - f[2] * r[1],
    f[2] * r[0] - f[0] * r[2],
    f[0] * r[1] - f[1] * r[0],
  ];
  return [r, d, f];
}

export function camera(azDeg, elDeg, meta, radius) {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  const R = radius === undefined ? meta.cam_radius : radius;
  const t = meta.target;
  const pos = [
    R * Math.cos(el) * Math.sin(az) + t[0],
    R * Math.sin(el) + t[1],
    R * Math.cos(el) * Math.cos(az) + t[2],
  ];
  return { az: azDeg, el: elDeg, pos, basis: lookAt(pos, t) };
}

/* Pixel focal length: the same expression the generator uses. */
export function focal(res, fovDeg) {
  return (0.5 * res) / Math.tan((fovDeg * Math.PI) / 360);
}

/* Unit direction for pixel (row i, col j). */
export function rayDir(cam, res, fovDeg, i, j) {
  const f = focal(res, fovDeg);
  const x = (j + 0.5 - 0.5 * res) / f;
  const y = (i + 0.5 - 0.5 * res) / f;
  const [r, d, fw] = cam.basis;
  return unit([
    fw[0] + x * r[0] + y * d[0],
    fw[1] + x * r[1] + y * d[1],
    fw[2] + x * r[2] + y * d[2],
  ]);
}

/* A whole frame of the true scene: colour, expected depth and how much weight
 * the ray picked up at all. The last two are what makes it possible to ask
 * whether a fit got the shape right rather than only the picture. */
export function renderTruthFull(scene, cam, res, steps, opts = {}) {
  const rgb = new Float32Array(res * res * 3);
  const depth = new Float32Array(res * res);
  const alpha = new Float32Array(res * res);
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const d = rayDir(cam, res, scene.m.fov_deg, i, j);
      const r = scene.march(cam.pos, d, steps, opts);
      const p = i * res + j;
      rgb[p * 3] = r.rgb[0];
      rgb[p * 3 + 1] = r.rgb[1];
      rgb[p * 3 + 2] = r.rgb[2];
      alpha[p] = r.alpha;
      depth[p] = r.alpha > 1e-6 ? r.depth / r.alpha : 0;
    }
  }
  return { rgb, depth, alpha };
}

/* Colour only, which is what most of the page wants. */
export function renderTruth(scene, cam, res, steps, opts = {}) {
  return renderTruthFull(scene, cam, res, steps, opts).rgb;
}

/* Depth as a picture: near is light, far is dark, and anything the ray passed
 * straight through keeps the page's own background rather than pretending to
 * be at some depth. */
export function depthImage(depth, alpha, bg, lo, hi) {
  const out = new Float32Array(depth.length * 3);
  const span = Math.max(hi - lo, 1e-6);
  for (let p = 0; p < depth.length; p++) {
    if (alpha[p] < 0.5) {
      out[p * 3] = bg[0];
      out[p * 3 + 1] = bg[1];
      out[p * 3 + 2] = bg[2];
      continue;
    }
    const t = Math.max(0, Math.min(1, (depth[p] - lo) / span));
    out[p * 3] = 0.98 - 0.49 * t;
    out[p * 3 + 1] = 0.94 - 0.80 * t;
    out[p * 3 + 2] = 0.99 - 0.19 * t;
  }
  return out;
}

/* Where the two agree there is something, how far apart they put it; and how
 * often they disagree about whether there is anything there at all. */
export function depthCompare(dA, aA, dB, aB) {
  let n = 0;
  let sum = 0;
  let worst = 0;
  let disagree = 0;
  for (let p = 0; p < dA.length; p++) {
    const inA = aA[p] > 0.5;
    const inB = aB[p] > 0.5;
    if (inA !== inB) disagree++;
    if (!inA || !inB) continue;
    const e = Math.abs(dA[p] - dB[p]);
    sum += e;
    if (e > worst) worst = e;
    n++;
  }
  return { mean: n ? sum / n : 0, worst, both: n, disagree, frac: disagree / dA.length };
}

export function psnr(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  const mse = s / a.length;
  return mse <= 0 ? Infinity : -10 * Math.log10(mse);
}
