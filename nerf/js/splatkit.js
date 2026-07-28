/* The blob cloud, rasterised in the browser.
 *
 * Same rendering equation as the field next door, and a completely different
 * way of getting to it. Each blob is projected to an ellipse in screen space
 * with the EWA jacobian, the ellipses are sorted by depth, and then they are
 * alpha composited front to back. A blob is cut off at three standard
 * deviations, which is the only reason this is fast: it touches a bounded
 * patch of the screen instead of every pixel.
 *
 * The generator's rasteriser walks every blob for every pixel and applies the
 * same cutoff; this one walks only the pixels inside each blob's screen
 * bounding box, which is a superset of the pixels the cutoff keeps, so the two
 * produce the same picture and the widget checks that they do.
 */
import { decodeF16 } from './nerfkit.js';

export class Cloud {
  constructor(model) {
    const n = model.n;
    this.n = n;
    this.viewdep = model.viewdep;
    this.mu = decodeF16(model.mu, n * 3);
    this.op = decodeF16(model.opacity, n);
    this.c0 = decodeF16(model.c0, n * 3);
    this.c1 = model.viewdep ? decodeF16(model.c1, n * 9) : null;
    this.cfg = model.cfg;
    /* Sigma = R S S' R' is rebuilt here rather than shipped, because what the
       generator holds are the scale and the rotation and those are what got
       rounded to half precision. Shipping the product would round a second
       time and the page would be running a different model from the one every
       number on it describes. */
    const logs = decodeF16(model.logs, n * 3);
    const q = decodeF16(model.quat, n * 4);
    this.sig = new Float32Array(n * 6);          // xx, xy, xz, yy, yz, zz
    const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let g = 0; g < n; g++) {
      const s = [Math.exp(logs[g * 3]), Math.exp(logs[g * 3 + 1]), Math.exp(logs[g * 3 + 2])];
      const nq = Math.hypot(q[g * 4], q[g * 4 + 1], q[g * 4 + 2], q[g * 4 + 3]) || 1;
      const w = q[g * 4] / nq;
      const x = q[g * 4 + 1] / nq;
      const y = q[g * 4 + 2] / nq;
      const z = q[g * 4 + 3] / nq;
      const R = [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
      ];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i * 3 + j] = R[i * 3 + j] * s[j];
      const dot = (a, b) => M[a * 3] * M[b * 3] + M[a * 3 + 1] * M[b * 3 + 1] + M[a * 3 + 2] * M[b * 3 + 2];
      this.sig[g * 6] = dot(0, 0);
      this.sig[g * 6 + 1] = dot(0, 1);
      this.sig[g * 6 + 2] = dot(0, 2);
      this.sig[g * 6 + 3] = dot(1, 1);
      this.sig[g * 6 + 4] = dot(1, 2);
      this.sig[g * 6 + 5] = dot(2, 2);
    }
  }
}

/* World blobs to screen blobs. Returns flat arrays plus the depth order. */
export function project(cloud, cam, res, fovDeg) {
  const n = cloud.n;
  const f = (0.5 * res) / Math.tan((fovDeg * Math.PI) / 360);
  const [rr, dd, ff] = cam.basis;
  const P = cam.pos;
  const zmin = cloud.cfg.z_min;
  const out = {
    x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n),
    i00: new Float32Array(n), i01: new Float32Array(n), i11: new Float32Array(n),
    s00: new Float32Array(n), s01: new Float32Array(n), s11: new Float32Array(n),
    rad: new Float32Array(n), keep: new Uint8Array(n),
    cr: new Float32Array(n), cg: new Float32Array(n), cb: new Float32Array(n),
    alpha: new Float32Array(n),
  };
  const order = [];
  for (let g = 0; g < n; g++) {
    const wx = cloud.mu[g * 3] - P[0];
    const wy = cloud.mu[g * 3 + 1] - P[1];
    const wz = cloud.mu[g * 3 + 2] - P[2];
    const cx = rr[0] * wx + rr[1] * wy + rr[2] * wz;
    const cy = dd[0] * wx + dd[1] * wy + dd[2] * wz;
    const cz = ff[0] * wx + ff[1] * wy + ff[2] * wz;
    out.z[g] = cz;
    if (cz <= zmin) continue;
    const z = cz;
    out.x[g] = 0.5 * res + (f * cx) / z;
    out.y[g] = 0.5 * res + (f * cy) / z;
    /* M = J W, two rows of three */
    const j00 = f / z;
    const j02 = (-f * cx) / (z * z);
    const j12 = (-f * cy) / (z * z);
    const m0 = [j00 * rr[0] + j02 * ff[0], j00 * rr[1] + j02 * ff[1], j00 * rr[2] + j02 * ff[2]];
    const m1 = [j00 * dd[0] + j12 * ff[0], j00 * dd[1] + j12 * ff[1], j00 * dd[2] + j12 * ff[2]];
    const S = cloud.sig;
    const o = g * 6;
    const sxx = S[o]; const sxy = S[o + 1]; const sxz = S[o + 2];
    const syy = S[o + 3]; const syz = S[o + 4]; const szz = S[o + 5];
    const mul = (m) => [
      sxx * m[0] + sxy * m[1] + sxz * m[2],
      sxy * m[0] + syy * m[1] + syz * m[2],
      sxz * m[0] + syz * m[1] + szz * m[2],
    ];
    const a0 = mul(m0);
    const a1 = mul(m1);
    let A = m0[0] * a0[0] + m0[1] * a0[1] + m0[2] * a0[2] + cloud.cfg.dilate;
    let B = m0[0] * a1[0] + m0[1] * a1[1] + m0[2] * a1[2];
    let C = m1[0] * a1[0] + m1[1] * a1[1] + m1[2] * a1[2] + cloud.cfg.dilate;
    const det = Math.max(A * C - B * B, 1e-8);
    out.s00[g] = A; out.s01[g] = B; out.s11[g] = C;
    out.i00[g] = C / det;
    out.i01[g] = -B / det;
    out.i11[g] = A / det;
    const tr = 0.5 * (A + C);
    const root = Math.sqrt(Math.max(tr * tr - det, 0));
    out.rad[g] = 3 * Math.sqrt(Math.max(tr + root, 1e-8));
    out.alpha[g] = 1 / (1 + Math.exp(-cloud.op[g]));
    /* colour is evaluated once per blob at its own view direction, exactly as
       the spherical harmonics are in the paper */
    const L = Math.hypot(wx, wy, wz) || 1e-12;
    const dx = wx / L; const dy = wy / L; const dz = wz / L;
    for (let k = 0; k < 3; k++) {
      let raw = cloud.c0[g * 3 + k];
      if (cloud.viewdep) {
        raw += cloud.c1[g * 9 + k * 3] * dx + cloud.c1[g * 9 + k * 3 + 1] * dy
             + cloud.c1[g * 9 + k * 3 + 2] * dz;
      }
      const v = 1 / (1 + Math.exp(-raw));
      if (k === 0) out.cr[g] = v;
      else if (k === 1) out.cg[g] = v;
      else out.cb[g] = v;
    }
    out.keep[g] = 1;
    order.push(g);
  }
  order.sort((p, q) => out.z[p] - out.z[q]);
  out.order = order;
  return out;
}

/* Alpha composite the projected blobs. `only` is an optional Uint8Array mask
   over blob indices; `weightOut` collects each blob's total contribution,
   which is what the "how many blobs carry this frame" slider is built on. */
export function rasterise(cloud, proj, res, {
  bg = [0.945, 0.953, 0.953], only = null, weightOut = null, stopAt = 1e-5,
} = {}) {
  const t0 = performance.now();
  const N = res * res;
  const T = new Float32Array(N).fill(1);
  const acc = new Float32Array(N);
  const rgb = new Float32Array(N * 3);
  /* expected depth, the same quantity the field reports, so the two
     representations can be asked the same question about where they put the
     surface and not only about what colour they made it */
  const dsum = new Float32Array(N);
  const cut = cloud.cfg.cutoff2;
  const amax = cloud.cfg.alpha_max;
  let touched = 0;   /* inside the cutoff, which is what the generator counts */
  let shaded = 0;    /* and actually composited, which is fewer once a pixel is opaque */
  let drawn = 0;
  for (let k = 0; k < proj.order.length; k++) {
    const g = proj.order[k];
    if (only && !only[g]) continue;
    const cx = proj.x[g];
    const cy = proj.y[g];
    const r = proj.rad[g];
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(res - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(res - 1, Math.ceil(cy + r));
    if (x1 < x0 || y1 < y0) continue;
    drawn++;
    const i00 = proj.i00[g];
    const i01 = proj.i01[g];
    const i11 = proj.i11[g];
    const op = proj.alpha[g];
    const cr = proj.cr[g];
    const cg = proj.cg[g];
    const cb = proj.cb[g];
    let wsum = 0;
    for (let py = y0; py <= y1; py++) {
      const dy = py + 0.5 - cy;
      for (let px = x0; px <= x1; px++) {
        const dx = px + 0.5 - cx;
        const m = i00 * dx * dx + 2 * i01 * dx * dy + i11 * dy * dy;
        if (m > cut) continue;
        const pix = py * res + px;
        touched++;
        if (T[pix] < stopAt) continue;
        shaded++;
        let a = op * Math.exp(-0.5 * m);
        if (a > amax) a = amax;
        const w = T[pix] * a;
        rgb[pix * 3] += w * cr;
        rgb[pix * 3 + 1] += w * cg;
        rgb[pix * 3 + 2] += w * cb;
        acc[pix] += w;
        dsum[pix] += w * proj.z[g];
        T[pix] *= (1 - a + 1e-10);
        wsum += w;
      }
    }
    if (weightOut) weightOut[g] = wsum;
  }
  const depth = new Float32Array(N);
  for (let p = 0; p < N; p++) {
    const rest = 1 - acc[p];
    rgb[p * 3] += rest * bg[0];
    rgb[p * 3 + 1] += rest * bg[1];
    rgb[p * 3 + 2] += rest * bg[2];
    depth[p] = acc[p] > 1e-6 ? dsum[p] / acc[p] : 0;
  }
  return { rgb, depth, alpha: acc, ms: performance.now() - t0, touched, shaded, drawn,
    perPixel: touched / N };
}

export function renderSplats(cloud, cam, res, fovDeg, opts = {}) {
  const proj = project(cloud, cam, res, fovDeg);
  const r = rasterise(cloud, proj, res, opts);
  r.proj = proj;
  return r;
}

/* The K blobs that carry most of this frame, measured rather than guessed. */
export function topBlobs(cloud, proj, res, k, opts = {}) {
  const w = new Float32Array(cloud.n);
  rasterise(cloud, proj, res, { ...opts, weightOut: w });
  const idx = proj.order.slice().sort((a, b) => w[b] - w[a]);
  const mask = new Uint8Array(cloud.n);
  let carried = 0;
  let total = 0;
  for (let i = 0; i < w.length; i++) total += w[i];
  for (let i = 0; i < Math.min(k, idx.length); i++) { mask[idx[i]] = 1; carried += w[idx[i]]; }
  return { mask, carried, total, share: total > 0 ? carried / total : 0 };
}

/* The 2 sigma outline of a projected blob, for the view that shows the
   ellipses instead of the picture they add up to. */
export function ellipse(proj, g, sigmas = 2) {
  const a = proj.s00[g];
  const b = proj.s01[g];
  const c = proj.s11[g];
  const tr = 0.5 * (a + c);
  const det = a * c - b * b;
  const root = Math.sqrt(Math.max(tr * tr - det, 0));
  const l1 = tr + root;
  const l2 = Math.max(tr - root, 1e-8);
  const theta = 0.5 * Math.atan2(2 * b, a - c);
  return { cx: proj.x[g], cy: proj.y[g], rx: sigmas * Math.sqrt(l1), ry: sigmas * Math.sqrt(l2), theta };
}
