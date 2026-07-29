/* The diffusion process, run in the page.
 *
 * Two things live here and they are deliberately separate:
 *
 *   the TRUTH, which the page recomputes in closed form. q(x_t) is the toy
 *   density scaled by sqrt(abar) and blurred by a gaussian of variance
 *   1 - abar, and both halves survive that exactly. The two blobs are
 *   gaussians convolved with a gaussian, which is a gaussian, so they are four
 *   lines of arithmetic. The ring is rotationally symmetric, so its
 *   convolution is too, and the generator ships one radial profile per noise
 *   level rather than a picture of the field: 400 numbers instead of 4,096, and
 *   what gets drawn is the function rather than a resampling of it.
 *
 *   the MODEL, which is the score network the generator trained, read weight
 *   for weight out of the same data file at float16.
 *
 * Keeping them apart is the whole point of the article: the page can draw the
 * field the network is trying to learn beside the field it learned, and
 * subtract them. Nobody doing this on images can.
 */
import { byName, decodeWeights, dense, lcg } from '../../assets/js/halfweights.js';

const TAU = Math.PI * 2;

export function makeDDPM(data) {
  const N = data.net;
  const W = decodeWeights(N.b64, N.count);
  const L = byName(N.spec);
  const abarAll = N.abar_all;
  const steps = N.steps;
  const silu = (v) => v.map((t) => t / (1 + Math.exp(-t)));

  /* the ring's radial profile at each level the page can show, unpacked into
     one array per level so a lookup is an index rather than a search */
  const rings = [];
  const ringDs = [];
  {
    const nAll = N.rho_n * N.page_ts.length;
    const flat = decodeWeights(N.ring_b64, nAll);
    const flatD = decodeWeights(N.ringd_b64, nAll);
    for (let i = 0; i < N.page_ts.length; i++) {
      rings.push(flat.subarray(i * N.rho_n, (i + 1) * N.rho_n));
      ringDs.push(flatD.subarray(i * N.rho_n, (i + 1) * N.rho_n));
    }
  }
  const dRho = N.rho_max / (N.rho_n - 1);

  /* ------------------------------------------------------------ the truth */
  const TOY = {
    ring: { w: 0.5, r0: 2.0, sigma: 0.16 },
    blobs: [
      { w: 0.25, mean: [-2.55, 2.55], sd: [0.35, 0.35], angle: 0.0 },
      { w: 0.25, mean: [2.55, -2.55], sd: [0.62, 0.17], angle: 35.0 },
    ],
  };

  function blobCov(b, a, v) {
    const th = (b.angle * Math.PI) / 180;
    const c = Math.cos(th);
    const s = Math.sin(th);
    /* Q diag(sd^2) Q', scaled by abar, then the blur added to the diagonal */
    const s0 = b.sd[0] * b.sd[0];
    const s1 = b.sd[1] * b.sd[1];
    const xx = a * (c * c * s0 + s * s * s1) + v;
    const yy = a * (s * s * s0 + c * c * s1) + v;
    const xy = a * (c * s * s0 - s * c * s1);
    const det = xx * yy - xy * xy;
    return { xx, yy, xy, det, ixx: yy / det, iyy: xx / det, ixy: -xy / det };
  }

  /* the ring's profile, and its derivative in the radius, by interpolating the
     shipped samples; the derivative is what the score needs along the radius */
  function ringAt(level, rho) {
    const p = rings[level];
    const q = ringDs[level];
    const u = rho / dRho;
    if (u >= N.rho_n - 1) return [0, 0];
    const i = Math.floor(u);
    const f = u - i;
    /* Both the value and the SLOPE are interpolated from shipped arrays. The
       slope is not recovered by differencing the value: doing that turns a
       smooth profile into a staircase, and the page's score came out 168%
       wrong while its density was only 6% wrong. The generator has the exact
       derivative from its spline, so it ships it. */
    return [p[i] * (1 - f) + p[i + 1] * f, q[i] * (1 - f) + q[i + 1] * f];
  }

  /** q(x_t) and grad log q(x_t), exactly, at one point. `level` indexes
   *  net.page_ts rather than being a raw t, because that is what the profile
   *  was shipped for. */
  function truthAt(level, x, y) {
    const a = N.abar[level];
    const v = Math.max(1 - a, 1e-12);
    const sa = Math.sqrt(a);
    let dens = 0;
    let gx = 0;
    let gy = 0;
    for (const b of TOY.blobs) {
      const C = blobCov(b, a, v);
      const dx = x - sa * b.mean[0];
      const dy = y - sa * b.mean[1];
      const qx = C.ixx * dx + C.ixy * dy;
      const qy = C.ixy * dx + C.iyy * dy;
      const val = (b.w * Math.exp(-0.5 * (dx * qx + dy * qy))) / (TAU * Math.sqrt(C.det));
      dens += val;
      gx -= qx * val;
      gy -= qy * val;
    }
    const rho = Math.hypot(x, y);
    const [rv, rd] = ringAt(level, rho);
    dens += TOY.ring.w * rv;
    if (rho > 1e-9) {
      gx += (TOY.ring.w * rd * x) / rho;
      gy += (TOY.ring.w * rd * y) / rho;
    }
    const d = Math.max(dens, 1e-300);
    return { q: dens, sx: gx / d, sy: gy / d };
  }

  /* ------------------------------------------------------------ the model */
  function embed(t) {
    /* the same sinusoidal embedding the generator trained with, written from
       the same definition rather than sampled and shipped */
    const out = new Float64Array(16);
    for (let i = 0; i < 8; i++) {
      const f = Math.exp((6.0 * i) / 7.0);
      const a = (t / steps) * f;
      out[i] = Math.sin(a);
      out[i + 8] = Math.cos(a);
    }
    return out;
  }

  /** eps_theta(x, t) for one point. `dense` in the shared decoder takes one
   *  vector at a time, which is the right granularity here: the widgets call
   *  this on a few hundred points, not a few hundred thousand. */
  function epsOne(x, y, t) {
    const e = embed(t);
    const inp = new Float64Array(18);
    inp[0] = x;
    inp[1] = y;
    for (let k = 0; k < 16; k++) inp[2 + k] = e[k];
    let h = silu(dense(W, L.l0, inp));
    h = silu(dense(W, L.l1, h));
    h = silu(dense(W, L.l2, h));
    return dense(W, L.l3, h);
  }

  /** The same for a batch laid out as [x0,y0, x1,y1, ...]. */
  function epsBatch(pts, t) {
    const n = pts.length / 2;
    const out = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const e = epsOne(pts[i * 2], pts[i * 2 + 1], t);
      out[i * 2] = e[0];
      out[i * 2 + 1] = e[1];
    }
    return out;
  }

  /** The score the network implies: eps and the score differ by a known
   *  factor, which is exactly why the two fields can be put on one chart. */
  function learnedAt(level, x, y) {
    const t = N.page_ts[level];
    const v = Math.max(1 - N.abar[level], 1e-12);
    const e = epsOne(x, y, t);
    return { sx: -e[0] / Math.sqrt(v), sy: -e[1] / Math.sqrt(v) };
  }

  /* ------------------------------------------------------- the reverse chain */
  /**
   * Run the reverse process and return every state it passed through, so the
   * page can scrub the chain rather than only show where it ended.
   *
   * `mode` is 'ancestral' (the original, which adds noise back at each step) or
   * 'deterministic' (the same trained network with that term removed). They are
   * two samplers for one model, not two models.
   */
  function reverse(n, stepsUsed, seed, mode = 'deterministic') {
    const rnd = lcg(seed);
    let x = new Float64Array(n * 2);
    for (let i = 0; i < n * 2; i++) x[i] = rnd.normal();
    const idx = [];
    for (let j = 0; j < stepsUsed; j++) {
      idx.push(Math.round(steps - 1 - (j * (steps - 1)) / (stepsUsed - 1)));
    }
    const frames = [Float64Array.from(x)];
    for (let j = 0; j < idx.length; j++) {
      const t = idx[j];
      const aT = abarAll[t];
      const aPrev = j + 1 < idx.length ? abarAll[idx[j + 1]] : 1.0;
      const eps = epsBatch(x, t);
      const nx = new Float64Array(n * 2);
      const sT = Math.sqrt(aT);
      const sN = Math.sqrt(Math.max(1 - aT, 0));
      const sP = Math.sqrt(aPrev);
      const sPN = Math.sqrt(Math.max(1 - aPrev, 0));
      for (let i = 0; i < n * 2; i++) {
        const x0 = (x[i] - sN * eps[i]) / sT;
        if (mode === 'ancestral' && aPrev < 1.0) {
          const varr = ((1 - aPrev) / (1 - aT)) * (1 - aT / aPrev);
          const mean = ((sP * (1 - aT / aPrev)) / (1 - aT)) * x0
            + ((Math.sqrt(aT / aPrev) * (1 - aPrev)) / (1 - aT)) * x[i];
          nx[i] = mean + (varr > 0 ? Math.sqrt(varr) * rnd.normal() : 0);
        } else {
          nx[i] = sP * x0 + sPN * eps[i];
        }
      }
      x = nx;
      frames.push(Float64Array.from(x));
    }
    return { frames, idx };
  }

  return { truthAt, learnedAt, epsOne, epsBatch, reverse,
    levels: N.page_ts, abar: N.abar };
}
