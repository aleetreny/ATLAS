/* A linear state space layer, both ways, in the browser.
 *
 * The article's first claim is that these two functions are the same function:
 *
 *   the recurrence   h[t] = a h[t-1] + b x[t],   y[t] = c h[t]
 *   the convolution  y = k * x,   k[t] = c a^t b
 *
 * That is not a resemblance to be illustrated, it is an identity to be checked,
 * so the page computes both from the same parameters and reports the worst
 * disagreement. Anything above the accumulated rounding of a few hundred
 * additions means one of the two is wrong.
 *
 * Which matters practically and not just aesthetically: the convolution can be
 * evaluated for every position at once, so training parallelises over the
 * sequence, while the recurrence needs constant memory and one step of work per
 * new token, so generation is cheap. Being able to switch between them is the
 * whole reason these models exist.
 */
import { decodeHalf } from '../../assets/js/textkit.js';

export function loadDemo(D) {
  return {
    n: D.n,
    L: D.L,
    a: decodeHalf(D.a, D.n),
    b: decodeHalf(D.b, D.n),
    c: decodeHalf(D.c, D.n),
    x: decodeHalf(D.x, D.L),
    kernelRef: D.kernel_ref,
    yRef: D.y_ref,
  };
}

/** Step by step, carrying a state of n numbers and nothing else. */
export function recurrent(a, b, c, x) {
  const n = a.length;
  const h = new Float64Array(n);
  const out = new Float64Array(x.length);
  const trace = [];
  for (let t = 0; t < x.length; t++) {
    let y = 0;
    for (let i = 0; i < n; i++) {
      h[i] = a[i] * h[i] + b[i] * x[t];
      y += c[i] * h[i];
    }
    out[t] = y;
    if (t < 64) trace.push(Array.from(h));
  }
  return { y: out, trace };
}

/** The kernel of the equivalent convolution: k[t] = sum_i c_i a_i^t b_i. */
export function kernel(a, b, c, L) {
  const n = a.length;
  const pow = new Float64Array(n).fill(1);
  const k = new Float64Array(L);
  for (let t = 0; t < L; t++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      s += c[i] * pow[i] * b[i];
      pow[i] *= a[i];
    }
    k[t] = s;
  }
  return k;
}

/** y = k * x, causal, which is the same output computed all at once. */
export function convolve(k, x) {
  const out = new Float64Array(x.length);
  for (let t = 0; t < x.length; t++) {
    let s = 0;
    const m = Math.min(t + 1, k.length);
    for (let j = 0; j < m; j++) s += k[j] * x[t - j];
    out[t] = s;
  }
  return out;
}

/**
 * The worst disagreement between the two, relative to the size of the output.
 *
 * Relative rather than absolute, because the output scale depends entirely on
 * how slow the dynamics are: a decay of 0.999 accumulates hundreds of terms and
 * produces much larger numbers than a decay of 0.5, and an absolute tolerance
 * would be measuring that instead of the implementations.
 */
export function dualityError(a, b, c, x) {
  const yr = recurrent(a, b, c, x).y;
  const yc = convolve(kernel(a, b, c, x.length), x);
  let worst = 0;
  let scale = 1e-12;
  for (let t = 0; t < yr.length; t++) scale = Math.max(scale, Math.abs(yr[t]));
  for (let t = 0; t < yr.length; t++) worst = Math.max(worst, Math.abs(yr[t] - yc[t]));
  return { worst: worst / scale, y: yr, yc, scale };
}

/** How many steps until the kernel is below a fraction of where it started. */
export function reachOf(k, frac = 0.01) {
  const k0 = Math.abs(k[0]) || 1e-30;
  for (let t = 0; t < k.length; t++) {
    if (Math.abs(k[t]) / k0 < frac) return t;
  }
  return null;
}
