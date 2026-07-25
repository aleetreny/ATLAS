/* Robust regression, from scratch.
 *
 * Mirrors src/models/robust_regression.py (ScratchRANSAC, ScratchTheilSen,
 * ScratchHuber) with two deliberate upgrades, both of which the article
 * explains rather than hides:
 *
 *  - RANSAC records every iteration it tries, so the page can replay the
 *    search instead of only showing its answer.
 *  - Huber is solved by iteratively reweighted least squares with an estimated
 *    scale, not by fixed-step gradient descent on raw units. A fixed delta on
 *    unscaled data plus an under-converged optimizer produces a flat line that
 *    looks like a property of the loss but is really a property of the solver,
 *    and that confound would undermine the comparison this article makes.
 */

/* ------------------------------------------------------------------ basics */

/* Ordinary least squares through (x, y). Returns {slope, intercept}. */
export function ols(xs, ys) {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  return { slope, intercept: my - slope * mx };
}

/* Weighted least squares, the engine underneath IRLS. */
export function wls(xs, ys, ws) {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const mx = sx / sw;
  const my = sy / sw;
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += ws[i] * (xs[i] - mx) * (ys[i] - my);
    den += ws[i] * (xs[i] - mx) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  return { slope, intercept: my - slope * mx };
}

export function median(arr) {
  if (!arr.length) return NaN;
  const a = [...arr].sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/* Median absolute deviation, rescaled to estimate a Gaussian sigma. */
export function mad(arr) {
  const m = median(arr);
  return median(arr.map((v) => Math.abs(v - m))) / 0.6745;
}

/* ------------------------------------------------------------- theil-sen */

/* Median of the slopes of every pair of points. Parameter free.
 * Also returns the full slope list, because seeing that distribution IS the
 * explanation of why the estimator resists outliers. */
export function theilSen(xs, ys) {
  const slopes = [];
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      if (xs[j] !== xs[i]) slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
    }
  }
  const slope = median(slopes);
  const intercept = median(xs.map((x, i) => ys[i] - slope * x));
  return { slope, intercept, slopes };
}

/* ---------------------------------------------------------------- ransac */

/* mulberry32: seeded, so a replay of the search is identical every time. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Hypothesize, verify, count, repeat, refine.
 * Returns the winning model plus the full history, one entry per iteration. */
export function ransac(xs, ys, { threshold, iterations = 200, seed = 7 } = {}) {
  const rand = rng(seed);
  const n = xs.length;
  const history = [];
  let best = null;

  for (let it = 0; it < iterations; it++) {
    // 1. hypothesize: the smallest sample that defines a line
    let i = Math.floor(rand() * n);
    let j = Math.floor(rand() * n);
    let guard = 0;
    while ((j === i || xs[j] === xs[i]) && guard++ < 50) j = Math.floor(rand() * n);
    if (xs[j] === xs[i]) continue;

    const slope = (ys[j] - ys[i]) / (xs[j] - xs[i]);
    const intercept = ys[i] - slope * xs[i];

    // 2 and 3. verify against everyone, and count who agrees
    const inliers = [];
    for (let k = 0; k < n; k++) {
      if (Math.abs(ys[k] - (intercept + slope * xs[k])) < threshold) inliers.push(k);
    }

    const entry = { it, sample: [i, j], slope, intercept, count: inliers.length, inliers, isBest: false };
    if (!best || inliers.length > best.count) {
      best = entry;
      entry.isBest = true;
    }
    history.push(entry);
  }

  if (!best) return { slope: 0, intercept: 0, inliers: [], history };

  // 5. refine: the two random points chose the committee, the whole committee
  // chooses the line
  const inX = best.inliers.map((k) => xs[k]);
  const inY = best.inliers.map((k) => ys[k]);
  const refined = ols(inX, inY);
  return { ...refined, inliers: best.inliers, best, history };
}

/* ----------------------------------------------------------------- huber */

export function huberWeight(r, k) {
  const a = Math.abs(r);
  return a <= k ? 1 : k / a;
}

/* Huber M-estimation by IRLS with a concurrently estimated scale, which is
 * what makes delta meaningful regardless of the units y happens to be in.
 * delta is in units of sigma (sklearn calls it epsilon; 1.35 is the classic
 * value giving 95% efficiency against Gaussian noise). */
export function huber(xs, ys, { delta = 1.35, maxIter = 200, tol = 1e-10 } = {}) {
  let { slope, intercept } = ols(xs, ys);
  let scale = 1;
  for (let it = 0; it < maxIter; it++) {
    const res = xs.map((x, i) => ys[i] - (intercept + slope * x));
    scale = Math.max(mad(res), 1e-9);
    const ws = res.map((r) => huberWeight(r / scale, delta));
    const next = wls(xs, ys, ws);
    const moved = Math.abs(next.slope - slope) + Math.abs(next.intercept - intercept);
    slope = next.slope;
    intercept = next.intercept;
    if (moved < tol) break;
  }
  const res = xs.map((x, i) => ys[i] - (intercept + slope * x));
  return { slope, intercept, scale, weights: res.map((r) => huberWeight(r / scale, delta)) };
}

/* ------------------------------------------------------------- diagnostics */

/* Hat matrix diagonal for simple regression: how much pull each x position has
 * on the fit, independent of its y value. This is what separates a point that
 * is merely unusual from one that is dangerous. */
export function leverage(xs) {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const sxx = xs.reduce((s, v) => s + (v - mx) ** 2, 0);
  return xs.map((x) => 1 / n + (x - mx) ** 2 / sxx);
}

/* Loss functions, for the comparison plot. r is a raw residual. */
export const losses = {
  squared: (r) => 0.5 * r * r,
  absolute: (r) => Math.abs(r),
  huber: (r, d) => (Math.abs(r) <= d ? 0.5 * r * r : d * (Math.abs(r) - 0.5 * d)),
};

/* Derivative magnitude, i.e. how hard a residual pulls on the fit. The whole
 * robustness story is that squared loss lets this grow without bound. */
export const pulls = {
  squared: (r) => Math.abs(r),
  absolute: () => 1,
  huber: (r, d) => Math.min(Math.abs(r), d),
};

export function rmse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return Math.sqrt(s / yTrue.length);
}
