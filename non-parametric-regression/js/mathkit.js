/* Non-parametric regression, from scratch.
 *
 * Mirrors src/models/non_parametric_regression.py.
 *
 *  - Local linear regression is parameterised around the query point. That is a
 *    conditioning choice, not a correctness one: the prediction is identical to
 *    fitting in raw x and evaluating at the query, to about 1e-14. What removes
 *    the boundary bias is the tilt, not where the origin sits.
 *  - The Gaussian process is solved by Cholesky factorisation of K + sigma^2 I
 *    and returns a posterior variance, so the page can draw what the model does
 *    not know rather than only what it predicts.
 */

/* -------------------------------------------------------------------- knn */

/* Lazy learner: there is no fit, only a lookup. weights 'uniform' averages the
 * k nearest, 'distance' weights them by 1/d so the closest neighbour dominates. */
export function knnPredict(xt, yt, xq, k, weights = 'uniform') {
  const n = xt.length;
  const kk = Math.max(1, Math.min(k, n));
  return xq.map((q) => {
    // partial selection is enough: we only need the kk smallest distances
    const d = new Array(n);
    for (let i = 0; i < n; i++) d[i] = [Math.abs(xt[i] - q), i];
    d.sort((a, b) => a[0] - b[0]);
    if (weights === 'uniform') {
      let s = 0;
      for (let i = 0; i < kk; i++) s += yt[d[i][1]];
      return s / kk;
    }
    let num = 0;
    let den = 0;
    for (let i = 0; i < kk; i++) {
      const w = 1 / Math.max(d[i][0], 1e-10);
      num += w * yt[d[i][1]];
      den += w;
    }
    return num / den;
  });
}

/* Which points a query actually consulted, for the "show me the neighbourhood"
 * annotation. */
export function knnNeighbours(xt, xq, k) {
  const d = xt.map((v, i) => [Math.abs(v - xq), i]).sort((a, b) => a[0] - b[0]);
  return d.slice(0, Math.max(1, Math.min(k, xt.length))).map((p) => p[1]);
}

/* ------------------------------------------------------------ kernel smoothers */

export const gaussianKernel = (d, h) => Math.exp(-0.5 * (d / h) ** 2);

/* Nadaraya-Watson: a kernel-weighted average. Locally CONSTANT, which is the
 * source of its boundary problem: at the edge of the data every neighbour lies
 * on one side, and an average of them cannot represent a trend. */
export function nadarayaWatson(xt, yt, xq, h) {
  const fallback = yt.reduce((a, b) => a + b, 0) / yt.length;
  return xq.map((q) => {
    let num = 0;
    let den = 0;
    for (let i = 0; i < xt.length; i++) {
      const w = gaussianKernel(xt[i] - q, h);
      num += w * yt[i];
      den += w;
    }
    return den < 1e-12 ? fallback : num / den;
  });
}

/* Local linear: a kernel-weighted straight line fitted around each query, with
 * x measured RELATIVE to the query. The intercept of that local line is then
 * the prediction, and because the line can tilt, a one-sided neighbourhood at
 * the boundary is no longer a problem. */
export function localLinear(xt, yt, xq, h) {
  const fallback = yt.reduce((a, b) => a + b, 0) / yt.length;
  return xq.map((q) => {
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let t0 = 0;
    let t1 = 0;
    for (let i = 0; i < xt.length; i++) {
      const dx = xt[i] - q;
      const w = gaussianKernel(dx, h);
      s0 += w;
      s1 += w * dx;
      s2 += w * dx * dx;
      t0 += w * yt[i];
      t1 += w * dx * yt[i];
    }
    if (s0 < 1e-12) return fallback;
    const det = s0 * s2 - s1 * s1;
    if (Math.abs(det) < 1e-12) return t0 / s0;
    return (t0 * s2 - t1 * s1) / det;
  });
}

/* The weights a single query assigns, for drawing the moving window. */
export function kernelWeights(xt, q, h) {
  return xt.map((v) => gaussianKernel(v - q, h));
}

/* --------------------------------------------------------- gaussian process */

export const kernels = {
  rbf: (a, b, l) => Math.exp(-0.5 * ((a - b) / l) ** 2),
  /* Matern 3/2: same length scale, far less smooth. The choice of kernel is a
   * statement about how the world behaves between observations. */
  matern: (a, b, l) => {
    const r = (Math.sqrt(3) * Math.abs(a - b)) / l;
    return (1 + r) * Math.exp(-r);
  },
  periodic: (a, b, l) => Math.exp(-2 * Math.sin((Math.PI * Math.abs(a - b)) / (2 * l)) ** 2),
};

/* Cholesky of a symmetric positive definite matrix. */
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-12));
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

function forwardSub(L, b) {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  return y;
}

function backSub(L, y) {
  const n = L.length;
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/* Posterior mean and standard deviation of a GP at the query points.
 * Returns predictions in the original units of y. */
export function gpPredict(xt, yt, xq, { kernel = 'rbf', length = 1, noise = 0.1, amplitude = 1 } = {}) {
  const n = xt.length;
  if (!n) return { mean: xq.map(() => 0), sd: xq.map(() => amplitude) };
  const kf = kernels[kernel] || kernels.rbf;
  const my = yt.reduce((a, b) => a + b, 0) / n;
  const yc = yt.map((v) => v - my);

  const K = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => amplitude * amplitude * kf(xt[i], xt[j], length) + (i === j ? noise * noise : 0))
  );
  const L = cholesky(K);
  const alpha = backSub(L, forwardSub(L, yc));

  const mean = [];
  const sd = [];
  for (const q of xq) {
    const ks = xt.map((v) => amplitude * amplitude * kf(v, q, length));
    let m = 0;
    for (let i = 0; i < n; i++) m += ks[i] * alpha[i];
    mean.push(m + my);
    const v = forwardSub(L, ks);
    let dot = 0;
    for (let i = 0; i < n; i++) dot += v[i] * v[i];
    sd.push(Math.sqrt(Math.max(amplitude * amplitude - dot, 1e-12)));
  }
  return { mean, sd };
}

/* ----------------------------------------------------------------- helpers */

export function rmse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return Math.sqrt(s / yTrue.length);
}

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
