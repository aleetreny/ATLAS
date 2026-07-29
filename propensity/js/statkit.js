/* The five estimators, in the browser, on the sample the page draws.
 *
 * They are here rather than inside a widget because main.js checks them against
 * the values the generator computed on the same rows: if the page draws a
 * cloud and prints an estimate underneath it, the estimate has to be the one
 * those points produce, and the only way to know is to compute it twice.
 */

/* Least squares by Householder QR.
 *
 * Not by solving the normal equations, which is a standing rule in this atlas
 * since a polynomial fit returned silently wrong coefficients: forming the
 * cross-product matrix squares the condition number. With five columns it
 * would have been fine, and writing the safe one costs thirty lines that never
 * have to be revisited.
 *
 * `A` is an array of rows, `b` the right-hand side. Returns the coefficients.
 */
export function lstsq(A, b) {
  const m = A.length;
  const n = A[0].length;
  const R = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] ** 2;
    norm = Math.sqrt(norm);
    if (norm < 1e-14) continue;
    const alpha = R[k][k] > 0 ? -norm : norm;
    const v = new Array(m).fill(0);
    v[k] = R[k][k] - alpha;
    for (let i = k + 1; i < m; i++) v[i] = R[i][k];
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] ** 2;
    if (vv < 1e-30) continue;
    for (let j = k; j <= n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += v[i] * R[i][j];
      const f = (2 * dot) / vv;
      for (let i = k; i < m; i++) R[i][j] -= f * v[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = R[i][n];
    for (let j = i + 1; j < n; j++) s -= R[i][j] * x[j];
    x[i] = Math.abs(R[i][i]) < 1e-14 ? 0 : s / R[i][i];
  }
  return x;
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

export function naive(X, Y) {
  return mean(Y.filter((_, i) => X[i] === 1)) - mean(Y.filter((_, i) => X[i] === 0));
}

/* Regression with the covariates in: intercept, treatment, then the columns. */
export function regression(V, X, Y) {
  const A = X.map((x, i) => [1, x, ...V[i]]);
  return lstsq(A, Y)[1];
}

/* Horvitz-Thompson, stabilised: each unit weighted by one over the probability
 * of the treatment it actually received, and each arm renormalised by its own
 * weights rather than by the sample size. */
export function ipw(X, Y, e) {
  let n1 = 0;
  let d1 = 0;
  let n0 = 0;
  let d0 = 0;
  for (let i = 0; i < Y.length; i++) {
    if (X[i] === 1) { n1 += Y[i] / e[i]; d1 += 1 / e[i]; } else { n0 += Y[i] / (1 - e[i]); d0 += 1 / (1 - e[i]); }
  }
  return n1 / d1 - n0 / d0;
}

export function strata(X, Y, e, k = 5) {
  const sorted = [...e].sort((a, b) => a - b);
  const edges = d3.range(1, k).map((i) => sorted[Math.floor((i * sorted.length) / k)]);
  let num = 0;
  let den = 0;
  for (let s = 0; s < k; s++) {
    const idx = [];
    for (let i = 0; i < e.length; i++) {
      const bin = d3.bisectRight(edges, e[i]);
      if (bin === s) idx.push(i);
    }
    const a = idx.filter((i) => X[i] === 1).map((i) => Y[i]);
    const b = idx.filter((i) => X[i] === 0).map((i) => Y[i]);
    if (a.length < 2 || b.length < 2) continue;
    num += idx.length * (mean(a) - mean(b));
    den += idx.length;
  }
  return den ? num / den : NaN;
}

/* Nearest neighbour on the logit of the score, with replacement, averaged over
 * both directions so that the estimand is the whole population and not just
 * the treated. */
export function matching(X, Y, e) {
  const lg = e.map((v) => Math.log(Math.min(Math.max(v, 1e-9), 1 - 1e-9)
    / Math.min(Math.max(1 - v, 1e-9), 1 - 1e-9)));
  const t = [];
  const c = [];
  X.forEach((v, i) => (v === 1 ? t : c).push(i));
  const nearest = (from, pool) => {
    let best = pool[0];
    let bd = Infinity;
    for (const j of pool) {
      const d = Math.abs(lg[from] - lg[j]);
      if (d < bd) { bd = d; best = j; }
    }
    return best;
  };
  const att = mean(t.map((i) => Y[i] - Y[nearest(i, c)]));
  const atc = mean(c.map((i) => Y[nearest(i, t)] - Y[i]));
  return (att * t.length + atc * c.length) / X.length;
}

/* The augmented estimator: an outcome model per arm, plus a weighted
 * correction on its residuals. */
export function aipw(V, X, Y, e) {
  const fit = (want) => {
    const rowsA = [];
    const rowsB = [];
    for (let i = 0; i < Y.length; i++) {
      if (X[i] === want) { rowsA.push([1, ...V[i]]); rowsB.push(Y[i]); }
    }
    const beta = lstsq(rowsA, rowsB);
    return V.map((v) => [1, ...v].reduce((s, u, j) => s + u * beta[j], 0));
  };
  const mu1 = fit(1);
  const mu0 = fit(0);
  let acc = 0;
  for (let i = 0; i < Y.length; i++) {
    const a = mu1[i] + (X[i] === 1 ? (Y[i] - mu1[i]) / e[i] : 0);
    const b = mu0[i] + (X[i] === 0 ? (Y[i] - mu0[i]) / (1 - e[i]) : 0);
    acc += a - b;
  }
  return acc / Y.length;
}

/* Standardised difference of means, with the pooled deviation in the
 * denominator, optionally weighted. It is the diagnostic the whole balance
 * section is about, so it is written once here and used by both the widget and
 * the guard. */
export function smd(V, X, w = null) {
  const p = V[0].length;
  const out = [];
  for (let j = 0; j < p; j++) {
    const a = [];
    const b = [];
    const wa = [];
    const wb = [];
    for (let i = 0; i < X.length; i++) {
      if (X[i] === 1) { a.push(V[i][j]); wa.push(w ? w[i] : 1); } else { b.push(V[i][j]); wb.push(w ? w[i] : 1); }
    }
    const wm = (vals, ws) => vals.reduce((s, v, i) => s + v * ws[i], 0) / ws.reduce((s, v) => s + v, 0);
    const va = d3.variance(a);
    const vb = d3.variance(b);
    const sd = Math.sqrt((va + vb) / 2);
    out.push(sd > 0 ? (wm(a, wa) - wm(b, wb)) / sd : 0);
  }
  return out;
}
