/* ATLAS series helpers: the recursions module 4.2 runs in the browser.
 *
 * Everything here is written to match a definition the generators also write
 * out, not to match a library, because the guard on each page compares two
 * implementations of one definition and a definition that only exists in one
 * of them cannot be checked. Two conventions are pinned down on purpose:
 *
 *   - autocovariance divides by n at every lag, the biased estimator, because
 *     the unbiased one is not positive definite and a partial autocorrelation
 *     solved from it can leave the unit circle;
 *   - variances are population variances (divide by n), because numpy divides
 *     by n - 1 by default and mixing the two shows up in the third decimal and
 *     looks exactly like an implementation error.
 *
 * No DOM in this file: it is imported by the pages and by a node script that
 * checks it against the generators without a browser.
 */

export function mean(y) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += y[i];
  return s / y.length;
}

export function variance(y) {
  const m = mean(y);
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (y[i] - m) ** 2;
  return s / y.length;
}

/* Autocorrelation up to `nlags`, index 0 being the trivial 1. */
export function acf(y, nlags) {
  const m = mean(y);
  const n = y.length;
  let d0 = 0;
  for (let i = 0; i < n; i++) d0 += (y[i] - m) ** 2;
  const out = [1];
  for (let k = 1; k <= nlags; k++) {
    let s = 0;
    for (let i = k; i < n; i++) s += (y[i] - m) * (y[i - k] - m);
    out.push(s / d0);
  }
  return out;
}

/* Partial autocorrelation by the Durbin-Levinson recursion. */
export function pacf(y, nlags) {
  const rho = acf(y, nlags);
  const phi = Array.from({ length: nlags + 1 }, () => new Array(nlags + 1).fill(0));
  const out = new Array(nlags + 1).fill(0);
  out[0] = 1;
  let v = 1;
  for (let k = 1; k <= nlags; k++) {
    let num = rho[k];
    for (let j = 1; j < k; j++) num -= phi[k - 1][j] * rho[k - j];
    phi[k][k] = num / v;
    for (let j = 1; j < k; j++) phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
    v *= 1 - phi[k][k] ** 2;
    out[k] = phi[k][k];
  }
  return out;
}

/* y[t] - y[t - lag]. */
export function difference(y, lag = 1) {
  const out = [];
  for (let i = lag; i < y.length; i++) out.push(y[i] - y[i - lag]);
  return out;
}

/* Simple exponential smoothing: l[t] = a*y[t] + (1-a)*l[t-1].
 * Returns the level after each observation, which is also the forecast of the
 * next one at every horizon. */
export function sesLevels(y, alpha, level0 = null) {
  let l = level0 === null ? y[0] : level0;
  const out = new Array(y.length);
  for (let i = 0; i < y.length; i++) {
    l = alpha * y[i] + (1 - alpha) * l;
    out[i] = l;
  }
  return out;
}

/* ARIMA(0,1,1) in error form: yhat[t+1] = y[t] + theta*e[t], e[t] = y[t]-yhat[t].
 * Started wherever the smoother is started, because the identity between the
 * two is exact only when they begin in the same place. */
export function ma1Forecasts(y, theta, level0 = null) {
  let prev = level0 === null ? y[0] : level0;
  const out = new Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const e = y[i] - prev;
    prev = y[i] + theta * e;
    out[i] = prev;
  }
  return out;
}

/* The GARCH(1,1) variance recursion: v[t] = w + a*r[t-1]^2 + b*v[t-1].
 * v[i] is the variance forecast made for day i using everything up to i-1, so
 * the array lines up with the returns it is a forecast of. */
export function garchFilter(r, omega, alpha, beta, var0) {
  const out = new Array(r.length);
  let v = var0;
  for (let i = 0; i < r.length; i++) {
    out[i] = v;
    v = omega + alpha * r[i] * r[i] + beta * v;
  }
  return out;
}

/* Gaussian negative log likelihood of those returns under that variance path. */
export function garchNLL(r, omega, alpha, beta, var0) {
  const v = garchFilter(r, omega, alpha, beta, var0);
  let s = 0;
  for (let i = 0; i < r.length; i++) {
    s += 0.5 * (Math.log(2 * Math.PI * v[i]) + (r[i] * r[i]) / v[i]);
  }
  return s;
}

/* Exponentially weighted variance, the RiskMetrics recursion. */
export function ewmaVar(r, lambda, var0) {
  const out = new Array(r.length);
  let v = var0;
  for (let i = 0; i < r.length; i++) {
    out[i] = v;
    v = lambda * v + (1 - lambda) * r[i] * r[i];
  }
  return out;
}

/* Least squares by Householder QR.
 *
 * Never by the normal equations: forming A'A squares the condition number, and
 * a lag matrix of a persistent series is exactly where that goes wrong in
 * silence. Same routine as the regression articles use. */
export function lstsq(A, b) {
  const m = A.length;
  const n = A[0].length;
  const R = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] * R[i][k];
    norm = Math.sqrt(norm);
    if (norm < 1e-300) continue;
    const a = R[k][k] > 0 ? -norm : norm;
    const v = new Array(m).fill(0);
    for (let i = k; i < m; i++) v[i] = R[i][k];
    v[k] -= a;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] * v[i];
    if (vv < 1e-300) continue;
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
    x[i] = Math.abs(R[i][i]) < 1e-300 ? 0 : s / R[i][i];
  }
  return x;
}

/* Mean absolute scaled error: absolute errors in units of the in sample
 * seasonal naive error, which is what makes two series comparable. */
export function maseScale(train, period) {
  let s = 0;
  for (let i = period; i < train.length; i++) s += Math.abs(train[i] - train[i - period]);
  return s / (train.length - period);
}
