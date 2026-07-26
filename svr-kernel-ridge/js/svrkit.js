/* The little linear algebra this article runs live: an RBF kernel matrix, a
 * Cholesky solve for (K + lambda I) alpha = y, and ordinary least squares for
 * the scrolly's opening straight line. Kernel ridge has a closed form, which
 * is the whole reason the page can refit it continuously under two sliders
 * and still be checked against scikit-learn at load time.
 */

export function rbfK(xa, xb, gamma) {
  const K = Array.from({ length: xa.length }, () => new Float64Array(xb.length));
  for (let i = 0; i < xa.length; i++) {
    for (let j = 0; j < xb.length; j++) {
      const d = xa[i] - xb[j];
      K[i][j] = Math.exp(-gamma * d * d);
    }
  }
  return K;
}

/* Cholesky decomposition of a symmetric positive-definite matrix (in place
 * on a copy), then forward/back substitution. n = 294 here: about a
 * millisecond, which is what makes the sliders feel like sliders. */
export function choleskySolve(A, b) {
  const n = A.length;
  const L = Array.from({ length: n }, (_, i) => Float64Array.from(A[i]));
  for (let j = 0; j < n; j++) {
    for (let k = 0; k < j; k++) {
      for (let i = j; i < n; i++) L[i][j] -= L[i][k] * L[j][k];
    }
    const d = Math.sqrt(Math.max(L[j][j], 1e-12));
    for (let i = j; i < n; i++) L[i][j] /= d;
  }
  const y = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < i; k++) y[i] -= L[i][k] * y[k];
    y[i] /= L[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let k = i + 1; k < n; k++) y[i] -= L[k][i] * y[k];
    y[i] /= L[i][i];
  }
  return y;
}

/* Fit kernel ridge on (x, y) and return a predictor over any query points. */
export function krrFit(x, y, gamma, lambda) {
  const K = rbfK(x, x, gamma);
  for (let i = 0; i < K.length; i++) K[i][i] += lambda;
  const alpha = choleskySolve(K, y);
  return (xq) => {
    const Kq = rbfK(xq, x, gamma);
    return Kq.map((row) => {
      let s = 0;
      for (let j = 0; j < x.length; j++) s += row[j] * alpha[j];
      return s;
    });
  };
}

/* Ordinary least squares in one dimension, for the scrolly's opener. */
export function ols1d(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
  }
  const b = sxy / sxx;
  const a = my - b * mx;
  return (xq) => xq.map((v) => a + b * v);
}

export const rmse = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
};
