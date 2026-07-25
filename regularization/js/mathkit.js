/* Penalized linear models, from scratch and numerically honest.
 *
 * Mirrors src/models/regularization.py (ScratchRidge, ScratchLasso) and is
 * validated at runtime against the scikit-learn reference fits shipped in
 * data/*.json.
 *
 * A note on why this file does NOT use the textbook (X'X + aI)^-1 X'y form.
 * Forming X'X squares the condition number of the problem. On the degree-15
 * polynomial in this article that is fatal: cond(X) is already around 1e10, so
 * cond(X'X) lands past the limit of double precision and the "unregularized"
 * fit silently comes back wrong. Everything here is therefore solved as a least
 * squares problem via Householder QR, which is what careful libraries do.
 *
 * All objectives use the same 1/(2n) scaling so one alpha slider is comparable
 * across penalties:
 *
 *   (1/2n)||y - Xb||^2  +  a*L1*||b||_1  +  a*(1-L1)/2*||b||^2
 *
 * L1 = 1 is pure Lasso, L1 = 0 is pure Ridge, anything between is ElasticNet.
 */

/* ------------------------------------------------------------------ QR least squares */

/* Solve min ||A x - b|| by Householder reflections. A is m x n with m >= n.
 * Works on a copy of the augmented matrix [A | b] so b is transformed with A. */
export function lstsq(A, b) {
  const m = A.length;
  const n = A[0].length;
  const R = A.map((row, i) => [...row, b[i]]);

  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] * R[i][k];
    norm = Math.sqrt(norm);
    if (norm < 1e-300) continue;

    // choose the sign that avoids cancellation
    const alpha = R[k][k] > 0 ? -norm : norm;
    const v = new Array(m).fill(0);
    for (let i = k; i < m; i++) v[i] = R[i][k];
    v[k] -= alpha;

    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] * v[i];
    if (vv < 1e-300) continue;

    // apply H = I - 2vv'/v'v to every remaining column, b included
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

/* ------------------------------------------------------------------ the penalties */

/* Soft thresholding: the whole reason L1 produces exact zeros.
 * If a feature's usefulness is smaller than the toll it pays nothing and
 * leaves. Otherwise it is admitted, shrunk by exactly the toll. */
export function softThreshold(rho, t) {
  if (rho > t) return rho - t;
  if (rho < -t) return rho + t;
  return 0;
}

/* Ridge, solved as an augmented least squares problem.
 *
 * Stacking sqrt(n*a)*I underneath X and zeros underneath y gives, in the normal
 * equations, exactly (X'X + n*a*I)b = X'y. Same answer as the textbook formula,
 * without ever forming X'X. Literally "invent n fake observations that all say
 * the coefficients should be zero", which is a fair description of what a
 * penalty does.
 */
export function ridgeFit(X, y, alpha) {
  const n = X.length;
  const p = X[0].length;
  const s = Math.sqrt(n * alpha);
  const A = X.map((r) => [...r]);
  const b = [...y];
  for (let j = 0; j < p; j++) {
    const row = new Array(p).fill(0);
    row[j] = s;
    A.push(row);
    b.push(0);
  }
  return lstsq(A, b);
}

/* Coordinate descent for Lasso / ElasticNet, maintaining the residual instead
 * of recomputing predictions per coordinate. warmStart lets an alpha path reuse
 * the previous solution, which is what makes a 120-position slider feel instant. */
export function coordFit(X, y, alpha, { l1Ratio = 1, warmStart = null, maxIter = 800, tol = 1e-10 } = {}) {
  const n = X.length;
  const p = X[0].length;
  const b = warmStart ? [...warmStart] : new Array(p).fill(0);

  const r = y.map((v, i) => {
    let s = v;
    for (let j = 0; j < p; j++) s -= X[i][j] * b[j];
    return s;
  });

  const z = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j] * X[i][j];
    z[j] = s / n;
  }

  const l1 = alpha * l1Ratio;
  const l2 = alpha * (1 - l1Ratio);

  for (let it = 0; it < maxIter; it++) {
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      if (z[j] < 1e-14) continue;
      let dot = 0;
      for (let i = 0; i < n; i++) dot += X[i][j] * r[i];
      const rho = dot / n + z[j] * b[j];
      const next = softThreshold(rho, l1) / (z[j] + l2);
      const delta = next - b[j];
      if (delta !== 0) {
        for (let i = 0; i < n; i++) r[i] -= X[i][j] * delta;
        b[j] = next;
      }
      maxDelta = Math.max(maxDelta, Math.abs(delta));
    }
    if (maxDelta < tol) break;
  }
  return b;
}

/* ------------------------------------------------------------------ model handle */

/* One object per dataset: caches nothing expensive, keeps call sites tidy. */
export function makeModel(X, y) {
  return {
    X,
    y,
    fit(alpha, l1Ratio = 1) {
      if (l1Ratio === 0) return ridgeFit(X, y, alpha);
      return coordFit(X, y, alpha, { l1Ratio });
    },
    /* A whole path, warm-started from the strongest penalty downward. */
    path(alphas, l1Ratio = 1) {
      const order = alphas.map((a, i) => [a, i]).sort((p, q) => q[0] - p[0]);
      const out = new Array(alphas.length);
      let warm = null;
      for (const [a, i] of order) {
        const b = l1Ratio === 0 ? ridgeFit(X, y, a) : coordFit(X, y, a, { l1Ratio, warmStart: warm });
        warm = b;
        out[i] = b;
      }
      return out;
    },
    /* X'X / n, for the two-dimensional geometry picture only. */
    gram() {
      const n = X.length;
      const p = X[0].length;
      const G = Array.from({ length: p }, () => new Array(p).fill(0));
      for (let i = 0; i < n; i++) {
        for (let a = 0; a < p; a++) {
          for (let b2 = a; b2 < p; b2++) G[a][b2] += X[i][a] * X[i][b2];
        }
      }
      for (let a = 0; a < p; a++) {
        for (let b2 = a; b2 < p; b2++) {
          G[a][b2] /= n;
          G[b2][a] = G[a][b2];
        }
      }
      return G;
    },
  };
}

/* ------------------------------------------------------------------ helpers */

export function predict(X, b) {
  return X.map((row) => row.reduce((s, v, j) => s + v * b[j], 0));
}

export function rmse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return Math.sqrt(s / yTrue.length);
}

export function nonZero(b, eps = 1e-8) {
  return b.reduce((c, v) => c + (Math.abs(v) > eps ? 1 : 0), 0);
}

/* Standardized polynomial design matrix, using the mu/sd shipped with the data
 * so the browser reproduces the Python pipeline exactly. */
export function polyDesign(xs, degree, mu, sd) {
  return xs.map((x) => {
    const row = [];
    for (let d = 1; d <= degree; d++) row.push((x ** d - mu[d - 1]) / sd[d - 1]);
    return row;
  });
}

/* Eigen-decomposition of a symmetric 2x2 matrix, for the RSS contour ellipses.
 * Returns eigenvalues (l1 >= l2) and the rotation angle of the first axis. */
export function eig2(A) {
  const a = A[0][0];
  const b = A[0][1];
  const c = A[1][1];
  const tr = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  const angle = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return { l1, l2, angle };
}
