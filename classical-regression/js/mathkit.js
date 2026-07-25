/* Small linear-algebra kit for the classical-regression article.
 * Mirrors the from-scratch NumPy implementations in src/models/: same math,
 * same results (validated against the reference values baked into data/*.json).
 */

/* Solve A x = b via Gaussian elimination with partial pivoting.
 * A: array of rows. Returns x. Small systems only (<= 10x10 here). */
export function solve(A, b) {
  const n = A.length;
  // augmented copy
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // pivot
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    [M[col], M[piv]] = [M[piv], M[col]];
    const p = M[col][col];
    if (Math.abs(p) < 1e-12) continue; // singular direction: leave as-is
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / p;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / (Math.abs(row[i]) < 1e-12 ? 1 : row[i]));
}

/* Ordinary least squares on a design matrix WITH bias handled here.
 * X: array of feature rows (no ones column), y: targets.
 * Returns coeffs [b0, b1, ..., bp] via the Normal Equation, exactly like ScratchOLS. */
export function olsFit(X, y) {
  const n = X.length;
  const p = X[0].length + 1;
  // Build X^T X and X^T y with the implicit ones column
  const XtX = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const row = [1, ...X[i]];
    for (let a = 0; a < p; a++) {
      Xty[a] += row[a] * y[i];
      for (let b = a; b < p; b++) XtX[a][b] += row[a] * row[b];
    }
  }
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < a; b++) XtX[a][b] = XtX[b][a];
    XtX[a][a] += 1e-9; // tiny jitter for numerical safety at high degree
  }
  return solve(XtX, Xty);
}

/* Solve min ||A x - b|| by Householder QR, without ever forming A'A.
 *
 * The article teaches the Normal Equation and olsFit above is its faithful
 * mirror, which is fine for the two-parameter line it fits. Polynomials are a
 * different animal: forming A'A squares the condition number, and the powers
 * x^1..x^9 stay nearly collinear even after each column is standardized. In
 * double precision the elimination then hits pivots it has to skip, and the
 * fit comes back wrong *silently*. Measured on this very widget: at degree 9
 * on all 294 cars the normal equation reported test r^2 = 0.681 when the true
 * value, which QR and the Python reference both give, is 0.649. That is the
 * difference between "degree 9 is fine" and "degree 9 is past the peak", which
 * is the entire point of the widget. */
export function lstsq(A, b) {
  const m = A.length;
  const n = A[0].length;
  const R = A.map((row, i) => [...row, b[i]]);

  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] * R[i][k];
    norm = Math.sqrt(norm);
    if (norm < 1e-300) continue;

    const alpha = R[k][k] > 0 ? -norm : norm; // sign that avoids cancellation
    const v = new Array(m).fill(0);
    for (let i = k; i < m; i++) v[i] = R[i][k];
    v[k] -= alpha;

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

/* Polynomial regression of given degree with column standardization (the same
 * scheme the Python reference used, so the R^2 values match) solved by QR.
 * Returns a predict(x) function plus the coefficients. */
export function polyFit(xs, ys, degree) {
  const n = xs.length;
  const mu = [];
  const sd = [];
  for (let d = 1; d <= degree; d++) {
    let m = 0;
    for (const x of xs) m += x ** d;
    m /= n;
    let s = 0;
    for (const x of xs) s += (x ** d - m) ** 2;
    s = Math.sqrt(s / n) || 1;
    mu.push(m);
    sd.push(s);
  }
  const X = xs.map((x) => {
    const row = [1];
    for (let d = 1; d <= degree; d++) row.push((x ** d - mu[d - 1]) / sd[d - 1]);
    return row;
  });
  const coeffs = lstsq(X, ys);
  const predict = (x) => {
    let yhat = coeffs[0];
    for (let d = 1; d <= degree; d++) yhat += coeffs[d] * ((x ** d - mu[d - 1]) / sd[d - 1]);
    return yhat;
  };
  return { predict, coeffs };
}

/* R^2 */
export function r2Score(yTrue, yPred) {
  const my = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < yTrue.length; i++) {
    ssRes += (yTrue[i] - yPred[i]) ** 2;
    ssTot += (yTrue[i] - my) ** 2;
  }
  return 1 - ssRes / ssTot;
}

/* MSE */
export function mse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return s / yTrue.length;
}

/* Poisson GLM trainer on standardized x, the exact ScratchPoissonGLM loop.
 * Returns an object you can .step(k) and read {w, b, iter, nll}. */
export function poissonGLM(xRaw, y, lr = 0.1) {
  const n = xRaw.length;
  const mx = xRaw.reduce((a, b) => a + b, 0) / n;
  const sx = Math.sqrt(xRaw.reduce((a, x) => a + (x - mx) ** 2, 0) / n);
  const xs = xRaw.map((x) => (x - mx) / sx);
  const model = {
    w: 0,
    b: 0,
    iter: 0,
    meanX: mx,
    stdX: sx,
    /* natural-scale coefficients: log(mu) = b0n + b1n * bill */
    natural() {
      return { b1: this.w / sx, b0: this.b - (this.w * mx) / sx };
    },
    predict(billValue) {
      return Math.exp(this.b + this.w * ((billValue - mx) / sx));
    },
    nll() {
      // negative mean Poisson log-likelihood (dropping the log(y!) constant)
      let s = 0;
      for (let i = 0; i < n; i++) {
        const eta = this.b + this.w * xs[i];
        s += Math.exp(eta) - y[i] * eta;
      }
      return s / n;
    },
    step(k = 1) {
      for (let it = 0; it < k; it++) {
        let dw = 0;
        let db = 0;
        for (let i = 0; i < n; i++) {
          const err = Math.exp(this.b + this.w * xs[i]) - y[i];
          dw += err * xs[i];
          db += err;
        }
        this.w -= (lr * dw) / n;
        this.b -= (lr * db) / n;
        this.iter++;
      }
    },
    reset() {
      this.w = 0;
      this.b = 0;
      this.iter = 0;
    },
  };
  return model;
}
