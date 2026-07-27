/* The linear algebra this article runs in the browser.
 *
 * Everything here is small on purpose: a 13x13 symmetric eigenproblem, a 25x25
 * positive definite solve, and a rotation sweep that is a closed form in five
 * numbers. None of it needs a library, and doing it here rather than shipping
 * the answers is what lets the page recompute a factorisation when the reader
 * moves a control instead of looking up a precomputed one.
 *
 * One convention, stated once because mixing it up is a real and invisible
 * bug: every variance, covariance and moment below is the POPULATION form,
 * dividing by n. numpy's np.cov divides by n-1 by default, so the generator
 * had to be told not to; with the two conventions mixed, the closed-form
 * rotation sweep sat 3e-2 away from the honest one, which looks exactly like
 * an implementation error and is not.
 */

/* ---------------------------------------------------------------- reductions */
export function mean(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return s / v.length;
}

/* Population variance. */
export function variance(v) {
  const m = mean(v);
  let s = 0;
  for (let i = 0; i < v.length; i++) s += (v[i] - m) ** 2;
  return s / v.length;
}

export function colMeans(X) {
  const p = X[0].length;
  const out = new Float64Array(p);
  for (const row of X) for (let j = 0; j < p; j++) out[j] += row[j];
  for (let j = 0; j < p; j++) out[j] /= X.length;
  return out;
}

export function colSds(X) {
  const m = colMeans(X);
  const p = X[0].length;
  const out = new Float64Array(p);
  for (const row of X) for (let j = 0; j < p; j++) out[j] += (row[j] - m[j]) ** 2;
  for (let j = 0; j < p; j++) out[j] = Math.sqrt(out[j] / X.length);
  return out;
}

/* Centre, and optionally scale each column to unit variance. The whole point
 * of the spectrum widget is that these two produce different models. */
export function prepare(X, standardise) {
  const m = colMeans(X);
  const s = standardise ? colSds(X) : null;
  return X.map((row) => row.map((v, j) => (v - m[j]) / (s ? s[j] : 1)));
}

/* Population covariance matrix of an n x p matrix. */
export function covariance(X) {
  const n = X.length;
  const p = X[0].length;
  const m = colMeans(X);
  const C = Array.from({ length: p }, () => new Float64Array(p));
  for (const row of X) {
    for (let i = 0; i < p; i++) {
      const di = row[i] - m[i];
      for (let j = i; j < p; j++) C[i][j] += di * (row[j] - m[j]);
    }
  }
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      C[i][j] /= n;
      C[j][i] = C[i][j];
    }
  }
  return C;
}

/* --------------------------------------------------------------- eigenvalues */
/* Cyclic Jacobi for a symmetric matrix: rotate away the largest off-diagonal
 * entry, repeat. Slow in the abstract and instant at p = 13, and unlike a
 * power method it returns every eigenvector at once and in full precision,
 * which matters because the article quotes the LAST component as well as the
 * first. Returns eigenvalues descending with matching eigenvector columns. */
export function jacobiEigen(Ain, { sweeps = 100, tol = 1e-12 } = {}) {
  const p = Ain.length;
  const A = Ain.map((row) => Float64Array.from(row));
  const V = Array.from({ length: p }, (_, i) => {
    const r = new Float64Array(p);
    r[i] = 1;
    return r;
  });

  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += A[i][j] ** 2;
    if (Math.sqrt(off) < tol) break;

    for (let i = 0; i < p; i++) {
      for (let j = i + 1; j < p; j++) {
        if (Math.abs(A[i][j]) < 1e-300) continue;
        const theta = (A[j][j] - A[i][i]) / (2 * A[i][j]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < p; k++) {
          const aik = A[i][k];
          const ajk = A[j][k];
          A[i][k] = c * aik - s * ajk;
          A[j][k] = s * aik + c * ajk;
        }
        for (let k = 0; k < p; k++) {
          const aki = A[k][i];
          const akj = A[k][j];
          A[k][i] = c * aki - s * akj;
          A[k][j] = s * aki + c * akj;
        }
        for (let k = 0; k < p; k++) {
          const vki = V[k][i];
          const vkj = V[k][j];
          V[k][i] = c * vki - s * vkj;
          V[k][j] = s * vki + c * vkj;
        }
      }
    }
  }

  const order = Array.from({ length: p }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  return {
    values: order.map((i) => A[i][i]),
    /* vectors[k] is the k-th eigenvector, with its sign fixed so that its
       largest-magnitude entry is positive. The sign of an eigenvector is
       arbitrary, and letting it flip between renders makes a loadings chart
       look like it changed its mind. */
    vectors: order.map((i) => {
      const v = Array.from({ length: p }, (_, k) => V[k][i]);
      let big = 0;
      for (let k = 1; k < p; k++) if (Math.abs(v[k]) > Math.abs(v[big])) big = k;
      return v[big] < 0 ? v.map((x) => -x) : v;
    }),
  };
}

/* ------------------------------------------------------------------- solving */
/* Cholesky solve for a symmetric positive definite system. Used to fit the
 * coefficients of a digit against a set of atoms: the normal equations are
 * fine here because the atom sets are tiny (k at most 25) and well
 * conditioned, unlike the polynomial designs of the regression articles where
 * forming the same product is a documented way to be wrong in silence. */
export function solveSPD(A, b) {
  const n = b.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        L[i][i] = Math.sqrt(Math.max(s, 1e-12));
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/* ------------------------------------------------------- fitting a recipe */
/* Least squares coefficients of `x - offset` against `atoms`: what PCA and ICA
 * both use, because neither constrains the sign of anything. */
export function fitLinear(x, atoms, offset) {
  const k = atoms.length;
  const G = Array.from({ length: k }, () => new Float64Array(k));
  const b = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      let s = 0;
      for (let t = 0; t < x.length; t++) s += atoms[i][t] * atoms[j][t];
      G[i][j] = s;
      G[j][i] = s;
    }
    let s = 0;
    for (let t = 0; t < x.length; t++) s += atoms[i][t] * (x[t] - offset[t]);
    b[i] = s;
  }
  return solveSPD(G, b);
}

/* Lee and Seung's multiplicative update for the coefficients with the atoms
 * held fixed. Every factor on the right is non-negative, so a coefficient that
 * starts positive stays positive and one that reaches zero stays there: the
 * constraint is enforced by the shape of the update rather than by clipping,
 * which is the trick the whole method is built on. */
export function fitNonNegative(x, atoms, iters = 400) {
  const k = atoms.length;
  const p = x.length;
  let atomMean = 0;
  for (let i = 0; i < k; i++) for (let t = 0; t < p; t++) atomMean += atoms[i][t];
  atomMean /= k * p;
  const start = mean(x) / Math.max(atomMean, 1e-9) / k + 1e-3;

  const w = new Float64Array(k).fill(start);
  const G = Array.from({ length: k }, () => new Float64Array(k));
  const xh = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      let s = 0;
      for (let t = 0; t < p; t++) s += atoms[i][t] * atoms[j][t];
      G[i][j] = s;
      G[j][i] = s;
    }
    let s = 0;
    for (let t = 0; t < p; t++) s += atoms[i][t] * x[t];
    xh[i] = s;
  }
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < k; i++) {
      let den = 0;
      for (let j = 0; j < k; j++) den += w[j] * G[i][j];
      w[i] = (w[i] * xh[i]) / Math.max(den, 1e-10);
    }
  }
  return w;
}

/* Rebuild a row from a recipe: offset + sum of coefficient times atom. */
export function reconstruct(coeffs, atoms, offset) {
  const p = offset.length;
  const out = new Float64Array(p);
  for (let t = 0; t < p; t++) out[t] = offset[t];
  for (let i = 0; i < coeffs.length; i++) {
    const c = coeffs[i];
    const a = atoms[i];
    for (let t = 0; t < p; t++) out[t] += c * a[t];
  }
  return out;
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}

/* --------------------------------------------------------------- whitening */
/* Centre and decorrelate 2-D data to identity covariance. Returns the whitened
 * points; the article only ever needs the two-dimensional case, where the
 * eigenproblem has a closed form and Jacobi would be theatre. */
export function whiten2(P) {
  const n = P.length;
  const mx = mean(P.map((d) => d[0]));
  const my = mean(P.map((d) => d[1]));
  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (const [x, y] of P) {
    cxx += (x - mx) ** 2;
    cxy += (x - mx) * (y - my);
    cyy += (y - my) ** 2;
  }
  cxx /= n;
  cxy /= n;
  cyy /= n;
  const tr = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  const ang = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  /* V diag(1/sqrt(lambda)) V^T, written out. */
  const i1 = 1 / Math.sqrt(Math.max(l1, 1e-12));
  const i2 = 1 / Math.sqrt(Math.max(l2, 1e-12));
  const w11 = c * c * i1 + s * s * i2;
  const w12 = c * s * (i1 - i2);
  const w22 = s * s * i1 + c * c * i2;
  return P.map(([x, y]) => {
    const dx = x - mx;
    const dy = y - my;
    return [w11 * dx + w12 * dy, w12 * dx + w22 * dy];
  });
}

/* Excess kurtosis, population form. Zero for a gaussian, which is the only
 * reason this quantity is in the article at all. */
export function kurtosis(v) {
  const m = mean(v);
  let m2 = 0;
  let m4 = 0;
  for (let i = 0; i < v.length; i++) {
    const d = v[i] - m;
    m2 += d * d;
    m4 += d * d * d * d;
  }
  m2 /= v.length;
  m4 /= v.length;
  return m4 / (m2 * m2) - 3;
}

/* The non-gaussianity of both rotated axes, for one angle, from the five mixed
 * fourth moments of the whitened data. A rotation of whitened data keeps unit
 * variance, so the fourth moment of the rotated axis is a quartic in
 * (cos, sin) and the entire 180-degree curve is arithmetic on five numbers. */
export function objectiveAt(m, deg) {
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  const [m40, m31, m22, m13, m04] = m;
  const k1 = c ** 4 * m40 + 4 * c ** 3 * s * m31 + 6 * c ** 2 * s ** 2 * m22
    + 4 * c * s ** 3 * m13 + s ** 4 * m04 - 3;
  const k2 = s ** 4 * m40 - 4 * s ** 3 * c * m31 + 6 * s ** 2 * c ** 2 * m22
    - 4 * s * c ** 3 * m13 + c ** 4 * m04 - 3;
  return { k1, k2, value: k1 * k1 + k2 * k2 };
}

export function rotate(P, deg) {
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  return P.map(([x, y]) => [c * x + s * y, -s * x + c * y]);
}

/* Pearson correlation. */
export function corr(a, b) {
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db || 1e-12);
}

/* How well a pair of recovered signals matches a pair of sources, allowing
 * both signs and both pairings, because no factorisation can do better than
 * recovering sources up to sign and order. Returns the worst source's match
 * and the worst amount of the other source leaking into it. */
export function recovery(rec, src) {
  const r0 = rec.map((d) => d[0]);
  const r1 = rec.map((d) => d[1]);
  const s0 = src.map((d) => d[0]);
  const s1 = src.map((d) => d[1]);
  const M = [[Math.abs(corr(r0, s0)), Math.abs(corr(r0, s1))],
    [Math.abs(corr(r1, s0)), Math.abs(corr(r1, s1))]];
  if (M[0][0] + M[1][1] >= M[0][1] + M[1][0]) {
    return { match: Math.min(M[0][0], M[1][1]), leak: Math.max(M[0][1], M[1][0]) };
  }
  return { match: Math.min(M[0][1], M[1][0]), leak: Math.max(M[0][0], M[1][1]) };
}
