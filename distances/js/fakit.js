/* The factor model, in the browser.
 *
 * Everything here mirrors, line for line, what generate_distances_data.py runs
 * in numpy, because the whole point of the widget is that the reader's own
 * machine fits both models rather than downloading two pictures of them. Where
 * the two could drift they have been pinned:
 *
 * - the covariance is the POPULATION one, divide by n, because numpy's default
 *   of n-1 has no counterpart here and mixing the two is a silent few parts in
 *   a thousand;
 * - the EM starts from the eigendecomposition of that covariance rather than
 *   from anything random, so the whole trajectory is determined and the two
 *   ends can be compared to eight decimals;
 * - a factor model is only ever identified up to a rotation of its factors, so
 *   nothing here compares loadings entry by entry. What gets compared is the
 *   subspace they span and the uniquenesses, which are identified.
 */
import { symEigen, covariance, centre, zeros } from '../../assets/js/embed.js';

/* PCA loadings, scaled the way a factor analyst writes them: column j is the
 * jth eigenvector times the square root of its eigenvalue, so that the sum of
 * the squared entries of a row is that variable's explained variance. */
export function pcaLoadings(X, k) {
  const S = covariance(X);
  const { values, vectors } = symEigen(S);
  const p = S.length;
  const L = zeros(p, k);
  for (let j = 0; j < k; j++) {
    const s = Math.sqrt(Math.max(values[j], 0));
    for (let i = 0; i < p; i++) L[i][j] = vectors[i][j] * s;
  }
  return { loadings: L, values };
}

function matmul(A, B) {
  const n = A.length;
  const m = B[0].length;
  const q = B.length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < q; t++) {
      const a = A[i][t];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += a * B[t][j];
    }
  }
  return C;
}

function transpose(A) {
  const T = zeros(A[0].length, A.length);
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) T[j][i] = A[i][j];
  return T;
}

/* Inverse of a small symmetric positive definite matrix by Gauss-Jordan. k is
 * two or three here, so nothing clever is warranted or wanted. */
function inverse(A) {
  const n = A.length;
  const M = A.map((r, i) => {
    const row = new Float64Array(2 * n);
    row.set(r, 0);
    row[n + i] = 1;
    return row;
  });
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    const t = M[c]; M[c] = M[piv]; M[piv] = t;
    const d = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((r) => r.slice(n));
}

function logdet(A) {
  /* Through the eigenvalues, which are already needed nowhere else but are the
     safest route for a matrix that can be close to singular when one variable
     is nearly all noise. */
  const { values } = symEigen(A);
  let s = 0;
  for (let i = 0; i < values.length; i++) s += Math.log(Math.max(values[i], 1e-300));
  return s;
}

/* Rubin and Thayer's EM for maximum likelihood factor analysis, on a covariance
 * matrix. Returns the loadings, the uniquenesses, the log likelihood per
 * observation and how many iterations it took. */
export function faEM(S, k, { iters = 5000, tol = 1e-12 } = {}) {
  const p = S.length;
  const psi = new Float64Array(p);
  for (let i = 0; i < p; i++) psi[i] = S[i][i];
  const { values, vectors } = symEigen(S);
  let L = zeros(p, k);
  for (let j = 0; j < k; j++) {
    const s = Math.sqrt(Math.max(values[j], 1e-6));
    for (let i = 0; i < p; i++) L[i][j] = vectors[i][j] * s;
  }
  let prev = -Infinity;
  let ll = prev;
  let used = iters;
  for (let it = 0; it < iters; it++) {
    /* M = I + L' Psi^-1 L */
    const M = zeros(k, k);
    for (let a = 0; a < k; a++) {
      M[a][a] += 1;
      for (let b = 0; b < k; b++) {
        let s = 0;
        for (let i = 0; i < p; i++) s += (L[i][a] * L[i][b]) / Math.max(psi[i], 1e-10);
        M[a][b] += s;
      }
    }
    const Minv = inverse(M);
    /* beta = Minv L' Psi^-1, k x p */
    const beta = zeros(k, p);
    for (let a = 0; a < k; a++) {
      for (let i = 0; i < p; i++) {
        let s = 0;
        for (let b = 0; b < k; b++) s += Minv[a][b] * L[i][b];
        beta[a][i] = s / Math.max(psi[i], 1e-10);
      }
    }
    const betaS = matmul(beta, S);                         // k x p
    const SB = transpose(betaS);                           // p x k, equals S beta'
    const Ezz = zeros(k, k);                               // Minv + beta S beta'
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < k; b++) {
        let s = Minv[a][b];
        for (let i = 0; i < p; i++) s += betaS[a][i] * beta[b][i];
        Ezz[a][b] = s;
      }
    }
    L = matmul(SB, inverse(Ezz));
    for (let i = 0; i < p; i++) {
      let s = 0;
      for (let a = 0; a < k; a++) s += L[i][a] * betaS[a][i];
      psi[i] = Math.max(S[i][i] - s, 1e-8);
    }
    /* the likelihood, up to the constant both ends drop */
    const Sigma = zeros(p, p);
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let a = 0; a < k; a++) s += L[i][a] * L[j][a];
        Sigma[i][j] = s + (i === j ? psi[i] : 0);
      }
    }
    const Sinv = inverse(Sigma);
    let tr = 0;
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) tr += Sinv[i][j] * S[j][i];
    ll = -0.5 * (logdet(Sigma) + tr);
    if (Math.abs(ll - prev) < tol * Math.max(Math.abs(ll), 1)) {
      used = it + 1;
      break;
    }
    prev = ll;
  }
  return { loadings: L, psi, loglik: ll, iters: used };
}

/* One minus the mean squared cosine of the principal angles between two column
 * spaces: zero when they are the same plane, one when they are perpendicular.
 * Both inputs are p x k with k small, so a Gram-Schmidt and a k x k SVD is all
 * it takes, and the SVD comes out of the eigenvalues of the k x k product. */
export function subspaceGap(A, B) {
  const qa = gramSchmidt(A);
  const qb = gramSchmidt(B);
  const k = A[0].length;
  const C = zeros(k, k);
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      let s = 0;
      for (let i = 0; i < qa.length; i++) s += qa[i][a] * qb[i][b];
      C[a][b] = s;
    }
  }
  const CtC = zeros(k, k);
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += C[t][a] * C[t][b];
      CtC[a][b] = s;
    }
  }
  const { values } = symEigen(CtC);
  let m = 0;
  for (let i = 0; i < k; i++) m += Math.max(values[i], 0);
  return 1 - m / k;
}

function gramSchmidt(A) {
  const n = A.length;
  const k = A[0].length;
  const Q = zeros(n, k);
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < n; i++) Q[i][j] = A[i][j];
    for (let t = 0; t < j; t++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += Q[i][t] * Q[i][j];
      for (let i = 0; i < n; i++) Q[i][j] -= dot * Q[i][t];
    }
    let nn = 0;
    for (let i = 0; i < n; i++) nn += Q[i][j] ** 2;
    nn = Math.sqrt(nn);
    if (nn > 1e-14) for (let i = 0; i < n; i++) Q[i][j] /= nn;
  }
  return Q;
}

/* Build the widget's dataset at one setting of the volume knob: the shared part
 * plus fixed standard normal noise, scaled column by column. The noise matrix
 * is the same one at every setting, so what the slider changes is one number
 * and not the draw, which is what makes the comparison a comparison. */
export function buildDataset(common, noise, psiSd, loud, sdLoud) {
  const n = common.length;
  const p = common[0].length;
  const sd = Float64Array.from(psiSd);
  sd[loud] = sdLoud;
  const X = zeros(n, p);
  for (let i = 0; i < n; i++) {
    /* Rounded to four decimals exactly where the generator rounds, because a
       fit of the unrounded numbers is a fit of different numbers. */
    for (let j = 0; j < p; j++) {
      X[i][j] = Math.round((common[i][j] + noise[i][j] * sd[j]) * 1e4) / 1e4;
    }
  }
  return { X: centre(X), sd };
}
