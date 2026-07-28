/* The other three detectors, written so that the page runs them rather than
 * replaying them. No DOM, so node can check the lot against the generator's
 * references before any of it is drawn.
 *
 * Two of the three are exactly reproducible against scikit-learn (LOF and the
 * concentration step are deterministic given the data), and the third is not
 * trained here at all: fitting a one-class SVM needs a quadratic programme, so
 * the generator ships the dual coefficients and this file evaluates the
 * decision function, which is the same split of labour the detection and
 * autoencoder articles use for their networks.
 */

export function sqDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

/* All pairwise squared distances, once. Everything below reads this. */
export function distanceTable(X) {
  const n = X.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = sqDist(X[i], X[j]);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
}

/* The k nearest neighbours of every point, excluding itself.
 *
 * Sorted on the SQUARED distance and reported as the root, because ordering by
 * d and by d^2 is not the same order in floating point and a single swapped tie
 * moved an eigenvalue by a whole decimal in an earlier article. One quantity,
 * one order, both sides. */
export function neighbours(D, k) {
  const n = D.length;
  return Array.from({ length: n }, (_, i) => {
    const idx = [];
    for (let j = 0; j < n; j++) if (j !== i) idx.push(j);
    idx.sort((a, b) => D[i][a] - D[i][b]);
    return idx.slice(0, k);
  });
}

/* Distance to the kth neighbour: the simplest global detector there is, and the
 * control LOF has to beat. */
export function knnDistance(D, k) {
  const nb = neighbours(D, k);
  return nb.map((row, i) => Math.sqrt(D[i][row[row.length - 1]]));
}

/* Breunig's local outlier factor.
 *
 * k-distance is the distance to the kth neighbour; the reachability distance
 * from a to b is max(k-distance of b, d(a,b)), which is the smoothing that
 * stops a point inside a tight cluster from having an explosive density; the
 * local reachability density is one over the mean of those; and LOF is the mean
 * ratio of a neighbour's density to your own. One means "as dense as the
 * company you keep". */
export function lof(X, k) {
  const D = distanceTable(X);
  const n = X.length;
  const nb = neighbours(D, k);
  const kdist = nb.map((row, i) => Math.sqrt(D[i][row[row.length - 1]]));
  const lrd = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const j of nb[i]) s += Math.max(kdist[j], Math.sqrt(D[i][j]));
    lrd[i] = s > 0 ? nb[i].length / s : Infinity;
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const j of nb[i]) s += lrd[j];
    out[i] = s / (nb[i].length * lrd[i]);
  }
  return { lof: Array.from(out), kdist, lrd: Array.from(lrd), nb };
}

/* ------------------------------------------------------------ one-class SVM */

/* f(x) = sum_i alpha_i K(x_i, x) - rho, with K the gaussian kernel. Positive is
 * inside the boundary, which is scikit-learn's sign convention. */
export function svmDecision(model, X, Q) {
  const { sv, alpha, rho, gamma } = model;
  return Q.map((q) => {
    let s = 0;
    for (let i = 0; i < sv.length; i++) s += alpha[i] * Math.exp(-gamma * sqDist(X[sv[i]], q));
    return s - rho;
  });
}

/* ------------------------------------------------------------------ the MCD */

export function colMeans(X, rows) {
  const p = X[0].length;
  const m = new Float64Array(p);
  for (const i of rows) for (let j = 0; j < p; j++) m[j] += X[i][j];
  for (let j = 0; j < p; j++) m[j] /= rows.length;
  return m;
}

export function covariance(X, rows, mu) {
  const p = X[0].length;
  const S = Array.from({ length: p }, () => new Float64Array(p));
  for (const i of rows) {
    for (let a = 0; a < p; a++) {
      const da = X[i][a] - mu[a];
      for (let b = a; b < p; b++) S[a][b] += da * (X[i][b] - mu[b]);
    }
  }
  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      S[a][b] /= rows.length - 1;
      S[b][a] = S[a][b];
    }
  }
  return S;
}

export function cholesky(S) {
  const p = S.length;
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = S[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 0)) return null;
        L[i][i] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

export function mahalanobis(X, mu, S) {
  const L = cholesky(S);
  if (!L) return null;
  const p = mu.length;
  return X.map((x) => {
    let acc = 0;
    const y = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      let t = x[i] - mu[i];
      for (let k = 0; k < i; k++) t -= L[i][k] * y[k];
      y[i] = t / L[i][i];
      acc += y[i] * y[i];
    }
    return acc;
  });
}

export function detFromChol(S) {
  const L = cholesky(S);
  if (!L) return 0;
  let d = 1;
  for (let i = 0; i < L.length; i++) d *= L[i][i] ** 2;
  return d;
}

/* One concentration step: fit on the support, score everybody, keep the h
 * smallest. Rousseeuw and Van Driessen's theorem is that the determinant never
 * goes up, so the widget can print it and the reader can watch. */
export function cStep(X, support) {
  const h = support.length;
  const mu = colMeans(X, support);
  const S = covariance(X, support, mu);
  const d2 = mahalanobis(X, mu, S);
  const next = d2.map((v, i) => i).sort((a, b) => d2[a] - d2[b]).slice(0, h).sort((a, b) => a - b);
  return { mu, S, d2, next, det: detFromChol(S) };
}

/* --------------------------------------------------------------- scoring */

/* Average precision, computed the way scikit-learn does it: the step-wise sum
 * of precision at each threshold weighted by the change in recall, with ties
 * broken by taking every point at the same score together. */
export function averagePrecision(labels, scores) {
  const n = labels.length;
  const order = scores.map((v, i) => i).sort((a, b) => scores[b] - scores[a]);
  const positives = labels.reduce((a, b) => a + b, 0);
  let tp = 0;
  let fp = 0;
  let prev = 0;
  let ap = 0;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && scores[order[j]] === scores[order[i]]) j += 1;
    for (let q = i; q < j; q += 1) (labels[order[q]] ? (tp += 1) : (fp += 1));
    const recall = tp / positives;
    ap += (recall - prev) * (tp / (tp + fp));
    prev = recall;
    i = j;
  }
  return ap;
}

/* Area under the ROC curve, by the rank-sum identity so that ties cost nothing
 * to handle correctly. */
export function rocAuc(labels, scores) {
  const n = labels.length;
  const order = scores.map((v, i) => i).sort((a, b) => scores[a] - scores[b]);
  const rank = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && scores[order[j]] === scores[order[i]]) j += 1;
    const r = (i + j + 1) / 2;                    /* average rank, 1 based */
    for (let q = i; q < j; q++) rank[order[q]] = r;
    i = j;
  }
  let sum = 0;
  let pos = 0;
  for (let q = 0; q < n; q++) {
    if (labels[q]) {
      sum += rank[q];
      pos += 1;
    }
  }
  const neg = n - pos;
  return (sum - (pos * (pos + 1)) / 2) / (pos * neg);
}
