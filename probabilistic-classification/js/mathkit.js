/* Probabilistic classification, from scratch.
 *
 * Mirrors src/models/probabilistic_classification.py. Everything here is
 * validated at runtime against the scikit-learn reference values shipped in
 * data/cancer.json, and the checks warn loudly rather than failing silently.
 */

/* The sigmoid, written so it cannot overflow. exp() of a large positive number
 * is Infinity and Infinity/Infinity is NaN, so branch on the sign and let the
 * exponent always be negative. Same function, no cliff at |z| ~ 710. */
export function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export const logit = (p) => Math.log(p / (1 - p));

/* Mean negative log-likelihood of a binary model, the loss logistic regression
 * is actually trained on. Clipped, because a confident mistake would otherwise
 * report Infinity and take the contour map with it. */
export function crossEntropy(ys, ps) {
  let s = 0;
  for (let i = 0; i < ys.length; i++) {
    const p = Math.min(Math.max(ps[i], 1e-12), 1 - 1e-12);
    s += ys[i] ? -Math.log(p) : -Math.log(1 - p);
  }
  return s / ys.length;
}

/* Mean squared error of the probability against the 0 or 1 that happened.
 * As a training objective for a sigmoid this is a mistake, which the article
 * demonstrates rather than asserts; as a SCORE on held-out data it is the
 * Brier score and is perfectly respectable. Same arithmetic, two jobs. */
export function brier(ys, ps) {
  let s = 0;
  for (let i = 0; i < ys.length; i++) s += (ps[i] - ys[i]) ** 2;
  return s / ys.length;
}

/* Area under the ROC curve, computed as the rank statistic rather than by
 * integrating a curve: it is the probability that a random positive outranks a
 * random negative, ties counting half. Exact, and O(n log n). */
export function auc(ys, ps) {
  const idx = ps.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const ranks = new Array(ps.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const r = (i + j) / 2 + 1; // average rank over the tied block
    for (let k = i; k <= j; k++) ranks[idx[k][1]] = r;
    i = j + 1;
  }
  let sumPos = 0;
  let nPos = 0;
  for (let k = 0; k < ys.length; k++) {
    if (ys[k] === 1) {
      sumPos += ranks[k];
      nPos++;
    }
  }
  const nNeg = ys.length - nPos;
  if (!nPos || !nNeg) return 0.5;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/* Batch gradient descent on the cross-entropy, for one feature plus intercept.
 * Returns the whole path, because the article's point is the shape of the
 * journey and not only where it ends up. */
export function descend(xs, ys, { b0 = 0, b1 = 0, lr = 0.5, steps = 400, loss = 'ce' } = {}) {
  const n = xs.length;
  const path = [{ b0, b1 }];
  for (let t = 0; t < steps; t++) {
    let g0 = 0;
    let g1 = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(b0 + b1 * xs[i]);
      /* Cross-entropy through a sigmoid collapses to (p - y). Squared error
       * does not: the chain rule leaves a p(1-p) factor behind, and that factor
       * is what flattens the surface to nothing once the model is confident,
       * whether it is confidently right or confidently wrong. */
      const d = loss === 'ce' ? p - ys[i] : 2 * (p - ys[i]) * p * (1 - p);
      g0 += d;
      g1 += d * xs[i];
    }
    b0 -= (lr * g0) / n;
    b1 -= (lr * g1) / n;
    path.push({ b0, b1 });
  }
  return path;
}

/* The loss of a one-feature logistic model at a given (b0, b1). */
export function lossAt(xs, ys, b0, b1, kind) {
  const ps = xs.map((x) => sigmoid(b0 + b1 * x));
  return kind === 'ce' ? crossEntropy(ys, ps) : brier(ys, ps);
}

/* Gaussian probability density. */
export const gauss = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

/* Reliability bins: what the model promised against what happened.
 * Returns only bins that got at least one case, because drawing an empty bin at
 * zero invents a failure the model never had the chance to commit. */
export function reliability(ys, ps, bins = 10) {
  const out = [];
  for (let i = 0; i < bins; i++) {
    const lo = i / bins;
    const hi = (i + 1) / bins;
    const inBin = [];
    for (let k = 0; k < ps.length; k++) {
      const last = i === bins - 1;
      if (ps[k] >= lo && (last ? ps[k] <= hi : ps[k] < hi)) inBin.push(k);
    }
    if (!inBin.length) continue;
    out.push({
      lo,
      hi,
      n: inBin.length,
      predicted: inBin.reduce((s, k) => s + ps[k], 0) / inBin.length,
      observed: inBin.reduce((s, k) => s + ys[k], 0) / inBin.length,
    });
  }
  return out;
}
