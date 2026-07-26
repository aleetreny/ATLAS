/* The little machinery the KDA article runs live: RBF projections from the
 * shipped dual coefficients, 1-D projections onto any direction for the
 * reader's dial, the best single threshold of a 1-D projection, and simple
 * histogram binning. The page's accuracies are recomputed here and must
 * match the generator's at load time.
 */

/* z(x) = sum_i alpha_i k(x, x_i): the kernel discriminant's 1-D score. */
export function kdaScores(Xq, Xtrain, alpha, gamma) {
  const out = new Float64Array(Xq.length);
  for (let q = 0; q < Xq.length; q++) {
    let s = 0;
    for (let i = 0; i < Xtrain.length; i++) {
      const dx = Xq[q][0] - Xtrain[i][0];
      const dy = Xq[q][1] - Xtrain[i][1];
      s += alpha[i] * Math.exp(-gamma * (dx * dx + dy * dy));
    }
    out[q] = s;
  }
  return out;
}

export function accuracy(pred, y) {
  let ok = 0;
  for (let i = 0; i < y.length; i++) ok += pred[i] === y[i] ? 1 : 0;
  return ok / y.length;
}

/* Project points onto the direction at angle theta. */
export const project = (X, theta) => {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return X.map((p) => p[0] * c + p[1] * s);
};

/* The best single cut of a 1-D projection, judged on the same points:
 * the most charitable possible reading of a linear projection. */
export function bestCut(z, y) {
  const order = [...z.keys()].sort((a, b) => z[a] - z[b]);
  let n1Left = 0;
  const total1 = y.reduce((a, b) => a + b, 0);
  let best = { acc: 0, cut: z[order[0]] - 1 };
  /* sweep cut positions between consecutive sorted points */
  for (let r = 0; r <= order.length; r++) {
    const rightOnes = total1 - n1Left;
    const leftZeros = r - n1Left;
    const accA = (leftZeros + rightOnes) / y.length;        // right side = class 1
    const accB = 1 - accA;                                   // or the flip
    const acc = Math.max(accA, accB);
    if (acc > best.acc) {
      const lo = r === 0 ? z[order[0]] - 1 : z[order[r - 1]];
      const hi = r === order.length ? z[order[order.length - 1]] + 1 : z[order[r]];
      best = { acc, cut: (lo + hi) / 2, flip: accB > accA };
    }
    if (r < order.length) n1Left += y[order[r]];
  }
  return best;
}

/* Fixed-range histogram: bins counts per class. */
export function histo(z, y, lo, hi, nbins) {
  const b0 = new Array(nbins).fill(0);
  const b1 = new Array(nbins).fill(0);
  for (let i = 0; i < z.length; i++) {
    const t = Math.max(0, Math.min(nbins - 1, Math.floor((z[i] - lo) / (hi - lo) * nbins)));
    (y[i] ? b1 : b0)[t] += 1;
  }
  return { b0, b1 };
}
