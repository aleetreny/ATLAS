/* Gradient boosting, from scratch, in one dimension.
 *
 * Mirrors src/models/gradient_boosting.py. Everything here operates on a single
 * feature, because the point of the visuals is to watch the ensemble assemble
 * itself, and a stump on one feature is the smallest object that can show it.
 *
 * The second-order quantities XGBoost introduced are computed too, since the
 * article draws the split-gain calculation rather than only asserting it.
 */

/* A decision stump: one threshold, two constant leaves. The weakest learner
 * that is still a learner. */
export function fitStump(xs, residuals, { candidates = 40 } = {}) {
  const n = xs.length;
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  let best = null;

  for (let c = 1; c < candidates; c++) {
    const thr = lo + ((hi - lo) * c) / candidates;
    let sl = 0;
    let nl = 0;
    let sr = 0;
    let nr = 0;
    for (let i = 0; i < n; i++) {
      if (xs[i] <= thr) {
        sl += residuals[i];
        nl++;
      } else {
        sr += residuals[i];
        nr++;
      }
    }
    if (!nl || !nr) continue;
    const ml = sl / nl;
    const mr = sr / nr;
    // squared error after this split
    let sse = 0;
    for (let i = 0; i < n; i++) {
      const p = xs[i] <= thr ? ml : mr;
      sse += (residuals[i] - p) ** 2;
    }
    if (!best || sse < best.sse) best = { thr, left: ml, right: mr, sse, nl, nr };
  }
  return best;
}

export const stumpPredict = (stump, x) => (x <= stump.thr ? stump.left : stump.right);

/* Boost: start from a constant, then repeatedly fit a stump to what is still
 * wrong and add a shrunken version of it. Returns the whole history so the page
 * can replay the construction. */
export function boost(xs, ys, { rounds = 60, lr = 0.1, candidates = 40 } = {}) {
  const n = xs.length;
  const base = ys.reduce((a, b) => a + b, 0) / n;
  const pred = new Array(n).fill(base);
  const stumps = [];
  const history = [];

  for (let m = 0; m < rounds; m++) {
    const residuals = ys.map((v, i) => v - pred[i]);
    const stump = fitStump(xs, residuals, { candidates });
    if (!stump) break;
    for (let i = 0; i < n; i++) pred[i] += lr * stumpPredict(stump, xs[i]);
    stumps.push(stump);
    let sse = 0;
    for (let i = 0; i < n; i++) sse += (ys[i] - pred[i]) ** 2;
    history.push({
      round: m + 1,
      stump,
      residualsBefore: residuals,
      rmse: Math.sqrt(sse / n),
    });
  }
  return { base, stumps, history, lr };
}

/* Evaluate an ensemble truncated to the first m stumps. */
export function ensemblePredict(model, xq, m = Infinity) {
  const k = Math.min(m, model.stumps.length);
  return xq.map((x) => {
    let v = model.base;
    for (let i = 0; i < k; i++) v += model.lr * stumpPredict(model.stumps[i], x);
    return v;
  });
}

/* ------------------------------------------------- the second-order picture
 *
 * XGBoost approximates the loss to second order, which turns the question
 * "what should this leaf output?" into arithmetic instead of a search. For
 * squared error the gradient of a residual r is g = -r and the hessian is
 * h = 1, so for a leaf holding a set of points:
 *
 *     optimal weight   w* = -G / (H + lambda)
 *     value of a leaf  -0.5 * G^2 / (H + lambda)
 *
 * and a split is worth making when splitting the leaf beats keeping it, minus
 * a fixed toll gamma charged per new leaf.
 */
export function leafStats(residuals) {
  let G = 0;
  let H = 0;
  for (const r of residuals) {
    G += -r; // gradient of 0.5*(y - p)^2 with respect to the prediction
    H += 1;
  }
  return { G, H };
}

export const leafWeight = (G, H, lambda) => -G / (H + lambda);
export const leafScore = (G, H, lambda) => (G * G) / (H + lambda);

export function splitGain(leftRes, rightRes, lambda, gamma) {
  const L = leafStats(leftRes);
  const R = leafStats(rightRes);
  const P = { G: L.G + R.G, H: L.H + R.H };
  const gain = 0.5 * (leafScore(L.G, L.H, lambda) + leafScore(R.G, R.H, lambda) - leafScore(P.G, P.H, lambda)) - gamma;
  return { gain, L, R, P, wL: leafWeight(L.G, L.H, lambda), wR: leafWeight(R.G, R.H, lambda) };
}

/* ----------------------------------------------------------------- helpers */

export function rmse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return Math.sqrt(s / yTrue.length);
}

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* The teaching curve: enough structure that a staircase of stumps visibly has
 * to work for it, and smooth enough that the staircase looks wrong at first. */
export const TRUTH = (x) => Math.sin(x * 1.4) * 2.2 + x * 0.55;

export function makeData(n = 90, seed = 5) {
  const rand = rng(seed);
  const gauss = () => {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 9.4 + 0.3 * (rand() - 0.5);
    xs.push(x);
    ys.push(TRUTH(x) + gauss() * 0.75);
  }
  return { xs, ys };
}
