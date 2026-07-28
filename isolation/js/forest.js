/* The isolation forest, written to be the same forest the generator built.
 *
 * numpy's random stream cannot be reproduced in a browser, so the page and the
 * generator share a linear congruential generator instead, and the order of the
 * draws is part of the contract: one for the feature, one for the split, then
 * the left subtree, then the right. With that fixed, the browser rebuilds the
 * generator's trees exactly and the guard can compare scores rather than
 * distributions.
 *
 * No DOM in here, so node can import it and check it without a browser.
 */

/* s <- (1664525 s + 1013904223) mod 2^32 */
export class LCG {
  constructor(seed) {
    this.s = seed >>> 0;
  }

  nextU32() {
    /* Math.imul keeps the multiply exact in 32 bits, which a plain * does not:
       1664525 * 4294967295 is past 2^53 and the low bits are what matter. */
    this.s = (Math.imul(this.s, 1664525) + 1013904223) >>> 0;
    return this.s;
  }

  unit() {
    return this.nextU32() / 4294967296;
  }

  below(n) {
    return Math.floor(this.unit() * n);
  }
}

const EULER = 0.5772156649015329;

/* The average path length of an unsuccessful search in a binary search tree of
 * n keys: what a subtree of n points would have cost on average, and therefore
 * the right thing to credit an unsplit leaf with. */
export function cFactor(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + EULER) - (2 * (n - 1)) / n;
}

/* A partial Fisher-Yates, drawing exactly one number per position so that the
 * generator's stream and this one stay in step. */
export function sampleWithoutReplacement(n, k, rng) {
  const order = Array.from({ length: n }, (_, i) => i);
  const take = Math.min(k, n);
  for (let i = 0; i < take; i++) {
    const j = i + rng.below(n - i);
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  return order.slice(0, take);
}

function buildTree(X, idx, depth, limit, rng) {
  const n = idx.length;
  if (depth >= limit || n <= 1) return { leaf: n };
  const f = rng.below(X[0].length);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = X[idx[i]][f];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) return { leaf: n };
  const s = lo + rng.unit() * (hi - lo);
  const left = [];
  const right = [];
  for (let i = 0; i < n; i++) (X[idx[i]][f] < s ? left : right).push(idx[i]);
  if (left.length === 0 || right.length === 0) return { leaf: n };
  return {
    f,
    s,
    l: buildTree(X, left, depth + 1, limit, rng),
    r: buildTree(X, right, depth + 1, limit, rng),
  };
}

export function buildForest(X, nTrees, psi, seed) {
  const rng = new LCG(seed);
  const size = Math.min(psi, X.length);
  const limit = Math.ceil(Math.log2(Math.max(2, size)));
  const trees = [];
  for (let t = 0; t < nTrees; t++) {
    const take = sampleWithoutReplacement(X.length, psi, rng);
    trees.push(buildTree(X, take, 0, limit, rng));
  }
  return { trees, psi: size, limit };
}

export function pathLength(node, x) {
  let d = 0;
  let cur = node;
  while (cur.leaf === undefined) {
    cur = x[cur.f] < cur.s ? cur.l : cur.r;
    d += 1;
  }
  return d + cFactor(cur.leaf);
}

/* The published score: 2^(-E[h] / c(psi)). One half is "everybody agrees",
 * everything above it is "shorter to isolate than a random point". */
export function forestScore(forest, Q) {
  const c = cFactor(forest.psi);
  return Q.map((x) => {
    let h = 0;
    for (let t = 0; t < forest.trees.length; t++) h += pathLength(forest.trees[t], x);
    h /= forest.trees.length;
    return c > 0 ? 2 ** (-h / c) : 0;
  });
}

/* The path a single point takes down a single tree, for the scrolly: the cuts
 * it met and where it ended. */
export function tracePath(tree, x) {
  const cuts = [];
  let cur = tree;
  const box = { x0: -Infinity, x1: Infinity, y0: -Infinity, y1: Infinity };
  while (cur.leaf === undefined) {
    const goLeft = x[cur.f] < cur.s;
    cuts.push({ f: cur.f, s: cur.s, left: goLeft, box: { ...box } });
    if (cur.f === 0) {
      if (goLeft) box.x1 = Math.min(box.x1, cur.s);
      else box.x0 = Math.max(box.x0, cur.s);
    } else if (goLeft) box.y1 = Math.min(box.y1, cur.s);
    else box.y0 = Math.max(box.y0, cur.s);
    cur = goLeft ? cur.l : cur.r;
  }
  return { cuts, leafSize: cur.leaf, depth: cuts.length, box,
    length: cuts.length + cFactor(cur.leaf) };
}

/* Every cut of one tree, as segments clipped to the region each node owns.
 * Used to draw a single tree over the scatter rather than a score map. */
export function treeCuts(tree, bounds) {
  const out = [];
  const walk = (node, box, depth) => {
    if (node.leaf !== undefined) return;
    out.push({ f: node.f, s: node.s, box: { ...box }, depth });
    const left = { ...box };
    const right = { ...box };
    if (node.f === 0) {
      left.x1 = node.s;
      right.x0 = node.s;
    } else {
      left.y1 = node.s;
      right.y0 = node.s;
    }
    walk(node.l, left, depth + 1);
    walk(node.r, right, depth + 1);
  };
  walk(tree, { ...bounds }, 0);
  return out;
}
