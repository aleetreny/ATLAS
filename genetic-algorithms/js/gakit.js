/* The landscapes and the search, written without touching the DOM.
 *
 * Everything here also exists in src/utils/generate_ga_data.py, including the
 * random number generator, which is a linear congruential one written out on
 * both sides for exactly that reason: numpy's generator cannot be reproduced in
 * a browser, and a run whose numbers cannot be checked is a cartoon. Genomes
 * are bits and fitnesses are integers, so the guard in main.js compares the two
 * runs with no tolerance at all.
 */

/* s = (s*1664525 + 1013904223) mod 2^32 */
export class LCG {
  constructor(seed) { this.s = seed >>> 0; }

  nextU32() {
    this.s = (Math.imul(this.s, 1664525) + 1013904223) >>> 0;
    return this.s;
  }

  random() { return this.nextU32() / 4294967296; }

  randint(n) { return this.nextU32() % n; }
}

export const onemax = (x) => x.reduce((a, b) => a + b, 0);

export function royal(x, block) {
  let total = 0;
  for (let i = 0; i < x.length; i += block) {
    let all = 1;
    for (let j = 0; j < block; j++) all &= x[i + j];
    total += all ? block : 0;
  }
  return total;
}

/* A trap of k bits: the local gradient points at the all zeros corner, which
 * is worth k-1, and the optimum is the all ones corner, worth k. Every single
 * bit flip away from the optimum looks like a step in the right direction. */
export function trap(x, block) {
  let total = 0;
  for (let i = 0; i < x.length; i += block) {
    let u = 0;
    for (let j = 0; j < block; j++) u += x[i + j];
    total += u === block ? block : block - 1 - u;
  }
  return total;
}

export const trapLoose = (x, block, shuffle) =>
  trap(shuffle.map((i) => x[i]), block);

export function fitnessOf(kind, block, shuffle) {
  if (kind === 'onemax') return (x) => onemax(x);
  if (kind === 'royal') return (x) => royal(x, block);
  if (kind === 'trap_loose') return (x) => trapLoose(x, block, shuffle);
  return (x) => trap(x, block);
}

/* The same generational loop as the generator, asking the generator of random
 * numbers for exactly the same things in exactly the same order. That order is
 * part of the contract: get it wrong and the two runs drift apart while both
 * remain perfectly valid runs, which is the hardest kind of bug to see. */
export function runGA(cfg, fitness, { captureFirst = false } = {}) {
  const rng = new LCG(cfg.seed);
  const pop = cfg.pop;
  const n = cfg.bits;
  let P = [];
  for (let i = 0; i < pop; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = rng.randint(2);
    P.push(row);
  }
  let fit = P.map(fitness);
  const uniq = (Q) => new Set(Q.map((q) => q.join(''))).size;
  const history = [{ gen: 0, best: Math.max(...fit), mean: fit.reduce((a, b) => a + b, 0) / pop,
    unique: uniq(P) }];
  let capture = null;

  for (let g = 0; g < cfg.gens; g++) {
    const winners = [];
    for (let i = 0; i < pop; i++) {
      let bi = 0;
      let bf = -1;
      for (let t = 0; t < cfg.tour; t++) {
        const k = rng.randint(pop);
        if (fit[k] > bf) { bf = fit[k]; bi = k; }
      }
      winners.push(bi);
    }
    const Q = winners.map((w) => P[w].slice());
    const afterSelect = captureFirst && g === 0 ? Q.map((q) => q.slice()) : null;
    const cuts = [];
    if (cfg.cross === 'one_point') {
      for (let i = 0; i + 1 < pop; i += 2) {
        const c = 1 + rng.randint(n - 1);
        cuts.push(c);
        for (let j = c; j < n; j++) {
          const tmp = Q[i][j]; Q[i][j] = Q[i + 1][j]; Q[i + 1][j] = tmp;
        }
      }
    } else if (cfg.cross === 'uniform') {
      for (let i = 0; i + 1 < pop; i += 2) {
        for (let j = 0; j < n; j++) {
          if (rng.random() < 0.5) {
            const tmp = Q[i][j]; Q[i][j] = Q[i + 1][j]; Q[i + 1][j] = tmp;
          }
        }
      }
    }
    const afterCross = captureFirst && g === 0 ? Q.map((q) => q.slice()) : null;
    const flips = [];
    for (let i = 0; i < pop; i++) {
      for (let j = 0; j < n; j++) {
        if (rng.random() < cfg.pmut) { Q[i][j] ^= 1; if (g === 0) flips.push([i, j]); }
      }
    }
    let qfit = Q.map(fitness);
    /* one elite: the best of the previous generation replaces the worst child */
    let w = 0;
    for (let i = 1; i < pop; i++) if (qfit[i] < qfit[w]) w = i;
    let e = 0;
    for (let i = 1; i < pop; i++) if (fit[i] > fit[e]) e = i;
    Q[w] = P[e].slice();
    qfit[w] = fit[e];
    if (captureFirst && g === 0) {
      capture = { parents: P.map((p) => p.slice()), fit: fit.slice(), winners,
        afterSelect, afterCross, flips, children: Q.map((q) => q.slice()), cuts };
    }
    P = Q;
    fit = qfit;
    history.push({ gen: g + 1, best: Math.max(...fit),
      mean: fit.reduce((a, b) => a + b, 0) / pop, unique: uniq(P) });
  }
  return { history, population: P, fitness: fit, capture,
    finalBest: Math.max(...fit) };
}

/* Tour length for the travelling salesman panel: the same sum the generator
 * takes, so a tour drawn here can be scored against the file. */
export function tourLength(order, pts) {
  let total = 0;
  for (let i = 0; i < order.length; i++) {
    const a = pts[order[i]];
    const b = pts[order[(i + 1) % order.length]];
    total += Math.hypot(a[0] - b[0], a[1] - b[1]);
  }
  return total;
}
