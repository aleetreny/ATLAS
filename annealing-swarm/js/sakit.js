/* The chain and the distribution, written without touching the DOM.
 *
 * Both also exist in src/utils/generate_annealing_data.py, including the random
 * number generator, so the reference run can be reproduced here step for step
 * and the guard can ask for equality rather than closeness.
 */

export class LCG {
  constructor(seed) { this.s = seed >>> 0; }

  nextU32() {
    this.s = (Math.imul(this.s, 1664525) + 1013904223) >>> 0;
    return this.s;
  }

  random() { return this.nextU32() / 4294967296; }

  randint(n) { return this.nextU32() % n; }
}

/* exp(-E/T) normalised over the whole ring: the distribution the chain is
 * supposed to be sampling, written out because on 64 states it can be. */
export function boltzmann(energy, T) {
  const w = energy.map((e) => Math.exp(-e / T));
  const z = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / z);
}

export function totalVariation(p, q) {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += Math.abs(p[i] - q[i]);
  return s / 2;
}

/* One Metropolis walk on the ring. Downhill is always taken; uphill is taken
 * with probability exp(-dE/T), which is the whole method. */
export function walk(energy, { T = 1, steps = 20000, seed = 1, T1 = null,
  keepPath = false } = {}) {
  const n = energy.length;
  const rng = new LCG(seed);
  const counts = new Array(n).fill(0);
  let i = rng.randint(n);
  const path = keepPath ? [i] : null;
  let accepted = 0;
  let uphillProposed = 0;
  let uphillAccepted = 0;
  for (let k = 0; k < steps; k++) {
    const temp = T1 === null ? T : T * (T1 / T) ** (k / steps);
    const j = (i + (rng.random() < 0.5 ? 1 : n - 1)) % n;
    const d = energy[j] - energy[i];
    let take = d <= 0;
    if (!take) {
      uphillProposed += 1;
      if (rng.random() < Math.exp(-d / temp)) { take = true; uphillAccepted += 1; }
    }
    if (take) { i = j; accepted += 1; }
    counts[i] += 1;
    if (keepPath) path.push(i);
  }
  const total = counts.reduce((a, b) => a + b, 0);
  return { visits: counts.map((c) => c / total), path, accepted,
    uphillProposed, uphillAccepted, final: i };
}

/* The two dimensional test function the swarm panel draws. */
export function rastrigin(x, y) {
  return 20 + (x * x - 10 * Math.cos(2 * Math.PI * x))
    + (y * y - 10 * Math.cos(2 * Math.PI * y));
}

/* The stability boundary of the classical swarm, from the recurrence with the
 * two attractors held still: past it the expected distance from the attractor
 * grows without bound. */
export const stabilityLimit = (w) => (24 * (1 - w * w)) / (7 - 5 * w);
