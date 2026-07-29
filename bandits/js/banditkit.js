/* The machines and the policies, written without touching the DOM.
 *
 * The same code exists in src/utils/generate_bandits_data.py, generator of
 * random numbers included, so the reference run can be replayed here pull by
 * pull and the guard can compare the two exactly.
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

/* One run of the epsilon greedy rule, recording everything a picture needs:
 * which arm was pulled, what came back, and the regret so far, which is exact
 * because the probabilities behind the machines are written down. */
export function runBandit(arms, { seed = 1, pulls = 400, eps = 0.1 } = {}) {
  const rng = new LCG(seed);
  const k = arms.length;
  const best = Math.max(...arms);
  const counts = new Array(k).fill(0);
  const sums = new Array(k).fill(0);
  const history = [];
  let regret = 0;
  for (let t = 0; t < pulls; t++) {
    let a;
    if (t < k) a = t;
    else if (rng.random() < eps) a = rng.randint(k);
    else {
      a = 0;
      let bestMean = -Infinity;
      for (let i = 0; i < k; i++) {
        const m = counts[i] ? sums[i] / counts[i] : 0;
        if (m > bestMean) { bestMean = m; a = i; }
      }
    }
    const reward = rng.random() < arms[a] ? 1 : 0;
    counts[a] += 1;
    sums[a] += reward;
    regret += best - arms[a];
    history.push({ t, arm: a, reward, regret });
  }
  return { history, counts, sums, regret,
    means: counts.map((c, i) => (c ? sums[i] / c : 0)) };
}

/* The divergence between two coins, which is what the lower bound is written
 * in: the gap over it is how many pulls a machine has to be given before it can
 * be told apart from the best one. */
export function kl(p, q) {
  const e = 1e-12;
  const a = Math.min(Math.max(p, e), 1 - e);
  const b = Math.min(Math.max(q, e), 1 - e);
  return a * Math.log(a / b) + (1 - a) * Math.log((1 - a) / (1 - b));
}

export function laiRobbins(arms, T) {
  const best = Math.max(...arms);
  let total = 0;
  arms.forEach((a) => { if (a < best) total += ((best - a) / kl(a, best)) * Math.log(T); });
  return total;
}

/* The length the textbook picks for the exploring phase: it minimises an upper
 * bound on the regret, which is not the same thing as minimising the regret,
 * and the sweep in the page measures the difference. */
export function etcFormula(delta, T) {
  const inner = (T * delta * delta) / 4;
  if (inner <= 1) return 1;
  return Math.max(1, Math.round((4 / (delta * delta)) * Math.log(inner)));
}
