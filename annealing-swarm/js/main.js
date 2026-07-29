/* Entry point.
 *
 * Three guards worth having. The reference chain is reproduced step for step,
 * which is an equality check because the landscape is a list of numbers and the
 * generator of random numbers is written out on both sides. The Boltzmann
 * distribution is recomputed with a tolerance derived from the rounding of the
 * file rather than guessed. And the stability boundary is recomputed from its
 * formula instead of read.
 */
import { initMetropolisScrolly } from './scrolly-metropolis.js';
import { initBoltzmannWidget } from './boltzmann-widget.js';
import { initScheduleWidget } from './schedule-widget.js';
import { initSwarmWidget } from './swarm-widget.js';
import { initStabilityWidget } from './stability-widget.js';
import { initProse } from './prose.js';
import { walk, boltzmann, totalVariation, stabilityLimit } from './sakit.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
}

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(label, ok, detail) {
  if (!ok) console.warn(`[annealing-swarm] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const R = data.reference;
  const E = R.energy;

  /* 1. the reference chain, step for step */
  const run = walk(E, { T: R.config.T0, T1: R.config.T1, steps: R.config.steps,
    seed: R.config.seed, keepPath: true });
  const mine = run.path.filter((_, i) => i % R.path_stride === 0);
  const same = mine.length === R.path.length && mine.every((v, i) => v === R.path[i]);
  check('the reference chain', same && run.accepted === R.accepted
    && run.uphillAccepted === R.uphill_accepted,
  `${mine.length} of ${R.path.length} positions compared, ${run.accepted} accepted here `
    + `against ${R.accepted} in the file`);

  /* 2. the exact distribution.
   *
   * The tolerance is arithmetic rather than taste: the energies travel rounded
   * to five decimals, and a weight divides that by T, so the error in a
   * probability is about its own size times 5e-6 over T, plus the half unit the
   * probabilities themselves were rounded to. */
  let worstBoltz = 0;
  let worstTol = 0;
  data.boltzmann.rows.forEach((row) => {
    const p = boltzmann(data.boltzmann.energy, row.T);
    const gap = Math.max(...p.map((v, i) => Math.abs(v - row.exact[i])));
    const tol = Math.max(...p) * (5e-6 / row.T) + 5e-7;
    worstBoltz = Math.max(worstBoltz, gap / tol);
    worstTol = Math.max(worstTol, gap);
  });
  check('the exact distribution', worstBoltz <= 1,
    `the worst entry is ${worstBoltz.toFixed(2)} times its own rounding budget`);

  /* 3. a chain run here lands on that distribution */
  if (deep) {
    const T = 1.0;
    const p = boltzmann(data.boltzmann.energy, T);
    const chain = walk(data.boltzmann.energy, { T, steps: 200000, seed: 4242 });
    const tv = totalVariation(chain.visits, p);
    /* the floor is what the generator measured for a chain of this length */
    const floor = data.boltzmann.convergence.distance[
      data.boltzmann.convergence.steps.indexOf(200000)] * 3;
    check('a fresh chain', tv < floor,
      `distance ${tv.toFixed(4)} against a budget of ${floor.toFixed(4)}`);
  }

  /* 4. the stability boundary, recomputed */
  let worstLimit = 0;
  data.stability.rows.forEach((row) => {
    worstLimit = Math.max(worstLimit, Math.abs(stabilityLimit(row.w) - row.limit));
  });
  check('the boundary', worstLimit < 1e-4,
    `recomputing it moves it by ${worstLimit}`);

  /* 5. the cities really are the previous article's */
  check('the shared instance', data.schedules.cities.length === data.schedules.n
    && data.schedules.same_instance,
  'the tour instance is not the one the file claims');

  /* 6. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info('[annealing-swarm] load-time checks complete: reference chain reproduced '
    + `${same ? 'exactly' : 'WRONG'}, distribution within ${worstTol.toExponential(1)}, `
    + `boundary within ${worstLimit.toExponential(1)}`);
  return { same, worstBoltz, worstLimit };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/anneal.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the walk', () => initMetropolisScrolly(data));
  safely('the distribution', () => initBoltzmannWidget(data));
  safely('the schedules', () => initScheduleWidget(data));
  safely('the swarm', () => initSwarmWidget(data));
  safely('the boundary', () => initStabilityWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
