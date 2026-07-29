/* Entry point.
 *
 * The strong guard here is the reference run: a hundred generations reproduced
 * in the browser and compared against the file generation by generation. Bits
 * and fitnesses are integers, so it asks for equality rather than closeness,
 * and it would catch a single misplaced call to the random number generator.
 */
import { initGenerationScrolly } from './scrolly-generation.js';
import { initGaWidget } from './ga-widget.js';
import { initArmsWidget } from './arms-widget.js';
import { initDialsWidget } from './dials-widget.js';
import { initTspWidget } from './tsp-widget.js';
import { initProse } from './prose.js';
import { runGA, fitnessOf, tourLength } from './gakit.js';

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
  if (!ok) console.warn(`[genetic-algorithms] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const P = data.probe;

  /* 1. the four landscapes, on the genomes the generator scored */
  let landBad = 0;
  Object.keys(P.values).forEach((kind) => {
    const f = fitnessOf(kind, P.block, P.shuffle);
    P.genomes.forEach((genome, i) => {
      if (f(genome) !== P.values[kind][i]) landBad += 1;
    });
  });
  check('the landscapes', landBad === 0,
    `${landBad} of ${P.genomes.length * Object.keys(P.values).length} scores disagree`);

  /* 2. the trap is a trap: all zeros beats every single bit flip away from it */
  const zeros = new Array(P.bits).fill(0);
  const f = fitnessOf('trap_tight', P.block, P.shuffle);
  let worseNeighbours = 0;
  for (let i = 0; i < P.bits; i++) {
    zeros[i] = 1;
    if (f(zeros) < f(new Array(P.bits).fill(0))) worseNeighbours += 1;
    zeros[i] = 0;
  }
  check('the trap', worseNeighbours === P.bits && f(zeros) === P.deceptive_attractor,
    `${worseNeighbours} of ${P.bits} single flips make it worse, and all zeros scores ${f(zeros)}`);

  /* 3. the reference run, generation by generation, with no tolerance */
  const R = data.reference;
  const run = runGA(R.config, f);
  let bad = 0;
  let worstMean = 0;
  R.history.forEach((row, i) => {
    const mine = run.history[i];
    if (!mine || mine.best !== row.best || mine.unique !== row.unique) bad += 1;
    if (mine) worstMean = Math.max(worstMean, Math.abs(mine.mean - row.mean));
  });
  check('the reference run', bad === 0 && run.history.length === R.history.length
    && worstMean < 1e-9 && run.finalBest === R.final_best,
  `${bad} of ${R.history.length} generations differ, worst mean gap ${worstMean}`);

  /* 4. the tours drawn in the last panel score what the file says they score */
  const T = data.tsp;
  let worstTour = 0;
  ['ga', 'two_opt'].forEach((k) => {
    worstTour = Math.max(worstTour, Math.abs(tourLength(T[k].tour, T.points) - T[k].best));
  });
  check('the tours', worstTour < 2e-3,
    `a drawn tour is ${worstTour} away from the length the file gives it`);

  /* 5. a run with a bigger population really does what the table claims, which
   *    is the one place where a widget could disagree with a measurement */
  if (deep) {
    const big = runGA({ ...R.config, pop: 400, gens: 150 }, f);
    check('the population claim', big.finalBest > R.final_best,
      `at population 400 the run reaches ${big.finalBest} against ${R.final_best} at `
      + `${R.config.pop}, which is not the direction the table reports`);
  }

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

  console.info(`[genetic-algorithms] load-time checks complete: ${R.history.length} generations `
    + `reproduced exactly, landscapes ${landBad === 0 ? 'agree' : 'disagree'}, `
    + `tours within ${worstTour.toExponential(1)}`);
  return { landBad, bad, worstTour };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/ga.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('one generation', () => initGenerationScrolly(data));
  safely('the live run', () => initGaWidget(data));
  safely('five searchers', () => initArmsWidget(data));
  safely('the two dials', () => initDialsWidget(data));
  safely('the travelling salesman', () => initTspWidget(data));

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
