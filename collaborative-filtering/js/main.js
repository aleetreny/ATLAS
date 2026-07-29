/* Entry point.
 *
 * The guard that matters here is the one on the live similarities. The page
 * recomputes four definitions from shared counts and dot products, and the
 * generator exports, for a sample of pairs, the same four computed the other
 * way: straight from the columns of the rating matrix, without ever forming
 * the packed upper triangle the browser reads. So what is being checked is
 * exactly what can go wrong, which is the index arithmetic of that packing,
 * and it has a floor: the dot products travel rounded, so the tolerance is
 * derived from the decimals written rather than picked.
 */
import { initHoleWidget } from './hole-widget.js';
import { initSimWidget } from './sim-widget.js';
import { initRankWidget } from './rank-widget.js';
import { initSliceWidget } from './slice-widget.js';
import { initProse } from './prose.js';
import { makeSim } from './simkit.js';

let PROSE_IDS = [];

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
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

function check(data, loud = false) {
  const problems = [];
  const fail = (label, detail) => problems.push(`${label}: ${detail}`);

  /* 1. the four similarities, against the generator's own */
  const sim = makeSim(data.widget);
  const flavours = { count: sim.cosine, jaccard: sim.jaccard, rating: sim.rating,
    centred: sim.centred };
  /* The tolerance is absolute and it is derived, not chosen. The dot products
     travel with three decimals and the norms with four; the norms here are of
     order a hundred, so a rounded dot product moves a cosine by about
     5e-4/1e4 = 5e-8 and a rounded norm by about the same. 1e-6 leaves two
     orders of magnitude of margin and still fires on the failure this guard
     exists for, which is an index into the packed upper triangle being off by
     one and returning a different pair's number entirely.
     A RELATIVE tolerance would be the wrong instrument: a centred cosine near
     zero is a perfectly good measurement whose relative error is unbounded,
     and the first version of this guard duly fired on one at 1.1e-4. */
  let worst = 0;
  let worstWhat = '';
  data.widget.check.forEach((row) => {
    Object.entries(flavours).forEach(([key, fn]) => {
      const d = Math.abs(fn(row.i, row.j) - row[key]);
      if (d > worst) { worst = d; worstWhat = `${key} for pair ${row.i},${row.j}`; }
    });
  });
  if (!(worst < 1e-6)) {
    fail('the live similarities', `${worstWhat} differs from the generator's by ${worst.toExponential(2)}, `
      + 'which means the packed upper triangle is being read with the wrong index');
  }

  /* 2. the catalogue counts against the summary computed from the full array */
  const counts = data.shape.counts_sorted;
  const total = counts.reduce((a, b) => a + b, 0);
  let run = 0;
  let half = 0;
  for (let i = 0; i < counts.length; i++) {
    run += counts[i];
    if (run / total >= 0.5) { half = i + 1; break; }
  }
  if (half !== data.shape.half_books) {
    fail('the long tail', `the counts sent say ${half} books carry half the ratings and the summary `
      + `says ${data.shape.half_books}`);
  }
  if (total !== data.shape.train_ratings) {
    fail('the counts', `they sum to ${total} and the file says ${data.shape.train_ratings} training ratings`);
  }

  /* 3. the shrinkage cannot make a similarity bigger, ever */
  let grew = 0;
  for (let i = 0; i < Math.min(sim.n, 20); i++) {
    const a = sim.neighbours(i, 'centred', 0, 8);
    const b = sim.neighbours(i, 'centred', 100, 8);
    const byId = new Map(a.map((x) => [x.j, x.s]));
    b.forEach((x) => { if (byId.has(x.j) && x.s > byId.get(x.j) + 1e-9) grew += 1; });
  }
  if (grew > 0) fail('shrinkage', `${grew} similarities got larger when shrunk, which is impossible`);

  /* 4. the composed text */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .neigh-table td,'
    + ' .widget-note, svg text').forEach((node) => {
    const t = node.textContent || '';
    if (/\bundefined\b|\bNaN\b|\[object Object\]|\$\{/.test(t)) bad.push(node.id || node.className);
  });
  if (bad.length) fail('composed text', `undefined or NaN appears in: ${bad.join(', ')}`);

  if (problems.length) {
    problems.forEach((p) => console.warn(`[collaborative-filtering] ${p}`));
  } else if (loud) {
    console.log('[collaborative-filtering] the browser reproduces the generator\'s similarities, '
      + 'the catalogue counts add up, and shrinkage only ever shrinks');
  }
  return problems;
}

async function boot() {
  renderMath();
  const data = await fetch('./data/cf.json').then((r) => r.json());

  PROSE_IDS = safely('the numbered sentences', () => initProse(data)) || [];
  safely('the hole in the table', () => initHoleWidget(data));
  safely('what similar means', () => initSimWidget(data));
  safely('the baseline', () => initRankWidget(data));
  safely('where it stops working', () => initSliceWidget(data));
  renderMath();

  const guards = () => safely('guards', () => check(data, true));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 800);

  window.__atlasCheck = (loud) => check(data, loud);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
