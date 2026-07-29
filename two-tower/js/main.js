/* Entry point.
 *
 * The guard is the live tower against the generator's own scoring: the page
 * recomputes a query from a fixed set of picks and has to return the same top
 * five, which is the check that the vectors are being read in the right order
 * and that the popularity term is not being added twice. And the rank ceiling
 * is checked as arithmetic, because the singular values that produced it were
 * exported and the curve has to be their tail sum.
 */
import { initColdWidget } from './cold-widget.js';
import { initCeilWidget } from './ceil-widget.js';
import { initTowerWidget } from './tower-widget.js';
import { initCostWidget } from './cost-widget.js';
import { initProse } from './prose.js';

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

function check(data, tower, loud = false) {
  const problems = [];
  const fail = (label, detail) => problems.push(`${label}: ${detail}`);

  /* 1. the live scoring against the generator's */
  if (tower) {
    let bad = 0;
    data.widget.check.forEach((row) => {
      const got = tower.score(new Set(row.picked), true)
        .filter((d) => !row.picked.includes(d.i))
        .sort((a, b) => b.s - a.s).slice(0, 5).map((d) => d.i);
      const want = row.top.filter((t) => !row.picked.includes(t)).slice(0, got.length);
      if (got.join(',') !== want.join(',')) bad += 1;
    });
    if (bad) {
      fail('the live tower', `${bad} of ${data.widget.check.length} query vectors return a `
        + 'different top five than the generator, so the item vectors are being read wrongly');
    }
  }

  /* 2. the ceiling is a tail sum of the singular values it exported, so the
     curve can be recomputed here and has to agree to the rounding of the file */
  const s = data.ceiling.singular;
  const total = s.reduce((a, v) => a + v * v, 0);
  const cells = data.ceiling.n_users * data.ceiling.n_items;
  let worst = 0;
  data.ceiling.rows.forEach((row) => {
    if (row.k > s.length) return;
    const keep = s.slice(0, row.k).reduce((a, v) => a + v * v, 0);
    /* the exported singular values are only the first 32, so the tail sum they
       reconstruct is a lower bound on the error unless the matrix has rank at
       most 32; the comparison is therefore one sided and stated as such */
    const got = Math.sqrt(Math.max(total - keep, 0) / cells);
    worst = Math.max(worst, got - row.rmse);
  });
  if (!(worst < 1e-6)) {
    fail('the ceiling', `recomputing it from the exported singular values overshoots by `
      + `${worst.toExponential(2)}, which cannot happen if both are tail sums of the same spectrum`);
  }

  /* 3. the cold start arm has to be scored against the right chance rate */
  const chance = 10 / data.arms.cold_books;
  if (Math.abs(chance - data.arms.chance_at_10) > 1e-9) {
    fail('the cold start control', `chance at ten should be ${chance} for ${data.arms.cold_books} `
      + `books and the file says ${data.arms.chance_at_10}`);
  }

  /* 4. the composed text */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .rank-table td,'
    + ' .widget-note, svg text').forEach((node) => {
    const t = node.textContent || '';
    if (/\bundefined\b|\bNaN\b|\[object Object\]|\$\{/.test(t)) bad.push(node.id || node.className);
  });
  if (bad.length) fail('composed text', `undefined or NaN appears in: ${bad.join(', ')}`);

  if (problems.length) problems.forEach((p) => console.warn(`[two-tower] ${p}`));
  else if (loud) {
    console.log('[two-tower] the page scores the catalogue the way the generator did, and the '
      + 'ceiling is the tail of the spectrum it was computed from');
  }
  return problems;
}

async function boot() {
  renderMath();
  const data = await fetch('./data/tower.json').then((r) => r.json());

  PROSE_IDS = safely('the numbered sentences', () => initProse(data)) || [];
  safely('the cold start', () => initColdWidget(data));
  safely('the ceiling', () => initCeilWidget(data));
  const tower = safely('the live tower', () => initTowerWidget(data));
  safely('the cost', () => initCostWidget(data));
  renderMath();

  const guards = () => safely('guards', () => check(data, tower, true));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 800);

  window.__atlasCheck = (loud) => check(data, tower, loud);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
