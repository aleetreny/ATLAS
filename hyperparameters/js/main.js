/* Entry point.
 *
 * The guards on this page have an easy job and a hard one. The easy one is the
 * usual: the numbers the browser recomputes have to match the ones the
 * generator wrote. The hard one is that almost everything here is an argmax
 * over a table, and an argmax is not continuous, so the guard checks the CELL
 * and not only its value: two cells with the same score are not the same
 * answer, and the article's claims are about which cell a method lands on.
 */
import { initSurfaceWidget } from './surface-widget.js';
import { initSearchWidget } from './search-widget.js';
import { initBayesWidget } from './bayes-widget.js';
import { initHalvingWidget } from './halving-widget.js';
import { initCurseWidget } from './curse-widget.js';
import { initProse, PROSE_IDS } from './prose.js';

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

function check(label, ok, detail) {
  if (!ok) console.warn(`[hyperparameters] ${label}: ${detail}`);
  return ok;
}

function argmax2d(a) {
  let bi = 0;
  let bj = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      if (a[i][j] > a[bi][bj]) { bi = i; bj = j; }
    }
  }
  return [bi, bj];
}

function runGuards(data) {
  const S = data.surface;

  /* the cell, not just the value: the page draws a ring on the winner and the
     prose names it, so a tie broken differently on the two sides would show up
     as a ring in the wrong place with a perfectly correct number under it */
  const [bi, bj] = argmax2d(S.cv);
  check('the best cell by cross validation',
    bi === S.best.at_cv[0] && bj === S.best.at_cv[1],
    `the page finds [${bi}, ${bj}] and the file says [${S.best.at_cv}]`);
  const [ti, tj] = argmax2d(S.test);
  check('the best cell on held out data',
    ti === S.best.at_test[0] && tj === S.best.at_test[1],
    `the page finds [${ti}, ${tj}] and the file says [${S.best.at_test}]`);
  check('the two best values',
    Math.abs(S.cv[bi][bj] - S.best.cv) < 1e-9
    && Math.abs(S.test[ti][tj] - S.best.test) < 1e-9,
    'the maxima recomputed here differ from the ones in the file');

  /* the importance of each axis, which is the quantity the second section
     rests on, recomputed from the surface rather than trusted */
  const rowMax = S.cv.map((r) => Math.max(...r));
  const colMax = S.cv[0].map((_, j) => Math.max(...S.cv.map((r) => r[j])));
  const a0 = Math.max(...rowMax) - Math.min(...rowMax);
  const a1 = Math.max(...colMax) - Math.min(...colMax);
  check('axis importance', Math.abs(a0 - S.importance.axis0) < 2e-6
    && Math.abs(a1 - S.importance.axis1) < 2e-6,
    `recomputed ${a0.toFixed(6)} and ${a1.toFixed(6)} against `
    + `${S.importance.axis0} and ${S.importance.axis1}`);

  /* the plateau: the single number the closing paragraph is built on */
  const flat = S.cv.flat();
  const target = Math.max(...flat) - data.floor.cv_sd;
  const near = flat.filter((v) => v >= target).length / flat.length;
  check('the size of the plateau', Math.abs(near - data.gp.near_optimal) < 1e-6,
    `the page counts ${(near * 100).toFixed(2)}% of cells within one wobble of the best `
    + `and the file says ${(data.gp.near_optimal * 100).toFixed(2)}%`);

  /* the two biases, re-derived from the rows: selection has to be the piece
     that does NOT involve the held out column */
  data.curse.rows.forEach((r) => {
    check(`the split of the bias at b = ${r.b}`,
      Math.abs((r.cv - r.cv2) - r.selection) < 2e-6
      && Math.abs((r.cv - r.test) - r.gap) < 2e-6,
      'the shipped selection and gap do not equal the differences they claim to be');
  });

  /* halving's saving, which is arithmetic and therefore checkable exactly */
  const expect = data.halving.cost_full / data.halving.cost_halving;
  check('what halving saves', Math.abs(expect - data.halving.saving) < 1e-6,
    `${expect.toFixed(4)} against the shipped ${data.halving.saving}`);

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[hyperparameters] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/hyperparameters.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('the surface', () => initSurfaceWidget(data));
  safely('grid against random', () => initSearchWidget(data));
  safely('the surrogate', () => initBayesWidget(data));
  safely('halving', () => initHalvingWidget(data));
  safely('the winner\'s curse', () => initCurseWidget(data));

  renderMath();

  window.__atlasProseIds = PROSE_IDS;
  window.__atlasCheck = () => safely('guards', () => runGuards(data));
  const guards = () => window.__atlasCheck();
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
