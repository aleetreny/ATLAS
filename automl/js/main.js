/* Entry point.
 *
 * The guards here recompute the quantities the page argues from, and one of
 * them is worth naming: the axis spans. They are maxima over the rest of the
 * catalogue, so a single mis-grouped configuration would move a bar without
 * moving anything else on the page, and nothing else would notice.
 */
import { initGainWidget } from './gain-widget.js';
import { initAxesWidget } from './axes-widget.js';
import { initCurseWidget } from './curse-widget.js';
import { initPortWidget } from './port-widget.js';
import { initEnsWidget } from './ens-widget.js';
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
  if (!ok) console.warn(`[automl] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  const cfgs = data.configs.map((c) => c.split('|'));
  const AX = { scaler: 0, reducer: 1, weight: 2, model: 3 };

  Object.entries(data.tables).forEach(([name, T]) => {
    check(`${name} has a score per pipeline`,
      T.cv.length === cfgs.length && T.test.length === cfgs.length,
      `${T.cv.length} cross validated and ${T.test.length} held out against `
      + `${cfgs.length} pipelines`);

    /* the axis spans, recomputed from the per pipeline scores */
    Object.entries(AX).forEach(([axis, k]) => {
      const levels = Array.from(new Set(cfgs.map((c) => c[k])));
      const best = levels.map((lv) => Math.max(
        ...cfgs.map((c, i) => (c[k] === lv ? T.cv[i] : -Infinity))));
      const span = Math.max(...best) - Math.min(...best);
      check(`${name}: how much ${axis} moves`,
        Math.abs(span - T.axes[axis].span) < 2e-6,
        `recomputed ${span.toFixed(6)} against the shipped ${T.axes[axis].span}`);
    });

    /* the winner, as a pipeline and not only as a value: two pipelines can tie
       on the score and the page names one of them */
    let bi = 0;
    T.cv.forEach((v, i) => { if (v > T.cv[bi]) bi = i; });
    check(`${name}: which pipeline won`,
      data.configs[bi] === T.best.config,
      `the page picks ${data.configs[bi]} and the file says ${T.best.config}`);
    check(`${name}: the default is in the catalogue`,
      data.configs.includes(data.meta.default),
      `${data.meta.default} is not one of the ${cfgs.length} pipelines`);
  });

  /* the portfolio has to be pipelines that exist, and the cold table must not
     be one of the tables it was learnt on */
  data.portfolio.picks.forEach((p) => check('a portfolio entry exists',
    data.configs.includes(p), `${p} is not in the catalogue`));
  check('the cold table stayed cold',
    !data.portfolio.learnt_on.includes(data.portfolio.cold),
    `${data.portfolio.cold} appears in the tables the list was learnt on`);

  /* The out of fold curve of a greedy ensemble is NOT monotone, and the first
     version of this guard said it was. Adding a member with equal weight
     changes the average rather than extending it, so the best available move at
     step k+1 can still be worse than where step k landed. The guard used to
     fire on correct data, which is the expensive way to have a guard; what it
     checks now is what is actually true of the construction. */
  Object.entries(data.ensemble).forEach(([name, E]) => {
    const distinct = E.rows.map((r) => r.distinct);
    check(`${name}: the ensemble never un-picks a pipeline`,
      distinct.every((v, i) => i === 0 || v >= distinct[i - 1]),
      'the count of distinct members went down, which greedy selection cannot do');
    check(`${name}: every step adds a pipeline that exists`,
      E.rows.every((r) => r.added >= 0 && r.added < cfgs.length),
      'a step added a configuration index outside the catalogue');
    check(`${name}: the size of the average is the step number`,
      E.rows.every((r, i) => r.k === i + 1),
      'the sizes are not consecutive');
  });

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[automl] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/automl.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('what the search bought', () => initGainWidget(data));
  safely('the four decisions', () => initAxesWidget(data));
  safely('the price of the catalogue', () => initCurseWidget(data));
  safely('the warm start', () => initPortWidget(data));
  safely('the ensemble', () => initEnsWidget(data));

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
