/* Entry point.
 *
 * The guards check the two properties that make the rest of the page readable
 * and that a subtle bug would leave looking fine: that recall rises with the
 * dial on every index, and that no index ever costs more than exhaustive
 * search while returning less than it. Either of those failing would mean the
 * counter and the search disagree about what the search did.
 */
import { initConcWidget } from './conc-widget.js';
import { initTradeWidget } from './trade-widget.js';
import { initDemoWidget } from './demo-widget.js';
import { initAblWidget } from './abl-widget.js';
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
  if (!ok) console.warn(`[ann-search] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  const I = data.indices;

  [['hnsw', 'ef'], ['ivf', 'nprobe'], ['pq', 'm']].forEach(([key, knob]) => {
    const rows = I[key].slice().sort((a, b) => a[knob] - b[knob]);
    const dips = rows.filter((r, i) => i > 0 && r.recall < rows[i - 1].recall - 1e-9).length;
    check(`${key}: turning the dial up does not lose recall`, dips === 0,
      `${dips} of ${rows.length} steps go backwards, so the dial and the search disagree`);
    const costDips = rows.filter((r, i) => i > 0 && r.cost < rows[i - 1].cost - 1e-9).length;
    check(`${key}: turning the dial up costs more`, costDips === 0,
      `${costDips} of ${rows.length} steps get cheaper as the dial goes up`);
    rows.forEach((r) => {
      check(`${key}: recall is a fraction`, r.recall >= 0 && r.recall <= 1,
        `recall of ${r.recall}`);
    });
  });

  /* nothing may be both more expensive than looking at everything and worse
     than looking at everything: that combination means the counter is wrong */
  ['hnsw', 'ivf'].forEach((key) => {
    const silly = I[key].filter((r) => r.cost > I.brute_cost && r.recall < 0.999).length;
    check(`${key} never pays more than exhaustive search for less`, silly === 0,
      `${silly} settings cost more than the ${I.brute_cost} of exhaustive search and still `
      + 'miss neighbours');
  });

  /* the exact search sets the ceiling for the quality view, so nothing may
     beat it there either */
  Object.entries(I).forEach(([key, rows]) => {
    if (!Array.isArray(rows)) return;
    const over = rows.filter((r) => r.quality > I.exact_quality + 0.02).length;
    check(`${key} does not beat the exact answer on quality`, over === 0,
      `${over} settings score above the ${I.exact_quality} of exhaustive search`);
  });

  /* the asymmetric distance has to beat the symmetric one: keeping the query
     unquantised is strictly more information, and if it did not, the two were
     computed on different things */
  I.pq.forEach((r) => {
    check(`the product quantiser at m = ${r.m} keeps the query exact`,
      r.adc_err <= r.sdc_err + 1e-9,
      `the asymmetric error ${r.adc_err} is above the symmetric ${r.sdc_err}`);
  });

  /* the small graph the page draws has to be connected enough to walk: an
     entry point in its own component would make every walk a straight line */
  const D = data.demo;
  const deg = new Array(D.points.length).fill(0);
  D.edges.forEach(([a, b]) => { deg[a] += 1; deg[b] += 1; });
  check('every node in the drawn graph has an edge',
    deg.every((v) => v > 0), `${deg.filter((v) => v === 0).length} nodes are isolated`);

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[ann-search] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/annsearch.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('the concentration', () => initConcWidget(data));
  safely('the trade', () => initTradeWidget(data));
  safely('the walk', () => initDemoWidget(data));
  safely('the ablations', () => initAblWidget(data));

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
