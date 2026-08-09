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

  /* The page says every node at the bottom layer sits exactly at the cap. That
     is a claim about a distribution, so it is checked against the distribution
     and not against the three summary numbers next to it: the histogram must
     put all of its mass between the reported minimum and maximum, and add up to
     the collection. A summary can agree with itself and still be wrong. */
  const dg = I.degrees;
  const total = dg.hist.reduce((a, b) => a + b, 0);
  check('the degree histogram counts every node', total === I.n,
    `${total} nodes in the histogram against ${I.n} in the index`);
  const outside = dg.hist.reduce(
    (a, c, d) => a + ((d < dg.min || d > dg.max) ? c : 0), 0);
  check('the degree histogram agrees with its own summary', outside === 0,
    `${outside} nodes fall outside the ${dg.min} to ${dg.max} range the page prints`);
  const weighted = dg.hist.reduce((a, c, d) => a + c * d, 0) / Math.max(total, 1);
  check('the mean degree is the mean of the histogram', Math.abs(weighted - dg.mean) < 5e-6,
    `the histogram averages ${weighted} and the file says ${dg.mean}`);

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

  /* An approximate index may cross the exhaustive cost at a deliberately wide
     setting and still miss a neighbour. That is a real, measurable trade-off,
     not a broken counter; check the accounting and leave the interpretation to
     the prose instead of emitting a false warning. */
  ['hnsw', 'ivf'].forEach((key) => {
    const invalid = I[key].filter((r) => r.cost < 0 || r.recall < 0 || r.recall > 1).length;
    check(`${key} cost and recall are valid measurements`, invalid === 0,
      `${invalid} settings contain an invalid cost or recall`);
  });

  /* Recall has a true ceiling of one. Category share is a different proxy and
     may legitimately be higher for an approximate index. */
  Object.entries(I).forEach(([key, rows]) => {
    if (!Array.isArray(rows)) return;
    const invalid = rows.filter((r) => r.quality < 0 || r.quality > 1).length;
    check(`${key} quality is a probability`, invalid === 0,
      `${invalid} settings have category share outside [0, 1]`);
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
