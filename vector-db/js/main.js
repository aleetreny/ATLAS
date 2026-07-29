/* Entry point.
 *
 * One guard here is the mechanical claim of the first section, and getting it
 * right took two tries. The first version asserted that the number of connected
 * pieces can only go up as the filter tightens, and the measurement says no:
 * the count runs 1, 6, 35, 101, 135, 92, 50 while the kept rows run 6000 down
 * to 60. It falls at the end for a reason that should have been obvious, since
 * a subgraph of 60 nodes cannot have more than 60 pieces. The count was never
 * the quantity the argument needs.
 *
 * What the argument needs is the SHARE of the surviving rows sitting in the
 * largest piece, because that is what bounds how much of the filtered set one
 * greedy walk can reach, and that one is monotone. So the guard checks it, plus
 * the two things that are true of any partition by construction.
 */
import { initFilterWidget } from './filter-widget.js';
import { initChurnWidget } from './churn-widget.js';
import { initHybridWidget } from './hybrid-widget.js';
import { initBytesWidget } from './bytes-widget.js';
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
  if (!ok) console.warn(`[vector-db] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  const rows = data.filter.rows.slice().sort((a, b) => b.selectivity - a.selectivity);

  rows.forEach((r, i) => {
    check(`filter at ${r.selectivity}: oversampling does not lose recall`,
      r.over_recall >= r.post_recall - 1e-9,
      `asking for more gives ${r.over_recall} against ${r.post_recall} for asking for ten`);
    check(`filter at ${r.selectivity}: oversampling costs more`,
      r.over_cost >= r.post_cost - 1e-9,
      `${r.over_cost} against ${r.post_cost}`);
    check(`filter at ${r.selectivity}: scanning the kept rows costs what it should`,
      Math.abs(r.pre_cost - r.kept) < 1.5,
      `${r.pre_cost} distances to scan ${r.kept} rows`);
    if (i > 0) {
      check(`filter at ${r.selectivity}: the reachable share only shrinks`,
        r.biggest <= rows[i - 1].biggest + 1e-9,
        `${r.biggest} of the kept rows in the largest piece at ${r.selectivity}, against `
        + `${rows[i - 1].biggest} at the looser ${rows[i - 1].selectivity}`);
    }
    check(`filter at ${r.selectivity}: the largest piece is a share`,
      r.biggest > 0 && r.biggest <= 1, `${r.biggest}`);
    /* a partition of the kept rows: at least one piece, never more than rows */
    check(`filter at ${r.selectivity}: the pieces partition the kept rows`,
      r.pieces >= 1 && r.pieces <= r.kept,
      `${r.pieces} pieces over ${r.kept} kept rows`);
    check(`filter at ${r.selectivity}: the largest piece fits in the kept rows`,
      Math.round(r.biggest * r.kept) >= r.kept / r.pieces - 1,
      `the largest of ${r.pieces} pieces holds ${Math.round(r.biggest * r.kept)} of `
      + `${r.kept} rows, below the average piece`);
  });

  /* deleting more can only make the marked index worse, never better */
  const ch = data.churn.rows.slice().sort((a, b) => a.deleted - b.deleted);
  const rises = ch.filter((r, i) => i > 0 && r.tombstone_recall > ch[i - 1].tombstone_recall + 0.02).length;
  check('deleting more does not improve the marked index', rises === 0,
    `${rises} steps of the deletion sweep raise recall, which would mean the tombstones help`);
  ch.forEach((r) => {
    if (r.rebuilt_recall == null) return;
    check(`at ${r.deleted} deleted, rebuilding is not worse than marking`,
      r.rebuilt_recall >= r.tombstone_recall - 0.02,
      `rebuilt ${r.rebuilt_recall} against marked ${r.tombstone_recall}`);
  });

  /* the fusion cannot beat the per query oracle, by construction */
  data.hybrid.rows.forEach((r) => {
    check(`hybrid at ${r.length} words: the oracle is an upper bound`,
      r.oracle >= Math.max(r.sparse, r.dense) - 1e-9,
      `the oracle scores ${r.oracle} and its own inputs reach `
      + `${Math.max(r.sparse, r.dense)}`);
  });

  /* rescoring more candidates cannot lose recall */
  const rr = data.bytes.rerank;
  const dips = rr.filter((r, i) => i > 0 && r.recall < rr[i - 1].recall - 1e-9).length;
  check('rescoring a wider list does not lose recall', dips === 0,
    `${dips} of ${rr.length} widths go backwards`);

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[vector-db] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/vectordb.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('the filter', () => initFilterWidget(data));
  safely('the deletions', () => initChurnWidget(data));
  safely('words and directions', () => initHybridWidget(data));
  safely('the bill', () => initBytesWidget(data));

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
