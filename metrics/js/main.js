/* Entry point.
 *
 * The strongest guard on this page recomputes the area under the curve in the
 * browser, by both routes, from the raw scores the file ships. That is not a
 * comparison against the generator: it is the identity the second section is
 * about, re-derived where the reader is, and it has no tolerance to choose
 * because both sides are exact arithmetic on the same numbers.
 */
import { initRocWidget } from './roc-widget.js';
import { initThresholdWidget } from './th-widget.js';
import { initEnumWidget } from './enum-widget.js';
import { initBleuWidget } from './bleu-widget.js';
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
  if (!ok) console.warn(`[metrics] ${label}: ${detail}`);
  return ok;
}

function aucPairs(y, s) {
  const idx = y.map((_, i) => i).sort((a, b) => s[a] - s[b]);
  const ranks = new Array(y.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j += 1;
    const r = 0.5 * (i + j) + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = r;
    i = j + 1;
  }
  let R = 0;
  let n1 = 0;
  y.forEach((v, k) => { if (v === 1) { R += ranks[k]; n1 += 1; } });
  const n0 = y.length - n1;
  return (R - (n1 * (n1 + 1)) / 2) / (n1 * n0);
}

function aucTrapezoid(y, s) {
  const idx = y.map((_, i) => i).sort((a, b) => s[b] - s[a]);
  let tp = 0;
  let fp = 0;
  const P = y.reduce((a, b) => a + b, 0);
  const N = y.length - P;
  let area = 0;
  let prevFp = 0;
  let prevTp = 0;
  idx.forEach((i) => {
    if (y[i] === 1) tp += 1; else fp += 1;
    area += ((fp - prevFp) / N) * ((tp + prevTp) / (2 * P));
    prevFp = fp;
    prevTp = tp;
  });
  return area;
}

function runGuards(data) {
  const y = data.roc.y;
  const p = data.roc.p;

  /* the identity, re-derived here rather than trusted */
  const a = aucTrapezoid(y, p);
  const b = aucPairs(y, p);
  check('the two routes to the area agree', Math.abs(a - b) < 1e-9,
    `the trapezoids give ${a.toFixed(12)} and the pair count gives ${b.toFixed(12)}`);

  /* and it is the number the file shipped, to the precision the file has */
  const shipped = data.tasks.ship.models.logistic.auc;
  check('and it is the shipped area', Math.abs(a - shipped) < 2e-5,
    `the page recomputes ${a.toFixed(6)} and the file says ${shipped}`);

  /* a monotone map cannot move it: this is the claim of the whole section, so
     the browser applies one and checks */
  const squashed = p.map((v) => 1 / (1 + Math.exp(-Math.log(Math.max(v, 1e-9)
    / Math.max(1 - v, 1e-9)) / 4)));
  const c = aucPairs(y, squashed);
  check('squashing the scores leaves the area alone', Math.abs(c - b) < 1e-9,
    `${c.toFixed(12)} against ${b.toFixed(12)} after a monotone map`);

  /* the enumeration is complete: the number of ways to write n as an ordered
     sum of four non negative integers is the tetrahedral number */
  const n = data.enumerate.n;
  const expect = ((n + 1) * (n + 2) * (n + 3)) / 6;
  check('every confusion matrix is there', data.enumerate.matrices === expect,
    `${data.enumerate.matrices} against the ${expect} there are`);

  /* the symmetric metric has to be symmetric, exactly */
  check('the correlation coefficient is symmetric', data.enumerate.mcc_asym === 0,
    `swapping the classes moves it by ${data.enumerate.mcc_asym}`);

  /* BLEU on identical text is one, and on nothing at all is zero */
  const same = data.bleu.rows.find((r) => r.kind === 'identical');
  check('BLEU of identical text is one', Math.abs(same.corpus - 1) < 1e-9,
    `it is ${same.corpus}`);
  const bp = data.bleu.brevity.find((r) => Math.abs(r.frac - 1) < 1e-9);
  if (bp) {
    check('the brevity penalty is one at equal length', Math.abs(bp.bp - 1) < 1e-9,
      `it is ${bp.bp} when the candidate and the reference are the same length`);
  }

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[metrics] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/metrics.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('the curve', () => initRocWidget(data));
  safely('the thresholds', () => initThresholdWidget(data));
  safely('the enumeration', () => initEnumWidget(data));
  safely('BLEU', () => initBleuWidget(data));

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
