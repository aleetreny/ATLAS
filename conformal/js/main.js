/* Entry point.
 *
 * The guard has an unusually strong version of its job available here, because
 * the central claim of the article is an identity rather than a measurement:
 * the rank the method uses and the coverage it produces are both closed forms,
 * so the browser can recompute them and demand exact agreement. What is left
 * to check with a tolerance is the live conformal interval, which the page
 * recomputes from the calibration scores every time the reader moves alpha.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initExactWidget } from './exact-widget.js';
import { initModelsWidget } from './models-widget.js';
import { initCondWidget } from './cond-widget.js';
import { initExchWidget } from './exch-widget.js';
import { initSetsWidget } from './sets-widget.js';
import { rankFor, quantileAt } from './conformalkit.js';

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

function check(data) {
  const problems = [];
  const M = data.meta;
  const X = data.exact;

  /* ---- the rank and the coverage formula, recomputed here.
     Both sides are integer arithmetic, so this is an identity and gets an
     exact comparison rather than a tolerance. */
  X.rows.forEach((row) => {
    const k = rankFor(row.n, M.alpha);
    if (k !== row.k) {
      problems.push(`the rank for n = ${row.n} is ${k} here and ${row.k} in the export`);
    }
    if (Math.abs(k / (row.n + 1) - row.formula) > 1e-12) {
      problems.push(`the coverage formula for n = ${row.n} is ${(k / (row.n + 1)).toFixed(12)} `
        + `here and ${row.formula} in the export`);
    }
    if (Math.abs(row.mean - row.formula) > 5e-4) {
      problems.push(`at n = ${row.n} the measured coverage is ${row.mean} and the formula `
        + `says ${row.formula}`);
    }
  });

  /* ---- the live interval, recomputed from the calibration scores.
     Scores travelled at five decimals, so anything past a thousandth of the
     interval width is a disagreement rather than rounding. */
  const C = data.cloud;
  C.ref.forEach((row) => {
    const q = quantileAt(C.cal_scores, row.alpha);
    if (Math.abs(q - row.q) > 1e-4) {
      problems.push(`the browser's quantile at alpha ${row.alpha} is ${q.toFixed(6)} and the `
        + `export says ${row.q}`);
    }
    const hit = C.test.y.filter((y, i) => Math.abs(y - C.test.pred[i]) <= q).length
      / C.test.y.length;
    if (Math.abs(hit - row.coverage) > 2e-3) {
      problems.push(`recomputing coverage at alpha ${row.alpha} gives ${hit.toFixed(4)} and `
        + `the export says ${row.coverage}`);
    }
  });

  /* ---- the directions the prose states */
  const MO = data.models.rows;
  const covs = MO.map((r) => r.coverage);
  if (Math.max(...covs) - Math.min(...covs) > 0.01) {
    problems.push('the page says all four models get the same coverage and they differ by '
      + `${(Math.max(...covs) - Math.min(...covs)).toFixed(4)}`);
  }
  const widths = MO.map((r) => r.width);
  if (!(Math.max(...widths) / Math.min(...widths) > 1.3)) {
    problems.push('the page says the widths separate and they do not');
  }
  const byC = Object.fromEntries(data.conditional.rows.map((r) => [r.key, r]));
  if (!(byC.plain.g0 - byC.plain.g1 > 0.1)) {
    problems.push('the two groups are supposed to be covered very differently by one '
      + 'interval, and they are not');
  }
  ['mondrian', 'cqr'].forEach((k) => {
    if (Math.abs(byC[k].g0 - byC[k].g1) > 0.03) {
      problems.push(`${k} is supposed to even out the two groups and gives ${byC[k].g0} `
        + `against ${byC[k].g1}`);
    }
  });
  const E = data.exchange;
  if (!(E.chrono.coverage < 1 - M.alpha - 0.05)) {
    problems.push('the page says splitting in time breaks coverage, and it does not');
  }
  if (Math.abs(E.shuffled.coverage - (1 - M.alpha)) > 0.02) {
    problems.push('the shuffled control does not recover coverage, so the cause is not the '
      + 'ordering');
  }
  const S = Object.fromEntries(data.classification.rows.map((r) => [r.key, r]));
  if (!(S.aps.cov > S.aps_rand.cov + 0.02)) {
    problems.push('the unrandomised adaptive sets are supposed to over-cover and they do not');
  }
  if (!(S.lac.size < S.aps_rand.size)) {
    problems.push('the simplest score is supposed to give the smallest sets and it does not');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the rank and the coverage formula recompute exactly, the live interval '
      + 'reproduces the generator on the drawn sample, and every direction the prose states '
      + 'matches its table');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/conformal.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('not about ninety, exactly ninety', () => initExactWidget(data));
  safely('the same guarantee for a useless model', () => initModelsWidget(data));
  safely('ninety percent of whom', () => initCondWidget(data));
  safely('the one assumption', () => initExchWidget(data));
  safely('when the prediction is a set', () => initSetsWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = () => check(data);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
