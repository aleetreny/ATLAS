/* Entry point.
 *
 * The guard checks the three things that would make this page wrong rather
 * than merely ugly: that the two estimators printed under the cloud are the
 * ones those points produce, that the closed forms the prose quotes are the
 * ones the page's own arithmetic gives, and that the two worlds really are
 * indistinguishable, since the whole last third of the article rests on that
 * being exact rather than approximate.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initLeverWidget } from './lever-widget.js';
import { initWeakWidget } from './weak-widget.js';
import { initLeakWidget } from './leak-widget.js';
import { initWorldsWidget } from './worlds-widget.js';
import { initLateWidget } from './late-widget.js';
import { ols, wald, firstStageF } from './ivkit.js';

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
  const C = data.cloud;

  /* ---- the two estimators on the drawn cloud */
  const gotOls = ols(C.x, C.y);
  const gotIv = wald(C.z, C.x, C.y);
  const gotF = firstStageF(C.z, C.x);
  if (Math.abs(gotOls - C.ols) > 1e-4) {
    problems.push(`the browser's least squares on the drawn cloud is ${gotOls.toFixed(6)} and `
      + `the export says ${C.ols}`);
  }
  if (Math.abs(gotIv - C.iv) > 1e-4) {
    problems.push(`the browser's ratio on the drawn cloud is ${gotIv.toFixed(6)} and the `
      + `export says ${C.iv}`);
  }
  if (Math.abs(gotF - C.f) > 1e-2) {
    problems.push(`the browser's first stage F is ${gotF.toFixed(3)} and the export says ${C.f}`);
  }

  /* ---- the closed form for the bias of least squares, recomputed */
  const closed = (M.gamma * M.delta) / (M.pi ** 2 + M.delta ** 2 + M.vex);
  if (Math.abs(closed - data.headline.ols_bias_closed) > 1e-6) {
    problems.push(`the page computes the closed-form bias as ${closed.toFixed(8)} and the `
      + `export says ${data.headline.ols_bias_closed}`);
  }

  /* ---- the leak divided by the lever, in every cell of the table */
  data.exclusion.forEach((row) => {
    row.cells.forEach((c) => {
      const want = c.kappa / row.pi;
      if (Math.abs(want - c.closed) > 1e-6) {
        problems.push(`the closed form at pi ${row.pi} and kappa ${c.kappa} is `
          + `${want.toFixed(6)} here and ${c.closed} in the export`);
      }
    });
  });

  /* ---- the two worlds, which the last third of the page depends on */
  const W = data.worlds;
  if (!(W.algebra_max_diff < 1e-12)) {
    problems.push(`the two worlds should share a covariance matrix exactly and differ by `
      + `${W.algebra_max_diff}`);
  }
  if (!(W.sampled_max_diff < 0.05)) {
    problems.push(`the simulated worlds disagree by ${W.sampled_max_diff}, which means the `
      + `exhibited parameters do not describe the world that was simulated`);
  }
  if (Math.abs(W.iv_a - W.iv_b) > 0.05) {
    problems.push(`the estimator is supposed to be unable to tell the two worlds apart and `
      + `it returns ${W.iv_a} and ${W.iv_b}`);
  }
  if (!(W.beta_lo < M.beta && M.beta < W.beta_hi)) {
    problems.push('the interval of compatible effects does not contain the true effect, '
      + 'which would mean the sweep is wrong rather than the point being made');
  }

  /* ---- the directions the prose states */
  const S = data.strength;
  const strongest = S[0];
  const weakest = S[S.length - 1];
  if (!(Math.abs(weakest.iv_median_bias) > Math.abs(strongest.iv_median_bias))) {
    problems.push('the page says a weaker lever biases the ratio more, and the sweep '
      + 'disagrees');
  }
  const means = data.seeds.map((s) => s.mean);
  const meds = data.seeds.map((s) => s.median);
  const spreadMean = Math.max(...means) - Math.min(...means);
  const spreadMed = Math.max(...meds) - Math.min(...meds);
  if (!(spreadMean > 5 * spreadMed)) {
    problems.push(`the page says the mean wanders between seeds while the median does not, `
      + `and they spread ${spreadMean.toFixed(3)} against ${spreadMed.toFixed(3)}`);
  }
  const L = data.late;
  if (Math.abs(L.wald - L.late) > 0.02) {
    problems.push(`the ratio should equal the compliers' effect and gives ${L.wald} against `
      + `${L.late}`);
  }
  if (Math.abs(L.ate - L.late) < 0.1) {
    problems.push('the population effect and the compliers effect coincide, so that section '
      + 'has nothing to show');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('both estimators reproduce on the drawn cloud, every closed form recomputes '
      + 'here, and the two worlds share a covariance matrix to machine precision');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/instruments.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('two ratios and a division', () => initLeverWidget(data));
  safely('an estimator with no mean', () => initWeakWidget(data));
  safely('a leak divided by the lever', () => initLeakWidget(data));
  safely('the interval the data cannot narrow', () => initWorldsWidget(data));
  safely('whose effect is it', () => initLateWidget(data));
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
