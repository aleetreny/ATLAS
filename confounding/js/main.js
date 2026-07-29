/* Entry point.
 *
 * The guard here has an unusual job. Most articles in this atlas check a
 * browser reimplementation against a reference the generator computed, because
 * the page recomputes something expensive. This page recomputes almost
 * nothing: the expensive part is four million draws that already happened.
 *
 * What it can check, and what would actually go wrong, is different: that the
 * closed form the page prints is the closed form the page's own arithmetic
 * produces, that the cloud it draws really does give the estimates written
 * underneath it, and that every direction the prose states in words matches the
 * table it came from. The third one is the cheap check that has caught real
 * errors elsewhere in this atlas, where a sentence said "above" about a number
 * that was below.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initEngineWidget } from './engine-widget.js';
import { initWineWidget } from './wine-widget.js';
import { initCostWidget } from './cost-widget.js';
import { initWorldsWidget } from './worlds-widget.js';
import { slope, ols2 } from './statkit.js';

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
  const E = data.engine;

  /* ---- the closed form, recomputed here rather than trusted.
     This is the one identity on the page that the browser can verify from
     scratch, so it gets the tight tolerance: both sides are arithmetic on
     numbers that travelled at full precision. */
  const millsGap = (a) => (2 * a) / Math.sqrt(1 + a * a) * Math.sqrt(2 / Math.PI);
  const gap = millsGap(M.alpha);
  if (Math.abs(gap - E.gap_closed) > 1e-6) {
    problems.push(`the page recomputes the closed-form gap as ${gap.toFixed(8)} and the `
      + `export says ${E.gap_closed}`);
  }
  data.alpha_sweep.forEach((row) => {
    const want = M.gamma * millsGap(row.alpha);
    if (Math.abs(want - row.bias_closed) > 1e-6) {
      problems.push(`the closed-form bias at alpha ${row.alpha} is ${want.toFixed(6)} here `
        + `and ${row.bias_closed} in the export`);
    }
  });

  /* ---- the cloud the widget draws has to be the cloud the readout describes.
     Coordinates were rounded to four decimals on the way out, so the floor
     under this comparison is that rounding and not the estimator. */
  const C = data.cloud;
  const naive = (() => {
    const t = C.y.filter((_, i) => C.x[i] === 1);
    const c = C.y.filter((_, i) => C.x[i] === 0);
    return d3.mean(t) - d3.mean(c);
  })();
  if (Math.abs(naive - C.naive) > 5e-4) {
    problems.push(`the drawn cloud gives a difference in means of ${naive.toFixed(6)} and `
      + `the export says ${C.naive}`);
  }
  const adj = ols2(C.x, C.u, C.y);
  if (Math.abs(adj - C.adjusted) > 5e-4) {
    problems.push(`adjusting on the drawn cloud gives ${adj.toFixed(6)} and the export says `
      + `${C.adjusted}`);
  }

  /* ---- the real reversal, recomputed from the coordinates the page has.
     The generator measured on these same rounded coordinates on purpose, so a
     disagreement here is a bug and not a rounding story. */
  const B = data.wine.best;
  const pooled = slope(B.x, B.y);
  if (Math.abs(pooled - B.pooled) > 1e-6) {
    problems.push(`the pooled slope recomputed from the drawn points is ${pooled.toFixed(6)} `
      + `and the export says ${B.pooled}`);
  }
  for (let g = 0; g < 3; g++) {
    const xs = B.x.filter((_, i) => B.g[i] === g);
    const ys = B.y.filter((_, i) => B.g[i] === g);
    const s = slope(xs, ys);
    if (Math.abs(s - B.within[g]) > 1e-6) {
      problems.push(`cultivar ${g} slope is ${s.toFixed(6)} here and ${B.within[g]} in the `
        + `export`);
    }
    if (Math.sign(s) === Math.sign(pooled)) {
      problems.push(`the page says every cultivar disagrees with the pooled slope, and `
        + `cultivar ${g} agrees with it`);
    }
  }

  /* ---- the directions the prose states in words */
  if (!(E.naive > E.truth)) {
    problems.push('the page says the difference in means overstates the effect, and it does not');
  }
  const S = data.strata;
  for (let i = 1; i < S.length; i++) {
    if (Math.abs(S[i].bias) > Math.abs(S[i - 1].bias) + 1e-6) {
      problems.push(`more strata should never leave more bias, and going from ${S[i - 1].k} `
        + `to ${S[i].k} does`);
    }
  }
  const P = data.proxy;
  if (!(Math.abs(P[P.length - 1].bias) > Math.abs(P[0].bias))) {
    problems.push('the page says a noisier proxy corrects less, and the sweep disagrees');
  }
  const K = data.collider;
  if (!(Math.abs(K.naive - K.truth) < Math.abs(K.with_collider - K.truth))) {
    problems.push('the page says conditioning on the collider is worse than doing nothing, '
      + 'and the numbers say otherwise');
  }

  /* ---- the two worlds: the whole section rests on this being exact */
  const W = data.worlds;
  if (!(W.cov_max_diff < 1e-12)) {
    problems.push(`the two worlds are supposed to share a covariance matrix exactly and they `
      + `differ by ${W.cov_max_diff}`);
  }
  if (Math.abs(W.world_a.adjusted - W.world_b.adjusted) > 0.02) {
    problems.push('the two worlds should hand an analyst who adjusts the same number, and '
      + `they give ${W.world_a.adjusted} and ${W.world_b.adjusted}`);
  }
  if (Math.abs(W.world_a.truth - W.world_b.truth) < 0.1) {
    problems.push('the two worlds have the same answer, so the section has no subject');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the closed form checks out, the drawn clouds reproduce the estimates '
      + 'printed under them, and every direction the prose states matches its table');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/confounding.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('the confounding machine', () => initEngineWidget(data));
  safely('a reversal in real bottles', () => initWineWidget(data));
  safely('what the correction costs', () => initCostWidget(data));
  safely('two worlds, one dataset', () => initWorldsWidget(data));
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
