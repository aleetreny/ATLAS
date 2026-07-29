/* Entry point.
 *
 * The guard's job here is the one the article's own argument suggests. Five
 * estimators run in the browser on a sample the generator also ran them on, so
 * the first thing to check is that the two agree: an estimate printed under a
 * cloud has to be the estimate those points produce. The second is the
 * balancing property itself, recomputed from the covariates rather than read
 * out of the export, because that theorem is what the rest of the page stands
 * on.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initBalanceWidget } from './balance-widget.js';
import { initWeightsWidget } from './weights-widget.js';
import { initDrWidget } from './dr-widget.js';
import { initOverlapWidget } from './overlap-widget.js';
import { naive, regression, ipw, strata, matching, aipw, smd } from './statkit.js';

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
  const C = data.cloud;

  /* ---- the five estimators, recomputed here.
     The tolerance comes from the export and not from taste: covariates went
     out at four decimals and the fitted score at five, so a difference of a
     few thousandths is rounding and anything larger is a disagreement. */
  const TOL = 3e-3;
  const got = {
    naive: naive(C.X, C.Y),
    regression: regression(C.V, C.X, C.Y),
    ipw: ipw(C.X, C.Y, C.e_hat),
    ipw_true: ipw(C.X, C.Y, C.e_true),
    strata: strata(C.X, C.Y, C.e_hat),
    matching: matching(C.X, C.Y, C.e_hat),
    aipw: aipw(C.V, C.X, C.Y, C.e_hat),
  };
  Object.entries(got).forEach(([k, v]) => {
    const want = C.ref[k];
    if (want === undefined) return;
    if (!Number.isFinite(v) || Math.abs(v - want) > TOL) {
      problems.push(`the browser's ${k} on the drawn sample is ${v.toFixed(6)} and the `
        + `generator got ${want}`);
    }
  });

  /* ---- the fitted score has to be the fitted score, not a different fit */
  let worstE = 0;
  C.V.forEach((v, i) => {
    const z = C.coef.reduce((s, b, j) => s + b * v[j], C.intercept);
    worstE = Math.max(worstE, Math.abs(1 / (1 + Math.exp(-z)) - C.e_hat[i]));
  });
  if (worstE > 2e-4) {
    problems.push(`recomputing the fitted score from the exported coefficients differs by `
      + `${worstE.toExponential(2)}`);
  }

  /* ---- the balancing property, recomputed from the covariates.
     Weighting by the inverse probability is the cheapest version of the claim
     and the one the browser has the data for, so that is the one checked. */
  const before = smd(C.V, C.X);
  const w = C.X.map((x, i) => (x === 1 ? 1 / C.e_true[i] : 1 / (1 - C.e_true[i])));
  const after = smd(C.V, C.X, w);
  const worstBefore = Math.max(...before.map(Math.abs));
  const worstAfter = Math.max(...after.map(Math.abs));
  if (!(worstAfter < 0.5 * worstBefore)) {
    problems.push(`weighting by the true score should balance the covariates and the worst `
      + `standardised difference goes from ${worstBefore.toFixed(3)} to ${worstAfter.toFixed(3)}`);
  }

  /* ---- the directions the prose states */
  const B = data.balance;
  if (!(Math.max(...B.after_true.map(Math.abs)) < Math.max(...B.before.map(Math.abs)))) {
    problems.push('the page says the true score balances and the export disagrees');
  }
  if (!(Math.abs(B.after_wrong[1]) > 0.3 * Math.abs(B.before[1]))) {
    problems.push('the wrong score is supposed to leave the covariates it ignores where they '
      + 'were, and it does not, so the control is not a control');
  }
  const D = data.double_robust.cells;
  ['10', '01'].forEach((cell) => {
    const broken = cell === '10' ? 'regression' : 'ipw';
    if (!(Math.abs(D[cell].aipw.bias) < Math.abs(D[cell][broken].bias))) {
      problems.push(`in cell ${cell} the doubly robust estimator should beat ${broken} and `
        + `it does not`);
    }
  });
  if (!(Math.abs(D['00'].aipw.bias) > 4 * Math.abs(D['11'].aipw.bias))) {
    problems.push('the page says both models wrong breaks the doubly robust estimator too, '
      + 'and the table does not show it');
  }
  const H = data.hidden;
  if (!(Math.abs(H.bias) > 5 * H.se && H.smd_seen < 0.1)) {
    problems.push('the hidden covariate section needs a real bias next to clean balance, and '
      + `it has ${H.bias} with balance ${H.smd_seen}`);
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the five estimators reproduce the generator on the drawn sample, the true '
      + 'score balances the covariates when recomputed here, and every direction the prose '
      + 'states matches its table');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/propensity.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('one number, four covariates', () => initBalanceWidget(data));
  safely('four ways to spend one score', () => initWeightsWidget(data));
  safely('two chances instead of one', () => initDrWidget(data));
  safely('when there is nobody to compare with', () => initOverlapWidget(data));
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
