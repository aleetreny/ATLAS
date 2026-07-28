/* Entry point: render maths, load the data, boot every widget in isolation,
 * then check the page's own arithmetic against the generator's.
 *
 * The guards below are written so they CAN fire. Each one compares a number
 * this page computed against a number scikit-learn or the generator computed,
 * with a tolerance tight enough to catch a real disagreement and loose enough
 * to survive the four decimal places the JSON is written with. A guard whose
 * tolerance swallows the error it is looking for is decoration, and this atlas
 * has paid for that one already.
 */
import { initScrollyMds } from './scrolly-mds.js';
import { initMetricWidget, metricSummary } from './metric-widget.js';
import { initShepardWidget, scalingBase, scalingRun } from './shepard-widget.js';
import { initFactorWidget, factorFit } from './factor-widget.js';
import { initDigitsWidget, digitsSolve } from './digits-widget.js';
import { initProse } from './prose.js';
import {
  toRows, pairwise, doubleCentre, classicalMDS, covariance, symEigen, procrustes,
} from '../../assets/js/embed.js';

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

/* One widget throwing must not take the rest of the page down with it. */
function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(label, ok, detail) {
  if (!ok) console.warn(`[distances] ${label}: ${detail}`);
  return ok;
}

/* PCA scores of the shipped matrix, so the scrolly can claim the identity
 * rather than assert it. Population covariance at both ends. */
function pcaScores(X, k) {
  const S = covariance(X);
  const { values, vectors } = symEigen(S);
  const m = new Float64Array(X[0].length);
  for (let i = 0; i < X.length; i++) for (let j = 0; j < m.length; j++) m[j] += X[i][j] / X.length;
  const Z = X.map((row) => {
    const out = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let t = 0; t < row.length; t++) s += (row[t] - m[t]) * vectors[t][j];
      out[j] = s;
    }
    return out;
  });
  return { scores: Z, values };
}

function runGuards(data, computed, caches) {
  const W = data.wine;

  /* 1. the identity the whole first section is built on, recomputed here */
  const gap = procrustes(computed.mds2, computed.pca2).disparity;
  check('mds equals pca', gap < 1e-12,
    `the two configurations differ by ${gap.toExponential(2)} after alignment`);
  /* and the stronger claim, coordinate by coordinate with the signs pinned */
  let worst = 0;
  for (let i = 0; i < computed.mds2.length; i++) {
    for (let j = 0; j < 2; j++) {
      worst = Math.max(worst, Math.abs(Math.abs(computed.mds2[i][j]) - Math.abs(computed.pca2[i][j])));
    }
  }
  check('mds equals pca coordinatewise', worst < 1e-8,
    `worst coordinate differs by ${worst.toExponential(2)}, the file says ${W.mds_pca_gap}`);

  /* 2. every metric's spectrum against scikit-learn's distances */
  data.metrics.rows.forEach((row) => {
    if (!caches.metric[row.key]) caches.metric[row.key] = metricSummary(computed.X, row.key);
    const S = caches.metric[row.key];
    let we = 0;
    for (let i = 0; i < row.eigenvalues.length; i++) {
      we = Math.max(we, Math.abs(S.values[i] - row.eigenvalues[i]));
    }
    /* The file writes eigenvalues to four decimals, so 5e-5 of disagreement is
       the rounding and nothing else. The bound is set just above it: a real
       disagreement between two eigensolvers on a 178 by 178 matrix is orders
       of magnitude larger than this, so the guard can still fire. */
    check(`spectrum ${row.key}`, we < 2e-4,
      `eigenvalues differ from the generator's by up to ${we.toExponential(2)}, `
      + `which is more than the four decimals the file was written with`);
    check(`stress ${row.key}`, Math.abs(S.stress - row.stress2) < 5e-5,
      `stress is ${S.stress.toFixed(6)}, the file says ${row.stress2}`);
    check(`negative mass ${row.key}`, Math.abs(S.negMass - row.negative_mass) < 5e-5,
      `negative mass is ${S.negMass.toExponential(4)}, the file says ${row.negative_mass}`);
    check(`negative count ${row.key}`, S.negCount === row.neg_count,
      `${S.negCount} negative eigenvalues here, ${row.neg_count} in the file`);
  });

  /* 3. the SMACOF runs, metric and non-metric, at every setting of the slider */
  data.scaling.rows.forEach((row) => {
    if (!caches.scaling[row.gamma]) {
      caches.scaling[row.gamma] = scalingRun(data, row.gamma, computed.base);
    }
    const R = caches.scaling[row.gamma];
    check(`metric smacof ${row.gamma}`, Math.abs(R.metricStress - row.metric_stress) < 5e-5,
      `stress ${R.metricStress.toFixed(6)} against ${row.metric_stress}`);
    check(`nonmetric smacof ${row.gamma}`,
      Math.abs(R.nonmetricStress - row.nonmetric_stress) < 5e-5,
      `stress ${R.nonmetricStress.toFixed(6)} against ${row.nonmetric_stress}`);
    check(`metric disparity ${row.gamma}`,
      Math.abs(R.metricDisparity - row.metric_disparity) < 5e-5,
      `disparity ${R.metricDisparity.toFixed(6)} against ${row.metric_disparity}`);
    check(`nonmetric disparity ${row.gamma}`,
      Math.abs(R.nonmetricDisparity - row.nonmetric_disparity) < 5e-5,
      `disparity ${R.nonmetricDisparity.toFixed(6)} against ${row.nonmetric_disparity}`);
  });
  /* and the claim that the non-metric answer never moved at all */
  const first = caches.scaling[data.scaling.gammas[0]].nonmetric;
  let spread = 0;
  data.scaling.gammas.forEach((g) => {
    const Z = caches.scaling[g].nonmetric;
    for (let i = 0; i < Z.length; i++) {
      spread = Math.max(spread, Math.abs(Z[i][0] - first[i][0]), Math.abs(Z[i][1] - first[i][1]));
    }
  });
  check('nonmetric invariance', spread <= Math.max(data.scaling.nonmetric_spread, 1e-9),
    `the nine non-metric maps differ by up to ${spread.toExponential(2)}, `
    + `the file says ${data.scaling.nonmetric_spread}`);

  /* 4. both fits at every setting of the factor slider */
  data.factor.sweep.forEach((row) => {
    if (!caches.factor[row.sd]) caches.factor[row.sd] = factorFit(data, row.sd);
    const R = caches.factor[row.sd];
    check(`pca subspace sd=${row.sd}`, Math.abs(R.pcaGap - row.pca_gap) < 5e-5,
      `${R.pcaGap.toFixed(6)} against ${row.pca_gap}`);
    check(`fa subspace sd=${row.sd}`, Math.abs(R.faGap - row.fa_gap) < 5e-5,
      `${R.faGap.toFixed(6)} against ${row.fa_gap}`);
    let wp = 0;
    for (let i = 0; i < R.psi.length; i++) wp = Math.max(wp, Math.abs(R.psi[i] - row.psi_fa[i]));
    check(`fa uniquenesses sd=${row.sd}`, wp < 5e-4,
      `the private variances differ by up to ${wp.toExponential(2)}`);
    check(`pca loud share sd=${row.sd}`, Math.abs(R.pcaLoudShare - row.pca_loud_share) < 5e-4,
      `${R.pcaLoudShare.toFixed(6)} against ${row.pca_loud_share}`);
  });

  /* 5. the digits, both fits */
  const G = data.digits;
  check('digits negative mass', Math.abs(computed.digits.negMass - G.negative_mass) < 5e-4,
    `${computed.digits.negMass.toFixed(6)} against ${G.negative_mass}`);
  check('digits negative count', computed.digits.negCount === G.neg_count,
    `${computed.digits.negCount} against ${G.neg_count}`);
  check('digits classical stress',
    Math.abs(computed.digits.stressClassical - G.stress_classical) < 5e-4,
    `${computed.digits.stressClassical.toFixed(6)} against ${G.stress_classical}`);
  check('digits nonmetric stress',
    Math.abs(computed.digits.stressNonmetric - G.stress_nonmetric) < 5e-4,
    `${computed.digits.stressNonmetric.toFixed(6)} against ${G.stress_nonmetric}`);
  const dc = procrustes(toRows(G.coords), computed.digits.classical).disparity;
  check('digits classical map', dc < 1e-6,
    `the browser's classical map is ${dc.toExponential(2)} from the generator's`);

  /* 6. nothing on the page may read "undefined" or "NaN" */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .dist-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[distances] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/distances.json').then((r) => r.json());

  const X = toRows(data.wine.matrix);
  const D = pairwise(X, 'euclidean');
  const B = doubleCentre(D);
  const { coords: mds2, values } = classicalMDS(D, 2);
  const { scores: pca2 } = pcaScores(X, 2);
  const computed = { X, D, B, mds2, pca2, values, base: scalingBase(data) };
  computed.digits = digitsSolve(data);

  const caches = { metric: {}, scaling: {}, factor: {} };

  safely('prose', () => initProse(data));
  safely('the map out of a table', () => initScrollyMds(data, computed));
  safely('which distance', () => initMetricWidget(data, caches.metric));
  safely('metric against non-metric', () => initShepardWidget(data, computed.base, caches.scaling));
  safely('factor analysis', () => initFactorWidget(data, caches.factor));
  safely('the digits', () => initDigitsWidget(data, computed.digits));

  /* KaTeX again: prose.js writes into paragraphs after the first pass. */
  renderMath();

  const guards = () => safely('guards', () => runGuards(data, computed, caches));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 6000 });
  else setTimeout(guards, 1200);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
