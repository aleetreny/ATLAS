/* Entry point: load the data, boot every widget in isolation, and then check
 * this page's arithmetic against the references the generator produced.
 *
 * The tolerances are not guesses. Each one is the rounding the JSON was written
 * with, times whatever the operation does to it, which means a guard here can
 * only stay quiet if the browser really is reproducing the measurement. A guard
 * that cannot fire is decoration, and this article is about numbers that flatter
 * themselves, so it would be a poor place to keep any.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initScrollyLine } from './scrolly-line.js';
import { initLineWidget } from './line-widget.js';
import { initCeilingWidget } from './ceiling-widget.js';
import { initEllipseWidget } from './ellipse-widget.js';
import { initDeepWidget } from './deep-widget.js';
import {
  zScores, modifiedZ, tukeyFences, zCeiling, selfMahalanobis, mahalanobis,
  colMeans, covariance, cholesky, chi2ppf, normPpf, d2Ceiling,
} from './statkit.js';

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
    fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
  }
}

function check(data, deep = false) {
  const problems = [];
  const some = (arr, n) => (deep ? arr : arr.filter((_, i) => i % Math.ceil(arr.length / n) === 0));

  /* ---- one column: every statistic the page prints, against numpy's */
  const mg = data.one_d.values;
  const zs = zScores(mg);
  const mz = modifiedZ(mg);
  const f = tukeyFences(mg);
  /* the file carries four decimals, so anything derived from it can differ by
     five in the fifth: that is the floor, and the guard sits just above it */
  const R4 = 6e-5;
  const R6 = 6e-7;
  const cmp = (a, b, tol, what) => {
    if (!(Math.abs(a - b) <= tol)) {
      problems.push(`${what}: ${a} here against ${b} offline (${Math.abs(a - b).toExponential(2)})`);
    }
  };
  cmp(zs.mean, data.one_d.mean, R4, 'the mean of the magnesium column');
  cmp(zs.sd, data.one_d.sd, R4, 'its standard deviation');
  cmp(mz.median, data.one_d.median, R4, 'its median');
  cmp(mz.mad, data.one_d.mad, R4, 'its median absolute deviation');
  cmp(f.q1, data.checks.quantiles.q1, R6, 'the lower quartile');
  cmp(f.q3, data.checks.quantiles.q3, R6, 'the upper quartile');
  cmp(f.hi, data.one_d.fence_hi, R4, "Tukey's upper fence");

  /* ---- the ceiling, which the generator got by pushing points to 1e9 */
  let worstCeil = 0;
  data.one_d.ceiling.rows.forEach((r) => {
    worstCeil = Math.max(worstCeil, Math.abs(zCeiling(r.n, 1) - r.measured));
  });
  data.one_d.ceiling.k_rows.forEach((r) => {
    worstCeil = Math.max(worstCeil, Math.abs(zCeiling(data.many_d.n, r.k) - r.measured));
  });
  if (worstCeil > R4) {
    problems.push(`the closed-form ceiling against the one measured by pushing points out: `
      + `differ by ${worstCeil.toExponential(2)}`);
  }

  /* ---- the bench, position by position: the whole curve, not its ends */
  const S = data.one_d.sub;
  let worstDrag = 0;
  let countMismatch = 0;
  [['one', S.one, false], ['two', S.two, true]].forEach(([, rows, withComp]) => {
    some(rows, 12).forEach((r) => {
      const i = rows.indexOf(r);
      const v = S.values.slice();
      v[S.suspect] = r.pos;
      if (withComp) v[S.companion] = S.comp_at;
      const z = zScores(v);
      const m = modifiedZ(v);
      const t = tukeyFences(v);
      worstDrag = Math.max(worstDrag,
        Math.abs(z.z[S.suspect] - r.z), Math.abs(m.z[S.suspect] - r.mz), Math.abs(t.hi - r.fence));
      const nz = z.z.filter((q) => Math.abs(q) > data.one_d.z_cut).length;
      const nm = m.z.filter((q) => Math.abs(q) > data.one_d.mz_cut).length;
      if (nz !== r.n_z || nm !== r.n_mz) countMismatch += 1;
      void i;
    });
  });
  if (worstDrag > R4) {
    problems.push(`the bench's curves: differ from the generator's by up to `
      + `${worstDrag.toExponential(2)}`);
  }
  if (countMismatch) {
    problems.push(`${countMismatch} slider position(s) where the number of bottles flagged here `
      + `is not the number the generator counted`);
  }

  /* ---- two columns */
  const A = data.two_d.points;
  const pair = selfMahalanobis(A);
  let worstPair = 0;
  data.checks.d2_pair.forEach((v, i) => {
    worstPair = Math.max(worstPair, Math.abs(v - pair.d2[i]));
  });
  if (worstPair > R6) {
    problems.push(`the squared distances on the pair: up to ${worstPair.toExponential(2)} apart`);
  }
  cmp(chi2ppf(data.two_d.joint_p, 2), data.two_d.chi_cut, 1e-3, "the ellipse's cutoff");
  cmp(normPpf(1 - (1 - data.two_d.marg_p) / 2), data.two_d.z_cut, 1e-3, "the margins' cutoff");

  /* the ray: the same suspect scored with and against itself */
  let worstRay = 0;
  some(data.two_d.ray.rows, 8).forEach((r) => {
    const both = selfMahalanobis(A.concat([r.point]));
    worstRay = Math.max(worstRay, Math.abs(both.d2[both.d2.length - 1] - r.d2_in));
    worstRay = Math.max(worstRay,
      Math.abs(mahalanobis([r.point], pair.mu, pair.S)[0] - r.d2_out));
  });
  if (worstRay > R4) {
    problems.push(`the in-fit against out-of-fit ray: up to ${worstRay.toExponential(2)} apart`);
  }

  /* ---- thirteen columns */
  const X = data.wine.matrix;
  const deep13 = selfMahalanobis(X);
  let worst13 = 0;
  data.checks.d2_13.forEach((v, i) => {
    worst13 = Math.max(worst13, Math.abs(v - deep13.d2[i]));
  });
  if (worst13 > R6) {
    problems.push(`the squared distances on all ${data.many_d.p} columns: up to `
      + `${worst13.toExponential(2)} apart`);
  }
  /* the identity, which is exact algebra rather than a measurement and is
     therefore the tightest guard on the page */
  const total = deep13.d2.reduce((a, b) => a + b, 0);
  const want = data.many_d.p * (data.many_d.n - 1);
  if (Math.abs(total - want) / want > 1e-10) {
    problems.push(`the squared distances sum to ${total} where the algebra says exactly ${want}`);
  }
  if (Math.max(...deep13.d2) > d2Ceiling(data.many_d.n)) {
    problems.push('a squared distance is above the ceiling, which cannot happen');
  }
  const L = cholesky(deep13.S);
  cmp(deep13.S.reduce((a, row, i) => a + row[i], 0), data.checks.cov13_trace, 1e-4,
    'the trace of the covariance');
  cmp(2 * L.reduce((a, row, i) => a + Math.log(row[i]), 0), data.checks.logdet13, 1e-5,
    'the log determinant of the covariance');
  cmp(chi2ppf(0.975, data.many_d.p), data.many_d.chi_cut, 1e-3, 'the chi-squared cutoff');

  /* ---- the planted batch, step by step */
  const CUT = chi2ppf(0.975, data.many_d.p);
  let worstMask = 0;
  let flagMismatch = 0;
  data.many_d.masking.rows.filter((r) => r.m > 0).forEach((r) => {
    const rows = X.concat(data.many_d.masking.planted.slice(0, r.m));
    const mu = colMeans(rows);
    const S2 = covariance(rows, mu);
    const d2 = mahalanobis(rows, mu, S2);
    const planted = d2.slice(X.length);
    worstMask = Math.max(worstMask,
      Math.abs(planted.reduce((a, b) => a + b, 0) / r.m - r.d2));
    const flagged = planted.filter((v) => v > CUT).length;
    const real = d2.slice(0, X.length).filter((v) => v > CUT).length;
    if (flagged !== r.flagged || real !== r.real_flagged) flagMismatch += 1;
  });
  if (worstMask > 2e-3) {
    problems.push(`the planted batch's own distances: up to ${worstMask.toExponential(2)} apart`);
  }
  if (flagMismatch) {
    problems.push(`${flagMismatch} planted-batch size(s) where this page flags a different number `
      + `of bottles than the generator did`);
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every number on this page matches the reference that owns it'
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/outliers.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('the push', () => initScrollyLine(data));
  safely('the bench', () => initLineWidget(data));
  safely('the ceiling', () => initCeilingWidget(data));
  safely('the ellipse', () => initEllipseWidget(data));
  safely('the planted batch', () => initDeepWidget(data));
  renderMath();
  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
