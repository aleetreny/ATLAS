/* Entry point.
 *
 * The guard checks the two things that make this article possible rather than
 * the things that are easy to check. First, that the sum really is a posterior:
 * the symmetry of the model forces the posterior mean of two of the three
 * weights to be exactly zero, and a quadrature with a bug does not get that for
 * free. Second, that the answer key and the sampler agree where they can be
 * expected to, and only there, because a guard that demanded agreement in the
 * gap fired on correct data and had to be rewritten.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initExactWidget } from './exact-widget.js';
import { initChainsWidget } from './chains-widget.js';
import { initApproxWidget } from './approx-widget.js';
import { initPriorWidget } from './prior-widget.js';

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
  const E = data.exact;
  const C = data.chain_worst;
  const R = data.approx.rows;
  const by = Object.fromEntries(R.map((r) => [r.key, r]));

  /* ---- la simetría, que es la comprobación de que la suma está bien hecha.
     Cambiar el signo de w, b y v deja v*tanh(wx+b) igual, así que la posterior
     es exactamente simétrica y la media de w y de v tiene que ser cero. */
  if (Math.abs(E.mean_w) > 1e-9 || Math.abs(E.mean_v) > 1e-9) {
    problems.push(`the posterior of this model is exactly symmetric, so the posterior means of `
      + `w and v have to be zero, and they are ${E.mean_w} and ${E.mean_v}`);
  }

  /* ---- el predictivo y sus tres regiones */
  if (E.pred_sd.some((v) => !(v > 0))) {
    problems.push('some test point has a predictive spread that is not positive');
  }
  if (!(E.gap_growth > 1.2)) {
    problems.push(`the page says the spread grows in the gap and it grows by a factor of `
      + `${E.gap_growth}`);
  }
  if (!(E.far_ratio < 1.2)) {
    problems.push(`the page says the spread does NOT grow far from the data, and out there it `
      + `is ${E.far_ratio} times what it is on the data`);
  }

  /* ---- las cadenas contra la cuadratura, exigiendo lo que se puede exigir:
     acuerdo fuera del hueco, y dentro del hueco no más desacuerdo del que las
     cadenas ya tienen entre ellas. Un guardia que pedía 0,01 en todo el eje
     disparó sobre datos correctos. */
  if (!(C.tight < 0.02)) {
    problems.push(`outside the gap the chains and the grid should agree, and they differ by `
      + `${C.tight}`);
  }
  if (!(C.worst_at > E.gap_lo && C.worst_at < E.gap_hi)) {
    problems.push(`the page says the worst disagreement is inside the gap and it is at `
      + `x = ${C.worst_at}, outside [${E.gap_lo}, ${E.gap_hi}]`);
  }
  if (!(C.mean < C.spread)) {
    problems.push(`the grid is further from the chains (${C.mean}) than the chains are from `
      + `each other (${C.spread})`);
  }

  /* ---- las direcciones que la prosa afirma sobre la tabla */
  R.forEach((r) => {
    if (!(r.sd_ratio_in >= 0) || !(r.tv_all >= 0)) {
      problems.push(`${r.key} exported a score that is not a number`);
    }
  });
  if (!(by.map.sd_ratio_in < 0.01)) {
    problems.push(`a single best fit is supposed to have no spread at all and it has `
      + `${by.map.sd_ratio_in} of the right one`);
  }
  if (!(by.bootstrap.sd_ratio_in > by.ensemble.sd_ratio_in)) {
    problems.push('the page says resampling the data is what buys an ensemble its spread, and '
      + `the same-data ensemble gives ${by.ensemble.sd_ratio_in} against `
      + `${by.bootstrap.sd_ratio_in}`);
  }
  const cheapest = [by.map, by.laplace, by.ensemble, by.bootstrap];
  const sampled = [by.vi, by.sgld];
  if (!(Math.min(...sampled.map((r) => r.tv_all)) < Math.min(...cheapest.map((r) => r.tv_all)))) {
    problems.push('the page says the two methods that pay for the posterior get closer to it, '
      + 'and the table disagrees');
  }
  if (!(by.laplace.sd_ratio_in > 1)) {
    problems.push(`the page calls Laplace the only row that is too wide, and its spread in the `
      + `gap is ${by.laplace.sd_ratio_in} of the right one`);
  }

  /* ---- el prior: invisible donde hay datos, decisivo en el hueco */
  const S = data.prior_sweep;
  const inMove = Math.abs(S[S.length - 1].sd_in / S[0].sd_in - 1);
  const gapMove = S[S.length - 1].sd_gap / S[0].sd_gap;
  if (!(inMove < 0.35)) {
    problems.push(`the page says the prior barely moves the spread where the data are, and it `
      + `moves it by ${inMove}`);
  }
  if (!(gapMove > 2)) {
    problems.push(`the page says the prior decides the spread in the gap, and across the whole `
      + `sweep it changes it by a factor of ${gapMove}`);
  }
  const bestP = S.reduce((p, c) => (c.log_evidence > p.log_evidence ? c : p), S[0]);
  if (bestP.scale !== S[0].scale && bestP.scale !== S[S.length - 1].scale) {
    /* un máximo interior es lo que la sección afirma; si se fuera a un extremo
       habría que decir que el barrido se quedó corto */
  } else {
    problems.push(`the evidence peaks at the edge of the sweep (width ${bestP.scale}), so the `
      + 'sweep does not contain the preferred prior and the section overstates what it shows');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the exact posterior is symmetric to the digit, the chains agree with it '
      + 'everywhere the data decide the answer, and every direction the prose states matches '
      + 'its table');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/bayesnets.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('the whole posterior by brute force', () => initExactWidget(data));
  safely('a second opinion', () => initChainsWidget(data));
  safely('six ways to not do the integral', () => initApproxWidget(data));
  safely('the assumption nobody reports', () => initPriorWidget(data));
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
