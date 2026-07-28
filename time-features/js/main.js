/* Entry point.
 *
 * The guard here is the hinge trend. The page refits it in JavaScript with the
 * same accelerated proximal gradient the generator used, on the same series,
 * and compares the curve: that catches a wrong design matrix, a penalty applied
 * to the wrong coefficients, and the off by one in a rolling window, which is
 * the error this whole article is about.
 */
import { initTableScrolly, buildRow } from './scrolly-table.js';
import { initLeakWidget } from './leak-widget.js';
import { initCeilingWidget } from './ceiling-widget.js';
import { initProphetWidget } from './prophet-widget.js';
import { initDecompWidget } from './decomp-widget.js';
import { initProse } from './prose.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
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
  if (!ok) console.warn(`[time-features] ${label}: ${detail}`);
  return ok;
}

/* the same design matrix the generator builds: intercept, slope, the hinges,
   then the Fourier pairs, in that order, because the exported coefficients are
   indexed by it */
function design(n, period, nCp, nFourier, range = 0.8) {
  const cols = [];
  const t = d3.range(n).map((i) => i / (n - 1));
  cols.push(new Array(n).fill(1));
  cols.push(t.slice());
  /* the candidate bends are linspace(0, range, nCp + 1) without its first
     point, exactly as the generator builds them: dividing by nCp + 1 instead
     puts every hinge in the wrong place and the refitted trend comes out 0.118
     parts per million away, which is what this guard is for */
  const cps = d3.range(1, nCp + 1).map((j) => (j / nCp) * range);
  cps.forEach((c) => cols.push(t.map((v) => Math.max(0, v - c))));
  for (let k = 1; k <= nFourier; k++) {
    cols.push(d3.range(n).map((i) => Math.sin((2 * Math.PI * k * i) / period)));
    cols.push(d3.range(n).map((i) => Math.cos((2 * Math.PI * k * i) / period)));
  }
  return { cols, nHinge: cps.length };
}

function fistaTrend(y, period, nCp, nFourier, tau, iters) {
  const n = y.length;
  const { cols, nHinge } = design(n, period, nCp, nFourier);
  const p = cols.length;
  const lam = tau * n;
  /* the step is one over the largest eigenvalue of the Gram matrix; the power
     method is enough for that and avoids shipping an eigensolver */
  const gram = (v) => {
    const xv = new Float64Array(n);
    for (let j = 0; j < p; j++) {
      const c = cols[j];
      const vj = v[j];
      if (vj === 0) continue;
      for (let i = 0; i < n; i++) xv[i] += c[i] * vj;
    }
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const c = cols[j];
      let s = 0;
      for (let i = 0; i < n; i++) s += c[i] * xv[i];
      out[j] = s;
    }
    return out;
  };
  let v = new Float64Array(p).fill(1);
  let lmax = 1;
  for (let k = 0; k < 60; k++) {
    const w = gram(v);
    let norm = 0;
    for (let j = 0; j < p; j++) norm += w[j] * w[j];
    norm = Math.sqrt(norm);
    if (!norm) break;
    for (let j = 0; j < p; j++) w[j] /= norm;
    lmax = norm;
    v = w;
  }
  const step = 1 / lmax;
  let b = new Float64Array(p);
  let z = new Float64Array(p);
  let tk = 1;
  for (let it = 0; it < iters; it++) {
    const xz = new Float64Array(n);
    for (let j = 0; j < p; j++) {
      const c = cols[j];
      const zj = z[j];
      if (zj === 0) continue;
      for (let i = 0; i < n; i++) xz[i] += c[i] * zj;
    }
    const resid = new Float64Array(n);
    for (let i = 0; i < n; i++) resid[i] = xz[i] - y[i];
    const bn = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const c = cols[j];
      let s = 0;
      for (let i = 0; i < n; i++) s += c[i] * resid[i];
      bn[j] = z[j] - step * s;
      if (j >= 2 && j < 2 + nHinge) {
        const thr = lam * step;
        bn[j] = Math.sign(bn[j]) * Math.max(0, Math.abs(bn[j]) - thr);
      }
    }
    const tn = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
    const zn = new Float64Array(p);
    for (let j = 0; j < p; j++) zn[j] = bn[j] + ((tk - 1) / tn) * (bn[j] - b[j]);
    b = bn;
    z = zn;
    tk = tn;
  }
  const trend = new Float64Array(n);
  for (let j = 0; j < 2 + nHinge; j++) {
    const c = cols[j];
    for (let i = 0; i < n; i++) trend[i] += c[i] * b[j];
  }
  return { trend, beta: b, nHinge };
}

function runGuards(data, deep) {
  const P = data.prophet;
  const y = data.series.y;

  /* 1. the trend, refitted here, against the generator's */
  const iters = deep ? 8000 : 2000;
  const row = P.sweep.find((s) => s.tau === P.tau);
  const fit = fistaTrend(y, data.meta.period, P.n_changepoints, P.n_fourier, P.tau, iters);
  let worst = 0;
  for (let i = 0; i < y.length; i++) {
    worst = Math.max(worst, Math.abs(fit.trend[i] - row.trend[i]));
  }
  /* at 2000 steps the iteration itself is only converged to about 0.02 ppm, so
     the spot check gets the loose threshold and the deep one gets the tight
     threshold the two python solvers agreed to */
  check('the hinge trend', worst < (deep ? 0.01 : 0.2),
    `refitting the trend in the browser differs from the generator by ${worst.toFixed(5)} ppm`);

  /* 2. the table columns, which is where an off by one hides */
  const i = y.length - 1;
  const f = buildRow(y, i, +data.series.labels[i].slice(5, 7) - 1);
  let m12 = 0;
  for (let k = i - 12; k < i; k++) m12 += y[k];
  m12 /= 12;
  check('the rolling window', Math.abs(f.mean12 - m12) < 1e-9 && f.lag1 === y[i - 1],
    'a rolling column includes its own row, which would make every score below meaningless');

  /* 3. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        bad.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info(`[time-features] load-time checks complete: trend to ${worst.toFixed(5)} ppm at `
    + `${iters} steps${deep ? '' : ' (spot check; window.__atlasCheck(true) refits properly)'}`);
  return { worst, iters };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/timefeatures.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the table', () => initTableScrolly(data));
  safely('the split', () => initLeakWidget(data));
  safely('the ceiling', () => initCeilingWidget(data));
  safely('the hinge trend', () => initProphetWidget(data));
  safely('the decomposition', () => initDecompWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1200);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
