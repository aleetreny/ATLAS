/* Entry point: render maths, load the data and the atom sprite, boot every
 * widget in isolation, then check the page's own arithmetic against the
 * generator's.
 *
 * The guards below are written so they CAN fire. Every one of them compares a
 * number this page computed against a number scikit-learn computed, with a
 * tolerance tight enough to catch a real disagreement and loose enough to
 * survive the rounding that the JSON and the 8-bit sprite deliberately
 * introduce. A guard whose tolerance swallows the error it is looking for is
 * decoration, and this atlas has paid for that one already.
 */
import { initScrollyLine } from './scrolly-line.js';
import { initDialWidget } from './dial-widget.js';
import { initSpectrumWidget, decompose } from './spectrum-widget.js';
import { initIcaWidget, fastICA2 } from './ica-widget.js';
import { initAtomsWidget, loadAtoms, fitOne } from './atoms-widget.js';
import { initProse } from './prose.js';
import { project, unit, centred } from './cloud.js';
import { whiten2, objectiveAt } from './linalg.js';

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
  if (!ok) console.warn(`[directions] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, computed, bank) {
  const C = data.cloud;
  const S = data.spectrum;
  const I = data.ica;
  const A = data.atoms;

  /* 1. the browser's own eigendecomposition against scikit-learn's */
  for (const key of ['raw', 'std']) {
    let worst = 0;
    computed[key].evr.forEach((v, i) => {
      worst = Math.max(worst, Math.abs(v - S[key].evr[i]));
    });
    check(`spectrum ${key}`, worst < 1e-5,
      `explained variance differs from scikit-learn by up to ${worst.toExponential(2)}`);
    /* the loadings too, up to the sign convention both ends apply */
    let wl = 0;
    computed[key].vectors[0].forEach((v, i) => {
      wl = Math.max(wl, Math.abs(Math.abs(v) - Math.abs(S[key].loadings1[i])));
    });
    check(`spectrum ${key} loadings`, wl < 1e-4,
      `first component's weights differ by up to ${wl.toExponential(2)}`);
  }

  /* 2. the dial's own sweep, and the identity the section is built on */
  for (const key of ['raw', 'std']) {
    const cur = C[key];
    /* The identity is algebra, so it is checked against a total this page
       computes from the same points, at 1e-12. Checking it against the total
       in the file instead only measures how many decimals that total was
       written with, which is a different and much less interesting question:
       it is asked separately, one line down. */
    const pts = centred(cur.points);
    const own = pts.reduce((s, [a, b]) => s + a * a + b * b, 0) / pts.length;
    let best = -Infinity;
    let bestDeg = 0;
    let gap = 0;
    for (let d = 0; d < 180; d += C.dial_step) {
      const pr = project(pts, d);
      gap = Math.max(gap, Math.abs(pr.captured + pr.lost - own));
      if (pr.captured > best) {
        best = pr.captured;
        bestDeg = d;
      }
    }
    check(`dial ${key} identity`, gap < 1e-12,
      `captured + thrown away misses the total by ${gap.toExponential(2)}`);
    check(`dial ${key} total`, Math.abs(own - cur.total) < 1e-5,
      `this page makes the total ${own.toFixed(8)}, the file says ${cur.total}`);
    check(`dial ${key}`, Math.abs(bestDeg - cur.dial_best_deg) < 1e-9,
      `best angle is ${bestDeg}, the file says ${cur.dial_best_deg}`);
    check(`dial ${key} share`, Math.abs(best / own - cur.dial_best_share) < 1e-4,
      `best share is ${(best / own).toFixed(6)}, the file says ${cur.dial_best_share}`);
    /* the exact eigenvector, from the other side: the second component must be
       perpendicular to the first, which is the claim the scrolly draws */
    const [ux, uy] = unit(cur.exact_deg);
    const perp = project(pts, cur.exact_deg + 90);
    check(`dial ${key} perpendicular`, Math.abs(perp.captured / own - (1 - cur.exact_share)) < 1e-4,
      `the perpendicular direction keeps ${(perp.captured / cur.total).toFixed(6)}, `
      + `expected ${(1 - cur.exact_share).toFixed(6)} (unit ${ux.toFixed(3)},${uy.toFixed(3)})`);
  }

  /* 3. the whitening and the closed-form rotation sweep */
  const Z = whiten2(I.mixed);
  const m = [0, 0, 0, 0, 0];
  for (const [z1, z2] of Z) {
    m[0] += z1 ** 4;
    m[1] += z1 ** 3 * z2;
    m[2] += z1 ** 2 * z2 ** 2;
    m[3] += z1 * z2 ** 3;
    m[4] += z2 ** 4;
  }
  for (let j = 0; j < 5; j++) m[j] /= Z.length;
  const worstM = Math.max(...m.map((v, i) => Math.abs(v - I.moments[i])));
  check('ica moments', worstM < 1e-5,
    `the browser's fourth moments differ from the generator's by up to ${worstM.toExponential(2)}`);

  let peak = { deg: 0, v: -Infinity };
  for (let d = 0; d < 180; d += I.dial_step) {
    const v = objectiveAt(m, d).value;
    if (v > peak.v) peak = { deg: d, v };
  }
  check('ica peak', Math.abs(peak.deg - I.peak_deg) < 1e-9,
    `the browser's peak is at ${peak.deg}, the file says ${I.peak_deg}`);

  /* the fixed point and the brute-force sweep must agree, up to the swap of
     the two recovered signals that a quarter turn performs */
  const f = fastICA2(Z);
  const deg = ((Math.atan2(f.w[1], f.w[0]) * 180) / Math.PI + 180) % 180;
  const off = Math.abs(deg - peak.deg) % 90;
  check('fastica', f.converged && Math.min(off, 90 - off) < 1,
    `the fixed point landed at ${deg.toFixed(2)}, the sweep at ${peak.deg} `
    + `(converged: ${f.converged})`);

  /* 4. the reconstructions, per digit and per rank, against the generator's */
  let worstFit = 0;
  let where = '';
  A.per_image.forEach((row) => {
    ['pca', 'ica', 'nmf'].forEach((method) => {
      row[method].forEach((ref, i) => {
        const got = fitOne(method, row.rank, bank.show[i], bank, A.nmf_fit_iters).error;
        const rel = Math.abs(got - ref) / Math.max(ref, 1e-12);
        if (rel > worstFit) {
          worstFit = rel;
          where = `${method} rank ${row.rank} digit ${A.showcase[i].label}: `
            + `${got.toFixed(8)} against ${ref}`;
        }
      });
    });
  });
  check('atoms', worstFit < 1e-4,
    `a reconstruction differs from the generator's by ${(worstFit * 100).toFixed(4)}% (${where})`);

  /* And the claim the section rests on: PCA and ICA reach the same error.
     Exactly the same, before the atoms are rounded to 8 bits. After rounding,
     two differently perturbed sets of vectors span very slightly different
     subspaces, so the bound here is the generator's own measured worst case
     over all the digits rather than a number chosen to pass. */
  A.ranks.forEach((r) => {
    const x = bank.show[0];
    const row = A.errors.find((e) => e.rank === r);
    const p = fitOne('pca', r, x, bank, A.nmf_fit_iters);
    const ic = fitOne('ica', r, x, bank, A.nmf_fit_iters);
    check(`subspace rank ${r}`, Math.abs(p.error - ic.error) <= row.ica_gap_worst_image * 1.5,
      `PCA and ICA should span the same subspace to within `
      + `${row.ica_gap_worst_image.toExponential(2)} but reach ${p.error.toFixed(8)} `
      + `and ${ic.error.toFixed(8)}`);
    /* NMF cannot produce a negative pixel. Not "usually": ever. */
    const n = fitOne('nmf', r, x, bank, A.nmf_fit_iters);
    check(`nmf non-negative rank ${r}`, Math.min(...n.recon) >= 0,
      `an NMF reconstruction went to ${Math.min(...n.recon).toExponential(2)}`);
  });

  /* 5. nothing on the page may read "undefined" or "NaN" */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .dir-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[directions] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/directions.json').then((r) => r.json());
  const bank = await loadAtoms(data);

  safely('prose', () => initProse(data));
  const computed = safely('spectrum maths', () => decompose(data));
  safely('the best line', () => initScrollyLine(data));
  safely('the dial', () => initDialWidget(data));
  if (computed) safely('the spectrum', () => initSpectrumWidget(data, computed));
  safely('the rotation', () => initIcaWidget(data));
  safely('the atoms', () => initAtomsWidget(data, bank));

  /* KaTeX again: prose.js writes into paragraphs after the first pass. */
  renderMath();

  const guards = () => safely('guards', () => runGuards(data, computed, bank));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 4000 });
  else setTimeout(guards, 800);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
