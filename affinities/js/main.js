/* Entry point: load the data, boot every widget in isolation, and then check
 * this page's two algorithms against the references the generator produced. A
 * page that claims to run affinity propagation and runs something else is
 * worse than a page with a picture of it, so the checks are loud.
 */
import { initProse } from './prose.js';
import { initScrollyMessages } from './scrolly-messages.js';
import { initPrefWidget } from './pref-widget.js';
import { initSpectralWidget } from './spectral-widget.js';
import { initCutWidget } from './cut-widget.js';
import {
  similarity, affinityPropagation, knnGraph, rbfGraph, spectral, components,
  jacobiEigen, ncut, ari,
} from './graphkit.js';

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

/* `deep` runs every recorded check, which takes a few seconds and is what a
 * verification pass wants. The automatic run at load time checks one of each
 * kind, which is enough to catch a page whose code and whose references have
 * drifted apart and cheap enough not to be felt. */
function check(data, deep = false) {
  const problems = [];
  const some = (arr, n) => (deep ? arr : arr.slice(0, n));

  /* affinity propagation against scikit-learn, exemplar for exemplar */
  const sims = {};
  some(data.ap.checks, 2).forEach((c) => {
    const pts = data.datasets[c.dataset].points;
    const n = pts.length;
    if (!sims[c.dataset]) sims[c.dataset] = similarity(pts);
    const r = affinityPropagation(sims[c.dataset], n, {
      preference: c.preference, damping: c.damping,
    });
    const same = r.exemplars.length === c.exemplars.length
      && r.exemplars.every((v, i) => v === c.exemplars[i]);
    if (!same) {
      problems.push(`affinity propagation on the ${c.dataset} at preference ${c.preference}: `
        + `${r.exemplars.length} exemplars here against ${c.exemplars.length} offline`
        + `${r.exemplars.length === c.exemplars.length ? ', and not the same points' : ''}`);
    }
    if (r.iters !== c.iters) {
      problems.push(`affinity propagation on the ${c.dataset} at preference ${c.preference}: `
        + `${r.iters} iterations here against ${c.iters} offline`);
    }
    const a = ari(data.datasets[c.dataset].true_labels, r.labels);
    if (Math.abs(a - c.ari) > 0.002) {
      problems.push(`affinity propagation on the ${c.dataset}: ARI ${a.toFixed(4)} here against `
        + `${c.ari} offline`);
    }
  });

  /* the eigenvalues against scipy's dense solver. The Lanczos iteration here
     agrees to machine precision where the eigenvalues are separated and to
     about 1e-6 where several of them crowd together, which is what the
     tolerance is set from rather than from hope. */
  some(data.spectral.checks, 2).forEach((c) => {
    const pts = data.datasets[c.dataset].points;
    const n = pts.length;
    const W = c.mode === 'knn' ? knnGraph(pts, c.param) : rbfGraph(pts, c.param);
    const sp = spectral(W, n, data.datasets[c.dataset].k);
    let worst = 0;
    sp.eigenvalues.forEach((v, i) => {
      if (i < c.eigenvalues.length) worst = Math.max(worst, Math.abs(v - c.eigenvalues[i]));
    });
    if (worst > 1e-5) {
      problems.push(`spectral eigenvalues on the ${c.dataset} (${c.mode} ${c.param}): differ by `
        + `${worst.toExponential(2)}`);
    }
    const a = ari(data.datasets[c.dataset].true_labels, sp.labels);
    if (Math.abs(a - c.ari) > 0.002) {
      problems.push(`spectral on the ${c.dataset} (${c.mode} ${c.param}): ARI ${a.toFixed(4)} here `
        + `against ${c.ari} offline`);
    }
  });

  /* the number of zero eigenvalues IS the number of connected pieces, which is
     a claim the article makes in bold and can therefore be tested */
  Object.entries(data.spectral.sets).forEach(([name, ref]) => {
    const pts = data.datasets[name].points;
    const W = knnGraph(pts, data.spectral.knn);
    const comp = components(W, pts.length).count;
    const zeros = ref.eigenvalues.filter((v) => v < 1e-8).length;
    if (comp !== zeros) {
      problems.push(`on the ${name}: ${comp} connected pieces but ${zeros} zero eigenvalues`);
    }
  });

  /* the exhaustive cut: the browser enumerates all of them, so the only thing
     to check is that its optimum and the generator's agree */
  const C = data.cut;
  const n = C.points.length;
  const W = rbfGraph(C.points, C.gamma);
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += W[i * n + j];
    deg[i] = s;
  }
  const M = Array.from({ length: n }, (_, i) => Float64Array.from(
    { length: n }, (_, j) => W[i * n + j] / Math.sqrt(deg[i] * deg[j])
  ));
  const lam2 = 1 - jacobiEigen(M).values[1];
  if (Math.abs(lam2 - C.lambda2) > 1e-6) {
    problems.push(`the relaxed value: ${lam2.toFixed(6)} here against ${C.lambda2} offline`);
  }
  /* the eigenvector itself, entry by entry, up to the sign that is arbitrary */
  const { vectors } = jacobiEigen(M);
  let f = Array.from(vectors[1], (v, i) => v / Math.sqrt(deg[i]));
  if (f.reduce((s, v) => s + v, 0) < 0) f = f.map((v) => -v);
  const ref = C.fiedler.reduce((s, v) => s + v, 0) < 0 ? C.fiedler.map((v) => -v) : C.fiedler;
  const worstF = Math.max(...f.map((v, i) => Math.abs(v - ref[i])));
  if (worstF > 1e-5) {
    problems.push(`the second eigenvector: entries differ by up to ${worstF.toExponential(2)}`);
  }

  let best = Infinity;
  let bestMask = null;
  for (let bits = 0; bits < 2 ** (n - 1); bits++) {
    const mask = new Array(n).fill(false);
    mask[0] = true;
    for (let i = 1; i < n; i++) if (bits & (1 << (i - 1))) mask[i] = true;
    if (mask.every(Boolean)) continue;
    const v = ncut(W, n, mask);
    if (v < best) {
      best = v;
      bestMask = mask;
    }
  }
  if (Math.abs(best - C.optimum) > 1e-6) {
    problems.push(`the exhaustive best cut: ${best.toFixed(6)} here against ${C.optimum} offline`);
  }
  /* and that it is the same split, not merely the same number: a partition and
     its complement are one split, so either match is a match */
  const same = C.optimum_mask.every((v, i) => !!v === bestMask[i])
    || C.optimum_mask.every((v, i) => !!v !== bestMask[i]);
  if (!same) {
    problems.push('the exhaustive best cut has the same value here but a different split');
  }
  /* the threshold that reaches it: the generator counts points from the low end
     of the eigenvector, so its t is directly comparable */
  const order = f.map((v, i) => i).sort((a, b) => f[a] - f[b]);
  let bestT = 0;
  let bestTv = Infinity;
  for (let t = 1; t < n; t++) {
    const mask = new Array(n).fill(false);
    for (let i = 0; i < t; i++) mask[order[i]] = true;
    const v = ncut(W, n, mask);
    if (v < bestTv) {
      bestTv = v;
      bestT = t;
    }
  }
  if (bestT !== C.sweep_t) {
    problems.push(`the eigenvector's best threshold: ${bestT} points here against ${C.sweep_t} offline`);
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every algorithm on this page matches its reference implementation'
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/affinity.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('the message passing scrolly', () => initScrollyMessages(data));
  safely('the two dials', () => initPrefWidget(data));
  safely('the graph', () => initSpectralWidget(data));
  safely('every cut', () => initCutWidget(data));
  renderMath();
  /* after the page is interactive: the checks take a second and nothing on
     screen depends on them */
  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, deep);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
