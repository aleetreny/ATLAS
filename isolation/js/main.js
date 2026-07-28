/* Entry point: load the data, boot every widget in isolation, and then check
 * the four detectors against the references the generator produced.
 *
 * Each tolerance is the rounding the JSON carries for that quantity, so a guard
 * here can only stay quiet if the browser is genuinely reproducing the
 * measurement. Where a quantity cannot be reproduced at all, the guard says so
 * rather than widening until it passes: scikit-learn's forest is a different
 * forest, so what is compared there is the ranking.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initScrollyIsolate } from './scrolly-isolate.js';
import { initForestWidget } from './forest-widget.js';
import { initBiasWidget } from './bias-widget.js';
import { initLocalWidget } from './local-widget.js';
import { initBoundaryWidget } from './boundary-widget.js';
import { initRobustWidget } from './robust-widget.js';
import { buildForest, forestScore, cFactor } from './forest.js';
import {
  lof, distanceTable, knnDistance, svmDecision, cStep, averagePrecision, rocAuc,
} from './detkit.js';

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

function check(data, deep = false) {
  const problems = [];
  const push = (s) => problems.push(s);

  /* ---- the normaliser, which every score divides by */
  let worstC = 0;
  data.forest.c_table.forEach((r) => {
    worstC = Math.max(worstC, Math.abs(cFactor(r.n) - r.c));
  });
  if (worstC > 1e-7) push(`c(n) differs from the generator's by ${worstC.toExponential(2)}`);

  /* ---- the forests the page rebuilds, tree for tree */
  Object.entries(data.forest.page).forEach(([name, spec]) => {
    const X = data.sets[name].points;
    const f = buildForest(X, spec.trees, spec.psi, spec.seed);
    if (f.limit !== spec.limit) {
      push(`the ${name} forest: depth limit ${f.limit} here against ${spec.limit} offline`);
    }
    const s = forestScore(f, spec.probe_index.map((i) => X[i]));
    let worst = 0;
    spec.probe_score.forEach((v, i) => {
      worst = Math.max(worst, Math.abs(v - s[i]));
    });
    if (worst > 1e-7) {
      push(`the ${name} forest: scores differ by up to ${worst.toExponential(2)}, so this page is `
        + 'not growing the same trees');
    }
  });

  /* ---- the score field of the corners, cell by cell */
  if (deep) {
    const G = data.forest.grid;
    const X = data.sets['two clusters and two corners'].points;
    const f = buildForest(X, G.trees, G.psi, G.seed);
    const pts = [];
    const step = (G.hi - G.lo) / (G.n - 1);
    for (let r = 0; r < G.n; r++) for (let c = 0; c < G.n; c++) pts.push([G.lo + c * step, G.lo + r * step]);
    const s = forestScore(f, pts);
    let worst = 0;
    G.scores.forEach((v, i) => {
      worst = Math.max(worst, Math.abs(v - s[i]));
    });
    if (worst > 1e-4) {
      push(`the score field: ${G.scores.length} cells differ by up to ${worst.toExponential(2)}`);
    }
  }

  /* ---- the local outlier factor, against scikit-learn, entry by entry */
  const X2 = data.sets['two densities'].points;
  const y2 = data.sets['two densities'].labels;
  const ks = deep ? data.lof.k_grid : data.lof.k_grid.slice(0, 3);
  let worstLof = 0;
  ks.forEach((k) => {
    const r = lof(X2, k);
    const ref = data.lof.ref[String(k)];
    ref.forEach((v, i) => {
      worstLof = Math.max(worstLof, Math.abs(v - r.lof[i]));
    });
    const row = data.lof.rows.find((q) => q.k === k);
    if (Math.abs(averagePrecision(y2, r.lof) - row.ap_lof) > 6e-4) {
      push(`LOF at k = ${k}: average precision ${averagePrecision(y2, r.lof).toFixed(4)} here `
        + `against ${row.ap_lof} offline`);
    }
    if (Math.abs(rocAuc(y2, r.lof) - row.auc_lof) > 6e-4) {
      push(`LOF at k = ${k}: area under the curve differs from the generator's`);
    }
  });
  if (worstLof > 6e-6) {
    push(`the local outlier factor: differs from scikit-learn's by up to `
      + `${worstLof.toExponential(2)}`);
  }
  const Dt = distanceTable(X2);
  const kk = data.lof.disagree.k;
  const rowK = data.lof.rows.find((q) => q.k === kk);
  if (Math.abs(averagePrecision(y2, knnDistance(Dt, kk)) - rowK.ap_knn) > 6e-4) {
    push(`the kth-neighbour distance at k = ${kk}: average precision does not match`);
  }

  /* ---- the kernel decision function, against libsvm's */
  let worstSvm = 0;
  let worstSvmAp = 0;
  const models = Object.entries(data.svm.models);
  (deep ? models : models.slice(0, 4)).forEach(([, m]) => {
    const model = { sv: m.sv, alpha: m.alpha, rho: m.rho, gamma: m.gamma };
    const f = svmDecision(model, X2, X2.filter((_, i) => i % 11 === 0));
    m.probe.forEach((v, i) => {
      worstSvm = Math.max(worstSvm, Math.abs(v - f[i]));
    });
    /* And the score the widget prints, not just the decision function it comes
       from: a difference of 2e-8 in the second moves average precision by up to
       0.35 on this data, so agreeing on the function is not the same as
       agreeing on the number, and the guard has to check the number. */
    const full = svmDecision(model, X2, X2);
    worstSvmAp = Math.max(worstSvmAp,
      Math.abs(averagePrecision(y2, full.map((v) => -v)) - m.ap));
  });
  if (worstSvm > 6e-6) {
    push(`the one-class SVM's decision function: differs by up to ${worstSvm.toExponential(2)}`);
  }
  if (worstSvmAp > 6e-5) {
    push(`the one-class SVM's average precision: differs by up to `
      + `${worstSvmAp.toExponential(2)} from what the generator recorded`);
  }

  /* ---- the concentration step, from the supports the generator recorded */
  let worstMu = 0;
  let worstDet = 0;
  data.mcd.steps.forEach((st) => {
    const r = cStep(X2, st.support);
    st.mu.forEach((v, i) => {
      worstMu = Math.max(worstMu, Math.abs(v - r.mu[i]));
    });
    worstDet = Math.max(worstDet, Math.abs(r.det - st.det) / Math.max(1e-12, st.det));
    /* the theorem the widget prints: the determinant never goes up */
    if (st.step > 0) {
      const prev = data.mcd.steps[st.step - 1];
      if (st.det > prev.det * (1 + 1e-9)) {
        push(`the determinant went up between steps ${prev.step} and ${st.step}, which the `
          + 'concentration step forbids');
      }
    }
  });
  if (worstMu > 6e-5) push(`the concentration step's centre differs by ${worstMu.toExponential(2)}`);
  if (worstDet > 1e-4) push(`its determinant differs by ${worstDet.toExponential(2)} relatively`);

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every detector on this page matches the reference that owns it'
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/isolation.json').then((r) => r.json());
  const scrolly = safely('the tree', () => initScrollyIsolate(data));
  safely('the numbered sentences', () => initProse(data, scrolly ? scrolly.facts : {
    limit: data.forest.page['two densities'].limit,
    psi: data.forest.page['two densities'].psi,
    cPsi: cFactor(data.forest.page['two densities'].psi),
    anomDepth: 0, normDepth: 0, anomLength: 0, normLength: 0, anomLeaf: 1, normLeaf: 1, cuts: 0,
  }));
  safely('the forest', () => initForestWidget(data));
  safely('the axis bias', () => initBiasWidget(data));
  safely('local against global', () => initLocalWidget(data));
  safely('the boundary', () => initBoundaryWidget(data));
  safely('the concentration step', () => initRobustWidget(data));
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
