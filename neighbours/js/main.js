/* Entry point.
 *
 * The guards here are deliberately not the ones the other articles in this
 * module use, and the reason is the article's own finding: this descent
 * amplifies a difference of one part in ten million in the bandwidths into a
 * visibly different picture. So what gets checked is what is stable and
 * checkable, and the tolerances say which is which:
 *
 *   exactly  - every point's neighbour distribution has the requested
 *              perplexity; that is a solved equation, not a trajectory
 *   tightly  - the bandwidths against the generator's, to 1e-4
 *   loosely  - the outcome of a thousand steps, where the article has measured
 *              how far two runs of the same code can legitimately land apart
 */
import { initScrollyTsne } from './scrolly-tsne.js';
import { initCrowdingWidget } from './crowding-widget.js';
import { initLiveWidget, buildDatasets } from './live-widget.js';
import { initBoardWidget } from './board-widget.js';
import { initProse } from './prose.js';
import { toRows, symEigen, covariance, zeros } from '../../assets/js/embed.js';
import { jointP, conditionalP, sqDists, countBlobs } from './tsnekit.js';

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

function check(label, ok, detail) {
  if (!ok) console.warn(`[neighbours] ${label}: ${detail}`);
  return ok;
}

function pca2(X) {
  const S = covariance(X);
  const { vectors } = symEigen(S);
  const n = X.length;
  const p = X[0].length;
  const m = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) m[j] += X[i][j] / n;
  const Z = zeros(n, 2);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 2; j++) {
      let s = 0;
      for (let t = 0; t < p; t++) s += (X[i][t] - m[t]) * vectors[t][j];
      Z[i][j] = s;
    }
  }
  return Z;
}

function runGuards(data, computed, sets, live) {
  const M = data.meta;
  const shown = String(M.perplexity_show);

  /* 1. the defining equation, which is exact */
  const D2 = computed.D2;
  M.perplexities.forEach((p) => {
    const { P, betas } = conditionalP(D2, p);
    let worst = 0;
    for (let i = 0; i < P.length; i++) {
      let H = 0;
      for (let j = 0; j < P.length; j++) {
        if (j === i || P[i][j] <= 0) continue;
        H -= P[i][j] * Math.log(P[i][j]);
      }
      worst = Math.max(worst, Math.abs(Math.exp(H) - p));
    }
    check(`perplexity ${p}`, worst < 1e-6,
      `a point's neighbour distribution has perplexity ${worst.toExponential(2)} away from ${p}`);
    if (String(p) === shown) {
      const ref = data.digits.perplexity[shown].betas;
      let wb = 0;
      for (let i = 0; i < betas.length; i++) wb = Math.max(wb, Math.abs(betas[i] - ref[i]));
      check('bandwidths', wb < 1e-4,
        `the bandwidths differ from the generator's by up to ${wb.toExponential(2)}`);
    }
  });

  /* 2. the joint distribution is a distribution */
  let total = 0;
  for (let i = 0; i < computed.P.length; i++) {
    for (let j = 0; j < computed.P.length; j++) if (i !== j) total += computed.P[i][j];
  }
  check('joint distribution', Math.abs(total - 1) < 1e-6,
    `the neighbour probabilities sum to ${total.toFixed(9)}`);

  /* 3. the two synthetic datasets, which the browser draws rather than
        downloads: their measured geometry must be the generator's */
  const B = data.blobs;
  const cent = [0, 1, 2].map((c) => {
    const acc = new Float64Array(M.pcs);
    let cnt = 0;
    sets.blobs.X.forEach((row, i) => {
      if (B.labels[i] !== c) return;
      for (let j = 0; j < M.pcs; j++) acc[j] += row[j];
      cnt += 1;
    });
    for (let j = 0; j < M.pcs; j++) acc[j] /= cnt;
    return acc;
  });
  const radius = [0, 1, 2].map((c) => {
    let s = 0;
    let cnt = 0;
    sets.blobs.X.forEach((row, i) => {
      if (B.labels[i] !== c) return;
      let d = 0;
      for (let j = 0; j < M.pcs; j++) d += (row[j] - cent[c][j]) ** 2;
      s += Math.sqrt(d);
      cnt += 1;
    });
    return s / cnt;
  });
  const gotRatio = radius[2] / radius[0];
  check('blob radii', Math.abs(gotRatio - B.true_radius_ratio) < 5e-3,
    `the browser's blobs have a radius ratio of ${gotRatio.toFixed(4)}, `
    + `the generator's ${B.true_radius_ratio}`);

  /* 4. the outcome, at the tolerance the article's own nudge experiment says is
        the most two runs of this code can be held to */
  if (live && live.run && live.run.it >= M.iters) {
    const clumps = countBlobs(live.run.Y, data.noise.counter_multiple).count;
    const ref = data.digits.runs[shown].clusters;
    check('clump count', Math.abs(clumps - ref) <= 2,
      `this run shows ${clumps} clumps, the generator's showed ${ref}`);
  }

  /* 5. nothing on the page may read "undefined" or "NaN" */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .nb-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[neighbours] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/neighbours.json').then((r) => r.json());

  const X = toRows(data.digits.scores);
  const shown = data.meta.perplexity_show;
  const D2 = sqDists(X);
  /* solved once and shared: the conditional distribution, the joint one, and
     the bandwidths all come out of the same search */
  const { P: Pcond, betas } = conditionalP(D2, shown);
  const n = X.length;
  const P = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) P[i][j] = Math.max((Pcond[i][j] + Pcond[j][i]) / (2 * n), 1e-12);
  }

  /* The point the scrolly follows is the one whose bandwidth is closest to the
     median, so its neighbour distribution is typical rather than chosen. */
  const sig = Array.from(betas, (b) => 1 / Math.sqrt(2 * b));
  const med = Array.from(sig).sort((a, b) => a - b)[Math.floor(sig.length / 2)];
  let focus = 0;
  let bestGap = Infinity;
  sig.forEach((v, i) => {
    if (Math.abs(v - med) < bestGap) {
      bestGap = Math.abs(v - med);
      focus = i;
    }
  });

  const pcaLive = pca2(X);
  const computed = {
    P: { cond: Pcond }, joint: P, D2, betas, focus,
    pca: pcaLive, mid: pcaLive, final: pcaLive,
    midIter: data.meta.exagg_iters,
  };

  const sets = buildDatasets(data);
  const cache = { [`digits:${shown}`]: P };

  safely('prose', () => initProse(data));
  const sc = safely('the mechanism', () => initScrollyTsne(data, computed));
  safely('what a plane cannot hold', () => initCrowdingWidget(data));
  const live = safely('the live run', () => initLiveWidget(data, sets, cache, (st) => {
    /* the scrolly's last two panels follow the run that is still going */
    if (st.set !== 'digits' || st.pi !== data.meta.perplexities.indexOf(shown)) return;
    const it = st.run.it;
    if (it < data.meta.exagg_iters && it > 0) return;
    const snap = st.run.Y.map((r) => Float64Array.from(r));
    if (it >= data.meta.exagg_iters && it < data.meta.exagg_iters + 60) computed.mid = snap;
    if (it >= data.meta.iters) computed.final = snap;
    if (sc && sc.refresh) sc.refresh();
  }));
  safely('the scoreboard', () => initBoardWidget(data));

  renderMath();

  const guards = () => safely('guards',
    () => runGuards(data, { ...computed, P }, sets, live));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 12000 });
  else setTimeout(guards, 3000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
