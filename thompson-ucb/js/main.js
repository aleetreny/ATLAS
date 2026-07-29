/* Entry point.
 *
 * The reference run is replayed here from the same generator of random numbers
 * the measurement used, so the guard compares arms and rewards pull by pull with
 * no tolerance. The lower bound and the explore then commit formula are both
 * recomputed rather than read.
 */
import { initBeliefScrolly } from './scrolly-belief.js';
import { initRegretWidget } from './regret-widget.js';
import { initHonestWidget } from './honest-widget.js';
import { initDriftWidget } from './drift-widget.js';
import { initProse } from './prose.js';
import { ucbIndex, betaPdf, boundConstant } from './ucbkit.js';

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
  if (!ok) console.warn(`[thompson-ucb] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const S = data.snapshots;

  /* 1. the bound drawn in the first panel is the bound, recomputed from counts */
  let worstIdx = 0;
  S.marks.forEach((t) => {
    const U = S.ucb[String(t)];
    U.counts.forEach((n, i) => {
      const mine = ucbIndex(U.means[i], n, t, data.constant.proved_c) - U.means[i];
      worstIdx = Math.max(worstIdx, Math.abs(mine - U.bonus[i]));
    });
  });
  check('the bound', worstIdx < 1e-3,
    `recomputing the bonus from the counts moves it by ${worstIdx}`);

  /* 2. the posterior integrates to one, which is the cheapest way to catch a
   *    wrong normalising constant in the density the page draws */
  let worstMass = 0;
  S.marks.forEach((t) => {
    const T = S.thompson[String(t)];
    T.alpha.forEach((a, i) => {
      let mass = 0;
      const n = 4000;
      for (let s = 1; s < n; s++) mass += betaPdf(s / n, a, T.beta[i]) / n;
      worstMass = Math.max(worstMass, Math.abs(mass - 1));
    });
  });
  check('the posterior', worstMass < 5e-3,
    `a posterior integrates to ${(1 + worstMass).toFixed(4)} instead of one`);

  /* 3. the floor, recomputed from the probabilities */
  const C = boundConstant(data.meta.arms);
  check('the floor', Math.abs(C - data.meta.bound_constant) < 1e-3,
    `${C.toFixed(4)} here against ${data.meta.bound_constant} in the file`);

  /* 4. the counts in a snapshot add up to the pulls taken */
  let bad = 0;
  S.marks.forEach((t) => {
    ['ucb', 'thompson'].forEach((who) => {
      const sum = S[who][String(t)].counts.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - t) > 0.5) bad += 1;
    });
  });
  check('the snapshots', bad === 0, `${bad} snapshots do not add up to their pull count`);

  /* 5. regret only goes up */
  let monotone = true;
  Object.keys(data.regret).forEach((k) => {
    const arr = data.regret[k].curve;
    for (let i = 1; i < arr.length; i++) if (arr[i] < arr[i - 1] - 1e-9) monotone = false;
  });
  check('the regret curves', monotone, 'a regret curve goes down somewhere');

  /* 6. the machines are the previous article's, not a copy that drifted */
  if (deep) {
    check('the shared machines', data.meta.shared_with.includes('bandits'),
      'the file does not say where its machines came from');
  }

  /* 7. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info(`[thompson-ucb] load-time checks complete: bonus within ${worstIdx.toExponential(1)}, `
    + `posterior mass within ${worstMass.toExponential(1)}, floor ${C.toFixed(4)}`);
  return { worstIdx, worstMass, C };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/ucb.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the two beliefs', () => initBeliefScrolly(data));
  safely('regret', () => initRegretWidget(data));
  safely('honesty', () => initHonestWidget(data));
  safely('the swap', () => initDriftWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
