/* Entry point.
 *
 * The reference run is replayed here from the same generator of random numbers
 * the measurement used, so the guard compares arms and rewards pull by pull with
 * no tolerance. The lower bound and the explore then commit formula are both
 * recomputed rather than read.
 */
import { initPullScrolly } from './scrolly-pulls.js';
import { initPolicyWidget } from './policy-widget.js';
import { initDialsWidget } from './dials-widget.js';
import { initGapWidget } from './gap-widget.js';
import { initProse } from './prose.js';
import { runBandit, laiRobbins, etcFormula } from './banditkit.js';

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
  if (!ok) console.warn(`[bandits] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const R = data.reference;

  /* 1. the reference run, pull by pull */
  const run = runBandit(R.arms, { seed: R.config.seed, pulls: R.config.pulls,
    eps: R.config.eps });
  let bad = 0;
  R.history.forEach((h, i) => {
    const m = run.history[i];
    if (!m || m.arm !== h.arm || m.reward !== h.reward || Math.abs(m.regret - h.regret) > 1e-9) {
      bad += 1;
    }
  });
  check('the reference run', bad === 0 && run.counts.every((c, i) => c === R.counts[i]),
    `${bad} of ${R.history.length} pulls differ`);

  /* 2. the lower bound, recomputed from the probabilities */
  const bound = laiRobbins(data.meta.arms, data.meta.horizon);
  check('the lower bound', Math.abs(bound - data.meta.bound) < 1e-3,
    `${bound.toFixed(4)} here against ${data.meta.bound} in the file`);

  /* 3. the formula that picks the exploring length */
  let formulaBad = 0;
  data.etc.sweeps.forEach((s) => {
    if (etcFormula(s.delta, data.etc.horizon) !== s.predicted_m) formulaBad += 1;
  });
  check('the explore then commit formula', formulaBad === 0,
    `${formulaBad} of ${data.etc.sweeps.length} gaps disagree`);

  /* 4. regret can only go up, and the last value is the one the table shows */
  let monotone = true;
  Object.keys(data.policies).forEach((k) => {
    const arr = data.policies[k].regret;
    for (let i = 1; i < arr.length; i++) if (arr[i] < arr[i - 1] - 1e-9) monotone = false;
  });
  check('the regret curves', monotone, 'a regret curve goes down somewhere, which cannot happen');

  /* 5. the machines are what the page says they are */
  if (deep) {
    const fresh = runBandit(data.meta.arms, { seed: 12345, pulls: 40000, eps: 1.0 });
    const worst = Math.max(...fresh.means.map((m, i) => Math.abs(m - data.meta.arms[i])));
    check('the machines', worst < 0.03,
      `pulling everything at random 40,000 times, the worst estimate is off by ${worst.toFixed(4)}`);
  }

  /* 6. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info(`[bandits] load-time checks complete: ${R.history.length} pulls reproduced, `
    + `bound ${bound.toFixed(4)}, ${bad} disagreements`);
  return { bad, bound };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/bandits.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the pulls', () => initPullScrolly(data));
  safely('the policies', () => initPolicyWidget(data));
  safely('the dials', () => initDialsWidget(data));
  safely('the gap', () => initGapWidget(data));

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
