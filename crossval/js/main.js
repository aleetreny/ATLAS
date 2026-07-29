/* Entry point.
 *
 * One guard here is unusual and worth naming: the page claims that the fold
 * spread understates how much the estimate really moves, and that claim is a
 * ratio between two shipped numbers. Recomputing the ratio checks the
 * arithmetic; checking that it is above one checks the claim. If a rerun ever
 * made it drop below one, the paragraph would be wrong and nothing else on the
 * page would notice.
 */
import { initTruthWidget } from './truth-widget.js';
import { initNoiseWidget } from './noise-widget.js';
import { initTimeWidget } from './time-widget.js';
import { initGroupWidget } from './group-widget.js';
import { initEmptyWidget } from './empty-widget.js';
import { initProse, PROSE_IDS } from './prose.js';

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
  if (!ok) console.warn(`[crossval] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  const T = data.truth;

  T.rows.forEach((r) => {
    check(`k = ${r.k}: the bias is the distance from the truth`,
      Math.abs((r.mean - T.expected_risk) - r.bias) < 2e-6,
      `${(r.mean - T.expected_risk).toFixed(6)} against the shipped ${r.bias}`);
    check(`k = ${r.k}: rows per fit`,
      r.n_train === T.n - Math.floor(T.n / r.k),
      `${r.n_train} rows against ${T.n} minus ${T.n} over ${r.k}`);
    check(`k = ${r.k}: the understatement factor`,
      Math.abs(r.sd_true / r.sd_claimed - r.ratio) < 1e-3,
      `${(r.sd_true / r.sd_claimed).toFixed(4)} against the shipped ${r.ratio}`);
    check(`k = ${r.k}: the printed error bar is the smaller one`, r.ratio > 1,
      `the fold spread claims ${r.sd_claimed} and the estimator really moves ${r.sd_true}, `
      + 'so the paragraph about understatement has its direction wrong');
  });

  /* Repeating the split averages away one component and not the other, so the
     sweep must fall and must stop falling. The first version of this guard
     compared the last point against T.true_sd, which is the spread of the TRUE
     RISK and a different quantity entirely: the sweep sits four times above it
     and always will, so the check passed for the wrong reason while the
     paragraph next to it printed true_sd as the floor. */
  const reps = T.repeats;
  const last = reps[reps.length - 1];
  check('repeating the split only helps', reps.every((r, i) => i === 0 || r.sd <= reps[i - 1].sd),
    `the sweep is not monotone: ${reps.map((r) => r.sd.toFixed(5)).join(', ')}`);
  const partVar = (reps[0].sd ** 2 - last.sd ** 2) / (1 - 1 / last.repeats);
  check('the two variance components separate', partVar > 0 && partVar < reps[0].sd ** 2,
    `the part repeating can average away comes out at ${partVar}, which is not between zero `
    + `and the whole ${reps[0].sd ** 2}`);
  check('the partition component matches the direct measurement',
    Math.abs(Math.sqrt(partVar) - T.partition_sd) < 0.6 * T.partition_sd,
    `${Math.sqrt(partVar).toFixed(5)} from the sweep against ${T.partition_sd} measured by `
    + `resplitting one fixed dataset, which are meant to be the same quantity`);

  /* the leakage arm must beat chance and the honest arm must not */
  const N = data.noise;
  check('leakage produces an answer above chance', N.outside > 0.5 + 3 * N.outside_sd / Math.sqrt(N.seeds),
    `${N.outside} is not above the ${N.chance} it should be`);
  check('doing it properly lands on chance',
    Math.abs(N.inside - 0.5) < 4 * N.inside_sd / Math.sqrt(N.seeds),
    `${N.inside} is not within sampling error of ${N.chance}`);

  /* the shuffled control has to undo the effect, or the effect was not order */
  const TM = data.time;
  const rand = TM.rows[0];
  const shuf = TM.rows[1];
  const fwd = TM.rows[2];
  check('the shuffle control lands back on the optimistic figure',
    Math.abs(shuf.mae - rand.mae) < Math.abs(fwd.mae - rand.mae) / 2,
    `shuffling gives ${shuf.mae}, random gives ${rand.mae} and forward chaining gives `
    + `${fwd.mae}: the control did not isolate the order`);

  /* the digest the generator asserted against and the one it actually computed
     have to be the same string, or "the same series as the forecasting article"
     is a claim resting on a constant nobody rechecked */
  check('the series is the one the constant names',
    data.time.digest === data.meta.co2_digest,
    `${data.time.digest} measured against ${data.meta.co2_digest} asserted`);

  /* the two routes to the same probability. The tolerance is not a round number
     chosen to pass: it is four standard errors of the simulation itself, so the
     guard tightens as the trial count rises and a real disagreement cannot hide
     behind a generous constant */
  data.empty.forEach((r) => {
    const se = Math.sqrt(Math.max(r.measured * (1 - r.measured), 1e-6) / r.trials);
    check(`empty folds at ${r.positives} positives`,
      Math.abs(r.exact - r.measured) < 4 * se,
      `the formula says ${r.exact} and ${r.trials} simulated splits say ${r.measured}, `
      + `apart by ${(Math.abs(r.exact - r.measured) / se).toFixed(1)} standard errors`);
  });

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[crossval] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/crossval.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('against a computed risk', () => initTruthWidget(data));
  safely('accuracy on noise', () => initNoiseWidget(data));
  safely('the order', () => initTimeWidget(data));
  safely('the groups', () => initGroupWidget(data));
  safely('folds without positives', () => initEmptyWidget(data));

  renderMath();

  window.__atlasProseIds = PROSE_IDS;
  window.__atlasCheck = () => safely('guards', () => runGuards(data));
  const guards = () => window.__atlasCheck();
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
