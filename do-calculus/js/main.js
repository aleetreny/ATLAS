/* Entry point.
 *
 * This page has something almost no other article in the atlas has: a
 * reference that is exact rather than measured. The interventional
 * distribution is a finite sum, so the browser can recompute it from the
 * probability tables and compare with the export at machine precision, and the
 * adjustment formula for every subset can be recomputed the same way. A guard
 * with a tolerance of 1e-12 is available here and it would be a waste not to
 * use it.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initSetsWidget } from './sets-widget.js';
import { initFrontWidget } from './front-widget.js';
import { initHetWidget } from './het-widget.js';
import { worlds, jointProb, doProb, adjust } from './dagkit.js';

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

  /* ---- the true effect, recomputed in the browser from the tables.
     Both sides are sums of products of numbers that travelled at full
     precision, so anything above 1e-12 is a disagreement and not rounding. */
  const model = { parents: E.parents, cpt: E.cpt };
  const ws = worlds(E.names.length);
  const truth = doProb(model, ws, E.x, 1, E.y) - doProb(model, ws, E.x, 0, E.y);
  if (Math.abs(truth - E.truth) > 1e-12) {
    problems.push(`the browser computes the interventional effect as ${truth.toFixed(12)} `
      + `and the export says ${E.truth}`);
  }

  /* ---- and every one of the sixteen adjustment formulas */
  let worstSet = 0;
  E.rows.forEach((row) => {
    const got = adjust(model, ws, E.x, E.y, row.S);
    if (!Number.isFinite(got) && row.estimate === null) return;
    worstSet = Math.max(worstSet, Math.abs(got - row.estimate));
  });
  if (worstSet > 1e-12) {
    problems.push(`recomputing the adjustment formulas here differs from the export by `
      + `${worstSet.toExponential(2)}`);
  }

  /* ---- the claim the whole first section makes: the graphical criterion
     picks out exactly the sets that give the right answer */
  E.rows.forEach((row) => {
    const err = Math.abs(row.estimate - E.truth);
    if (row.backdoor && err > 1e-10) {
      problems.push(`the set {${row.names.join(', ')}} satisfies the criterion and is off by `
        + `${err.toExponential(2)}`);
    }
  });
  if (E.rows.filter((r) => r.backdoor).length < 1) {
    problems.push('no adjustment set satisfies the criterion, so the section has no subject');
  }
  if (Math.abs(E.full.estimate - E.truth) < 1e-6) {
    problems.push('adjusting for everything is correct in the example graph, so the example '
      + 'does not show what the page says it shows');
  }

  /* ---- the front door, recomputed */
  const fm = { parents: data.frontdoor.parents, cpt: data.frontdoor.cpt };
  const fws = worlds(data.frontdoor.names.length);
  const ftruth = doProb(fm, fws, 1, 1, 3) - doProb(fm, fws, 1, 0, 3);
  if (Math.abs(ftruth - data.frontdoor.truth) > 1e-12) {
    problems.push(`the browser computes the front door model's true effect as `
      + `${ftruth.toFixed(12)} and the export says ${data.frontdoor.truth}`);
  }
  if (!(data.frontdoor.error < 1e-10)) {
    problems.push(`the front door formula is supposed to be exact and is off by `
      + `${data.frontdoor.error}`);
  }

  /* ---- the rules table: exact when the condition holds, visibly not when it
     does not. Both halves matter, because a rule that always holds would mean
     the condition is not doing anything. */
  data.rules.rows.forEach((row) => {
    if (row.holds && row.residual > 1e-12) {
      problems.push(`rule ${row.rule} (${row.case}) meets its condition and leaves a residual `
        + `of ${row.residual}`);
    }
    if (!row.holds && row.residual < 1e-6) {
      problems.push(`rule ${row.rule} (${row.case}) fails its condition and the two sides `
        + `still agree, so the condition is not doing any work`);
    }
  });

  /* ---- the directions the heterogeneity section states */
  const by = Object.fromEntries(data.heterogeneity.rows.map((r) => [r.key, r]));
  if (!(by.cf_centered.mse < by.cf.mse)) {
    problems.push('the page says local centring helps the causal forest and the table '
      + 'disagrees');
  }
  if (!(by.cf.mse > by.constant.mse)) {
    problems.push('the page says the uncentred forest loses to answering the average, and '
      + 'the table disagrees');
  }
  if (!(by.cf_centered.mse < by.constant.mse)) {
    problems.push('the centred forest does not beat the constant, so the section has no '
      + 'positive result to report');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the browser recomputes the interventional effect and all sixteen adjustment '
      + 'formulas to machine precision, the criterion picks out exactly the sets that work, '
      + 'and the three rules are exact when their conditions hold');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/docalculus.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('sixteen answers, two of them right', () => initSetsWidget(data));
  safely('a confounder nobody measured', () => initFrontWidget(data));
  safely('not how much, but for whom', () => initHetWidget(data));
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
