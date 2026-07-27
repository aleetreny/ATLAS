/* Every numbered sentence on the page, written from the measurement.
 *
 * Same discipline as the depth article: nothing with a figure in it is typed
 * by hand, and every comparison is written as a comparison so that a rerun
 * which moves a result moves the sentence with it instead of leaving it
 * quietly wrong.
 */
const pct = (v) => `${(v * 100).toFixed(2)}%`;
const pts = (v) => `${(v * 100).toFixed(2)} points`;

function set(id, html) {
  const n = document.querySelector(`#${id}`);
  if (n) n.innerHTML = html;
}

export function initProse(data) {
  const rungs = data.ladder.rungs;
  const rung0 = rungs[0];
  const top = rungs[rungs.length - 1];
  const e2e = data.ladder.end_to_end;
  const gains = rungs.filter((r) => r.against_previous?.verdict === 'real gain');
  const losses = rungs.filter((r) => r.against_previous?.verdict === 'real loss');
  const noise = rungs.length - 1 - gains.length - losses.length;

  /* the ladder, in one honest sentence */
  set('ladder-note',
    `Eleven rungs, each changing one thing and keeping the rest fixed, from a plain residual `
    + `block to a block with every ConvNeXt modernisation on it. End to end it is `
    + `${pts(e2e.delta)}, <span class="bold">${e2e.verdict}</span>. But the interesting number is `
    + `not the end: of the ${rungs.length - 1} steps, ${gains.length} cleared their own threshold, `
    + `${losses.length} lost ground, and <span class="bold">${noise} could not be distinguished `
    + `from re-rolling the random seed</span>. An ablation table with no noise floor on it would `
    + `have reported all eleven as findings.`);

  /* the matched comparison, which is the one the article stands on */
  const fm = data.ladder.flop_matched;
  const pm = data.ladder.param_matched;
  let matched = '';
  if (fm && fm.contrast) {
    matched += `At equal multiply-accumulates the modernised block is ${pts(fm.contrast.delta)} `
      + `away from the plain one, ${fm.contrast.verdict}.`;
  }
  if (pm && pm.contrast) {
    matched += ` At equal parameter count it is ${pts(pm.contrast.delta)}, ${pm.contrast.verdict}.`;
  }
  set('matched-note', matched
    ? `${matched} Those two are this article's headline comparison rather than the raw ladder `
      + `endpoint, because every rung in between moves cost and design at the same time, and a `
      + `network that is better because it is bigger has not told you anything about its design.`
    : `The matched comparisons were cut from this run's schedule; the ladder endpoint is what `
      + `survives, and it carries the cost change with it.`);

  /* the isolated ablations against the cumulative ladder */
  const iso = data.ladder.isolated;
  if (iso) {
    set('isolated-note',
      `Applied one at a time to the same starting block, the changes add up to `
      + `${pts(iso.sum_isolated_deltas)}. Applied cumulatively, the ladder moved `
      + `${pts(iso.cumulative_total_delta)}. The gap of ${pts(iso.disagreement)} is the `
      + `interaction that an ablation table cannot show you, and it is `
      + `<span class="bold">${iso.disagreement_exceeds_noise ? 'larger than the noise floor' : 'inside the noise floor'}</span>. `
      + `Modernisations are not independent, and adding up the rows of a paper's ablation table `
      + `is not a prediction of what you would get by applying them all.`);
  }

  /* compound scaling */
  const arms = data.scaling.arms;
  if (arms && arms.length) {
    const best = arms.reduce((a, b) => (b.test_acc_mean > a.test_acc_mean ? b : a));
    const worst = arms.reduce((a, b) => (b.test_acc_mean < a.test_acc_mean ? b : a));
    const eff = arms.find((a) => a.axis === 'efficientnet');
    const base = data.runs[data.scaling.base];
    set('scaling-note',
      `The same budget, ${data.scaling.budget_target.toFixed(1)} times the base network, spent `
      + `${arms.length} different ways. The best is <span class="bold">${best.axis}</span> at `
      + `${pct(best.test_acc_mean)} and the worst is <span class="bold">${worst.axis}</span> at `
      + `${pct(worst.test_acc_mean)}: a spread of ${pts(best.test_acc_mean - worst.test_acc_mean)} `
      + `between two networks that cost the same to run.`
      + (eff
        ? ` EfficientNet's own exponents land at ${pct(eff.test_acc_mean)}, `
          + `${pts(eff.test_acc_mean - base.test_acc_mean)} from the base network they were scaling.`
        : '')
      + ` And a caution that belongs next to the result rather than in a footnote: the achieved `
      + `budgets span a factor of ${data.scaling.budget_spread.toFixed(2)}, because block counts and `
      + `channel counts are integers, and that spread is larger than several of the gaps above.`);
  }

  /* what the compound rule actually multiplies by */
  const ef = data.scaling.efficientnet;
  if (ef) {
    set('compound-note',
      `One more piece of arithmetic worth doing yourself. The compound rule scales depth by `
      + `\\(\\alpha^\\phi\\), width by \\(\\beta^\\phi\\) and resolution by \\(\\gamma^\\phi\\), with the `
      + `constraint that \\(\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2\\) so that a unit step of `
      + `\\(\\phi\\) doubles the cost. With the paper's own published values `
      + `(${ef.alpha}, ${ef.beta}, ${ef.gamma}) that product is `
      + `<span class="bold">${ef.compound_per_phi.toFixed(3)}</span>, not 2. The familiar reading of `
      + `the rule as doubling, quadrupling, octupling is off by ${((2 / ef.compound_per_phi - 1) * 100).toFixed(0)}% `
      + `per step, which compounds. Here the exponents were used to generate the allocation and the `
      + `budget was allowed to land where it landed, measured rather than assumed.`);
  }

  /* the honest limitations, straight from the generator */
  if (Array.isArray(data.limitations) && data.limitations.length) {
    set('limits-note', data.limitations.map((l) => (typeof l === 'string' ? l : l.note || '')).filter(Boolean).join(' '));
  }

  /* The closing verdict is the one paragraph most likely to be written before
     the numbers arrive and then never checked against them, so it is composed
     from the counts and it says whichever of the three things is true. */
  const worst = rungs.slice(1).reduce((a, b) =>
    ((b.against_previous?.delta ?? 0) < (a.against_previous?.delta ?? 0) ? b : a));
  const bestStep = rungs.slice(1).reduce((a, b) =>
    ((b.against_previous?.delta ?? 0) > (a.against_previous?.delta ?? 0) ? b : a));

  let headline;
  if (e2e.verdict === 'real loss') {
    headline = `At this scale the finished article is <span class="bold">worse</span> than what it `
      + `started from: ${pts(e2e.delta)} against the plain residual block, ${e2e.verdict}. `
      + `The single most expensive change was "${worst.change}" at ${pts(worst.against_previous.delta)}, `
      + `and the only clear winner was "${bestStep.change}" at ${pts(bestStep.against_previous.delta)}.`;
  } else if (e2e.verdict === 'real gain') {
    headline = `End to end the modernised block does win, by ${pts(e2e.delta)}, driven mostly by `
      + `"${bestStep.change}" at ${pts(bestStep.against_previous.delta)} and held back by `
      + `"${worst.change}" at ${pts(worst.against_previous.delta)}.`;
  } else {
    headline = `End to end the two blocks are indistinguishable here: ${pts(e2e.delta)}, inside the `
      + `noise floor. Eleven changes, and the network they produce is the same network as far as `
      + `this experiment can tell.`;
  }

  set('closing-note',
    `${headline} That is not a claim that the original papers are wrong, and it would be a `
    + `misreading of this page to take it as one. It is a claim about what an ablation table is `
    + `evidence <i>for</i>. At three hundred epochs on a million images, a tenth of a point may be `
    + `real and worth having, and the design that produces it may need every one of its parts to be `
    + `there. At ${data.meta.epochs} epochs on ${data.meta.n_train.toLocaleString('en-US')}, most of `
    + `these changes are asking for a budget they cannot yet repay, and ${noise} of them are asking `
    + `for it with evidence that a different random seed would have produced anyway. The number that `
    + `tells you which situation you are in is the seed spread, and it is the number almost nobody `
    + `publishes.`);
}
