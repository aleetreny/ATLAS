/* Every sentence with a measured number, composed from the measurement.
 *
 * Three of the paragraphs here are comparisons that could go either way on a
 * rerun (which approximation is closest, whether an ensemble of eight spreads
 * at all, which prior scale the evidence picks), so all three are written as
 * branches on the table rather than as a verdict that was true once.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-why', 'intro-model',
  'exact-intro', 'exact-gap',
  'chains-intro', 'chains-note',
  'approx-intro', 'approx-ensemble',
  'prior-intro', 'prior-note',
  'v-map-cost', 'v-map-good', 'v-map-bad',
  'v-lap-cost', 'v-lap-good', 'v-lap-bad',
  'v-ens-cost', 'v-ens-good', 'v-ens-bad',
  'v-vi-cost', 'v-vi-good', 'v-vi-bad',
  'v-mc-cost', 'v-mc-good', 'v-mc-bad',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

function rows(sel, data, cells) {
  const body = d3.select(sel).select('tbody');
  if (body.empty()) return;
  body.html('');
  data.forEach((d) => {
    const tr = body.append('tr');
    cells(d).forEach((c) => {
      const td = tr.append('td');
      if (c && c.html !== undefined) td.html(c.html);
      else td.text(c);
    });
  });
}

export function initProse(data) {
  const M = data.meta;
  const E = data.exact;
  const C = data.chain_worst;
  const A = data.approx.rows;
  const P = data.prior_sweep;
  const by = Object.fromEntries(A.map((r) => [r.key, r]));

  const cube = M.grid ** 3;
  const ordered = [...A].sort((a, b) => a.tv_all - b.tv_all);
  const best = ordered[0];
  const worstRow = ordered[ordered.length - 1];
  /* La comparación que decide el párrafo del conjunto: si los ocho miembros
   * caen en la misma función, sus cifras son las del ajuste único. */
  const flat = Math.abs(by.ensemble.tv_all - by.map.tv_all) < 1e-6
    && by.ensemble.sd_ratio_in < 0.01;
  const bestPrior = P.reduce((p, c) => (c.log_evidence > p.log_evidence ? c : p), P[0]);
  const lowPrior = P[0];
  const highPrior = P[P.length - 1];

  set('intro-why',
    `A Bayesian neural network is a simple idea with an impossible integral in the middle of `
    + `it. Instead of one setting of the weights, keep every setting, weighted by how well it `
    + `explains the data, and average the predictions. The averaging is the integral, it runs `
    + `over as many dimensions as the network has weights, and for anything you would actually `
    + `deploy it cannot be done. So the field is a catalogue of approximations, and every paper `
    + `that proposes one compares it against the others, because there is nothing else to `
    + `compare it against.`);

  set('intro-model',
    `So here is a network small enough that the integral is a sum. One input, one hidden unit, `
    + `one output, no biases except the hidden one: \\(f(x) = v\\tanh(wx + b)\\), which is `
    + `${word(3)} weights. Put a Gaussian prior of width ${M.prior} on each, take ${M.n} `
    + `observations with noise of known size ${M.sigma}, and lay a grid of ${M.grid} points per `
    + `axis over the cube \\([-${M.lim}, ${M.lim}]^3\\). That is `
    + `<span class="bold">${n0(cube)}</span> settings of the weights, every one of them `
    + `evaluated. The posterior is not approximated here: it is added up. Everything below is `
    + `marked against that sum.`);

  set('exact-intro',
    `The recipe is the one from the textbook, run without any of the approximations the `
    + `textbook then spends four chapters on. For each of the ${n0(cube)} weight settings, `
    + `compute how well it fits the ${M.n} points, multiply by the prior, and that is the `
    + `posterior up to a constant. The constant is the sum, which comes out to `
    + `${E.mass.toExponential(3)}, so the log evidence of this model on this data is `
    + `<span class="bold">${f4(E.log_evidence)}</span>. The predictive mean and spread at any `
    + `test point are then weighted averages over the same grid. One thing falls out for free `
    + `and is worth using as a check: flipping the sign of all ${word(3)} weights leaves `
    + `\\(v\\tanh(wx+b)\\) unchanged, so the posterior is exactly symmetric and the posterior `
    + `mean of \\(w\\) and of \\(v\\) has to be zero. Measured: `
    + `${E.mean_w === 0 ? '0 to every digit exported' : E.mean_w.toExponential(1)} and `
    + `${E.mean_v === 0 ? '0' : E.mean_v.toExponential(1)}.`);

  set('exact-gap',
    `And the shape of that band is the first result, because it is not the shape the phrase `
    + `everybody uses would lead you to draw. "Uncertainty grows away from the data" predicts a `
    + `band that flares at both ends. It does not. Averaged over the region where the points `
    + `are, the predictive spread is ${f4(E.regions.on_data)}. Out beyond the last point on `
    + `either side it is ${f4(E.regions.far)}, which is `
    + `<span class="bold">${f2(E.far_ratio)} times</span> as much, that is to say the same `
    + `number. A \\(\\tanh\\) saturates, so far from the origin every function the posterior `
    + `still believes in has flattened out at the same height, and the model is not confused `
    + `out there at all. Where the spread does grow is the <span class="bold">gap</span> `
    + `between the two clumps of data, from ${f2(E.gap_lo)} to ${f2(E.gap_hi)}: `
    + `${f4(E.regions.in_gap)}, a factor of <span class="bold">${f2(E.gap_growth)}</span>. `
    + `That is the honest version of the sentence: uncertainty grows where the data do not `
    + `pin the function down, and being far away is only one way of not being pinned down.`);

  set('chains-intro',
    `A grid is not how anybody computes a posterior, so before this one is used as an answer `
    + `key it is worth checking against the method that is. ${word(data.chains.length)} `
    + `Metropolis chains, ${n0(data.chains[0].n + 1)} steps each, acceptance rate around `
    + `${f2(data.chains.reduce((s, c) => s + c.rate, 0) / data.chains.length)}, started from `
    + `different places and never told about each other. Their predictive means are drawn `
    + `against the grid's.`);

  set('chains-note',
    `Where the disagreement is, and why, is the useful part. Outside the gap, chains and grid `
    + `agree to <span class="bold">${f4(C.tight)}</span>, which for a predictive mean that runs `
    + `from ${f2(Math.min(...E.pred_mean))} to ${f2(Math.max(...E.pred_mean))} is agreement. `
    + `The worst disagreement on the whole `
    + `axis is ${f4(C.mean)}, and it is at \\(x = ${f2(C.worst_at)}\\), inside the gap. Look at `
    + `where the chains ended up and it stops being mysterious: two of them settled around `
    + `\\(w = ${f2(data.chains[0].theta_mean[0])}\\) and two around `
    + `\\(w = ${f2(data.chains[1].theta_mean[0])}\\). Those are the two symmetric modes, they `
    + `describe <span class="bold">the same function</span>, and no chain crossed between them `
    + `in ${n0(data.chains[0].n + 1)} steps. Where the data decide the answer that costs `
    + `nothing. In the gap, where what decides the answer is the shape of the posterior's `
    + `tails, it costs ${f4(C.mean)}, and the ${word(data.chains.length)} chains already `
    + `disagree <span class="bold">with each other</span> by ${f4(C.spread)} there. The grid `
    + `is not off to one side of them: it sits inside their envelope at `
    + `${C.n_points - C.outside_band} of ${C.n_points} points, and where it steps outside it `
    + `steps out by ${C.excursion.toExponential(0)}.`);

  set('approx-intro',
    `Now the part that needs an answer key. These are the things people actually run: keep the `
    + `single best fit and quote it; put a Gaussian at the best fit using the curvature there; `
    + `train ${word(8)} networks and average them; train ${word(8)} on resampled data and `
    + `average those; fit a factorised Gaussian by optimisation; and run gradient descent with `
    + `noise injected so it wanders the posterior instead of settling. Each one is scored `
    + `against the exact predictive distribution, point by point, by total variation distance, `
    + `which is a real distance between distributions and not a summary of two of their `
    + `moments.`);

  set('approx-ensemble',
    `One row of that table is worth stopping on, because it is the row that contradicts the `
    + `reason people reach for the method. `
    + `${flat
      ? `An ensemble of ${word(8)} networks trained on the same data with different random `
        + `starts gives a spread, in the gap, of exactly `
        + `<span class="bold">${f3(by.ensemble.sd_ratio_in)}</span> of the right one. Not a `
        + `small fraction: none. Its distance to the exact predictive, `
        + `${f4(by.ensemble.tv_all)}, is the same number as the single best fit's to six `
        + `decimals, because all ${word(8)} members converged to the same function. With `
        + `${word(3)} weights there is one optimum, up to the sign flip that changes nothing, `
        + `and a deterministic optimiser finds it from anywhere. The spread that makes deep `
        + `ensembles work is not a property of ensembling; it is a property of not being able `
        + `to solve the optimisation problem. Change what varies between members from the `
        + `starting point to the data, by resampling, and the spread appears: `
        + `${f3(by.bootstrap.sd_ratio_in)} of the right one.`
      : `An ensemble of ${word(8)} networks trained on the same data recovers `
        + `${f3(by.ensemble.sd_ratio_in)} of the right spread in the gap, against `
        + `${f3(by.bootstrap.sd_ratio_in)} when the data are resampled instead.`} `
    + `And the two rows underneath are the ones that pay: `
    + `<span class="bold">${best.label}</span> lands at ${f4(best.tv_all)}, `
    + `${(worstRow.tv_all / best.tv_all).toFixed(1)} times closer to the truth than `
    + `${worstRow.label} at ${f4(worstRow.tv_all)}, and `
    + `${by.sgld.sd_ratio_in > by.vi.sd_ratio_in
      ? `SGLD is the one that gets the size of the gap nearly right `
        + `(${f3(by.sgld.sd_ratio_in)} of it)`
      : `mean field VI is the one that gets the size of the gap nearly right `
        + `(${f3(by.vi.sd_ratio_in)} of it)`}. `
    + `Laplace is the interesting failure: at ${f3(by.laplace.sd_ratio_in)} it is the only row `
    + `that is <span class="bold">too wide</span> rather than too narrow, because a Gaussian `
    + `fitted to the curvature at one mode knows nothing about the other one.`);

  set('prior-intro',
    `Every number on this page depends on a choice that is almost never mentioned when a `
    + `Bayesian network is presented: how wide the prior on the weights is. It is not a `
    + `formality. A prior over weights is a prior over <span class="bold">functions</span>, `
    + `and the functions it puts mass on are what the posterior falls back to wherever the `
    + `data are silent. The first view draws what each width actually believes, before it has `
    + `seen anything.`);

  set('prior-note',
    `Read those two columns against each other and the prior turns out to be two different `
    + `things at once. Where the data are, it is invisible: the predictive spread goes from `
    + `${f4(lowPrior.sd_in)} at width ${lowPrior.scale} to ${f4(highPrior.sd_in)} at width `
    + `${highPrior.scale}, which is a change of `
    + `${pc(Math.abs(highPrior.sd_in / lowPrior.sd_in - 1))} across a factor of `
    + `${highPrior.scale / lowPrior.scale} in the assumption. In the gap it is everything: `
    + `${f3(lowPrior.sd_gap)} against <span class="bold">${f3(highPrior.sd_gap)}</span>, a `
    + `factor of ${f1(highPrior.sd_gap / lowPrior.sd_gap)}. The width you never reported `
    + `decides the entire answer in exactly the region where anybody would look at a Bayesian `
    + `model for guidance. There is a defence, and it is the log evidence in the same table: `
    + `it peaks at width <span class="bold">${bestPrior.scale}</span> `
    + `(${f2(bestPrior.log_evidence)}, against ${f2(lowPrior.log_evidence)} at `
    + `${lowPrior.scale} and ${f2(highPrior.log_evidence)} at ${highPrior.scale}), so the data `
    + `do have an opinion about the assumption. That opinion is worth `
    + `${f1(bestPrior.log_evidence - highPrior.log_evidence)} in log evidence, which is a `
    + `preference and not a proof.`);

  rows('#approx-table', [...A].sort((a, b) => a.tv_all - b.tv_all), (d) => [
    { html: `<span class="bold">${d.label}</span>` },
    f4(d.tv_all),
    f3(d.sd_ratio_in),
    f3(d.sd_ratio_out),
  ]);

  set('v-map-cost', `One optimisation, and the cheapest thing on the page.`);
  set('v-map-good', `The centre of the band: its mean is off by ${f4(by.map.mean_abs)} on `
    + `average.`);
  set('v-map-bad', `There is no band. Spread ${f3(by.map.sd_ratio_in)} of the right one, `
    + `everywhere, by construction.`);
  set('v-lap-cost', `One optimisation and one Hessian, so barely more than the above.`);
  set('v-lap-good',
    `It is the only cheap row that is not too narrow: ${f3(by.laplace.sd_ratio_in)} of the `
    + `right spread in the gap.`);
  set('v-lap-bad',
    `It only knows the mode it was fitted at. Distance ${f4(by.laplace.tv_all)}, worse than `
    + `${best.label}.`);
  set('v-ens-cost', `${word(8)} optimisations, so ${word(8)} times the above.`);
  set('v-ens-good',
    `With resampled data it does buy spread: ${f3(by.bootstrap.sd_ratio_in)} of the right one `
    + `in the gap.`);
  set('v-ens-bad',
    flat
      ? `With the same data it buys none at all: ${f3(by.ensemble.sd_ratio_in)}, identical to `
        + `one fit, because every member finds the same optimum.`
      : `With the same data it buys ${f3(by.ensemble.sd_ratio_in)}, most of the way to one `
        + `fit.`);
  set('v-vi-cost', `One optimisation over twice as many parameters, plus a sampler inside the `
    + `gradient.`);
  set('v-vi-good', `Distance ${f4(by.vi.tv_all)}, ${by.vi.tv_all <= best.tv_all
    ? 'the closest row on the page' : `behind ${best.label}`}.`);
  set('v-vi-bad',
    `Factorised, so it cannot represent a posterior whose weights depend on each other, and it `
    + `comes out ${by.vi.sd_ratio_in < 1 ? 'narrow' : 'wide'} at `
    + `${f3(by.vi.sd_ratio_in)} in the gap.`);
  set('v-mc-cost', `Thousands of passes over the data, and no way to know when to stop.`);
  set('v-mc-good',
    `The size of the uncertainty, better than anything cheaper: ${f3(by.sgld.sd_ratio_in)} of `
    + `the right spread in the gap.`);
  set('v-mc-bad',
    `It samples one mode at a time. The ${word(data.chains.length)} exact chains above never `
    + `crossed between the two in ${n0(data.chains[0].n + 1)} steps.`);

  set('closing-1',
    `The honest summary of this page is shorter than the page. Every approximation here is `
    + `wrong in the same direction: they get the centre of the prediction nearly right and the `
    + `size of the doubt wrong, and they get it wrong precisely in the region where the doubt `
    + `was the whole reason for being Bayesian. The gap is where a real deployment meets the `
    + `input it was not trained on, and it is where the spread runs from `
    + `${f3(by.map.sd_ratio_in)} of the truth to ${f3(by.sgld.sd_ratio_in)} of it depending on `
    + `which method you happened to pick. None of that is visible without the answer key, and `
    + `the answer key exists here only because the network has ${word(3)} weights.`);

  set('closing-2',
    `The next article keeps the model and throws away the uncertainty, which is the other way `
    + `to be honest about a prediction: instead of asking what the model believes, ask what it `
    + `can be held to. That is <a href="../attribution/">the first article of 7.3</a>, where `
    + `the question stops being how sure the model is and becomes what it used, and the same `
    + `discipline applies: build the answer key first, then look at the picture.`);

  set('ref-note',
    `The model is deliberately the smallest thing that is still a neural network: `
    + `\\(f(x) = v\\tanh(wx+b)\\), ${word(3)} weights, prior width ${M.prior}, noise `
    + `${M.sigma}, ${M.n} observations, seed ${M.seed}. The exact posterior is a sum over a `
    + `${M.grid} cubed grid on \\([-${M.lim}, ${M.lim}]^3\\), evaluated in slices because the `
    + `predictive density over ${M.y_grid} output values would otherwise be a tensor of `
    + `several gigabytes. Total variation distances are computed on that same output grid.`);
}
