/* Every sentence with a measured number, composed from the measurement.
 *
 * The one to be careful with is the coverage paragraph, because "exactly
 * ninety percent" is a claim that a reader can check against the table two
 * inches below it: the formula, the measurement and the spread all come from
 * the same export and none of them is typed.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const f6 = (v) => v.toFixed(6);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-point', 'intro-recipe',
  'exact-intro', 'exact-note',
  'models-intro', 'models-note',
  'cond-intro', 'cond-note',
  'exch-intro', 'exch-note',
  'sets-intro', 'sets-note',
  'v-any', 'v-any-watch', 'v-exact', 'v-exact-watch',
  'v-cond', 'v-cond-watch', 'v-exch', 'v-exch-watch',
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
  const X = data.exact;
  const MO = data.models;
  const C = data.conditional;
  const E = data.exchange;
  const S = data.classification;

  const hundred = X.rows.find((r) => r.n === 100) ?? X.rows[2];
  const smallest = X.rows[0];
  const biggest = X.rows[X.rows.length - 1];
  const worstGap = Math.max(...X.rows.map((r) => Math.abs(r.mean - r.formula)));
  const byModel = Object.fromEntries(MO.rows.map((r) => [r.key, r]));
  const covs = MO.rows.map((r) => r.coverage);
  const widths = MO.rows.map((r) => r.width);
  const byCond = Object.fromEntries(C.rows.map((r) => [r.key, r]));
  const bySet = Object.fromEntries(S.rows.map((r) => [r.key, r]));

  set('intro-point',
    `Almost every number this atlas has published is an average: a mean squared error, an `
    + `accuracy, a log likelihood. An average error of 0.3 does not say what to do with `
    + `tomorrow's prediction. This article is about the other output, an interval, and about `
    + `one particular way of building one that has a property the rest do not: it works for `
    + `<span class="bold">any</span> model, including a bad one, and its guarantee holds at `
    + `${n0(M.n)} rows rather than in the limit.`);

  set('intro-recipe',
    `The recipe is three lines and never mentions the model. Fit on one part of the data. On `
    + `a second part, the calibration set, record how wrong the model was on each row. Then `
    + `for a new row, take the interval of everything within the `
    + `\\(\\lceil (n+1)(1-\\alpha) \\rceil\\)-th smallest of those errors. That is all. The `
    + `model can be a neural network or the mean of the training set, and the sentence that `
    + `follows is true either way.`);

  set('exact-intro',
    `The guarantee is not asymptotic and it is not approximate: coverage is exactly `
    + `\\(\\lceil (n+1)(1-\\alpha)\\rceil / (n+1)\\). With `
    + `${n0(hundred.n)} calibration points and a target of ${pc(1 - M.alpha, 0)} that comes `
    + `to <span class="bold">${f6(hundred.formula)}</span>, and over ${n0(X.trials)} `
    + `calibration sets the measurement is <span class="bold">${f6(hundred.mean)}</span>. `
    + `The worst disagreement anywhere in the table is `
    + `${worstGap.toExponential(1)}. Nothing was fitted to make that happen: the guarantee `
    + `depends only on ranks, so the scores can be drawn from any continuous distribution `
    + `and the coverage of each calibration set computed in closed form.`);

  set('exact-note',
    `The last two columns are the ones that change how this should be read. That exact `
    + `${f6(hundred.formula)} is an average over calibration sets, and any particular one is `
    + `a draw. With ${n0(hundred.n)} calibration points the spread between them is `
    + `<span class="bold">${f3(hundred.sd)}</span>, which the Beta distribution predicts as `
    + `${f3(hundred.sd_closed)}, and <span class="bold">${pc(hundred.below)}</span> of `
    + `calibration sets end up below the target. At ${n0(smallest.n)} points the spread is `
    + `${f3(smallest.sd)}, so a calibration set of that size can easily deliver `
    + `${pc(smallest.q05)} instead of ${pc(1 - M.alpha, 0)}. It takes ${n0(biggest.n)} points `
    + `to get the spread down to ${f3(biggest.sd)}. The guarantee is about the procedure, and `
    + `a practitioner runs it once.`);

  set('models-intro',
    `Nothing in the recipe mentioned the model, and that is testable. Four models on the same `
    + `data: a random forest, a linear fit, a linear fit that only sees one of the three `
    + `features, and one that ignores the features and returns the mean of the training set. `
    + `Over ${n0(MO.splits)} splits their coverages are `
    + `<span class="bold">${MO.rows.map((r) => f4(r.coverage)).join(', ')}</span>, which is `
    + `${f4(Math.max(...covs) - Math.min(...covs))} apart. All four are valid, including the `
    + `one that is not a model.`);

  set('models-note',
    `Validity is free. Width is what the model is for, and there the four are not the same at `
    + `all: <span class="bold">${f3(byModel.forest.width)}</span> for the forest against `
    + `<span class="bold">${f3(byModel.constant.width)}</span> for the mean, a factor of `
    + `${f2(Math.max(...widths) / Math.min(...widths))}. That is the trade this method makes `
    + `explicit and most others hide. A useless model does not produce a wrong interval here; `
    + `it produces an honest interval so wide that nobody would act on it, which is the `
    + `correct thing for it to do and is not what a confidence interval from a badly `
    + `specified regression would have done.`);

  set('cond-intro',
    `Here is the part that decides whether any of this is useful. The data below have two `
    + `groups, an easy one and a hard one whose noise is ${f1(C.sigma[1] / C.sigma[0])} times `
    + `larger, in equal proportion. One interval for everybody gets `
    + `<span class="bold">${f4(byCond.plain.all)}</span> overall, which is the promise kept. `
    + `Inside the easy group it covers <span class="bold">${f4(byCond.plain.g0)}</span> and `
    + `inside the hard one <span class="bold">${f4(byCond.plain.g1)}</span>. The average is `
    + `right and neither group is getting what it says on the label.`);

  set('cond-note',
    `Two repairs, and they buy different things. Calibrating separately within each group `
    + `brings both to ${f4(byCond.mondrian.g0)} and ${f4(byCond.mondrian.g1)}, and the `
    + `intervals stop being the same size: ${f2(byCond.mondrian.w0)} for the easy group `
    + `against ${f2(byCond.mondrian.w1)} for the hard one. It needs somebody to name the `
    + `groups. Conformalised quantile regression gets `
    + `${f4(byCond.cqr.g0)} and ${f4(byCond.cqr.g1)} without being told the groups exist, by `
    + `calibrating a model that already predicts quantiles, and pays for it in width `
    + `(${f2(byCond.cqr.w0)} and ${f2(byCond.cqr.w1)}). Neither is free and both are honest `
    + `about the same thing: <span class="bold">marginal coverage is a property of the `
    + `average customer, and there is no average customer</span>.`);

  set('exch-intro',
    `Everything above needs one assumption, and it is not normality, independence or a `
    + `correct model. It is that the calibration rows and the row you are predicting are `
    + `<span class="bold">exchangeable</span>: any ordering of them is as likely as any `
    + `other. Time series break that by construction, and the test below uses the same `
    + `monthly series of atmospheric carbon dioxide the ARIMA article published, asserted by `
    + `digest before anything runs. Twelve lags as features, a linear model, the first third `
    + `to fit, the second to calibrate, the last to test.`);

  set('exch-note',
    `Split in time order, the intervals cover <span class="bold">${pc(E.chrono.coverage)}</span> `
    + `of the ${n0(E.chrono.n_test)} months they were supposed to cover ${pc(1 - M.alpha, 0)} `
    + `of. The control is what makes that a measurement rather than an anecdote: shuffle the `
    + `same rows, fit the same model, calibrate the same way, and coverage comes back to `
    + `<span class="bold">${pc(E.shuffled.coverage)}</span> with a spread of `
    + `${f3(E.shuffled.coverage_sd)} over ${word(E.reps)} shuffles. The data are identical `
    + `and the model is identical. The only thing that changed is the order, and the residuals `
    + `tell you why: they average ${f3(E.chrono.mean_abs_cal)} on the calibration block and `
    + `${f3(E.chrono.mean_abs_test)} on the test block, because the series is still rising `
    + `and the second half is not the first half.`);

  set('sets-intro',
    `In classification the same recipe returns a set of labels rather than an interval, and `
    + `it inherits the same guarantee: the true label is in the set `
    + `${pc(1 - M.alpha, 0)} of the time. What changes is that the score has to be chosen, and `
    + `two obvious choices behave completely differently on ${n0(S.n)} handwritten digits `
    + `where a forest already gets ${pc(S.top1)} of them right.`);

  set('sets-note',
    `The middle row of that table is the one worth keeping, because it is a failure that `
    + `looks like a success. Adaptive sets built the usual way cover `
    + `<span class="bold">${f4(bySet.aps.cov)}</span> of the time, which is far above the `
    + `${pc(1 - M.alpha, 0)} asked for, and pay for it with `
    + `<span class="bold">${f2(bySet.aps.size)}</span> labels per prediction out of `
    + `${S.classes}. The reason is that the score saturates: with a confident classifier, `
    + `almost every calibration row has nearly all the probability mass on the true label, so `
    + `the quantile lands next to one and the sets swallow everything. Randomising the last `
    + `label's contribution, which is what the paper actually specifies, brings coverage to `
    + `<span class="bold">${f4(bySet.aps_rand.cov)}</span> and the size to `
    + `${f2(bySet.aps_rand.size)}. And the smallest sets of all come from the simplest score `
    + `(${f2(bySet.lac.size)} labels), which pays elsewhere: its worst class is covered `
    + `${pc(bySet.lac.worst)} of the time and ${pc(bySet.lac.empty)} of its predictions are `
    + `<span class="bold">empty sets</span>, a valid answer that says nothing.`);

  rows('#exact-table', X.rows, (d) => [
    { html: `<span class="bold">${n0(d.n)}</span>` },
    n0(d.k),
    f6(d.formula),
    f6(d.mean),
    `${f4(d.sd)} (formula ${f4(d.sd_closed)})`,
    pc(d.below),
  ]);

  rows('#sets-table', S.rows, (d) => [
    { html: `<span class="bold">${d.label}</span>` },
    f4(d.cov),
    f3(d.size),
    f4(d.worst),
    pc(d.empty),
  ]);

  set('v-any', `four models from a forest to a constant, coverage between `
    + `${f4(Math.min(...covs))} and ${f4(Math.max(...covs))}`);
  set('v-any-watch',
    `The constant model's intervals are ${f2(Math.max(...widths) / Math.min(...widths))} `
    + `times wider than the forest's. Valid and useless are compatible.`);
  set('v-exact', `${f6(hundred.mean)} measured against ${f6(hundred.formula)} predicted, with `
    + `${n0(hundred.n)} calibration points`);
  set('v-exact-watch',
    `That is an average over calibration sets. ${pc(hundred.below)} of individual calibration `
    + `sets land below the target, and the spread is ${f3(hundred.sd)}.`);
  set('v-cond', `${f4(byCond.plain.g0)} for the easy group and ${f4(byCond.plain.g1)} for the `
    + `hard one, averaging to ${f4(byCond.plain.all)}`);
  set('v-cond-watch',
    `Group conditional coverage needs the groups named; conformalised quantile regression `
    + `does not, and both cost width.`);
  set('v-exch', `${pc(E.chrono.coverage)} in time order against ${pc(E.shuffled.coverage)} on `
    + `the same rows shuffled`);
  set('v-exch-watch',
    `Nothing in the output flags it. The intervals look exactly like the ones that work.`);

  set('closing-1',
    `This article is the odd one out in this module. The four causal articles all end the `
    + `same way: the estimate is correct if an assumption holds, and no diagnostic fires when `
    + `it does not. Conformal prediction ends differently. Its assumption is about the data `
    + `collection rather than about arrows, it is often knowable, and when it fails the `
    + `failure is at least measurable if you have held out data from the right period. What `
    + `you get in exchange is narrow: a guarantee about a long run of predictions, not about `
    + `the one in front of you, and not about any subgroup you can name.`);

  set('closing-2',
    `What it does not give is the shape of the uncertainty. An interval of width `
    + `${f3(byModel.forest.width)} says nothing about whether the truth is more likely near `
    + `the middle, whether there are two plausible answers rather than one, or how much of `
    + `the doubt would go away with more data. For that you need a distribution over what the `
    + `model itself could have been, which is what the `
    + `<a href="../bayesian-nets/">next article</a> is about. It is the harder path, because `
    + `that distribution cannot be computed for any network anybody uses; so the model there `
    + `is made small enough that it can, and the approximations everybody actually runs are `
    + `scored against the answer.`);

  set('ref-note',
    `The split is the simplest version of conformal prediction and the only one measured `
    + `here: one fit, one calibration set, one quantile. The coverage table uses `
    + `${n0(X.trials)} calibration sets per row with the conditional coverage of each computed `
    + `in closed form rather than estimated. The regression sections use ${n0(M.n)} rows and `
    + `${n0(MO.splits)} splits, the classification section ${n0(S.splits)} splits of `
    + `${n0(S.n)} digits with ${n0(S.cal)} calibration rows, and the carbon dioxide series `
    + `carries digest ${M.co2_digest}, the same one the ARIMA article asserts.`);
}
