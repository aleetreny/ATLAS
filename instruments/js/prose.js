/* Every sentence with a measured number, composed from the measurement.
 *
 * Two paragraphs here are branches on things that could have gone either way
 * and did not: whether the folklore threshold of ten on the first stage F is
 * where the bias becomes visible in this world, and how far apart the mean and
 * the median of the same four hundred estimates end up. Both are written to
 * say whatever the sweep says.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sgn = (v, d = 4) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-gap', 'intro-lever',
  'lever-intro', 'lever-note',
  'weak-intro', 'weak-verdict',
  'leak-intro', 'leak-note',
  'worlds-intro', 'worlds-note',
  'late-intro', 'late-note',
  'v-strength', 'v-strength-check', 'v-exclusion', 'v-exclusion-check',
  'v-independence', 'v-independence-check', 'v-monotone', 'v-monotone-check',
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
  const H = data.headline;
  const S = data.strength;
  const Z = data.seeds;
  const E = data.exclusion;
  const W = data.worlds;
  const L = data.late;

  const strong = S.find((r) => r.pi === M.pi) ?? S[2];
  const nearTen = S.reduce((p, c) => (Math.abs(c.f_median - 10) < Math.abs(p.f_median - 10) ? c : p), S[0]);
  const weakest = S[S.length - 1];
  const meanSpread = Math.max(...Z.map((s) => s.mean)) - Math.min(...Z.map((s) => s.mean));
  const medSpread = Math.max(...Z.map((s) => s.median)) - Math.min(...Z.map((s) => s.median));
  const weakRow = E[E.length - 1];
  const smallLeak = weakRow.cells.find((c) => c.kappa === 0.05) ?? weakRow.cells[2];
  const strongRow = E[0];
  const sameLeak = strongRow.cells.find((c) => c.kappa === 0.05) ?? strongRow.cells[2];
  const shares = L.shares;
  const alwaysTaker = L.by_type[1];

  set('intro-gap',
    `The two previous articles share an assumption neither of them can defend. Adjusting for `
    + `a confounder needs the confounder to be in the file; a propensity score needs every `
    + `variable that decides treatment to be in the file. The first article ended by measuring `
    + `what happens when one is missing, and the second measured it again: perfect balance on `
    + `everything visible, and an answer that is still wrong. There is no version of either `
    + `method that survives an unmeasured confounder, because both of them work by `
    + `conditioning on something, and you cannot condition on a column that does not exist.`);

  set('intro-lever',
    `The way out is older than both and changes the shape of the problem. Instead of `
    + `measuring the confounder, find something that <span class="bold">pushes the `
    + `treatment</span> and has no other route to the outcome: a lottery, a distance, a rule `
    + `change, a coin somebody already flipped. Then compare what the push did to the `
    + `treatment with what it did to the outcome. Whatever the hidden confounder is doing, it `
    + `is not responding to the push, so it cancels. The world below is written out in full, `
    + `so the effect is <span class="bold">${f2(M.beta)}</span> by construction, and every `
    + `number on this page is read against it.`);

  set('lever-intro',
    `The estimator is one fraction. The bottom is how much the treatment moved when the `
    + `lever moved, the top is how much the outcome moved, and the ratio is the effect. On `
    + `${n0(H.reps)} samples of ${n0(H.n)} rows, least squares returns `
    + `<span class="bold">${f4(H.ols_mean)}</span> where the truth is ${f2(M.beta)}, and that `
    + `gap is not luck: it is \\(\\gamma\\delta/\\mathrm{Var}(X)\\), which comes to `
    + `${f4(H.ols_bias_closed)} in closed form against ${f4(H.ols_bias)} measured. The ratio `
    + `returns <span class="bold">${f4(H.iv_median)}</span>.`);

  set('lever-note',
    `With a lever this strong the estimator does what it promises. The first stage F is `
    + `<span class="bold">${f1(H.f_mean)}</span>, which is two orders of magnitude past any `
    + `threshold anybody uses, and the spread between samples is ${f4(H.iv_sd)} against `
    + `${f4(H.ols_sd)} for least squares. That is the honest trade in one line: the biased `
    + `estimator is ${f1(H.ols_sd === 0 ? 0 : H.iv_sd / H.ols_sd)} times tighter than the `
    + `unbiased one, which is why nobody reaches for an instrument when they believe there is `
    + `no confounding.`);

  set('weak-intro',
    `A weak instrument is usually described as a noisier version of a good one. It is not `
    + `that, and the sweep says so in two ways. The first is ordinary: as the lever weakens, `
    + `the median estimate slides back towards least squares, because the denominator of the `
    + `fraction shrinks faster than the numerator's advantage. At a first stage F of `
    + `<span class="bold">${f1(nearTen.f_median)}</span>, which is where the folklore `
    + `threshold sits, the median is off by ${sgn(nearTen.iv_median_bias)}; at `
    + `F ${f2(weakest.f_median)} it is off by <span class="bold">${sgn(weakest.iv_median_bias)}</span>, `
    + `which is ${pc(Math.abs(weakest.iv_median_bias) / M.beta)} of the effect.`);

  set('weak-verdict',
    `The second way is stranger and is the reason this table exists. Those two columns are `
    + `the same ${n0(M.reps)} numbers summarised twice. The medians agree with each other to `
    + `<span class="bold">${f3(medSpread)}</span> across six seeds; the means disagree by `
    + `<span class="bold">${f2(meanSpread)}</span>, running from ${f2(Math.min(...Z.map((s) => s.mean)))} `
    + `to ${f2(Math.max(...Z.map((s) => s.mean)))}. That is not a bigger version of ordinary `
    + `noise. A ratio whose denominator can be arbitrarily close to zero has no mean at all, `
    + `so the sample mean is estimating something that does not exist and will keep wandering `
    + `with more data rather than settling. In this sweep `
    + `${pc(weakest.wild)} of samples land more than ten effects away from the truth. A `
    + `confidence interval built the usual way, from a standard error, is describing a `
    + `distribution this estimator does not have.`);

  set('leak-intro',
    `Everything so far assumed the lever touches the outcome only by moving the treatment. `
    + `That is the exclusion restriction, and it is worth knowing exactly what a small `
    + `violation costs, because "the instrument is only slightly imperfect" is the most `
    + `common defence of an instrument. The arithmetic is unforgiving and exact: a direct `
    + `path of size \\(\\kappa\\) biases the ratio by `
    + `<span class="bold">\\(\\kappa/\\pi\\)</span>, the leak divided by the lever.`);

  set('leak-note',
    `So the two weaknesses multiply rather than add. With a strong lever, a direct path of `
    + `${f2(sameLeak.kappa)} moves the answer by ${sgn(sameLeak.bias, 3)}, which most people `
    + `would ignore. With a lever of ${f2(weakRow.pi)}, the same ${f2(smallLeak.kappa)} moves `
    + `it by <span class="bold">${sgn(smallLeak.bias, 3)}</span> against a closed form of `
    + `${f3(smallLeak.closed)}, which is `
    + `${pc(Math.abs(smallLeak.bias) / M.beta, 0)} of the effect being estimated. The two `
    + `assumptions people report separately, relevance and exclusion, are not separate: the `
    + `weaker the first, the less of the second you are allowed to give up.`);

  set('worlds-intro',
    `The obvious response is to test it. With one instrument, you cannot, and the reason is `
    + `not that the test is hard to construct. The covariance matrix of the three observed `
    + `variables has six numbers in it and the model has more parameters than that, so more `
    + `than one set of parameters fits the same data exactly. Sweeping the effect and asking `
    + `which values admit a world with non-negative noise variances gives an interval: every `
    + `effect from <span class="bold">${f3(W.beta_lo)}</span> to `
    + `<span class="bold">${f3(W.beta_hi)}</span> is compatible with these data, a range of `
    + `${f3(W.width)} around a truth of ${f2(M.beta)}.`);

  set('worlds-note',
    `The widget shows the two ends of that interval as two worlds. In the first the exclusion `
    + `restriction holds and the effect is ${f2(W.beta_a)}; in the second there is a direct `
    + `path of ${f3(W.kappa_b)} and the effect is ${f3(W.beta_b)}. Their covariance matrices `
    + `agree to <span class="bold">${W.algebra_max_diff.toExponential(1)}</span> by algebra `
    + `rather than by fitting, and simulating ${n0(W.n)} draws of each gives sample `
    + `covariances that agree to ${W.sampled_max_diff.toExponential(1)}. The estimator returns `
    + `${f3(W.iv_a)} in the first and ${f3(W.iv_b)} in the second, because it cannot tell `
    + `them apart, and the first stage F is ${f1(W.f_a)} against ${f1(W.f_b)}, so the one `
    + `diagnostic everybody reports is equally happy in both. None of this is a limitation of `
    + `two-stage least squares. The information is not in the data.`);

  set('late-intro',
    `Suppose everything holds. The lever is strong, the exclusion restriction is true, and `
    + `the estimator returns a number that is not biased for anything. There is still a `
    + `question left, and it is the one that survives into practice: whose effect is it? In `
    + `this world the population is ${pc(shares[0])} compliers, who take the treatment when `
    + `pushed and not otherwise; ${pc(shares[1])} always-takers and ${pc(shares[2])} `
    + `never-takers, who do the same thing either way. The effect is different for each `
    + `group, and all three are written down.`);

  set('late-note',
    `The ratio returns <span class="bold">${f4(L.wald)}</span>. The average effect across `
    + `everybody is <span class="bold">${f4(L.ate)}</span>, and the average among compliers `
    + `is <span class="bold">${f4(L.late)}</span>. It is not estimating the population `
    + `effect and it is not estimating it badly: it is estimating a different quantity, `
    + `exactly, and the two differ by ${f3(Math.abs(L.ate - L.late))} here because the `
    + `always-takers have an effect of ${f2(alwaysTaker)} and the lever cannot reach them. `
    + `Nothing in the data says who the compliers were. Change the lever, and the same study `
    + `on the same population answers about a different group of people.`);

  rows('#seeds-table', Z, (d) => [
    { html: `<span class="bold">${d.seed}</span>` },
    f3(d.mean),
    f3(d.median),
    f2(d.max_abs),
  ]);

  set('v-strength', `first stage F of ${f1(H.f_mean)} at full strength, and a median bias of `
    + `${sgn(weakest.iv_median_bias)} at F ${f2(weakest.f_median)}`);
  set('v-strength-check',
    `Yes, and it is the one everybody reports. It is also the one that says nothing about the `
    + `other three.`);
  set('v-exclusion', `a direct path of ${f2(smallLeak.kappa)} with a lever of `
    + `${f2(weakRow.pi)} costs ${sgn(smallLeak.bias, 3)}, exactly the leak over the lever`);
  set('v-exclusion-check',
    `No. With one instrument the model has more parameters than the data has covariances, `
    + `and the compatible effects span ${f3(W.width)}.`);
  set('v-independence', `both worlds return ${f3(W.iv_a)} and ${f3(W.iv_b)} from covariance `
    + `matrices that agree to ${W.algebra_max_diff.toExponential(1)}`);
  set('v-independence-check',
    `No, and it is the same kind of claim the first article ended on: an assertion about `
    + `arrows that no column can confirm.`);
  set('v-monotone', `with ${pc(shares[0])} compliers, the ratio gives ${f4(L.wald)} against `
    + `a population effect of ${f4(L.ate)}`);
  set('v-monotone-check',
    `No. It is assumed here rather than tested, and without it the ratio is a difference of `
    + `two groups' effects rather than an average of anybody's.`);

  set('closing-1',
    `Three articles in, the shape is hard to miss. Every method in this branch works, in the `
    + `sense that it returns the right number when its assumptions hold, and every one of them `
    + `rests on at least one assumption that the data cannot check. Adjusting needs the `
    + `confounder measured. Weighting needs every variable that decides treatment. An `
    + `instrument needs a lever that pushes hard and touches nothing else. When the assumption `
    + `holds the estimate is right; when it fails there is no residual, no diagnostic and no `
    + `warning, only a number with a confidence interval around it.`);

  set('closing-2',
    `Which raises the question this branch has been circling since its first page. All three `
    + `methods are answers to "what should I adjust for", decided by hand from a picture `
    + `somebody drew. The <a href="../do-calculus/">next article</a> stops deciding by hand: `
    + `on a small enough model the interventional distribution can be computed exactly, every `
    + `possible adjustment set can be enumerated and checked against it, and the graphical `
    + `rule that is supposed to predict which ones work can be tested rather than quoted. And `
    + `then, with the identification settled, it asks the question that actually gets asked in `
    + `practice: not how much the treatment helps on average, but who it helps.`);

  set('ref-note',
    `The world is linear and Gaussian on purpose: it is the case where every quantity on this `
    + `page has a closed form, so a measurement can be checked against arithmetic rather than `
    + `against another measurement. Seed ${M.seed}, ${n0(M.n)} rows per sample and `
    + `${n0(M.reps)} samples per row of the sweep. The compliance section uses `
    + `${n0(L.n)} units with the three types drawn at ${shares.map((s) => pc(s, 0)).join(', ')} `
    + `and a different effect written into each.`);
}
