/* Every sentence with a measured number, composed from the measurement.
 *
 * The rule this article leans on hardest is the one about directions: almost
 * every claim here is "bigger than", "the wrong sign", "still there". Those
 * are figures too, so they are branches on the data rather than sentences
 * somebody typed while looking at one run.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));
const times = (v) => (v >= 10 ? `${Math.round(v)}` : f1(v));
const f1 = (v) => v.toFixed(1);

export const PROSE_IDS = [
  'intro-predict', 'intro-truth', 'intro-machine',
  'engine-intro', 'engine-note', 'engine-strata',
  'wine-intro', 'wine-count',
  'cost-intro', 'cost-collider',
  'worlds-intro', 'worlds-verdict',
  'v-naive', 'v-naive-watch', 'v-adjust', 'v-adjust-watch',
  'v-collider', 'v-collider-watch', 'v-graph', 'v-graph-watch',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const E = data.engine;
  const S = data.strata;
  const P = data.proxy;
  const C = data.collider;
  const W = data.worlds;
  const N = data.wine;

  const strataFive = S.find((r) => r.k === 5) ?? S[3];
  const strataBest = S[S.length - 1];
  const proxyHalf = P.find((r) => r.noise === 0.5) ?? P[3];
  const colliderLoss = Math.abs(C.with_collider - C.truth) / C.truth;
  const best = N.best;
  const withinMax = Math.max(...best.within.map(Math.abs));
  const sameSign = best.within.every((v) => Math.sign(v) === Math.sign(best.within[0]));

  set('intro-predict',
    `Every model in this atlas so far has answered the same shape of question: given `
    + `what I can see, what is the number I cannot see yet. Prices, categories, the next `
    + `word, tomorrow's reading. This article asks a different one, and the difference is `
    + `not a matter of degree. <span class="bold">What would happen if somebody reached in `
    + `and changed the treatment</span> is not a prediction, and no column in any dataset `
    + `contains it, because a dataset records a world where nobody reached in.`);

  set('intro-truth',
    `That creates a problem for the rule this atlas runs on, which is that every figure is `
    + `measured rather than cited. There is no ground truth column to score against. So the `
    + `world is <span class="bold">written down instead of collected</span>, the same move `
    + `the scene in the radiance field article and the sprites in the style article are `
    + `built on: if the mechanism is a formula, the causal effect is a parameter you chose, `
    + `and every method can be scored against it exactly.`);

  set('intro-machine',
    `Here is the whole mechanism. A hidden variable \\(U\\) that nobody writes down; a `
    + `treatment \\(X\\) that happens more often when \\(U\\) is large; and an outcome `
    + `\\(Y = ${M.base} + ${M.beta}X + ${M.gamma}U + \\varepsilon\\). The effect of the `
    + `treatment is <span class="bold">${f2(M.beta)}</span>, because that is what was `
    + `typed. Comparing the treated with the untreated over ${n0(M.n_big)} draws gives `
    + `<span class="bold">${f4(E.naive)}</span>, which is `
    + `${f1(E.naive / M.beta)} times the truth and is not a small-sample problem: the `
    + `standard error of that difference is ${f4(E.se_naive)}.`);

  set('engine-intro',
    `The gap is not a mystery to be discovered by simulation. With this assignment rule the `
    + `hidden variable is a normal cut in half, so the difference in \\(U\\) between the two `
    + `groups is the inverse Mills ratio and comes out in closed form: `
    + `\\(2\\alpha/\\sqrt{1+\\alpha^2}\\cdot\\sqrt{2/\\pi}\\), which at `
    + `\\(\\alpha = ${M.alpha}\\) is ${f4(E.gap_closed)}. Multiply by ${M.gamma} and the `
    + `predicted bias is <span class="bold">${f4(E.bias_closed)}</span>. The measured one is `
    + `<span class="bold">${f4(E.bias_measured)}</span>. The formula was written before the `
    + `sample existed and the slider below moves both of them together.`);

  set('engine-note',
    `The two agree to ${f4(Math.abs(E.bias_measured - E.bias_closed))}, against a standard `
    + `error of ${f4(E.se_naive)} on the measured side, which is the only reason that `
    + `agreement means anything. The other line the widget draws is the honest one: putting `
    + `\\(U\\) into the regression returns <span class="bold">${f4(E.adjusted)}</span> `
    + `against a truth of ${f2(M.beta)}. And the effect itself was checked by a second `
    + `route, simulating the world with the arrow into \\(X\\) cut and the treatment set by `
    + `hand, which gives ${f4(E.ate_sim)}.`);

  set('engine-strata',
    `Adjusting is rarely done by writing the confounder into a linear model, though. It is `
    + `done by splitting people into comparable groups, and that is where the first quiet `
    + `failure lives. With ${word(strataFive.k)} strata of \\(U\\), which is more than most `
    + `published analyses use, the leftover bias is `
    + `<span class="bold">${f4(strataFive.bias)}</span>, or `
    + `${pc(strataFive.bias / M.beta)} of the effect being estimated. It takes `
    + `${n0(strataBest.k)} strata to get it down to ${f4(strataBest.bias)}. Inside a stratum `
    + `the confounder still varies, and inside a stratum there is still confounding.`);

  set('wine-intro',
    `A reversal is usually introduced with one famous table, which makes it look like a `
    + `museum piece: a curiosity that happens to admissions data from 1973. So here it is on `
    + `data this atlas has already used five times, the ${n0(N.n)} bottles of wine from `
    + `three cultivars. Across all the bottles, ${best.xname.replace(/_/g, ' ')} and `
    + `${best.yname.replace(/_/g, ' ')} move <span class="bold">together</span>, with a `
    + `slope of ${f3(best.pooled)}. Inside each of the three cultivars they move `
    + `<span class="bold">apart</span>: ${best.within.map((v) => f3(v)).join(', ')}. `
    + `${sameSign ? 'All three' : 'The cultivars'} disagree with the pooled line, and nobody `
    + `chose this pair to make the point.`);

  set('wine-count',
    `Nobody chose it because it was found by search. Every ordered pair of the `
    + `${n0(N.p)} chemical columns was tested, and `
    + `<span class="bold">${word(N.reversals)} of the ${n0(N.pairs)} pairs</span> reverse `
    + `in all three cultivars at once, which is ${pc(N.rate)} of them. That number only `
    + `means something next to a control, so the cultivar labels were shuffled and the `
    + `search rerun ${n0(200)} times: the average count under shuffling is `
    + `<span class="bold">${f2(N.null_mean)}</span> and ${pc(N.null_zero)} of shuffles find `
    + `none at all. Using a third chemical cut into thirds instead of the cultivar, the rate `
    + `drops to ${pc(N.third_reversals / N.third_pairs, 2)}, which says the reversals track `
    + `a real grouping rather than any grouping.`);

  set('cost-intro',
    `So adjust for the confounder. The trouble is that "adjust" hides three separate things `
    + `that can go wrong, and from outside the three look identical: a number, reported with `
    + `a confidence interval, that is not the effect. The first is coarseness, measured `
    + `above. The second is measurement: adjusting for a noisy copy of the confounder `
    + `corrects only the part the copy explains. With noise equal to half the confounder's `
    + `own spread, the leftover bias is <span class="bold">${f4(proxyHalf.bias)}</span>, `
    + `still ${pc(proxyHalf.bias / M.beta)} of the effect.`);

  set('cost-collider',
    `The third is the one worth stopping on, because it inverts the usual advice. Here the `
    + `treatment is assigned by a coin flip, so there is no confounding at all and the plain `
    + `difference in means is right: <span class="bold">${f4(C.naive)}</span> against a truth `
    + `of ${f2(C.truth)}. Now add a variable that is caused by both the treatment and the `
    + `outcome, the kind of thing that gets recorded because it is downstream of everything, `
    + `and control for it. The answer becomes <span class="bold">${f4(C.with_collider)}</span>, `
    + `which has lost ${pc(colliderLoss)} of the effect, and doing it non-parametrically by `
    + `thirds gives the same verdict (${f4(C.stratified)}). Nothing was measured badly. The `
    + `bias was created by the correction.`);

  set('worlds-intro',
    `Adjusting fixed the first world and broke the third. The natural next question is which `
    + `situation the data are in, and the answer is that the data do not know. Two mechanisms `
    + `are written below. In the first, \\(Z\\) causes both the treatment and the outcome, so `
    + `it is a confounder and must be adjusted for. In the second, the treatment causes `
    + `\\(Z\\) and \\(Z\\) causes the outcome, so \\(Z\\) is a mediator, part of how the `
    + `effect happens, and adjusting for it removes part of the answer.`);

  set('worlds-verdict',
    `The two are not merely similar. Their covariance matrices agree to `
    + `<span class="bold">${W.cov_max_diff.toExponential(1)}</span>, which is exact rather `
    + `than close, because the second world's parameters were solved for rather than fitted; `
    + `simulating ${n0(W.n)} draws of each and comparing sample covariances gives `
    + `${W.cov_sampled_diff.toExponential(1)}. Every statistic anybody can compute is the `
    + `same in both. And the answers are not: the effect is `
    + `<span class="bold">${f3(W.world_a.truth)}</span> in the first and `
    + `<span class="bold">${f3(W.world_b.truth)}</span> in the second. An analyst who `
    + `adjusts reports ${f3(W.world_a.adjusted)} and an analyst who does not reports `
    + `${f3(W.world_b.unadjusted)}. Each is exactly right in one world and wrong in the `
    + `other, and no test on these data can say which.`);

  set('v-naive',
    `${f4(E.naive)} against a truth of ${f2(M.beta)}, with the gap predicted in closed form `
    + `to ${f4(Math.abs(E.bias_measured - E.bias_closed))}`);
  set('v-naive-watch',
    `Right only when nothing decides who is treated, which is what a randomised experiment `
    + `buys and an observational dataset does not.`);
  set('v-adjust',
    `${f4(E.adjusted)} with the confounder itself, but ${f4(strataFive.bias)} of bias left `
    + `at ${word(strataFive.k)} strata and ${f4(proxyHalf.bias)} with a noisy copy`);
  set('v-adjust-watch',
    `Coarse groups and noisy measurements both leave a fraction of the bias behind, and `
    + `neither shows up as anything unusual in the output.`);
  set('v-collider',
    `${f4(C.with_collider)} where doing nothing gives ${f4(C.naive)} and the truth is `
    + `${f2(C.truth)}`);
  set('v-collider-watch',
    `"Control for everything you have" is a rule that manufactures this failure, because the `
    + `variables easiest to collect are often downstream of both.`);
  set('v-graph',
    `Two graphs, one covariance matrix agreeing to ${W.cov_max_diff.toExponential(1)}, and `
    + `two answers ${f3(Math.abs(W.world_b.truth - W.world_a.truth))} apart`);
  set('v-graph-watch',
    `The graph is an input, not a conclusion. It comes from knowing what the variables are, `
    + `and that knowledge is not in the file.`);

  set('closing-1',
    `The uncomfortable part of this article is not the arithmetic, which is a few lines. It `
    + `is that all four rows of that table are the same regression, run four times, and the `
    + `output looks equally respectable every time. There is no diagnostic on the data that `
    + `separates ${f3(W.world_a.truth)} from ${f3(W.world_b.truth)}, no residual plot that `
    + `flags a collider, no goodness-of-fit statistic that notices ${f4(proxyHalf.bias)} of `
    + `leftover bias. What decides the answer is a claim about which variable causes which, `
    + `and that claim comes from outside the file every single time.`);

  set('closing-2',
    `Which leaves a practical problem, and it is the one the rest of this branch is about. `
    + `Suppose you do know the graph and the confounders really are measured. With one `
    + `confounder you can compare like with like. With ten continuous ones there are no two `
    + `people alike anywhere in the data, so the strategy that just worked has no subjects `
    + `left to apply it to. The <a href="../propensity/">next article</a> is about the `
    + `result that rescues it, which says that all of that information can be squeezed into `
    + `a single number without losing what matters.`);

  set('ref-note',
    `The bottles are the ones this atlas keeps reusing: the same ${n0(N.n)} wines and `
    + `${n0(N.p + 1)} measurements that appear in the regression, directions, assumptions, `
    + `distances and outlier articles, loaded from scikit-learn and asserted against their `
    + `shape and totals before the search runs. The synthetic worlds use seed ${M.seed}, `
    + `${n0(M.n_big)} draws for the headline figures, and ${n0(200)} permutations for the `
    + `shuffled control.`);
}
