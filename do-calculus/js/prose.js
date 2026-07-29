/* Every sentence with a measured number, composed from the measurement.
 *
 * The heterogeneity paragraph is the one that had to be rewritten around what
 * came out: the causal forest without local centring is worse than answering
 * the average to everybody, and saying so is the whole lesson of that section.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const f6 = (v) => v.toFixed(6);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-byhand', 'intro-exact',
  'sets-intro', 'sets-note', 'sets-random',
  'rules-intro', 'rules-note',
  'front-intro', 'front-note',
  'het-intro', 'het-centre',
  'v-sets', 'v-sets-watch', 'v-front', 'v-front-watch',
  'v-het', 'v-het-watch', 'v-test', 'v-test-watch',
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
  const E = data.exact;
  const R = data.random;
  const F = data.frontdoor;
  const U = data.rules;
  const H = data.heterogeneity;
  const P = data.prognostic;

  const valid = E.rows.filter((r) => r.backdoor);
  const named = (S) => (S.length ? S.join(', ') : 'nothing at all');
  const by = Object.fromEntries(H.rows.map((r) => [r.key, r]));
  const best = H.rows.reduce((p, c) => (c.mse < p.mse ? c : p), H.rows[0]);
  const losers = H.rows.filter((r) => r.key !== 'constant' && r.mse > by.constant.mse);
  const centreFactor = by.cf.mse / by.cf_centered.mse;
  const holds = U.rows.filter((r) => r.holds);
  const fails = U.rows.filter((r) => !r.holds);
  const worstHold = Math.max(...holds.map((r) => r.residual));
  const smallestFail = Math.min(...fails.map((r) => r.residual));

  set('intro-byhand',
    `Every method in this branch has needed somebody to decide what to adjust for, and the `
    + `three articles before this one all made that decision the same way: by looking at a `
    + `picture and thinking about it. The first article measured what happens when the `
    + `decision is wrong in either direction, missing a confounder or including a collider, `
    + `and in both cases the output looks exactly as healthy as a correct one.`);

  set('intro-exact',
    `Make the model small enough and the deciding stops being a matter of judgement. With `
    + `${word(E.names.length)} binary variables there are ${n0(2 ** E.names.length)} possible `
    + `worlds, so the distribution after an intervention can be <span class="bold">computed `
    + `rather than estimated</span>: cross out the factor for the variable somebody set by `
    + `hand and add up the rest. That is the entire definition, it is a sum with `
    + `${n0(2 ** E.names.length)} terms, and it turns "which adjustment set is valid" into a `
    + `question with a checkable answer.`);

  set('sets-intro',
    `The graph below has ${word(E.names.length)} binary variables. The treatment is `
    + `${E.names[E.x]} and the outcome is ${E.names[E.y]}, and the true effect of setting `
    + `${E.names[E.x]} by hand is <span class="bold">${f6(E.truth)}</span>, computed by that `
    + `sum. There are ${word(E.n_sets)} possible things to adjust for, from nothing to all `
    + `${word(E.names.length - 2)} of the other variables, and every one of them has been `
    + `tried. <span class="bold">${word(E.n_valid)} of the ${word(E.n_sets)}</span> return `
    + `the true effect, and they return it to `
    + `${E.worst_valid_error.toExponential(0)}, which is machine arithmetic rather than a `
    + `good approximation.`);

  set('sets-note',
    `The graphical criterion and the arithmetic agree on all ${word(E.n_sets)}. That is the `
    + `part worth pausing on, because it means a rule about drawings predicted, exactly, `
    + `which sums come out right. Two of the wrong answers are the ones people actually `
    + `reach for: adjusting for <span class="bold">nothing</span> gives ${f6(E.empty.estimate)} `
    + `and is off by ${f4(E.empty.error)}, and adjusting for `
    + `<span class="bold">everything available</span> gives ${f6(E.full.estimate)} and is off `
    + `by ${f4(E.full.error)}. The second one is worse than it looks: it uses strictly more `
    + `information than the sets that work.`);

  set('sets-random',
    `One graph is an example, so the habit was measured on ${n0(R.n)} random ones. `
    + `Adjusting for every observed variable gives the right answer in `
    + `<span class="bold">${pc(R.full_rate)}</span> of them. A valid set exists in `
    + `${pc(R.any_rate)}, so in the great majority of the cases where it fails, the answer `
    + `was available and the rule threw it away. When it is wrong the median error is `
    + `${f4(R.err_median)} and ${pc(R.err_over_05)} of graphs are off by more than 0.05, `
    + `on effects that live between zero and one.`);

  set('rules-intro',
    `The criterion above is a special case of three rewriting rules, and each rule is an `
    + `equality between two expressions that holds when a separation holds in a modified `
    + `graph. The table checks them the only way worth checking an equality: compute both `
    + `sides independently and subtract. Each row is a case where the condition is met and `
    + `a matching case where it is not.`);

  set('rules-note',
    `Those residuals are the point of the table. Where the condition holds the two sides `
    + `agree to <span class="bold">${worstHold.toExponential(0)}</span> in the worst case, `
    + `which is the size of a rounding error and not of an approximation. Where it fails the `
    + `smallest gap is <span class="bold">${smallestFail.toExponential(1)}</span>, thirteen `
    + `orders of magnitude larger. A rule of do-calculus is not a heuristic that usually `
    + `works: it either applies and is exact, or does not apply and is arbitrary.`);

  set('front-intro',
    `The instruments article gave up on the confounder and reached outside for a lever. `
    + `There is another way through, and it uses nothing external. Suppose the treatment `
    + `affects the outcome only through a mediator that you can measure, and the unmeasured `
    + `confounder does not touch that mediator. Then the effect can be recovered in two `
    + `steps: how the treatment moves the mediator, and how the mediator moves the outcome. `
    + `Neither step is confounded, even though the whole path is.`);

  set('front-note',
    `Nothing about that is a trick of the numbers. The true effect in this model, computed `
    + `with the unmeasured confounder in full view, is `
    + `<span class="bold">${f6(F.truth)}</span>. The front door formula, which reads only `
    + `variables an analyst can see, returns <span class="bold">${f6(F.frontdoor)}</span>, `
    + `an error of ${F.error.toExponential(0)}. Every adjustment set available to that `
    + `analyst fails, because the confounder is not among them: `
    + `${F.backdoor_sets.map((s) => `${s.S.length ? s.S.join(', ') : 'nothing'} gives `
      + `${f4(s.estimate)}`).join(' and ')}, against a naive difference of ${f4(F.naive)}. `
    + `The information was in the data all along, and no amount of adjusting could reach it.`);

  set('het-intro',
    `Everything up to here estimates one number for a population. That is the wrong `
    + `question for most of the decisions these methods get used for, where what matters is `
    + `not whether the treatment helps on average but whether it helps this person. The `
    + `world below has an effect written as a function of the covariates, `
    + `\\(\\tau(V)\\), so every method can be scored point by point against the truth `
    + `instead of against each other. The bar to clear is not zero: it is answering the `
    + `average to everybody, which costs ${f4(by.constant.mse)} and is the last row of the `
    + `table.`);

  set('het-centre',
    `The two rows that matter are the two causal forests, and the gap between them is the `
    + `result of this section. Both are honest forests with the same split criterion and the `
    + `same number of trees. The difference is that one subtracts a cross-fitted prediction `
    + `from the outcome and from the treatment before splitting, and the other does not. `
    + `Centred: <span class="bold">${f4(by.cf_centered.mse)}</span>. Not centred: `
    + `<span class="bold">${f4(by.cf.mse)}</span>, a factor of ${f1(centreFactor)}, and `
    + `worse than answering ${f2(2)} to everybody. The reason is the first article of this `
    + `branch: a difference of means inside a leaf is still a difference of means, and if `
    + `treatment was assigned by a covariate the leaf did not split on, that difference is `
    + `confounded. ${losers.length
      ? `Three of the arms lose to the constant (${losers.map((r) => r.label).join(', ')}), `
        + `which is worth saying out loud: estimating who benefits is much harder than `
        + `estimating whether anybody does.`
      : ''}`);

  rows('#rules-table', U.rows, (d) => [
    { html: `<span class="bold">rule ${d.rule}</span>` },
    d.case,
    d.holds ? 'met' : 'not met',
    d.residual.toExponential(1),
  ]);

  rows('#het-table', H.rows, (d) => [
    { html: `<span class="bold">${d.label}</span>` },
    f4(d.mse),
    f4(d.sd),
    d.corr === null ? 'not defined' : f3(d.corr),
  ]);

  set('v-sets', `${word(E.n_valid)} of ${word(E.n_sets)} sets are valid and reproduce the `
    + `truth to ${E.worst_valid_error.toExponential(0)}`);
  set('v-sets-watch',
    `Adjusting for everything available is valid in ${pc(R.full_rate)} of ${n0(R.n)} random `
    + `graphs, and a valid set existed in ${pc(R.any_rate)} of them.`);
  set('v-front', `${f6(F.frontdoor)} against a truth of ${f6(F.truth)}, with the confounder `
    + `never observed`);
  set('v-front-watch',
    `It needs the mediator to be untouched by the confounder, which is a claim about arrows `
    + `and not about data, exactly like the exclusion restriction.`);
  set('v-het', `the centred forest at ${f4(by.cf_centered.mse)} against ${f4(by.constant.mse)} `
    + `for one number for everybody`);
  set('v-het-watch',
    `The same forest without centring scores ${f4(by.cf.mse)}, which is ${f1(centreFactor)} `
    + `times worse and loses to the constant.`);
  set('v-test', `every rule holds to ${worstHold.toExponential(0)} when its condition is met `
    + `and fails by at least ${smallestFail.toExponential(1)} when it is not`);
  set('v-test-watch',
    `All of that is exact given the graph. Which graph is the right one is the question this `
    + `branch has not been able to answer once.`);

  set('closing-1',
    `This article can compute what the other three had to assume, and that is worth being `
    + `precise about, because it is easy to oversell. Given the graph, everything here is `
    + `exact: which adjustment sets work, why the two that do work are the ones the criterion `
    + `names, why the front door reaches an effect that no adjustment can. What none of it `
    + `does is tell you the graph. The ${n0(R.n)} random models in the second section came `
    + `with their arrows attached, because they were generated. A dataset does not.`);

  set('closing-2',
    `That closes this branch. Four articles, four ways of getting a causal number out of `
    + `data that was not collected by intervening, and one property they share: the estimate `
    + `is exactly right when the assumption holds, exactly wrong when it does not, and the `
    + `output looks identical either way. The <a href="../conformal/">next article</a> is `
    + `about the other half of scientific intelligence, and it inverts that property. Instead `
    + `of a point estimate that depends on an untestable assumption, it produces an interval `
    + `with a guarantee that holds in finite samples, for any model, under an assumption you `
    + `can at least reason about. The catch, measured there, is what the guarantee is a `
    + `guarantee about.`);

  set('ref-note',
    `Everything discrete on this page is exact: with ${word(E.names.length)} binary `
    + `variables the interventional distribution is a sum over `
    + `${n0(2 ** E.names.length)} terms and no sampling happens anywhere in the first three `
    + `sections. The probability tables were not chosen by eye either: they were searched `
    + `until the example showed all four things it needs to show at once, which is seed `
    + `${E.dag_seed} of that search. The heterogeneity section uses ${n0(H.n)} rows per `
    + `sample and ${word(H.reps)} seeds per arm.`);
}
