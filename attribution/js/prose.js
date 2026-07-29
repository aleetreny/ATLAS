/* Every sentence with a measured number, composed from the measurement.
 *
 * The paragraph about the two value functions is written as a branch on which
 * of them gives the unused column more credit, because that direction is the
 * whole point of the section and a sentence typed from memory would have got
 * it right by luck rather than by measurement.
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
const pretty = (s) => s.replace(/_/g, ' ').replace('od280/od315 of diluted wines', 'od280/od315');

export const PROSE_IDS = [
  'intro-nokey', 'intro-keys',
  'exact-intro', 'exact-note', 'exact-interaction',
  'kernel-intro',
  'two-intro', 'two-note',
  'lime-intro', 'lime-note',
  'models-intro',
  'v-eff', 'v-eff-watch', 'v-exact', 'v-exact-watch',
  'v-two', 'v-two-watch', 'v-lime', 'v-lime-watch',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const K = data.answer_key;
  const I = data.interaction;
  const KE = data.kernel;
  const T = data.two_shaps;
  const L = data.lime;
  const MO = data.models;

  const coalitions = 2 ** M.p;
  const cheap = KE.rows[0];
  const dear = KE.rows[KE.rows.length - 1];
  const enough = KE.rows.find((r) => r.top1 >= 1) ?? dear;
  const widths = L.rows;
  const tightest = widths[0];
  const stable = widths.find((r) => r.top_share >= 1) ?? widths[widths.length - 1];
  const tops = new Set(widths.map((r) => r.top));
  const condMore = T.unused_conditional > T.unused_interventional;

  set('intro-nokey',
    `A model returns a number and somebody asks why. The answer that has taken over the field `
    + `is an <span class="bold">attribution</span>: a list with one number per column, saying `
    + `how much of this prediction is that column's doing. It is the most quoted output in `
    + `applied machine learning and almost the only one nobody validates, for a reason that `
    + `sounds decisive: there is no dataset with a column called "the true attribution".`);

  set('intro-keys',
    `There is, if you build one, and this page builds two. The first is a `
    + `<span class="bold">model whose attributions are known in closed form</span>: if the `
    + `model is a sum of one function per column, then the credit due to a column is that `
    + `column's term, by the axioms and not by argument. The second is a setting where `
    + `<span class="bold">both</span> versions of the estimator can be written down exactly, `
    + `so that their disagreement is a subtraction rather than a debate. Everything runs on `
    + `the ${n0(M.rows)} bottles of wine that five other articles of this atlas have used, `
    + `predicting ${pretty(M.target)} from ${word(M.p)} of the other measurements.`);

  set('exact-intro',
    `The definition is a sum over every subset of the columns: for each one, how much the `
    + `prediction moves when this column is added to it, averaged with the weights the axioms `
    + `force. With ${word(M.p)} columns that is <span class="bold">${n0(coalitions)}</span> `
    + `subsets and ${n0(coalitions * M.p / 2)} differences, and nothing about it is `
    + `approximate. The bars below are that sum, and the line across them is the answer key: `
    + `the additive model's own term for each column, centred on the background. The worst `
    + `disagreement across ${word(K.n)} bottles and ${word(M.p)} columns is `
    + `<span class="bold">${K.worst.toExponential(0)}</span>.`);

  set('exact-note',
    `Two identities hold on every instance and both are checked in this page rather than `
    + `asserted. The attributions sum to the prediction minus the average prediction, to `
    + `<span class="bold">${K.worst_eff.toExponential(0)}</span>, which is the property the `
    + `waterfall drawing depends on. And each one equals the model's own additive term, to `
    + `${K.worst.toExponential(0)}. The second is the one worth having: the first would hold `
    + `for a badly computed attribution as long as the errors cancelled.`);

  set('exact-interaction',
    `That answer key exists because the model was additive, and the interesting case is when `
    + `it is not. Take a model written by hand, `
    + `\\(${I.a}x_0 + ${I.b}x_1 + ${I.c}x_0x_1\\), and the axioms say something specific `
    + `about the cross term: it belongs to both columns and gets split `
    + `<span class="bold">exactly in half</span>. Measured across six instances, the two `
    + `halves differ by at most ${I.halves.toExponential(0)} and each column's total matches `
    + `its own term plus its half to ${I.worst.toExponential(0)}. That is worth knowing `
    + `before reading any attribution of a model with interactions in it: a feature that does `
    + `nothing on its own still gets credit, and the credit is real.`);

  set('kernel-intro',
    `Nobody enumerates coalitions for a model with forty columns, because ${n0(coalitions)} `
    + `becomes a trillion. The estimator everybody runs samples coalitions instead and solves `
    + `a weighted regression, and its error is measurable here because the exact answer is `
    + `available. With ${n0(cheap.m)} sampled coalitions the worst attribution is off by `
    + `<span class="bold">${f4(cheap.err)}</span>, which is ${pc(cheap.relative)} of the `
    + `largest attribution in the problem, and the largest column is identified correctly `
    + `${pc(cheap.top1, 0)} of the time. It takes ${n0(enough.m)} to get that to `
    + `${pc(enough.top1, 0)}, and at ${n0(dear.m)} coalitions the error is `
    + `${f4(dear.err)}. The ranking stabilises long before the values do, which is the `
    + `useful thing to know: an attribution plot is trustworthy at a sample size where the `
    + `numbers printed next to the bars are not.`);

  set('two-intro',
    `There is a decision hidden inside the definition, and the name does not mention it. To `
    + `ask what the prediction would be without a column, you have to put something in its `
    + `place. One answer is to substitute a value from another row, which breaks whatever `
    + `correlation that column had with the others; the other is to average over what that `
    + `column would plausibly have been given the rest. Both are called SHAP. With a linear `
    + `model and a Gaussian fit to the columns, both have closed forms, so what follows has `
    + `no sampling noise in it at all.`);

  set('two-note',
    `Across the columns and instances measured, the two versions differ by `
    + `${f4(T.mean_gap)} on average and up to <span class="bold">${f4(T.max_gap)}</span>. `
    + `The clearest case is <span class="bold">${pretty(T.unused_name)}</span>, which the `
    + `model barely uses and which correlates ${sgn(T.unused_corr, 3)} with `
    + `${pretty(T.partner_name)}, a column it does use. Breaking the correlation gives it `
    + `<span class="bold">${f4(T.unused_interventional)}</span> of credit; averaging over the `
    + `conditional gives it <span class="bold">${f4(T.unused_conditional)}</span>, `
    + `${f1(T.unused_conditional / Math.max(T.unused_interventional, 1e-9))} times more. `
    + `${condMore
      ? `Neither number is wrong. The first says what the model does, and it does not read `
        + `that column; the second says what the column tells you, and it tells you a lot, `
        + `because it stands in for one the model reads. A recommendation to drop the column `
        + `follows from the first and a claim that it matters follows from the second, and `
        + `the library that produced the plot decided which.`
      : `The direction came out the other way here, which is worth its own sentence rather `
        + `than a rewrite: on this fit, breaking the correlation gives the unused column more `
        + `credit than averaging over it.`}`);

  set('lime-intro',
    `LIME is quoted in the same breath as SHAP and is a different kind of object. It does not `
    + `divide a prediction into parts: it fits a straight line to the model in a `
    + `neighbourhood of one point and reports the line's slopes. That makes it fast, `
    + `model agnostic and dependent on a parameter that the plots never carry, which is how `
    + `big the neighbourhood is.`);

  set('lime-note',
    `The width is not a display setting. Across ${word(widths.length)} widths on one bottle, `
    + `the column that comes out first `
    + `${tops.size > 1
      ? `<span class="bold">changes ${tops.size === 2 ? 'once' : `${word(tops.size - 1)} times`}</span>`
      : 'stays the same'}, and at the narrowest setting the twelve repetitions with different `
    + `random seeds agree with each other only <span class="bold">${f3(tightest.stability)}</span> `
    + `of the way, on a rank correlation where one is perfect agreement: at that width the `
    + `explanation is mostly the seed. By ${f2(stable.width)} the runs agree at `
    + `${f3(stable.stability)} and pick the same top column every time. And the thing they `
    + `agree on is not the model's local behaviour: against the true local gradient, measured `
    + `by finite differences on the model itself, the rank correlation is `
    + `<span class="bold">${sgn(stable.grad_corr, 3)}</span>. A wider neighbourhood buys `
    + `stability by explaining something further away.`);

  set('models-intro',
    `One last comparison, and it decides how much weight an explanation can carry. Two `
    + `forests, the same data, different randomisation. Their test errors are `
    + `${f4(MO.mse_a)} and ${f4(MO.mse_b)}, a difference of ${f4(MO.mse_gap)} against a `
    + `spread between seeds of ${f4(MO.mse_seed_spread)}, so by any measurement on this page `
    + `they are the same model. Their exact attributions on the same bottles correlate `
    + `<span class="bold">${f3(MO.rank_corr)}</span> in rank, and they name the same most `
    + `important column <span class="bold">${pc(MO.same_top)}</span> of the time. Everything `
    + `about the first four sections is exact, and it is exact about a model, not about the `
    + `world: swap in an equally good model and one prediction in ${Math.round(1 / (1 - MO.same_top))} `
    + `gets a different headline.`);

  set('v-eff', `the attributions sum to the prediction minus the average, to `
    + `${K.worst_eff.toExponential(0)}`);
  set('v-eff-watch',
    `That identity is forced by the definition. It holds for a wrong set of values too, as `
    + `long as their errors cancel, so it is not evidence of anything on its own.`);
  set('v-exact', `enumeration matches the closed form to ${K.worst.toExponential(0)}; `
    + `${n0(cheap.m)} sampled coalitions leave ${f4(cheap.err)} of error`);
  set('v-exact-watch',
    `The ranking settles by ${n0(enough.m)} coalitions, and the values are still moving in `
    + `the third decimal at ${n0(dear.m)}.`);
  set('v-two', `${f4(T.unused_interventional)} against ${f4(T.unused_conditional)} for the `
    + `same column, from the two value functions`);
  set('v-two-watch',
    `Which one your library computes is usually decided by the model type, not by the `
    + `question you asked.`);
  set('v-lime', `top column stable only above width ${f2(stable.width)}; rank correlation `
    + `with the true local gradient ${sgn(stable.grad_corr, 3)}`);
  set('v-lime-watch',
    `Two equally accurate models agree on the top column ${pc(MO.same_top)} of the time.`);

  set('closing-1',
    `Everything here is about ten columns and a prediction that is one number, and within `
    + `those limits the picture is clear: the values can be made exact, the identities hold, `
    + `the approximation error is measurable, and the two remaining disagreements are choices `
    + `rather than errors. It is worth being precise about what that buys. An exact `
    + `attribution tells you how <span class="bold">this model</span> arrived at `
    + `<span class="bold">this number</span>. It does not tell you what causes what, which is `
    + `what the first four articles of this module were about and what a reader will assume `
    + `on seeing a bar chart of feature importances.`);

  set('closing-2',
    `The harder half is what happens when the columns are pixels. A hundred and fifty `
    + `thousand of them, no additive answer key, no closed form, and the explanation has to `
    + `be an image rather than a list. The <a href="../saliency/">next article</a> is about `
    + `that, and it can score the answers, because it borrows two models this atlas already `
    + `published along with something almost no saliency paper has: the exact pixel mask of `
    + `the object being explained.`);

  set('ref-note',
    `The dataset is the one this atlas keeps returning to: ${n0(M.rows)} bottles from three `
    + `cultivars, loaded from scikit-learn and asserted against its shape before anything `
    + `runs. The ${word(M.p)} columns used are the ones with the largest standardised `
    + `coefficients in a single linear fit, and they are named in the widget above. The `
    + `background for the value function is ${n0(M.background)} training rows, and the exact `
    + `enumeration covers all ${n0(coalitions)} coalitions for every instance reported.`);
}
