/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const f3 = (v) => Number(v).toFixed(3);
const signed = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(4)}`;

export const PROSE_IDS = [
  'opening-note', 'truth-intro', 'truth-note', 'noise-intro', 'noise-note',
  'time-intro', 'time-note', 'group-intro', 'empty-intro', 'verdict-intro',
  'verdict-plain', 'verdict-plain-fix', 'verdict-leak', 'verdict-leak-fix',
  'verdict-time', 'verdict-time-fix', 'verdict-group', 'verdict-group-fix',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.truth;
  const N = data.noise;
  const TM = data.time;
  const G = data.groups;
  const E = data.empty;

  const two = T.rows[0];
  const ten = T.rows.find((r) => r.k === 10) || T.rows[T.rows.length - 2];
  const loo = T.rows[T.rows.length - 1];
  const worstRatio = T.rows.reduce((a, b) => (b.ratio > a.ratio ? b : a));
  const rand = TM.rows[0];
  const shuf = TM.rows[1];
  const fwd = TM.rows[2];
  const emb = TM.embargo;
  const noGap = emb[0];
  const bigGap = emb[emb.length - 1];
  const groupGap = G.by_row - G.by_group;
  const groupFloor = G.by_row_sd + G.by_group_sd;
  const groupReal = Math.abs(groupGap) > groupFloor;
  const rare = E[0];
  const rep = T.repeats;
  const repFirst = rep[0];
  const repLast = rep[rep.length - 1];

  set('opening-note',
    `The generator is eight standard normal columns and a logistic model with coefficients `
    + `drawn once and then written down, so a fresh population of ${n0(T.pop)} rows gives the `
    + `risk of any fitted model to four decimals without any estimation in it. Two truths `
    + `come out of that and they are not the same: the risk of the model fitted on `
    + `<span class="bold">these</span> ${T.n} rows, and the expected risk of fitting on `
    + `${T.n} rows at all, which is ${f4(T.expected_risk)}. Cross validation estimates the `
    + `second, and the second is not what anyone wants; they want the first. The gap between `
    + `those two is most of what makes this subject confusing.`);

  set('truth-intro',
    `Every fold trains on fewer rows than the model you are going to ship, so cross `
    + `validation is measuring a slightly worse model on purpose. With ${T.sets} independent `
    + `datasets that bias can be read straight off: ${T.n} rows in ${two.k} folds trains each `
    + `fit on ${two.n_train} of them and reports ${f4(two.mean)}, which is `
    + `${signed(two.bias)} from the truth; ${ten.k} folds trains on ${ten.n_train} and `
    + `reports ${f4(ten.mean)}, ${signed(ten.bias)}; leaving one out trains on `
    + `${loo.n_train} and reports ${f4(loo.mean)}, ${signed(loo.bias)}.`);

  set('truth-note',
    `The second view carries the number this section is for. The error bar printed next to a `
    + `cross validated score is almost always the spread of the fold scores divided by the `
    + `root of k, which assumes the folds are independent samples. They are not: any two `
    + `folds share ${Math.round((1 - 2 / ten.k) * 100)}% of their training rows at k = `
    + `${ten.k}. Measuring how much the estimate really moves, by redrawing the dataset, `
    + `against how much it claims to move: at k = ${ten.k} it is ${f4(ten.sd_true)} against `
    + `${f4(ten.sd_claimed)}, <span class="bold">understated by a factor of `
    + `${ten.ratio.toFixed(2)}</span>, and the worst of the sweep is `
    + `${worstRatio.ratio.toFixed(2)} at k = ${worstRatio.k}. There is no fix inside one `
    + `dataset, which is a theorem and not an oversight: no unbiased estimator of that `
    + `variance exists from a single split. Repeating the whole split helps with part of it `
    + `and only part: ${repFirst.repeats} repeat gives ${f4(repFirst.sd)} and `
    + `${repLast.repeats} give ${f4(repLast.sd)}, but the floor is ${f4(T.true_sd)}, which is `
    + `how much the dataset itself moves and no amount of resplitting can touch.`);

  set('noise-intro',
    `Everything above assumes the split is the only thing standing between the model and the `
    + `answer. It usually is not, because a pipeline has steps before the model, and the `
    + `question of where those steps stand relative to the split is the single most `
    + `expensive detail in this subject.`);

  set('noise-note',
    `The demonstration uses data with nothing in it at all: ${N.n} rows, `
    + `${n0(N.p)} columns of independent noise, labels from a coin. Keep the ${N.keep} `
    + `columns that separate the classes best, chosen while looking at every row, and cross `
    + `validate on those: <span class="bold">${f4(N.outside)}</span>. Choose them again `
    + `inside each fold and it is ${f4(N.inside)}, which is the truth. The reason is that `
    + `with ${n0(N.p)} columns and ${N.n} rows, some columns separate the classes by luck, `
    + `and choosing them while looking at the test rows makes that luck available to the `
    + `model in every fold. It is worth noticing how respectable ${f4(N.outside)} looks. `
    + `Nobody would report the ${f4(N.resub)} you get from scoring on the training rows; this `
    + `is the same mistake wearing cross validation as a disguise.`);

  set('time-intro',
    `The third failure is an ordering that is in the data and not in the table. Rows of a `
    + `series are not exchangeable, so a fold drawn at random asks the model to fill a gap `
    + `with the answer visible on both sides of it, which is not the question anyone will `
    + `ask it in production.`);

  set('time-note',
    `Both numbers below are honest arithmetic on the same rows, which is what makes the pair `
    + `useful: ${f3(rand.mae)} parts per million splitting at random and ${f3(fwd.mae)} `
    + `training only on the past, a factor of ${(fwd.mae / rand.mae).toFixed(1)}. The `
    + `control isolates the cause rather than assuming it: shuffle the same rows so the `
    + `order is gone and split at random again, and the answer is ${f3(shuf.mae)}, within `
    + `${f3(Math.abs(shuf.mae - rand.mae))} of the optimistic one. And then the part that `
    + `survives doing it properly. Put a ${TM.ma} month rolling mean in the features and rows `
    + `on either side of a cut share observations even though no future row is used to `
    + `predict a past one: forward chaining reports ${f3(noGap.mae)} with the cut flush and `
    + `${f3(bigGap.mae)} once ${bigGap.embargo} months are dropped either side of it.`);

  set('group-intro',
    `The fourth is the one nobody can see in their own data, because the groups are not a `
    + `column. Reuters contains the same story filed more than once, and this atlas has run `
    + `on that corpus twice already without asking how many.`);

  set('empty-intro',
    `The last one is smaller and sharper, and it is the reason stratification is a default `
    + `rather than an option. With few enough positives, some fold gets none of them, and `
    + `the average over folds is then an average over a different number of folds than it `
    + `claims.`);

  set('verdict-intro',
    `Four situations, and what they have in common is that the naive split does not fail `
    + `loudly. In every one of them it returns a number, the number looks reasonable, and it `
    + `is answering a question nobody asked.`);

  set('verdict-plain',
    `${f4(ten.mean)} at k = ${ten.k}, which is ${signed(ten.bias)} from the risk of fitting `
    + `on all ${T.n} rows.`);
  set('verdict-plain-fix',
    `Nothing, but quote the right error bar: the printed one is understated by `
    + `${ten.ratio.toFixed(2)} times here, and repeating the split has a floor at `
    + `${f4(T.true_sd)}.`);
  set('verdict-leak', `${f4(N.outside)} on data containing nothing.`);
  set('verdict-leak-fix',
    `Every step that looks at the labels goes inside the fold. Done that way the same data `
    + `gives ${f4(N.inside)}, which is chance.`);
  set('verdict-time',
    `${f3(rand.mae)} parts per million against ${f3(fwd.mae)} when the model is only allowed `
    + `the past.`);
  set('verdict-time-fix',
    `Forward chaining, plus a gap if any feature spans several rows: ${bigGap.embargo} months `
    + `of gap moved this one from ${f3(noGap.mae)} to ${f3(bigGap.mae)}.`);
  set('verdict-group',
    groupReal
      ? `${f4(G.by_row)} by row against ${f4(G.by_group)} by group, `
        + `${f4(Math.abs(groupGap))} apart.`
      : `${f4(G.by_row)} by row against ${f4(G.by_group)} by group, which the seed spread of `
        + `${f4(groupFloor)} cannot separate.`);
  set('verdict-group-fix',
    `Find the groups before trusting either. Here ${n0(G.pairs)} near duplicate pairs leave `
    + `${n0(G.in_groups)} of ${n0(G.docs)} documents inside a group with somebody else.`);

  set('closing-1',
    `The thread through all five sections is one sentence: a held out score answers exactly `
    + `the question its split encodes, and the split is usually chosen without anyone `
    + `writing down what question that is. Random folds encode "another row from the same `
    + `table, drawn the same way, with everything I did to the columns already done". If `
    + `that is what you will face, the number is right. If the next row will arrive later in `
    + `time, or from a story you have already half seen, or before the column selection has `
    + `been done, then the number is answering a friendlier question than the one you have.`);

  set('closing-2',
    `And there is a decision this page has kept out of the way. Every number here was an `
    + `accuracy or an absolute error, chosen because those are the two summaries that need `
    + `least explaining. They are also two of the worst summaries in common use. `
    + `<a href="../metrics/">The next article</a> is about what happens when the same model, `
    + `on the same data, with the same split, is ranked by one summary rather than another, `
    + `and it starts by enumerating every confusion matrix of a hundred cases.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/utils/generate_crossval_data.py</span>, seed ${M.seed}. The `
    + `simulated section is ${n0(M.truth_sets)} independent datasets of ${M.truth_n} rows and `
    + `${M.truth_d} columns from a logistic process, with the risk of each fitted model `
    + `computed on ${n0(M.pop)} fresh rows rather than estimated. The leakage section is `
    + `${N.seeds} datasets of ${N.n} rows and ${n0(N.p)} noise columns with coin flip labels. `
    + `The time section runs on the ${TM.n} usable months of the Mauna Loa carbon dioxide `
    + `series that the <a href="../arima/">forecasting article</a> published, asserted by its `
    + `digest <span class="mono">${M.co2_digest}</span> before anything else runs, with `
    + `${M.lags} lags and a ridge fit. The group section finds near duplicates among `
    + `${n0(G.docs)} Reuters newswires by cosine similarity in ${G.dims} dimensions and joins `
    + `them transitively. The empty fold section compares ${n0(rare.trials)} simulated splits `
    + `against a ratio of binomial coefficients with a second order inclusion and exclusion `
    + `term. No wall clocks.`);
}
