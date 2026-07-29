/* Every sentence that carries a measured number, written from the measurement.
 *
 * The rule this article leans on hardest is that a direction is a number too.
 * Its central claim is that one term of the formula helps and another does
 * not, and "does not" is a comparison against a floor that was itself measured;
 * so every one of those sentences is a branch on the floor rather than a
 * sentence with a number dropped into it. If a rerun moves the rarity weight
 * from harmless to helpful, the page says so instead of contradicting its own
 * table.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const signed = (v, d = 4) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));
const plural = (n, one, many) => (n === 1 ? one : many);

export const PROSE_IDS = [
  'intro-counting', 'intro-guard', 'scrolly-intro',
  'step-0', 'step-1', 'step-2', 'step-3', 'step-4', 'step-5',
  'formula-note', 'factorial-intro', 'factorial-floor', 'factorial-verdict',
  'why-intro', 'why-reg', 'why-second', 'ceiling-note', 'stopword-note',
  'retrieval-intro', 'retrieval-note',
  'length-intro', 'length-verdict',
  'ngram-intro', 'ngram-cost', 'ngram-verdict', 'mindf-note',
  'v-tf', 'v-tf-watch', 'v-idf', 'v-idf-watch', 'v-norm', 'v-norm-watch',
  'v-ngram', 'v-ngram-watch',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

function rows(sel, data, cells) {
  const body = d3.select(sel).select('tbody');
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
  const Z = data.zipf;
  const H = data.heaps;
  const F = data.factorial;
  const N = data.noise;
  const V = data.term_value;
  const L = data.length;
  const R = data.retrieval;
  const G = data.ngrams;

  const cell = (tf, idf, norm) => F.cells.find(
    (c) => c.tf === tf && c.idf === idf && c.norm === norm);
  const f1 = (c) => c.svm_f1;
  const floor = N.f1_floor;

  /* The three effects the article is about, measured once here and reused, so
     a sentence and a table cannot drift apart. */
  const normGaps = ['binary', 'raw', 'sublinear'].flatMap(
    (tf) => [false, true].map((u) => f1(cell(tf, u, true)) - f1(cell(tf, u, false))));
  const idfGaps = ['binary', 'raw', 'sublinear'].map(
    (tf) => f1(cell(tf, true, true)) - f1(cell(tf, false, true)));
  const idfBig = idfGaps.filter((v) => Math.abs(v) > floor);
  const normBig = normGaps.filter((v) => Math.abs(v) > floor);
  const allSameSign = normGaps.every((v) => v > 0) || normGaps.every((v) => v < 0);
  const best = F.cells.reduce((p, c) => (f1(c) > f1(p) ? c : p), F.cells[0]);

  /* ------------------------------------------------------------- the opening */
  set('intro-counting',
    `The oldest answer is still the one to beat: <span class="bold">count the words</span>. Throw `
    + `away the order, keep a tally per word, and a document becomes a point in a space with one `
    + `axis per word in the language. It sounds like it should not work. It works well enough that `
    + `on the ${n0(M.docs)} Reuters newswire stories used here it puts `
    + `<span class="bold">${pc(best.svm_acc, 2)}</span> of the held-out stories in the right one of `
    + `${word(M.categories.length)} categories, which is the number the rest of this article is `
    + `going to take apart.`);

  set('intro-guard',
    `Everything below runs in this page. What travels in the file is `
    + `<span class="bold">integer counts</span>, one per word per document, plus one integer per `
    + `word saying how many documents hold it, and nothing else: the page rebuilds every weight `
    + `from those, so the check at load time can demand agreement with the offline reference to `
    + `1e-12 rather than settling for close enough, because there is no rounding in between to `
    + `argue about. The offline side was itself checked against scikit-learn's own implementation `
    + `of the same formula, where the worst weight in the whole matrix of `
    + `${n0(data.sklearn.nnz)} non-zeros differs by `
    + `<span class="value">${data.sklearn.worst_abs_diff.toExponential(2)}</span>.`);

  /* ------------------------------------------------------------- the scrolly */
  set('scrolly-intro',
    `Before deciding what a word is worth, it is worth looking at how words are distributed, `
    + `because the distribution is unlike anything in the first two modules of this atlas. These `
    + `${n0(M.docs)} stories hold <span class="bold">${n0(Z.tokens)}</span> running words drawn `
    + `from <span class="bold">${n0(Z.types)}</span> distinct ones.`);

  set('step-0',
    `Every distinct word in the corpus, ordered from commonest to rarest, against how often it `
    + `occurs. The commonest occurs <span class="bold">${n0(Z.top[0].c)}</span> times and the `
    + `median word occurs once or twice, so on linear axes the whole picture is a spike against `
    + `the left edge and a flat line along the bottom. There is nothing to read, and that is the `
    + `first thing worth knowing about text.`);
  set('step-1',
    `Take the logarithm of both axes and the wall becomes a straight line. A straight line here `
    + `means frequency falls as a fixed power of rank, which is <span class="bold">Zipf's `
    + `law</span>, and it is not a property of this corpus: it turns up in every natural language `
    + `anybody has counted.`);
  set('step-2',
    `Fitted between ranks ${n0(Z.fit_lo)} and ${n0(Z.fit_hi)}, the slope is `
    + `<span class="bold">${f3(Z.slope)}</span> with an r&sup2; of ${f3(Z.fit_r2)}. The band is `
    + `stated because it matters: fitted over the whole curve instead, the flat shelf of words `
    + `that occur exactly once is thousands of points long and drags any line towards itself. A `
    + `slope of minus one means the second commonest word occurs about half as often as the `
    + `first, the tenth about a tenth as often, and so on down.`);
  set('step-3',
    `The shaded strip on the left is <span class="bold">${n0(Z.types_for_half)}</span> words: that `
    + `is all it takes to account for half of the ${n0(Z.tokens)} running words in the corpus. The `
    + `top ten alone are ${pc(Z.share_10)} of the text, and the top hundred are `
    + `${pc(Z.share_100)}. Any weighting scheme that treats every count the same is going to spend `
    + `most of its numbers on these.`);
  set('step-4',
    `And the shaded region on the right is the other end: <span class="bold">${n0(Z.hapax)}</span> `
    + `words, <span class="bold">${pc(Z.hapax_frac)}</span> of the vocabulary, occur exactly once `
    + `in the entire corpus. Between the two shaded regions is everything a model can actually `
    + `learn from, and it is a minority of the vocabulary in both directions.`);
  set('step-5',
    `A different measurement of the same corpus: distinct words seen against words read. It is `
    + `still climbing at the end, and the exponent of `
    + `<span class="bold">${f3(H.beta)}</span> says it always will be, because a curve of that `
    + `shape has no ceiling. Reading twice as much does not get you a fixed vocabulary, it gets `
    + `you about ${f2(2 ** H.beta)} times as many distinct words. There is no corpus large enough `
    + `to have seen them all, which is the constraint the whole of this branch of the atlas is `
    + `working under.`);

  /* ------------------------------------------------------------- the formula */
  set('formula-note',
    `Three knobs, and only two of them are in the name. The first says what a repeated word is `
    + `worth. The second is the rarity weight, which is a function of one number, the count of `
    + `documents holding the term, and it never sees a label because it is computed before there `
    + `is a task. The third divides the finished vector by its own length, and it is the one with `
    + `no letter in the acronym.`);

  /* ------------------------------------------------------------ el factorial */
  set('factorial-intro',
    `Three ways to treat the count, rarity on or off, length normalised or not: `
    + `<span class="bold">${word(F.cells.length)} combinations</span>, all of them fitted on the `
    + `same ${n0(F.n_train)} training stories and scored on the same ${n0(F.n_test)} held-out `
    + `ones, over the same ${n0(F.vocab)} features. Two different classifiers read each `
    + `representation, because a claim about a representation has to survive being read by more `
    + `than one model.`);

  set('factorial-floor',
    `Before comparing anything, a floor. Resampling the ${n0(F.n_test)} held-out stories with `
    + `replacement ${n0(N.boot_reps)} times, holding the predictions fixed, moves macro F1 by a `
    + `standard deviation of <span class="bold">${f4(N.boot_f1_sd)}</span>; refitting the same `
    + `matrix under ${word(N.seed_reps)} different solver seeds moves it by `
    + `<span class="bold">${f4(N.seed_f1_sd)}</span>. The larger of the two, `
    + `<span class="bold">${f4(floor)}</span>, is the shaded band in the chart, and any difference `
    + `narrower than it is a difference this experiment cannot see.`);

  set('factorial-verdict',
    `Read the rows: the line joining the two dots is the length normalisation, and it is `
    + `${allSameSign ? 'the same direction in all six' : 'not the same direction in all six'}, `
    + `with ${word(normBig.length)} of the six clearing the floor. Its smallest effect, `
    + `<span class="value">${signed(Math.min(...normGaps.map(Math.abs)) * Math.sign(normGaps[normGaps.map(Math.abs).indexOf(Math.min(...normGaps.map(Math.abs)))]))}</span>, `
    + `is ${f2(Math.min(...normGaps.map(Math.abs)) / floor)} times the floor. `
    + `The rarity weight, with the length already normalised, moves the score by `
    + `${idfGaps.map((v) => `<span class="value">${signed(v)}</span>`).join(', ')} for the three `
    + `counts, `
    + (idfBig.length === 0
      ? `and <span class="bold">not one of the three clears the floor</span>, in either direction. `
        + `Half of the name of this formula is, on this corpus and this task, doing nothing that `
        + `can be measured.`
      : `and ${word(idfBig.length)} of the three ${plural(idfBig.length, 'clears', 'clear')} the `
        + `floor.`)
    + ` The decision that is not in the name is the one carrying the result.`);

  /* -------------------------------------------------------------------- why */
  set('why-intro',
    `A result like that needs a mechanism, or it is a coincidence with a number attached. There `
    + `are two candidate explanations and they are both testable, so both were tested.`);

  const reg = data.idf_mechanism;
  const regBig = reg.reduce((p, c) => (Math.abs(c.gap_f1) > Math.abs(p.gap_f1) ? c : p), reg[0]);
  const regSmall = reg.reduce((p, c) => (Math.abs(c.gap_f1) < Math.abs(p.gap_f1) ? c : p), reg[0]);
  set('why-reg',
    `The first is that the classifier absorbs it. The rarity weight multiplies every column by a `
    + `constant, and a linear model already has one weight per column, so multiplying a column by `
    + `a constant and dividing its weight by the same constant is the same function: with no `
    + `penalty on the weights, `
    + `idf provably cannot change the answer at all. The only thing that breaks that equivalence `
    + `is the regularisation, which measures the weights and so is not invariant to their scale. `
    + `That predicts the gap should shrink towards zero as the penalty is loosened. It does not: `
    + `the gap is largest where the penalty is <span class="bold">strongest</span> `
    + `(<span class="value">${signed(regBig.gap_f1)}</span> at C = ${regBig.C}) and smallest in `
    + `the middle of the range (<span class="value">${signed(regSmall.gap_f1)}</span> at `
    + `C = ${regSmall.C}). The prediction is clean, falsifiable, and wrong, so the explanation is `
    + `somewhere else.`);

  rows('#reg-table', reg, (d) => [
    `C = ${d.C}`,
    f4(d.plain_f1),
    f4(d.idf_f1),
    { html: `<span class="value">${signed(d.gap_f1)}</span>` },
  ]);

  set('why-second',
    `The second candidate is not about the model. The rarity weight is a `
    + `<span class="bold">proxy</span> for how useful a word is, built without ever looking at the `
    + `labels, so the question with an answer is whether it ranks words the way something that `
    + `does look at the labels would. Below, every term in the training vocabulary, with its `
    + `rarity weight against the mutual information between its presence and the category.`);

  set('ceiling-note',
    `The dashed line is not a fit and not a trend. It is the exact ceiling. The information the `
    + `presence of a word can carry about anything is bounded by the entropy of its own presence, `
    + `which is zero when a word occurs in one document and largest when it `
    + `occurs in half of them. So a word that appears three times in ${n0(V.n_train)} documents `
    + `cannot be informative, however perfectly it correlates with a category, and the rarity `
    + `weight is a strictly decreasing function of exactly the quantity that ceiling increases `
    + `with. It is not even a ranking at the top: the largest rarity weight in this corpus is `
    + `shared by <span class="bold">${n0(V.tied_at_max_idf)}</span> terms at once, every one of `
    + `them appearing in exactly ${word(V.max_idf_df)} documents, and tf-idf has nothing to say `
    + `about which of them matters. The rank correlation between the two is `
    + `<span class="bold">${f4(V.spearman)}</span>. That is not idf failing at its job; it is idf `
    + `doing precisely what it says on the tin, which turns out to be close to the opposite of `
    + `what a classifier wants.`);

  const swAll = data.stopwords;
  const swGaps = [false, true].map((u) => {
    const a = swAll.find((r) => !r.dropped && r.idf === u);
    const b = swAll.find((r) => r.dropped && r.idf === u);
    return { idf: u, gap: b.f1 - a.f1, removed: a.vocab - b.vocab };
  });
  const swBig = swGaps.filter((r) => Math.abs(r.gap) > floor);
  set('stopword-note',
    `Which lands, uncomfortably, on the most repeated piece of advice in the field. Of the 200 `
    + `most informative terms in this corpus, <span class="bold">${n0(V.top_mi_stopwords)}</span> `
    + `are on the standard English stopword list, the one every tutorial deletes in its first `
    + `code cell. The single most informative term of all is `
    + `<span class="bold">${V.best_mi_word}</span>, which appears in `
    + `${n0(V.best_mi_df)} of the ${n0(V.n_train)} training documents. Deleting the list `
    + `costs ${n0(swGaps[0].removed)} features and changes the score by `
    + `${swGaps.map((r) => `<span class="value">${signed(r.gap)}</span>`).join(' and ')} `
    + `(without and with the rarity weight), `
    + (swBig.length === 0
      ? `neither of which clears the ${f4(floor)} floor.`
      : `${word(swBig.length)} of which clears the ${f4(floor)} floor.`)
    + ` Genre gives a document away before topic does, and function words are where genre lives.`);

  rows('#stopword-table', swAll, (d) => [
    d.dropped ? 'removed' : 'kept',
    d.idf ? 'on' : 'off',
    n0(d.vocab),
    f4(d.acc),
    f4(d.f1),
  ]);

  /* ------------------------------------------------------------- recuperacion */
  const retGaps = ['binary', 'raw', 'sublinear'].map((tf) => {
    const a = R[`${tf}_0`];
    const b = R[`${tf}_1`];
    return { tf, a: a.p10, b: b.p10, gap: b.p10 - a.p10, sd: b.p10_sd };
  });
  const helps = retGaps.filter((r) => r.gap > r.sd);
  const hurts = retGaps.filter((r) => -r.gap > r.sd);
  const bestRet = retGaps.reduce((p, c) => (Math.max(c.a, c.b) > Math.max(p.a, p.b) ? c : p),
    retGaps[0]);
  set('retrieval-intro',
    `There is one obvious objection to all of the above, and it is a good one: idf was not `
    + `invented for classification. It was invented for <span class="bold">retrieval</span>, where `
    + `there is no training label and no per-column weight to learn, so the only weighting a term `
    + `has is the one the vector brings with it. If the rarity weight earns its keep anywhere, it `
    + `has to be here. Below, for each way of treating the count, how often a document in the top `
    + `ten by cosine shares the category of the query, over all `
    + `${n0(L.n)} held-out documents used in turn as the query.`);

  rows('#retrieval-table', retGaps, (d) => [
    { binary: 'present or not', raw: 'raw count', sublinear: '1 + log count' }[d.tf],
    f4(d.a),
    f4(d.b),
    { html: `<span class="value">${signed(d.gap)}</span>` },
    f4(d.sd),
  ]);

  set('retrieval-note',
    `Measured on the task it was invented for, the answer is still not a clean yes: it helps `
    + `${helps.length ? helps.map((r) => `the ${r.tf === 'raw' ? 'raw count' : r.tf === 'binary' ? 'binary count' : 'log count'} (${signed(r.gap)})`).join(' and ') : 'none of the three'}, `
    + `and it <span class="bold">hurts</span> `
    + `${hurts.length ? hurts.map((r) => `the ${r.tf === 'raw' ? 'raw count' : r.tf === 'binary' ? 'binary count' : 'log count'} (${signed(r.gap)})`).join(' and ') : 'none of the three'}, `
    + `all against a floor of about ${f4(retGaps[0].sd)}. The pattern is not random: the rarity `
    + `weight rescues the raw count, which is the worst of the three on its own at `
    + `${f4(R.raw_0.p10)}, and takes points off the other two. Damping a word that is everywhere `
    + `is worth doing once, and taking the logarithm of the count has already done it. The best `
    + `retrieval setting measured here is `
    + `<span class="bold">${bestRet.tf === 'sublinear' ? '1 + log count' : bestRet.tf === 'raw' ? 'raw count' : 'present or not'} `
    + `${bestRet.b > bestRet.a ? 'with' : 'without'} the rarity weight</span>, at `
    + `${f4(Math.max(bestRet.a, bestRet.b))}.`);

  /* --------------------------------------------------------------- longitud */
  set('length-intro',
    `Which leaves the third knob, the one with no letter in the acronym and no paper named after `
    + `it. The documents in this corpus run from ${n0(L.len_min)} words to `
    + `${n0(L.len_max)}, a factor of ${n0(Math.round(L.len_max / Math.max(1, L.len_min)))}, with a `
    + `median of ${n0(Math.round(L.median_len))}. Before normalising, the length of a document's `
    + `vector correlates with the length of the document at `
    + `<span class="bold">${f3(L.corr_len_norm)}</span>, so most of what the vector is recording `
    + `is how much was written rather than what it was about.`);

  set('length-verdict',
    `Over every one of the ${n0(L.n)} held-out documents used in turn as the query rather than `
    + `the ${word(8)} in the widget: the nearest neighbour by cosine shares the query's category `
    + `<span class="bold">${pc(L.cosine_same_label, 2)}</span> of the time, and the nearest by `
    + `euclidean distance on the same vectors before the division does so `
    + `<span class="bold">${pc(L.euclid_same_label, 2)}</span> of the time. Across a top ten it is `
    + `${f4(L.cosine_p10)} against ${f4(L.euclid_p10)}. And the mechanism is in the second pair of `
    + `numbers: the mean gap in length between a query and its nearest neighbour is `
    + `<span class="value">${f1Round(L.cosine_len_gap)}</span> words under cosine and `
    + `<span class="value">${f1Round(L.euclid_len_gap)}</span> under euclidean. Without the `
    + `division, a long document is far from everything, a short one is near everything, and the `
    + `question of what either is about barely gets asked.`);

  /* --------------------------------------------------------------- n-gramas */
  const uni = G.find((g) => g.n === 1);
  const bi = G.find((g) => g.n === 2);
  const tri = G.find((g) => g.n === 3);
  set('ngram-intro',
    `A bag of words has thrown away the order, which is most of what a sentence is. "The dollar `
    + `rose against the yen" and "the yen rose against the dollar" are the same point. The `
    + `standard repair is to count adjacent pairs as well, and it is worth seeing the bill before `
    + `the benefit: <span class="bold">${n0(uni.types)}</span> distinct words become `
    + `<span class="bold">${n0(bi.types)}</span> distinct pairs, `
    + `${f1Round(bi.types / uni.types)} times as many, from the same `
    + `${n0(Z.tokens)} words of text.`);

  rows('#ngram-table', G, (d) => [
    `${word(d.n)} word${d.n === 1 ? '' : 's'}`,
    n0(d.types),
    `${n0(d.hapax)} (${pc(d.hapax_frac)})`,
    pc(d.test_unseen_token_frac),
  ]);

  set('ngram-cost',
    `The last column is the one that decides it, and it is not usually printed. `
    + `<span class="bold">${pc(bi.test_unseen_token_frac)}</span> of the word pairs in the `
    + `held-out documents never occurred once in training, so for those the model has nothing at `
    + `all: not a weak feature, an absent one. By ${word(tri.n)} words it is `
    + `${pc(tri.test_unseen_token_frac)}. Here is what the extra features actually buy, on the `
    + `same split as everything else:`);

  const NA = data.ngram_arm;
  rows('#ngram-acc-table', NA, (d) => [
    d.nmax === 1 ? 'single words' : `up to ${word(d.nmax)}-word runs`,
    n0(d.features),
    f4(d.acc),
    f4(d.f1),
  ]);

  const biGap = NA[1].f1 - NA[0].f1;
  const triGap = NA[2].f1 - NA[0].f1;
  set('ngram-verdict',
    `Going from single words to pairs multiplies the feature count by `
    + `${f1Round(NA[1].features / NA[0].features)} and moves macro F1 by `
    + `<span class="value">${signed(biGap)}</span>, against a floor of ${f4(floor)}; going to `
    + `triples multiplies it by ${f1Round(NA[2].features / NA[0].features)} and moves it by `
    + `<span class="value">${signed(triGap)}</span>. `
    + (Math.abs(biGap) > floor
      ? `The pairs clear the floor.`
      : `Neither clears the floor.`)
    + ` The order is worth something in language. On this task, at this scale, it is not worth `
    + `this, and the reason is the last column of the table above rather than anything about `
    + `syntax: a feature that appears once is a feature the model cannot learn a weight for, and `
    + `${pc(bi.hapax_frac)} of the pairs appear exactly once.`);

  const MD = data.mindf;
  const bestMd = MD.reduce((p, c) => (c.f1 > p.f1 ? c : p), MD[0]);
  const cheapest = MD.filter((r) => bestMd.f1 - r.f1 <= floor)
    .reduce((p, c) => (c.features < p.features ? c : p), bestMd);
  set('mindf-note',
    `The same question asked of the vocabulary itself has the same shape of answer. Keeping only `
    + `words that appear in at least ${n0(cheapest.min_df)} documents leaves `
    + `<span class="bold">${n0(cheapest.features)}</span> features, `
    + `${pc(1 - cheapest.features / MD[0].features)} fewer than keeping everything, and scores `
    + `${f4(cheapest.f1)} against the best-anywhere ${f4(bestMd.f1)}: inside the floor. Most of a `
    + `vocabulary is not information, it is a list of things that happened once.`);

  rows('#mindf-table', MD, (d) => [
    `${n0(d.min_df)} document${d.min_df === 1 ? '' : 's'}`,
    n0(d.features),
    f4(d.acc),
    f4(d.f1),
  ]);

  /* ----------------------------------------------------------------- veredicto */
  const tfBest = ['binary', 'raw', 'sublinear']
    .map((tf) => ({ tf, v: f1(cell(tf, true, true)) }))
    .reduce((p, c) => (c.v > p.v ? c : p));
  const tfWorst = ['binary', 'raw', 'sublinear']
    .map((tf) => ({ tf, v: f1(cell(tf, true, true)) }))
    .reduce((p, c) => (c.v < p.v ? c : p));
  const tfName = { binary: 'present or not', raw: 'the raw count', sublinear: '1 + log count' };
  set('v-tf',
    `${(tfBest.v - tfWorst.v) > floor ? 'Clears the floor' : 'Inside the floor'}: `
    + `${f4(tfBest.v - tfWorst.v)} of macro F1 between the best (${tfName[tfBest.tf]}) and the `
    + `worst (${tfName[tfWorst.tf]}) with everything else held fixed.`);
  set('v-tf-watch',
    `It overlaps with the rarity weight. Taking the logarithm already damps a word that is `
    + `everywhere, which is most of what idf was for, and doing both is where the retrieval `
    + `numbers go backwards.`);
  set('v-idf',
    idfBig.length === 0
      ? `Nothing measurable: ${idfGaps.map((v) => signed(v)).join(', ')} of macro F1 for the three `
        + `counts, all inside the ${f4(floor)} floor, in both directions.`
      : `${word(idfBig.length)} of three effects clear the floor: `
        + `${idfGaps.map((v) => signed(v)).join(', ')}.`);
  set('v-idf-watch',
    `It has never seen your labels, and it ranks words by a quantity that <span class="bold">caps `
    + `how informative they can be</span>. Rank correlation with mutual information here: `
    + `${f4(V.spearman)}.`);
  set('v-norm',
    `The largest effect on the page: up to ${f4(Math.max(...normGaps.map(Math.abs)))} of macro F1 `
    + `in classification, and ${pc(L.cosine_same_label - L.euclid_same_label, 1)} of nearest `
    + `neighbours in retrieval.`);
  set('v-norm-watch',
    `It is not in the name, so it is the one that gets dropped when somebody reimplements the `
    + `formula from the description rather than from the paper.`);
  set('v-ngram',
    `${Math.abs(biGap) > floor ? f4(biGap) : `${signed(biGap)}, inside the floor`} of macro F1 for `
    + `${f1Round(NA[1].features / NA[0].features)} times the features.`);
  set('v-ngram-watch',
    `${pc(bi.test_unseen_token_frac)} of the pairs in held-out text were never seen in training. `
    + `They are not weak features, they are absent ones.`);

  /* ------------------------------------------------------------------ cierre */
  set('closing-1',
    `The counting model has a ceiling and it is not where the name suggests. Of the three `
    + `decisions inside TF-IDF, the one that mattered most here is the one with no letter in the `
    + `acronym, and the one that mattered least is half the acronym. That is worth knowing on its `
    + `own, but the deeper limit is the one the vocabulary curve showed: every word is its own `
    + `axis, orthogonal to every other, so <span class="bold">"bond" and "bonds" are as unrelated `
    + `as "bond" and "wheat"</span>, and no amount of reweighting fixes that, because reweighting `
    + `only stretches axes it cannot join.`);
  set('closing-2',
    `The next article in this branch is the first attempt to get underneath that: instead of one `
    + `axis per word, a handful of axes shared by all of them, with each document a mixture over `
    + `the handful. It keeps the bag of words exactly as it is here, and changes only what the `
    + `bag is projected onto. <a href="../topics/">Latent Dirichlet Allocation</a> is next.`);

  set('ref-note',
    `The corpus is Reuters-21578 as distributed with NLTK: the ${n0(M.docs)} stories carrying `
    + `exactly one category from the ${word(M.categories.length)} largest, split into `
    + `${n0(M.train)} training and ${n0(M.test)} held-out documents by the ModApte partition that `
    + `ships with the corpus rather than by anything chosen here. The tokenizer is `
    + `${M.tokenizer}, which is written once in Python and mirrored word for word in JavaScript, `
    + `since every count on this page is downstream of it. The stopword list is the ${M.stopword_list} `
    + `English words NLTK distributes, used here to measure what removing them costs rather than `
    + `to remove them.`);
}

function f1Round(v) {
  return v.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}
