/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const pct = (v) => `${(v * 100).toFixed(2)}%`;

export const PROSE_IDS = [
  'opening-note', 'roc-intro', 'roc-note', 'th-intro', 'th-note',
  'enum-intro', 'enum-note', 'bleu-intro', 'bleu-note', 'verdict-intro',
  'verdict-auc', 'verdict-auc-warn', 'verdict-ap', 'verdict-ap-warn',
  'verdict-mcc', 'verdict-mcc-warn', 'verdict-bleu', 'verdict-bleu-warn',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.tasks;
  const MO = data.monotone;
  const PR = data.prevalence;
  const TH = data.threshold;
  const E = data.enumerate;
  const B = data.bleu;
  const D = data.disagree;

  const ship = T.ship;
  const earn = T.earn;
  const worstAuc = Math.max(...Object.values(T).flatMap(
    (t) => Object.values(t.models).map((m) => m.auc_gap)));
  const rare = PR.rows[PR.rows.length - 1];
  const common = PR.rows[0];
  const accRow = TH.rows.find((r) => r.metric === 'accuracy');
  const mccRow = TH.rows.find((r) => r.metric === 'mcc');
  const thSpread = Math.max(...TH.rows.map((r) => r.threshold))
    - Math.min(...TH.rows.map((r) => r.threshold));
  const shuf = B.rows.find((r) => r.kind === 'shuffled');
  const same = B.rows.find((r) => r.kind === 'identical');
  const nn = B.rows.find((r) => r.kind === 'nearest neighbour');
  const segSpread = Math.max(...B.segmentation.map((s) => s.bleu))
    - Math.min(...B.segmentation.map((s) => s.bleu));
  const flips = D.reduce((s, r) => s + r.flips, 0);
  const comps = D.reduce((s, r) => s + r.comparisons, 0);

  set('opening-note',
    `The classifiers are on the corpus this atlas keeps coming back to: `
    + `${n0(M.docs)} Reuters newswires with one category each, turned into ${M.dims} dense `
    + `columns, with the ModApte split. Two of the eight categories are used as binary tasks `
    + `on purpose, because they are very different shapes: predicting `
    + `<span class="mono">earn</span> is a ${pct(earn.prevalence)} prevalence problem and `
    + `predicting <span class="mono">ship</span> is a ${pct(ship.prevalence)} one, `
    + `${ship.positives} positives out of ${n0(ship.n_test)} test rows. Every metric on this `
    + `page behaves differently on those two, which is the point.`);

  set('roc-intro',
    `Start with the one that is usually treated as the default. The receiver operating `
    + `curve sweeps a threshold from one end of the scores to the other and plots what `
    + `fraction of positives and of negatives ends up flagged; the area under it summarises `
    + `the sweep.`);

  set('roc-note',
    `They agree to ${worstAuc.toExponential(1)} at worst across every model on both tasks, `
    + `which is float arithmetic and not a chosen tolerance. That identity is not a curiosity: `
    + `<span class="bold">it is the reason the area cannot see anything except the ordering</span>. `
    + `Squash all the scores towards a half and no pair changes places, so the area cannot `
    + `move, and the sweep below shows it moving by at most `
    + `${MO.worst_shift.toExponential(1)} while the Brier score runs across a range of `
    + `${f4(Math.max(...MO.rows.map((r) => r.brier)) - Math.min(...MO.rows.map((r) => r.brier)))}. `
    + `That is exactly what the <a href="../calibration/">calibration article</a> measured `
    + `from the other end, and it is why an area of 0.98 tells you nothing about whether the `
    + `probabilities mean anything. The same argument covers the base rate: dropping the `
    + `prevalence from ${pct(common.prevalence)} to ${pct(rare.prevalence)} leaves the area `
    + `at ${f4(rare.auc)} and takes average precision from ${f4(common.ap)} to `
    + `${f4(rare.ap)}, because the ordering did not change and the share of flagged rows `
    + `that are right did.`);

  set('th-intro',
    `A ranking metric does not tell you what to do. Somewhere between the score and the `
    + `decision there is a threshold, and it is not a property of the model: it is chosen by `
    + `whichever summary you decided to maximise.`);

  set('th-note',
    `On the ${pct(TH.prevalence)} task the four common choices land ${thSpread.toFixed(3)} `
    + `apart and flag between `
    + `${Math.min(...TH.rows.map((r) => r.predicted_positive))} and `
    + `${Math.max(...TH.rows.map((r) => r.predicted_positive))} rows. Maximising accuracy `
    + `reaches ${f4(accRow.value)}, and the honest way to read that figure is next to the `
    + `dumbest possible model: flagging nothing at all scores `
    + `${f4(TH.always_negative.accuracy)} on accuracy, `
    + `${f4(TH.always_negative.f1)} on F1 and ${f4(TH.always_negative.mcc)} on the `
    + `correlation coefficient. The <a href="../imbalanced-learning/">imbalanced learning `
    + `article</a> called that the accuracy trap and measured what it costs; here it is `
    + `visible as a threshold that has moved to keep a number high.`);

  set('enum-intro',
    `Which leaves the question of which summary of a two by two table to use, and that one `
    + `can be settled by enumeration rather than by preference. There are `
    + `${n0(E.matrices)} confusion matrices with ${E.n} cases in them, and all of them are in `
    + `the file behind this page.`);

  set('enum-note',
    `Across ${n0(E.pairs)} random pairs of them, F1 and the correlation coefficient disagree `
    + `about which matrix is better <span class="bold">${pct(E.flip_f1_mcc)}</span> of the `
    + `time, accuracy and the correlation coefficient ${pct(E.flip_acc_mcc)}, accuracy and `
    + `F1 ${pct(E.flip_acc_f1)}. Two structural facts explain most of that. F1 never reads `
    + `the true negatives, so two matrices that differ only in how many rows the model `
    + `correctly left alone get the same F1 and different correlation coefficients. And F1 `
    + `is not symmetric: swap which class is called positive and it moves by up to `
    + `${E.f1_asym_max.toFixed(3)}, while the correlation coefficient moves by exactly `
    + `${E.mcc_asym.toFixed(3)}, because it is a correlation between two binary variables `
    + `and correlation does not care which one is on which axis. That is not a small `
    + `advantage: it means the number cannot be improved by relabelling.`);

  set('bleu-intro',
    `The third metric summarises something that is not a table at all. BLEU counts how many `
    + `of the candidate's word sequences appear in the reference, clipped so that repeating a `
    + `word does not help, up to four words at a time, with a penalty for being short.`);

  set('bleu-note',
    `There is no translation system on this page, which is a decision and not a shortcut: a `
    + `system's output differs from a reference in a way nobody can write down, so a score on `
    + `it says only "worse by an unknown amount". Perturbing ${n0(B.sentences)} real `
    + `sentences in ways that <span class="bold">are</span> written down says what the `
    + `statistic responds to. Shuffling every word of a sentence, which keeps `
    + `${(shuf.word_overlap * 100).toFixed(0)}% of its words and destroys the sentence, `
    + `scores ${f4(shuf.corpus)} against ${f4(same.corpus)} for leaving it alone. Replacing `
    + `each word with its nearest neighbour among the vectors the `
    + `<a href="../embeddings/">embeddings article</a> published, which is close to a `
    + `paraphrase, scores ${f4(nn.corpus)}: a metric that counts exact word matches gives no `
    + `credit for a synonym, by construction. And the thing that gets left out of every `
    + `comparison table: changing the tokenisation moves the number by `
    + `${f4(segSpread)} on the same pairs of sentences.`);

  set('verdict-intro',
    `Four rows, and the middle column is what each one is a statement about rather than what `
    + `it is "for". Across the models measured on this page, two metrics disagreed about `
    + `which of two models was better in ${flips} of ${comps} comparisons.`);

  set('verdict-auc',
    `The probability that a positive outranks a negative, counted exactly and agreeing with `
    + `the area to ${worstAuc.toExponential(1)}.`);
  set('verdict-auc-warn',
    `Anything monotone in the scores. Calibration is invisible to it, and so is the base `
    + `rate: it moved ${f4(Math.abs(rare.auc - common.auc))} while the prevalence went from `
    + `${pct(common.prevalence)} to ${pct(rare.prevalence)}.`);
  set('verdict-ap',
    `How much of what you flag is right, averaged over every cut. It moved from `
    + `${f4(common.ap)} to ${f4(rare.ap)} over that same prevalence sweep.`);
  set('verdict-ap-warn',
    `Its scale is the prevalence: a coin scores ${f4(rare.ap_chance)} at `
    + `${pct(rare.prevalence)} and ${f4(common.ap_chance)} at ${pct(common.prevalence)}, so `
    + `the number is not comparable across datasets without saying the base rate.`);
  set('verdict-mcc',
    `Both summarise one two by two table, and they disagree about which of two tables is `
    + `better in ${pct(E.flip_f1_mcc)} of pairs.`);
  set('verdict-mcc-warn',
    `F1 cannot see the true negatives and changes by up to ${E.f1_asym_max.toFixed(3)} when `
    + `you swap which class is positive. Neither can see the threshold that made the table.`);
  set('verdict-bleu',
    `Overlap of word sequences, up to four long, with a length penalty. Identical text `
    + `scores ${f4(same.corpus)}.`);
  set('verdict-bleu-warn',
    `Word order beyond four, synonyms (${f4(nn.corpus)} for nearest neighbour `
    + `substitution), and the tokenisation, worth ${f4(segSpread)} on unchanged text.`);

  set('closing-1',
    `The pattern across the three is the same and it is not that the metrics are bad. Each `
    + `one is an exact statement about something: an ordering, a table, a set of overlapping `
    + `sequences. Each becomes wrong at the moment it is read as a statement about something `
    + `else, and the distance between the two is measurable in every case on this page. The `
    + `only general advice that survives is to report the thing the metric is about `
    + `alongside it: the base rate next to an average precision, the calibration next to an `
    + `area, the tokenisation next to a BLEU.`);

  set('closing-2',
    `That closes the evaluation branch, and with it every question this module has asked `
    + `about a model that already exists. What is left is the part that starts once the `
    + `model is finished and something has to be done with its output at scale. `
    + `<a href="../ann-search/">The next article</a> takes the vectors that the embedding `
    + `articles produced and asks how to find the nearest ones among millions, which turns `
    + `out to have its own version of the trade this page has been making all along: an `
    + `exact answer, or a cheap one.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/utils/generate_metrics_data.py</span>, seed ${M.seed}. The `
    + `classifiers run on a singular value decomposition of the same weighted counts the `
    + `<a href="../tf-idf/">TF-IDF article</a> measured, ${M.dims} dimensions over `
    + `${n0(M.vocab)} terms and ${n0(M.docs)} single label newswires. The area is computed `
    + `twice, once by trapezoids over the curve and once by ranking every pair, and the two `
    + `have to agree. The enumeration is every confusion matrix with ${M.enum_n} cases, all `
    + `${n0(E.matrices)} of them, with the disagreement rates taken over ${n0(E.pairs)} `
    + `random pairs and the pair count printed next to them. Calibration error uses `
    + `${M.bins} equal width bins. BLEU is written in the generator with clipping, a brevity `
    + `penalty and up to four grams, over ${n0(B.sentences)} sentences of the Brown corpus `
    + `between ${8} and 30 words; the nearest neighbour substitution uses the skip gram `
    + `vectors of the <a href="../embeddings/">embeddings article</a>, ${B.embed_meta.shipped} `
    + `words at ${B.embed_meta.dim} dimensions, read from its published file.`);
}
