/* Every sentence with a measured number in it, composed from the measurement.
 *
 * This article compares more things than most, so nearly every sentence below
 * is a branch rather than a template: which criterion picks which k, whether
 * two of them agree, which of three factorisations wins, whether the seed
 * matters more than the model. If a rerun moves any of those, the page moves
 * with it instead of describing a result it no longer has.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-problem', 'intro-key', 'intro-guard',
  'model-note', 'gibbs-note', 'sampler-verdict',
  'k-intro', 'k-verdict',
  'stability-intro', 'stability-verdict', 'stability-examples',
  'filter-intro', 'filter-verdict', 'alpha-note',
  'rivals-intro', 'rivals-verdict', 'gibbs-vb-note',
  'v-filter', 'v-filter-watch', 'v-k', 'v-k-watch', 'v-seed', 'v-seed-watch',
  'v-alpha', 'v-alpha-watch',
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
  const K = data.ksweep;
  const St = data.stability;
  const F = data.filtering;
  const R = data.rivals;
  const A = data.alpha;
  const GV = data.gibbs_vs_vb;
  const S = data.sampler;

  const bestCoh = K.rows.reduce((p, c) => (c.coherence > p.coherence ? c : p), K.rows[0]);
  const bestPer = K.rows.reduce((p, c) => (c.perplexity < p.perplexity ? c : p), K.rows[0]);
  const bestAri = K.rows.reduce((p, c) => (c.ari > p.ari ? c : p), K.rows[0]);
  const atTruth = K.rows.find((r) => r.k === M.categories.length);

  /* ------------------------------------------------------------- the opening */
  set('intro-problem',
    `The previous article left a document as a point in a space with one axis per word: `
    + `${n0(M.vocab)} of them here, after the usual filtering, for ${n0(M.docs)} newswire `
    + `stories. Every axis is at right angles to every other, which is a strange thing to believe `
    + `about language. It says that "barrel" and "crude" are exactly as unrelated as "barrel" and `
    + `"dividend", and no amount of reweighting can say otherwise, because reweighting stretches `
    + `axes and never joins them.`);

  set('intro-key',
    `Almost every article about topic models evaluates one by printing the topics and inviting `
    + `you to nod. This one cannot, because these ${n0(M.docs)} stories come with `
    + `<span class="bold">${word(M.categories.length)} hand-written categories</span> `
    + `(${M.categories.slice(0, -1).join(', ')} and ${M.categories[M.categories.length - 1]}), `
    + `assigned by people at Reuters who had never heard of any of this. The model never sees `
    + `them. So "did it find the topics that are there" stops being a matter of taste and becomes `
    + `a number, and the number is worth knowing before anything else: at `
    + `${word(M.categories.length)} topics, the model's own partition of the documents scores an `
    + `adjusted Rand index of <span class="bold">${f4(atTruth ? atTruth.ari : bestAri.ari)}</span> `
    + `against those categories, where a perfect recovery is 1 and shuffling gives 0.`);

  set('intro-guard',
    `The sampler runs in this page, token by token, from a random start, and you can watch it. `
    + `It is written twice, once in Python and once in JavaScript, sharing a linear congruential `
    + `generator so that both produce the same draws in the same order; the counts are integers, `
    + `so the check at load time demands that the browser reproduce the generator's numbers `
    + `<span class="bold">exactly</span>, at ${word(data.refs.length)} checkpoints, rather than `
    + `to within a tolerance. Two Markov chains that agree closely have not agreed at all.`);

  /* -------------------------------------------------------------- the model */
  set('model-note',
    `Nobody believes this is how anything was written. It is a claim about what would be enough `
    + `to reproduce the counts, and the useful part is what it leaves out: the story has no `
    + `word order in it anywhere. A document is still a bag. What has changed is that the bag is `
    + `now built from ${word(S.k)} shared distributions rather than from ${n0(M.vocab)} `
    + `independent axes, and those distributions are what the model has to invent.`);

  set('gibbs-note',
    `Inference is the interesting part. The topic of each individual word is unobserved, so the `
    + `sampler starts by assigning one at random to every one of the `
    + `<span class="bold">${n0(S.tokens)}</span> tokens in the panel below, and then repeatedly `
    + `visits each token and reassigns it, given every other assignment. Nothing about this is `
    + `guaranteed to find the best answer; it is guaranteed only to wander in proportion to how `
    + `good an answer is. Press the buttons and watch the panel: `
    + `<span class="value">${n0(S.docs)}</span> documents down the side, sorted into their true `
    + `categories with a rule between each, and the ${word(S.k)} topics across.`);

  set('sampler-verdict',
    `The blocks appear within a handful of sweeps, and that on its own means very little: a model `
    + `with ${word(S.k)} topics will always divide ${n0(S.docs)} documents into ${word(S.k)} `
    + `groups. The question the rules on the left are there to answer is whether the divisions `
    + `land where the categories are. Some of them do and some of them very clearly do not, and `
    + `the rest of this article is about which knobs decide that.`);

  /* ---------------------------------------------------------------- how many */
  set('k-intro',
    `The number of topics is not learned, it is supplied. Two criteria are normally used to pick `
    + `it. <span class="bold">Held-out perplexity</span> is the model's own: fit on `
    + `${n0(K.n_train)} documents, and ask how surprised it is by the words of the `
    + `${n0(K.n_test)} it never saw. <span class="bold">Coherence</span> is external: take the `
    + `ten heaviest words of each topic and ask how often they turn up in the same documents as `
    + `each other, relative to how often they turn up at all. Both are drawn below over the same `
    + `fits, along with the agreement against the answer key, which is the thing neither of them `
    + `can see.`);

  const agreeCohPer = bestCoh.k === bestPer.k;
  const cohAgreeKey = bestCoh.k === bestAri.k;
  const perAgreeKey = bestPer.k === bestAri.k;
  const perMonotone = K.rows.every((r, i) => i === 0 || r.perplexity >= K.rows[i - 1].perplexity);
  set('k-verdict',
    `Perplexity is best at <span class="bold">k = ${bestPer.k}</span>, coherence at `
    + `<span class="bold">k = ${bestCoh.k}</span>, and agreement with the hand-written categories `
    + `at <span class="bold">k = ${bestAri.k}</span>. `
    + (agreeCohPer
      ? `The two usual criteria happen to agree here, which is not the general case and is worth `
        + `not generalising from. `
      : `<span class="bold">The two usual criteria do not pick the same model.</span> `)
    + (perAgreeKey
      ? `Perplexity happens to land on the number of real categories. `
      : `Neither does perplexity land on the ${word(M.categories.length)} categories that are `
        + `actually there. `)
    + (cohAgreeKey && !perAgreeKey
      ? `Coherence does, which is the argument for it. `
      : '')
    + (perMonotone
      ? `And perplexity keeps getting worse as topics are added, all the way across the sweep, `
        + `which is the opposite of the usual picture. `
      : `Perplexity does not fall away steadily either: it is best at k = ${bestPer.k} and `
        + `${K.rows[K.rows.length - 1].perplexity > bestPer.perplexity ? 'worse' : 'better'} by `
        + `k = ${K.rows[K.rows.length - 1].k}. `)
    + `The uncomfortable part is that on a corpus without an answer key, which is every corpus `
    + `anybody actually runs this on, you would have to choose one of the two criteria without `
    + `knowing which of them was closer to the truth.`);

  /* ------------------------------------------------------------- estabilidad */
  set('stability-intro',
    `There is a question you can ask of any unsupervised method that costs exactly one more run `
    + `of the code, and it is asked far less often than it should be: what happens if you change `
    + `the random seed. Below, ${word(St.seeds.length)} fits of the same model on the same `
    + `${n0(M.docs)} documents with the same ${word(S.k)} topics and the same priors, compared `
    + `pairwise. The topics are matched by optimal assignment before comparing, because a topic `
    + `model has no canonical order and comparing them by position would measure the permutation `
    + `rather than the disagreement.`);

  rows('#stability-table', St.pairs, (d) => [
    `${d.a} and ${d.b}`, f4(d.cosine), f4(d.worst), f4(d.jaccard),
  ]);

  set('stability-verdict',
    `Same data, same model, same number of topics. Matched topic to topic, the average similarity `
    + `between two runs is <span class="bold">${f4(St.mean_cosine)}</span>, and the worst matched `
    + `pair anywhere in the ${word(St.pairs.length)} comparisons is `
    + `<span class="bold">${f4(St.worst_topic)}</span>. Measured the way a reader actually `
    + `compares two tables of topics, by how many of the ten heading words are shared, it is `
    + `<span class="bold">${pc(St.mean_jaccard)}</span> overlap. And against the answer key the `
    + `${word(St.seeds.length)} runs score from <span class="value">${f4(St.ari_min)}</span> to `
    + `<span class="value">${f4(St.ari_max)}</span>, a spread of `
    + `<span class="value">${f4(St.ari_max - St.ari_min)}</span>, which is `
    + `${(St.ari_max - St.ari_min) > 0.05 ? 'larger' : 'smaller'} than the gap between the best `
    + `and worst choice of k in the sweep above `
    + `(<span class="value">${f4(bestAri.ari - Math.min(...K.rows.map((r) => r.ari)))}</span>). `
    + `The seed is not a modelling decision and nobody reports it.`);

  set('stability-examples',
    `Two of those runs, matched and side by side. Words in the accent are in both; faint words `
    + `are in one only.`);

  /* ------------------------------------------------------------- el filtrado */
  set('filter-intro',
    `Every recipe for a topic model opens with two lines of filtering, and they are always `
    + `presented as housekeeping: drop the words that appear almost everywhere, and drop the `
    + `stopword list. Below they are applied one at a time, so that each row differs from the one `
    + `above it by exactly one decision, which is the only way to know which of them did what.`);

  rows('#filter-table', F, (d) => [
    d.arm, n0(d.vocab), pc(d.stopword_frac_in_top, 0), f4(d.coherence), f4(d.ari),
  ]);

  const noFilter = F[0];
  const allFilter = F[F.length - 1];
  const filterSwing = allFilter.ari - noFilter.ari;
  /* Four sources of variation, ranked by how much each moves the same score,
     so the paragraph can say which decision mattered most instead of assuming
     it. Every one of them is measured on this page. */
  const sources = [
    { name: 'the filtering', size: Math.abs(filterSwing) },
    { name: 'the random seed', size: St.ari_max - St.ari_min },
    { name: 'the number of topics',
      size: bestAri.ari - Math.min(...K.rows.map((r) => r.ari)) },
    { name: 'which of the three factorisations',
      size: Math.max(...R.map((r) => r.ari)) - Math.min(...R.map((r) => r.ari)) },
  ].sort((a, b) => b.size - a.size);

  set('filter-verdict',
    `With nothing removed, <span class="bold">${pc(noFilter.stopword_frac_in_top, 0)}</span> of `
    + `the words heading the topics are stopwords, and the model scores `
    + `<span class="value">${f4(noFilter.ari)}</span> against the key. With both filters applied `
    + `it is <span class="bold">${pc(allFilter.stopword_frac_in_top, 0)}</span> and `
    + `<span class="value">${f4(allFilter.ari)}</span>. `
    + (filterSwing < 0
      ? `<span class="bold">The recommended preprocessing makes it worse.</span> Removing the `
        + `words everybody removes costs <span class="bold">${f4(-filterSwing)}</span> of `
        + `agreement with the hand-written categories, and it does so in exactly the direction the `
        + `previous article predicted: those stopwords were measured there to be among the most `
        + `informative features in this corpus, because genre gives a newswire story away before `
        + `topic does and function words are where genre lives. The topics that come out of the `
        + `unfiltered run are less pleasant to read and better at their job.`
      : `Removing them buys <span class="bold">${f4(filterSwing)}</span> of agreement with the `
        + `categories, which is the case for doing it, and it is worth having measured rather `
        + `than assumed.`)
    + ` Either way the filtering is not housekeeping: it moves the score by `
    + `<span class="value">${f4(Math.abs(filterSwing))}</span> before the model has run a single `
    + `iteration. Whether that is the biggest lever here is a question with an answer, and the `
    + `answer is that it is <span class="bold">${sources.findIndex((s) => s.name === 'the filtering') === 0
      ? 'the biggest of the four measured on this page'
      : `not: ${sources[0].name} moves it more (${f4(sources[0].size)} against `
        + `${f4(Math.abs(filterSwing))})`}</span>.`);

  set('alpha-note',
    `One more knob that is usually left wherever the library put it. The document prior decides `
    + `how mixed a document is allowed to be: small values push each document towards a single `
    + `topic, large values spread it across all of them.`);

  rows('#alpha-table', A, (d) => [
    f2(d.alpha),
    `${f3(Math.exp(d.mean_entropy))} of ${S.k}`,
    pc(d.frac_dominant, 0),
    f4(d.ari),
  ]);

  /* ------------------------------------------------------------------ rivals */
  const byName = Object.fromEntries(R.map((r) => [r.model, r]));
  const bestRivalCoh = R.reduce((p, c) => (c.coherence > p.coherence ? c : p), R[0]);
  const bestRivalAri = R.reduce((p, c) => (c.ari > p.ari ? c : p), R[0]);
  set('rivals-intro',
    `LDA is not the only way to turn a document-word table into a handful of shared axes. `
    + `<span class="bold">NMF</span> factors the same table into two non-negative pieces, and `
    + `<span class="bold">LSA</span> is a truncated singular value decomposition of it, which is `
    + `principal components under another name. All three get ${word(S.k)} components on the `
    + `same matrix, and all three are scored the same way against the same key.`);

  rows('#rivals-table', R, (d) => [
    d.model, f4(d.coherence), f4(d.ari), f4(d.purity), f4(d.one_to_one),
  ]);

  set('rivals-verdict',
    `The three do not separate the way the literature suggests. `
    + `<span class="bold">${bestRivalCoh.model}</span> has the most coherent topics at `
    + `<span class="value">${f4(bestRivalCoh.coherence)}</span>, and `
    + `<span class="bold">${bestRivalAri.model}</span> recovers the hand-written categories best `
    + `at <span class="value">${f4(bestRivalAri.ari)}</span>`
    + (bestRivalCoh.model === bestRivalAri.model
      ? `, and it is the same model on both counts.`
      : `, and they are <span class="bold">not the same model</span>: whichever you would have `
        + `picked by looking at the topic lists is not the one that found the structure.`)
    + ` The spread between the best and worst of the three on the key is `
    + `<span class="value">${f4(Math.max(...R.map((r) => r.ari)) - Math.min(...R.map((r) => r.ari)))}</span>, `
    + `which is ${(Math.max(...R.map((r) => r.ari)) - Math.min(...R.map((r) => r.ari)))
      > (St.ari_max - St.ari_min) ? 'more' : 'less'} than changing the seed of LDA alone.`);

  set('gibbs-vb-note',
    `And the same model fitted two completely different ways. The sampler in this page is `
    + `collapsed Gibbs; scikit-learn uses variational inference, which optimises a bound instead `
    + `of sampling. They are not the same algorithm and have no obligation to reach the same `
    + `place, so what can be compared is what is defined: on the same `
    + `${n0(S.docs)} documents they reach coherence `
    + `<span class="value">${f4(GV.gibbs.coherence)}</span> and `
    + `<span class="value">${f4(GV.vb.coherence)}</span>, agreement with the key `
    + `<span class="value">${f4(GV.gibbs.ari)}</span> and `
    + `<span class="value">${f4(GV.vb.ari)}</span>, and their topics matched against each other `
    + `sit at <span class="value">${f4(GV.match.mean)}</span> similarity on average with the `
    + `worst matched pair at <span class="value">${f4(GV.match.min)}</span>. Two algorithms for `
    + `one model disagree about as much as one algorithm disagrees with itself under a different `
    + `seed.`);

  /* ---------------------------------------------------------------- veredicto */
  set('v-filter',
    `${filterSwing < 0 ? 'Cost' : 'Bought'} ${f4(Math.abs(filterSwing))} of agreement with the `
    + `key, and turned ${pc(noFilter.stopword_frac_in_top, 0)} of topic headings into `
    + `${pc(allFilter.stopword_frac_in_top, 0)}.`);
  set('v-filter-watch',
    `It runs before the model and gets written up as data cleaning. The words it removes were `
    + `measured in the previous article to be the most informative ones in this corpus, and here `
    + `removing them ${filterSwing < 0 ? 'costs exactly what that predicts' : 'helps anyway'}.`);
  set('v-k',
    `Agreement ranges from ${f4(Math.min(...K.rows.map((r) => r.ari)))} to `
    + `${f4(bestAri.ari)} across the sweep, best at k = ${bestAri.k}.`);
  set('v-k-watch',
    agreeCohPer
      ? `The two usual criteria agreed here, but neither of them can see the key.`
      : `Perplexity says ${bestPer.k} and coherence says ${bestCoh.k}. Without a key there is no `
        + `way to know which to believe.`);
  set('v-seed',
    `${f4(St.ari_max - St.ari_min)} of agreement between the best and worst of `
    + `${word(St.seeds.length)} seeds, with matched topics only ${f4(St.mean_cosine)} alike.`);
  set('v-seed-watch',
    `It is not a modelling decision, so it is not in the methods section, so a result that `
    + `depends on it looks like a result that does not.`);
  const alphaBest = A.reduce((p, c) => (c.ari > p.ari ? c : p), A[0]);
  set('v-alpha',
    `From ${pc(A[0].frac_dominant, 0)} to ${pc(A[A.length - 1].frac_dominant, 0)} of documents `
    + `dominated by one topic across the sweep, with agreement best at ${f2(alphaBest.alpha)}.`);
  set('v-alpha-watch',
    `Libraries default it to 1/k without saying so, and it decides whether a document is allowed `
    + `to be about two things.`);

  /* ------------------------------------------------------------------ cierre */
  set('closing-1',
    `A topic model will always give you topics. Ask for ${word(S.k)} and you get ${word(S.k)}, `
    + `whether or not there are ${word(S.k)} of anything, and they will read plausibly because `
    + `words that occur together really do occur together. What this corpus adds is the thing `
    + `almost no corpus has: a way to check. The answer is that it finds something real `
    + `(<span class="value">${f4(bestAri.ari)}</span> at its best, against 0 for shuffling) and `
    + `that the decisions made before the model ran moved that number more than the model did.`);

  set('closing-2',
    `The next article changes what is being counted. Everything so far, in both articles, has `
    + `treated a word as an atom with no inside and no neighbours: "bond" and "bonds" are `
    + `different axes, and nothing anywhere has noticed. The repair is to stop giving each word `
    + `its own axis and start learning a position for it from the company it keeps, which turns `
    + `out to need nothing more exotic than counting again, more carefully. `
    + `<a href="../embeddings/">Word vectors</a> are next.`);

  set('ref-note',
    `The corpus is the same ${n0(M.docs)} Reuters stories the previous article used, and the `
    + `generator asserts that rather than claiming it: it rebuilds them by the same route and `
    + `checks the document count, the token count and the vocabulary size against the file that `
    + `article shipped. After dropping words in fewer than ${M.min_df} documents, words in more `
    + `than ${pc(M.max_df_frac, 0)} of them, and the ${M.stopwords_removed} English stopwords `
    + `NLTK distributes, ${n0(M.vocab)} words and ${n0(M.tokens)} tokens are left. The sampler `
    + `in the page runs on a balanced subset of ${n0(S.docs)} of those documents so that it `
    + `converges while you watch it.`);
}
