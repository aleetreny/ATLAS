/* Every sentence with a measured number, composed from the measurement.
 *
 * Two things in this article are unusually easy to overclaim and both are
 * written as branches here. The first is the theorem: a high correlation with
 * shifted mutual information is not the same as reproducing it, and the
 * sentence that says which one happened reads the measured slope rather than
 * assuming. The second is the analogy accuracy, which on a corpus this size is
 * a fact about this corpus and is said that way.
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
  'intro-problem', 'intro-idea', 'intro-guard',
  'objective-intro', 'objective-theorem', 'pmi-note',
  'nb-intro', 'window-intro', 'window-verdict',
  'models-intro', 'analogy-note', 'oov-intro', 'oov-result',
  'v-sg', 'v-sg-watch', 'v-gl', 'v-gl-watch', 'v-ft', 'v-ft-watch',
  'v-win', 'v-win-watch',
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

export function initProse(data, space) {
  const M = data.meta;
  const P = data.pmi;
  const W = data.windows;
  const R = data.models;
  const O = data.oov;

  const byName = Object.fromEntries(R.map((r) => [r.model, r]));
  const small = W.reduce((p, c) => (c.window < p.window ? c : p), W[0]);
  const large = W.reduce((p, c) => (c.window > p.window ? c : p), W[0]);
  const posDrop = small.same_pos - large.same_pos;
  const slopeClose = Math.abs(P.slope - 1) < 0.15;

  /* ------------------------------------------------------------- the opening */
  set('intro-problem',
    `Two articles of this branch have now represented a document by counting its words, and both `
    + `ran into the same wall. Every word gets an axis of its own, at right angles to every other, `
    + `so "bond" and "bonds" are exactly as unrelated as "bond" and "wheat", and nothing in either `
    + `article could notice. Reweighting the axes cannot fix it, because reweighting stretches `
    + `axes and never joins them. Neither can a topic model: it groups <span class="bold">`
    + `documents</span> by the words in them, not words by the company they keep.`);

  set('intro-idea',
    `The repair is a sentence from a linguist, written in 1957 and quoted ever since: you shall `
    + `know a word by the company it keeps. Instead of an axis per word, give each word a `
    + `<span class="bold">position</span> in a space of ${M.dim} dimensions, and choose the `
    + `positions so that words appearing in similar company end up near each other. Nothing about `
    + `that requires a neural network, and the second figure below is the proof: what the most `
    + `famous method is doing turns out to be factorising a table of counts.`);

  set('intro-guard',
    `Three models, one corpus, one vocabulary, one budget: skip-gram with negative sampling, `
    + `GloVe and FastText, all trained here on the ${n0(M.tokens)} words of the Brown corpus with `
    + `${M.dim} dimensions, ${word(M.epochs)} passes and the same ${n0(M.vocab)} words. Sharing `
    + `the hyperparameters is the whole point: most of the published gap between these three is `
    + `hyperparameters rather than method. All three then travel to this page in full, so every `
    + `neighbour list and every analogy below is computed in your browser rather than chosen in `
    + `advance, and the check at load time is that the vectors survived the trip to half `
    + `precision and still come back sorted.`);

  /* ---------------------------------------------------------------- objetivo */
  set('objective-intro',
    `Skip-gram with negative sampling is usually introduced as a neural network, which makes it `
    + `sound more mysterious than it is. There is no hidden layer and no non-linearity. There are `
    + `two vectors per word, and for every pair of words that actually occurred near each other, `
    + `the objective pushes their dot product up, while for ${word(M.negatives)} pairs drawn from `
    + `noise it pushes them down:`);

  set('objective-theorem',
    `And there is a theorem about where that lands. Levy and Goldberg showed that this objective `
    + `is optimised when the dot product of a word and a context equals a quantity computed purely `
    + `by counting: the pointwise mutual information of the pair, minus the log of the number of `
    + `noise samples. Below, that quantity on one axis and the dot product the training actually `
    + `produced on the other, over ${n0(P.pairs)} pairs. The dashed diagonal is not a fit. It is `
    + `the theorem.`);

  set('pmi-note',
    `Which reframes the whole thing. `
    + (slopeClose
      ? `The measured slope is <span class="bold">${f3(P.slope)}</span>, close enough to one that `
        + `the model is reproducing the counting quantity and not merely ranking pairs like it.`
      : `The correlation is <span class="bold">${f3(P.pearson)}</span> and the slope is `
        + `<span class="bold">${f3(P.slope)}</span>, and the gap between those two numbers is the `
        + `honest reading: this model has learned to <span class="bold">order</span> pairs much `
        + `the way mutual information does without reaching its scale. The theorem is about the `
        + `optimum, and ${word(M.epochs)} passes over a million words is not the optimum. Quoting `
        + `the correlation alone would have made the claim look exact.`)
    + ` Either way the direction of travel is the same, and it is the useful part: an embedding is `
    + `not learning something a count cannot see. It is compressing a table of counts, and the `
    + `compression is the point, because the table has ${n0(M.vocab)} rows and columns and almost `
    + `all of it is zero.`);

  /* -------------------------------------------------------------- los vecinos */
  set('nb-intro',
    `The usual way to show off an embedding is a list of nearest neighbours for a well-chosen `
    + `word. That proves nothing, so all three models are asked about the same word at the same `
    + `time, and every neighbour carries the part-of-speech tag the corpus gives it. The tags are `
    + `what turns "these look similar" into a count.`);

  /* -------------------------------------------------------------- la ventana */
  set('window-intro',
    `There is one hyperparameter here that changes the <span class="bold">meaning</span> of the `
    + `output rather than its quality, and it is usually reported as a detail. The window is how `
    + `far either side of a word counts as its company. Narrow, and the company is grammatical `
    + `neighbours; wide, and the company is whatever the passage is about. Because this corpus is `
    + `tagged, that stops being a matter of impression: ask, for each word, how many of its ten `
    + `nearest neighbours carry its own part of speech.`);

  rows('#window-table', W, (d) => [
    `${d.window} either side`, n0(d.pairs), pc(d.same_pos), pc(d.stem_share),
  ]);

  set('window-verdict',
    `Same algorithm, same corpus, same number of dimensions, same number of passes. With a window `
    + `of ${small.window}, <span class="bold">${pc(small.same_pos)}</span> of a word's ten nearest `
    + `neighbours share its part of speech; with a window of ${large.window} it is `
    + `<span class="bold">${pc(large.same_pos)}</span>, `
    + (posDrop > 0.02
      ? `a fall of <span class="bold">${pc(posDrop)}</span>. The narrow window is finding words `
        + `that could stand in the same slot of a sentence; the wide one is finding words that `
        + `turn up in the same kind of passage. Both are "similar" and they are not the same `
        + `relation.`
      : posDrop < -0.02
        ? `which goes <span class="bold">up</span> rather than down, against the usual account, `
          + `and at this corpus size that is what the measurement says.`
        : `which is not a meaningful change on this corpus. The effect is real in the literature `
          + `at much larger scale; here it does not clear anything worth calling a difference, and `
          + `saying so is better than reporting the sign of a rounding error.`)
    + ` Nobody writing "window=5" in a configuration file is usually aware they are choosing which `
    + `of those two questions the vectors will answer.`);

  /* ------------------------------------------------------------- tres modelos */
  set('models-intro',
    `All three on the same corpus, the same vocabulary, the same ${M.dim} dimensions and the same `
    + `window of ${data.windows[1] ? data.windows[1].window : 5}. The last two columns are the `
    + `same analogy test run twice with one line changed, which is explained under the table.`);

  rows('#models-table', R, (d) => [
    d.model, pc(d.same_pos), pc(d.stem_share), pc(d.analogy),
    pc(d.analogy_lazy),
  ]);

  const bestAn = R.reduce((p, c) => (c.analogy > p.analogy ? c : p), R[0]);
  const ft = byName.fasttext;
  const sg = byName['skip-gram'];
  set('analogy-note',
    `The last two columns are the same test with one line changed. Solving "a is to b as c is to `
    + `what" means finding the word nearest to b minus a plus c, and every published accuracy on `
    + `this task <span class="bold">removes a, b and c from the search first</span>. Without that, `
    + `the answer is very often one of the three words handed back, because c is the nearest thing `
    + `to something that is almost c: across all ${n0(bestAn.analogy_asked)} questions the `
    + `accuracy falls from ${pc(bestAn.analogy)} to ${pc(bestAn.analogy_lazy)} for the best of the `
    + `three when the exclusion is removed. The widget below lets you turn it off and watch which `
    + `answers change. And the honest caveat on all of these numbers: `
    + `${n0(M.tokens)} words is a small corpus, so ${pc(bestAn.analogy)} is a statement about this `
    + `corpus and not about the method.`);

  /* -------------------------------------------------------------------- OOV */
  set('oov-intro',
    `Every model in this branch so far has had the same hard edge: a word outside the training `
    + `vocabulary has no representation at all. Not a poor one, none. TF-IDF has no column for it, `
    + `a topic model has no probability for it, and skip-gram and GloVe have no row. FastText is `
    + `the same objective as skip-gram with one change: a word's vector is the average of its own `
    + `row and the rows of its character n-grams, ${M.subword_range[0]} to ${M.subword_range[1]} `
    + `characters long, hashed into ${n0(M.subword_buckets)} buckets. A word it has never seen `
    + `still has n-grams it has seen.`);

  if (O) {
    const works = O.wins > 0.65 || O.lift > 0.05;
    set('oov-result',
      `Measured on ${n0(O.words)} words that fell just below the vocabulary cut, so neither model `
      + `has a row for any of them. The target is not either model's opinion: it is the average of `
      + `the vectors of the words that actually surrounded it in the corpus, which is the `
      + `distributional definition of what the word means. The control is the one that matters and `
      + `it took two attempts to get right: comparing against two words picked at random measures `
      + `the <span class="bold">anisotropy of the space</span> rather than anything about `
      + `composition, because trained vectors share a dominant direction and any two of them `
      + `already sit at a cosine of about 0.36. The control here is the same composed vector `
      + `against a <span class="bold">different word's</span> target, which holds both `
      + `distributions fixed and varies only the pairing.</div><div>`
      + (works
        ? `FastText's composed vector lands <span class="bold">${f3(O.mean_cosine)}</span> from `
          + `its own target against <span class="bold">${f3(O.mismatched_mean)}</span> from `
          + `somebody else's, and picks its own target over a stranger's `
          + `<span class="bold">${pc(O.wins)}</span> of the time. The character n-grams are `
          + `carrying real information about a word the model never saw.`
        : `<span class="bold">And at this scale it does not work.</span> The composed vector lands `
          + `<span class="value">${f3(O.mean_cosine)}</span> from its own target and `
          + `<span class="value">${f3(O.mismatched_mean)}</span> from a stranger's, a difference `
          + `of <span class="bold">${f3(O.lift)}</span>, and it prefers its own target `
          + `<span class="bold">${pc(O.wins)}</span> of the time where chance is 50%. That is `
          + `not a demonstration of anything. The mechanism is real and is used at scale every `
          + `day; what this corpus shows is that ${n0(M.tokens)} words is not enough to learn `
          + `useful representations for ${n0(M.subword_buckets)} hashed character n-grams, most `
          + `of which are seen a handful of times. The result stays here rather than being `
          + `quietly dropped, because a version of this measurement without the matched control `
          + `would have printed a comfortable-looking ${f3(O.mean_cosine)} and said nothing true.`)
      + `</div><div>Skip-gram and GloVe are not in this comparison at all, and that is not a low `
      + `score, it is the absence of one: there is no vector to score. Words like `
      + `<span class="bold">${O.examples.slice(0, 5).join(', ')}</span> are the kind that fall `
      + `below a cut, and an open vocabulary means there are always more of them.`);
  } else {
    set('oov-result',
      `The experiment needs words below the vocabulary cut with enough surviving context to `
      + `estimate a target from, and this corpus did not supply enough of them, so there is no `
      + `number here rather than a number measured on a handful of words.`);
  }

  /* ---------------------------------------------------------------- veredicto */
  set('v-sg',
    `${pc(sg.same_pos)} part-of-speech agreement among neighbours and ${pc(sg.analogy)} on the `
    + `analogy sets, and it is the model the theorem in the second figure is about.`);
  set('v-sg-watch',
    `No answer at all for a word it never saw, and the window quietly decides what its "similar" `
    + `means.`);
  set('v-gl',
    `${pc(byName.glove.same_pos)} agreement and ${pc(byName.glove.analogy)} on the analogies, `
    + `fitted to the whole co-occurrence table at once rather than pair by pair.`);
  set('v-gl-watch',
    `Same hard edge on unseen words, and it needs the ${n0(M.cooc_cells)} non-zero cells of that `
    + `table in memory.`);
  set('v-ft',
    `${pc(ft.same_pos)} agreement and ${pc(ft.stem_share)} of neighbours sharing a prefix against `
    + `${pc(sg.stem_share)} for skip-gram, which is the morphology showing up. `
    + (O
      ? (O.wins > 0.65 || O.lift > 0.05
        ? `On words it never saw it picks the right target ${pc(O.wins)} of the time.`
        : `On words it never saw it picks the right target only ${pc(O.wins)} of the time at this `
          + `corpus size, which is barely above chance.`)
      : 'It is the only one of the three with any answer for a word it never saw.'));
  set('v-ft-watch',
    `Character n-grams tie words that merely look alike, which is why its neighbours share `
    + `prefixes ${byName.fasttext.stem_share > byName['skip-gram'].stem_share
      ? 'more often than the others do' : 'no more often than the others'}.`);
  set('v-win',
    Math.abs(posDrop) > 0.02
      ? `${pc(Math.abs(posDrop))} of part-of-speech agreement between the narrowest and widest `
        + `window tried, in the direction the literature describes.`
      : `${pc(Math.abs(posDrop))} between the narrowest and widest window, which is not a `
        + `difference this corpus can resolve.`);
  set('v-win-watch',
    `It is reported as a detail and it behaves as a definition of the word "similar".`);

  /* ------------------------------------------------------------------ cierre */
  set('closing-1',
    `A word now has a position rather than an axis, and the position knows things: that this word `
    + `and that one fill the same slot, that these two are about the same subject, that a word `
    + `nobody has ever seen is probably a plural. All of it came out of counting which words turn `
    + `up near which, and the second figure says that in the strongest possible way, by putting `
    + `what the training produced against what pure counting predicts.`);

  set('closing-2',
    `What none of it knows is <span class="bold">where</span> the word was. Every vector on this `
    + `page is one fixed position per word: "bank" has a single vector doing duty for the river `
    + `and the money, and the model has no way to ask which one this is, because it never sees `
    + `this one. It sees the word type. Fixing that means a representation that is computed `
    + `per occurrence and therefore has to read the sentence in order, which is where the next `
    + `article starts, on a task where the answer for every single token is known. `
    + `<a href="../sequences/">Sequence models</a> are next.`);

  set('ref-note',
    `The corpus is the Brown corpus as distributed with NLTK, ${n0(M.sentences)} sentences and `
    + `${n0(M.tokens)} words, chosen over the newswire of the previous two articles for one `
    + `reason that the article then uses: it is tagged with parts of speech, so "what does similar `
    + `mean here" has an answer that can be counted. Words occurring fewer than ${M.min_count} `
    + `times are dropped, leaving ${n0(M.vocab)} of them. Frequent words are subsampled at `
    + `${M.subsample} and the noise distribution is the frequency raised to three quarters, both `
    + `as specified in the original paper and both often left out of reimplementations. The `
    + `${n0(M.shipped)} most frequent words travel to this page in half precision, and every `
    + `figure was measured on those rounded values rather than on the ones training produced.`);
}
