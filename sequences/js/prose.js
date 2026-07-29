/* Every sentence carrying a measured number, composed from the measurement.
 *
 * The article's central comparison is a direction ("the gated cells hold on
 * longer"), and this site has paid twice for writing a direction by hand, so
 * every one of them below is a branch on the measured half lives rather than a
 * sentence with numbers dropped in. If a rerun puts the plain RNN ahead, the
 * page says that instead of contradicting its own figure.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));
const NAME = { rnn: 'plain RNN', gru: 'GRU', lstm: 'LSTM' };

export const PROSE_IDS = [
  'intro-order', 'intro-task', 'intro-guard',
  'hmm-intro', 'hmm-note', 'hmm-score', 'hmm-verdict',
  'nets-intro', 'nets-verdict',
  'reach-intro', 'reach-verdict',
  'v-base', 'v-base-watch', 'v-hmm', 'v-hmm-watch', 'v-rnn', 'v-rnn-watch',
  'v-gated', 'v-gated-watch',
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
  const H = data.hmm;
  const N = data.nets;
  const F = data.flow;
  const C = data.copy;

  const cells = ['rnn', 'gru', 'lstm'];
  const half = Object.fromEntries(cells.map((k) => [k, F[k] && F[k].half_life]));
  const accs = Object.fromEntries(cells.map((k) => [k, N[k].acc]));
  const bestNet = cells.reduce((p, c) => (accs[c] > accs[p] ? c : p), cells[0]);
  const worstNet = cells.reduce((p, c) => (accs[c] < accs[p] ? c : p), cells[0]);
  /* A half life of null means the gradient never fell to half within the
     sentence, which is the strongest possible result and has to sort as such
     rather than as a missing value. */
  const reachOf = (k) => (half[k] === null || half[k] === undefined
    ? Number.POSITIVE_INFINITY : half[k]);
  const longest = cells.reduce((p, c) => (reachOf(c) > reachOf(p) ? c : p), cells[0]);
  const shortest = cells.reduce((p, c) => (reachOf(c) < reachOf(p) ? c : p), cells[0]);
  const gatedWin = reachOf('rnn') < Math.max(reachOf('gru'), reachOf('lstm'));

  const breaksAt = (k) => {
    const bad = (C[k] || []).find((r) => r.acc < 0.6);
    return bad ? bad.distance : null;
  };

  /* ------------------------------------------------------------- the opening */
  set('intro-order',
    `Three articles of this branch have now thrown the word order away and got surprisingly far `
    + `without it. That works because topic is mostly carried by which words are present. It `
    + `stops working the moment the question is about a word's <span class="bold">role</span> `
    + `rather than a document's subject, and the cheapest example is that "the dollar rose against `
    + `the yen" and "the yen rose against the dollar" are the same bag.`);

  set('intro-task',
    `The oldest task in the field is a good place to put the order back. Every one of the `
    + `${n0(M.train_tokens + M.test_tokens)} words in this corpus was given a `
    + `<span class="bold">part-of-speech tag</span> by hand, from a set of `
    + `${word(M.tags.length)}, so unlike a topic model there is a right answer for every single `
    + `token and no argument about what counts as one. The split is by sentence and not by word, `
    + `which matters: splitting inside a sentence would leave half of a context in training and `
    + `half in test, and the test would stop measuring what it is for. `
    + `${n0(M.train_sents)} sentences to train on, ${n0(M.test_sents)} held out.`);

  set('intro-guard',
    `The Markov model runs in this page in full, which is not a boast: a hidden Markov model `
    + `<span class="bold">is</span> three tables of numbers, and all three fit in the file. So the `
    + `page decodes every sentence below from scratch, and the check at load time is that its `
    + `decode matches the generator's tag for tag, plus one thing a tolerance cannot express: `
    + `that the greedy decoder never scores higher than Viterbi, which is impossible if the `
    + `dynamic program is right and is the way a wrong one usually shows itself.`);

  /* ------------------------------------------------------------------ el HMM */
  set('hmm-intro',
    `A hidden Markov model makes two assumptions and both are wrong on purpose. The first is that `
    + `a tag depends only on the tag immediately before it, so "${M.tags[0].toLowerCase()} after `
    + `${M.tags[1].toLowerCase()}" has a probability and everything further back has no vote at `
    + `all. The second is that a word depends only on its own tag and not on any neighbouring `
    + `word. Written down, the entire model is one product:`);

  set('hmm-note',
    `Every tag depends on exactly one tag, and every word depends on exactly one tag. Nothing `
    + `else. That is ${M.tags.length * M.tags.length} transition numbers and `
    + `${n0(M.tags.length * M.vocab)} emission numbers, counted rather than fitted, with one `
    + `smoothing constant of ${data.hmm_alpha} applied identically to all three tables so that no `
    + `later comparison can turn out to have depended on tuning one of them.`);

  set('hmm-score',
    `Decoding a sentence means choosing the best of ${M.tags.length}<sup>n</sup> taggings, which `
    + `for the sentences above is more than a billion, and Viterbi does it in a number of steps `
    + `linear in the length by keeping only the best path into each tag at each word. Over the `
    + `whole held-out split, and against the one baseline that matters:`);

  rows('#hmm-table', [H], (d) => [
    n0(d.tokens),
    pc(d.baseline),
    pc(d.hmm),
    { html: `<span class="value">${d.hmm >= d.baseline ? '+' : ''}${pc(d.hmm - d.baseline)}</span>` },
  ]);

  const gain = H.hmm - H.baseline;
  const ambGain = H.hmm_ambiguous - H.baseline_ambiguous;
  set('hmm-verdict',
    `The baseline is embarrassingly strong, and saying so is the point: just giving every word `
    + `the tag it wore most often in training, with no context whatsoever, gets `
    + `<span class="bold">${pc(H.baseline)}</span>. The model that reads the sentence gets `
    + `<span class="bold">${pc(H.hmm)}</span>, `
    + `${Math.abs(gain) < 0.002
      ? 'which is barely a difference at all'
      : `${gain > 0 ? 'a gain of' : 'a loss of'} <span class="bold">${pc(Math.abs(gain))}</span>`}. `
    + `The reason the gap is small is that most words are not ambiguous: only `
    + `<span class="value">${pc(H.ambiguous_share, 1)}</span> of tokens are words that ever wore `
    + `more than one tag in training. Restricted to those, which is where a sequence model is `
    + `supposed to earn its keep, it is `
    + `<span class="bold">${pc(H.hmm_ambiguous)}</span> against `
    + `<span class="bold">${pc(H.baseline_ambiguous)}</span>`
    + `${Math.abs(ambGain) > Math.abs(gain)
      ? `, a gap ${(Math.abs(ambGain) / Math.max(Math.abs(gain), 1e-9)).toFixed(1)} times larger `
        + `than the one on the whole split`
      : ''}. And on the ${n0(H.unknown_tokens)} tokens whose word never appeared in training at `
    + `all, the baseline has nothing to say and guesses one tag for all of them `
    + `(<span class="value">${pc(H.baseline_unknown)}</span>) while the model still has the `
    + `transitions (<span class="value">${pc(H.hmm_unknown)}</span>).`);

  /* ---------------------------------------------------------------- las redes */
  set('nets-intro',
    `A recurrent network drops the Markov assumption entirely. Instead of one tag of memory it `
    + `carries a vector, updated at every word, that can in principle hold anything about `
    + `everything that came before. The three cells below differ only in how that vector gets `
    + `updated, and they are given the <span class="bold">same parameter budget</span> rather than `
    + `the same hidden width: an LSTM has four gates and a plain RNN has one, so at equal width `
    + `the LSTM would be nearly four times the model and the table would be measuring size.`);

  rows('#nets-table', cells.map((k) => ({ k, ...N[k] })), (d) => [
    NAME[d.k], n0(d.hidden), n0(d.recurrent_params), pc(d.acc),
  ]);

  const spread = accs[bestNet] - accs[worstNet];
  set('nets-verdict',
    `Best of the three is the <span class="bold">${NAME[bestNet]}</span> at `
    + `<span class="bold">${pc(accs[bestNet])}</span>, worst is the `
    + `<span class="bold">${NAME[worstNet]}</span> at `
    + `<span class="bold">${pc(accs[worstNet])}</span>, and the spread across all three is `
    + `<span class="value">${pc(spread)}</span>. Against the hidden Markov model's `
    + `${pc(H.hmm)} that is `
    + `${accs[bestNet] > H.hmm
      ? `a gain of <span class="bold">${pc(accs[bestNet] - H.hmm)}</span>`
      : `<span class="bold">no gain at all</span>`} for a model with `
    + `${n0(N[bestNet].params)} parameters against three counted tables. `
    + `Which is a small difference for a large change of machinery, and the honest reading is not `
    + `that recurrence is useless: it is that <span class="bold">this task does not need `
    + `it</span>. Tagging is decided almost entirely by the word itself and its immediate `
    + `neighbours, so a model that can remember thirty words back has nothing to remember. To `
    + `tell these three apart you have to ask them something that does.`);

  /* ------------------------------------------------------------------ memoria */
  set('reach-intro',
    `Tagging cannot tell these three apart, which is exactly why the next two measurements do not `
    + `use it. The first asks the models directly: take a long sentence, and measure how much the `
    + `input at each earlier position still moves the state at the end. That number is what `
    + `training propagates backwards, so where it vanishes is where learning from that distance `
    + `stops. The second asks a task built for nothing else: remember one symbol across a growing `
    + `wall of distractors.`);

  const nf = (v) => (v === null || v === undefined ? 'never, within the sentence' : `${v} steps`);
  const F0 = data.flow_untrained || {};
  const half0 = Object.fromEntries(cells.map((k) => [k, F0[k] && F0[k].half_life]));
  const reach0 = (k) => (half0[k] === null || half0[k] === undefined
    ? Number.POSITIVE_INFINITY : half0[k]);
  const gated0 = Math.max(reach0('gru'), reach0('lstm')) > reach0('rnn');
  const spreadTrained = Math.max(...cells.map(reachOf)) - Math.min(...cells.map(reachOf));

  set('reach-verdict',
    `Two views, and the second is the one that resolves anything. Before training, the gradient `
    + `falls to half in ${nf(half0.rnn)} for the plain RNN and ${nf(half0.gru)} and `
    + `${nf(half0.lstm)} for the GRU and the LSTM, `
    + (gated0
      ? `so the gated cells do carry further at initialisation, which is the direction the `
        + `textbook gives`
      : `so the gated cells do not carry further at initialisation here`)
    + `. After training on tagging they are ${nf(half.rnn)}, ${nf(half.gru)} and ${nf(half.lstm)}, `
    + `spread over <span class="bold">${spreadTrained} step${spreadTrained === 1 ? '' : 's'}</span> `
    + `between the three.</div><div>Which is a difference of one or two steps either way, and the `
    + `honest reading is that <span class="bold">this measurement does not separate these three `
    + `architectures</span> on this task. It is worth saying rather than dressing up, and the `
    + `reason is visible in the numbers above it: trained on a problem whose answers are local, `
    + `all three learned to be local, so the curve is a picture of what tagging asked for rather `
    + `than of what the cell could do. A probe that only ever runs on a task that needs no memory `
    + `cannot measure memory.</div><div>The copy task is the one that separates them, because it `
    + `is nothing but memory: `
    + `${cells.map((k) => `<span class="bold">${NAME[k]}</span> `
      + (breaksAt(k) === null
        ? `holds to ${data.copy_distances[data.copy_distances.length - 1]}`
        : `breaks at ${breaksAt(k)} distractors`)).join(', ')}. `
    + `There the three are not within a step of each other at all, and the ordering is `
    + `${(breaksAt('rnn') !== null && (breaksAt('gru') === null || breaksAt('gru') > breaksAt('rnn')))
      ? 'the one the gates were invented to produce'
      : 'not the clean one either, which is what the table says'}.`);

  /* ---------------------------------------------------------------- veredicto */
  set('v-base',
    `${pc(H.baseline)} of tokens right with no model at all, and ${pc(H.baseline_ambiguous)} on `
    + `the ambiguous ones.`);
  set('v-base-watch',
    `${pc(H.baseline_unknown)} on words it never saw, because it can only fall back to one tag `
    + `for all of them.`);
  set('v-hmm',
    `${pc(H.hmm)} overall and ${pc(H.hmm_ambiguous)} on ambiguous words, from three counted `
    + `tables and no gradient descent anywhere.`);
  set('v-hmm-watch',
    `One step of memory by construction. Anything the answer depends on from two tags back is `
    + `invisible to it, whatever the sentence says.`);
  set('v-rnn',
    `${pc(accs.rnn)} on tagging, and a gradient half life of ${nf(half.rnn)}.`);
  set('v-rnn-watch',
    breaksAt('rnn') === null
      ? `It held the copy task to the longest distance tried, which is not the usual result.`
      : `It stops solving the copy task at ${breaksAt('rnn')} distractors.`);
  set('v-gated',
    `${pc(Math.max(accs.gru, accs.lstm))} on tagging, ${pc(spread)} of spread across all three, `
    + `and a longer reach: ${NAME[longest]} keeps its gradient furthest.`);
  set('v-gated-watch',
    `Still strictly one token at a time. Nothing on this page can look at position 40 until it `
    + `has been through the other 39, which is the constraint the next article removes.`);

  /* ------------------------------------------------------------------ cierre */
  set('closing-1',
    `Every model on this page has the same shape, and it is worth naming before leaving it: a `
    + `state that gets updated once per token, in order. That shape is why an HMM can be counted `
    + `instead of fitted, and it is why the ${NAME[shortest]}'s influence from `
    + `${nf(half[shortest])} back has effectively gone. The gates in an LSTM do not remove the `
    + `constraint, they postpone it: the path from word 1 to word 60 is still sixty multiplications long.`);

  set('closing-2',
    `The next article removes the recurrence instead of improving it. If the problem is that the `
    + `distance between two words is the number of steps between them, one fix is to make every `
    + `pair of positions <span class="bold">one step apart</span>, whatever the sentence length, `
    + `and pay for it in a different currency. The same gradient probe and the same copy task run `
    + `there, on the same split, so the curves go on the same axes. `
    + `<a href="../attention/">Self-attention</a> is next.`);

  set('ref-note',
    `The corpus is the Brown corpus as distributed with NLTK, ${n0(M.train_sents + M.test_sents)} `
    + `sentences mapped to the ${word(M.tags.length)} universal part-of-speech tags rather than to `
    + `the 472 fine-grained ones the corpus ships with: with 472 tags the transition table has `
    + `222,784 cells to estimate from this many sentences, which is not a hard problem but an `
    + `unanswerable one. Words seen fewer than ${M.min_count} times in training are folded into a `
    + `single unknown token, leaving a vocabulary of ${n0(M.vocab)}. The split is `
    + `${n0(M.train_sents)} sentences to train and ${n0(M.test_sents)} held out, drawn with seed `
    + `${M.seed}.`);
}
