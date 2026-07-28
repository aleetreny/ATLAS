/* Every sentence with a measured number, composed from the measurement.
 *
 * This article compares three things on four axes, so almost every sentence
 * is a branch on which of them won rather than a description of a result. The
 * mask rate paragraph in particular is written so that it says whatever the
 * sweep says, including the possibility that the folklore value is the best
 * one, which would also be worth knowing.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-labels', 'intro-three', 'intro-guard',
  'arms-intro', 'objective-note', 'arms-verdict',
  'rate-intro', 'rate-verdict',
  'v-bert', 'v-bert-watch', 'v-gpt', 'v-gpt-watch', 'v-t5', 'v-t5-watch',
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
  const A = data.arms;
  const S = data.rate_sweep;

  const by = Object.fromEntries(A.map((a) => [a.kind, a]));
  const bestProbe = A.reduce((p, c) => (c.probe > p.probe ? c : p), A[0]);
  const bestCloze = A.reduce((p, c) => (c.cloze > p.cloze ? c : p), A[0]);
  const bestSignal = A.reduce((p, c) => (c.signal_per_token > p.signal_per_token ? c : p), A[0]);
  const pmin = Math.min(...A.map((a) => a.params));
  const pmax = Math.max(...A.map((a) => a.params));

  set('intro-labels',
    `Every model in this branch so far has been trained on labels somebody wrote by hand: eight `
    + `categories assigned by people at Reuters, or a part of speech for each of a million words. `
    + `Those labels are the scarcest thing in the field. Raw text is not scarce at all, and the `
    + `question that reorganised the last decade is what you can train on text that has no labels `
    + `whatsoever.`);

  set('intro-three',
    `There are three well-known answers, usually presented as three different models with three `
    + `different names. They are not three architectures. They are the <span class="bold">same `
    + `architecture asked to predict three different things</span>: hide some words and predict `
    + `them from both sides, predict each next word from the left only, or delete whole spans and `
    + `regenerate them. Everything people say about BERT, GPT and T5 that is not about size comes `
    + `out of that choice.`);

  set('intro-guard',
    `One architecture, one corpus, one budget. All three arms below are ${M.layers} layers at `
    + `${M.d_model} dimensions with ${M.heads} heads, trained on the same `
    + `${n0(M.chunks_train)} chunks of ${M.seq} tokens from the same corpus the last two articles `
    + `used, for the same <span class="bold">${n0(M.steps)} optimiser steps</span> at the same `
    + `batch size, over the same vocabulary of ${n0(M.vocab)} words, with parameter counts inside `
    + `${pc((pmax - pmin) / pmin, 1)} of each other. Equal steps rather than equal wall clock, `
    + `because a clock measures the machine. The only thing that differs between the three is what `
    + `the model was asked to predict.`);

  set('arms-intro',
    `What separates them is a mask and a target, and nothing else. The masked arm replaces `
    + `${pc(M.mask_rate, 0)} of the tokens with a placeholder and is graded on guessing them, `
    + `with the eighty-ten-ten recipe the original paper specifies and most reimplementations `
    + `skip. The causal arm sees a triangular mask and is graded on every next word. The span arm `
    + `deletes runs of words, leaves a sentinel behind, and a decoder regenerates them.`);

  const E = data.examples;
  if (E) {
    const gradedT5 = E.t5.target.length - 1;
    set('objective-note',
      `Those are real training examples, corrupted by the same code that trained the three arms `
      + `rather than redrawn here, which is why the masked one has a stray word that was replaced `
      + `by a different word instead of a placeholder. Count the bars: on the same `
      + `<span class="value">${E.words.length}</span> tokens, the masked arm is graded on `
      + `<span class="bold">${E.bert.graded.length}</span> positions, the causal arm on `
      + `<span class="bold">${E.gpt.graded.length}</span>, and the span arm on `
      + `<span class="bold">${gradedT5}</span> tokens of a second sequence it has to produce from `
      + `nothing. That difference is not a detail of the picture. It is the entire table below.`);
  }

  rows('#arms-table', A, (d) => [
    d.model, n0(d.params), n0(d.supervised), f3(d.signal_per_token),
    pc(d.probe), pc(d.cloze),
  ]);

  const sameWinner = bestProbe.kind === bestCloze.kind;
  set('arms-verdict',
    `The column nobody prints is the fourth one. All three read `
    + `<span class="value">${n0(A[0].tokens_seen)}</span> tokens, and the number of predictions `
    + `they were graded on differs by a factor of `
    + `<span class="bold">${(bestSignal.signal_per_token / Math.min(...A.map((a) => a.signal_per_token))).toFixed(1)}</span>: `
    + `${bestSignal.model} extracts <span class="value">${f3(bestSignal.signal_per_token)}</span> `
    + `supervised predictions from each token it reads, and the masked objective extracts `
    + `<span class="value">${f3(by.bert.signal_per_token)}</span>, because it only grades itself `
    + `on the words it hid. That single ratio is most of the reason the causal family kept `
    + `scaling.</div><div>On the other two measures: the frozen probe is best for `
    + `<span class="bold">${bestProbe.model}</span> at ${pc(bestProbe.probe)}, and filling a gap `
    + `in the middle is best for <span class="bold">${bestCloze.model}</span> at `
    + `${pc(bestCloze.cloze)}`
    + (sameWinner
      ? `, the same arm on both.`
      : `, a different arm on each. Which is the point of running all three rather than one: `
        + `there is no ranking here, there is a choice of what you are going to do next.`)
    + ` And the asymmetry the table cannot show: the masked arm `
    + `<span class="bold">cannot generate at all</span>, whatever its numbers, because nothing in `
    + `its training ever asked it to produce a word without seeing the words after it.`);

  const bestRate = S.reduce((p, c) => (c.probe > p.probe ? c : p), S[0]);
  const folklore = S.find((r) => Math.abs(r.rate - 0.15) < 1e-9);
  const spread = Math.max(...S.map((r) => r.probe)) - Math.min(...S.map((r) => r.probe));
  set('rate-intro',
    `One number in the masked objective has been copied from paper to paper since 2018 without `
    + `being re-derived: mask ${pc(0.15, 0)} of the tokens. It is a reasonable number and it was `
    + `never claimed to be optimal. Here it is swept, with everything else held fixed and the same `
    + `frozen probe used to score the result.`);

  rows('#rate-table', S, (d) => [
    pc(d.rate, 0), n0(d.supervised), f4(d.final_loss), pc(d.probe),
  ]);

  const beats = folklore ? bestRate.probe - folklore.probe : null;
  set('rate-verdict',
    `Two things pull against each other here and the shape of the curve is the argument. Masking `
    + `more gives the model more predictions to learn from, which is the fourth column going up. `
    + `Masking more also destroys the context it needs in order to make them, so past some point `
    + `each prediction is a worse one. `
    + (bestRate.rate === 0.15
      ? `On this corpus the best of the rates tried is <span class="bold">${pc(0.15, 0)}</span>, `
        + `which is the folklore value, and it is worth having checked rather than assumed.`
      : `On this corpus the best of the rates tried is `
        + `<span class="bold">${pc(bestRate.rate, 0)}</span>, not ${pc(0.15, 0)}`
        + (beats !== null
          ? `, and it beats it by <span class="value">${pc(beats)}</span> on the frozen probe`
          : '')
        + `. That is one small corpus and one small model, so it is not a recommendation; what it `
        + `is, is evidence that the number was never a constant of nature.`)
    + ` The whole sweep spans <span class="value">${pc(spread)}</span> of probe accuracy, so the `
    + `choice is worth about that much and no more.`);

  set('v-bert',
    `${pc(by.bert.probe)} on the frozen probe and ${pc(by.bert.cloze)} filling a gap, from `
    + `${f3(by.bert.signal_per_token)} supervised predictions per token read.`);
  set('v-bert-watch',
    `It is graded on a fraction of what it reads, so the same amount of text buys it far less `
    + `signal than the causal arm gets.`);
  set('v-gpt',
    `${pc(by.gpt.probe)} on the frozen probe and ${pc(by.gpt.cloze)} on the gap, from `
    + `${f3(by.gpt.signal_per_token)} predictions per token, which is the most of the three.`);
  set('v-gpt-watch',
    `Every representation it builds is missing everything to the right of the word, which is why `
    + `the gap task is the one it does worst.`);
  set('v-t5',
    `${pc(by.t5.probe)} on the frozen probe and ${pc(by.t5.cloze)} on the gap, and it is the only `
    + `arm that both sees both sides and generates.`);
  set('v-t5-watch',
    `The cross-attention has to come out of the same budget, so at equal parameters it is a `
    + `shallower model on each side than the other two.`);

  set('closing-1',
    `The objective is not a detail of the training script. It decided, on one budget and one `
    + `corpus, which of these models can write a sentence at all, which one gets to see the word `
    + `after the gap, and how much supervision each token of text was worth: `
    + `${f3(Math.max(...A.map((a) => a.signal_per_token)))} against `
    + `${f3(Math.min(...A.map((a) => a.signal_per_token)))}. Everything else about them was held `
    + `equal on purpose so that those three sentences could be said.`);

  set('closing-2',
    `One thing has been constant since the attention article and it is the part that scales worst: `
    + `every layer here compares every position with every other, so doubling the context `
    + `quadruples that part of the bill. The last article of this branch goes back to the `
    + `structure attention replaced, a recurrence, and asks whether it can be made to train in `
    + `parallel like a convolution while still running one step at a time like a recurrence. It `
    + `can, exactly, and the page checks the two forms against each other. `
    + `<a href="../state-space/">State space models</a> are next.`);

  set('ref-note',
    `The corpus and split are the previous two articles', unchanged: the Brown corpus cut into `
    + `${n0(M.chunks_train)} training chunks and ${n0(M.chunks_test)} held-out chunks of `
    + `${M.seq} tokens each. Cutting at a fixed length rather than at sentence boundaries is what `
    + `all three of these models really do, and it means a chunk can start mid-sentence; it is `
    + `done identically in all three arms. The vocabulary is capped at the `
    + `${n0(M.vocab_cap)} most frequent words, which covers `
    + `<span class="value">${pc(M.coverage)}</span> of the tokens in the corpus and leaves the `
    + `rest to be read as a single unknown-word symbol. That cap is what makes eight training runs `
    + `affordable, because the output layer is a softmax over the whole vocabulary at every `
    + `position and it, not the depth, is the dominant bill on this page; real models avoid the `
    + `cap with subword vocabularies that never leave anything out. It is applied identically to `
    + `all three arms and to every row of the sweep. The frozen probe is trained on the part-of-speech tags `
    + `of the same corpus, so its column can be read next to the accuracies in the two previous `
    + `articles, keeping in mind that a linear layer on a frozen body is a much weaker thing than `
    + `a model trained for the task. Seed ${M.seed} throughout.`);
}
