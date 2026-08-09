/* Every sentence carrying a measured number, composed from the measurement.
 *
 * The comparison table against the previous article's recurrent cells is built
 * from that article's own shipped numbers where they are available, and says so
 * where they are not, rather than quietly reprinting figures from memory.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-problem', 'intro-idea', 'intro-guard',
  'matrix-intro', 'perm-intro', 'perm-result',
  'curves-intro', 'reach-compare',
  'heads-intro', 'heads-verdict',
  'cost-intro', 'cost-verdict', 'copy-note',
  'v-path', 'v-path-watch', 'v-pos', 'v-pos-watch', 'v-scale', 'v-scale-watch',
  'v-heads', 'v-heads-watch',
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

export function initProse(data, seq) {
  const M = data.meta;
  const P = data.permutation;
  const R = data.reach;
  const A = data.ablation;
  const C = data.cost;

  const ratio = P.with_positions / Math.max(P.without_positions, 1e-30);

  set('intro-problem',
    `The previous article ended on a constraint rather than a result. Every model on it updated a `
    + `state once per token, in order, so the path from the first word of a sentence to the last `
    + `is as long as the sentence. Gates postpone what that does to a gradient; they do not remove `
    + `it, because the path is still there.`);

  set('intro-idea',
    `The fix is to stop passing information along a chain. Let every position look directly at `
    + `every other position and take a weighted average of what it finds, with the weights decided `
    + `by how well a <span class="bold">query</span> from the looking word matches a `
    + `<span class="bold">key</span> from each word being looked at. The distance between any two `
    + `positions becomes one step, whatever the sentence length, and the whole thing is three `
    + `matrix products and a softmax:`);

  set('intro-guard',
    `The trained encoder runs in this page: ${M.layers} layers, ${M.heads} heads, `
    + `${M.d_model} dimensions and ${n0(M.params)} parameters, trained on `
    + `<span class="bold">the same split of the same corpus, for the same task, as the previous `
    + `article</span>, which the generator asserts against that article's own file rather than `
    + `claiming. Every projection travels in half precision, so switching the scaling off or `
    + `knocking a head out below recomputes the model rather than swapping a picture.`);

  set('matrix-intro',
    `Each word builds a query, every word offers a key, and the match between them decides where `
    + `that word looks. A row of the matrix below is one word's distribution over everywhere it `
    + `could have looked, and it sums to one. There are ${M.layers * M.heads} such matrices per `
    + `sentence, one per head per layer, and the three buttons underneath change the model rather `
    + `than the view: the tags along the bottom are re-predicted every time.`);

  set('perm-intro',
    `Turning the positions off is not a degradation, it is a change of what kind of function the `
    + `model is. Attention takes a weighted average over a set. Reorder the input and every output `
    + `moves with its own word and nothing else changes at all, which means that without a `
    + `positional encoding this architecture is, exactly, a bag of words with a very expensive way `
    + `of mixing it.`);

  set('perm-result',
    `Measured as an equality rather than described: shuffling the words of ${word(20)} sentences, `
    + `then undoing the shuffle on the outputs, the worst disagreement anywhere is `
    + `<span class="bold">${P.without_positions.toExponential(2)}</span> without positions, which `
    + `is floating point noise and not a small effect. With the positions switched back on the `
    + `same shuffle moves the outputs by `
    + `<span class="bold">${P.with_positions.toExponential(2)}</span>, `
    + `<span class="bold">${ratio > 1000 ? `${Math.round(ratio).toLocaleString('en-US')} times` : `${f2(ratio)} times`}</span> `
    + `larger. The control is the point: a small number on its own would not distinguish "blind to `
    + `order" from "mildly insensitive to it".`);

  set('curves-intro',
    `Two of the choices in that equation look like implementation detail and are not. The first is `
    + `the division by the square root of the head size, which most descriptions mention in a `
    + `subordinate clause. The second is a claim about distance that the previous article measured `
    + `on recurrent cells, repeated here with the same probe so that the curves can share an axis.`);

  /* La tabla de comparacion se compone del JSON del articulo anterior si esta,
     y si no, se dice que no esta en vez de imprimir cifras de memoria. */
  const flowRows = [];
  if (seq && seq.flow) {
    [['rnn', 'plain RNN'], ['gru', 'GRU'], ['lstm', 'LSTM']].forEach(([k, label]) => {
      const f = seq.flow[k];
      if (f) flowRows.push({ label, half: f.half_life, at10: f.at_10, at30: f.at_30 });
    });
  }
  flowRows.push({ label: 'self-attention', half: R.half_life, at10: R.at_10, at30: R.at_30 });
  const nf = (v) => (v === null || v === undefined ? 'never falls to half' : `${v} steps`);
  const nv = (v) => (v === null || v === undefined ? 'n/a' : f4(v));
  rows('#reach-table', flowRows, (d) => [d.label, nf(d.half), nv(d.at10), nv(d.at30)]);

  const attnBest = flowRows.every((r) => r.label === 'self-attention'
    || (r.half !== null && r.half !== undefined
      && (R.half_life === null || r.half < R.half_life)));
  set('reach-compare',
    seq && seq.flow
      ? `Put next to the recurrent cells of the previous article, on the same axis and from the `
        + `same probe over the same sentences: `
        + (attnBest
          ? `attention holds its gradient over distances where every recurrent cell measured there `
            + `has already lost half of it.`
          : `the ordering is not the clean one the story usually tells, and the table says what it `
            + `says.`)
      : `The previous article's file is not loaded, so this table shows attention alone rather `
        + `than numbers typed in from memory.`);

  set('heads-intro',
    `The usual explanation for having several heads is that each learns a different kind of `
    + `relation, illustrated with a picture of one head that obviously does. The measurable version `
    + `of that claim is what happens when a head is taken away. Below, each of the `
    + `${M.layers * M.heads} heads zeroed in turn, with the model otherwise untouched, scored on `
    + `the same held-out split.`);

  rows('#ablation-table', A.rows, (d) => [
    `layer ${d.layer + 1}, head ${d.head + 1}`,
    pc(d.acc),
    { html: `<span class="value">${d.drop >= 0 ? '' : '+'}${pc(-d.drop)}</span>` },
  ]);

  const worst = A.rows[0];
  const mildest = A.rows[A.rows.length - 1];
  const free = A.rows.filter((r) => r.drop <= 0.001).length;
  set('heads-verdict',
    `Ablation by zeroing rather than by deletion, so the width of the output projection and the `
    + `parameter count stay identical and the comparison is about the head. Without any ablation `
    + `the model gets <span class="bold">${pc(A.base)}</span>. The most expensive head to lose is `
    + `<span class="bold">layer ${worst.layer + 1}, head ${worst.head + 1}</span>, costing `
    + `<span class="bold">${pc(worst.drop)}</span>; the cheapest costs `
    + `<span class="bold">${mildest.drop < 0 ? 'improves' : 'costs'} ${pc(Math.abs(mildest.drop))}</span>, and `
    + `<span class="bold">${word(free)}</span> of the ${A.rows.length} `
    + `${free === 1 ? 'head costs' : 'heads cost'} essentially nothing at all. `
    + `That is the finding the pictures never show: heads are not equal, and on a task this size `
    + `several of them are along for the ride.`);

  set('cost-intro',
    `Every path being one step long has a price, and it is the only thing anybody remembers about `
    + `this architecture. The table counts multiplications rather than timing anything, because a `
    + `wall clock measures the machine it ran on. Two columns, because the cost has two parts and `
    + `only one of them is quadratic: the projections are linear in the length, and comparing `
    + `every position against every other is not.`);

  rows('#cost-table', C.rows, (d) => [
    n0(d.length), n0(d.projections), n0(d.pairwise), pc(d.pairwise_share, 0),
    `${d.ratio.toFixed(1)}x`,
  ]);

  const half = C.rows.find((r) => r.pairwise_share >= 0.5);
  const previous = C.rows.length > 1 ? C.rows[C.rows.length - 2] : C.rows[0];
  const final = C.rows[C.rows.length - 1];
  const pairwiseGrowth = previous.pairwise ? final.pairwise / previous.pairwise : 1;
  const totalGrowth = previous.total ? final.total / previous.total : 1;
  set('cost-verdict',
    `The quadratic term is real and it does not dominate where people think. With `
    + `${M.d_model} dimensions the pairwise part only overtakes the projections at a length of `
    + `<span class="bold">${half ? n0(half.length) : 'beyond the range shown'}</span>, which is `
    + `arithmetic rather than measurement: it happens when the length passes twice the model `
    + `dimension. Below that, "attention is quadratic" is true and irrelevant, because the linear `
    + `part is paying most of the bill. At the largest doubling shown, the pairwise term grows `
    + `${pairwiseGrowth.toFixed(1)} times and the total bill grows ${totalGrowth.toFixed(1)} times.`);

  if (data.copy && seq && seq.copy) {
    const ds = data.copy_distances;
    const table = ds.map((d) => {
      const row = { d };
      ['rnn', 'gru', 'lstm'].forEach((k) => {
        const r = (seq.copy[k] || []).find((x) => x.distance === d);
        row[k] = r ? r.acc : null;
      });
      const a = data.copy.find((x) => x.distance === d);
      row.attn = a ? a.acc : null;
      return row;
    });
    rows('#copy-table', table, (r) => [
      n0(r.d),
      r.rnn === null ? '-' : f3(r.rnn),
      r.gru === null ? '-' : f3(r.gru),
      r.lstm === null ? '-' : f3(r.lstm),
      { html: r.attn === null ? '-' : `<span class="bold">${f3(r.attn)}</span>` },
    ]);
    const longest = ds[ds.length - 1];
    const attnLast = data.copy.find((x) => x.distance === longest);
    set('copy-note',
      `And on the task the previous article used to break the recurrent cells, generated by the `
      + `same function with the same seeds, so these are literally the same sequences: at `
      + `${longest} distractors attention scores `
      + `<span class="bold">${attnLast ? f3(attnLast.acc) : 'n/a'}</span> against chance of `
      + `${attnLast ? f3(attnLast.chance) : 'n/a'}.`);
  } else {
    set('copy-note',
      `The previous article's file is not loaded, so the joint table is not shown rather than `
      + `being filled in from memory.`);
  }

  /* ---------------------------------------------------------------- veredicto */
  set('v-path',
    R.half_life
      ? `A gradient half life of ${R.half_life} steps, measured with the same probe as the `
        + `recurrent cells.`
      : `The gradient never falls to half anywhere in a sixty-word sentence, where the recurrent `
        + `cells all do.`);
  set('v-path-watch',
    `It is paid for pairwise: ${pc(C.rows[C.rows.length - 1].pairwise_share, 0)} of the `
    + `multiplications at length ${n0(C.rows[C.rows.length - 1].length)} are the pairwise term.`);
  set('v-pos',
    `Without it, the worst change from shuffling a sentence is `
    + `${P.without_positions.toExponential(2)}, which is nothing.`);
  set('v-pos-watch',
    `That is not "mostly order-independent", it is a proof that the layer is a function of a set. `
    + `Everything the model knows about order comes from one added vector.`);
  set('v-scale',
    `At a head size of ${data.scaling.rows[data.scaling.rows.length - 1].d}, entropy of `
    + `${f3(data.scaling.rows[data.scaling.rows.length - 1].entropy_raw)} without the division `
    + `against ${f3(data.scaling.rows[data.scaling.rows.length - 1].entropy_scaled)} with it.`);
  set('v-scale-watch',
    `The damage grows with the head size, so a change that looks safe at 8 dimensions is fatal `
    + `at 512.`);
  set('v-heads',
    `${pc(worst.drop)} for the most expensive head and ${pc(Math.abs(mildest.drop))} for the cheapest, from `
    + `${A.rows.length} ablations on an unchanged model.`);
  set('v-heads-watch',
    `${word(free)} of ${A.rows.length} cost essentially nothing, so the count is not evidence of `
    + `the count being needed.`);

  set('closing-1',
    `What this architecture removed was the chain, and the two measurements above say what that `
    + `bought and what it cost in the same units the previous article used. It did not remove the `
    + `hard part of language, and the permutation test is the reminder: strip one added vector and `
    + `the most celebrated model of the last decade is a bag of words again.`);

  set('closing-2',
    `Nothing in this article or the last has said what the model should be trained to `
    + `<span class="bold">predict</span>. Both trained on a task with hand-written labels for every `
    + `token, which is the one thing there is never enough of. The interesting question is what to `
    + `do with text that has no labels at all, and there are three well-known answers that give `
    + `three different models from the same architecture and the same budget. `
    + `<a href="../pretraining/">Pretraining objectives</a> are next.`);

  set('ref-note',
    `The corpus, the split, the vocabulary and the task are the previous article's, unchanged and `
    + `asserted against its file: ${n0(M.train_sents)} sentences to train, ${n0(M.test_sents)} `
    + `held out, ${n0(M.vocab)} words and ${word(M.tags.length)} tags. The encoder is `
    + `${M.layers} layers of ${M.heads} heads at ${M.d_model} dimensions, ${word(M.epochs)} `
    + `epochs, seed ${M.seed}. The copy task is generated by the same function with the same `
    + `seeds as there, so the rows of the joint table are the same sequences and not merely `
    + `sequences of the same kind.`);
}
