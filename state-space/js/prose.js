/* Every sentence with a measured number, composed from the measurement.
 *
 * The joint tables at the end pull in the numbers the three previous articles
 * shipped, and say so when a file is missing rather than filling the gap from
 * memory. That is the whole reason those articles export their copy-task rows:
 * this one is the place where all five architectures finally share a table.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-problem', 'intro-idea', 'intro-guard',
  'duality-intro', 'duality-verdict',
  'init-intro', 'init-verdict',
  'sel-intro', 'sel-verdict',
  'cost-intro', 'cost-verdict',
  'v-dual', 'v-dual-watch', 'v-init', 'v-init-watch', 'v-sel', 'v-sel-watch',
  'v-cost', 'v-cost-watch',
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

export function initProse(data, seq, attn) {
  const D = data.duality;
  const R = data.reach;
  const S = data.selective;
  const C = data.cost;

  set('intro-problem',
    `Attention bought a path of length one between any two positions, and the bill for it grows `
    + `with the square of the length. The previous articles measured both halves of that: the `
    + `gradient no longer decaying with distance, and the pairwise term overtaking everything else `
    + `once the sequence passes twice the model dimension. If context is going to keep growing, `
    + `something has to give.`);

  set('intro-idea',
    `The way out starts by noticing which part of a recurrence caused the trouble. The problem was `
    + `never the state; it was that the update was <span class="bold">non-linear</span>, so the `
    + `only way to reach step 1,000 was to compute steps 1 to 999 first, and the gradient had to `
    + `come back through a thousand different Jacobians. Take the non-linearity out of the `
    + `recurrence and put it between layers instead, and something changes completely:`);

  set('intro-guard',
    `Those two are the same function. Not similar, not an approximation: expand the recurrence and `
    + `group the terms and the convolution falls out, so the left-hand form can generate one token `
    + `at a time from a state of ${data.demo.n} numbers, and the right-hand form can be evaluated `
    + `for every position at once during training. This page computes both, from the same `
    + `parameters, and reports the worst disagreement it finds. Offline, over `
    + `${D.trials} random systems at each of ${word(D.rows.length)} speeds, that worst `
    + `disagreement is <span class="bold">${D.worst.toExponential(2)}</span>.`);

  set('duality-intro',
    `The left-hand form carries a state of ${data.demo.n} numbers and touches each input once. `
    + `The right-hand form never mentions a state at all: it is a single convolution with a kernel `
    + `as long as the sequence, and that kernel is the layer's entire memory written out. The first `
    + `view below draws it. The second runs both computations and puts them on the same axes.`);

  rows('#duality-table', D.rows, (d) => [
    f3(d.decay), d.worst_relative.toExponential(2), d.median_relative.toExponential(2),
  ]);

  set('duality-verdict',
    `Which is why these models exist at all. Every other architecture in this branch had to pick: `
    + `a recurrence is cheap to run and impossible to parallelise over the sequence, attention is `
    + `the reverse. A linear recurrence is the only one that gets both, and the price it pays for `
    + `them is exactly the linearity. The error grows with how slowly the state forgets, from `
    + `<span class="value">${D.rows[0].worst_relative.toExponential(1)}</span> at a decay of `
    + `${D.rows[0].decay} to <span class="value">${D.rows[D.rows.length - 1].worst_relative.toExponential(1)}</span> `
    + `at ${D.rows[D.rows.length - 1].decay}, which is what accumulating more terms does to `
    + `floating point and not a difference between the two forms.`);

  const good = R.find((r) => !r.unstable && r.hundredth_at !== null
    && r.hundredth_at === Math.max(...R.filter((q) => !q.unstable && q.hundredth_at !== null)
      .map((q) => q.hundredth_at)));
  const bad = R.find((r) => r.unstable);
  set('init-intro',
    `A recurrence that multiplies by the same number every step either forgets quickly or explodes, `
    + `and which one it does is decided before training starts. Below, three ways of initialising `
    + `those multipliers, with everything else identical: a plain gaussian, uniform on the unit `
    + `interval, and decays spread evenly in log space, which is the idea the structured `
    + `initialisations are built on.`);

  rows('#init-table', R, (d) => [
    d.init,
    f3(d.max_abs_a),
    d.unstable ? 'never, it grows' : (d.hundredth_at === null ? 'beyond the range' : `${d.hundredth_at} steps`),
    d.unstable
      ? `diverges to ${d.growth.toExponential(1)} times its start`
      : 'decays',
  ]);

  set('init-verdict',
    `Which is the whole content of the structured initialisations these models are named after. `
    + (bad
      ? `The obvious initialisation, a gaussian, puts at least one multiplier at `
        + `<span class="bold">${f3(bad.max_abs_a)}</span>, and a multiplier above one applied `
        + `repeatedly does what multipliers above one do: that kernel reaches `
        + `<span class="bold">${bad.growth.toExponential(1)}</span> times its starting value `
        + `before the sequence is over. It is worth being careful with the words here, because a `
        + `kernel that does not decay can be the best case or the worst one and they look the same `
        + `in a summary: this one is not remembering, it is exploding.`
      : `None of the three initialisations tried came out unstable, which is not the usual case `
        + `and is what this run measured.`)
    + ` The log-spaced decays reach a hundredth of their starting weight at `
    + `<span class="bold">${good ? `${good.hundredth_at} steps` : 'the far end of the range'}</span>, `
    + `with different channels forgetting at deliberately different rates, so one layer holds `
    + `several timescales at once. Nothing about that is learned. It is chosen, before any data `
    + `arrives, and it is most of why these models work.`);

  set('sel-intro',
    `Everything above is linear and <span class="bold">time-invariant</span>: the same multiplier `
    + `is applied at every step, whatever is being read. That is what makes it a convolution, and `
    + `it is also a real limit, because a system whose dynamics cannot depend on the input cannot `
    + `decide to remember this token and not that one. The copy task the previous articles used `
    + `does not expose it, because the thing to remember is always in the same place, and a fixed `
    + `system can learn "hold on to the first one". The task below moves the targets around and `
    + `marks them, so deciding what to keep requires looking at what is being read. Two lengths `
    + `rather than a curve, and the reason is worth stating: the scan here is written out step by `
    + `step, because the page runs the same one, so the gradient has to come back through every `
    + `step of it and a single model past length 40 takes more than an hour on the machine these `
    + `numbers were measured on. Both rows are trained on an identical budget, which is what the `
    + `comparison needs.`);

  rows('#sel-table', S, (d) => [
    n0(d.length), pc(d.invariant.acc), pc(d.selective.acc),
    pc(d.invariant.exact), pc(d.selective.exact),
  ]);

  const gaps = S.map((r) => r.selective.acc - r.invariant.acc);
  const biggest = Math.max(...gaps);
  const anyGap = gaps.some((v) => v > 0.05);
  const widest = S[gaps.indexOf(biggest)];
  const exactRatio = widest.selective.exact / Math.max(1e-9, widest.invariant.exact);
  set('sel-verdict',
    anyGap
      ? `The gap is the one idea in Mamba that changes what the model <span class="bold">can `
        + `do</span> rather than how fast it does it: at its widest, at length `
        + `${n0(widest.length)}, <span class="bold">${pc(biggest)}</span> more of the marked `
        + `tokens recalled, ${pc(widest.selective.acc)} against ${pc(widest.invariant.acc)}. `
        + `Per token is the kind way to score it. Asking for the whole sequence right, which is `
        + `what copying actually means, the same two are `
        + `<span class="bold">${pc(widest.selective.exact)}</span> and `
        + `<span class="bold">${pc(widest.invariant.exact)}</span>, a factor of `
        + `${f1(exactRatio)}. Making the step size and `
        + `the input and output maps depend on the token being read costs the convolution form, `
        + `because the kernel is no longer the same at every position, and that is the trade the `
        + `whole architecture is organised around: a parallel scan replaces the convolution and `
        + `recovers most of the speed.`
      : `On this task at these lengths the two are closer than expected: the widest gap measured `
        + `is <span class="bold">${pc(biggest)}</span>. The mechanism is real and the task is the `
        + `one designed to expose it, so what this says is that at this scale the fixed system is `
        + `still finding a way, and the honest version of the paragraph is this one rather than `
        + `the one that was planned.`);

  set('cost-intro',
    `Three families, one count of multiplications per sequence, and no clocks anywhere: a wall `
    + `clock measures the machine it ran on, and this branch has been counting operations `
    + `throughout so that its tables mean the same thing on any computer. Model dimension `
    + `${C.d_model}, state size ${C.state}.`);

  rows('#cost-table', C.rows, (d) => [
    n0(d.length), n0(d.attention), n0(d.recurrent), n0(d.ssm),
    `${d.attention_over_ssm.toFixed(1)}x`,
  ]);

  const first = C.rows[0];
  const last = C.rows[C.rows.length - 1];
  set('cost-verdict',
    `At length ${n0(first.length)} attention costs `
    + `<span class="value">${first.attention_over_ssm.toFixed(1)}</span> times the state space `
    + `layer, and at ${n0(last.length)} it costs `
    + `<span class="value">${last.attention_over_ssm.toFixed(1)}</span> times. The ratio grows `
    + `because one of them is quadratic, but notice where it starts: even at short lengths the `
    + `state space layer is cheaper, and that is the constant rather than the exponent. The `
    + `quadratic term only overtakes attention's own linear term past a length of `
    + `<span class="bold">${C.predicted}</span>, which is twice the model dimension and falls `
    + `between two rows of this table: the last one where the projections still dominate and the `
    + `${n0(C.quadratic_takes_over_at)} where they no longer do.`);

  /* --------------------------- la tabla conjunta de la tarea de copia --------- */
  const ds = data.copy_distances;
  const haveSeq = seq && seq.copy;
  const haveAttn = attn && attn.copy;
  const table = ds.map((d) => {
    const row = { d };
    if (haveSeq) {
      ['rnn', 'gru', 'lstm'].forEach((k) => {
        const r = (seq.copy[k] || []).find((x) => x.distance === d);
        row[k] = r ? r.acc : null;
      });
    }
    if (haveAttn) {
      const a = attn.copy.find((x) => x.distance === d);
      row.attn = a ? a.acc : null;
    }
    const s = data.copy.find((x) => x.distance === d);
    row.ssm = s ? s.acc : null;
    return row;
  });
  const cell = (v) => (v === null || v === undefined ? '-' : f3(v));
  rows('#copy-table', table, (r) => [
    n0(r.d), cell(r.rnn), cell(r.gru), cell(r.lstm), cell(r.attn),
    { html: r.ssm === null ? '-' : `<span class="bold">${f3(r.ssm)}</span>` },
  ]);

  /* ---------------------------------------------------------------- veredicto */
  set('v-dual',
    `The two computations of the same layer agree to `
    + `${D.worst.toExponential(2)} relative, across ${word(D.rows.length)} speeds and `
    + `${D.trials} random systems each.`);
  set('v-dual-watch',
    `Linear is the price. Put a non-linearity back inside the recurrence and the convolution form `
    + `disappears, which is why the non-linearity lives between layers instead.`);
  set('v-init',
    good
      ? `Log-spaced decays keep a hundredth of their weight ${good.hundredth_at} steps back, from `
        + `a state of a few dozen numbers.`
      : `The initialisations tried differ in reach by more than an order of magnitude.`);
  set('v-init-watch',
    bad
      ? `The obvious initialisation diverges: a gaussian puts a multiplier at ${f3(bad.max_abs_a)} `
        + `and the kernel reaches ${bad.growth.toExponential(1)} times its start.`
      : `A multiplier above one applied repeatedly diverges, so the initialisation is not free to `
        + `be arbitrary.`);
  set('v-sel',
    anyGap
      ? `Up to ${pc(biggest)} more tokens recalled on the selective copy task, and `
        + `${pc(widest.selective.exact)} of whole sequences against ${pc(widest.invariant.exact)}.`
      : `${pc(biggest)} at its widest on the selective copy task, which is less than the mechanism `
        + `predicts at this scale.`);
  set('v-sel-watch',
    `It gives up the convolution: with input-dependent dynamics the kernel is different at every `
    + `position, so training needs a parallel scan instead.`);
  set('v-cost',
    `${last.attention_over_ssm.toFixed(1)} times cheaper than attention at length `
    + `${n0(last.length)}, and ${first.attention_over_ssm.toFixed(1)} times at ${n0(first.length)}.`);
  set('v-cost-watch',
    `The constant matters more than the exponent until the sequence passes ${C.predicted} tokens, `
    + `which is twice the model dimension.`);

  set('closing-1',
    `This branch started by throwing the order away and has spent six articles putting it back. `
    + `Counting words gave a document a vector and no notion of sequence at all. A topic model `
    + `changed the axes and kept the bag. Word vectors gave each word a position from its company. `
    + `A recurrence read in order and forgot; attention read everything at once and paid for it; `
    + `and this layer is a recurrence again, restricted until it can be computed both ways. Every `
    + `one of those was a trade between what the model can see and what it costs to let it see, `
    + `and every table in the six articles is a measurement of one of those trades.`);

  set('closing-2',
    `That closes the natural language branch of module 4. What comes next in this module is the `
    + `other kind of sequence, the kind where the values are numbers rather than words and the `
    + `question is what happens next rather than what was meant: time series and audio. Much of `
    + `the machinery carries over, including this layer, which arrived in sequence modelling from `
    + `exactly that direction. <a href="../arima/">Putting the rows in order</a> is next, and the `
    + `first thing it takes away is the right to shuffle them.`);

  set('ref-note',
    `The copy task is generated by the same function and the same seeds as in the three previous `
    + `articles, so the rows of the joint table are the same sequences rather than sequences of `
    + `the same kind`
    + (haveSeq && haveAttn ? '.' : ', and the columns whose files are not loaded are left empty '
      + 'rather than filled in from memory.')
    + ` The selective copy task is new here and is described where it is used. Every cost figure `
    + `is a count of multiply-accumulate operations implied by the definitions, not a measurement `
    + `of any machine. Seed ${data.meta.seed} throughout.`);
}
