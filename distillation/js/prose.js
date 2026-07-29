/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const signed = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(4)}`;

export const PROSE_IDS = [
  'opening-note', 'transfer-intro', 'transfer-note', 'freeze-intro',
  'distil-intro', 'distil-note', 'temp-intro', 'verdict-intro',
  'verdict-frozen', 'verdict-frozen-warn', 'verdict-tune', 'verdict-tune-warn',
  'verdict-distil', 'verdict-distil-warn', 'verdict-unlab', 'verdict-unlab-warn',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const FW = data.transfer.forward;
  const BW = data.transfer.backward;
  const FZ = data.freeze;
  const D = data.distil;
  const TE = data.teacher;

  const small = FW.rows[0];
  const big = FW.rows[FW.rows.length - 1];
  const bsmall = BW.rows[0];
  const learned = small.transfer - small.random;
  const projection = small.random - small.pixels;
  const softNames = Object.keys(D.arms).filter(
    (k) => k.startsWith('soft T=') && !k.includes('unlabelled'));
  const bestSoft = softNames.reduce((a, b) => (D.arms[b].mean > D.arms[a].mean ? b : a));
  const hard = D.arms['hard labels'];
  const best = D.arms[bestSoft];
  const argmax = D.arms['teacher argmax'];
  const scr = D.arms['scrambled wrong classes'];
  const ceiling = D.arms['student ceiling'];
  const unlab = D.arms['soft T=4 plus unlabelled'];
  const shapeFloor = best.sd + scr.sd;
  const shapeMatters = best.mean - scr.mean > shapeFloor;
  const distilGain = best.mean - hard.mean;
  const distilReal = distilGain > hard.sd + best.sd;
  const unlabGain = unlab.mean - best.mean;
  const unlabReal = unlabGain > unlab.sd + best.sd;

  set('opening-note',
    `Everything here runs on two tables of ${28 * 28} pixel images with ten classes each, `
    + `MNIST and Fashion-MNIST, in both directions, and the networks are plain multilayer `
    + `perceptrons written in numpy: ${M.hidden.join(' and ')} hidden units, `
    + `no convolutions. That limits what can be concluded and the limit is worth stating `
    + `once. Nothing on this page says anything about what a visual hierarchy transfers, `
    + `because there is no hierarchy here. What it can say is what happens to a `
    + `<span class="bold">matrix of weights</span> and to a `
    + `<span class="bold">distribution over classes</span>, which is what both claims on `
    + `this page are actually about.`);

  set('transfer-intro',
    `The experiment is the standard one. Train an encoder on the first task, freeze it, and `
    + `fit a linear head on n labelled rows of the second, sweeping n from ${small.n} to `
    + `${n0(big.n)}. Against raw pixels it wins comfortably at every n, which is the result `
    + `everybody publishes and which does not distinguish between three completely `
    + `different explanations.`);

  set('transfer-note',
    `So there are four controls rather than one. The <span class="bold">untrained</span> arm `
    + `is the same architecture with the weights it was initialised with, frozen: whatever it `
    + `reaches is what a random projection into ${M.hidden[M.hidden.length - 1]} dimensions `
    + `buys, with no learning in it at all. The <span class="bold">shuffled</span> arm was `
    + `trained to convergence on the first task with the labels dealt at random, so it has `
    + `had exactly as much optimiser applied as the transferred one and has learned nothing `
    + `about anything; it reached ${f4(FW.shuffled_train)} on its own scrambled training set, `
    + `which is memorisation and not structure. The <span class="bold">oracle</span> arm is `
    + `an encoder trained on the destination task itself, which is the ceiling. At `
    + `${small.n} labelled rows the transferred encoder is ${f4(small.transfer)} against `
    + `${f4(small.random)} untrained and ${f4(small.pixels)} for raw pixels: `
    + `<span class="bold">${f4(projection)} of the gap over pixels is the projection and `
    + `${f4(learned)} is the learning</span>. And the direction matters: going the other way, `
    + `clothes to digits, the same comparison at ${bsmall.n} rows is ${f4(bsmall.transfer)} `
    + `against ${f4(bsmall.random)} untrained, a difference of `
    + `${f4(bsmall.transfer - bsmall.random)}.`);

  set('freeze-intro',
    `A frozen encoder is one end of a dial. The other is fine tuning everything, and in `
    + `between is the choice of how many layers to let the gradient into, which is usually `
    + `made by folklore. It is one sweep.`);

  set('distil-intro',
    `The second half of the page changes what is inherited. The teacher here is an ensemble `
    + `of ${TE.members} networks scoring ${f4(TE.ensemble_acc)}, its members averaging `
    + `${f4(TE.member_acc.reduce((a, b) => a + b, 0) / TE.member_acc.length)}; the student is `
    + `a single network ${(D.teacher_params / D.student_params).toFixed(0)} times smaller. `
    + `Trained on the ${n0(D.labelled)} labels directly it reaches ${f4(hard.mean)}. Trained `
    + `on the teacher's probabilities instead it reaches ${f4(best.mean)}, `
    + (distilReal
      ? `a gain of ${signed(distilGain)} that is larger than the `
        + `${f4(hard.sd + best.sd)} of seed spread behind it.`
      : `a difference of ${signed(distilGain)} that is inside the `
        + `${f4(hard.sd + best.sd)} of seed spread behind it.`));

  set('distil-note',
    `The explanation always given for why that works is "dark knowledge": the teacher's `
    + `wrong classes are not equally wrong, and a five that looks a bit like a three is `
    + `teaching the student something a one hot label cannot. It is a good story and it is `
    + `testable, because it makes a prediction about two arms nobody runs. `
    + `<span class="bold">Harden</span> the teacher to its argmax and the label correction `
    + `survives while the shape dies: ${f4(argmax.mean)}. `
    + `<span class="bold">Scramble</span> the mass among the wrong classes, keeping the `
    + `winner, its probability and the entropy exactly, and only which wrong class got which `
    + `share is destroyed: ${f4(scr.mean)}. `
    + (shapeMatters
      ? `That is ${f4(best.mean - scr.mean)} below the real distribution against `
        + `${f4(shapeFloor)} of seed spread, so on these two models the shape is carrying `
        + `something of its own and the story survives its own test.`
      : `That is ${f4(Math.abs(best.mean - scr.mean))} from the real distribution against `
        + `${f4(shapeFloor)} of seed spread, which this measurement cannot separate. `
        + `<span class="bold">On these two models the shape of the wrong classes is not `
        + `where the gain lives</span>: what the student is getting is a softer, better `
        + `calibrated target, and the teacher's ordering of its mistakes is not doing the `
        + `work the story assigns to it.`));

  set('temp-intro',
    `Two dials remain, and both are usually copied from the paper that introduced them. `
    + `They are one sweep each, and the sweeps are cheap because the teacher is already `
    + `trained.`);

  set('verdict-intro',
    `Four claims, each with the control that decides it. The right hand column is where most `
    + `of the reading of this page lives: three of these four are only interpretable next to `
    + `something that was measured on purpose to be uninteresting.`);

  set('verdict-frozen',
    `At ${small.n} rows, ${f4(small.transfer)} against ${f4(small.pixels)} for raw pixels; `
    + `by ${n0(big.n)} rows, ${f4(big.transfer)} against ${f4(big.pixels)}.`);
  set('verdict-frozen-warn',
    `An untrained encoder of the same shape already reaches ${f4(small.random)} at `
    + `${small.n} rows, so ${(projection / (small.transfer - small.pixels) * 100).toFixed(0)}% `
    + `of the advantage over pixels is the projection and not the training.`);
  set('verdict-tune',
    `Thawing everything reaches ${f4(FZ.rows[FZ.rows.length - 1].mean)} at ${FZ.n} rows `
    + `against ${f4(FZ.rows[0].mean)} for the head alone.`);
  set('verdict-tune-warn',
    Math.abs(FZ.rows[FZ.rows.length - 1].mean - FZ.rows[0].mean)
      <= FZ.rows[0].sd + FZ.rows[FZ.rows.length - 1].sd
      ? `A difference the seed spreads cannot separate, for `
        + `${(FZ.rows[FZ.rows.length - 1].trainable / FZ.rows[0].trainable).toFixed(0)} times `
        + `the trainable parameters.`
      : `Costs ${(FZ.rows[FZ.rows.length - 1].trainable / FZ.rows[0].trainable).toFixed(0)} `
        + `times the trainable parameters, and every one of them needs an optimiser state.`);
  set('verdict-distil',
    `${f4(best.mean)} against ${f4(hard.mean)} from labels alone, with a student ceiling of `
    + `${f4(ceiling.mean)} when it sees every label there is.`);
  set('verdict-distil-warn',
    shapeMatters
      ? `The scrambling control lands ${f4(best.mean - scr.mean)} below, so the shape of the `
        + `wrong classes is doing measurable work.`
      : `The scrambling control matches it to ${f4(Math.abs(best.mean - scr.mean))}, so `
        + `whatever is being taught here, it is not which wrong class the teacher preferred.`);
  set('verdict-unlab',
    `Adding ${n0(D.unlabelled)} rows the student has no labels for takes it to `
    + `${f4(unlab.mean)}.`);
  set('verdict-unlab-warn',
    unlabReal
      ? `${signed(unlabGain)} over the same recipe without them, larger than the `
        + `${f4(unlab.sd + best.sd)} of seed spread. This is the part of distillation that `
        + `has no equivalent in training on labels.`
      : `${signed(unlabGain)} over the same recipe without them, inside the `
        + `${f4(unlab.sd + best.sd)} of seed spread, so here the extra rows did not pay.`);

  set('closing-1',
    `Both halves of this page came out the same way, and it is worth naming the pattern `
    + `rather than the two results. In each case the headline comparison was comfortably `
    + `won, and in each case a control built to be boring took most of the win away: `
    + `${f4(projection)} of the transfer advantage at ${small.n} rows belongs to an `
    + `untrained projection, and `
    + (shapeMatters
      ? `the scrambled teacher is ${f4(best.mean - scr.mean)} behind the real one, which is `
        + `the part of distillation the story is actually about.`
      : `a teacher whose wrong classes have been shuffled teaches just as well as the real `
        + `one. Neither method is in doubt. What is in doubt is the mechanism each is `
        + `usually explained by, and mechanisms are what you need if you want to know when `
        + `something will stop working.`));

  set('closing-2',
    `The freezing sweep leaves the question this module has been circling since the first `
    + `page: how few parameters can be allowed to move and still land somewhere useful. `
    + `Here the answer was a head, ${n0(FZ.rows[0].trainable)} of `
    + `${n0(FZ.total)} parameters. <a href="../lora/">The next article</a> asks the same `
    + `question the other way round, by looking at what full fine tuning actually does to a `
    + `weight matrix, and finds that the answer has a shape.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/utils/generate_distillation_data.py</span> and the small numpy `
    + `network kit in <span class="mono">src/utils/tinynet.py</span>, seed ${M.seed}. `
    + `Encoders are ${M.hidden.join(' and ')} hidden units trained for ${M.source_epochs} `
    + `epochs on ${n0(M.source_n)} rows of the source task; probes are logistic regressions `
    + `on the frozen features; fine tuning and training from scratch are ${M.tune_epochs} `
    + `epochs of Adam. Every point is ${M.seeds} seeds and the spreads quoted are across `
    + `them. The teacher is ${TE.members} networks of `
    + `${M.teacher_hidden.join(' and ')} units; the student is `
    + `${M.student_hidden.join(' and ')}, trained for ${M.distil_epochs} epochs. Accuracy is `
    + `on ${n0(M.test_n)} held out rows of the destination task throughout. No wall clocks: `
    + `the only costs quoted are parameter counts.`);
}
