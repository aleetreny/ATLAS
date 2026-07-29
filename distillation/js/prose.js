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
  const bbig = BW.rows[BW.rows.length - 1];
  /* Lo que el entrenamiento del origen compra sobre la MISMA arquitectura sin
     entrenar, y lo que el cuello de botella cuesta sobre no tenerlo. Se miden
     por separado porque salieron con signos distintos, que no era el guion. */
  const learned = big.transfer - big.random;
  const bottleneck = big.transfer - big.pixels;
  const beatsPixels = FW.rows.filter((r) => r.transfer > r.pixels).length;
  const beatsRandom = FW.rows.filter((r) => r.transfer > r.random).length;
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
  const soft4 = D.arms['soft T=4'];
  const unlabGain = unlab.mean - soft4.mean;
  const unlabReal = unlabGain > unlab.sd + soft4.sd;

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
    + `${n0(big.n)}. The comparison it is usually reported against is raw pixels into the `
    + `same head, and a win there is taken to mean the encoder learned something worth `
    + `carrying. That comparison cannot distinguish between three different explanations, `
    + `which is why this page runs four controls; and here it does not even go the way it is `
    + `supposed to.`);

  set('transfer-note',
    `So there are four controls rather than one, and on this pair of tasks they are what `
    + `saves the section, because <span class="bold">the transferred encoder loses to the raw `
    + `pixels at ${FW.rows.length - beatsPixels} of the ${FW.rows.length} sample sizes</span>. `
    + `At ${n0(big.n)} labelled rows it scores ${f4(big.transfer)} against ${f4(big.pixels)} `
    + `for feeding the ${28 * 28} pixels straight to the same linear head. Transfer here is `
    + `negative, and the controls say which of the two obvious explanations is right. `
    + `The <span class="bold">untrained</span> arm is the same architecture with the weights `
    + `it was initialised with, frozen, and it scores ${f4(big.random)}: so the source `
    + `training is worth ${signed(learned)} over not having happened, and the encoder as a `
    + `whole still costs ${signed(bottleneck)} against not having one. `
    + `<span class="bold">The bottleneck loses more than the training adds.</span> `
    + `The <span class="bold">shuffled</span> arm, trained to convergence on the source task `
    + `with the labels dealt at random, reached ${f4(FW.shuffled_train)} on its own scrambled `
    + `training set and ${f4(FW.shuffled_acc)} on real source labels, which is memorisation `
    + `rather than structure by both readings, and its features score `
    + `${f4(big.shuffled)}: below the honest encoder, as it should be. The honest encoder for `
    + `comparison reaches ${f4(FW.source_acc)} on the task it was trained for. And the `
    + `<span class="bold">oracle</span>, an encoder of exactly the same shape trained on the `
    + `destination task itself, reaches ${f4(big.oracle)} through a frozen probe and `
    + `${f4(FW.oracle_full)} trained end to end, comfortably above the pixels either way. That `
    + `is the sentence the whole section is for: the ${M.hidden[M.hidden.length - 1]} unit `
    + `bottleneck is not the problem, because a bottleneck filled with the right features `
    + `wins easily. What fails is the assumption that digits and clothing are close enough `
    + `for one to furnish the other. Going the other way is the same story with different `
    + `numbers: at ${n0(bbig.n)} rows, ${f4(bbig.transfer)} transferred against `
    + `${f4(bbig.pixels)} for the pixels and ${f4(bbig.random)} untrained.`);

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
    `${f4(big.transfer)} at ${n0(big.n)} rows against ${f4(big.pixels)} for the raw pixels `
    + `and ${f4(big.oracle)} for an encoder of the same shape trained on this task.`);
  set('verdict-frozen-warn',
    `It beat the pixels at ${beatsPixels} of ${FW.rows.length} sample sizes and beat an `
    + `untrained encoder of the same shape at ${beatsRandom}. The source training is worth `
    + `${signed(learned)}; the bottleneck costs ${signed(bottleneck)}.`);
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
        + `${f4(unlab.sd + soft4.sd)} of seed spread. This is the part of distillation that `
        + `has no equivalent in training on labels.`
      : `${signed(unlabGain)} over the same recipe without them, inside the `
        + `${f4(unlab.sd + soft4.sd)} of seed spread, so here the extra rows did not pay.`);

  set('closing-1',
    `Both halves of this page went against the draft, in the same direction and for related `
    + `reasons. Transfer between these two tasks is <span class="bold">negative</span>: the `
    + `encoder costs ${signed(bottleneck)} against feeding the pixels straight in, even `
    + `though the source training is worth ${signed(learned)} against not having happened, `
    + `and the oracle arm proves the bottleneck itself is fine. And distillation from a `
    + `teacher ${(D.teacher_params / D.student_params).toFixed(0)} times larger is worth `
    + `${signed(distilGain)} over training on the labels directly, which the seed spread `
    + `cannot separate from nothing. Neither method is broken. What both results say is that `
    + `these methods buy something specific: transfer buys features when the source task is `
    + `related, and distillation buys supervision when the student is short of it. The `
    + `student here is ${f4(hard.mean)} against a ceiling of ${f4(ceiling.mean)}, so it is `
    + `not short of anything, and MNIST is not what Fashion-MNIST needed.`);

  set('closing-2',
    `The freezing sweep leaves the question this module has been circling since the first `
    + `page: how few parameters can be allowed to move and still land somewhere useful. `
    + `Here the answer was a head, ${n0(FZ.rows[0].trainable)} of `
    + `${n0(FZ.total)} parameters. <a href="../lora/">The next article</a> asks the same `
    + `question the other way round, by looking at what full fine tuning actually does to a `
    + `weight matrix, and finds that the answer has a shape.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/<wbr>utils/<wbr>generate_<wbr>distillation_<wbr>data.py</span> and the small numpy `
    + `network kit in <span class="mono">src/<wbr>utils/<wbr>tinynet.py</span>, seed ${M.seed}. `
    + `Encoders are ${M.hidden.join(' and ')} hidden units trained for ${M.source_epochs} `
    + `epochs on ${n0(M.source_n)} rows of the source task; probes are logistic regressions `
    + `on the frozen features; fine tuning and training from scratch are ${M.tune_epochs} `
    + `epochs of Adam. Every point in the transfer section is ${M.seeds} seeds and every point `
    + `in the distillation section is ${D.seeds}; the spreads quoted are the population `
    + `spread across them, and with counts this small they are an indication of scale rather `
    + `than an interval. The teacher is ${TE.members} networks of `
    + `${M.teacher_hidden.join(' and ')} units; the student is `
    + `${M.student_hidden.join(' and ')}, trained for ${M.distil_epochs} epochs. The `
    + `${TE.members} teacher members are the ${M.teacher_members} the meta block names and `
    + `carry ${n0(TE.params)} parameters between them, and they agree with each other on `
    + `${(D.teacher_agree * 100).toFixed(1)}% of the test rows, which is the number that says `
    + `how much of the ensemble's advantage is disagreement rather than depth. The transfer `
    + `curves are swept at ${M.sizes.length} sizes from ${M.sizes[0]} to `
    + `${n0(M.sizes[M.sizes.length - 1])} labelled rows. Accuracy is `
    + `on ${n0(M.test_n)} held out rows of the destination task throughout. No wall clocks: `
    + `the only costs quoted are parameter counts.`);
}
