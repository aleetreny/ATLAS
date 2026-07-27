/* The sentences that carry a number, written from the measurement.
 *
 * The previous article shipped with a training time in its table that a rerun
 * of the generator had already made stale. Nothing here is typed by hand: if
 * the ladder is rerun and a rung moves, every claim below moves with it.
 */

const pct = (v) => `${(v * 100).toFixed(2)}%`;
const pts = (v) => `${(v * 100).toFixed(1)} points`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const L = data.ladder;
  const first = L[0];
  const deepest = L[L.length - 1];
  const peak = L.reduce((a, b) => (b.plain.train_acc > a.plain.train_acc ? b : a));
  const resPeak = L.reduce((a, b) => (b.residual.train_acc > a.residual.train_acc ? b : a));
  const drop = peak.plain.train_acc - deepest.plain.train_acc;
  const gLo = first.plain.grad_first_init;
  const gHi = deepest.plain.grad_first_init;
  const grew = gHi / gLo;

  /* Every verdict below is a comparison, so each one is written from the
     comparison rather than from the result that would be convenient. */
  const turnsDown = drop > 0.002;
  set('verdict-depth',
    turnsDown
      ? `Training accuracy peaks at ${pct(peak.plain.train_acc)} with ${peak.depth} plain layers `
        + `and falls to ${pct(deepest.plain.train_acc)} at ${deepest.depth}.`
      : `Training accuracy rises the whole way, from ${pct(first.plain.train_acc)} at `
        + `${first.depth} layers to ${pct(deepest.plain.train_acc)} at ${deepest.depth}.`);
  set('verdict-depth-warn',
    turnsDown
      ? `Past the peak, more layers made the network worse at the one thing extra capacity `
        + `is supposed to guarantee: fitting its own training set.`
      : `At this scale the depth penalty never arrived, so the ladder here is a demonstration `
        + `of the mechanism and not a reproduction of the failure.`);

  let gradWord;
  if (grew >= 1.2) gradWord = `${grew.toFixed(0)} times larger, not vanishing`;
  else if (grew > 0.8) gradWord = 'barely changed, and certainly not vanishing';
  else if (grew > 0.2) gradWord = `${(1 / grew).toFixed(1)} times smaller, which is nothing like vanishing`;
  else gradWord = `${(1 / grew).toFixed(0)} times smaller, which is the collapse the folklore predicts`;
  set('verdict-grad',
    `At initialisation the gradient reaching the first convolution is `
    + `${gHi.toExponential(2)} at ${deepest.depth} layers against ${gLo.toExponential(2)} at `
    + `${first.depth}: ${gradWord}.`);

  set('verdict-res',
    `${pct(deepest.residual.train_acc)} training accuracy at ${deepest.depth} layers against `
    + `${pct(deepest.plain.train_acc)} without them, for `
    + `${(deepest.residual.params - deepest.plain.params).toLocaleString('en-US')} extra parameters.`);

  /* The objection a careful reader should raise, answered with the three
     controls the generator runs for exactly this purpose. */
  const tail = deepest.plain;
  const spread = data.seed_spread;
  const lr = data.lr_control;
  const bestLr = lr ? lr.reduce((a, b) => (b.train_acc > a.train_acc ? b : a)) : null;

  let note = `Three objections deserve answering before any of this is believed. `
    + `<span class="bold">That the deep network was merely stopped too early:</span> `
    + `the ${deepest.depth}-layer plain network improved its training loss by `
    + `${tail.last_two_epochs.toFixed(4)} over its last two epochs, from `
    + `${tail.history[tail.history.length - 3].toFixed(4)} to `
    + `${tail.history[tail.history.length - 1].toFixed(4)}. It had stopped, and it stopped in a `
    + `worse place.`;

  if (spread) {
    /* Whether the seed spread settles the question or undermines it is a
       comparison, so make it one rather than assuming the flattering answer. */
    const decisive = drop > spread.train_range * 3;
    note += ` <span class="bold">That one seed proves nothing:</span> the same ${spread.depth}-layer `
      + `plain network at three seeds spans ${pts(spread.train_range)} of training accuracy, `
      + `against the ${pts(drop)} the ladder loses between its peak and its deepest rung`
      + (decisive
        ? `, so the fall is far outside what a different draw of the initial weights could produce.`
        : `. Those are close enough that this ladder alone cannot separate the effect from the noise, and the honest reading is that more seeds are needed.`);
  }
  if (bestLr) {
    const beatsPeak = bestLr.train_acc >= peak.plain.train_acc;
    note += ` <span class="bold">That the learning rate was simply wrong for it:</span> the same `
      + `${deepest.depth} layers trained at ${lr.map((r) => r.lr).join(', ')} reach at best `
      + `${pct(bestLr.train_acc)}, at a rate of ${bestLr.lr}`
      + (beatsPeak
        ? `, which does reach the ${pct(peak.plain.train_acc)} of the best shallower network. On this data the depth penalty was a tuning problem after all, and the article says so.`
        : `, still short of the ${pct(peak.plain.train_acc)} a ${peak.depth}-layer network of the same family gets without any tuning at all.`);
  }
  note += ` The residual family, meanwhile, keeps climbing: its best is `
    + `${pct(resPeak.residual.train_acc)} at ${resPeak.depth} layers.`;
  set('closing-note', note);

  /* The promise made at the end of the previous article: that this one would
     retest ReLU where depth is the subject. */
  const acts = data.activations;
  if (acts && acts.length) {
    const byKey = {};
    acts.forEach((r) => { byKey[`${r.depth}-${r.act}`] = r; });
    const depthsSeen = [...new Set(acts.map((r) => r.depth))];
    let reluWins = 0;
    const lines = depthsSeen.map((d) => {
      const relu = byKey[`${d}-relu`];
      const tanh = byKey[`${d}-tanh`];
      if (!relu || !tanh) return '';
      const winner = relu.train_acc > tanh.train_acc ? 'ReLU' : 'tanh';
      if (winner === 'ReLU') reluWins += 1;
      return `at ${d} layers, ReLU reaches ${pct(relu.train_acc)} on the training set and tanh `
        + `${pct(tanh.train_acc)}, so ${winner} wins by `
        + `${pts(Math.abs(relu.train_acc - tanh.train_acc))}`;
    }).filter(Boolean);

    /* The previous article made a prediction in print. Say plainly whether it
       held, including if it did not. */
    let verdict;
    if (reluWins === lines.length) {
      verdict = `The prediction held. What lost by six tenths of a point on two layers wins `
        + `comfortably on twenty and on fifty-six, which is what "it is a story about depth" `
        + `was supposed to mean.`;
    } else if (reluWins === 0) {
      verdict = `The prediction did not hold: tanh is still ahead at both depths measured here. `
        + `The sentence written at the end of the previous article was wrong, and this is it `
        + `being retracted rather than quietly dropped.`;
    } else {
      verdict = `The prediction held at some depths and not others, which is the least tidy `
        + `outcome and the one worth reporting: the crossover is somewhere inside this range.`;
    }

    set('act-note',
      `The previous article measured tanh beating ReLU on a two-layer network and said the `
      + `advantage of ReLU is a story about depth, to be retested here. Same networks, same seeds, `
      + `only the activation swapped: ${lines.join('; ')}. ${verdict}`);
  }
}
