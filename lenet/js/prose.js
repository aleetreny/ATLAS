/* Every numbered sentence on this page, written from the measurement.
 *
 * This article is older than the pattern and was typed by hand, which is
 * exactly how a table ended up saying 4.4 s after a rerun had produced a
 * different number. Now nothing with a figure in it is written by hand, and
 * every comparison is composed as a comparison, so that a rerun which moves a
 * result moves the sentence with it instead of leaving it quietly wrong.
 *
 * Training times get one extra rule of their own. A second of wall clock is a
 * property of the machine that measured it, not of the network: the same
 * generator run on a four-core container reported 42.8 s where this one
 * reports 4.4. So a time is never stated as a bare constant here. It is stated
 * against the other network in the same run, which is the part that travels.
 */
const pct = (v) => `${(v * 100).toFixed(2)}%`;
const num = (v) => v.toLocaleString('en-US');

const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
  'eight', 'nine', 'ten', 'eleven', 'twelve'];
/* "six times" reads better than "6 times", but only while the ratio is small
   enough to have a short word. Past twelve, fall back to the figure. */
const word = (v) => {
  const r = Math.round(v);
  return r >= 0 && r < WORDS.length ? WORDS[r] : num(r);
};
/* A composed sentence can start with a composed word, and "six times slower"
   at the head of a table cell has to be capitalised like anything else. */
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function set(id, html) {
  const n = document.querySelector(`#${id}`);
  if (n) n.innerHTML = html;
}

export function initProse(data) {
  const cnn = data.models.cnn;
  const mlp = data.models.mlp;
  const nTest = data.data.n_test;
  const nTrain = data.data.n_train;
  const v = data.versions;

  /* the two networks, and the budget they share */
  set('p-ntrain', num(nTrain));
  set('p-ntest', num(nTest));

  const extraParams = mlp.params - cnn.params;
  set('p-budget',
    `To make the saving concrete rather than theoretical, both networks here were given the `
    + `<i>same budget</i>. The convolutional net has <span class="bold">${num(cnn.params)}</span> `
    + `parameters. The fully connected one was sized to match, with ${mlp.hidden} hidden units and `
    + `<span class="bold">${num(mlp.params)}</span> parameters, ${num(Math.abs(extraParams))} `
    + `${extraParams >= 0 ? 'more' : 'fewer'}. Same data, same optimiser, same ${WORDS[v.epochs] || v.epochs} `
    + `epochs, same seed:`);

  /* the table. The time column is a ratio with a unit attached, never a
     constant to be compared against a reader's own laptop. */
  set('t-cnn-params', num(cnn.params));
  set('t-cnn-acc', pct(cnn.acc));
  set('t-cnn-time', `${cnn.seconds.toFixed(1)} s`);
  set('t-mlp-params', num(mlp.params));
  set('t-mlp-acc', pct(mlp.acc));
  set('t-mlp-time', `${mlp.seconds.toFixed(1)} s`);

  /* errors, which is the same gap counted in a way that hurts */
  const errCnn = Math.round(nTest * (1 - cnn.acc));
  const errMlp = Math.round(nTest * (1 - mlp.acc));
  const removed = (errMlp - errCnn) / errMlp;
  const gap = (cnn.acc - mlp.acc) * 100;
  const slower = cnn.seconds / mlp.seconds;
  set('p-errors',
    `${gap > 0 ? 'A gap of ' : 'A deficit of '}${Math.abs(gap).toFixed(1)} points does not sound like `
    + `much until you count errors instead of successes: <span class="bold">${num(errCnn)}</span> `
    + `mistakes out of ${num(nTest)} against <span class="bold">${num(errMlp)}</span>, which is `
    + `${(removed * 100).toFixed(1)}% of the errors ${removed >= 0 ? 'removed' : 'added'}, with `
    + `${num(Math.abs(extraParams))} ${extraParams >= 0 ? 'fewer' : 'more'} parameters to do it with. `
    + `The structure is worth more than the capacity. It is also ${word(slower)} times slower per run `
    + `on the machine that measured it, which is the honest price and the reason the fully connected `
    + `net was still competitive on small problems for years.`);

  /* what a shift costs each of them */
  const at = (px) => data.shift.find((r) => r.shift === px);
  const s0 = at(0);
  const s1 = at(1);
  const s2 = at(2);
  if (s0 && s1 && s2) {
    const dCnn = (s0.cnn - s1.cnn) * 100;
    const dMlp = (s0.mlp - s1.mlp) * 100;
    set('p-shift-cost',
      `At one pixel the convolutional net loses ${dCnn.toFixed(1)} points and the fully connected one `
      + `loses ${dMlp.toFixed(1)}. At two pixels it is ${pct(s0.cnn)} down to ${pct(s2.cnn)} against `
      + `${pct(s0.mlp)} down to ${pct(s2.mlp)}. So the structure `
      + `${dCnn < dMlp ? 'genuinely helps' : 'does not help here'}, and the feature map in the widget `
      + `shows why: it travelled with the digit instead of arriving at a different set of weights.`);
    set('t-equi-measured',
      `A one-pixel shift costs ${dCnn.toFixed(1)} points instead of the ${dMlp.toFixed(1)} it costs a `
      + `fully connected net.`);
  }

  /* the cliff, where the claim that the curves cross has to be checked and
     not assumed: past the crossing the gap says which way each net fails, and
     if a rerun stopped them crossing the sentence has to stop saying it */
  const s4 = at(4);
  const tail = data.shift.filter((r) => r.shift > 4);
  const crossed = tail.length > 0 && tail.every((r) => r.mlp > r.cnn);
  if (s4) {
    const last = data.shift[data.shift.length - 1].shift;
    set('p-cliff',
      `And then <span class="bold">both of them fall off a cliff</span>. By four pixels the `
      + `convolutional net is at ${pct(s4.cnn)}, which is what guessing would give you, and by `
      + `${WORDS[5] || 5} and ${WORDS[last] || last} `
      + (crossed
        ? `the two curves have crossed: both sit so far below chance that neither is uncertain, they `
          + `are confidently wrong, and the remaining gap says which way each one fails rather than `
          + `which one still recognises a digit.`
        : `both are still below chance, close enough that the remaining gap says which way each one `
          + `fails rather than which one still recognises a digit.`)
      + ` Four pixels, a seventh of the width of the image, is enough to destroy a network that was `
      + `supposed to be translation invariant.`);
  }

  /* the ingredients table, and the augmentation paragraph that follows it */
  const ing = (needle) => data.ingredients.find((r) => r.name.includes(needle));
  const relu = ing('ReLU');
  const drop = ing('dropout');
  const aug = ing('shifts during training');

  if (relu) {
    set('t-relu-without', pct(relu.baseline));
    set('t-relu-with', pct(relu.acc));
    set('t-relu-verdict', relu.acc >= relu.baseline
      ? `Worth ${((relu.acc - relu.baseline) * 100).toFixed(1)} points on this network.`
      : `Backwards. On this network tanh converged sooner and scored higher, by `
        + `${((relu.baseline - relu.acc) * 100).toFixed(1)} points.`);
  }
  if (drop) {
    set('t-drop-without', pct(drop.baseline));
    set('t-drop-with', pct(drop.acc));
    set('t-drop-verdict', drop.acc >= drop.baseline
      ? `Worth ${((drop.acc - drop.baseline) * 100).toFixed(1)} points, but only once the training set `
        + `is small enough to overfit.`
      : `It costs ${((drop.baseline - drop.acc) * 100).toFixed(1)} points here: there was nothing to `
        + `regularise away.`);
  }
  if (aug) {
    const factor = aug.acc / aug.baseline;
    set('t-aug-without', pct(aug.baseline));
    set('t-aug-with', pct(aug.acc));
    set('t-aug-verdict', cap(`${word(factor)} times better on shifted digits, and free on clean ones.`));
    /* The clean-digit control lives in the generator's note, because the whole
       point is that the gain on shifted digits cost nothing on centred ones. */
    const m = /([\d.]+)\s+against\s+([\d.]+)/.exec(aug.note || '');
    const cleanWith = m ? parseFloat(m[1]) : null;
    const cleanWithout = m ? parseFloat(m[2]) : null;
    set('t-aug-measured',
      `${pct(aug.acc)} against ${pct(aug.baseline)} on four-pixel shifts, `
      + (cleanWith !== null && cleanWith >= cleanWithout
        ? 'at no cost on centred digits.'
        : `costing ${((cleanWithout - cleanWith) * 100).toFixed(2)} points on centred digits.`));
    set('p-augment',
      `What does buy it is showing the network the shifts. Retrained with random shifts of up to two `
      + `pixels applied during training, and nothing else changed, the same architecture scores `
      + `<span class="bold">${pct(aug.acc)}</span> on four-pixel shifts instead of ${pct(aug.baseline)}, `
      + `a factor of ${word(factor)}`
      + (cleanWith !== null
        ? `, while ${cleanWith >= cleanWithout ? 'giving up nothing' : 'costing a little'} on centred `
          + `digits (${pct(cleanWith)} against ${pct(cleanWithout)})`
        : '')
      + `. The invariance was never in the model. It was in the data all along.`);
  }

  /* the closing summary table */
  set('t-filters-measured',
    `${pct(cnn.acc)} against ${pct(mlp.acc)} at equal parameter count, `
    + `${removed >= 0.5 ? 'over half the errors' : `${(removed * 100).toFixed(0)}% of the errors`} removed.`);
  set('t-filters-watch', cap(
    `${word(slower)} times slower to train here, and what the filters learn is only as good as what `
    + `the data contains.`));

  /* The saving, counted rather than asserted: one shared bank of filters
     against a dense layer wired to produce the same output map. */
  const { n_filters: nF, k } = data.conv1;
  const side = data.digits.tile;
  const shared = nF * k * k + nF;
  const outUnits = side * side * nF;
  const dense = side * side * outUnits + outUnits;
  set('t-share-measured',
    `${num(shared)} parameters where a fully connected layer of the same shape needs ${num(dense)}.`);

  /* the references block, which repeats the run's settings and so goes stale
     in exactly the same way as the body text */
  set('p-refs-data',
    `Data: MNIST, ${num(nTrain)} training digits and ${num(nTest)} test digits drawn with a fixed `
    + `seed, stratified by class. Both networks trained with Adam at a learning rate of 0.001, batch `
    + `size ${v.batch}, for ${v.epochs} epochs, seed ${v.seed}.`);

  /* the sign-off, where the time gets its caveat spelled out */
  set('p-signoff',
    `Thanks for reading! Every convolution on this page runs in its JavaScript, and the trained `
    + `weights, accuracies and reference feature maps come from `
    + `<code>src/<wbr>utils/<wbr>generate_<wbr>lenet_<wbr>data.py</code>, which trains every network here from scratch `
    + `with seed ${v.seed}, ${v.epochs} epochs and batches of ${v.batch}. The two training times in `
    + `the table above, ${cnn.seconds.toFixed(1)} s and ${mlp.seconds.toFixed(1)} s, are wall clock on `
    + `one laptop CPU: the ${word(slower)}-to-one ratio between them is the part that would survive on `
    + `your machine, the seconds are not.`);
}
