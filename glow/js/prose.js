/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const A = data.arms;
  const T = data.trap;
  const C = data.compress;
  const O = data.ood;
  const P = data.temperature;
  const B = data.board;
  const N = data.net;
  const full = A.rows.find((r) => r.name === 'full');
  const resolved = A.rows.filter((r) => r.name !== 'full' && r.resolves);
  const best = P.rows.find((r) => r.t === P.best);
  const one = P.rows.find((r) => r.t === 1.0);
  const dd = O.table['digits->digits'];
  const df = O.table['digits->fashion'];
  const fd = O.table['fashion->digits'];
  const ff = O.table['fashion->fashion'];

  set('opening-note',
    `The number this page is about is ${C.flow} bits per pixel: an exact log likelihood, `
    + `divided by ${M.dim} pixels, converted to base two, on ${n0(M.n_digits_test)} held out `
    + `images the model never saw. Beside it, on the same bytes, gzip needs ${C.gzip} and PNG `
    + `needs ${C.png}. `
    + `${C.flow < C.gzip
      ? `So a flow of ${n0(N.count)} numbers beats both, which is the claim this kind of model `
        + `exists to make.`
      : `So a flow of ${n0(N.count)} numbers does not beat either of them, and that is worth `
        + `saying plainly rather than quoting the flow's number on its own: at this size, on `
        + `this data, a program from 1996 compresses these pictures better than the model does.`}`);

  set('scrolly-intro',
    `Bits per pixel is a code length, so the honest way to read one is against other code `
    + `lengths for the same file. Everything below is measured on the same `
    + `${n0(M.n_digits_test)} held out 8 bit images, with the compressors actually run rather `
    + `than quoted.`);

  set('scrolly-step-0',
    `Eight bits a pixel is what the file costs if nobody models anything: ${M.dim} pixels, `
    + `${M.levels} possible values each, written down.`);
  set('scrolly-step-1',
    `Most of those ${M.levels} values almost never occur. The entropy of the pixel histogram, `
    + `which is what an ideal coder would pay knowing only how often each grey appears and `
    + `nothing about where, is ${C.histogram} bits.`);
  set('scrolly-step-2',
    `Learn the histogram separately for each of the ${M.dim} positions, still with no `
    + `relationship between them, and it falls to ${C.per_pixel}. That is the number any model `
    + `worth having must beat, because it took no modelling at all.`);
  set('scrolly-step-3',
    `Then two programs people ship. gzip gets ${C.gzip} bits per pixel on these images and PNG `
    + `gets ${C.png}, both measured by compressing the bytes and counting them.`);
  set('scrolly-step-4',
    `And the flow: ${C.flow}. `
    + `${C.flow < Math.min(C.gzip, C.png)
      ? 'Below both of them, on a model small enough to run in this page.'
      : `Which is ${(C.flow - Math.min(C.gzip, C.png)).toFixed(4)} bits worse than the better of `
        + `the two. A larger flow closes that gap; this one is sized to run in your browser, and `
        + `the page says so instead of quoting the number alone.`}`);

  set('deq-note',
    `The noise in that second expression is not a regulariser and not a trick. Pixel values are `
    + `integers, so the target distribution is a set of spikes, and a continuous density is `
    + `allowed to be arbitrarily tall on spikes. Adding uniform noise turns the spikes into `
    + `boxes of width one level, and then the continuous model's expected log density is a lower `
    + `bound on the discrete model's log likelihood, so the bits it reports are an upper bound on `
    + `a code length. The next section removes the noise and watches what happens.`);

  set('trap-intro',
    `One line removed, everything else identical: the same architecture, the same data, the same `
    + `${M.epochs} epochs, the same seed.`);

  set('trap-note',
    `${T.rows[T.rows.length - 1].bpd < 0
      ? `The point is not that the number is bad. The point is that it is not a number: bits per `
        + `pixel below zero means a code shorter than nothing, and what the model has actually `
        + `found is that a density can be tall wherever the data is a set of points.`
      : `On this data at this size the arm without noise has not yet gone below zero, and this `
        + `page reports where it got to (${T.rows[T.rows.length - 1].bpd}) rather than the `
        + `argument's prediction.`} `
    + `It is also why no likelihood in this module is quoted without saying what it is a `
    + `likelihood of. The <a href="../vae/">variational autoencoder</a> reported a bound on a `
    + `density over continuous pixels, the <a href="../vq-vae/">codebook</a> reported an exact `
    + `count of bits over a discrete code, the <a href="../ebm/">energy model</a> reported a `
    + `number that was not normalised at all, and this article reports bits under a specific `
    + `dequantisation. Four numbers, four different objects.`);

  set('arms-intro',
    `Glow is three pieces, and each one can be removed. Two seeds per arm, the same data, the `
    + `same budget, and the widest seed range in the table becomes the floor every verdict is `
    + `read against.`);

  set('arms-note',
    `${resolved.length
      ? `The changes that move the result by more than the ${A.floor} bit floor are `
        + `${resolved.map((r) => `${r.label} (${r.gap > 0 ? '+' : ''}${r.gap})`).join(', ')}.`
      : `Nothing in the table moves the result by more than the ${A.floor} bit seed floor, which `
        + `is a result about this size of model rather than about the architecture.`} `
    + `The one worth dwelling on is the learned mixing, because it is the piece Glow is named `
    + `for. `
    + `${(() => {
      const p = A.rows.find((r) => r.name === 'permute');
      return p.resolves
        ? `Replacing it with a fixed random permutation costs ${p.gap} bits, which is outside `
          + `the floor, so at this size the learning really is doing something a permutation `
          + `cannot.`
        : `Replacing it with a fixed random permutation costs ${p.gap} bits, which is inside the `
          + `floor: with four channels after a single squeeze there are only ${4 * 3 * 2} `
          + `permutations to choose between, and a learned mixing has very little room to beat `
          + `them. This is a measurement about a small model, and it is exactly the kind of `
          + `result that gets quietly dropped when the ablation is run once.`;
    })()}`);

  set('temp-intro',
    `A flow samples by running backwards, which is one pass rather than a chain, so the pictures `
    + `below are made in this page while you read. The temperature scales the gaussian before it `
    + `goes in: at one it is the model's own distribution, and below one it concentrates on the `
    + `middle of it.`);

  set('temp-note',
    `The judge likes ${P.best} best, at ${best.confidence} against ${one.confidence} at `
    + `temperature one and ${M.judge_confidence} on real held out digits. But the second column `
    + `is the one that stops a bad model looking good: a model that produced one convincing `
    + `digit over and over would score well on confidence and badly on class spread, and at `
    + `${P.best} the spread is ${best.class_entropy} nats where ten classes equally often would `
    + `be ${P.uniform_entropy}, with the most popular taking `
    + `${(best.top_class_share * 100).toFixed(1)}%. Lowering the temperature is not making the `
    + `model better, it is refusing to ask it about the parts of its own distribution it is `
    + `least sure of, and the fraction of pixels that come back outside the range a picture can `
    + `have says the same thing: ${(one.outside * 100).toFixed(1)}% at temperature one against `
    + `${(best.outside * 100).toFixed(1)}% at ${P.best}.`);

  set('ood-intro',
    `And now the question that only a model with an exact likelihood can be embarrassed by. Two `
    + `models, the same architecture, one trained on handwritten digits and one on photographs `
    + `of clothes, both evaluated on both held out sets.`);

  const rowsOod = [
    ['digits', dd, df],
    ['Fashion-MNIST', fd, ff],
  ];
  const tb = document.querySelector('#ood-table');
  if (tb) {
    tb.innerHTML = rowsOod.map(([name, onDig, onFas]) => `<tr>
      <td><span class="bold">${name}</span></td>
      <td>${onDig.mean} <span style="opacity:.7">(${onDig.min} to ${onDig.max})</span></td>
      <td>${onFas.mean} <span style="opacity:.7">(${onFas.min} to ${onFas.max})</span></td>
      <td>${onDig.mean < onFas.mean ? 'the digits' : 'the clothes'}, by
          ${Math.abs(onDig.mean - onFas.mean).toFixed(4)} bits</td>
    </tr>`).join('');
  }

  set('ood-note',
    `${O.gap > 0
      ? `It reproduces. The model trained on clothes spends ${Math.abs(O.gap)} bits per pixel `
        + `FEWER on digits it has never seen than on the held out clothes it was trained for, `
        + `against a seed floor of ${O.floor}. A likelihood that has never seen a digit prefers `
        + `digits.`
      : `It does not reproduce here. The model trained on clothes spends ${Math.abs(O.gap)} bits `
        + `per pixel MORE on digits than on its own held out images, against a seed floor of `
        + `${O.floor}, so on this data at this size the likelihood does prefer what it was `
        + `trained on.`} `
    + `${O.gap * O.reverse_gap < 0
      ? `And the two directions disagree, which is the asymmetry that makes the result strange: `
        + `the digit model spends ${Math.abs(O.reverse_gap)} bits `
        + `${O.reverse_gap > 0 ? 'fewer' : 'more'} on clothes than on digits.`
      : `Both directions point the same way, so whatever this is, it is not the one sided effect `
        + `the literature describes.`} `
    + `The competing explanation is testable and this page tests it: a likelihood may simply `
    + `prefer whatever is easy to compress. gzip needs ${O.gzip_digits} bits per pixel for a `
    + `digit and ${O.gzip_fashion} for a piece of clothing, so the two datasets are not equally `
    + `complex to begin with, and across every held out image of both the model's own bits and `
    + `gzip's correlate at <span class="value">${O.correlation}</span>. `
    + `${Math.abs(O.correlation) > 0.5
      ? `That is most of the story: the likelihood is largely reporting compressibility, which `
        + `is a property of the picture and not of the model.`
      : `That is weaker than the compressibility account needs, so on this data the simple `
        + `explanation does not carry the result either.`}`);

  set('board-intro',
    `Five articles, one module, one set of digits and one two dimensional density, and five `
    + `different objects called a likelihood.`);

  const board = [];
  if (B.vae) {
    board.push(['<a href="../vae/">The variational autoencoder</a>',
      'a lower bound on the log likelihood',
      `bound ${B.vae.toy_bound} against an exact ${B.vae.toy_exact} on the two dimensional `
      + `density, and ${B.vae.digit_mse} squared error on held out digits`,
      'say how loose the bound is, on anything real']);
  }
  if (B.vqvae) {
    board.push(['<a href="../vq-vae/">The codebook</a>',
      'an exact count of bits, and a distortion',
      'a fixed rate per image, with the reconstruction measured at that rate',
      'assign a probability to a picture at all']);
  }
  if (B.ebm) {
    board.push(['<a href="../ebm/">The energy model</a>',
      'an unnormalised score',
      'the only article whose model puts the hole in the ring where the truth does',
      'produce a probability without an integral nobody can compute']);
  }
  if (B.flows) {
    board.push(['<a href="../flows/">The flow</a>',
      'the exact log likelihood, by an identity',
      `${B.flows.toy_exact} nats per held out row, ${B.flows.toy_kl} from the truth`,
      `open a hole: ${B.flows.hole} where the truth puts ${B.flows.truth_hole}`]);
  }
  board.push(['This article',
    'the same identity, on images, in bits',
    `${B.glow.bpd} bits per pixel against gzip's ${B.glow.gzip} and PNG's ${B.glow.png}`,
    'tell you whether a picture came from its own training distribution']);
  const bt = document.querySelector('#board-table');
  if (bt) {
    bt.innerHTML = board.map((r) => `<tr>${r.map((c, i) => (i === 0
      ? `<td><span class="bold">${c}</span></td>` : `<td>${c}</td>`)).join('')}</tr>`).join('');
  }

  set('board-note',
    `Read the last column downwards. Every one of these models is best at something and blind to `
    + `something, and the blindness is never in the loss curve: the bound's looseness, the `
    + `codebook's missing probability, the energy's missing constant, the flow's missing hole `
    + `and this article's missing sense of what it was trained on are all things that had to be `
    + `measured from outside the objective. That is the argument of the whole module, and it is `
    + `why every article in it was fitted to a density whose truth was written down first.`);

  set('verdict-intro',
    `Four questions, and this model answers the first one and loses the fourth.`);
  set('verdict-bits',
    `${C.flow} bits per pixel, exactly, and it is a bound on a real code length because the `
    + `dequantisation makes it one.`);
  set('verdict-bits-warn',
    `Without that noise the number is not a code length at all: the same architecture reports `
    + `${T.rows[T.rows.length - 1].bpd}.`);
  set('verdict-compress',
    `${C.flow < C.gzip
      ? `This one already does: ${C.flow} against gzip's ${C.gzip}.`
      : `Not at ${n0(N.count)} weights: ${C.flow} against gzip's ${C.gzip} and PNG's ${C.png}. `
        + `It does beat the per pixel model at ${C.per_pixel}, which is the line that separates `
        + `modelling from counting.`}`);
  set('verdict-compress-warn',
    `Bits per pixel is only comparable between models that dequantise the same way, and it says `
    + `nothing about whether the samples look like anything.`);
  set('verdict-temp',
    `The judge prefers ${P.best} at ${best.confidence} over ${one.confidence} at one, and the `
    + `class spread stays at ${best.class_entropy} nats, so it is not collapsing.`);
  set('verdict-temp-warn',
    `It is not an improvement to the model, and the sample it refuses to draw is exactly the `
    + `part of its own distribution it was least sure of.`);
  set('verdict-ood',
    `The two by two table says what the likelihood prefers, and its correlation with gzip says `
    + `why: ${O.correlation}.`);
  set('verdict-ood-warn',
    `${O.gap > 0
      ? 'It gives higher likelihood to a dataset it was never trained on, so used as a detector '
        + 'it fails in the most dangerous direction.'
      : 'On this data it does prefer its own training set, but the correlation with '
        + 'compressibility means the ordering is about the pictures rather than about the fit.'}`);

  set('closing-note',
    `So the module ends where it started, at the same question asked five ways. A distribution `
    + `over pictures can be a bound you cannot check, a rate you can count, a score you cannot `
    + `normalise, an identity that cannot make a hole, or a number of bits that turns out to be `
    + `mostly about brightness and compressibility. Every one of those is a real answer, and not `
    + `one of them is the answer.`
    + `<br /><br />`
    + `What none of them do is start from a picture and destroy it slowly, learning only how to `
    + `undo one small step of the damage. That model has no bound, no constant, no invertibility `
    + `requirement and no chain that has to mix, and it is what module 3.3 is about.`);

  set('resources-note',
    `${n0(M.n_digits_train)} MNIST digits to fit on and ${n0(M.n_digits_test)} held out, plus `
    + `${n0(M.n_fashion_train)} and ${n0(M.n_fashion_test)} from Fashion-MNIST, all mean pooled `
    + `to ${M.side} by ${M.side} and rounded to ${M.levels} levels, which is what the `
    + `compressors are given too. The model is one squeeze into ${M.side / 2} by ${M.side / 2} `
    + `by 4 and then ${M.k} steps of actnorm, a learned four by four mixing and an affine `
    + `coupling with ${M.hidden} channels inside it: one level and no split, smaller than the `
    + `paper's, and every ablation is against the same shape. Trained for ${M.epochs} epochs on `
    + `batches of ${M.batch} at a step of ${M.lr}, over seeds ${M.seeds.join(', ')}, with fresh `
    + `dequantisation noise every epoch. The shared judge is the module's, `
    + `${(M.judge_accuracy * 100).toFixed(2)}% on the held out digits. The weights the page runs `
    + `are ${n0(N.count)} numbers in float16, and encoding a held out digit and inverting it `
    + `returns the pixels to ${N.round_trip}. Run on ${M.threads} threads with torch ${M.torch}.`);
}
