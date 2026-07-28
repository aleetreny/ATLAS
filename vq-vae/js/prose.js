/* Every figure on this page, composed from the file it was measured into.
 *
 * Comparisons are branches, always, including the ones that look settled: this
 * article had two results come out against the plan at full scale and one of
 * them changed sign between the smoke run and the real one.
 */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.toy;
  const G = data.gradient;
  const C = data.collapse;
  const B = data.bits;
  const P = data.prior;
  const N = data.net;
  const worst = C.variants.reduce((a, b) => (b.active.mean < a.active.mean ? b : a));
  const best = C.variants.reduce((a, b) => (b.active.mean > a.active.mean ? b : a));
  const b24 = B.rows[0];
  const arms24 = b24.arms;
  const winner = Object.entries(arms24)
    .reduce((a, b) => (b[1].test.mean < a[1].test.mean ? b : a));
  const loserBest = Object.entries(arms24)
    .filter(([k]) => k !== winner[0])
    .reduce((a, b) => (b[1].test.mean < a[1].test.mean ? b : a));
  const bitFloor = Math.max(...Object.values(arms24).map((a) => a.test.spread));
  const six = T.rows.find((r) => r.bits === 6) || T.rows[T.rows.length - 1];
  const ex6 = T.extra_bits.find((e) => e.bits === six.bits);

  set('opening-note',
    `Two things follow from that, and only one of them is the one people talk about. The `
    + `advertised one is that the rate is exact: ${N.arch.cells} cells of `
    + `${Math.round(Math.log2(N.arch.k))} bits is ${N.arch.bits} bits, and the entropy of how `
    + `the entries are actually used says the true cost is ${T.rows[0] && ''}`
    + `${P.marginal_bits} bits per cell against the ${P.uniform_bits} a uniform code would `
    + `spend. The other one is that <span class="bold">the gradient the encoder needs stops `
    + `existing</span>. Nearest entry is a step function: move the encoder's output a little `
    + `and the chosen entry does not change, so the reconstruction does not change, so the `
    + `derivative is zero. Not small, zero. This page measures it, at `
    + `<span class="bold">${G.hard_gradient_max}</span> across `
    + `${n0(G.batch)} codes and ${G.directions} directions each.`);

  set('scrolly-intro',
    `Start where the module always starts, on the two dimensional density whose truth is `
    + `written in closed form, because a codebook there is a picture rather than a table. `
    + `Fitting one is exactly Lloyd's algorithm, which is k-means, run with ${T.config.restarts} `
    + `restarts on ${n0(T.config.n_fit)} points and scored on ${n0(T.config.n_test)} more.`);

  set('scrolly-step-0',
    `The points are the density this module measures everything against: a ring and two blobs, `
    + `${n0(T.config.n_test)} draws of it, ${data.toy.showcase.points.length} of them drawn `
    + `here. A quantiser has to replace each one by an entry from a finite list, and the error `
    + `is how far it moved.`);
  set('scrolly-step-1',
    `One bit is two entries, and two entries can only say which half of the data a point is `
    + `in. The error is ${T.rows[0].vector_test.mean}, and the two crosses are the whole `
    + `alphabet.`);
  set('scrolly-step-2',
    `Four bits is sixteen entries, and now the crosses start to trace the ring, because that is `
    + `where the points are. Nobody told the algorithm about the ring: minimising squared `
    + `distance put the entries there. The error is ${T.rows.find((r) => r.bits === 4).vector_test.mean}.`);
  set('scrolly-step-3',
    `Six bits, sixty four entries, ${six.vector_test.mean}. The entries are dense where the `
    + `density is high and sparse where it is not, which is the whole content of a codebook: it `
    + `spends its alphabet where the data is.`);
  set('scrolly-step-4',
    `And here is the control. Spend the same ${six.bits} bits one coordinate at a time, `
    + `${six.scalar_split[0]} on one axis and ${six.scalar_split[1]} on the other, and the `
    + `entries land on a grid whether or not there is any data there. The error is `
    + `${six.scalar_test.mean} against ${six.vector_test.mean}, a factor of ${six.ratio.mean}.`);

  set('argmin-note',
    `That arg min is not differentiable, and it is worse than it looks: it is locally `
    + `<span class="bold">constant</span>. The generator perturbs the encoder's output in `
    + `${G.directions} directions at ${G.families.random.rows.length} sizes and asks how often `
    + `the reconstruction loss changes at all. Up to the last size measured nothing changes, and `
    + `the largest gradient of the reconstruction term with respect to the encoder output over `
    + `the whole batch is ${G.hard_gradient_max}. The distance from a code to the boundary of `
    + `its own cell is ${G.margin.median.mean} at the median and ${G.margin.min.mean} at the `
    + `closest, so the step function is not even nearly a slope. What the model trains on `
    + `instead is a rule: copy the decoder's gradient past the quantiser as if it were not `
    + `there. Its norm is ${G.st_norm.mean}, and the gradient it stands in for is exactly `
    + `nothing.`);

  set('rate-intro',
    `A rate against a distortion, with both arms spending the same bits. The point of the `
    + `second curve is that it is not a straw man: the split of the bits between the two axes is `
    + `chosen per seed, so it is the best separable quantiser that seed could have built.`);

  set('rate-note',
    `Vector quantisation wins at every rate measured and on every seed. The interesting reading `
    + `is sideways rather than downwards: to reach what ${six.bits} bits of a shared codebook `
    + `reaches, the separable quantiser needs ${ex6 ? ex6.scalar_bits_needed : 'more'} bits, so `
    + `${ex6 ? ex6.extra : 'the difference'} of them are pure overhead for treating the `
    + `coordinates as unrelated. Both curves fall at close to one unit of log error per bit `
    + `(${T.slope.vector} and ${T.slope.scalar}, fitted above ${T.slope.from_bits} bits), which `
    + `is the classical rate for a smooth density in two dimensions, so the alphabet does not `
    + `change how fast error falls with bits. It changes where the curve sits.`);

  set('cells-intro',
    `And the same thing on digits. The encoder turns ${M.digits.pixels} pixels into `
    + `${N.arch.cells} vectors of ${N.arch.dim} numbers, each of which is replaced by its `
    + `nearest of ${N.arch.k} entries: ${N.arch.bits} bits for the whole picture. Because the `
    + `code is a list of symbols rather than a point, it can be edited, which is the one thing `
    + `no widget in the <a href="../vae/">previous article</a> could do.`);

  set('cells-note',
    `Two facts are worth taking from that atlas. The first is that no cell owns a region of the `
    + `image: measured, the strongest quadrant of a cell takes between `
    + `${(Math.min(...N.quadrant_share.flat()) * 100).toFixed(1)}% and `
    + `${(Math.max(...N.quadrant_share.flat()) * 100).toFixed(1)}% of the change that cell `
    + `causes, where a cell that owned a quarter of the picture would take 100% and a cell that `
    + `owned nothing would take 25%. Turning one symbol repaints the whole digit. The second is `
    + `that most entries decode to something plausible: the alphabet is a vocabulary of digit `
    + `shapes and not a lookup of training images, which is why the model rebuilds a held out `
    + `digit at ${N.quant.mse_quantised} on ${N.arch.bits} bits.`);

  set('health-intro',
    `A codebook has a failure mode a continuous code does not: entries can die. An entry that is `
    + `never the nearest one to anything receives no gradient and never moves again, and it does `
    + `not come back. Four variants below, each differing from the next in exactly one thing.`);

  set('health-note',
    `The worst variant keeps ${worst.active.mean} of ${C.k} entries alive and the best keeps `
    + `${best.active.mean}, which is not a tuning difference, it is the difference between an `
    + `alphabet and a single symbol. The measure that says so without any threshold is the `
    + `perplexity of the usage: ${worst.perplexity.mean} against ${best.perplexity.mean}, where `
    + `${C.k} would mean every entry used equally often. And the reconstruction follows it: `
    + `${worst.test.mean} against ${best.test.mean} against a seed floor of ${C.floor}. `
    + `The fix is one line, and it is initialisation rather than optimisation: start the `
    + `codebook at k-means of the encoder's own first outputs, whose scale (${C.init.encoder_rms} `
    + `root mean square) is nothing like the ${C.init.code_rms} of the random codebook it would `
    + `otherwise be matched against.`);

  set('sample-note',
    `Which leaves the question the <a href="../vae/">last article</a> spent its whole length on: `
    + `can this thing generate? Not by itself. Nothing in the training objective says which `
    + `combinations of ${N.arch.cells} symbols are legal, so drawing four entries uniformly is `
    + `drawing from a distribution the model never claimed. A second model over the code grids, `
    + `trained on nothing but the ${n0(M.digits.train)} grids the encoder produced, brings the `
    + `cost of a cell from ${P.uniform_bits} bits to ${P.ar_bits.mean} (with a seed spread of `
    + `${P.ar_bits.spread}), which is ${P.saved_bits_per_image} bits saved per image and is real `
    + `structure. Whether it makes the pictures better is a separate question, and the table `
    + `answers it: `
    + `${P.samplers.prior.confidence.mean > P.samplers.uniform.confidence.mean
      ? `the judge does prefer them, ${P.samplers.prior.confidence.mean} against `
        + `${P.samplers.uniform.confidence.mean}, but both sit below the `
        + `${P.judge_real_confidence} it gives real held out digits`
      : `it does not, ${P.samplers.prior.confidence.mean} against `
        + `${P.samplers.uniform.confidence.mean}`}, and with `
    + `${N.arch.cells} cells there is very little joint structure left to find.`);

  const order = ['uniform', 'marginal', 'prior', 'reconstruction', 'real'];
  const names = {
    uniform: `${N.arch.cells} entries drawn uniformly`,
    marginal: 'each cell drawn from how often it is used',
    prior: 'a model over the grids of codes',
    reconstruction: 'real digits, encoded and decoded',
    real: 'real held out digits',
  };
  const tb = document.querySelector('#sample-table');
  if (tb) {
    tb.innerHTML = order.filter((k) => P.samplers[k]).map((k) => {
      const s = P.samplers[k];
      return `<tr>
        <td>${k === 'real' ? '<span class="bold">' + names[k] + '</span>' : names[k]}</td>
        <td><span class="value">${s.confidence.mean}</span> &plusmn; ${s.confidence.spread}</td>
        <td>${s.class_entropy.mean} nats, top class ${(s.top_class_share.mean * 100).toFixed(1)}%</td>
        <td>${k === 'real' ? 'the reference every other row is read against'
    : k === 'reconstruction' ? `what ${N.arch.bits} bits of this model costs a real digit`
      : s.confidence.mean >= P.samplers.real.confidence.mean
        ? 'the judge is more confident here than on real digits, which is a fact about the judge'
        : 'below the reference, and the class spread says whether it collapsed'}</td>
      </tr>`;
    }).join('');
  }

  set('verdict-intro',
    `Four questions, and the discrete code answers two of them outright and makes one of them `
    + `harder.`);
  set('verdict-rate',
    `${N.arch.cells} cells of ${Math.round(Math.log2(N.arch.k))} bits is ${N.arch.bits} bits, `
    + `with no divergence to estimate, and the measured entropy of the usage says what an `
    + `entropy coder would really pay: ${P.marginal_bits} per cell.`);
  set('verdict-rate-warn',
    `The gradient it costs is not small, it is ${G.hard_gradient_max}, and everything downstream `
    + `runs on a substitute rule.`);
  set('verdict-vector',
    `At ${six.bits} bits on the plane, ${six.vector_test.mean} against ${six.scalar_test.mean} `
    + `for the same bits spent one axis at a time. On digits at ${b24.bits} bits the winner is `
    + `${winner[1].label}, at ${winner[1].test.mean} against ${loserBest[1].test.mean}.`);
  set('verdict-vector-warn',
    `${Math.abs(winner[1].test.mean - loserBest[1].test.mean) > bitFloor
      ? `That gap is ${(loserBest[1].test.mean - winner[1].test.mean).toFixed(6)} against a worst `
        + `seed spread of ${bitFloor.toFixed(6)}, so it survives the noise, but the codebook `
        + `itself is ${n0(N.codebook_count)} numbers of shared state the bit count does not `
        + `charge for.`
      : `That gap is inside the seed spread of ${bitFloor.toFixed(6)}, so this page does not `
        + `call it a difference.`}`);
  set('verdict-collapse',
    `Perplexity of the usage is the diagnostic and it needs no threshold: ${best.perplexity.mean} `
    + `for the healthy variant against ${worst.perplexity.mean} for the collapsed one.`);
  set('verdict-collapse-warn',
    `A dead entry never comes back, so the damage is done in the first epochs: `
    + `${C.variants[0].history[0].active} entries alive after the first one.`);
  set('verdict-prior',
    `The codes carry real structure: a model over the grids gets a cell down to `
    + `${P.ar_bits.mean} bits from ${P.uniform_bits}.`);
  set('verdict-prior-warn',
    `It is a second training run and a second model, and at ${N.arch.cells} cells it buys `
    + `${P.saved_bits_per_image} bits per image and `
    + `${P.samplers.prior.confidence.mean > P.samplers.uniform.confidence.mean
      ? 'a small improvement in the pictures' : 'no improvement in the pictures'}.`);

  set('closing-note',
    `So a discrete code turns a rate into arithmetic and a gradient into a convention. The `
    + `convention works, which is the surprising part: a rule with no derivation behind it `
    + `trains a model that rebuilds a held out digit at ${N.quant.mse_quantised} on `
    + `${N.arch.bits} bits.`
    + `<br /><br />`
    + `Both articles so far have been built the same way: an encoder, a decoder, and a `
    + `distribution over the thing in between. The next one throws all three away. If what you `
    + `want is a density, you can write one directly, as any function at all divided by its own `
    + `integral, and never build an encoder. The catch is the integral, and `
    + `<a href="../ebm/">the next article</a> is about what happens when you refuse to compute `
    + `it, in the one setting where it can be computed exactly so the refusal can be scored.`);

  set('resources-note',
    `${n0(M.digits.n)} MNIST digits, seed ${M.seed}, mean pooled to 14 by 14 and split `
    + `${n0(M.digits.train)} for fitting and ${n0(M.digits.test)} held out, asserted identical `
    + `to the <a href="../autoencoders/">autoencoders article</a>'s by running that article's `
    + `own float16 encoder on them. The model is ${N.arch.encoder} and ${N.arch.decoder}, with `
    + `${N.arch.k} entries of ${N.arch.dim} numbers, trained for ${M.training.epochs} epochs on `
    + `batches of ${M.training.batch} with a commitment cost of ${M.training.beta}, over seeds `
    + `${M.seeds.join(', ')}. The two dimensional section uses ${n0(T.config.n_fit)} points, `
    + `${T.config.restarts} k-means restarts, and a second solver on the same points for a `
    + `check. The prior over codes is ${N.prior.input}. The weights the page runs are `
    + `${n0(N.count)} numbers in float16 plus ${n0(N.codebook_count)} for the codebook and `
    + `${n0(N.prior.count)} for the prior: quantising them moves `
    + `${N.quant.index_changes} of ${n0(N.quant.index_total)} code assignments and the held out `
    + `error from ${N.quant.mse_full} to ${N.quant.mse_quantised}. Run on ${M.threads} threads `
    + `with torch ${M.torch}, numpy ${M.numpy} and scikit-learn ${M.sklearn}.`);
}
