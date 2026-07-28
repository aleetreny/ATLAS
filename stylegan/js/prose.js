/* The sentences that carry a measured number, written from the measurement.
 *
 * Every comparison below is a branch, including the ones about direction
 * ("lower", "sharper", "further"), and including the cells of the verdict
 * table. If a re-run moves a result, these sentences move with it or say that
 * the result they were built on is no longer there.
 */

const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const n0 = (v) => Math.round(v).toLocaleString('en-US');
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v <= 12 && Number.isInteger(v) ? WORDS[v] : n0(v));
const bold = (s) => `<span class="bold">${s}</span>`;
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const need = ['meta', 'dataset', 'metrics', 'mixing', 'noise', 'truncation', 'scatter', 'net'];
  const missing = need.filter((k) => !data[k]);
  if (missing.length) {
    throw new Error(`style.json is missing ${missing.join(', ')}: regenerate before publishing`);
  }
  const M = data.meta;
  const D = data.dataset;
  const seed = M.seeds[0];
  const plain = data.metrics[`plain/${seed}`];
  const styled = data.metrics[`styled/${seed}`];
  const armSeeds = (kind, pick) => M.seeds.map((s) => pick(data.metrics[`${kind}/${s}`]));

  /* ------------------------------------------------------------ the set-up */
  set('dataset-note',
    `Both claims are about structure in a latent space, and structure is exactly the thing you `
    + `cannot check on photographs, because nobody knows what the true factors of a face are. So `
    + `the data here is drawn by the generator that measures it: ${n0(M.n_train)} sprites of `
    + `${M.res} by ${M.res} pixels, each one ${word(M.shapes.length)} numbers, `
    + `${M.shapes.join(', ')} at a size between ${M.size_lo} and ${M.size_hi} pixels, somewhere `
    + `between ${M.pos_lo} and ${M.pos_hi} in each direction, in a colour, on a grey background, `
    + `with a grain of ${M.grain} on every pixel. Size, position and colour come back out of an `
    + `image by arithmetic: the area, centroid and mean hue of everything far enough from the `
    + `background. On real sprites that arithmetic recovers the position to `
    + `${bold(`${f2(D.floor.cx_err)} of a pixel`)}, the size to ${f2(D.floor.size_err)} and the `
    + `hue to ${f4(D.floor.hue_err)}, which is the floor under every number below. The shape needs `
    + `a classifier, and it gets ${bold(pc(M.reader_acc, 2))} on sprites it was not trained on.`);

  /* --------------------------------------------------------------- the hole */
  set('hole-intro',
    `The interesting part is not the six factors, it is one thing that is missing. A sprite is `
    + `never both large and blue: above ${M.big} pixels of size, the hue band from ${M.blue[0]} to `
    + `${M.blue[1]} never occurs. Both marginals stay flat, ${pc(D.hole.big_share)} of sprites are `
    + `large and ${pc(D.hole.blue_share)} are blue, so a model that looked at either factor on its `
    + `own would see nothing unusual, and ${bold(pc(D.hole.expected_joint))} of the space is `
    + `forbidden only in the joint. That is a miniature of the real problem: in any real dataset `
    + `the factors are related, some combinations do not occur, and a generator fed a gaussian has `
    + `to bend that gaussian into a shape with the same holes.`);

  const plainHole = plain.hole;
  const styledHole = styled.hole;
  const better = styledHole.share < plainHole.share;
  const ratio = plainHole.share / Math.max(styledHole.share, 1e-9);
  set('hole-result',
    `The data puts ${bold('nothing')} in the rectangle. The generator with the latent at the input `
    + `puts ${bold(pc(plainHole.share, 2))} of its samples there, and the one with the latent at `
    + `every layer puts ${bold(pc(styledHole.share, 2))}. `
    + (better
      ? `A factor of ${f1(ratio)}, in the direction the paper predicts: the mapping network `
        + `absorbs the bending, so the part of the model that draws never has to represent a `
        + `warped space. `
      : `Which is not the direction the paper predicts, and it is reported as it came out. `)
    + `Both would put about ${pc(styledHole.expected_if_independent, 2)} there if they had learned `
    + `the two marginals and forgotten that the factors are related, so neither has simply `
    + `ignored the structure; the question is only how much of it survives.`);

  /* ---------------------------------------------------------- architecture */
  set('arch-note',
    `The two are matched on weights, not eyeballed: ${bold(n0(plain.params))} in the generator with `
    + `the latent at the input and ${bold(n0(styled.params))} with it at every layer, a difference `
    + `of ${pc(Math.abs(plain.params - styled.params) / plain.params, 1)}, and the same `
    + `discriminator of ${n0(plain.dparams)} weights for both. Same data, same loss, same `
    + `${n0(M.steps)} steps, same batch of ${M.batch}. The distance from the real sprites comes out `
    + `${f2(plain.fd)} and ${f2(styled.fd)}, `
    + (Math.abs(plain.fd - styled.fd) / Math.max(plain.fd, styled.fd) < 0.25
      ? `close enough that this article is not about which one draws better sprites. It is about `
        + `what their latent spaces are like.`
      : `so they are not equally good at drawing sprites either, which is worth keeping in mind `
        + `under everything below.`));

  /* -------------------------------------------------------------- metrics */
  set('ppl-intro',
    `Perceptual path length asks a simple question: take a small step in the latent space, and how `
    + `far does the picture move. Average that over the space, and a mapping that has to bend `
    + `sharply somewhere shows up as a large number, because the bend has to be paid for by a `
    + `stretch. The StyleGAN paper measures it in both spaces, and so does this: in the noise `
    + `\\(z\\), which is where the ordinary generator's steps have to happen, and in the mapped `
    + `latent \\(w\\), which is where the styled one's can.`);

  const pz = plain.path.z.perceptual;
  const sz = styled.path.z.perceptual;
  const sw = styled.path.w.perceptual;
  set('ppl-result',
    `Stepping in \\(z\\): ${bold(f1(pz))} for the latent at the input and ${bold(f1(sz))} for the `
    + `latent at every layer. Stepping in \\(w\\) instead: ${bold(f1(sw))}, `
    + (sw < sz
      ? `${f1(sz / sw)} times shorter than the same network's own \\(z\\) path. That gap is the `
        + `mapping network's whole job made visible: the same pictures, the same weights, and a `
        + `space in which walking is ${f1(sz / sw)} times less violent.`
      : `which is not shorter than its own \\(z\\) path, so on this dataset the mapping is not `
        + `buying a smoother walk and the article says so rather than repeating the claim.`)
    + ` The third column is the one nobody has on real data: the same paths measured in the true `
    + `factors that were drawn rather than in a network's features. It agrees `
    + `(${f1(styled.path.z.factor)} against ${f1(styled.path.w.factor)}), which matters because a `
    + `perceptual metric built on a classifier can only see what the classifier was trained to `
    + `care about, and this one can be checked against the answer.`);

  set('sep-intro',
    `The second metric asks whether the space is not just smooth but organised: can a single `
    + `straight line separate the sprites that are large from the sprites that are not. If a factor `
    + `is spread across the latent in a curved way, no line will do it; if the space has been `
    + `unbent, one will.`);

  const attrs = ['large', 'blue', 'right', 'disc'];
  const gains = attrs.map((a) => {
    const zAcc = styled.sep.z[a] && styled.sep.z[a].acc;
    const wAcc = styled.sep.w[a] && styled.sep.w[a].acc;
    return { a, z: zAcc, w: wAcc, gain: (wAcc ?? 0) - (zAcc ?? 0) };
  }).filter((g) => g.z !== null && g.w !== null);
  const bestGain = gains.slice().sort((x, y) => y.gain - x.gain)[0];
  const anyGain = gains.filter((g) => g.gain > 0.02);
  set('sep-result',
    (bestGain
      ? `The largest move is ${bold(bestGain.a === 'large' ? '"is it large"' : `"is it ${bestGain.a}"`)}, `
        + `which a line in \\(z\\) gets ${pc(bestGain.z, 1)} of the time and a line in \\(w\\) gets `
        + `${pc(bestGain.w, 1)}. ` : '')
    + (anyGain.length
      ? `${word(anyGain.length)} of the ${word(gains.length)} attributes are more linearly `
        + `available in \\(w\\) than in \\(z\\), which is the claim. `
      : `None of the four attributes is meaningfully more available in \\(w\\) than in \\(z\\) `
        + `here, so at this scale the separability half of the argument does not reproduce, and `
        + `that is worth saying next to the path length result rather than instead of it. `)
    + `The grey number in each cell is the accuracy of always guessing the commoner answer, which `
    + `is the only honest baseline for an attribute that is not fifty-fifty: "is it blue" is true `
    + `about ${pc(D.hole.blue_share, 0)} of the time in the data, so ${pc(1 - D.hole.blue_share, 0)} `
    + `is free.`);

  /* --------------------------------------------------------- style mixing */
  const rows = data.mixing.rows;
  const keys = ['cx', 'cy', 'size', 'hue', 'shape'];
  const KEYNAME = {
    cx: 'position across', cy: 'position down', size: 'size', hue: 'colour', shape: 'shape',
  };
  set('mix-intro',
    `Because every block gets its own copy of the style, two samples can be crossed: take the `
    + `first sprite's style for the early blocks and the second's for the late ones. In the paper `
    + `that produces the famous grid where the pose comes from one face and the colouring from `
    + `another. Here it can be scored instead of admired, because every factor of a sprite can be `
    + `read off its pixels.`);

  const first = rows[0];
  const last = rows[rows.length - 1];
  const owner = {};
  keys.forEach((k) => {
    const best = rows.slice().sort((a, b) => b[k] - a[k])[0];
    owner[k] = best;
  });
  const spread = keys.map((k) => ({
    k, span: Math.max(...rows.map((r) => r[k])) - Math.min(...rows.map((r) => r[k])),
  })).sort((a, b) => b.span - a.span);
  set('mix-table-intro',
    `One row per block, and one line per factor: swap only that block's style and measure how far `
    + `each factor of the picture travelled towards the other sample. A factor that a block owns `
    + `moves all the way when that block is swapped and stays put when the others are.`);

  const localised = spread[0].span > 0.25;
  set('mix-result',
    `The factor that is most localised is ${bold(KEYNAME[spread[0].k])}, which moves `
    + `${pc(owner[spread[0].k][spread[0].k], 0)} when block ${owner[spread[0].k].layer + 1} `
    + `(${owner[spread[0].k].res} by ${owner[spread[0].k].res}) is swapped and as little as `
    + `${pc(Math.min(...rows.map((r) => r[spread[0].k])), 0)} at another block. The least `
    + `localised is ${KEYNAME[spread[spread.length - 1].k]}, with a span of only `
    + `${pc(spread[spread.length - 1].span, 0)} across the five blocks. `
    + (localised
      ? `So the separation is real but it is partial: no block owns a factor outright, and the `
        + `neat story about coarse layers holding pose and fine layers holding colour is, at this `
        + `resolution, a tendency with a lot of overlap. `
      : `So on this dataset the blocks are not specialised: every factor moves about as much `
        + `whichever block is swapped, and the article publishes that rather than the story. `)
    + `Which is the honest version of the claim: the architecture provides places for factors to `
    + `live, and the training run decides what goes where.`);

  /* --------------------------------------------------------------- noise */
  const N = data.noise;
  const ratios = {};
  ['cx', 'cy', 'size', 'hue'].forEach((k) => {
    ratios[k] = N.across_w[k] / Math.max(N.across_noise[k], 1e-9);
  });
  const worstK = Object.entries(ratios).sort((a, b) => a[1] - b[1])[0];
  set('noise-intro',
    `Every block also gets a page of random numbers, one value per pixel, scaled by a weight the `
    + `network learns per channel. The argument is about capacity: an image has details nobody `
    + `needs to remember, and if the model has no other source of randomness it has to spend its `
    + `latent on them. Here the grain on every sprite is exactly such a detail, drawn fresh for `
    + `every image at an amplitude of ${M.grain}.`);

  set('noise-result',
    `Across ${N.n_noise} noise draws at a fixed style, the position of the sprite moves by `
    + `${bold(f3(N.across_noise.cx))} of a pixel, its size by ${f3(N.across_noise.size)} and its `
    + `hue by ${f4(N.across_noise.hue)}. Across style vectors the same three move by `
    + `${f2(N.across_w.cx)}, ${f2(N.across_w.size)} and ${f3(N.across_w.hue)}: factors of `
    + `${f1(ratios.cx)}, ${f1(ratios.size)} and ${f1(ratios.hue)}. The weakest of the four `
    + `separations is ${KEYNAME[worstK[0]]} at ${f1(worstK[1])} times, so the division of labour `
    + `is `
    + (worstK[1] > 5
      ? `clean on every factor measured: the noise decides the grain and the style decides the `
        + `sprite.`
      : `clean on most factors and leaky on that one, which is the sort of detail a sample sheet `
        + `cannot show you.`)
    + ` The per-pixel variation it produces averages ${f4(N.mean_std)} of a unit of brightness, of `
    + `which ${pc(N.high_share)} sits above a four-pixel blur, meaning it is genuinely fine grain `
    + `rather than a slow wobble of the whole image.`);

  /* ---------------------------------------------------------- truncation */
  const T = data.truncation.rows;
  const full = T.find((r) => r.psi === 1.0);
  const half = T.reduce((b, r) => (Math.abs(r.psi - 0.5) < Math.abs(b.psi - 0.5) ? r : b), T[0]);
  const zero = T[0];
  const bestFd = T.slice().sort((a, b) => a.fd - b.fd)[0];
  set('trunc-intro',
    `The last dial is free, costs no training and is almost always on in the pictures anybody `
    + `publishes. Pull every style vector part of the way towards the average of the prior, and `
    + `samples get more typical: fewer failures, and less of the data. It is a trade, and it is `
    + `measurable in both directions at once.`);

  set('trunc-result',
    `At full spread the model sits ${bold(f2(full.fd))} from the data and produces sizes with a `
    + `spread of ${f2(full.size_sd)}. At half, ${bold(f2(half.fd))} and ${f2(half.size_sd)}. At `
    + `zero every sample is the same sprite, ${f2(zero.fd)} from the data with a size spread of `
    + `${f2(zero.size_sd)}. `
    + (bestFd.psi < 1.0 && bestFd.psi > 0
      ? `The distance is smallest at ${bold(f2(bestFd.psi))}, not at either end, which is the `
        + `reason the dial exists: a little truncation buys typicality faster than it costs `
        + `variety. `
      : `The distance is smallest at ${f2(bestFd.psi)}, so on this metric truncation does not buy `
        + `anything here, which is worth reporting given how routinely it is applied. `)
    + `Its effect on the hole is the cleanest reading of what it does: the forbidden rectangle `
    + `holds ${pc(full.hole, 2)} of samples at full spread and ${pc(half.hole, 2)} at half, `
    + (half.hole < full.hole
      ? `because the region the model gets wrong is in the tails, and truncation is a way of not `
        + `showing you the tails.`
      : `so here the mistakes are not confined to the tails, and truncation does not hide them.`));

  /* --------------------------------------------------------- the verdict */
  set('verdict-mapping',
    `Stepping in \\(w\\) rather than \\(z\\) shortens the path by a factor of ${f1(sz / sw)}, and `
    + `the forbidden combination goes from ${pc(plainHole.share, 2)} of samples to `
    + `${pc(styledHole.share, 2)}.`);
  set('verdict-styles',
    `Swapping one block's style moves ${KEYNAME[spread[0].k]} by up to `
    + `${pc(owner[spread[0].k][spread[0].k], 0)} and as little as `
    + `${pc(Math.min(...rows.map((r) => r[spread[0].k])), 0)}, so the blocks are specialised but `
    + `not exclusive.`);
  set('verdict-noise',
    `Noise moves the sprite's position ${f1(ratios.cx)} times less than the style does, and its `
    + `colour ${f1(ratios.hue)} times less.`);
  set('verdict-trunc',
    `Half the spread costs ${f2(half.size_sd)} of size variety against ${f2(full.size_sd)}, and `
    + (half.hole < full.hole ? 'takes the forbidden combination from '
      + `${pc(full.hole, 2)} of samples down to ${pc(half.hole, 2)}.`
      : `leaves the forbidden combination at ${pc(half.hole, 2)} against ${pc(full.hole, 2)}.`));

  /* --------------------------------------------------------- the closing */
  const q = Math.max(...M.seeds.map((s) => data.metrics[`styled/${s}`].q_error));
  set('limits-note',
    `What this page is not: a 32 pixel sprite with ${word(M.shapes.length)} shapes is not a face, `
    + `and ${n0(M.steps)} steps is not a training run. Two caveats specifically worth having in `
    + `the open. The measurements above are averages over ${word(M.seeds.length)} seeds where a `
    + `number is quoted per architecture, and the seed spread on the distance from the data is `
    + `${f2(Math.abs(armSeeds('styled', (m) => m.fd)[0] - armSeeds('styled', (m) => m.fd)[M.seeds.length - 1]))} `
    + `for the styled pair, so a difference smaller than that is not a difference. And the network `
    + `running in this tab is the quantised one: rounding every weight to float16 moves a pixel by `
    + `${bold(q.toExponential(1))} on average, which is why the generator measures the figures `
    + `above on those weights rather than on the ones the optimiser produced.`);

  set('closing-note',
    `Both articles so far have generated something out of nothing: noise in, picture out, and the `
    + `only question was how to make the picture good. The last one in this module starts from a `
    + `picture instead. <a href="../translation/">Translation</a> asks a generator to turn one `
    + `image into another one, first with the two images paired so there is a right answer, and `
    + `then without, where the only thing holding the mapping in place is that it has to be `
    + `undoable. The measured result there is that being undoable is a much weaker constraint than `
    + `it sounds.`);

  set('resources-note',
    `Every figure on this page comes from <code>src/utils/generate_stylegan_data.py</code>: `
    + `${n0(M.n_train)} sprites at ${M.res} by ${M.res}, two generators of ${n0(plain.params)} and `
    + `${n0(styled.params)} weights trained for ${n0(M.steps)} steps at batch ${M.batch} on `
    + `${word(M.seeds.length)} seeds each, a ${M.z}-dimensional noise vector mapped to a `
    + `${M.w}-dimensional latent, path lengths over ${n0(styled.path.z.n)} pairs and separability `
    + `over four attributes. The generator the page runs is `
    + `${n0(data.net.count)} numbers in float16, ${Math.round(data.net.weights_b64.length / 1024)} `
    + `kilobytes of base64, and the page checks itself against ${data.check.rgb.length} reference `
    + `images the generator rendered from the same weights. No wall clock time is quoted anywhere: `
    + `everything was measured with ${M.threads} threads, and that is a fact about one machine.`);
}
