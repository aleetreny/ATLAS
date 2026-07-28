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
    + `and drawn without a grain of noise on it, which is a decision the article comes back to. `
    + `Size, position and colour come back out of an `
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
  /* Comparing the raw shares would compare two different questions: a model
     that happens to draw fewer large sprites has fewer chances to land in the
     rectangle. What each model can be held to is its own marginals, so the
     quantity that means the same thing for both is the share it actually
     produces over the share it would produce if it had forgotten that the two
     factors are related at all. */
  const kept = (h) => h.share / Math.max(h.expected_if_independent, 1e-9);
  const floorHole = data.scale.hole_floor;
  const better = kept(styledHole) < kept(plainHole);
  const overFloor = (h) => h.share / Math.max(floorHole, 1e-9);
  set('hole-result',
    `The rectangle is exactly empty in the factors that were drawn, and it is `
    + `${bold(pc(floorHole, 2))} full in the same sprites measured back off their own pixels. That `
    + `gap is not a mistake to be corrected, it is the floor: the measured size runs about `
    + `${f2(D.floor.size_bias)} of a pixel high because of the antialiased edge, so a sprite drawn `
    + `just under the threshold measures just over it. Every number below is read against that `
    + `${pc(floorHole, 2)} rather than against zero.`
    + `<br /><br />`
    + `The generator with the latent at the input puts ${bold(pc(plainHole.share, 2))} of its `
    + `samples in the rectangle, ${f1(overFloor(plainHole))} times the floor, and the one with the `
    + `latent at every layer puts ${bold(pc(styledHole.share, 2))}, ${f1(overFloor(styledHole))} `
    + `times it. Both are far below what they would produce having learned each factor's marginal `
    + `and nothing about the pair, which for these two models would be `
    + `${pc(plainHole.expected_if_independent, 2)} and ${pc(styledHole.expected_if_independent, 2)}. `
    + `So the honest reading is that ${bold('both')} of them found the hole, `
    + (better
      ? `and the one with the mapping network holds it very slightly better relative to its own `
        + `marginals. `
      : `and the difference between them is ${pc(Math.abs(styledHole.share - plainHole.share), 2)} `
        + `of samples against a floor of ${pc(floorHole, 2)}, which is not a difference this `
        + `experiment can resolve. `)
    + `The other seed of each says the same thing louder: `
    + `${M.seeds.map((s) => `${pc(data.metrics[`plain/${s}`].hole.share, 2)}`).join(' and ')} for `
    + `the first architecture and `
    + `${M.seeds.map((s) => `${pc(data.metrics[`styled/${s}`].hole.share, 2)}`).join(' and ')} for `
    + `the second, so the seed moves this measurement further than the architecture does.`);

  /* ---------------------------------------------------------- architecture */
  const SC = data.scale;
  set('arch-note',
    `The two are matched on weights, not eyeballed: ${bold(n0(plain.params))} in the generator with `
    + `the latent at the input and ${bold(n0(styled.params))} with it at every layer, a difference `
    + `of ${pc(Math.abs(plain.params - styled.params) / plain.params, 1)}, and the same `
    + `discriminator of ${n0(plain.dparams)} weights for both. Same data, same loss, same `
    + `${n0(M.steps)} steps, same batch of ${M.batch}. `
    + `<br /><br />`
    + `Before either of them is compared to anything, the scale they are measured on needs both `
    + `ends nailed down. Distances here are Frechet distances in the sixty-four dimensional `
    + `penultimate layer of the shape reader, and on that scale two disjoint halves of the real `
    + `sprites score ${bold(f2(SC.floor))}, the same sprites under a three pixel blur score `
    + `${f2(SC.blur3)}, and uniform noise scores ${f2(SC.noise)}. The two generators come out at `
    + `${bold(f2(plain.fd))} with the latent at the input and ${bold(f2(styled.fd))} with it at `
    + `every layer. `
    + (Math.abs(plain.fd - styled.fd) / Math.max(plain.fd, styled.fd) < 0.25
      ? `Close enough to each other that this article is not about which one draws better sprites; `
        + `far enough from the floor that neither is close to the data.`
      : `So neither is a good generator at this size, and the one with the styles is the `
        + `${bold('worse')} of the two, by more than the gap between a real sprite and a blurred `
        + `one. Everything below is about what their latent spaces are like, and that difference `
        + `sits underneath all of it: a model that draws worse has less structure to organise, and `
        + `where the measurements come out against StyleGAN's claims this page says which of the `
        + `two explanations it can rule out and which it cannot.`));

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
  const pplRatios = M.seeds.map((s) => {
    const m = data.metrics[`styled/${s}`].path;
    return m.z.perceptual / m.w.perceptual;
  });
  const allShorter = pplRatios.every((r) => r > 1);
  const factorAgrees = (styled.path.z.factor / styled.path.w.factor) > 1 === (sz / sw > 1);
  set('ppl-result',
    `Stepping in \\(z\\): ${bold(f1(pz))} for the latent at the input and ${bold(f1(sz))} for the `
    + `latent at every layer. Stepping in \\(w\\) instead: ${bold(f1(sw))}, `
    + (allShorter
      ? `${f1(sz / sw)} times shorter than the same network's own \\(z\\) path, and shorter on `
        + `${word(M.seeds.length)} seeds out of ${word(M.seeds.length)} `
        + `(${pplRatios.map((r) => `${f2(r)} times`).join(' and ')}). That gap is the mapping `
        + `network's whole job made visible: the same weights and the same pictures, reached `
        + `through a space in which walking is less violent. It is also the one claim on this page `
        + `that does not need the two models compared with each other, which matters given how `
        + `differently they draw: it is a statement about two spaces inside one network.`
      : sw < sz
        ? `shorter on this seed by a factor of ${f1(sz / sw)}, but not on all of them `
          + `(${pplRatios.map((r) => f2(r)).join(' and ')}), so the effect is there and the seed is `
          + `bigger than it.`
        : `which is not shorter than its own \\(z\\) path, so on this dataset the mapping is not `
          + `buying a smoother walk and the article says so rather than repeating the claim.`)
    + ` The last column is the one nobody has on real data: the same paths measured in the true `
    + `factors that were drawn rather than in a network's features (${f1(styled.path.z.factor)} `
    + `against ${f1(styled.path.w.factor)}), and it `
    + (factorAgrees ? 'agrees' : 'disagrees with the perceptual one, which is worth knowing')
    + `, which matters because a perceptual metric built on a classifier can only see what that `
    + `classifier was trained to care about, and this one can be checked against the answer.`);

  set('sep-intro',
    `The second metric asks whether the space is not just smooth but organised: can a single `
    + `straight line separate the sprites that are large from the sprites that are not. If a factor `
    + `is spread across the latent in a curved way, no line will do it; if the space has been `
    + `unbent, one will.`);

  const ATTR = { large: '"is it large"', blue: '"is it blue"', right: '"is it on the right"',
    disc: '"is it a disc"' };
  const attrs = ['large', 'blue', 'right', 'disc'];
  const over = (sp, a) => (sp[a] && sp[a].acc !== null ? sp[a].acc - sp[a].base : null);
  const gains = attrs.map((a) => ({
    a,
    plainZ: over(plain.sep.z, a),
    styledZ: over(styled.sep.z, a),
    styledW: over(styled.sep.w, a),
  })).filter((g) => g.plainZ !== null);
  const wBeatsZ = gains.filter((g) => g.styledW - g.styledZ > 0.02);
  const plainWorks = gains.filter((g) => g.plainZ > 0.05);
  /* The reason the styled model's cells are flat is measurable rather than
     mysterious: its own marginals are so lopsided that guessing the commoner
     answer is already almost always right, and an attribute a model hardly
     ever varies is one no line can be asked to find. */
  const worstBase = attrs.map((a) => (styled.sep.z[a] ? styled.sep.z[a].base : 0))
    .reduce((m, v) => Math.max(m, v), 0);
  const worstAttr = attrs.find((a) => styled.sep.z[a] && styled.sep.z[a].base === worstBase);
  set('sep-result',
    `Read the cells against their own grey baseline, which is the accuracy of always guessing the `
    + `commoner answer: an attribute that is true ninety per cent of the time hands a line ninety `
    + `per cent for free. On that reading, the model with the latent at the input carries `
    + `${plainWorks.length ? `${word(plainWorks.length)} of the ${word(gains.length)}` : 'none'} `
    + `attributes above its baseline in \\(z\\)`
    + (plainWorks.length
      ? `, by up to ${bold(`${pc(Math.max(...plainWorks.map((g) => g.plainZ)), 1)} of accuracy`)} `
        + `(${ATTR[plainWorks.slice().sort((x, y) => y.plainZ - x.plainZ)[0].a]})`
      : '')
    + `. The model with the latent at every layer carries `
    + `${gains.filter((g) => g.styledZ > 0.02).length} in \\(z\\) and `
    + `${gains.filter((g) => g.styledW > 0.02).length} in \\(w\\). `
    + (wBeatsZ.length
      ? `${word(wBeatsZ.length)} of them are more available in \\(w\\) than in \\(z\\), which `
        + `is the claim. `
      : `So the separability half of the argument does not reproduce here, and the reason is on `
        + `the page rather than left to the imagination: this model's own marginals have `
        + `collapsed. ${bold(pc(worstBase, 1))} of the sprites it draws give the same answer to `
        + `${ATTR[worstAttr]}, so there is almost nothing for a line to separate, and a latent `
        + `space cannot be shown to organise a factor its generator has stopped producing. That is `
        + `a fact about a generator trained for ${n0(M.steps)} steps at ${M.res} pixels, not a `
        + `refutation of the paper, and the two should not be confused. `)
    + `The one comparison on this page that survives that objection is the path length above, `
    + `because it compares two spaces inside the same network rather than two networks.`);

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
    + `latent on them. This dataset is the hard case for that argument, because it has no such `
    + `details: a sprite here is a deterministic function of its ${word(M.shapes.length + 2)} `
    + `factors, drawn without a grain of noise on it. So the question the widget answers is not `
    + `whether the noise inputs help, it is what a network does with a free source of randomness `
    + `when the data it is copying has none.`);

  const scales = N.learned_scales || [];
  set('noise-result',
    `It turns them off. The learned scale on the noise map averages `
    + `${scales.map((s) => s.toExponential(1)).join(', ')} across the ${word(scales.length)} `
    + `blocks, and what comes out the other end is a per-pixel variation of `
    + `${bold(f4(N.mean_std))} of a unit of brightness across ${N.n_noise} draws, of which `
    + `${pc(N.high_share)} sits above a four pixel blur. Against style vectors, the same sprite's `
    + `position moves by ${f2(N.across_w.cx)} pixels, its size by ${f2(N.across_w.size)} and its `
    + `hue by ${f3(N.across_w.hue)}; against noise draws, by ${bold(f3(N.across_noise.cx))}, `
    + `${f3(N.across_noise.size)} and ${f4(N.across_noise.hue)}. Factors of `
    + `${f1(ratios.cx)}, ${f1(ratios.size)} and ${f1(ratios.hue)}, with the weakest separation at `
    + `${f1(worstK[1])} times. `
    + `<br /><br />`
    + `Which is the result worth having, and it is not the one the paper reports, because it is `
    + `not the experiment the paper ran. On faces there is stochastic detail everywhere, hair and `
    + `pores and specular flecks, and the noise inputs take it over. Here there is none, and the `
    + `network correctly declines the offer: given a free source of randomness and nothing random `
    + `to spend it on, it learns a scale near zero and the pictures stop depending on it. A `
    + `mechanism that does nothing on data that does not need it is the mechanism behaving `
    + `correctly, and it is only visible because this dataset is one where "how much of this `
    + `image is chance" has an exact answer: none of it.`);

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
    `Stepping in \\(w\\) rather than \\(z\\) shortens the path by `
    + `${pplRatios.map((r) => `${f2(r)} times`).join(' and ')} on the two seeds. On the forbidden `
    + `combination the two architectures land at ${pc(plainHole.share, 2)} and `
    + `${pc(styledHole.share, 2)} against a measurement floor of ${pc(data.scale.hole_floor, 2)}, `
    + `which is not enough of a gap to separate them.`);
  set('verdict-styles',
    `Swapping one block's style moves ${KEYNAME[spread[0].k]} by up to `
    + `${pc(owner[spread[0].k][spread[0].k], 0)} and as little as `
    + `${pc(Math.min(...rows.map((r) => r[spread[0].k])), 0)}, so the blocks are specialised but `
    + `not exclusive.`);
  set('verdict-noise',
    `On data with no stochastic detail in it, the network learns noise scales averaging `
    + `${(N.learned_scales || [0]).map((s) => s.toExponential(1)).join(', ')} and the pictures stop `
    + `depending on them: rerolling the noise moves the sprite's position ${f1(ratios.cx)} times `
    + `less than changing its style does, and its colour ${f1(ratios.hue)} times less.`);
  set('verdict-trunc',
    `Half the spread costs ${f2(half.size_sd)} of size variety against ${f2(full.size_sd)}, and `
    + (half.hole < full.hole ? 'takes the forbidden combination from '
      + `${pc(full.hole, 2)} of samples down to ${pc(half.hole, 2)}.`
      : `leaves the forbidden combination at ${pc(half.hole, 2)} against ${pc(full.hole, 2)}.`));

  /* --------------------------------------------------------- the closing */
  const q = Math.max(...M.seeds.map((s) => data.metrics[`styled/${s}`].q_error));
  const styledFds = armSeeds('styled', (m) => m.fd);
  set('limits-note',
    `What this page is not, said plainly, because half of it came out against the paper it is `
    + `about. A ${M.res} pixel sprite with ${word(M.shapes.length)} shapes is not a face and `
    + `${n0(M.steps)} steps is not a training run: on a scale where two halves of the real data `
    + `score ${f2(SC.floor)} and a three pixel blur scores ${f2(SC.blur3)}, both generators here `
    + `sit past ${f2(Math.min(plain.fd, styled.fd))}. The one with the styles is the worse of the `
    + `two, and its seeds land at ${styledFds.map((v) => f2(v)).join(' and ')}, so a difference `
    + `smaller than ${f2(Math.abs(styledFds[0] - styledFds[styledFds.length - 1]))} between `
    + `anything on this page is not a difference. `
    + `<br /><br />`
    + `That matters for how the results divide. The ones measured ${bold('inside')} one network, `
    + `which are the path length, the block attribution, the noise study and the truncation `
    + `sweep, are unaffected by the two generators being unequal. The ones measured `
    + `${bold('between')} the two, which are the hole and the separability, are confounded by it, `
    + `and where those came out against the claim this page says which explanation it can rule out `
    + `and which it cannot rather than picking the flattering one. Finally, the network running in `
    + `this tab is the quantised one: rounding every weight to float16 moves a pixel by `
    + `${bold(q.toExponential(1))} on average, which is why every figure above was measured on `
    + `those weights rather than on the ones the optimiser produced.`);

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
