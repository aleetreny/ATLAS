/* Every number on this page, composed from the file it was measured into, with
 * a branch on every comparison so a re-run moves the sentence with it. */
const IDS = [
  'opening-note', 'ceiling-intro', 'ceiling-note', 'ladder-intro', 'trade-intro', 'trade-note',
  'live-intro', 'verdict-intro', 'verdict-cost', 'verdict-cost-warn', 'verdict-ceiling',
  'verdict-ceiling-warn', 'verdict-which', 'verdict-which-warn', 'closing-1', 'closing-2',
  'ref-note',
];

/* The sweep harness reads this list to demand that every one of these ids
   ends up saying something other than the placeholder the HTML ships. */
export const PROSE_IDS = IDS;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');

export function initProse(data) {
  const M = data.meta;
  const S = data.shared;
  const T = data.trade;
  const bestLat = data.ladder.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));
  const bestPix = data.pixel_ladder.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));
  const ratio = bestPix.params / bestLat.params;
  const better = Math.abs(bestPix.mmd2) / Math.abs(bestLat.mmd2);
  const smallest = data.ceiling[0];
  const largest = data.ceiling[data.ceiling.length - 1];
  const short = T.filter((r) => !r.at_ceiling);
  const bestK = T.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));

  set('opening-note',
    `The space this article works in is not a new one. It is the latent the `
    + `<a href="../vae/">variational autoencoder article</a> published, and that is checked `
    + `rather than asserted: its own float16 encoder is run here on these digits and has to `
    + `reproduce the codes it published, which it does to ${S.worst} against the quantisation `
    + `floor of ${S.floor} that article measured. Everything below is scored with the same two `
    + `sample statistic, on the same held out digits, against the same floor as `
    + `<a href="../ddpm/">the pixel article</a>, so the two sets of numbers can be put on one `
    + `chart without an argument about what they mean.`);

  set('ceiling-intro',
    `Start with the half of this that has nothing to do with diffusion. Whatever a latent model `
    + `samples, the reader sees the decoder's output, so the model cannot produce anything the `
    + `decoder cannot produce. Encode a real digit and decode it again, with no diffusion `
    + `anywhere, and what comes back is the best that space will ever manage.`);

  set('ceiling-note',
    `${smallest.resolves
      ? `At ${smallest.k} numbers that limit is <span class="bold">${smallest.mmd2}</span>, and `
        + `the pixel article's largest model, on ${n0(bestPix.params)} weights, scored `
        + `${bestPix.mmd2}. `
        + `${Math.abs(smallest.mmd2) > Math.abs(bestPix.mmd2)
          ? `So a latent model in that space loses to it <i>before it is trained</i>. Not because `
            + `diffusion is hard in there, but because the picture has to come back out through a `
            + `two number bottleneck and cannot.`
          : `So even the smallest space here has room to beat it.`}`
      : `Even at ${smallest.k} numbers the round trip is inside the floor, so on this data the `
        + `bottleneck is not costing anything measurable.`} `
    + `${largest.resolves
      ? `At ${largest.k} numbers the ceiling is ${largest.mmd2}, still outside the floor.`
      : `At ${largest.k} numbers the round trip is <span class="bold">inside</span> the floor `
        + `(${largest.mmd2} against ${data.floor.mmd2_max}), which means a perfect sampler in `
        + `that space would be indistinguishable from real digits at this sample size. That is `
        + `the ceiling stopping being a constraint.`}`);

  set('ladder-intro',
    `Now the diffusion, and the comparison the previous article set up. Both curves below count `
    + `<i>everything the model needs to make a picture</i>, so the latent one carries its decoder: `
    + `it would be an easy and dishonest win otherwise. `
    + `${ratio > 2
      ? `Even counted that way, the best latent model here reaches `
        + `<span class="bold">${bestLat.mmd2}</span> on ${n0(bestLat.params)} weights against `
        + `${bestPix.mmd2} on ${n0(bestPix.params)} in pixels: `
        + `<span class="bold">${ratio.toFixed(0)} times fewer weights for ${better.toFixed(1)} `
        + `times the score</span>.`
      : `Counted that way the two are close: ${bestLat.mmd2} on ${n0(bestLat.params)} weights `
        + `against ${bestPix.mmd2} on ${n0(bestPix.params)}.`}`);

  set('trade-intro',
    `Which raises the obvious question, and the answer is not "make the bottleneck as small as `
    + `possible". Two things move when that number changes and they move in opposite directions. `
    + `A smaller space is an easier distribution to learn and a lower ceiling to learn it under. `
    + `A larger one lifts the ceiling and hands the score network a harder job.`);

  set('trade-note',
    `${data.at_ceiling.length && short.length
      ? `The measurement is unambiguous about where the change happens. At `
        + `${data.at_ceiling.join(', ')} numbers the sampler arrives at its own ceiling, so the `
        + `diffusion is doing everything it usefully can and the autoencoder is what holds the `
        + `result down: more weights in the score network buy nothing. At ${short[0].k} it does `
        + `not arrive (${short[0].mmd2} against a ceiling of ${short[0].ceiling_mmd2}), and the `
        + `binding half has swapped. The crossover on this data is between `
        + `${data.at_ceiling[data.at_ceiling.length - 1]} and ${short[0].k}.`
      : data.at_ceiling.length
        ? `Every size measured here arrives at its own ceiling, so on this data the autoencoder `
          + `is the binding half everywhere and the way to a better sample is a better decoder.`
        : `No size measured here arrives at its own ceiling, so on this data the diffusion is the `
          + `binding half throughout and the way to a better sample is a bigger score network.`} `
    + `The practical version of that is a diagnostic rather than a rule: measure your own round `
    + `trip. If your samples score near it, spend on the autoencoder. If they score well short of `
    + `it, spend on the diffusion. It costs one forward pass over held out data and it is the `
    + `only way to know which half you are actually fighting.`);

  set('live-intro',
    `And the thing the previous article could not do. Its digit model was `
    + `${bestPix.kb_half} kB of float16, which is not something to put in a page, so that article `
    + `shipped photographs of its samples. This one ships the model: `
    + `${n0(data.net.count)} numbers carrying a score network, a decoder and an encoder. `
    + `Everything below is computed while you look at it.`);

  set('verdict-intro',
    `Three questions this article can answer with numbers rather than with taste.`);

  set('verdict-cost',
    `${ratio > 2
      ? `${ratio.toFixed(0)} times fewer weights than the same recipe in pixels, decoder included, `
        + `for ${better.toFixed(1)} times the score.`
      : `About the same weights as the same recipe in pixels once the decoder is counted.`}`);
  set('verdict-cost-warn',
    `The saving is real and it is also a ceiling you have accepted. It does not appear in the `
    + `sample statistic until you look for it.`);

  set('verdict-ceiling',
    `The round trip through your autoencoder bounds everything downstream of it: `
    + `${smallest.mmd2} at ${smallest.k} numbers here, ${largest.mmd2} at ${largest.k}.`);
  set('verdict-ceiling-warn',
    `${largest.resolves
      ? 'None of the sizes here has a ceiling inside the floor, so all of them cost something.'
      : `Past ${largest.k} numbers the ceiling stops being measurable at this sample size, which `
        + `is not the same as it not being there.`}`);

  set('verdict-which',
    `Samples near the round trip mean the autoencoder is the limit. Samples well short of it mean `
    + `the diffusion is. Here it is ${data.at_ceiling.length ? data.at_ceiling.join(', ') : 'none'} `
    + `for the first and ${short.length ? short.map((r) => r.k).join(', ') : 'none'} for the second.`);
  set('verdict-which-warn',
    `One forward pass over held out data, and almost nobody reports it.`);

  set('closing-1',
    `The number this article is named for is not hard to compute. It is one encode, one decode `
    + `and the same statistic you were already using, and it tells you which half of a two part `
    + `model is costing you. What makes it worth a page is that it is almost never quoted: a `
    + `latent diffusion result is normally reported as one number, and that number is the sum of `
    + `two very different limitations that call for opposite responses.`);

  set('closing-2',
    `So far every model in this module has been asked to produce something plausible and nothing `
    + `more. Nobody has told one what to produce. The next article keeps this architecture exactly `
    + `as it is and adds a second input, a condition, which the sampler has to obey while still `
    + `producing something that looks real, and the interesting part is what that costs: a model `
    + `that follows an instruction perfectly and a model that produces good pictures are not the `
    + `same model, and the distance between them can be measured. `
    + `<a href="../controlnet/">That is the next article</a>, where the two peaks turn out to sit `
    + `at different settings of the same dial and the more obedient one is not the one you want.`);

  set('ref-note',
    `${n0(M.n_train)} of the ${M.side} by ${M.side} digits the `
    + `<a href="../autoencoders/">bottleneck article</a> published for fitting and ${n0(M.n_test)} `
    + `held out, seed ${M.seed}. The autoencoders are that article's shape unchanged, `
    + `${M.pixels} to ${M.hidden} to k and back, trained for ${M.ae_epochs} epochs on squared `
    + `error, at k of ${M.ks.join(', ')}. The diffusion is the previous article's recipe on a `
    + `cosine schedule over ${M.diff_steps} steps, trained on the latent codes standardised, `
    + `which matters: the forward process pushes towards a standard normal, so a latent at a `
    + `different scale would spend the schedule undoing an offset. Every score is the unbiased `
    + `two sample statistic against ${n0(M.n_test)} held out digits, read against a floor of `
    + `${data.floor.mmd2_max} measured by splitting real digits in two ${data.floor.splits} `
    + `times. The pixel ladder is <a href="../ddpm/">the previous article's</a>, taken from its `
    + `data file rather than re-run. The weights the page runs are ${n0(data.net.count)} numbers `
    + `in float16, checked against a ${data.net.probe.z.length} point probe before anything is `
    + `drawn.`);

  return IDS;
}
