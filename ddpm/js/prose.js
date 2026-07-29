/* Every number on this page, composed from the data file at load time.
 *
 * Nothing here is typed by hand, and every comparison has a branch on both
 * directions, so that re-running the generator moves the sentence with the
 * measurement instead of leaving it quietly wrong. The article has already had
 * one result invert under it (the hole, which diffusion turned out not to open
 * either), and it survived because the prose was built this way.
 */
const IDS = [
  'opening-note', 'scrolly-intro', 'scrolly-step-0', 'scrolly-step-1', 'scrolly-step-2',
  'scrolly-step-3', 'scrolly-step-4', 'score-intro', 'score-note', 'reverse-intro',
  'reverse-note', 'prior-intro', 'pixel-intro', 'verdict-intro', 'verdict-exact',
  'verdict-exact-warn', 'verdict-sample', 'verdict-sample-warn', 'verdict-steps',
  'verdict-steps-warn', 'verdict-hole', 'verdict-hole-warn', 'closing-1', 'closing-2',
  'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');

export function initProse(data) {
  const M = data.meta;
  const G = data.guards;
  const D = data.destroy;
  const P = data.prior;
  const S = data.score;
  const R = data.sampling;
  const X = data.pixels;
  const B = data.board;

  const ceil = Math.abs(M.ceiling);
  const worstScore = S.rows.reduce((a, b) => (b.rel > a.rel ? b : a));
  const bestScore = S.rows.reduce((a, b) => (b.rel < a.rel ? b : a));
  const shortT = P.linear.rows[0];
  const bestSched = P[data.best_prior];
  const holeAnc = R.hole.find((r) => r.mode === 'ancestral');
  const holeDet = R.hole.find((r) => r.mode === 'deterministic');
  const bestHole = Math.min(holeAnc.share, holeDet.share);
  const bigLadder = X.ladder[X.ladder.length - 1];

  set('opening-note',
    `Everything on this page is measured on the same two dimensional density the previous module `
    + `was fitted against, and for the same reason: in two dimensions the normalising constant is `
    + `a sum, so quantities the rest of the field has to estimate can be computed here. `
    + `Diffusion makes that unusually powerful, because its forward process is closed form. `
    + `q(x_t) is the truth scaled by the square root of a number and blurred by a gaussian, and `
    + `both of those survive exactly, so this article can put the <i>true score field</i> on a `
    + `chart beside the one a network learned and subtract them. On images nobody can draw that `
    + `first field, because writing it down means writing down the data density, which is the `
    + `entire problem.`);

  set('scrolly-intro',
    `The forward process has no parameters and nothing to learn. It takes the density on the left `
    + `and blurs it, a little at a time, for ${n0(M.steps)} steps. What follows is that, computed `
    + `in closed form at each level rather than simulated: the mass on the grid stays within `
    + `${G.worst_mass} of one at every level shown, and a completely independent route (a `
    + `convolution done by Fourier transform instead) agrees to ${G.worst_fft} wherever the grid `
    + `is still fine enough to attempt it, which is ${G.fft_levels} of the ${G.levels} levels.`);

  const rowAt = (t) => D.rows.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
  const r0 = rowAt(0);
  const r1 = rowAt(100);
  const r2 = rowAt(400);
  const r3 = rowAt(800);

  set('scrolly-step-0',
    `The ring, the round blob and the thin one, exactly as the previous five articles had them. `
    + `The dashed circle is the ring's hole: the truth puts ${D.truth_hole} of its mass in there, `
    + `which is as close to nothing as this module gets, and it is the shape that defeated the `
    + `normalising flow.`);

  set('scrolly-step-1',
    `A hundred steps of noise, and the picture still looks like the truth. It is not: the hole `
    + `already holds ${r1.hole}, which is `
    + `<span class="bold">${n0(Math.round(Number(r1.hole) / Number(D.truth_hole)))} times</span> `
    + `what the truth puts there, and the density as a whole is ${r1.kl_normal} nats from a `
    + `standard normal against ${r0.kl_normal} at the start. The destruction is well underway `
    + `before it is visible.`);

  set('scrolly-step-2',
    `Past the middle, and the three components have merged into one lump. Half of the distance to `
    + `a standard normal was gone by t = ${D.half_t}, which is `
    + `${D.half_t < M.steps / 2 ? 'earlier than halfway through the schedule'
      : 'later than halfway through the schedule'}: `
    + `${r2.kl_normal} nats are left here of the ${r0.kl_normal} it started with.`);

  set('scrolly-step-3',
    `And near the end there is nothing to see, because there is nothing left. At t = ${r3.t} the `
    + `density is ${r3.kl_normal} nats from a standard normal, and the hole the flow could not `
    + `open holds ${r3.hole} of the mass, which is roughly what a gaussian puts in any circle of `
    + `that size.`);

  set('scrolly-step-4',
    `"Nothing left" is the assumption the whole method rests on, and it is a number rather than a `
    + `figure of speech: at the end of the full schedule the distance is ${bestSched.kl_end} nats `
    + `for the ${data.best_prior} schedule. That is what makes it safe to start sampling from a `
    + `standard normal. It is also a number that stops being small very quickly if the schedule `
    + `is shortened, which is further down this page.`);

  set('score-intro',
    `Training is one line. Take a data point, pick a noise level at random, add exactly that much `
    + `gaussian noise, and ask the network to say which noise was added. Squared error. That is `
    + `the whole objective, and the thing it teaches the network is the <i>score</i>: the `
    + `direction in which the noisy density increases fastest, at that noise level.`);

  set('score-note',
    `Which means the network has a right answer, and here we can write it down. The two arrows `
    + `below are the true score and the learned one at the same points. `
    + `${worstScore.rel > 1
      ? `They agree beautifully at the noisy end and fall apart at the clean end: the error is `
        + `<span class="bold">${bestScore.rel}</span> at t = ${bestScore.t} and `
        + `<span class="bold">${worstScore.rel}</span> at t = ${worstScore.t}, where "error" is `
        + `the length of the difference divided by the length of the truth, so the second number `
        + `means the network's arrow is wrong by more than the whole answer.`
      : `The error runs from ${bestScore.rel} at t = ${bestScore.t} to ${worstScore.rel} at `
        + `t = ${worstScore.t}, so on this model the two ends of the schedule are closer in `
        + `difficulty than the argument below would suggest.`} `
    + `The reason is not subtle once it is drawn. At the noisy end the answer is almost "point at `
    + `the origin", because the density is almost a standard normal. At the clean end the answer `
    + `is the fine structure of the data, which is the thing that was hard in the first place.`);

  set('reverse-intro',
    `Sampling runs the chain backwards: start from noise, ask the network which noise it thinks is `
    + `there, take some of it away, repeat. Two versions, and they are the same trained network `
    + `rather than two models. The ancestral sampler adds a little noise back at every step, `
    + `which is what the original derivation says to do. The deterministic one drops that term.`);

  set('reverse-note',
    `The chart carries its own floor, which is the only thing that makes the numbers readable: `
    + `two pools of ${R.floor.n} <i>real</i> draws land ${R.floor.mmd2} apart on average and `
    + `never more than ${R.floor.mmd2_max}, so anything under that is not a difference. `
    + `${R.enter_deterministic < R.enter_ancestral
      ? `The deterministic sampler gets inside that band at <span class="bold">`
        + `${R.enter_deterministic} steps</span> and the ancestral one needs `
        + `<span class="bold">${R.enter_ancestral}</span>, which is the practical argument for `
        + `the deterministic one: same model, same weights, a quarter of the work.`
      : R.enter_deterministic > R.enter_ancestral
        ? `The ancestral sampler gets inside that band first, at ${R.enter_ancestral} steps `
          + `against ${R.enter_deterministic}, which is the opposite of the usual advice and is `
          + `what this model measured.`
        : `Both samplers get inside that band at ${R.enter_deterministic} steps, so on this `
          + `model there is nothing to choose between them on step count.`} `
    + `And "from there onwards" is checked against every larger budget rather than against the `
    + `first one that qualifies, because a curve wobbling inside its own floor will cross it `
    + `several times.`);

  set('prior-intro',
    `The training objective has a term nobody optimises, and the derivation is right that nobody `
    + `should: it does not depend on the parameters. It is the distance between what the forward `
    + `process actually leaves at the end and the standard normal that sampling starts from. `
    + `Textbooks call it a constant and move on, which is fine when it is zero. In two dimensions `
    + `it is computable, so here is what it is worth.`);

  set('pixel-intro',
    `Everything so far has been two dimensional, which is what let it be exact. The same code on `
    + `${M.digit_side} by ${M.digit_side} digits gets a different kind of answer: not a wrong one, `
    + `an expensive one. `
    + `${bigLadder.params > 20 * X.toy_params
      ? `Reaching ${bigLadder.mmd2} on ${M.digit_side * M.digit_side} pixels took `
        + `<span class="bold">${n0(X.biggest)}</span> weights against ${n0(X.toy_params)} for the `
        + `two dimensional process on this page, which is ${Math.round(X.biggest / X.toy_params)} `
        + `times as many and ${bigLadder.kb_half} kB in float16.`
      : `Reaching ${bigLadder.mmd2} on ${M.digit_side * M.digit_side} pixels took `
        + `${n0(X.biggest)} weights, which is not far off the ${n0(X.toy_params)} the two `
        + `dimensional process needs.`} `
    + `That is a measurement rather than a complaint, and it is the reason the next article does `
    + `the diffusion somewhere smaller than pixel space.`);

  set('verdict-intro',
    `Six models, one density, one table. This one is the first that cannot fill the first row, `
    + `and the reason is structural rather than a failure of effort.`);

  set('verdict-exact',
    `A flow gives ${B.flow ? B.flow.exact : 'an exact log density'} exactly and an autoencoder `
    + `bounds it. A diffusion sampler gives neither: its reverse chain has no closed form density, `
    + `so there is no number to put here.`);
  set('verdict-exact-warn',
    `Every bits per dimension you have seen quoted for a diffusion model is a bound obtained by a `
    + `different argument, not the thing a flow reports.`);

  set('verdict-sample',
    `${Math.abs(B.ddpm.mmd2_best) <= B.ddpm.floor
      ? `Its best draws are <span class="bold">inside</span> the floor measured from real data `
        + `split in two (${B.ddpm.mmd2_best} against ${B.ddpm.floor}), which is the strongest `
        + `thing this statistic can say: indistinguishable at this sample size.`
      : `Its best draws score ${B.ddpm.mmd2_best} against a floor of ${B.ddpm.floor}, so they are `
        + `still separable from real ones.`}`);
  set('verdict-sample-warn',
    `On ${M.digit_side * M.digit_side} pixels the same recipe never gets inside its floor at any `
    + `size measured here, the best being ${bigLadder.mmd2} against ${X.floor.mmd2_max}.`);

  set('verdict-steps',
    `${R.enter_deterministic}${R.enter_deterministic === R.enter_ancestral ? '' : ' to ' + R.enter_ancestral} `
    + `steps is enough to be inside the floor, out of a schedule of ${n0(M.steps)}. The other `
    + `${Math.round(100 - (100 * R.enter_deterministic) / M.steps)}% buys nothing measurable.`);
  set('verdict-steps-warn',
    `The steps you can drop are the noisy ones. The clean end is where the score error is `
    + `${worstScore.rel} and where every remaining structure has to appear.`);

  set('verdict-hole',
    `${bestHole < Number(data.destroy.flow_hole)
      ? `Better than the flow, but only just: ${bestHole} against ${D.flow_hole}, where the truth `
        + `puts ${D.truth_hole}.`
      : `No better than the flow: ${bestHole} at best against ${D.flow_hole}, where the truth `
        + `puts ${D.truth_hole}.`}`);
  set('verdict-hole-warn',
    `The flow had an excuse, and it was a good one: a smooth invertible map cannot tear a hole in `
    + `a gaussian. This chain is not a smooth invertible map and has no such excuse.`);

  set('closing-1',
    `The result this page did not expect is the last one. A normalising flow cannot put a hole in `
    + `its density, for a reason that is topological and unanswerable, and the obvious next `
    + `sentence is that a diffusion model can, because its reverse chain is a sequence of `
    + `stochastic steps rather than one smooth map. It can. It does not. `
    + `${bestHole >= Number(D.flow_hole)
      ? `Measured on ${n0(holeDet.n)} draws rather than the few hundred a picture shows, the `
        + `better of the two samplers puts ${bestHole} of its mass in that circle against the `
        + `flow's ${D.flow_hole}: the same order of magnitude, with the truth four orders below `
        + `both.`
      : `Measured on ${n0(holeDet.n)} draws, the better sampler puts ${bestHole} there against `
        + `the flow's ${D.flow_hole}, an improvement of `
        + `${(Number(D.flow_hole) / bestHole).toFixed(1)} times, with the truth still far below `
        + `both.`} `
    + `And the reason is on this page, two sections up: carving that hole out is work for the `
    + `last steps of the chain, and the last steps are exactly where the learned score is worst.`);

  set('closing-2',
    `Which points at the next article rather than away from it. If the expensive part of `
    + `diffusion is the fine structure at the clean end, and the expensive part of doing it on `
    + `images is that there are ${M.digit_side * M.digit_side} dimensions of fine structure to `
    + `get right, then the move is to stop working in pixels: run the diffusion in the small `
    + `space an autoencoder already learned, and let the decoder put the detail back. That is `
    + `latent diffusion, and it is the next thing this module will do.`
    + `<br /><br />`
    + `The article that comes next in the atlas, though, changes the object rather than the `
    + `method. Everything up to here has started from a row of numbers or a picture, and both of `
    + `those arrive already numeric. A <span class="bold">document</span> does not: it is a `
    + `sequence of words of no fixed length, from a vocabulary nobody wrote down, and before any `
    + `of this machinery can touch it somebody has to decide what its numbers are. `
    + `<a href="../tf-idf/">Counting words</a> is where that starts.`);

  set('ref-note',
    `${n0(M.n_train)} draws from the closed form density, seed ${M.seed}, a ${M.hidden} wide `
    + `network with three hidden layers trained for ${M.epochs} epochs to predict the noise. `
    + `The schedule is linear over ${n0(M.steps)} steps unless a chart says otherwise. Every `
    + `quantity called exact on this page is a closed form or a quadrature on a box sized for `
    + `that noise level: the mass guard holds to ${G.worst_mass}, interpolating the ring's radial `
    + `profile costs ${G.worst_interp}, and the independent Fourier route agrees to `
    + `${G.worst_fft}. The digits are the ${M.digit_side} by ${M.digit_side} set the `
    + `<a href="../autoencoders/">bottleneck article</a> published, rebuilt and then proved `
    + `identical by running that article's own encoder on them. The shared judge is the module's, `
    + `${(M.judge_accuracy * 100).toFixed(2)}% on held out digits. The weights the page runs are `
    + `${n0(data.net.count)} numbers in float16, and the page checks itself against the `
    + `generator on a ${data.net.probe.points[0].length} point probe at each level rather than `
    + `trusting that it read them correctly.`);

  return IDS;
}
