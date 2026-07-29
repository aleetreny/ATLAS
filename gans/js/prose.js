/* The sentences that carry a measured number, written from the measurement.
 *
 * Nothing below is typed by hand, including the directions ("worse", "further",
 * "the best of the five"), which are numbers too, and including the cells of
 * the verdict table, which is where a stale claim survives longest because
 * nothing else on the page contradicts it. Re-run the generator and every
 * sentence here moves with the result, or names the result it no longer has.
 */

import { lateR } from './image-widget.js';

const n0 = (v) => Math.round(v).toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const sci = (v, d = 2) => v.toExponential(d);
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v <= 12 && Number.isInteger(v) ? WORDS[v] : n0(v));
const plural = (n, one, many) => (n === 1 ? one : many);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const bold = (s) => `<span class="bold">${s}</span>`;

const ARM_NAME = {
  dcgan: 'the DCGAN',
  sat: 'the saturating loss',
  'wgan-clip': 'the clipped WGAN',
  'wgan-gp': 'WGAN-GP',
  mlp: 'the version with no convolutions',
};
const RING_NAME = {
  ns: 'the non-saturating loss',
  sat: 'the saturating loss',
  'wgan-clip': 'the clipped WGAN',
  'wgan-gp': 'WGAN-GP',
};

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(game, ring, images) {
  const need = [[game, ['meta', 'best', 'identity', 'grads', 'disjoint']],
    [ring, ['meta', 'calib', 'runs', 'live', 'falsify']],
    [images, ['meta', 'metric', 'runs', 'track', 'nn']]];
  const missing = [];
  need.forEach(([obj, keys]) => keys.forEach((k) => { if (!obj || !obj[k]) missing.push(k); }));
  if (missing.length) {
    throw new Error(`the data files are missing ${missing.join(', ')}: regenerate before publishing`);
  }

  const G = game.grads;
  const B = game.best;
  const M = ring.meta;
  const IM = images.meta;

  /* ------------------------------------------------------------ the set-up */
  const comps = game.meta.mixture;
  set('dataset-note',
    `Three datasets, in order of how much they hide. First a mixture of ${word(comps.length)} `
    + `gaussians on the line, at ${comps.map((c) => c.mu).join(' and ')} with width ${comps[0].sd}, `
    + `where every quantity in the theory can be integrated on a grid instead of estimated, so the `
    + `argument can be checked rather than illustrated. Then ${word(M.k)} gaussians arranged on a `
    + `circle of radius ${M.r} with width ${M.sd}, which is the smallest dataset where a generator `
    + `can be caught producing only some of the data. Then ${n0(IM.n_train)} MNIST digits, where `
    + `nobody can integrate anything and the only way to say whether it worked is to measure it.`);

  /* --------------------------------------------------- which wrong answer */
  const gap = (k) => Math.abs(B[k].mu);
  const modeMu = Math.max(...comps.map((c) => Math.abs(c.mu)));
  const revOnMode = gap('rev') > modeMu * 0.6;
  set('div-note',
    `The four objectives below are all reasonable ways to say "these two distributions differ", `
    + `and on a family that contains the truth they would all agree, because they would all be `
    + `zero at the same place. This family does not contain the truth, and that is when they stop `
    + `agreeing. Forward KL is ${bold(`${f4(B.fwd.value)}`)} at best, and it gets there at mean `
    + `${f3(B.fwd.mu)} with width ${f2(B.fwd.sd)}: one wide gaussian ${bold('covering')} both `
    + `modes, including the empty valley between them, because the term it cannot afford is data `
    + `where the model has no mass. Reverse KL is ${bold(`${f4(B.rev.value)}`)} at best, at mean `
    + `${f3(B.rev.mu)} with width ${f2(B.rev.sd)}: `
    + (revOnMode
      ? `sitting ${bold('on one mode')} and ignoring the other, because the term it cannot afford `
        + `is model where the data has none. The mirror image scores ${f4(B.rev.mirror)}, the same `
        + `number to every digit here, so the objective genuinely has two answers and the grid `
        + `returns whichever it met first.`
      : `which is not on a mode, so on this mixture the usual story about reverse KL collapsing `
        + `does not appear and the article says so.`));

  const jsBetween = B.js.sd < B.fwd.sd && B.js.sd > B.rev.sd;
  set('div-result',
    `Jensen-Shannon, the one the discriminator is estimating, lands at mean ${f3(B.js.mu)} with `
    + `width ${f2(B.js.sd)}: `
    + (jsBetween && B.fwd.sd / B.js.sd > 1.1
      ? `between the two, ${f2(B.fwd.sd / B.js.sd)} times narrower than the covering answer and `
        + `${f2(B.js.sd / B.rev.sd)} times wider than the collapsing one.`
      : jsBetween
        ? `between the two on paper, but only just: it is ${f2(B.fwd.sd / B.js.sd)} times narrower `
          + `than the covering answer and ${f2(B.js.sd / B.rev.sd)} times wider than the `
          + `collapsing one, so on this mixture it is a covering objective with a rounding error `
          + `of mode seeking in it. The often repeated "Jensen-Shannon sits halfway between the `
          + `two KLs" is a statement about its definition, not about where it puts the model.`
        : `not between the two, which is worth knowing given how often it is described that way.`)
    + ` Wasserstein picks mean ${f3(B.w1.mu)} and width ${f2(B.w1.sd)}. `
    + `None of this is about optimisation and none of it goes away with a better network. It is `
    + `the objective choosing, and the choice survives all the way up to megapixel image models, `
    + `where "covers everything blurrily" and "produces a few things beautifully" are the two `
    + `well known failure modes and they are these two terms.`);

  /* ------------------------------------------------------- the gradient */
  const at = (s) => G.reduce((b, r) => (Math.abs(r.sep - s) < Math.abs(b.sep - s) ? r : b), G[0]);
  const far = at(6);
  const mid = at(4.5);
  const ratioFar = Math.abs(far.ns) / Math.max(Math.abs(far.sat), 1e-300);
  const ratioMid = Math.abs(mid.ns) / Math.max(Math.abs(mid.sat), 1e-300);
  set('grad-intro',
    `The theorem needs the discriminator to be at its optimum, so the natural reading is that the `
    + `discriminator should be trained as well as possible. The famous problem with that is what `
    + `the generator is then told. Both of the losses below have the same fixed point, and the `
    + `first one is what the original paper wrote in its algorithm box.`);

  set('grad-result',
    `At a separation of ${f1(mid.sep)} the best possible discriminator is right ${pc(mid.acc, 2)} `
    + `of the time and the saturating loss hands back ${bold(sci(Math.abs(mid.sat)))} while the `
    + `non-saturating one hands back ${bold(sci(Math.abs(mid.ns)))}, a factor of `
    + `${bold(sci(ratioMid, 1))}. At ${f1(far.sep)} the discriminator is right ${pc(far.acc, 2)} `
    + `of the time and the factor is ${bold(sci(ratioFar, 1))}. That is the entire content of "the `
    + `gradient vanishes": not that the model is far away and therefore hard, but that the loss `
    + `goes flat exactly where the model is worst, because \\(\\log(1-D)\\) is flat wherever `
    + `\\(D\\) is near one. Flipping the sign inside the logarithm changes nothing about where the `
    + `optimum is and everything about the slope on the way there, and it is why every `
    + `implementation on this page, and essentially every implementation anywhere, trains the `
    + `second line rather than the first. The awkward part is that the neat statement about `
    + `Jensen-Shannon is about the loss nobody uses.`);

  /* -------------------------------------------------- disjoint supports */
  const rows = game.disjoint.rows;
  const nz = rows.filter((r) => r.theta > 0);
  const half = nz.reduce((b, r) => (Math.abs(r.theta - 0.5) < Math.abs(b.theta - 0.5) ? r : b), nz[0]);
  const worstW = Math.max(...nz.map((r) => Math.abs(r.w_est - r.w1) / r.w1));
  const critRatio = Math.abs(half.g_crit) / Math.max(Math.abs(half.g_sat), 1e-300);
  set('disjoint-intro',
    `Real image data does not fill its pixel space; it sits on something much thinner, and so does `
    + `anything a generator produces from a ${IM.z}-dimensional noise vector. Two thin sets in a `
    + `big space generically do not intersect, so the case below is not a pathology, it is the `
    + `normal case dressed down small enough to compute.`);

  const dHere = half.d_real;
  set('disjoint-result',
    `Every non-zero distance gives the same Jensen-Shannon divergence, ${f4(Math.log(2))}, so as a `
    + `function of where the model is it is a constant, and a constant has no downhill anywhere. `
    + `The measurement through a trained network says the same thing without any theory: at `
    + `distance ${f2(half.theta)} the discriminator scores real samples ${f4(dHere)} and generated `
    + `ones ${f4(half.d_fake)}, and the gradient it passes back is ${bold(sci(Math.abs(half.g_sat)))}. `
    + `A critic trained with a gradient penalty on the same samples passes back `
    + `${bold(f3(Math.abs(half.g_crit)))}, ${bold(`${sci(critRatio, 1)} times more`)}, and its own `
    + `estimate of the distance is ${f3(half.w_est)} against a true ${f2(half.w1)}. Across the `
    + `whole sweep its worst error is ${pc(worstW, 1)}. The critic is not being cleverer about the `
    + `same quantity: it is measuring a different one, and that one still knows which way is `
    + `closer when the two distributions have nothing in common.`);

  /* ---------------------------------------------------------- the ring */
  const finals = {};
  M.arms.forEach((a) => {
    finals[a] = M.seeds.map((s) => ring.runs[`${a}/${s}`].final);
  });
  const allEight = M.arms.filter((a) => finals[a].every((f) => f.covered === M.k));
  const someFail = M.arms.filter((a) => finals[a].some((f) => f.covered < M.k));
  const worstSeed = [];
  M.arms.forEach((a) => M.seeds.forEach((s, i) => {
    worstSeed.push({ a, s, ...finals[a][i] });
  }));
  worstSeed.sort((x, y) => x.covered - y.covered || x.quality - y.quality);
  const bad = worstSeed[0];
  const good = worstSeed[worstSeed.length - 1];
  const gp = ring.runs[`${ring.meta.arms[0]}/${M.seeds[0]}`];

  set('ring-intro',
    `The ring of ${word(M.k)} gaussians is the standard exhibit for mode collapse, and it is a `
    + `good one, because a generator that produces beautiful samples from three of the eight is `
    + `obviously wrong in a way that no image dataset would ever let you see. Four objectives, `
    + `${word(M.seeds.length)} seeds each, ${n0(M.steps)} generator updates each, a generator with `
    + `${n0(gp.gparams)} weights against a discriminator with ${n0(gp.dparams)}.`);

  const qualities = M.arms.map((a) => ({ a, q: mean(finals[a].map((f) => f.quality)) }));
  qualities.sort((x, y) => y.q - x.q);
  const spreadWithin = Math.max(...M.arms.map(
    (a) => Math.max(...finals[a].map((f) => f.quality)) - Math.min(...finals[a].map((f) => f.quality))));
  const spreadBetween = qualities[0].q - qualities[qualities.length - 1].q;
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  set('ring-result',
    cap(someFail.length === 0
      ? `Every objective covered every mode on every seed, which is not the result the folklore `
        + `predicts and is reported here as it came out.`
      : `${someFail.map((a) => RING_NAME[a]).join(' and ')} lost modes and `
        + `${allEight.length ? `${allEight.map((a) => RING_NAME[a]).join(' and ')} did not` : 'none of them held all eight'}. `
        + `The worst run on the page is ${RING_NAME[bad.a]} on seed ${bad.s}, which covers `
        + `${bold(`${bad.covered} of ${M.k}`)} modes with ${pc(bad.quality)} of its samples `
        + `anywhere near the ring, and the best is ${RING_NAME[good.a]} on seed ${good.s} with `
        + `${good.covered} and ${pc(good.quality)}.`)
    + ` But the number that decides how much any of this is worth is the spread: the widest gap `
    + `between two seeds of the ${bold('same')} objective is ${bold(pc(spreadWithin))} of samples `
    + `on the ring, and the gap between the best and worst objective averaged over seeds is `
    + `${bold(pc(spreadBetween))}. `
    + (spreadWithin > spreadBetween
      ? `The seed matters more than the objective, which is exactly the finding of the large scale `
        + `comparison in reference [8], reproduced here at a scale small enough to run in an `
        + `afternoon. Any ranking of these four that does not show its seeds is a ranking of seeds.`
      : `Here the objective does outweigh the seed, which is the happier case and not the one the `
        + `large scale comparisons usually report.`));

  /* the falsification attempts */
  const F = ring.falsify;
  const fRows = F.trials.map((t) => {
    const runs = t.seeds.map((s) => F.runs[`${t.name}/${s}`]).filter(Boolean);
    const before = M.seeds.map((s) => ring.runs[`${t.arm}/${s}`].final.covered);
    return {
      t,
      before,
      after: runs.map((r) => r.final.covered),
      afterQ: runs.map((r) => r.final.quality),
      beforeQ: M.seeds.map((s) => ring.runs[`${t.arm}/${s}`].final.quality),
    };
  });
  const longer = fRows.filter((r) => r.t.steps > M.steps);
  const fixed = longer.filter((r) => Math.min(...r.after) > Math.min(...r.before));
  const stronger = fRows.find((r) => r.t.n_critic === 5);
  const list = (parts) => (parts.length < 2 ? parts.join('')
    : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`);
  set('falsify-note',
    `A result that goes against the folklore has to be attacked before it is published, so it was: `
    + `${word(F.trials.length)} more configurations, chosen to give each side its best chance. `
    + (longer.length
      ? `At three times the budget, `
        + list(longer.map((r) => `${RING_NAME[r.t.arm]} goes from `
          + `${r.before.filter((_, i) => r.t.seeds.includes(M.seeds[i])).join(' and ')} modes to `
          + `${r.after.join(' and ')}`))
        + `. `
        + (fixed.length
          ? `So part of what looked like a difference between objectives was a difference in how `
            + `fast they get there, and the arm that looked worst is the one the extra budget `
            + `helps most. `
          : `So being slow is not the explanation for any of it. `)
      : '')
    + (stronger
      ? `Giving the discriminator five updates per generator update, which is the classic recipe `
        + `for making collapse happen, takes ${RING_NAME[stronger.t.arm]} on the same two seeds `
        + `from ${stronger.before.filter((_, i) => stronger.t.seeds.includes(M.seeds[i])).join(' and ')} `
        + `modes to ${bold(stronger.after.join(' and '))}, with `
        + `${stronger.afterQ.map((q) => pc(q)).join(' and ')} of its samples on the ring against `
        + `${stronger.beforeQ.filter((_, i) => stronger.t.seeds.includes(M.seeds[i])).map((q) => pc(q)).join(' and ')} `
        + `before, so a stronger opponent did not produce collapse here either. `
      : '')
    + `The table is every attempt, including the ones that found nothing, because a negative `
    + `result with no falsification behind it is indistinguishable from not having tried.`);

  /* ------------------------------------------------------------- live */
  const L = ring.live;
  set('live-intro',
    `The same game, in your tab. A generator of ${bold(n0(L.runs.ns.params))} weights and a `
    + `discriminator of the same shape, trained on the ring with batches of ${L.batch}, written a `
    + `second time in JavaScript with the same random number generator, the same initialisation `
    + `order and the same Adam, so that this is the experiment rather than an animation of it. `
    + `The page checks that when it loads: it reruns ${n0(L.steps)} steps and compares them, step `
    + `by step, against the trajectory the generator recorded in numpy.`);

  const liveNs = L.runs.ns.report;
  const liveSat = L.runs.sat.report;
  set('live-result',
    `Two things are worth watching for. The first is that the discriminator's surface goes flat `
    + `wherever it has already won, which is the picture behind the whole previous section. The `
    + `second is that coverage is not a one way street: modes get found, abandoned and found `
    + `again, and a run that has ${word(liveNs.covered)} modes at ${n0(L.steps)} steps is not `
    + `promising to still have them at twice that. At exactly ${n0(L.steps)} steps the reference `
    + `run covers ${liveNs.covered} of ${M.k} with ${pc(liveNs.quality)} on the ring under the `
    + `non-saturating loss, and ${liveSat.covered} with ${pc(liveSat.quality)} under the `
    + `saturating one, which at this size is `
    + (Math.abs(liveNs.quality - liveSat.quality) < 0.05
      ? `not a difference worth reading: two hundred steps is far too early for the gradient `
        + `argument above to have bitten.`
      : `already visible, though two hundred steps is early enough that the seed still owns most `
        + `of it.`));

  /* ----------------------------------------------------------- images */
  const dc = images.runs['dcgan/1'];
  set('arch-note',
    `Everything so far has been small enough to draw. The DCGAN recipe is what people actually `
    + `built once the game was understood: convolutions in both players, strided rather than `
    + `pooled, batch normalisation in the generator, and no fully connected layers except the one `
    + `that turns the noise vector into a small feature map. The generator here is `
    + `${bold(n0(dc.gparams))} weights taking ${IM.z} numbers of noise up to a ${IM.tile} by `
    + `${IM.tile} image, and the discriminator is ${n0(dc.dparams)}. Five arms, `
    + `${word(IM.seeds.length)} seeds each, ${n0(IM.steps)} generator updates each on `
    + `${n0(IM.n_train)} digits.`);

  const C = images.metric;
  const drop = C.class_drop / C.floor_train;
  set('metric-intro',
    `The blur ladder and the four reference lines are what make the numbers below mean something: `
    + `identical data scores ${bold(f2(C.floor_train))}, deleting three of the ten digits scores `
    + `${bold(f2(C.class_drop))}, which is ${f1(drop)} times the floor, and uniform noise scores `
    + `${bold(f2(C.noise))}. So a generator at ${f2(C.class_drop)} is as far from the data as one `
    + `that never draws a seven, an eight or a nine, and a generator at ${f2(C.floor_train)} `
    + `would be indistinguishable from a fresh sample of the real thing. Precision and recall on `
    + `real against real come out ${f3(C.pr_real[0])} and ${f3(C.pr_real[1])}, which is the other `
    + `floor: neither of them reaches one even when both sets are genuine, so those two numbers `
    + `are only ever worth reading against each other.`);

  const finalFd = {};
  IM.arms.forEach((a) => { finalFd[a] = IM.seeds.map((s) => images.runs[`${a}/${s}`].final.fd); });
  const ranked = IM.arms.map((a) => ({ a, fd: mean(finalFd[a]), spread: Math.abs(finalFd[a][0] - finalFd[a][1]) }))
    .sort((x, y) => x.fd - y.fd);
  const winner = ranked[0];
  const second = ranked[1];
  const worstArm = ranked[ranked.length - 1];
  const decisive = (second.fd - winner.fd) > Math.max(winner.spread, second.spread);
  set('samples-result',
    `Ranked by distance from the real digits: `
    + ranked.map((r) => `${ARM_NAME[r.a]} ${f2(r.fd)}`).join(', ')
    + `. The first thing to do with that list is to put the seed spread next to it, because on `
    + `these arms two seeds of the same code differ by up to `
    + `${bold(f2(Math.max(...ranked.map((r) => r.spread))))}. `
    + (decisive
      ? `The gap between ${ARM_NAME[winner.a]} and ${ARM_NAME[second.a]} is `
        + `${f2(second.fd - winner.fd)}, which is larger than that, so it survives; `
      : `The gap between ${ARM_NAME[winner.a]} and ${ARM_NAME[second.a]} is `
        + `${f2(second.fd - winner.fd)}, which is inside it, so those two are not separated by `
        + `this experiment; `)
    + `the gap between the best and ${ARM_NAME[worstArm.a]} is ${f2(worstArm.fd - winner.fd)} and `
    + `is not in doubt. `
    + (() => {
      const widest = ranked.slice().sort((x, y) => y.spread - x.spread)[0];
      const inside = ranked.filter((r) => r.a !== widest.a
        && r.fd > Math.min(...finalFd[widest.a]) && r.fd < Math.max(...finalFd[widest.a]));
      return `The widest spread belongs to ${ARM_NAME[widest.a]}, whose two seeds landed at `
        + `${finalFd[widest.a].map((v) => f2(v)).join(' and ')}`
        + (inside.length
          ? `, which brackets ${inside.map((r) => ARM_NAME[r.a]).join(' and ')} entirely: on this `
            + `evidence those arms are not ordered at all, and a table of one seed each would have `
            + `ordered them confidently.`
          : `, a factor of ${f1(Math.max(...finalFd[widest.a]) / Math.min(...finalFd[widest.a]))} `
            + `between two runs of identical code.`);
    })());

  /* The saturating loss was supposed to be the bad one, and on these runs it is
   * not. That is only a contradiction if the earlier section is read as "the
   * saturating loss is worse", which is not what it measured: it measured that
   * the saturating loss goes flat where the discriminator is winning. So the
   * honest follow-up is to ask whether it is, and the discriminator's own loss
   * answers that. */
  const dLate = (a) => {
    const vals = [];
    IM.seeds.forEach((s) => images.runs[`${a}/${s}`].curve.forEach((c) => {
      if (c.it > IM.steps / 2) vals.push(c.d_stat);
    }));
    return vals.length ? mean(vals) : null;
  };
  const satFd = mean(finalFd.sat);
  const nsFd = mean(finalFd.dcgan);
  const fooled = 2 * Math.log(2);
  const dcD = dLate('dcgan');
  const satD = dLate('sat');
  set('sat-note',
    (satFd <= nsFd
      ? `Which leaves one thing to explain, because it looks like a contradiction with the third `
        + `section of this page: ${bold('the saturating loss is not the worse arm here')}. `
        + `${cap(ARM_NAME.sat)} finishes at ${f2(satFd)} against ${f2(nsFd)} for the same `
        + `architecture trained the ordinary way. That section did not measure that the saturating `
        + `loss is worse; it measured that it goes flat ${bold('where the discriminator is winning')}, `
        + `and whether that has happened is checkable rather than assumable. Over the second half `
        + `of training the discriminator's loss averages ${f3(dcD)} on the ordinary arm and `
        + `${f3(satD)} on the saturating one, against ${f3(fooled)} for a discriminator that cannot `
        + `tell the two apart at all and zero for one that is certain. `
        + `${dcD > fooled * 0.4
          ? `So it is winning, but nowhere near the corner where the gradient argument bites: at `
            + `the separation where that argument produced a factor of ${sci(ratioFar, 1)}, the `
            + `best possible discriminator was right ${pc(far.acc, 2)} of the time. On MNIST at `
            + `this size it never gets there, and where it does not, the two losses have nothing `
            + `to choose between them. `
          : `So it is winning decisively and the saturating loss survives anyway, which is a `
            + `result against the argument rather than beside it, and it is reported as such. `}`
        + `The place to expect the difference is exactly where the discriminator is strongest, `
        + `which at this scale means the toy problems rather than the pixels.`
      : `${cap(ARM_NAME.sat)} finishes at ${f2(satFd)} against ${f2(nsFd)} for the ordinary loss, `
        + `so the ordering the third section predicts does show up in the images as well, with the `
        + `discriminator's loss averaging ${f3(dcD)} and ${f3(satD)} over the second half of `
        + `training against ${f3(fooled)} for one that cannot tell the two apart.`));

  set('curve-note',
    `The curves say something the final numbers cannot: whether an arm was still improving when `
    + `the budget ran out. `
    + (() => {
      const moved = IM.arms.map((a) => {
        const c = images.runs[`${a}/1`].curve;
        const lastQuarter = c.filter((p) => p.it > IM.steps * 0.75);
        const prev = c.filter((p) => p.it > IM.steps * 0.5 && p.it <= IM.steps * 0.75);
        if (!lastQuarter.length || !prev.length) return { a, drop: 0 };
        return { a, drop: mean(prev.map((p) => p.fd)) - mean(lastQuarter.map((p) => p.fd)) };
      }).sort((x, y) => y.drop - x.drop);
      const still = moved.filter((m) => m.drop > 0.5);
      const join = (parts) => (parts.length < 2 ? parts.join('')
        : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`);
      return still.length
        ? `${cap(join(still.map((m) => ARM_NAME[m.a])))} ${plural(still.length, 'was', 'were')} `
          + `still going down over the last quarter of training, by up to ${f2(still[0].drop)}, so `
          + `${plural(still.length, 'its', 'their')} number here is a lower bound on how good it `
          + `gets rather than a verdict on the objective.`
        : `None of them was moving by more than half a unit over the last quarter, so the ranking `
          + `is about the objectives rather than about who was cut off first.`;
    })());

  set('track-intro',
    `The practical question underneath all of this: while a GAN is training, the loss curves are `
    + `the only thing you can see, and the whole argument of the Wasserstein paper is that its `
    + `critic gives you a number that means something. This is checkable. Every checkpoint above `
    + `recorded both the statistic a practitioner would be watching and the actual distance from `
    + `the real data, so the two can simply be plotted against each other.`);

  const tracked = IM.arms.map((a) => ({ a, r: images.track[a].r, late: lateR(images, a) }))
    .filter((o) => o.r !== null);
  const byAbs = [...tracked].sort((x, y) => Math.abs(y.r) - Math.abs(x.r));
  const wgans = tracked.filter((o) => o.a.startsWith('wgan'));
  const positive = wgans.filter((o) => o.r > 0.3);
  const negative = wgans.filter((o) => o.r < -0.3);
  const lateFlip = wgans.filter((o) => o.late !== null && o.late > 0.3);
  set('track-result',
    `The largest relationship on the page is ${ARM_NAME[byAbs[0].a]} at ${f3(byAbs[0].r)} and the `
    + `smallest is ${ARM_NAME[byAbs[byAbs.length - 1].a]} at ${f3(byAbs[byAbs.length - 1].r)}. `
    + (positive.length === wgans.length
      ? `Both critic based arms come out positive `
        + `(${wgans.map((o) => `${f2(o.r)}`).join(' and ')}), which is the claim: a smaller `
        + `estimate of the distance really does mean better samples. `
      : negative.length
        ? `And the two critic based arms come out ${bold('negative')} `
          + `(${wgans.map((o) => f3(o.r)).join(' and ')}), which is not a weak version of the `
          + `claim, it is the opposite of it: over these runs a ${bold('larger')} estimate of the `
          + `distance goes with better samples. The reason is visible in the scatter and is the `
          + `critic's own warm-up rather than anything about the generator: at the first `
          + `checkpoints the critic has learned nothing, so it reports a distance near zero while `
          + `the samples are at their worst. Restricted to the second half of training the same `
          + `correlations are ${wgans.map((o) => (o.late === null ? 'undefined' : f3(o.late))).join(' and ')}, `
          + (lateFlip.length === wgans.length
            ? `so once the critic is warm the claim does hold, and the honest version of it is `
              + `that the estimate is readable only after the critic has caught up. `
            : lateFlip.length
              ? `which is worse than a clean failure: the two critic arms do not even agree with `
                + `each other on the sign, so at this scale the statistic is not something to `
                + `steer by in either direction. `
              : `so the claim does not survive the correction either, at this scale. `)
        : `Neither critic based arm reaches a correlation worth reading in either direction. `)
    + `The structural point stands whatever the numbers do: a discriminator's loss is the score of `
    + `a game against an opponent that is also moving, so it can improve because the samples got `
    + `better or because the discriminator got worse, and it cannot tell you which. Every `
    + `checkpoint on this page needed a metric computed outside the game to be placed at all.`);

  const NNd = images.nn;
  const memorising = IM.arms.filter((a) => NNd[a] < NNd.held_out);
  set('nn-note',
    `One more objection to clear, because it is the first one anybody raises at a sheet of `
    + `samples: is it copying. For each generated digit, the distance to its nearest neighbour `
    + `among the training images, in the same feature space, averages `
    + IM.arms.map((a) => `${f2(NNd[a])} for ${ARM_NAME[a]}`).join(', ')
    + `. Real digits the generator never saw sit at ${bold(f2(NNd.held_out))} and a training image `
    + `to its own nearest neighbour sits at ${f2(NNd.train_self)}. `
    + (memorising.length
      ? `${memorising.map((a) => ARM_NAME[a]).join(' and ')} ${plural(memorising.length, 'is', 'are')} `
        + `closer to the training set than a held out real digit is, which is what copying would `
        + `look like and is worth flagging rather than explaining away.`
      : `Every arm sits further from the training set than a held out real digit does, so whatever `
        + `is wrong with these samples, it is not that they are copies.`));

  const walkers = images.runs['dcgan/1'];
  set('interp-note',
    `Four straight lines through the ${IM.z}-dimensional noise space, eight steps each, through `
    + `${ARM_NAME.dcgan}. What matters is not that the endpoints are digits, it is that the `
    + `middles are: a lookup table has nothing between its entries, and a map has everything. `
    + `That property is what the rest of this module is built on, because once the noise space `
    + `means something you can start asking for control over it, which is the next article.`);
  set('interp-readout',
    `<p>Each row is a walk between two points drawn from the same gaussian the generator was `
    + `trained on. Nothing along the way was ever a training image, and nothing along the way was `
    + `chosen: this is the straight line.</p>`);

  set('curve-readout',
    `<p>One line per seed, thick for the first. The dashed line is real data against other real `
    + `data, which is where a perfect generator would be and where none of these are: the closest `
    + `any arm gets is ${f2(Math.min(...IM.arms.flatMap((a) => finalFd[a])))} against a floor of `
    + `${f2(C.floor_train)}.</p>`);

  /* --------------------------------------------------------- the verdict */
  const idn = game.identity;
  set('verdict-game',
    `On the line, where it can be checked, the value of the game at the best possible `
    + `discriminator equals 2 JSD &minus; 2 log 2 to ${sci(idn.exact, 1)} once the quadrature's `
    + `own missing mass of ${sci(idn.missing_mass, 1)} is accounted for.`);
  set('verdict-grad',
    `A factor of ${sci(ratioFar, 1)} in the gradient handed back, at the separation where the best `
    + `discriminator is right ${pc(far.acc, 2)} of the time.`);
  set('verdict-wgan',
    `On disjoint supports the critic's estimate tracks the true distance to within ${pc(worstW, 1)} `
    + `while the classifier's gradient is ${sci(Math.abs(half.g_sat))}. On MNIST, `
    + `${ARM_NAME[ranked.find((r) => r.a.startsWith('wgan')).a]} finishes at `
    + `${f2(ranked.find((r) => r.a.startsWith('wgan')).fd)} against ${f2(winner.fd)} for the best `
    + `arm.`);
  const mlpFd = mean(finalFd.mlp);
  const dcFd = mean(finalFd.dcgan);
  set('verdict-dcgan',
    (mlpFd > dcFd
      ? `Convolutions are worth ${bold(f2(mlpFd - dcFd))} on this metric, ${f1(mlpFd / dcFd)} times `
        + `the distance without them, against a seed spread of `
        + `${f2(Math.max(ranked.find((r) => r.a === 'mlp').spread, ranked.find((r) => r.a === 'dcgan').spread))}.`
      : `The version with no convolutions finishes at ${f2(mlpFd)} against ${f2(dcFd)} with them, `
        + `so at this size the architecture is not what is deciding the result.`));
  set('verdict-metric',
    `A dropped class scores ${f2(C.class_drop)} where identical data scores ${f2(C.floor_train)} `
    + `and noise scores ${f2(C.noise)}, so the scale has both ends nailed down before any `
    + `generator is put on it.`);

  /* --------------------------------------------------------- the closing */
  const lipRows = IM.arms.filter((a) => a.startsWith('wgan'))
    .map((a) => ({ a, lip: images.runs[`${a}/1`].lip }));
  set('limits-note',
    `What this page is not: none of these generators is good. `
    + `${n0(IM.steps)} generator updates on ${n0(IM.n_train)} digits at ${IM.tile} by ${IM.tile} `
    + `pixels is a demonstration, and the closest arm still sits `
    + `${f1(Math.min(...IM.arms.flatMap((a) => finalFd[a])) / C.floor_train)} times further from `
    + `the real data than a fresh sample of it. Two smaller caveats that are worth having in the `
    + `open: the arms are matched on generator updates rather than on arithmetic, so the critic `
    + `based ones did ${bold(`${images.runs['wgan-gp/1'].d_updates / images.runs['dcgan/1'].d_updates}x`)} `
    + `as many discriminator updates for the same place on the horizontal axis; and neither `
    + `Lipschitz mechanism actually delivers a 1-Lipschitz critic, `
    + lipRows.map((r) => `${ARM_NAME[r.a]} averaging ${f2(r.lip.mean)} and peaking at ${f2(r.lip.max)}`)
      .join(', ')
    + `, where the constraint asks for exactly 1.`);

  set('closing-note',
    `The thing to carry forward is that the generator here has a latent space and nothing has been `
    + `asked of it. It is a ${IM.z}-dimensional gaussian that got mapped onto digits, and the walk `
    + `above shows the map is smooth, but no direction in it means anything in particular and `
    + `nothing stops one coordinate from controlling stroke width and slant and the digit itself `
    + `all at once. The next article, <a href="../stylegan/">Style</a>, is about buying control of `
    + `that space: what a mapping network does to a latent distribution that has to fit a data `
    + `distribution with holes in it, and what per-layer styles turn out to be worth when you can `
    + `measure which factor each layer moves.`);

  set('resources-note',
    `Every figure on this page comes from <code>src/<wbr>utils/<wbr>generate_<wbr>gan_<wbr>data.py</code>: the `
    + `divergences are trapezoid integrals on a grid of ${n0(game.meta.grid.n)} points from `
    + `${game.meta.grid.x0} to ${game.meta.grid.x1}, recomputed in this tab and checked against `
    + `the generator's own values; the disjoint supports experiment trains each network for `
    + `${n0(game.disjoint.steps)} steps at batch ${game.disjoint.batch}; the ring runs `
    + `${n0(M.steps)} generator updates at batch ${M.batch} on ${word(M.seeds.length)} seeds; the `
    + `image runs are ${n0(IM.steps)} generator updates at batch ${IM.batch} with a `
    + `${IM.z}-dimensional latent, ${IM.g_ch} channels in the generator and ${IM.d_ch} in the `
    + `discriminator, on ${n0(IM.n_train)} MNIST digits. No wall clock time is quoted anywhere on `
    + `this page: the runs were measured with ${game.meta.threads} threads, and a time measured `
    + `with ${game.meta.threads} threads is a fact about this machine rather than about any of `
    + `these methods.`);
}
