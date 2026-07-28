/* The sentences that carry a measured number, written from the measurement.
 *
 * Nothing below is typed by hand. Every comparison is a branch, including the
 * directions ("better", "further", "more compact"), which are numbers too, and
 * including the cells of the verdict table, which is where a stale claim
 * survives longest because nothing on the page contradicts it.
 */

import { halfSeparation } from './encoding-widget.js';

const n0 = (v) => Math.round(v).toLocaleString('en-US');
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const db = (v) => `${v.toFixed(2)} dB`;
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v <= 12 ? WORDS[v] : n0(v));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
/* Seven digits of ratio is a number nobody reads. */
const big = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)} million`
  : v >= 1e4 ? `${Math.round(v / 1e3)} thousand` : n0(v));

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

/* "Was it trained long enough" is the first thing anyone asks of a fit with a
 * step count in it, so it gets answered from the training curve rather than
 * left to the reader's charity: how much the batch loss still moved over the
 * last quarter of the run, in the same units as everything else on the page. */
function convergence(curve, iters) {
  if (!curve || curve.length < 8) return '';
  const at = (lo, hi) => {
    const rows = curve.filter(([it]) => it >= lo * iters && it < hi * iters);
    if (!rows.length) return null;
    return rows.reduce((s, [, v]) => s + v, 0) / rows.length;
  };
  const third = at(0.5, 0.75);
  const last = at(0.75, 1.01);
  if (!third || !last) return '';
  const moved = 10 * Math.log10(third / last);
  return `Over the last quarter of training the batch loss still moved `
    + `${moved.toFixed(2)} dB, so this is near a floor rather than on one: longer would be `
    + (moved > 1.0 ? 'noticeably better' : 'a little better') + `, and it would be better for both `
    + `arms of every comparison below.`;
}

export function initProse(data) {
  /* Named up front, and named in the error too. A block missing from the JSON
     throws somewhere in the middle of composing and leaves every sentence
     after it saying "Measured below.", which is loud enough to catch but says
     nothing about what went wrong; this says which block is absent. */
  const need = ['meta', 'nerf', 'splat', 'cost', 'quadrature', 'viewdep', 'geometry', 'views',
    'cameras', 'shapes', 'box'];
  const missing = need.filter((k) => !data[k]);
  if (missing.length) {
    throw new Error(`scene.json is missing ${missing.join(', ')}: regenerate before publishing`);
  }
  const M = data.meta;
  const N = data.nerf;
  const S = data.splat;
  const C = data.cost;
  const Q = data.quadrature;
  const V = data.viewdep;
  const G = data.geometry;
  const bandLo = Math.min(...M.rings);
  const bandHi = Math.max(...M.rings);
  const outEl = data.cameras.outside.map((c) => c.el);
  const belowIdx = outEl.map((e, i) => (e < bandLo ? i : -1)).filter((i) => i >= 0);
  const aboveIdx = outEl.map((e, i) => (e > bandHi ? i : -1)).filter((i) => i >= 0);
  const pick = (arr, idx) => idx.map((i) => arr[i]);
  const nBelow = mean(pick(N.per_outside, belowIdx));
  const nAbove = mean(pick(N.per_outside, aboveIdx));
  const sBelow = mean(pick(S.per_outside, belowIdx));
  const sAbove = mean(pick(S.per_outside, aboveIdx));
  const belowEl = Math.min(...outEl);
  const aboveEl = Math.max(...outEl);

  /* ------------------------------------------------------------- the scene */
  const solids = data.shapes.map((s) => s.name);
  set('dataset-note',
    `The scene is ${word(solids.length)} solids, and it is not a mesh anyone modelled: it is `
    + `${word(solids.length)} signed distance functions, a shell of half-width ${M.shell} across which `
    + `each of them fades from solid to nothing, and Blinn-Phong radiance on top. So there is a closed `
    + `form for how much stuff sits at any point in space and what colour it sends in any direction, `
    + `and a photograph is that closed form integrated along ${n0(M.res * M.res)} rays. That is the whole `
    + `reason for building the scene this way rather than downloading one: the ground truth here is not `
    + `an annotation, it is the function, so every error on this page is measured against the answer `
    + `rather than against somebody's opinion of it. The solids take up ${pc(data.box.solid_fraction, 1)} `
    + `of the box they live in, and ${pc(M.coverage, 1)} of the average frame has anything in it at all.`);

  set('rig-note',
    `The cameras sit on ${word(M.rings.length)} rings, ${word(M.ring_n)} to a ring, at `
    + `${M.rings.map((e) => `${e}°`).join(', ')} of elevation. Held out: ${word(M.n_inside)} cameras `
    + `between the rings, which is the ordinary novel view question, and ${word(M.n_outside)} more `
    + `outside the band entirely, ${word(belowIdx.length)} at ${belowEl}° and `
    + `${word(aboveIdx.length)} at ${aboveEl}°. The second group is the one that matters, and it is the `
    + `group that usually goes unreported: it asks what a fitted scene knows about a direction nobody `
    + `ever pointed a camera in.`);

  /* -------------------------------------------------------- the quadrature */
  const rows = Q.rows;
  const at = (s) => rows.find((r) => r.steps === s);
  const target = 40;
  const crossing = rows.find((r) => r.psnr >= target);
  const eight = at(8) || rows[0];
  const sixtyfour = at(64) || rows[rows.length - 1];
  const perDouble = rows.length > 2
    ? (rows[rows.length - 1].psnr - rows[0].psnr) / Math.log2(rows[rows.length - 1].steps / rows[0].steps)
    : 0;
  set('quad-note',
    `Before a single weight is fitted, that sum is already an approximation, and it is worth knowing how `
    + `bad. Against the same integral evaluated at ${n0(Q.ref_steps)} steps, `
    + `${eight.steps} steps score ${db(eight.psnr)} and are wrong by as much as `
    + `${f1(eight.worst * 255)} of 255 on a single channel; ${sixtyfour.steps} steps score `
    + `${db(sixtyfour.psnr)} and are wrong by ${f1(sixtyfour.worst * 255)}. `
    + (crossing
      ? `The sum passes ${target} dB at <span class="bold">${crossing.steps}</span> steps. `
      : `The sum never reaches ${target} dB in this sweep. `)
    + `The rate is about <span class="bold">${f1(perDouble)} dB per doubling</span>, `
    + (perDouble < 8
      ? `which is first-order quadrature of an integrand with a wall in it. `
      : perDouble < 11
        ? `between the 6 dB a first-order rule gives and the 12 dB of a second-order one, which is what a `
          + `midpoint rule does on an integrand that is smooth but only just: the shell each solid fades `
          + `across is ${M.shell * 2} units wide and a step at the low end of this sweep is wider than `
          + `that. `
        : `faster than the 6 dB a first-order rule gives, because the shell each solid fades across `
          + `makes the integrand smooth rather than discontinuous. `)
    + `Everything below renders at `
    + `${M.eval_steps} steps, and the reason that number is not larger is the next section.`);

  /* ------------------------------------------------------------- the field */
  set('nerf-intro',
    `Now delete the closed form and keep only the photographs. Put a small network where `
    + `\\(\\sigma\\) and \\(\\mathbf{c}\\) used to be: ${n0(N.params)} weights, `
    + `${word(N.arch.D)} hidden layers of ${N.arch.W}, taking a point and `
    + `a direction and returning a density and a colour. Render a pixel with the sum above, compare it `
    + `with the photograph, and push the error back through the sum into the weights. `
    + `${n0(M.iters)} steps of Adam later, over batches of ${n0(M.batch)} rays with `
    + `${M.train_steps} samples each, the field agrees with the ${M.n_train} photographs to `
    + `<span class="bold">${db(N.train)}</span> and with the ${M.n_inside} it was never shown to `
    + `<span class="bold">${db(N.inside)}</span>. The gap between those two numbers is `
    + `${f2(N.train - N.inside)} dB, which is how much of the fit is memory. `
    + convergence(N.loss_curve, M.iters));

  set('encoding-intro',
    `There is one famous caveat, and it is usually stated as an absolute: that this does not work at all `
    + `if you hand the network the coordinate. A ReLU network of a raw coordinate is a smooth, `
    + `low-frequency thing, so to make its output change across the width of one pixel it needs enormous `
    + `weights, and gradient descent will not find them. The repair is to stop feeding it the coordinate `
    + `and feed it a bank of sines and cosines of the coordinate instead:`);

  const lad = N.encoding.ladder;
  const best = lad.reduce((a, b) => (b.inside > a.inside ? b : a));
  const zero = lad.find((r) => r.L === 0);
  const wide = N.encoding.wide;
  /* the row the page is actually running, read off the architecture rather
     than assumed, and "where the ladder goes flat" measured against the same
     seed floor that every other difference on this page is measured against */
  const usedL = lad.find((r) => r.L === N.arch.Lx) || best;
  const gain = usedL.inside - zero.inside;
  const gainWide = usedL.inside - wide.inside;
  const floor = Math.max(0.2, N.seed_spread || 0);
  const saturates = lad.filter((r) => r.inside >= best.inside - floor)[0];
  /* Where the trained ladder makes its biggest step, and what the closed-form
     panel says the encoding's resolution is on either side of it. The two
     panels are the same story told twice and the prose should join them. */
  const halfX = (M.aabb.hi[0] - M.aabb.lo[0]) / 2;
  const pixelU = (2 * M.cam_radius * Math.tan((M.fov_deg * Math.PI) / 360)) / M.res;
  let jump = null;
  for (let i = 1; i < lad.length; i++) {
    const step = lad[i].inside - lad[i - 1].inside;
    if (!jump || step > jump.step) {
      jump = {
        step, lo: lad[i - 1], hi: lad[i],
        loPx: (halfSeparation(lad[i - 1].L) * halfX) / pixelU,
        hiPx: (halfSeparation(lad[i].L) * halfX) / pixelU,
      };
    }
  }
  set('encoding-result',
    `Measured, and the measurement is smaller than the usual telling of it. Going from no bands to `
    + `${usedL.L} is worth <span class="bold">${f2(gain)} dB</span> on the held-out cameras, `
    + `${db(zero.inside)} against ${db(usedL.inside)}, which is a factor of `
    + `${f1(10 ** (gain / 10))} in squared error, though in the strip above you have to look at the `
    + `edges to find it. It is not the difference between a picture and a fog. `
    /* The control is the whole point of the section, so which way it came out
       is a branch: how much of the gap is still there once the encoding-free
       network is given the same number of weights to spend on width. */
    + (gainWide > gain + 0.15
      ? `It more than survives the honest control. Give an encoding-free network the same `
        + `${n0(wide.params)} weights and let it spend them on width instead, at ${wide.W} units a `
        + `layer, and it does not close the gap: it reaches ${db(wide.inside)}, which is `
        + `<span class="bold">${f2(gain - gainWide)} dB worse</span> than the narrow encoding-free `
        + `network it was supposed to improve on. So none of the ${f2(gain)} dB is the parameter `
        + `count, and on this scene handing a coordinate network more width without giving it any `
        + `frequencies is not neutral, it is a liability. `
      : gainWide >= 0.75 * gain
        ? `It survives the honest control almost intact: an encoding-free network given the same `
          + `${n0(wide.params)} weights to spend on width instead reaches ${db(wide.inside)}, so `
          + `${f2(gainWide)} of the ${f2(gain)} dB is the bands and not the budget. `
        : gainWide <= 0.35 * gain
          ? `And most of it is the budget rather than the bands: an encoding-free network given the same `
            + `${n0(wide.params)} weights to spend on width reaches ${db(wide.inside)}, closing `
            + `${f2(gain - gainWide)} of the ${f2(gain)} dB on its own. `
          : `Some of it is the budget rather than the bands: an encoding-free network given the same `
            + `${n0(wide.params)} weights to spend on width reaches ${db(wide.inside)}, which closes `
            + `${f2(gain - gainWide)} of the ${f2(gain)} dB and leaves ${f2(gainWide)}. `)
    + `The ladder also stops climbing: ${best.L} bands is the best row at ${db(best.inside)} and `
    + `${saturates.L} bands is already within ${f2(best.inside - saturates.inside)} dB of it. `
    /* The page runs one particular row and it is not necessarily the winning
       one, which is worth saying out loud rather than letting the reader find
       the discrepancy between the bar chart and the model they are dragging. */
    + (best.L === usedL.L
      ? `That best row is the one running on this page. `
      : `The model running on this page is the ${usedL.L} band one, `
        + `${f2(best.inside - usedL.inside)} dB behind the best row, which is `
        + (best.inside - usedL.inside <= N.seed_spread
          ? `less than the ${f2(N.seed_spread)} dB this recipe moves between random seeds, so the two `
            + `rows are not distinguishable. `
          : `more than the ${f2(N.seed_spread)} dB this recipe moves between random seeds, so that one `
            + `is a real difference and the page is not showing the best fit. `))
    + `The reason is in the left panel and it is a length scale, and the two panels line up: the ladder's `
    + `biggest step is from L = ${jump.lo.L} to L = ${jump.hi.L}, `
    + `<span class="bold">${f2(jump.hi.inside - jump.lo.inside)} dB</span>, and that is exactly where the `
    + `encoding stops needing ${f1(jump.loPx)} pixels to tell two points apart and starts needing `
    + `${f1(jump.hiPx)}. Every band after that halves a number that is already below a pixel, and this `
    + `scene is made of large smooth solids with no texture anywhere in it: a network that can hold an `
    + `edge is all it needs, and past that the extra bands have nothing left to resolve.`);

  /* --------------------------------------------------------- running it here */
  const kB = (b) => `${Math.round(b / 1024)} kB`;
  set('live-intro',
    `Everything above was measured offline. What follows is not: the ${n0(N.params)} weights are on this `
    + `page, rounded to half precision, ${kB(M.json_bytes.field)} of base64 including the occupancy `
    + `grid. Rounded and unrounded agree to <span class="bold">${db(N.fp16_psnr)}</span>, which is to `
    + `say the rounding is invisible, and every number quoted on this page was measured `
    + `<span class="bold">after</span> the rounding, so the model being described is the model you are `
    + `about to drag around.`);

  const sc = N.steps_curve;
  const scAt = (s) => sc.find((r) => r.steps === s) || sc[sc.length - 1];
  /* the largest step count below the one the page renders at, whatever the
     sweep happens to contain: asking for exactly half of it once produced an
     empty interpolation and a sentence that read "the two are 15.21 dB and :" */
  const lower = sc.filter((r) => r.steps < M.eval_steps).slice(-1)[0] || sc[0];
  const here = scAt(M.eval_steps);
  set('skip-note',
    `Two things make that possible at all. The first is the box: a ray that misses the scene's bounding `
    + `box renders the background exactly, and no network is consulted. The second is an occupancy grid, `
    + `${N.grid_res} cells on a side, marking every cell where the fitted density is not negligible and `
    + `then dilated by one cell so it can only ever be too generous. It keeps `
    + `<span class="bold">${pc(N.skip_kept)}</span> of the samples and throws away the rest, and the `
    + `picture it produces differs from evaluating every single sample by `
    + `<span class="bold">${db(N.skip_psnr)}</span>. That is not zero. It is `
    + `${f1(N.skip_psnr - N.inside)} dB quieter than the error the field already has against the scene, `
    + `a factor of ${big(10 ** ((N.skip_psnr - N.inside) / 10))} in squared error, so it is invisible, and `
    + `it removes <span class="bold">${pc(1 - N.skip_kept)}</span> of the network evaluations. `
    + `The sample slider is the other half of the same story: at ${M.eval_steps} samples the field `
    + `renders itself to ${db(here.self)} of its own converged answer and at ${lower.steps} to `
    + `${db(lower.self)}, while against the scene those same two renders score ${db(here.truth)} and `
    + `${db(lower.truth)}. Past a point the sum stops being the thing that is wrong.`);

  /* ------------------------------------------------------------- the views */
  const vc = data.views;
  const v0 = vc[0];
  const vN = vc[vc.length - 1];
  set('views-intro',
    `The field above holds ${db(N.inside)} on cameras it never saw and ${db(nBelow)} on cameras at an `
    + `elevation of ${belowEl}°, which is ${bandLo - belowEl}° below the lowest ring. That is a gap of `
    + `<span class="bold">${f2(N.inside - nBelow)} dB</span> between two questions that are usually `
    + `reported as one number, and it is not a defect of the fit. Nothing constrains the field where no `
    + `photograph looked, so what it says there is whatever happened to be cheapest for the photographs `
    + `it did have. Above the band, at ${aboveEl}°, it scores ${db(nAbove)} instead, because that side is `
    + `only ${aboveEl - bandHi}° past a ring while the other is ${bandLo - belowEl}° past one and looking `
    + `at the underside of a floor that no camera ever saw.`);

  const dInside = vN.inside - v0.inside;
  const belowOf = (r) => mean(pick(r.per_outside, belowIdx));
  const dBelow = belowOf(vN) - belowOf(v0);
  /* The below-the-band curve does not have to go the same way as the one
     inside it, and here it does not: it peaks in the middle. That is worth its
     own sentence, because "more photographs is better" is exactly the belief
     this section exists to complicate. */
  const bestBelow = data.views.reduce((a, b) => (belowOf(b) > belowOf(a) ? b : a));
  const nonMono = bestBelow.views !== vN.views
    && belowOf(bestBelow) - belowOf(vN) > Math.max(0.3, N.seed_spread);
  set('views-result',
    `Quadrupling the photographs, ${v0.views} to ${vN.views}, moves the held-out cameras inside the band `
    + `by <span class="bold">${f2(dInside)} dB</span> and the ones below it by `
    + `<span class="bold">${f2(dBelow)} dB</span>. `
    + (dInside > dBelow + 0.3
      ? `More cameras in the same band buy the band, and buy much less outside it, which is the honest `
        + `way to read every novel view number ever published: it is interpolation inside the hull of the `
        + `cameras, and the word "novel" is doing less work than it sounds like it is.`
        + (nonMono
          ? ` And below the band it is not even monotone. The best score down there belongs to the `
            + `${bestBelow.views}-photograph fit, ${db(belowOf(bestBelow))} against `
            + `<span class="bold">${db(belowOf(vN))}</span> for the ${vN.views}-photograph one: `
            + `<span class="bold">${f2(belowOf(bestBelow) - belowOf(vN))} dB worse for twice the `
            + `data</span>, and well clear of the ${f2(N.seed_spread)} dB this recipe moves between `
            + `seeds. Extra photographs inside the band are extra reasons to explain the band, and `
            + `nothing whatsoever is holding the field honest underneath it, so the better it fits what `
            + `it was shown the freer it is to drift where it was not. That is the whole argument of `
            + `this section in one non-monotone curve, and it was not the curve I expected to draw.`
          : '')
      : dBelow > dInside + 0.3
        ? `Here the cameras outside the band gained more than the ones inside it, which is not what the `
          + `hull argument predicts and is worth more than the prediction was.`
        : `The two move together, so on this scene the extra photographs help everywhere about equally, `
          + `which is not what the hull argument predicts.`));

  set('viewdep-intro',
    `One more thing the field is given that a mesh with a texture is not: the direction. The colour it `
    + `returns is a function of \\(\\mathbf{x}\\) <span class="bold">and</span> \\(\\mathbf{d}\\), which `
    + `is what lets it hold a highlight that slides across the sphere as you walk around it. Whether `
    + `that is worth anything is a question the frame average cannot answer.`);

  const hotGain = V.on_hot - V.off_hot;
  const coldGain = V.on_cold - V.off_cold;
  const allGain = V.on_all - V.off_all;
  /* every one of these three can come out with either sign */
  const swing = (v) => (Math.abs(v) < 0.005 ? 'changes nothing'
    : v > 0 ? `is worth ${f2(v)} dB` : `costs ${f2(-v)} dB`);
  set('viewdep-result',
    `The specular lobe covers <span class="bold">${pc(V.frac, 2)}</span> of the frame, so over the whole `
    + `frame the direction input ${swing(allGain)} and could be dismissed as noise. Split the `
    + `pixels by whether the true radiance there depends on the direction at all, and it stops hiding: `
    + `inside the lobe the direction <span class="bold">${swing(hotGain)}</span> `
    + `(${db(V.off_hot)} to ${db(V.on_hot)}) and outside it, where nothing depends on where you stand, `
    + `it <span class="bold">${swing(coldGain)}</span>. `
    + (hotGain > 3 * Math.abs(coldGain) + 0.2
      ? `The effect is where the theory says it is and nowhere else, which is the cleanest form this kind `
        + `of result comes in: an aggregate that resolves nothing and a decomposition that does.`
      : hotGain <= N.seed_spread
        /* The decomposition was supposed to find the effect inside the lobe. It
           found it everywhere else instead, and the honest reading is the one
           the numbers force, not the one the section was written for. */
        ? `That is the decomposition finding the opposite of what it was built to find. Inside the lobe, `
          + `where the radiance genuinely depends on where you stand, the direction input is worth `
          + `${f2(hotGain)} dB, which is inside the ${f2(N.seed_spread)} dB this recipe moves between `
          + `random seeds: the two are not distinguishable there. All of the gain is outside the lobe, `
          + `where nothing depends on direction at all. `
          + `Two things could do that and this page cannot separate them. The lobe is the hardest part `
          + `of the frame for both models, ${db(V.on_hot)} against ${db(V.on_cold)} elsewhere, so there `
          + `may simply be no room to win there at this scale; and the direction input is also `
          + `${3 + 6 * N.arch.Ld} extra numbers into the colour head, which the network is free to spend `
          + `on anything, including the parts of the scene that never move. The measurement stands and `
          + `the usual explanation for it does not.`
        : `The effect is not confined to the lobe, which the theory did not predict: the direction is `
          + `worth ${f2(coldGain)} dB where the radiance does not depend on it at all, against `
          + `${f2(hotGain)} where it does, so the extra input is doing something besides the highlight.`));

  /* ------------------------------------------------------------- the blobs */
  set('splat-intro',
    `The field is one answer to "what is a scene". Here is a completely different one, fitted to the `
    + `same ${M.n_train} photographs with the same loss: <span class="bold">${n0(S.n)}</span> gaussian `
    + `blobs floating in the same box, ${S.dof_each} numbers each, `
    + `${n0(S.dof)} numbers in all against the field's ${n0(N.params)}. They start uniformly at random `
    + `and are moved by Adam for ${n0(M.splat_iters)} steps; every ${M.splat_recycle} steps the ones that have gone `
    + `transparent are recycled next to the ones whose centres are pulling hardest, which is a `
    + `fixed-budget stand-in for the clone-and-split heuristic in the paper. Fitted, they hold `
    + `<span class="bold">${db(S.inside)}</span> on the held-out cameras inside the band, against the `
    + `field's ${db(N.inside)}.`);

  const ratio = S.axis_ratio_median;
  set('blob-note',
    `The median blob has a scale of ${S.scale_median} scene units, so about `
    + `${(4 * S.scale_median).toFixed(2)} across at two standard deviations, and a median axis ratio of `
    + `<span class="bold">${ratio}</span>. `
    + (ratio > 2.2
      ? `That is a flake, not a ball: the optimiser spent the freedom, because a blob that lies flat `
        + `against a surface covers more of it for the same number of numbers. `
      : ratio < 1.35
        ? `That is very nearly a ball, so on this scene the optimiser barely used the freedom the full `
          + `covariance gave it, and the section below is where that stops being a curiosity. `
        : `That is neither a ball nor a flake: the optimiser used some of the freedom the full `
          + `covariance gave it and not all of it. `)
    + `A pixel has to look at <span class="bold">${S.touched_per_pixel}</span> of them on average, out of `
    + `${n0(S.n)}, because a blob is cut off at three standard deviations and reaches nothing beyond `
    + `that. That single fact is the whole performance argument, and the section after next puts a `
    + `number on it.`);

  set('aniso-intro',
    `Two groups inside those ${S.dof_each} numbers per blob are worth arguing about. Six of them are the `
    + `covariance, where a sphere would need one, and nine are the term that makes the colour depend on the viewing `
    + `direction. The comparison that gets published is ellipsoids against spheres at the same blob `
    + `count, which is not a fair fight, because the ellipsoids are also carrying `
    + `${n0(S.ablations.find((a) => a.key === 'aniso').dof - S.ablations.find((a) => a.key === 'iso').dof)} `
    + `more numbers. So both comparisons are here: at equal count, and at equal storage.`);

  const A = Object.fromEntries(S.ablations.map((a) => [a.key, a]));
  const shapeWorth = A.aniso.inside - A.iso_matched.inside;
  set('aniso-result',
    (shapeWorth > 0.25
      ? `The shape survives the fair comparison. At equal storage the ellipsoids are `
        + `<span class="bold">${f2(shapeWorth)} dB</span> ahead of the sphere cloud, which is `
        + `${f1(10 ** (shapeWorth / 10))} times less squared error for the same file, and the median `
        + `axis ratio of ${S.axis_ratio_median} says where it came from: the optimiser used the freedom.`
      : shapeWorth < -0.25
        ? `The shape does not survive the fair comparison. At equal storage the sphere cloud is `
          + `<span class="bold">${f2(-shapeWorth)} dB</span> ahead, so on this scene the six covariance `
          + `numbers would have been better spent on more blobs, and the usual equal-count comparison was `
          + `measuring the extra numbers rather than the extra shape.`
        : `At equal storage the two are within ${f2(Math.abs(shapeWorth))} dB of each other, so on this `
          + `scene the anisotropy is worth roughly what the numbers it costs would buy as extra spheres, `
          + `and the usual equal-count comparison was measuring the budget.`)
    + ` The seed spread is ${f2(S.seed_spread)} dB over ${S.seed_psnr.length} `
    + `run${S.seed_psnr.length === 1 ? '' : 's'}, which is the floor under every difference on this `
    + `chart` + (Math.abs(shapeWorth) > S.seed_spread
      ? `, and this one clears it.` : `, and this one does not clear it.`)
    /* The direction term costs storage too, so whether it earns that storage is
       read off the ladder at its own budget rather than at its own blob count.
       On this scene it does not, which the chart shows and the prose should
       not leave to the eye either. */
    + (() => {
      const nv = A.no_viewdep;
      const lo2 = [...S.counts].reverse().find((r) => r.dof <= nv.dof);
      const hi2 = S.counts.find((r) => r.dof >= nv.dof);
      if (!lo2 || !hi2 || lo2.n === hi2.n) return '';
      const t = (nv.dof - lo2.dof) / (hi2.dof - lo2.dof);
      const ladderHere = lo2.inside + t * (hi2.inside - lo2.inside);
      return ` The direction term is the other thing that costs storage, and at its own budget of `
        + `${n0(nv.dof)} numbers the ellipsoid ladder is worth about ${db(ladderHere)}, against the `
        + `${db(nv.inside)} the direction-free cloud actually reaches: dropping it `
        + (nv.inside > ladderHere
          ? `and spending the numbers on blobs instead would have been the worse trade here.`
          : `and spending the numbers on blobs instead would have been the better trade here.`);
    })()
    /* The count ladder flattening is the answer to the obvious objection about
       the gap to the field, so it gets said here rather than left to the eye. */
    + (S.counts.length > 1 ? ` And the ladder itself is flattening: the last doubling, `
      + `${n0(S.counts[S.counts.length - 2].n)} blobs to ${n0(S.counts[S.counts.length - 1].n)}, buys `
      + `<span class="bold">${f2(S.counts[S.counts.length - 1].inside
        - S.counts[S.counts.length - 2].inside)} dB</span> where the first one bought `
      + `${f2(S.counts[1].inside - S.counts[0].inside)}. Whatever separates this cloud from the field `
      + `on this scene, it is not simply a shortage of blobs.` : ''));

  /* -------------------------------------------------------------- the cost */
  /* Whether the two land close or far apart decides how this section opens,
     so it is a branch and not an assumption. */
  const gap = N.inside - S.inside;
  set('cost-intro',
    (Math.abs(gap) < 1.5
      ? `Both models on this page answer the same question at the same camera, and they land within `
        + `${f2(Math.abs(gap))} dB of each other. `
      : `Both models on this page answer the same question at the same camera, and they do not answer `
        + `it equally well: the ${gap > 0 ? 'field' : 'blob cloud'} is `
        + `<span class="bold">${f2(Math.abs(gap))} dB</span> ahead of the other on the held-out `
        + `cameras, ${db(N.inside)} against ${db(S.inside)}. `)
    + `They also do not cost the same, and the two facts point in opposite directions.`);

  /* Neither loss ever mentioned geometry, so whether either of them got it is
     a separate question with a separate answer, in scene units rather than in
     decibels, and it is the one the colour frames cannot be asked. */
  set('geometry-note',
    `Neither of these models was ever told where anything is. Both were shown photographs, and a wrong `
    + `shape that happened to produce the right photographs from the ${M.n_train} places somebody stood `
    + `would have scored exactly as well. So here is the other question, in scene units instead of `
    + `decibels: over the pixels where the fit and the scene agree there is a surface at all, the field `
    + `puts it <span class="bold">${G.field_inside.depth_error}</span> away from where it is and the `
    + `blobs <span class="bold">${G.splat_inside.depth_error}</span>. A pixel is ${G.pixel} units across `
    + `at this distance, so that is `
    + `${f2(G.field_inside.depth_error / G.pixel)} against `
    + `${f2(G.splat_inside.depth_error / G.pixel)} pixels of depth, a factor of `
    + `<span class="bold">${f1(G.splat_inside.depth_error / G.field_inside.depth_error)}</span> `
    + `between two models the photographs put ${f2(Math.abs(gap))} dB apart. They disagree with the `
    + `scene about whether there is anything there at all on ${pc(G.field_inside.silhouette, 2)} and `
    + `${pc(G.splat_inside.silhouette, 2)} of the frame. `
    + `Below the photographed band both get worse, to ${G.field_outside.depth_error} and `
    + `${G.splat_outside.depth_error}, `
    + (G.field_outside.depth_error > G.field_inside.depth_error * 1.5
      ? `which is the same lesson the pictures told, arriving through a different door: what a fit knows `
        + `about a direction is what a camera told it.`
      : `which is a smaller penalty than the pictures suggested, so the shape survives leaving the band `
        + `better than the colour does.`));

  const compact = N.params < S.dof;
  /* The two do not hold the same number of numbers, so the blob ladder is read
     at the field's own budget rather than at the count that happened to be
     chosen: the rows either side of it are the honest bracket. */
  const below18 = [...S.counts].reverse().find((r) => r.dof <= N.params);
  const above18 = S.counts.find((r) => r.dof >= N.params);
  const bracket = (below18 && above18 && below18.n !== above18.n)
    ? `The two do not store the same amount, so read the blob ladder at the field's own budget: `
      + `${n0(N.params)} numbers falls between the ${n0(below18.n)} blob row at ${db(below18.inside)} `
      + `and the ${n0(above18.n)} blob row at ${db(above18.inside)}. `
    : '';
  set('cost-result',
    bracket
    + `So: the field is the more compact of the two, ${n0(N.params)} numbers against ${n0(S.dof)}, `
    + `a factor of ${f1(compact ? S.dof / N.params : N.params / S.dof)}, and the blobs are the cheaper `
    + `to draw, by ${f1(C.ratio)} times in arithmetic. Neither of those is an accident of this scene: `
    + `a field pays per pixel because every pixel is an integral and every sample on it is a network `
    + `evaluation, and blobs pay per primitive because a primitive is looked at only where it lands. `
    + `Below the band the two also fail differently: the field holds ${db(nBelow)} and the blobs `
    + `${db(sBelow)}`
    + (Math.abs(nBelow - sBelow) < 0.4
      ? `, which is the same answer to within ${f2(Math.abs(nBelow - sBelow))} dB. `
      : `, ${f2(Math.abs(nBelow - sBelow))} dB apart in favour of the `
        + `${nBelow > sBelow ? 'field' : 'blobs'}. `)
    + `The seed spread for the field is ${f2(N.seed_spread)} dB over ${N.seed_psnr.length} `
    + `run${N.seed_psnr.length === 1 ? '' : 's'} and for `
    + `the blobs ${f2(S.seed_spread)} dB, so a difference smaller than about `
    + `${f2(Math.max(N.seed_spread, S.seed_spread))} dB on this page is not a difference.`);

  /* ------------------------------------------------------------ the verdict */
  set('verdict-quad',
    `${eight.steps} steps score ${db(eight.psnr)} against the converged integral and `
    + `${sixtyfour.steps} score ${db(sixtyfour.psnr)}, about ${f1(perDouble)} dB per doubling.`);
  set('verdict-encoding',
    `${f2(gain)} dB on held-out cameras against no bands, and ${f2(gainWide)} dB against the same weight `
    + `budget spent on width instead`
    + (gainWide > gain ? `, which is to say the width made it worse. ` : `. `)
    + `The ladder is flat from L = ${saturates.L} onwards.`);
  set('verdict-field',
    `${n0(N.params)} numbers hold ${db(N.inside)} on held-out cameras, ${db(nBelow)} below the `
    + `photographed band, and put the surface ${G.field_inside.depth_error} scene units off. One frame `
    + `is ${n0(C.nerf.points_per_frame)} network evaluations.`);
  set('verdict-splat',
    `${n0(S.n)} blobs, ${n0(S.dof)} numbers, ${db(S.inside)} held out, ${G.splat_inside.depth_error} `
    + `units of depth error, and ${f1(C.ratio)} times less arithmetic a frame than the field. A pixel `
    + `looks at ${S.touched_per_pixel} of them.`);

  /* ------------------------------------------------------------- the close */
  set('limits-note',
    `What this page cannot show. The scene is ${word(solids.length)} smooth solids with no texture on `
    + `them, photographed at ${M.res} by ${M.res} pixels from ${M.n_train} known cameras with no noise, `
    + `no motion blur and no calibration error; real capture has all of those and the pose estimation `
    + `alone is a research field. Both models here are two or three orders of magnitude smaller than `
    + `the ones in the papers, and trained for ${n0(M.iters)} and ${n0(M.splat_iters)} steps rather than `
    + `the hundreds of thousands those use. `
    + `The comparison between them is fair because they share the photographs, the loss and the `
    + `rendering equation; it is not a statement about which idea wins at scale, and the ${f2(C.ratio)}x `
    + `arithmetic gap in particular would be a different number with a hash grid or a tiled rasteriser `
    + `in the loop.`);

  set('runtime-note',
    `Every renderer on this page checks itself. At load time the browser rebuilds the scene from its `
    + `closed form, evaluates the field with and without empty space skipping, and rasterises the blobs, `
    + `all at a camera the generator also rendered, and compares pixel by pixel against references `
    + `computed from the same half-precision weights. Open the console to see the worst disagreement; `
    + `<code>window.__atlasCheck()</code> runs them again.`);

  set('closing-note',
    `The through line of this module was that a picture is a grid of numbers and everything else is `
    + `arithmetic on it. This article breaks that: the pictures were never the object, they were `
    + `${M.n_train} measurements of something else, and both methods here are estimators of the `
    + `something else. That is why the ${db(nBelow)} below the band matters more than the `
    + `${db(N.inside)} above it. A model that reproduces the photographs it was given has done `
    + `nothing; the only interesting number is the one from where nobody stood.`);

  set('resources-note',
    `Data: a scene defined in closed form in <code>src/utils/generate_nerf_data.py</code>, photographed `
    + `at ${n0(M.gt_steps)} quadrature steps and rounded to bytes. ${M.n_train} training cameras, `
    + `${M.n_inside} held out inside the photographed band and ${M.n_outside} outside it. The field is `
    + `${n0(N.params)} weights trained for ${n0(M.iters)} Adam steps on batches of ${n0(M.batch)} rays at `
    + `${M.train_steps} samples a ray, seed ${M.seed}; the blob cloud is ${n0(S.n)} gaussians trained for `
    + `${n0(M.splat_iters)} steps. Both were quantised to half precision before anything was measured `
    + `about them, and both run in this page from those quantised weights. Deliberately excluded: hash `
    + `grids, tiled rasterisation, hierarchical sampling with a second network, spherical harmonics past `
    + `first order, and pose estimation, all of which matter at scale and all of which would confound the `
    + `comparisons made here.`);
}
