/* Every number on this page, composed from the file it was measured into.
 *
 * The branches matter more here than anywhere else in this module, because the
 * page's three interesting claims are all claims that could fail on a re-run:
 * that reflow straightens the paths, that one evaluation afterwards is as good
 * as a hundred, and that distillation adds nothing on top. Each is written as a
 * question with both answers, so re-running the generator moves the prose
 * instead of leaving it asserting something the chart underneath contradicts.
 *
 * And one rule is enforced throughout: a comparison the redraw measurement did
 * not resolve is never written as a ranking, however tempting the gap looks.
 */
const IDS = ['opening-note', 'strip-intro', 'path-intro', 'path-note', 'reflow-intro',
  'steps-intro', 'steps-note', 'spread-intro', 'spread-note', 'live-intro', 'verdict-intro',
  'verdict-flow', 'verdict-flow-warn', 'verdict-reflow', 'verdict-reflow-warn',
  'verdict-student', 'verdict-student-warn', 'closing-1', 'closing-2', 'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initProse(data) {
  const M = data.meta;
  const S = data.straight;
  const A = data.arms;
  const F = data.setup.floor;
  const SP = data.spread;
  const row = (arm, steps) => A.find((r) => r.arm === arm && r.steps === steps);
  const bestOf = (arm) => A.filter((r) => r.arm === arm)
    .reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));

  const oneFlow = row('rectified flow', 1);
  const oneRe = row('after reflow', 1);
  const student = row('distilled student', 1);
  const bestFlow = bestOf('rectified flow');
  const bestRe = bestOf('after reflow');
  const gapFlow = data.gaps.find((r) => r.arm === 'rectified flow' && r.steps === 1);

  /* the pairs the redraw measurement was run for, looked up by name so a
     re-run that changes which cells were measured cannot leave a stale claim */
  const pair = (a, b) => SP.pairs.find(
    (p) => (p.a === a && p.b === b) || (p.a === b && p.b === a));
  const pBudget = pair('after reflow x1', 'after reflow x100');
  const pStudent = pair('after reflow x1', 'distilled student x1');
  const pCollapse = pair('rectified flow x1', 'after reflow x1');
  const pFree = pair('rectified flow x100', 'after reflow x1');
  const ranked = SP.pairs.filter((p) => p.resolved);
  const onlyCollapsed = ranked.length > 0 && ranked.every(
    (p) => p.a === 'rectified flow x1' || p.b === 'rectified flow x1');

  const straightened = S.straighter && S.factor > 1.5;
  const budgetFree = pBudget && !pBudget.resolved;
  const studentAdds = pStudent && pStudent.resolved
    && Math.abs(student.mmd2) < Math.abs(oneRe.mmd2);
  const inFloor = A.filter((r) => !r.resolves).length;
  const ratio = Math.abs(oneFlow.mmd2) / Math.abs(oneRe.mmd2);

  set('opening-note',
    `The latent underneath is the previous two articles', unchanged and checked: the same `
    + `autoencoder, whose round trip scores ${data.setup.ceiling} here against the `
    + `${data.setup.published} that article published for the same ${M.k} number bottleneck. `
    + `That matters more than usual on this page, because a sampler that takes one step and a `
    + `sampler that takes a hundred are being compared against a ceiling neither of them set, `
    + `and if the ceiling had moved the comparison would be with something else.`);

  set('strip-intro',
    `Four samplers and a row of real digits, from the same ${data.strips.per_pool} pieces of `
    + `noise. Three of these rows cost one pass through a network per picture and one of them `
    + `costs ${data.strips.show[1].steps}. Look at them before reading why.`);

  set('path-intro',
    `The reason you cannot simply take fewer, larger steps is geometric, and it is worth seeing `
    + `rather than being told. A sampler walks a path from noise to data. If that path were a `
    + `straight line, one step of the right size would land exactly where a hundred small ones `
    + `land, and the chain would be a waste of arithmetic. It is not a straight line, so a big `
    + `step leaves the path and arrives somewhere else.`);

  set('path-note',
    `How far from straight is measurable, and it is measured in all ${M.k} dimensions rather `
    + `than read off the drawing: the mean distance between the walked path and the straight `
    + `line joining its two ends, over the length of that line. Zero is a straight line. `
    + `Trained the ordinary way, this flow scores <span class="value">${S.flow}</span>, so its `
    + `paths wander ${pc(S.flow, 0)} of their own length off the line joining their ends. `
    + `${S.flow < 0.05
      ? `That is nearly straight, which is not what the next section expected.`
      : `That is not "nearly straight", which is how rectified flow is usually described, and `
        + `it is the honest starting point for the next section rather than an embarrassment: `
        + `the training target is straight lines, and the field that comes out of training on `
        + `them is not straight.`} `
    + `Walked with a single step, this model lands ${pc(gapFlow.gap)} of the whole trip away `
    + `from where a hundred steps land, which is not a near miss. It is further from the right `
    + `answer than the right answer is from where it started.`);

  set('reflow-intro',
    `The fix is stranger than the problem. Take the trained model, sample `
    + `${n0(M.pairs)} pictures with it, and keep the noise each one started from. That gives `
    + `pairs of (noise, sample) that the model itself considers matched, which independent `
    + `sampling never does: a random piece of noise and a random digit are not on any path. `
    + `Then retrain on those pairs, with the same objective and the same architecture. Nothing `
    + `changes except the pairing. `
    + `${straightened
      ? `Here it works, and by a lot: the deviation falls from ${S.flow} to `
        + `<span class="value">${S.reflow}</span>, a factor of ${S.factor}.`
      : `Here it does not work: the deviation goes from ${S.flow} to `
        + `<span class="value">${S.reflow}</span>, and the rest of this page reports that `
        + `rather than the result it was hoping for.`} `
    + `The comparison is at equal compute rather than at equal epochs, which is not a detail: `
    + `the retrained arm sees ${n0(M.pairs)} invented pairs against the first arm's `
    + `${n0(M.n_train)} real codes, so equal epochs would have handed it `
    + `${(M.pairs / M.n_train).toFixed(0)} times the gradient steps and any improvement would `
    + `have been bought with arithmetic instead of with the idea. Both arms get exactly `
    + `${n0(M.base_steps)} steps.`);

  set('steps-intro',
    `Straightness is a means, so here is the end: what each sampler's output is worth at each `
    + `budget, by the same two sample statistic every article in this module has used, against `
    + `the same real against real floor. The horizontal axis is network evaluations rather than `
    + `steps, because those two stopped being the same number in the previous article, where `
    + `guidance made every step cost two.`);

  set('steps-note',
    `${budgetFree
      ? `The reflowed curve is flat, and flat is a strong word, so the next section spends `
        + `itself proving it: one evaluation scores ${oneRe.mmd2} and a hundred score `
        + `${bestRe.mmd2}, a difference of ${(Math.abs(pBudget.diff)).toExponential(1)}. `
      : `After reflow one evaluation scores ${oneRe.mmd2} against ${bestRe.mmd2} at `
        + `${bestRe.steps}, so the chain got shorter without disappearing. `} `
    + `The arm before reflow is the contrast: a single evaluation there costs `
    + `${oneFlow.mmd2}, which is ${ratio.toFixed(0)} times the reflowed model's, and it needs `
    + `${bestFlow.steps} evaluations to come back down to ${bestFlow.mmd2}. `
    + `${inFloor
      ? `${inFloor} of these ${A.length} entries land inside the floor real digits set against `
        + `each other (${F.mmd2_max}), and they are printed without emphasis for that reason.`
      : `Every entry here is outside the floor real digits set against each other `
        + `(${F.mmd2_max}), so each of them is distinguishable from real data.`} `
    + `A last note on the second view: by the classifier test, everything except the collapsed `
    + `arm sits at or under the ${F.c2st_max} that real digits score against other real digits. `
    + `Two statistics, two answers, and the honest reading is that the kernel one still `
    + `separates these pools and the classifier one no longer does.`);

  set('spread-intro',
    `Now the part that decides what this article is allowed to say. The floor above answers `
    + `one question, "is this pool distinguishable from real data", and the chart invites a `
    + `different one, "is this pool closer to real data than that one". Those are not the same `
    + `question, and after the paths are straightened the fast arms land within one percent of `
    + `each other, which is exactly where the first gets mistaken for the second.`);

  set('spread-note',
    `So every cell worth arguing about was measured ${SP.draws} times with nothing changed but `
    + `the noise. A pair counts as ranked only if ${SP.draws} redraws of one never reach `
    + `${SP.draws} redraws of the other, and the band comes from the two cells compared rather `
    + `than from the whole chart: the collapsed arm's value is thirty times the others', its `
    + `spread is proportionally wider, and one global band taken from it would have swallowed `
    + `every comparison worth making. `
    + `${onlyCollapsed
      ? `${SP.resolved} of the ${SP.pairs.length} pairs survive, and every one of them is the `
        + `collapsed arm against something else. `
      : `${SP.resolved} of the ${SP.pairs.length} pairs survive. `}`
    + `${budgetFree
      ? `In particular, one evaluation after reflow against a hundred comes out `
        + `${pBudget.sigmas.toFixed(2)} combined standard deviations apart, which is to say `
        + `they are the same measurement. `
      : `One evaluation after reflow against a hundred is ${pBudget.sigmas.toFixed(2)} combined `
        + `standard deviations apart, which is a real difference. `}`
    + `${pFree && !pFree.resolved
      ? `And the sentence worth carrying out of this page: one evaluation after reflow is not `
        + `separable from <i>${bestFlow.steps}</i> evaluations of the model before it `
        + `(${pFree.sigmas.toFixed(2)} combined standard deviations). A hundred times fewer `
        + `passes through the network, for a change this measurement cannot see.`
      : `One evaluation after reflow against ${bestFlow.steps} evaluations before it does come `
        + `apart, at ${pFree ? pFree.sigmas.toFixed(2) : '?'} combined standard deviations, so `
        + `the saving is real but it is not free.`}`);

  set('live-intro',
    `All three, running here, with the budget as a dial. The counter under the pictures is the `
    + `honest cost: pictures times evaluations each. Drag the budget down on the first arm and `
    + `watch it come apart; drag it down on the second and `
    + `${budgetFree ? 'watch nothing happen that the measurement above can see' : 'watch it degrade more gently'}.`);

  set('verdict-intro',
    `Three ideas for shortening a sampling chain, and what each one actually bought here.`);

  set('verdict-flow',
    `Train on straight lines between noise and data instead of on a noise schedule. Free: a `
    + `change of objective, not of budget.`);
  set('verdict-flow-warn',
    `${S.flow < 0.05
      ? `Its paths came out nearly straight here (${S.flow}), which is the advertised behaviour.`
      : `Its paths are not straight (${S.flow} of their own length off the chord), so on its `
        + `own it buys no short chains at all: one evaluation costs `
        + `${(Math.abs(oneFlow.mmd2) / Math.abs(bestFlow.mmd2)).toFixed(0)} times what `
        + `${bestFlow.steps} cost.`}`);

  set('verdict-reflow',
    `${straightened
      ? `Retrain on the model's own (noise, sample) pairs. Straightens the paths by a factor of `
        + `${S.factor}, and ${budgetFree ? 'flattens the quality curve to within redraw noise' : 'shortens the chain'}.`
      : `Retrain on the model's own pairs. Straightened nothing here (${S.flow} to ${S.reflow}).`}`);
  set('verdict-reflow-warn',
    `A second full training run plus ${n0(M.pairs)} samples from the first model, and it can `
    + `only inherit what that first model knew: everything it learns, it learns from a teacher `
    + `that has already made all of its mistakes. `
    + `${pCollapse && pCollapse.resolved
      ? `The gain is not marginal, though: at one evaluation it is ${ratio.toFixed(0)} times `
        + `closer to real digits than the arm it came from, at `
        + `${pCollapse.sigmas.toFixed(0)} combined standard deviations.`
      : `And here the gain at one evaluation did not resolve against redraw noise.`}`);

  set('verdict-student',
    `Train a student to name the endpoint of a teacher trajectory from anywhere on it. One `
    + `evaluation by construction, with no budget to tune.`);
  set('verdict-student-warn',
    `${studentAdds
      ? `Here it improves on its teacher's own single evaluation, ${student.mmd2} against `
        + `${oneRe.mmd2}, at ${pStudent.sigmas.toFixed(2)} combined standard deviations.`
      : `Here it is not separable from its teacher's own single evaluation (${student.mmd2} `
        + `against ${oneRe.mmd2}, ${pStudent ? pStudent.sigmas.toFixed(2) : '?'} combined `
        + `standard deviations), so it adds nothing this model had not already got for free. `
        + `That is a result about this teacher rather than about distillation: when the `
        + `teacher's own one step answer is already as good as its hundred step answer, there `
        + `is nothing left for a student to recover.`}`);

  set('closing-1',
    `The shape worth carrying away is a chain of causes, not a winner. Diffusion sampling was `
    + `never expensive because the network was big; it was expensive because it needed a chain, `
    + `and it needed a chain because the path from noise to data is curved. `
    + `${straightened
      ? `Straighten the path and the chain shortens by itself, with no change to the sampler, `
        + `no change to the architecture and no distillation. Every technique that comes after `
        + `that is trying to recover something already recovered, which is exactly what the `
        + `student on this page failed to earn.`
      : `On this model straightening did not happen and the chain stayed as long as it was, `
        + `which is a reminder that these are measurements and not properties.`} `
    + `${inFloor || onlyCollapsed
      ? `The caveat is honest and it is in the last chart: on ${M.k} numbers standing for a `
        + `14 by 14 digit, the only comparison this measurement can actually rank is the one `
        + `where a sampler collapses. Everything subtler than that is inside the noise, and `
        + `the ordering at a scale where it would not be is an open question rather than `
        + `something this article measured.`
      : `Every ordering on this page survived the redraw test, which is more than the usual `
        + `standard for a claim of this kind.`}`);

  set('closing-2',
    `That closes the diffusion module and, with it, everything the atlas has to say about `
    + `making pictures. <a href="../ddpm/">The pixel article</a> built the chain, `
    + `<a href="../latent-diffusion/">the latent article</a> made each link cheaper, `
    + `<a href="../controlnet/">the conditioning article</a> made each link cost twice as much `
    + `in exchange for being told what to make, and this one asked how few links there have to `
    + `be.`
    + `<br /><br />`
    + `What comes next changes the object rather than the method. Everything so far has started `
    + `from a row of numbers or a picture, and both arrive already numeric. A `
    + `<span class="bold">document</span> does not: it is a sequence of words of no fixed `
    + `length, from a vocabulary nobody wrote down, and something has to decide what its `
    + `numbers are before any of this machinery can touch it. `
    + `<a href="../tf-idf/">Counting words</a> is where that starts.`);

  set('ref-note',
    `The autoencoder and the digits are the previous two articles', checked by reproducing the `
    + `published round trip. All three networks share one architecture, ${M.hidden} wide and `
    + `${M.depth} deep on a ${M.k} number latent, so the comparison is of training and never of `
    + `capacity. Every arm gets ${n0(M.base_steps)} gradient steps: the base flow as `
    + `${n0(M.epochs)} epochs over ${n0(M.n_train)} real codes in batches of ${M.flow_batch}, `
    + `the reflowed arm as ${n0(M.reflow_epochs)} epochs over ${n0(M.pairs)} of its teacher's `
    + `own pairs, the student as ${n0(M.student_epochs)} epochs over the same pairs in batches `
    + `of ${M.student_batch}. Sampling is Euler from noise to data at the budget shown. `
    + `Straightness is the mean distance from the walked path to the chord joining its ends, `
    + `over the length of that chord, in all ${M.k} dimensions, on 400 trajectories walked at `
    + `100 steps. Quality is the unbiased RBF two sample statistic on 250 samples against 500 `
    + `real digits, with the floor from ${F.splits} real against real splits, and the second `
    + `view is a two sample classifier test on the same pools. What may be ranked is decided `
    + `separately: ${SP.draws} redraws of each cell with nothing changed but the noise, and a `
    + `pair is called ranked only when those ${SP.draws} never overlap. The drawing of the `
    + `trajectories is a projection onto the latent's two strongest directions and is `
    + `illustration, not evidence. The weights the page runs are ${n0(data.net.count)} numbers `
    + `in float16, checked before anything is drawn against a ${data.net.probe.z.length} point `
    + `probe covering all three networks, an ${data.net.probe.walk_steps} step walk, the `
    + `projection and the decoder.`);

  return IDS;
}
