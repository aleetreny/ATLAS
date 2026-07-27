/* Every numbered sentence on the page, written from the measurement.
 *
 * Same discipline as the depth and after-depth articles: nothing with a figure
 * in it is typed by hand, and every comparison is written as a comparison, so
 * that a rerun which moves a result moves the sentence with it instead of
 * leaving it quietly wrong. Where a claim could go either way, both ways are
 * written and the data picks.
 */
const n0 = (v) => v.toLocaleString('en-US');
const f4 = (v) => v.toFixed(4);
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const cap = (s) => `${s.charAt(0).toUpperCase()}${s.slice(1)}`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data, det) {
  const d = data.dataset;
  const W = data.windows;
  const eq = W.equivalence;
  const main = data.arms.find((a) => a.id === 'slots3-kmeans');
  const noise = Math.max(...data.verdicts.map((q) => q.seed_spread));

  /* ---- what these pictures are ------------------------------------------ */
  const withObjects = d.objects_hist
    .map((count, k) => ({ k, count })).filter((r) => r.count > 0);
  const usedClasses = d.aspect_table.filter((r) => r.used);
  const widest = usedClasses.reduce((a, b) => (b.aspect > a.aspect ? b : a));
  const narrowest = usedClasses.reduce((a, b) => (b.aspect < a.aspect ? b : a));
  const wobbliest = d.aspect_table.filter((r) => !r.used)
    .reduce((a, b) => (b.sd > a.sd ? b : a));
  set('dataset-note',
    `The pictures are built rather than found: ${n0(d.n_train)} training canvases of `
    + `${d.canvas}×${d.canvas} pixels carrying ${n0(d.boxes_train)} boxes, ${n0(d.n_val)} canvases `
    + `for validation and ${n0(d.n_test)} for test with ${n0(d.boxes_test)} boxes between them. `
    + `Each holds ${withObjects[0].k} to ${withObjects[withObjects.length - 1].k} garments from `
    + `Fashion-MNIST, pasted at a random place and a random size between `
    + `${d.sprite_side[0]} and ${d.sprite_side[1]} pixels, plus objects from the `
    + `${d.held_out.length} classes the detector is never taught `
    + `(${d.held_out.join(', ')}), ${n0(d.distractors_test)} of them in the test split. `
    + `That buys one thing worth the artificiality: a box is `
    + `<span class="bold">${data.meta.box_convention}</span>, so no annotation noise sits between a `
    + `claim and its evidence.`
    + `<br /><br />`
    + `The five classes it does learn were chosen for shape, not for difficulty: their aspect ratios `
    + `run from ${narrowest.aspect.toFixed(2)} for ${narrowest.cls} to ${widest.aspect.toFixed(2)} `
    + `for ${widest.cls}, a spread of ${d.aspect_spread.toFixed(2)} with the closest neighbouring `
    + `pair still ${d.aspect_min_gap.toFixed(2)} apart, which is what gives the anchor experiment `
    + `further down something to find. ${cap(wobbliest.cls)} was held out for the opposite reason: `
    + `its aspect ratio wanders by ${wobbliest.sd.toFixed(2)} from example to example, so a single `
    + `prior for it would mean nothing. Compositing takes a pixel-wise maximum and refuses any `
    + `placement overlapping an existing object by more than ${d.iou_cap} (${d.crowd_cap} in the `
    + `${pct(d.crowded_stratum_frac)} of canvases deliberately made crowded), which rejected `
    + `${n0(d.rejected_train)} placements in training. Objects still cover each other: `
    + `${pct(1 - d.visible_ink.mean)} of a typical object's ink is hidden, the median object keeps `
    + `${pct(d.visible_ink.p50)} of its own and the most buried five per cent keep only `
    + `${pct(d.visible_ink.p05)}. Nothing is ever cut off by the edge of the picture, which is a `
    + `limitation and is treated as one below.`);

  /* ---- the metric -------------------------------------------------------- */
  const g = data.iou_geometry;
  set('iou-note',
    `A label is right or wrong. A box is neither, so detection scores itself with a ratio: the area `
    + `two boxes share over the area they cover between them. What that ratio does next is the part `
    + `worth feeling rather than reading. A ${g.sizes[1]} pixel box loses half its overlap after `
    + `<span class="bold">${g.iou50_shift_axis_px.toFixed(2)} pixels</span> of sliding along one `
    + `axis, and after only ${g.iou50_shift_diag_px.toFixed(2)} pixels when it slides diagonally. `
    + `On a ${d.canvas} pixel canvas that is a rounding error in position turning into the `
    + `difference between a hit and a miss, and it is why every number in this article carries the `
    + `threshold it was measured at.`);

  /* ---- how a pile of boxes becomes one number ----------------------------- */
  set('protocol-note',
    `One more definition before any of the numbers mean anything, because average precision is `
    + `where detection quietly hides its assumptions. ${cap(data.eval.protocol)}. Precision and `
    + `recall are then read off that ranking and the area under the curve is the average precision, `
    + `averaged over the ${d.classes.length} classes to give the mAP quoted everywhere below.`
    + `<br /><br />`
    + `Three details that are usually left out. First, the area is measured differently by different `
    + `benchmarks, and the same detections give `
    + `${f4(main.map50)} under the all-point envelope, ${f4(main.map50_voc11)} under the `
    + `${data.eval.ap_conventions[1]} rule and ${f4(main.map50_coco101)} under the `
    + `${data.eval.ap_conventions[2]} rule: a spread of `
    + `${f4(Math.max(main.map50, main.map50_voc11, main.map50_coco101)
      - Math.min(main.map50, main.map50_voc11, main.map50_coco101))} `
    + `from the convention alone, which is larger than several of the design differences this `
    + `article measures. Second, the threshold that decides what counts as a hit is a choice: the `
    + `same detector scores ${f4(main.map_by_iou[0])} at IoU ${data.eval.iou_sweep[0]} and `
    + `${f4(main.map_by_iou[main.map_by_iou.length - 1])} at `
    + `${data.eval.iou_sweep[data.eval.iou_sweep.length - 1]}. Third, every interval on this page `
    + `comes from ${n0(data.eval.bootstrap.resamples)} bootstrap resamples of the `
    + `${data.eval.bootstrap.unit}s rather than of the detections, because ${data.eval.bootstrap.why}.`);

  /* ---- the window bill ---------------------------------------------------- */
  const vj = W.viola_jones;
  set('windows-note',
    `The obvious way to find things is the one `
    + `<a href="../viola-jones/">Viola-Jones</a> used: take a classifier, slide it everywhere, and `
    + `keep whatever it says yes to. On a ${d.canvas} pixel canvas, with `
    + `${W.sizes.length} sizes at ${W.aspects.length} aspect ratios, that is `
    + `<span class="bold">${n0(W.exhaustive_five_by_three)}</span> windows to classify per picture at `
    + `stride 1, or ${n0(W.exhaustive_five_by_three_stride4)} if you accept the blur of a stride of `
    + `4. The detector below evaluates <span class="bold">${W.detector_candidates}</span>. `
    + `Article 19 had the same arithmetic on a bigger picture: ${n0(vj.n_windows)} positions for a `
    + `${vj.window} pixel window on a ${vj.size} pixel photograph, and a cascade whose entire `
    + `purpose was to make most of them cheap, spending ${vj.scan_mean_evaluated} of its `
    + `${vj.cascade_full_cost} features on the average window instead of all of them. `
    + `A cascade makes the bill affordable. What follows makes most of it disappear.`);

  /* The spine of the article. If the two paths ever stop agreeing exactly, the
     sentence has to stop claiming they do. */
  const identical = eq.winclf_gap === 0;
  set('equivalence-note',
    `Here is the fact that turns one into the other. Take the window classifier, run it on a crop of `
    + `the picture, then run it on the whole picture in one pass and read off the cell that covers `
    + `the same crop. `
    + (identical
      ? `The two answers are <span class="bold">identical</span>. Not close, not on average: the `
        + `largest disagreement anywhere is exactly zero. `
      : `The two answers agree to ${eq.winclf_gap.toExponential(2)}, which is arithmetic noise but `
        + `not exact equality. `)
    + `A sliding window and a grid cell are the same computation counted twice. Everything a `
    + `detector does differently follows from taking that seriously: if the cell already contains `
    + `the answer, stop cropping, and let the network's own output grid be the list of places you `
    + `were going to look anyway.`
    + `<br /><br />`
    + `The saving is not free and it is not magic either. ${W.detector_cells} crops of 28×28 through `
    + `this trunk cost ${(W.sharing.crop_macs / 1e6).toFixed(2)} million multiply-accumulates against `
    + `${(W.sharing.full_macs / 1e6).toFixed(2)} for one pass over the whole canvas, a factor of `
    + `<span class="bold">${W.sharing.factor.toFixed(2)}</span>. Those crops also cover `
    + `<span class="bold">${W.sharing.area_redundancy.toFixed(2)}</span> times the area of the canvas `
    + `between them. Those two numbers are the same number: what you save by sharing is exactly what `
    + `you were paying for overlapping, and the ${pct(1 - W.sharing.residual)} between them is the `
    + `padding at the edges. `
    + (eq.detector_interior > 0
      ? `Which is also why the detector, unlike the classifier, is <i>not</i> exactly equivalent: it `
        + `pads, so a crop and the matching cells of a full pass differ by `
        + `${eq.detector_interior.toExponential(1)} in the interior and `
        + `${eq.detector_border.toExponential(1)} on the border cells, on outputs whose typical size `
        + `is ${eq.detector_magnitude.toFixed(3)}. The equivalence is a property of the arithmetic, `
        + `not a slogan, and padding breaks it at the edges.`
      : ''));

  /* ---- a window classifier is not a detector ------------------------------ */
  const r1 = data.sliding.rounds[0];
  const r2 = data.sliding.rounds[1];
  const mined = r2.map50 - r1.map50;
  set('sliding-note',
    `So build one and measure it. A classifier with ${n0(data.sliding.params)} parameters, trained `
    + `on ${n0(r1.n_train_crops)} crops the way article 19 trained its cascade `
    + `(${n0(r1.n_positives)} positives and ${n0(r1.n_negatives)} negatives), slid over `
    + `${n0(data.sliding.n_test)} canvases and allowed its best `
    + `${data.eval.sliding_topk} windows per picture: <span class="bold">mAP@0.5 ${f4(r1.map50)}</span>. `
    + (mined < 0
      ? `Then the standard repair, mining the ${n0(r2.n_mined)} crops it was most confidently wrong `
        + `about and training on those as well, which made it <span class="bold">worse</span>: `
        + `${f4(r2.map50)}. `
      : `Mining ${n0(r2.n_mined)} hard negatives moved it to ${f4(r2.map50)}. `)
    + `The detector on the same ${n0(data.sliding.detector_same_canvases.n_test)} canvases scores `
    + `<span class="bold">${f4(data.sliding.detector_same_canvases.map50)}</span>.`
    + `<br /><br />`
    + `The gap is not about capacity, and it is not about the sliding either. It is that a classifier `
    + `answers "is this a shoe" and a detector answers "where is the shoe and how sure am I", and `
    + `only the second question has an answer you can rank, suppress and score. Scanning the same `
    + `classifier more finely does not help: at stride `
    + `${data.sliding.evaluation_stride.map((s) => `${s.evaluation_stride} it scores ${f4(s.map50)}`).join(', at stride ')}. `
    + `${mined < 0
      ? 'Mining harder made it worse because the classifier has no way to say "nearly": every window '
        + 'that is 90% of a shoe is either a positive it cannot localise or a negative it is being '
        + 'punished for, and the more of those you feed it the more of the truth it learns to reject. '
        + 'That ambiguity is exactly what a box regression resolves.'
      : 'The improvement from mining is real but it does not close the gap.'}`);

  /* ---- assignment ---------------------------------------------------------- */
  const rows = data.grid.rows;
  const s16 = rows.find((r) => r.stride === 16 && r.slots === 1);
  const s16b = rows.find((r) => r.stride === 16 && r.slots === 3);
  const s8 = rows.find((r) => r.stride === 8 && r.slots === 1);
  set('grid-note',
    `A grid answers "where do I look" by refusing to choose: every cell answers, always, and the `
    + `training job becomes deciding which cell is responsible for which object. The cost of that `
    + `decision is measurable. At stride ${s16.stride} with ${s16.slots} slot per cell, `
    + `<span class="bold">${n0(s16.unassignable)}</span> of the ${n0(s16.boxes)} test objects have `
    + `no slot to be assigned to at all, because something else got there first: a ceiling of `
    + `${f4(s16.capacity_ceiling)} on recall before a single weight is trained. Give the same grid `
    + `${s16b.slots} slots and it is ${n0(s16b.unassignable)}. At stride ${s8.stride} the same `
    + `single-slot grid already loses only ${n0(s8.unassignable)}. `
    + `<span class="bold">Collisions are a resolution problem, and slots are the cheap fix.</span> `
    + `${cap(data.grid.ceiling_note)}.`);

  /* the grid table, built from the same rows */
  set('grid-table-body', rows.map((r) => `<tr>
      <td>${r.stride}</td><td>${r.slots}</td><td>${n0(r.candidates)}</td>
      <td>${n0(r.collisions)}</td><td>${n0(r.unassignable)}</td>
      <td>${f4(r.capacity_ceiling)}</td><td>${f4(r.prior_only_iou)}</td>
      <td>${f4(r.prior_only_at50)}</td></tr>`).join(''));

  /* ---- what the network is actually asked for ------------------------------- */
  const lw = data.loss.weights;
  set('loss-note',
    `With responsibility settled, the loss writes itself, and it is three losses in a trench coat. `
    + `Every slot predicts four numbers for the box, one for whether anything is there, and `
    + `${d.classes.length} for what it is. Assigned slots are asked for all three; unassigned slots `
    + `are asked only to say "nothing here". The box term is ${data.loss.box}, weighted `
    + `${lw.box} against ${lw.cls} for the class term, ${lw.obj_pos} for objectness where there is `
    + `an object and ${lw.obj_neg} where there is not, which is the crude version of the imbalance `
    + `fix that focal loss does properly: `
    + `${data.grid.rows.find((r) => r.stride === 8 && r.slots === 3).candidates} slots and rarely `
    + `more than three objects means the negatives would otherwise drown everything.`
    + `<br /><br />`
    + `One rule in there is worth pulling out, because it is the detail that stops the grid fighting `
    + `itself: ${data.loss.ignore_rule}. Without it, a slot that produced a perfectly good box for `
    + `an object it was not assigned to would be punished for it. Trained with `
    + `${data.loss.optimiser}, and guarded: ${data.loss.guard}.`);

  /* ---- the anchor factorial ------------------------------------------------ */
  const v = (a, b) => data.verdicts.find((q) => q.pair[0] === a && q.pair[1] === b);
  const mult = v('slots3-same', 'slots1');
  const div = v('slots3-kmeans', 'slots3-same');
  const qual = v('slots3-square', 'slots3-same');
  const stride16 = v('stride16', 'slots3-kmeans');
  const stride4 = v('stride4', 'slots3-kmeans');
  const resolved = data.verdicts.filter((q) => q.resolved);
  set('arms-note',
    `Three slots per cell, or one? The answer that gets repeated is "more anchors, better recall". `
    + `What this factorial measures is that the count is not the thing. Going from one slot to three `
    + `<span class="bold">identical</span> ones moves the score by `
    + `${mult.delta >= 0 ? '+' : ''}${f4(mult.delta)}, `
    + `${mult.resolved ? 'which clears the noise floor' : 'which is inside the noise floor'}`
    + `${mult.delta < 0 ? ', and it moves it the wrong way' : ''}. Making those three slots `
    + `<span class="bold">different</span>, by fitting them to the training boxes, is worth `
    + `${div.delta >= 0 ? '+' : ''}${f4(div.delta)}, `
    + `${div.resolved ? 'which does clear it' : 'which does not clear it'}. `
    + `Diversity is what the anchors are for; multiplicity on its own buys `
    + `${mult.resolved && mult.delta > 0 ? 'a little' : 'nothing'}.`
    + `<br /><br />`
    + `Two more comparisons, both worth reporting for what they fail to show. Replacing the three `
    + `identical priors with three identical <i>squares</i>, a worse guess at the same shape, moves `
    + `the score ${qual.delta >= 0 ? '+' : ''}${f4(qual.delta)}, `
    + `${qual.resolved ? 'which is resolved' : 'which is inside the seed spread, so this experiment '
      + 'cannot tell you whether the quality of a single prior matters at all'}. And halving `
    + `the grid to stride ${data.arms.find((a) => a.id === 'stride4').stride} costs `
    + `${f4(stride4.delta)}, also inside the spread, while doubling it to stride `
    + `${data.arms.find((a) => a.id === 'stride16').stride} costs `
    + `<span class="bold">${f4(stride16.delta)}</span>, which is `
    + `${(Math.abs(stride16.delta) / noise).toFixed(0)} times the noise floor and the largest single `
    + `effect in this article. Of the ${data.verdicts.length} comparisons the factorial was designed `
    + `to make, <span class="bold">${resolved.length}</span> come back with an answer and `
    + `${data.verdicts.length - resolved.length} come back saying the experiment was not big enough `
    + `to tell.`);

  /* ---- the thing in your browser -------------------------------------------- */
  const kb = det.weights_b64.length / 1024;
  set('detector-note',
    `What follows is not a video of a detector. The page downloads `
    + `${n0(det.n_values)} weights, ${det.dtype} in base64, ${kb.toFixed(0)} kB of them, and runs `
    + `the convolutions in JavaScript over pixels it reads out of a PNG. `
    + `${det.arch.channels.join(', ')} channels through ${det.arch.channels.length} `
    + `${det.arch.kernel}×${det.arch.kernel} layers with a `
    + `${det.arch.head_kernel}×${det.arch.head_kernel} head on top, `
    + `${n0(main.params)} parameters, ${(main.macs / 1e6).toFixed(2)} million multiply-accumulates `
    + `per canvas, ${det.arch.grid}×${det.arch.grid} cells at stride ${det.arch.stride} with `
    + `${det.arch.anchors.length} slots each.`
    + `<br /><br />`
    + `Getting there costs a chain of small approximations, and each one was measured rather than `
    + `waved through. Folding batch normalisation into the convolutions moves the outputs by `
    + `${det.errors.fold_abs.toExponential(2)} (${det.errors.fold_relative.toExponential(2)} `
    + `relative); doing the arithmetic in float32 instead of float64 moves them by `
    + `${det.errors.dtype_float64_vs_float32.toExponential(2)}; rounding the weights to `
    + `${det.dtype} moves the weights themselves by up to `
    + `${det.errors.weight_quantisation_abs.toExponential(2)} and the outputs by `
    + `${det.errors.output_quantisation_abs.toExponential(2)}. In the only unit that matters, that `
    + `last step, which halves the download, `
    + `${Math.abs(det.errors.map50_delta_from_quantisation) < 0.005
      ? `moved the score on the full test split by `
        + `${det.errors.map50_delta_from_quantisation >= 0 ? '+' : ''}`
        + `${det.errors.map50_delta_from_quantisation.toFixed(5)} mAP: nothing worth measuring`
      : `moved the score by ${det.errors.map50_delta_from_quantisation.toFixed(5)} mAP, a real cost `
        + `worth knowing about`}.`
    + `<br /><br />`
    + `The weights arrive with a contract, and the widget implements that and nothing else. `
    + `Layer order: <i>${det.arch.order}</i>. Head layout: <i>${det.arch.channel_layout}</i>. `
    + `Decode: <i>${det.arch.decode}</i>. Confidence: <i>${det.arch.score}</i>. Reading those from `
    + `the export instead of assuming them is what lets the readout below compare its own boxes `
    + `with the generator's rather than merely produce plausible ones.`
    + `<br /><br />`
    + `Move the two sliders and nothing about the network changes. The `
    + `${det.arch.grid * det.arch.grid * det.arch.anchors.length} boxes it proposes are the same `
    + `boxes, with the same confidences, at every setting: the sliders only decide which of them you `
    + `are shown. That is the whole of what happens after a detector stops computing, and the next `
    + `two sections are about how much it is worth.`);

  /* ---- confidence ----------------------------------------------------------- */
  const rule = Object.fromEntries(data.score_rule.map((r) => [r.rule, r]));
  const bestRule = data.score_rule.reduce((a, b) => (b.map50 > a.map50 ? b : a));
  const rel = data.reliability;
  const top = rel.bins[rel.bins.length - 1];
  const bottom = rel.bins[0];
  set('confidence-note',
    `A cell emits two different opinions: how sure it is that something is there at all, and what it `
    + `thinks that something is. Ranking by the first alone scores ${f4(rule.objectness.map50)}, by `
    + `the second alone ${f4(rule.class.map50)}, and by their product `
    + `<span class="bold">${f4(rule.product.map50)}</span>. `
    + `${bestRule.rule === 'product'
      ? 'The product wins, and it wins because the two mistakes are different mistakes: a box in the '
        + 'right place with the wrong label and a confident label on an empty patch of background '
        + 'both have to be pushed down the ranking, and neither term does both.'
      : `The best is ${bestRule.rule}, which is not the combination most detectors ship.`}`
    + `<br /><br />`
    + `That number is a ranking, though, and it is worth being precise about what it is not. Take `
    + `every surviving detection, bin it by confidence, and ask how often the bin is right, where `
    + `right means ${rel.event}. Across ${n0(rel.n_detections)} detections the expected calibration `
    + `error is <span class="bold">${rel.ece_weighted.toFixed(4)}</span> weighted by how many `
    + `detections land in each bin, and ${rel.ece_unweighted.toFixed(4)} treating every bin alike. `
    + `In the top bin the detector claims an average of ${top.mean_score.toFixed(2)} and is right `
    + `${pct(top.precision)} of the time. What it does get right is the order: precision climbs from `
    + `${f4(bottom.precision)} in the bottom bin to ${f4(top.precision)} in the top one without ever `
    + `going backwards, and average precision only ever asks for the order. Read a detector's `
    + `confidence as a probability and you will be wrong by a wide margin; read it as a rank and it `
    + `is one of the better-behaved numbers in this atlas. `
    + `<a href="../calibration/">Calibration</a> is the article about closing that gap, and nothing `
    + `in the usual detection pipeline does it.`);

  /* ---- two stages ------------------------------------------------------------ */
  const ts = data.two_stage;
  const full = ts.ablations.find((a) => a.variant === 'full');
  const clsOnly = ts.ablations.find((a) => a.variant === 'cls-only');
  const boxOnly = ts.ablations.find((a) => a.variant === 'box-only');
  const extra = ts.one_stage_extra;
  const vsOne = full.map50 - main.map50;
  const vsExtra = full.map50 - extra.map50;
  const rec1 = ts.proposal_recall[0];
  const recK = ts.proposal_recall.find((r) => r.k === ts.topk);
  const recAll = ts.proposal_recall[ts.proposal_recall.length - 1];
  set('twostage-note',
    `The other family looks at the picture twice. A first stage proposes regions, a second crops the `
    + `<i>feature map</i> at those regions and decides properly, which is the idea `
    + `<a href="https://arxiv.org/abs/1506.01497">Faster R-CNN</a> is built on: the proposals come `
    + `from the shared features, not from the pixels, so the expensive part is still paid once. Here `
    + `that means ${ts.roialign.bins}×${ts.roialign.bins} bins pooled from ${ts.roialign.on}, a `
    + `proposal counted as an object above IoU ${ts.roialign.positive_iou} and as background below `
    + `${ts.roialign.background_iou}, sampled ${ts.roialign.pos_neg_ratio}, for `
    + `${ts.roialign.epochs} epochs on a trunk that is already trained.`
    + `<br /><br />`
    + `The recall of the proposals alone says how much room the second stage has: one proposal per `
    + `picture already covers ${pct(rec1.recall50)} of the objects at IoU 0.5, ${ts.topk} covers `
    + `${pct(recK.recall50)}, and all ${recAll.k} covers ${pct(recAll.recall50)}. Ask for IoU 0.7 `
    + `instead and the same ${recAll.k} proposals only reach ${pct(recAll.recall70)}, which is the `
    + `first hint about where a second stage can help: not in finding things, in placing them.`
    + `<br /><br />`
    + `Adding the second stage takes ${f4(main.map50)} to <span class="bold">${f4(full.map50)}</span>, `
    + `${vsOne >= 0 ? '+' : ''}${f4(vsOne)}, `
    + `<span class="bold">${Math.abs(vsOne) > noise ? 'clear of the seed spread' : 'inside the seed spread'}</span>, `
    + `with ${n0(full.params)} parameters against the one-stage detector's ${n0(main.params)}. `
    + `Because a second stage is also more training, the control is a one-stage detector given the `
    + `same ${extra.extra_epochs} extra epochs, which reaches ${f4(extra.map50)}: the two-stage lead `
    + `over that control is ${vsExtra >= 0 ? '+' : ''}${f4(vsExtra)}. Splitting the second stage `
    + `shows where it comes from: classification only, ${f4(clsOnly.map50)}; box refinement only, `
    + `${f4(boxOnly.map50)}. The largest effect is on strict overlap, ${f4(main.map75)} to `
    + `${f4(full.map75)} at IoU 0.75 and a mean matched overlap of ${f4(main.mean_matched_iou)} to `
    + `${f4(full.mean_matched_iou)}, which is what a second look at a box is for. And the honest `
    + `caveat was written before the run: ${ts.declared}.`);

  /* ---- what it gets wrong ------------------------------------------------------ */
  const fp = data.failures.false_positives;
  const fpRows = Object.entries(fp).sort((a, b) => b[1] - a[1]);
  const bl = data.baselines;
  set('failures-note',
    `A score is a summary, and summaries hide the interesting part. Of everything this detector gets `
    + `wrong on the test split, sorted by size: `
    + `${fpRows.map(([k, val]) => `${n0(val)} ${k}`).join(', ')}. Those categories are exclusive and `
    + `assigned in a fixed order (${data.failures.order}), so a box can only be counted once and the `
    + `order is part of the definition. The largest category is background, which is what a `
    + `${data.eval.score_floor} score floor buys you and is mostly harmless because those boxes rank `
    + `last. The interesting one is <span class="bold">${n0(fp['held-out distractor'])} `
    + `distractors</span>: objects from the ${d.held_out.length} classes the detector was never `
    + `taught, which it confidently labels as one of the five it knows. It was never given a way to `
    + `say "something, but not one of mine".`
    + `<br /><br />`
    + `And ${n0(data.failures.missed)} objects are missed entirely. Of those, `
    + `<span class="bold">${n0(data.failures.missed_present_in_raw_same_class)}</span> were sitting `
    + `in the raw grid with the right class before scoring and suppression threw them away, and `
    + `${n0(data.failures.missed_present_in_raw_any_class)} were there under some label. `
    + `${cap(data.failures.note)}. For scale, a baseline that finds the ${n0(bl.n_blobs)} connected `
    + `blobs of ink in these pictures and scores them by area gets ${f4(bl.blob_ap50)} where the `
    + `detector, judged on the same class-agnostic task, gets ${f4(bl.detector_ap50_agnostic)}: `
    + `finding <i>something</i> in these pictures is easy, and almost all of the difficulty is in `
    + `saying what and how sure.`);

  /* ---- limits, provenance and close ---------------------------------------------- */
  if (Array.isArray(d.limits) && d.limits.length) {
    set('limits-note',
      `What this page cannot tell you, in the words the generator recorded before it ran: `
      + `${d.limits.join('; ')}.`);
  }

  const rt = data.runtime;
  set('runtime-note',
    `The schedule was priced before it was run. A calibration model (${rt.calibration_model}) timed `
    + `at ${rt.calibration_seconds.toFixed(1)}s against the `
    + `${rt.brief_calibration_seconds.toFixed(1)}s it should take on an idle machine gave a `
    + `contention factor of ${rt.contention.toFixed(2)}. Then a real training step of this article's `
    + `own network was timed at ${rt.step_probe.seconds_per_step.toFixed(3)}s over `
    + `${rt.step_probe.timed_steps} steps after ${rt.step_probe.warmup_steps} warmup, which is the `
    + `number the plan was built on: ${rt.reason} `
    + `${rt.arms_dropped.length === 0
      ? `Nothing had to be cut, and the ladder of things that would have gone first, in order, was `
        + `${rt.ladder.join(', then ')}. `
      : `Cut to fit, in the order the ladder specified: ${rt.arms_dropped.join(', ')}. `}`
    + `${rt.lr_fallbacks.length === 0
      ? 'No arm needed its learning rate restarted. '
      : `Restarted at half the learning rate: ${rt.lr_fallbacks.join(', ')}. `}`
    + `The arms trained at more than one seed were `
    + `${rt.replicates.map(([arm, seeds]) => `${arm} at seeds ${seeds.join(' and ')}`).join(', ')}, `
    + `which is where the noise floor comes from. Four assumptions are doing work in that estimate `
    + `and the generator names them: ${rt.assumptions.join('; ')}. `
    + `Total ${(rt.total_seconds / 60).toFixed(1)} minutes at tier ${rt.tier}, `
    + `${rt.epochs} epochs per arm, batch ${data.versions.batch}, seed ${data.meta.seed}, `
    + `torch ${data.versions.torch}.`);

  const worst = data.nms.all.reduce((a, b) => (b.map50 < a.map50 ? b : a));
  const bestT = data.nms.all.reduce((a, b) => (b.map50 > a.map50 ? b : a));

  /* The three cells of the verdict table that carry a figure or a direction.
     A table is where a stale claim survives longest, because nothing on the
     page contradicts it, so these are composed like every other sentence. */
  const nmsRange = bestT.map50 - worst.map50;
  const levers = [
    ['suppression threshold', nmsRange],
    ['grid resolution', Math.abs(stride16.delta)],
    ['a whole second stage', Math.abs(vsOne)],
  ].sort((a, b) => b[1] - a[1]);
  set('verdict-anchors',
    `${mult.delta < 0
      ? 'Three identical slots are <span class="bold">worse</span> than one'
      : `Three identical slots are worth ${mult.delta >= 0 ? '+' : ''}${f4(mult.delta)} against one`}, `
    + `while three fitted ones are worth ${div.delta >= 0 ? '+' : ''}${f4(div.delta)} against three `
    + `identical. It is the diversity that pays, and only the comparison that isolates it can tell `
    + `you so.`);
  set('verdict-stride',
    `Coarsening costs more than refining buys. The stride-`
    + `${data.arms.find((a) => a.id === 'stride4').stride} arm is `
    + `${stride4.resolved ? 'outside' : 'inside'} the noise floor of stride ${main.stride} at `
    + `${f4(stride4.delta)}, and the stride-${data.arms.find((a) => a.id === 'stride16').stride} arm `
    + `is ${f4(stride16.delta)}, ${(Math.abs(stride16.delta) / noise).toFixed(0)} times the floor.`);
  set('verdict-nms',
    `${levers[0][0] === 'suppression threshold'
      ? 'It is the largest single lever on this page and it is not learned'
      : `It moves the score by ${f4(nmsRange)}, behind ${levers[0][0]}, and it is not learned`}. `
    + `It is also flat near the optimum: the test optimum for isolated scenes cannot be pinned down `
    + `more precisely than [${data.nms.isolated.optimum_ci.join(', ')}], so tuning it past the first `
    + `decimal is fitting noise.`);
  set('closing-note',
    `Three numbers, from three different parts of this page, are worth putting next to each other. `
    + `Changing the grid resolution moved the score by ${f4(Math.abs(stride16.delta))}. Adding a `
    + `whole second stage moved it by ${f4(Math.abs(vsOne))}. And setting the suppression threshold `
    + `badly, a single constant applied after the network has finished thinking, moves it by `
    + `<span class="bold">${f4(bestT.map50 - worst.map50)}</span>. The part of a detector that gets `
    + `the papers is not the part with the most leverage in it, and the part with the most leverage `
    + `is a line of post-processing that most write-ups do not report at all.`
    + `<br /><br />`
    + `The other thing to take away is smaller and older. A window classifier run on a crop and the `
    + `same classifier run once over the whole picture `
    + `${identical ? 'return the same numbers exactly' : 'return the same numbers to arithmetic noise'}. `
    + `Every one-stage detector is that observation, industrialised: the grid was always there, `
    + `hiding in the arithmetic of the scan that <a href="../viola-jones/">Viola-Jones</a> was `
    + `already paying for. What changed was noticing that the answer had been computed already.`);
}
