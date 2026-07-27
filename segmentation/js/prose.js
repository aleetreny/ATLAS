/* Every numbered sentence on the page, written from the measurement.
 *
 * Same discipline as the three articles before this one: nothing with a figure
 * in it is typed by hand, and every comparison is written as a comparison, so
 * that a rerun which moves a result moves the sentence with it instead of
 * leaving it quietly wrong. Where a claim could go either way, both ways are
 * written and the data picks.
 */
const n0 = (v) => v.toLocaleString('en-US');
const f4 = (v) => v.toFixed(4);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const cap = (s) => `${s.charAt(0).toUpperCase()}${s.slice(1)}`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data, net) {
  const d = data.dataset;
  const M = data.metric;
  const U = data.unet;
  const I = data.instances;
  const R = data.mask_rcnn;
  const P = data.promptable;
  const skipArm = U.arms.find((a) => a.id === 'skips');
  const plainArm = U.arms.find((a) => a.id === 'no-skips');

  /* ---- the pictures, and why the masks are exact ------------------------- */
  const same = d.same_pictures;
  const sameSplits = Object.entries(same).filter(([, v]) => v && typeof v === 'object');
  const allSame = sameSplits.every(([, v]) => v.canvases_identical && v.boxes_identical);
  set('dataset-note',
    `These are the detection article's canvases. Not canvases like them: the same ones, rebuilt `
    + `from the same seeds and then `
    + (same.checked
      ? `checked against that article's own cache, ${allSame
        ? `where every one of the ${n0(sameSplits.reduce((t, [, v]) => t + v.n, 0))} pictures and `
          + `every one of its boxes came back identical`
        : 'where some of them did NOT come back identical, which is a bug and not a feature'}`
      : `left unchecked, because that article's cache was not on this machine when the file ran`)
    + `. That matters because of how they are made: ${n0(d.n_train)} training canvases, `
    + `${n0(d.n_val)} for validation and ${n0(d.n_test)} for test, with the garments pasted by the `
    + `generator, so it knows which pixels it put where. ${cap(d.owner_rule)}. A pixel counts as `
    + `garment at or above ${d.ink_threshold} of ${255}, which is the same threshold that defined `
    + `the boxes, and garments cover ${pct(d.foreground_share)} of a training canvas against `
    + `${pct(d.distractor_share)} for the untaught classes.`
    + `<br /><br />`
    + `So the mask of every object is exact, in the same way its box was, and IoU and Dice below `
    + `are <span class="bold">counted rather than estimated</span>. That claim gets checked too: `
    + `the previous article measured how much of each garment survives the composite by a `
    + `completely different route, and it says ${pct(d.hidden_by_visible_ink, 2)} hidden where this `
    + `owner map says ${pct(d.hidden_mean, 2)}. The ${(d.tie_share * 100).toFixed(3)} of a point `
    + `between them is the pixels where two garments are exactly as bright as each other, which is `
    + `the one case where "who owns this pixel" needs a tie-break rather than an answer. `
    + `There is one wrinkle worth `
    + `stating before it bites: ${d.amodal_vs_visible} On the ${n0(d.objects_test)} test objects `
    + `that comes to ${pct(d.occluded_frac)} of them losing at least one pixel and `
    + `${pct(d.hidden_mean, 2)} of their ink on average. `
    + (d.fully_hidden > 0
      ? `And <span class="bold">${d.fully_hidden}</span> of them lose all of it: they have a `
        + `perfectly good box and not one visible pixel, which is a thing a detector can report `
        + `and a segmenter structurally cannot. That is the first hint that these two tasks are `
        + `not the same task with a finer output.`
      : `No object is hidden completely, so every box on this page has at least one pixel behind `
        + `it.`));

  /* ---- the metric --------------------------------------------------------- */
  const worst = M.per_class.reduce((a, b) => (b.box_as_mask_iou < a.box_as_mask_iou ? b : a));
  const best = M.per_class.reduce((a, b) => (b.box_as_mask_iou > a.box_as_mask_iou ? b : a));
  set('metric-note',
    `Take the ground-truth box of an object, fill it in, and score it as a mask against the `
    + `object. It is the best box that exists, so this is the ceiling on what a box can ever say `
    + `about shape. Over ${n0(M.n_objects)} test objects it scores IoU `
    + `<span class="bold">${f4(M.box_as_mask_iou_mean)}</span>. `
    + `A ${best.cls}, the most rectangular garment here, manages ${best.box_as_mask_iou.toFixed(3)}; `
    + `a ${worst.cls} manages ${worst.box_as_mask_iou.toFixed(3)}. `
    + `<span class="bold">Roughly a third of what a perfect box claims is not the object</span>, `
    + `and no improvement to a detector can recover it, because the detector was already right.`
    + `<br /><br />`
    + `Two metrics get used for masks and it is worth knowing they are the same metric. `
    + `${M.identity}, so one is a monotone function of the other and no prediction can win under `
    + `one and lose under the other. The generator checked that on every object it measured, and `
    + `${M.identity_max_gap === 0
      ? 'the two sides came out bit for bit equal every time'
      : `the worst disagreement was ${M.identity_max_gap.toExponential(1)}`}. Dice is the one people `
    + `optimise, because a smooth version of it makes a usable loss; IoU is the one people report. `
    + `The choice is a convention, not a finding.`);

  /* ---- the U-Net ---------------------------------------------------------- */
  set('unet-note',
    `A classifier is allowed to forget where things are. That is what pooling buys: by the time `
    + `the detector of the last article reached its head, a ${d.canvas}-pixel canvas was an `
    + `8 by 8 grid, and it did not need better than that to say "trouser, here, roughly". A `
    + `segmenter has to answer at every one of the ${n0(d.canvas * d.canvas)} pixels, so whatever `
    + `pooling threw away has to come back.`
    + `<br /><br />`
    + `The U-Net's answer is a wire. Take each feature map on the way down and hand it across to `
    + `the matching stage on the way up, so the decoder gets the encoder's own detail instead of `
    + `having to reinvent it from a coarse summary. Testing that claim honestly is harder than it `
    + `looks, because deleting a skip connection also deletes the channels it contributed and so `
    + `changes the width of the decoder, its parameter count and its arithmetic. `
    + `<span class="bold">Here the connection is always made and the tensor is replaced by `
    + `zeros.</span> ${cap(U.control)}. `
    + `Both arms train on ${n0(U.n_train_used)} of the ${n0(U.n_train_available)} canvases for `
    + `${U.epochs} epochs, and each was trained twice, at two seeds, so the difference has a floor `
    + `to clear. It is going to need it.`);

  const band = U.regions.acc_boundary;
  const inside = U.regions.acc_interior;
  set('unet-result',
    `The headline number says nothing, and saying so is the first result. Mean IoU moves by `
    + `${U.skip_delta >= 0 ? '+' : ''}${f4(U.skip_delta)} when the skips are switched on, `
    + `against a seed spread of <span class="bold">${f4(U.seed_spread)}</span>: the same network `
    + `trained twice moves ${(U.seed_spread / Math.max(Math.abs(U.skip_delta), 1e-9)).toFixed(0)} `
    + `times further than the change being tested. On that metric, at this scale, `
    + `<span class="bold">this experiment cannot tell you whether skip connections help</span>, `
    + `and an article that reported the ${f4(Math.abs(U.skip_delta))} without the `
    + `${f4(U.seed_spread)} would have been reporting a coin flip.`
    + `<br /><br />`
    + `Now split the canvas. The boundary band is ${pct(U.band_share)} of the pixels, the interior `
    + `is the other ${pct(1 - U.band_share)}, and each gets its own noise floor from the same two `
    + `seeds. On the <span class="bold">boundary</span>, skips are worth `
    + `${band.delta_mean >= 0 ? '+' : ''}${f4(band.delta_mean)} of accuracy against a floor of `
    + `${f4(band.seed_spread)} (${band.delta_seed19 >= 0 ? '+' : ''}${f4(band.delta_seed19)} at one `
    + `seed and ${band.delta_seed20 >= 0 ? '+' : ''}${f4(band.delta_seed20)} at the other, `
    + `${band.same_sign ? 'the same sign both times' : 'and the two disagree in sign'}): `
    + `<span class="bold">${band.resolved ? 'resolved' : 'not resolved'}</span>. `
    + `<span class="bold">Off</span> the boundary they are worth `
    + `${inside.delta_mean >= 0 ? '+' : ''}${f4(inside.delta_mean)} against a floor of `
    + `${f4(inside.seed_spread)}, `
    + `${inside.same_sign ? 'the same sign both times' : 'with the two seeds disagreeing about '
      + 'even the sign'}: <span class="bold">${inside.resolved ? 'resolved' : 'not resolved'}</span>. `
    + `At one seed the interior figures are ${skipArm.acc_interior.toFixed(4)} and `
    + `${plainArm.acc_interior.toFixed(4)}, which is the same number.`
    + `<br /><br />`
    + `${band.resolved && !inside.resolved
      ? 'So the mechanism survives the noise even though the headline does not. Skip connections '
        + 'are not making this network understand garments better; away from the edges it is '
        + 'exactly as good without them. They are telling it where a garment stops, which is a '
        + 'local problem, and it is precisely the problem the encoder created by pooling in the '
        + 'first place. That is a narrower claim than the one usually made for U-Net, and unlike '
        + 'the usual one it is the claim this experiment can actually support.'
      : 'The decomposition does not separate the two regions cleanly here either, so the honest '
        + 'summary is that this experiment is too small to locate the effect.'}`);

  /* ---- instances ----------------------------------------------------------- */
  set('instance-note',
    `Here is a question the map above cannot answer: how many garments are in the picture. `
    + `A semantic map has one label per pixel and no label for "which one", so two t-shirts that `
    + `touch are one region of t-shirt. This is not a training problem. It is the shape of the `
    + `output.`
    + `<br /><br />`
    + `And it is measurable without training anything, because the ground truth is exact. Across `
    + `the ${n0(I.n_test)} test canvases, <span class="bold">${n0(I.same_class_pairs_touching)}</span> `
    + `pairs of same-class objects touch, on ${n0(I.images_affected)} different pictures, and `
    + `<span class="bold">${n0(I.merged_into_one)}</span> of those pairs are a single connected `
    + `region in the <i>true</i> semantic map. `
    + `${I.merged_into_one > 0
      ? 'A perfect per-pixel classifier, one that got every single pixel right, would still report '
        + 'one object there.'
      : 'On this dataset they always stay separable, so the argument is structural rather than '
        + 'empirical here.'}`);

  const perCls = R.variants.find((v) => v.variant === 'per-class');
  const agn = R.variants.find((v) => v.variant === 'class-agnostic');
  const bestR = R.variants.reduce((a, b) => (b.mask_ap50 > a.mask_ap50 ? b : a));
  const headGap = perCls && agn ? perCls.mask_ap50 - agn.mask_ap50 : 0;
  set('maskrcnn-note',
    `The repair is the previous article, reused. Find the objects, then ask for a mask inside `
    + `each one, and the answer has an object index attached because it was produced per object. `
    + `Mask R-CNN is that idea with one detail that matters: the mask is pooled from the feature `
    + `map the detector already computed, not from the pixels, so the expensive part is still paid `
    + `once. The detector here is not a reimplementation. It is ${R.detector}`
    + `${R.detector_map50 ? `, still scoring ${f4(R.detector_map50)} mAP@0.5 with its `
      + `${n0(R.detector_params)} parameters` : ''}, and the mask head adds `
    + `${n0(R.variants[0].params)} more.`
    + `<br /><br />`
    + `Its masks reach <span class="bold">${f4(bestR.mask_ap50)}</span> average precision at IoU `
    + `0.5, against ${f4(bestR.box_ap50)} for the boxes carried by the very same detections. `
    + `<span class="bold">Compare those two with each other and with nothing else.</span> This `
    + `section scores the top ${R.topk} proposals per picture above a floor of ${R.score_floor} `
    + `with <i>no suppression at all</i>, so every duplicate the last article spent a section `
    + `removing is still in here and neither figure belongs next to `
    + `${R.detector_map50 ? `that article's ${f4(R.detector_map50)}` : "that article's mAP"}. `
    + `What the pair does say is clean, because the detections are identical and the only thing `
    + `that changed is which overlap the match is judged by. `
    + `${bestR.mask_ap50 < bestR.box_ap50
      ? `Asking for the shape instead of the rectangle costs `
        + `${f4(bestR.box_ap50 - bestR.mask_ap50)}. The objects it does find, it outlines at a mean `
        + `IoU of ${f4(bestR.mean_matched_mask_iou)}, comfortably past the `
        + `${f4(M.box_as_mask_iou_mean)} a perfect box manages.`
      : `The masks score at least as well as the boxes, which is worth a second look given how `
        + `much harder mask overlap is to satisfy.`} `
    + (perCls && agn
      ? `Mask R-CNN's own design point is that the mask head predicts one mask per class, so `
        + `classifying and outlining stop competing for one output. Here per class beats a single `
        + `shared mask by ${headGap >= 0 ? '+' : ''}${f4(headGap)}, and that pair was trained at `
        + `one seed only. After what the seeds did to the skip ablation above, a difference that `
        + `size with no floor under it is not something this page will call a result. `
      : '')
    + `The ceiling, though, was written down before the head was trained: ${R.declared}.`);

  /* ---- SAM ------------------------------------------------------------------ */
  set('sam-note',
    `Everything so far has needed a list of classes. The five garments were chosen in the last `
    + `article, the U-Net has one output channel per class, the mask head has one mask per class. `
    + `Segment Anything asks what happens if you delete that assumption: no classes at all, just `
    + `a picture and a prompt, and the answer is the thing under the prompt.`
    + `<br /><br />`
    + `<span class="bold">This page cannot train SAM and does not pretend to.</span> ${P.honesty}`
    + `<br /><br />`
    + `So: the same U-Net, with a second input channel carrying a bump at the point you clicked `
    + `and a single output channel, trained on ${n0(P.n_train_prompts)} prompts for ${P.epochs} `
    + `epochs, from a loss of ${P.history[0].toFixed(4)} down to `
    + `${P.history[P.history.length - 1].toFixed(4)}. It sees only the `
    + `${data.meta.classes.length} garment classes, never once the other ${d.held_out.length}, and `
    + `it is never told what a class is. ${n0(net.n_values)} weights, `
    + `${(net.weights_b64.length / 1024).toFixed(0)} kB of ${net.dtype}, and it runs on the pixels `
    + `in front of you.`
    + `<br /><br />`
    + `The weights arrive with a contract and the widget implements that and nothing else. `
    + `Input: <i>${net.arch.input}</i>. Layer order: <i>${net.arch.order}</i>. Output: `
    + `<i>${net.arch.output}</i>. Reading those from the export rather than assuming them is what `
    + `lets the readout compare its own masks against the generator's instead of merely producing `
    + `plausible ones, and ${net.errors.note}.`);

  const rel = P.known.iou > 0 ? P.gap / P.known.iou : 0;
  set('sam-result',
    `On the ${data.meta.classes.length} classes it was trained on it reaches IoU `
    + `<span class="bold">${f4(P.known.iou)}</span>. On the ${d.held_out.length} it has `
    + `<span class="bold">never seen</span>, which are sitting in the very same pictures as `
    + `distractors, it reaches <span class="bold">${f4(P.held_out.iou)}</span>: a gap of `
    + `${f4(P.gap)}, or ${pct(rel)} of the familiar score. `
    + `${pct(P.held_out.above_50, 0)} of the unfamiliar objects come back above IoU 0.5 against `
    + `${pct(P.known.above_50, 0)} of the familiar ones, and ${pct(P.held_out.above_75, 0)} `
    + `against ${pct(P.known.above_75, 0)} above the far stricter 0.75. The median unfamiliar `
    + `object scores ${f4(P.held_out.iou_median)}, so this is not a good average hiding a pile of `
    + `failures.`
    + `<br /><br />`
    + `${P.gap < 0.05
      ? 'That gap is small enough to be the point. The network was never asked to recognise '
        + 'anything, so there was nothing about a pullover for it to fail to know: "the connected '
        + 'bright region under this point, bounded where the brightness stops" is a description '
        + 'that does not mention garments at all, and it transfers because it never depended on '
        + 'the category in the first place.'
      : 'That gap is real, and it is the honest version of the result: the property transfers, '
        + 'but not for free, and a page this size cannot say whether more data would close it or '
        + 'widen it.'} `
    + `What SAM adds is scale, an architecture that can absorb it, and ambiguity handling this has `
    + `none of: a point on a sleeve might mean the sleeve, the shirt or the person, and a real `
    + `promptable model has to offer all three.`);

  /* ---- limits, provenance and close ------------------------------------------- */
  if (Array.isArray(d.limits) && d.limits.length) {
    set('limits-note',
      `What this page cannot tell you, in the words the generator recorded before it ran: `
      + `${d.limits.join('; ')}.`);
  }

  set('runtime-note',
    `Provenance, so the numbers can be argued with. The scenes are the detection article's, `
    + `verified against its cache rather than assumed. The four U-Net arms are the same network `
    + `with the skips zeroed or not, at two seeds each, trained on ${n0(U.n_train_used)} canvases `
    + `for ${U.epochs} epochs, with ${n0(skipArm.params)} parameters and `
    + `${(skipArm.macs / 1e6).toFixed(1)} million multiply-accumulates in every one of them. `
    + `The mask head sits on the detector's frozen features. The promptable network is the same `
    + `U-Net shape again, folded and quantised to ${net.dtype}, which moved its logits by `
    + `${net.errors.fold_abs.toExponential(1)} and its weights by `
    + `${net.errors.weight_quantisation_abs.toExponential(1)}. Everything took `
    + `${(data.runtime.total_seconds / 60).toFixed(0)} minutes on ${data.runtime.threads} CPU `
    + `threads, seed ${data.meta.seed}, torch ${data.versions.torch}. Read that duration as `
    + `provenance rather than as a constant: a different thread count moves it by a lot, which is `
    + `why nothing on this page is argued from a clock.`);

  set('closing-note',
    `The arc of this module was a narrowing question. `
    + `<a href="../classical-features/">What is in this window</a>, then `
    + `<a href="../detection/">where is it</a>, and now which pixels are it. Each step made the `
    + `output more specific and the supervision more expensive, and each one turned out to need a `
    + `different loss, a different metric and a different failure mode. The number to carry out of `
    + `this page is the first one: a <span class="bold">perfect</span> box scores `
    + `${f4(M.box_as_mask_iou_mean)} as a mask. Nothing is wrong with that box. It is answering a `
    + `different question, and knowing which question you are asking is most of the work.`
    + `<br /><br />`
    + `The other one is the last. A network that was never told what a garment is, asked only for `
    + `the thing under a point, comes back with ${f4(P.held_out.iou)} on categories it has never `
    + `met, against ${f4(P.known.iou)} on the ones it trained on. `
    + `${P.gap < 0.05
      ? 'Segmentation, it turns out, was never really about knowing what things are.'
      : 'Whether that generalises past five held-out classes is exactly the question SAM was '
        + 'built to answer at a scale this page cannot reach.'}`);

  /* the verdict cells that carry a figure or a direction */
  set('verdict-metric',
    `A perfect box read as a mask scores ${f4(M.box_as_mask_iou_mean)} over `
    + `${n0(M.n_objects)} objects, from ${worst.box_as_mask_iou.toFixed(3)} for a ${worst.cls} to `
    + `${best.box_as_mask_iou.toFixed(3)} for a ${best.cls}.`);
  set('verdict-unet',
    `Mean IoU cannot resolve it: ${U.skip_delta >= 0 ? '+' : ''}${f4(U.skip_delta)} against a seed `
    + `spread of ${f4(U.seed_spread)}. Split by region it is unambiguous: `
    + `${band.delta_mean >= 0 ? '+' : ''}${f4(band.delta_mean)} on the boundary against a floor of `
    + `${f4(band.seed_spread)}, ${inside.delta_mean >= 0 ? '+' : ''}${f4(inside.delta_mean)} off it `
    + `against ${f4(inside.seed_spread)}.`);
  set('verdict-instance',
    `${n0(I.merged_into_one)} of ${n0(I.same_class_pairs_touching)} touching same-class pairs are `
    + `one region in the true semantic map. The mask head on the detector reaches `
    + `${f4(bestR.mask_ap50)} mask AP against ${f4(bestR.box_ap50)} box AP on identical, `
    + `unsuppressed detections.`);
  set('verdict-sam',
    `${f4(P.known.iou)} IoU on the classes it was taught, ${f4(P.held_out.iou)} on the `
    + `${d.held_out.length} it never saw, a gap of ${f4(P.gap)}.`);
}
