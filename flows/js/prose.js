/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const G = data.guards;
  const D = data.depth;
  const A = data.additive;
  const K = data.mask;
  const T = data.topology;
  const B = data.board;
  const N = data.net;
  const shown = D.rows.find((r) => r.depth === M.depth_show);
  const best = D.rows.find((r) => r.depth === D.best);
  const one = D.rows[0];
  const deep = T.rows[T.rows.length - 1];
  const shallow = T.rows[0];
  const rr = T.radii[T.radii.length - 1];

  set('opening-note',
    `Fitted to the ring and the two blobs this module measures everything against, a flow of `
    + `${M.depth_show} coupling layers lands ${shown.kl} nats from the truth, and that number is `
    + `an integral rather than an estimate. Its density integrates to ${G.mass} over the `
    + `quadrature square, its analytic stretching factor matches a central difference to `
    + `${G.jacobian_gap}, and pushing points forward and back returns them to `
    + `${G.round_trip}. Then there is the ring's hole: the truth puts `
    + `${shallow[`truth_${rr}`]} of its mass inside radius ${rr}, and this flow puts `
    + `<span class="bold">${shown ? deep[`model_${rr}`] : ''}</span>. That is not a failure of `
    + `training, and no amount of depth fixes it.`);

  set('scrolly-intro',
    `A coupling layer is the trick that makes all of this cheap. Split the coordinates in two, `
    + `let one half through untouched, and use it to decide a scale and a shift for the other `
    + `half. The map is invertible by inspection, its Jacobian is triangular so its determinant `
    + `is the product of the scales, and the network inside it can be anything at all because it `
    + `is never inverted.`);

  set('scrolly-step-0',
    `The flow starts from a standard gaussian, and the lines are a square grid drawn on it. `
    + `Everything the model does is move this grid: the density at a point is the gaussian's `
    + `density divided by how much the grid was stretched there.`);
  set('scrolly-step-1',
    `One coupling layer. One coordinate is untouched, so the lines running along it stay exactly `
    + `where they were, and the other coordinate is scaled and shifted by an amount that depends `
    + `on the first. Half a transformation.`);
  set('scrolly-step-2',
    `The second layer swaps which half is which, which is the whole reason the mask alternates. `
    + `Without it the first coordinate would never move at all, and the article measures exactly `
    + `what that leaves: a marginal that stays standard normal to within ${K.gap} across the `
    + `whole grid.`);
  set('scrolly-step-3',
    `After all ${M.depth_show}, with the data it was fitted to drawn on top. The grid is dense `
    + `where the density is high, which is the same thing as saying the map stretched space `
    + `there, which is the same thing as saying the determinant is small there.`);
  set('scrolly-step-4',
    `And look at what the grid never does: cross itself, fold, or open. Every one of those `
    + `lines started as a straight line in the gaussian and is still a curve that does not meet `
    + `any other, because the map is invertible. That is the constraint the last section of this `
    + `page charges the flow for.`);

  set('coupling-note',
    `Both halves of that bargain are checked here rather than assumed. Over ${G.probe} points, `
    + `the log determinant the formula gives and the one a central difference of the map gives `
    + `agree to ${G.jacobian_gap} at a step of ${G.jacobian_step}, and pushing those points `
    + `forward and inverting them returns them to ${G.round_trip}. The model's density `
    + `integrates to ${G.mass} on the same square where the closed form truth integrates to one, `
    + `so ${G.leak} of it sits outside the box: a flow spreads its gaussian everywhere the `
    + `gaussian reaches, and the truth does not.`);

  set('depth-intro',
    `Depth, scored by the exact distance from the truth rather than by an average over held out `
    + `rows. The two are not the same number and this module has both: the second is what every `
    + `other article has to use, and the first is what two dimensions buy.`);

  set('depth-note',
    `From ${one.depth} layer to ${best.depth}, the distance from the truth falls from ${one.kl} `
    + `to ${best.kl}. `
    + `${D.flat_from
      ? `The staircase is flat from ${D.flat_from} layers on, and that is a claim about every `
        + `later row rather than about the first one that happened to qualify: no depth after it `
        + `is more than twice the seed floor of ${D.floor} above the best.`
      : `No depth in this sweep passes the flatness test, which asks that every later row stay `
        + `within twice the seed floor of ${D.floor} of the best, so the page does not use the `
        + `word.`}`
    + `${D.backwards.length
      ? ` It does go backwards between ${D.backwards.map((p) => `${p[0]} and ${p[1]}`).join(' and between ')} layers, which a curve drawn without seed ranges would have hidden.`
      : ' It never goes backwards by more than that floor.'}`
    + `<br /><br />`
    + `Two ablations sit on the same axes. Removing the scale, so that every layer only shifts `
    + `and the map preserves volume exactly, costs ${A.rows.find((r) => r.depth === M.depth_show).gap} `
    + `nats at ${M.depth_show} layers, and the guard that it really is volume preserving is that `
    + `its log determinant is ${A.logdet_max} at worst. Removing the alternation of the mask is `
    + `not an ablation, it is a different model: the coordinate that no layer ever transforms `
    + `comes out exactly standard normal, so its marginal sits ${K.marginal_vs_model} nats from `
    + `the truth's, which is the ${K.marginal_vs_normal} of the untouched prior, and the total `
    + `distance is ${K.kl} against ${K.kl_alternating} with the mask alternating.`);

  set('hole-intro',
    `And now the restriction that has no workaround. A flow is a diffeomorphism, a smooth `
    + `invertible map with a smooth inverse, applied to a gaussian. A gaussian is connected and `
    + `has no hole, and a diffeomorphism cannot make one: it can stretch a region until it is `
    + `thin, but it cannot cut it. The density this module uses has a ring, and a ring has a `
    + `hole in it.`);

  set('hole-note',
    `So the flow builds a bridge. The mass it puts inside radius ${rr} is `
    + `${deep[`model_${rr}`]} against the truth's ${deep[`truth_${rr}`]}, a factor of `
    + `${n0(T.ratio_deepest)}, and it does not fall with depth. What does change with depth is `
    + `how hard the map is working: the worst stretch at the data goes from ${shallow.cond_max} `
    + `to ${deep.cond_max}. That is the shape of the whole limitation, and it is why the `
    + `<a href="../ebm/">energy model</a> has no trouble here at all: an energy is free to be `
    + `high in the middle of the ring, because nothing requires it to be the image of anything.`);

  set('board-intro',
    `Which makes this the place to put the module's three models side by side, on one density, `
    + `with one grid, and see what each of them can actually report.`);

  const boardRows = [];
  if (B.vae) {
    boardRows.push(['A variational autoencoder',
      'a lower bound on the log likelihood, and nothing else',
      `bound ${B.vae.bound}, and the number it bounds is ${B.vae.exact}`,
      `the bound is loose by ${B.vae.gap} nats, and on any real data that gap is unknown`]);
  }
  if (B.ebm && B.ebm.kl !== undefined && B.ebm.kl !== null) {
    boardRows.push(['An energy model',
      'an unnormalised score: a density only after an integral nobody can do',
      `${B.ebm.kl} nats from the truth, which only two dimensions make computable`,
      'in any real dimension the constant is unavailable, so two of them cannot be compared']);
  }
  boardRows.push(['A flow',
    'the exact log likelihood, by an identity',
    `${B.flow.exact} nats per held out row, ${B.flow.kl} from the truth`,
    `it cannot make a hole: ${B.flow.hole} inside radius ${rr} against ${B.truth_hole}`]);
  const tb = document.querySelector('#board-table');
  if (tb) {
    tb.innerHTML = boardRows.map((r) => `<tr>${r.map((c, i) => (i === 0
      ? `<td><span class="bold">${c}</span></td>` : `<td>${c}</td>`)).join('')}</tr>`).join('');
  }

  set('board-note',
    `${B.vae
      ? `The first row is the one to sit with. The variational autoencoder's own objective `
        + `reported ${B.vae.bound} and the quantity it was bounding was ${B.vae.exact}: a gap of `
        + `${B.vae.gap} nats that nothing in its training would have revealed. This flow reports `
        + `${B.flow.exact} and that is the number, with no gap, because the change of variables `
        + `is an identity. The truth's own is ${B.ceiling}, and it is the same ${B.ceiling} in `
        + `all three articles because it is a property of the density rather than of any model.`
      : `The other articles' files were not present when this ran, so the table shows only what `
        + `this one measured.`}`);

  set('verdict-intro',
    `Four questions. A flow answers two of them better than anything else in the module and one `
    + `of them not at all.`);
  set('verdict-exact',
    `${B.flow.exact} nats per held out row, exactly, with no bound and no constant. On the same `
    + `data the distance from the truth is ${B.flow.kl} nats, and both numbers are integrals.`);
  set('verdict-exact-warn',
    `The identity only holds if the map is invertible and its determinant is right, so both are `
    + `guards here: ${G.round_trip} and ${G.jacobian_gap}.`);
  set('verdict-sample',
    `One pass through the inverse per sample, which is why the pictures on this page are drawn `
    + `by the reader's own machine and the energy article's had to be animated.`);
  set('verdict-sample-warn',
    `The architecture is constrained in exchange: at ${M.depth_show} layers the model has `
    + `${n0(shown.params)} weights and half of them are spent conditioning rather than `
    + `transforming.`);
  set('verdict-hole',
    `Nothing in this module handles the hole except the energy model. The flow puts `
    + `${deep[`model_${rr}`]} where the truth puts ${deep[`truth_${rr}`]}.`);
  set('verdict-hole-warn',
    `And it hides: the density looks right, the likelihood is high, and only measuring the mass `
    + `in the region shows it. Nothing in the training loss can see this.`);
  set('verdict-mask',
    `Alternating masks are not a detail: without them the model reaches ${K.kl} instead of `
    + `${K.kl_alternating}, and one coordinate is left exactly as it started.`);
  set('verdict-mask-warn',
    `Volume preserving layers cost `
    + `${A.rows.find((r) => r.depth === M.depth_show).gap} nats at the same depth, so "cheaper `
    + `determinant" is not free either.`);

  set('closing-note',
    `So the model that refuses to approximate gets the exact number, and pays for it with a `
    + `shape it cannot make. The trade is visible on one page only because the data is two `
    + `dimensional, and that is the last time this module will have that luxury.`
    + `<br /><br />`
    + `Because an exact likelihood on a plane is a curiosity, and on images it is a claim about `
    + `compression: so many bits per pixel, comparable with programs people actually ship. `
    + `<a href="../glow/">The last article of the module</a> puts this same machinery on `
    + `pictures, measures its bits against gzip and PNG, and asks the question that only a model `
    + `with an exact likelihood can be embarrassed by: which of two datasets does it think is `
    + `more likely?`);

  set('resources-note',
    `${n0(M.n_train)} points from the closed form density of this module, ${n0(M.n_test)} held `
    + `out, and a RealNVP style flow of affine coupling layers with two hidden layers of `
    + `${M.hidden} units inside each one and the scale bounded at ${M.s_max}, trained by exact `
    + `maximum likelihood for ${M.epochs} epochs on batches of ${M.batch} at a step of ${M.lr}, `
    + `over seeds ${M.seeds.join(', ')}. Every distance from the truth on this page is the exact `
    + `integral on a ${n0(M.grid)} by ${n0(M.grid)} grid, where the closed form density `
    + `integrates to one to ${G.quadrature[G.quadrature.length - 1].error}. The flow the page `
    + `runs is ${n0(N.count)} numbers in float16: quantising them moves a log density by at most `
    + `${N.quant.logp_shift} and the held out average from ${N.quant.held_full} to `
    + `${N.quant.held_quantised}. Run on ${M.threads} threads with torch ${M.torch}.`);
}
