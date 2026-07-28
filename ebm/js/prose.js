/* Every figure on this page, composed from the file it was measured into.
 *
 * This article has the widest seed spreads in the module, so almost every
 * sentence here is a branch on whether a difference is bigger than the floor,
 * and several of them come out saying that it is not.
 */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.truth;
  const A = data.arms;
  const S = data.sampler;
  const NN = data.nonoise;
  const C = data.classifier;
  const MD = data.model.arms.exact;
  const exact = A.rows.find((r) => r.mode === 'exact');
  const cd1 = A.rows.find((r) => r.mode === 'cd' && r.k === 1);
  const cdN = A.rows.filter((r) => r.mode === 'cd').slice(-1)[0];
  const pcd1 = A.rows.find((r) => r.mode === 'pcd' && r.k === 1);
  const ulaSmall = S.rows.filter((r) => r.sampler === 'ula')[0];
  const ulaBig = S.rows.filter((r) => r.sampler === 'ula').slice(-1)[0];
  const malaBig = S.rows.filter((r) => r.sampler === 'mala').slice(-1)[0];
  const noise0 = NN.rows[0];
  const real = C.rows.find((r) => r.pool === 'real');
  const noise = C.rows.find((r) => r.pool === 'uniform');
  const matched = C.rows.find((r) => r.pool === 'uniform_matched');

  set('opening-note',
    `The catch is the integral. Divide by it and the model is a density; refuse to and it is a `
    + `score with no units. Nobody computes it, because in any real problem it is an integral `
    + `over every image that could exist. So this page does it. The density this module is `
    + `measured against lives in two dimensions, where the integral is a sum over a grid, and `
    + `the sum is exact enough to be worth trusting: the closed form truth integrates to one on `
    + `it to ${T.quadrature[T.quadrature.length - 1].error}. That makes it possible to train the `
    + `same energy <span class="bold">with the gradient nobody can have</span>, and then measure `
    + `what every practical shortcut costs against it.`);

  set('scrolly-intro',
    `An energy is a landscape and training is two forces on it: push down where the data is, `
    + `push up where the model currently thinks the data is. The first force is free. The second `
    + `one requires samples from the model, which is the whole problem, because sampling from a `
    + `density you only know up to a constant means running a chain.`);

  set('scrolly-step-0',
    `${n0(M.n_train)} points from the closed form density, and an energy that starts flat. Low `
    + `energy will mean likely, so the landscape has to end up with valleys exactly where these `
    + `points are and nowhere else.`);
  set('scrolly-step-1',
    `The first force is easy: the gradient of the energy at the data, pushing it down. On its `
    + `own it would push the whole landscape down forever, which is why nothing here is a `
    + `probability until the second force arrives.`);
  set('scrolly-step-2',
    `The second force pushes the landscape up wherever the model puts mass, and that requires `
    + `samples from the model itself. The exactly trained arm gets them for free, because in two `
    + `dimensions the expectation is a sum over the same grid: it is the arm nobody in a real `
    + `problem can have, and it costs ${n0(exact.negative_evals)} energy evaluations against `
    + `${n0(cd1.negative_evals)} for one Langevin step per update.`);
  set('scrolly-step-3',
    `What ${n0(M.updates)} updates of the exact gradient produce. Its distance from the truth is `
    + `${exact.kl} nats, and the ring is there, and the two blobs are there, and the hole in the `
    + `middle of the ring is there, which is the part the <a href="../flows/">flows article</a> `
    + `cannot manage at all.`);
  set('scrolly-step-4',
    `The shortcut's landscape, and then the difference. One sampling step per update instead of `
    + `an exact expectation lands ${A.cd1_bias} nats further from the truth, and the difference `
    + `picture says where those nats went: the energy is not wrong everywhere, it is wrong in a `
    + `shape.`);

  set('gradient-note',
    `Read the second line again. The first term is the gradient of the energy at a data point, `
    + `which is a backward pass. The second is an expectation under the model, and the model is `
    + `the thing we are trying to fit, so the only way to take it is to sample from it, and the `
    + `only way to sample from an unnormalised density is to run a chain and hope it has `
    + `finished. Everything else on this page is about that hope. The gradient itself is not in `
    + `doubt: the generator checks its closed form against automatic differentiation to `
    + `${data.guards.true_gradient.closed_vs_autograd} and against central differences to `
    + `${data.guards.true_gradient.closed_vs_differences}.`);

  set('chains-intro',
    `A Langevin chain walks downhill on the energy and adds a specific amount of noise on every `
    + `step, and if the steps were infinitely small it would end up distributed exactly as the `
    + `model says. The steps are not infinitely small. The walkers below are running in this `
    + `page, on the trained energy, and the chart beside them is what the generator measured `
    + `about the same settings.`);

  set('chains-note',
    `The error of an unadjusted chain does not go to zero: it goes down with the step size, and `
    + `at ${ulaBig.eps} it is ${ulaBig.kl_ring} nats while at ${ulaSmall.eps} it is `
    + `${ulaSmall.kl_ring}, against a floor of ${S.floor_ring} that a perfect sampler would `
    + `still show with this many samples. `
    + `${S.fit_ring.exponent
      ? `Over the ${S.fit_ring.points} steps that are far enough above that floor to say `
        + `anything, from ${S.fit_ring.eps_min} to ${S.fit_ring.eps_max}, the error falls as the `
        + `step to the power ${S.fit_ring.exponent} with a fit quality of ${S.fit_ring.r2}. `
        + `Three points is three points, and the page says so rather than drawing a law through `
        + `them.`
      : `There were not enough steps above the floor to fit an exponent, and the page says so `
        + `instead of drawing a line through two points.`} `
    + `The Metropolis correction removes the bias rather than shrinking it: at the same `
    + `${malaBig.eps} it scores ${malaBig.kl_ring} while accepting `
    + `${(malaBig.accept * 100).toFixed(1)}% of its proposals. `
    + `There is a second measurement in the widget that matters more for training than either: `
    + `at every step size where the ring still has roughly the right width, the fraction of `
    + `walkers that ever move between the ring and a blob is at most `
    + `${(Math.max(...S.rows.filter((r) => r.ring_inflation < 1.5).map((r) => r.switch)) * 100).toFixed(1)}%. `
    + `The chains are not mixing between the parts of the distribution. They are sitting in `
    + `whichever part they started in.`);

  set('noise-note',
    `And the noise is not decoration. Press the third button: the same walkers, the same step, `
    + `no noise, and they stop being a sampler and become gradient descent. On this landscape `
    + `the bottom of the ring is a circle rather than a point, so they do not all pile onto one `
    + `spot, which is what "they collapse onto the modes" would have predicted: `
    + `${n0(noise0.distinct_3)} of ${n0(NN.chains)} endpoints are still distinct at three `
    + `decimals. What collapses is the direction that carries the distribution. The ring they `
    + `draw has a width of ${noise0.ring_sd} against the true ${T.ring_width}, its radius sits `
    + `at ${noise0.ring_radius} against the ${NN.ring_minimum} the closed form predicts for the `
    + `minimum of that energy, and the error of the radial distribution is ${noise0.kl_ring} `
    + `nats against ${ulaSmall.kl_ring} for the identical chains with the noise put back.`);

  set('bias-intro',
    `Which brings the question this whole module has been building towards: how wrong is the `
    + `shortcut? Nine arms below, the same architecture and the same ${n0(M.updates)} updates, `
    + `differing only in where the second term of the gradient comes from.`);

  set('bias-note',
    `The ordering is the reliable part and it holds on every seed: more sampling steps land `
    + `closer to the truth, and the arm with the exact expectation lands closest. The sizes are `
    + `less reliable than they look, and the honest thing is to say so. The median spread `
    + `between seeds of the same arm is ${A.floor} nats, so `
    + `${A.cd1_bias > A.floor
      ? `one step costing ${A.cd1_bias} is outside it, ${cdN.k} steps costing ${A.cd100_bias} is `
        + `well inside it, and this page will not call the second one a difference`
      : `even the cost of one step, ${A.cd1_bias}, sits inside it`}. `
    + `What is not ambiguous is the persistent buffer: at one step it costs ${A.pcd1_bias} nats `
    + `and its three seeds land at ${pcd1.kl_seeds.join(', ')}, which is not an estimate of `
    + `anything. A buffer that never restarts is supposed to give the chain the whole run to mix `
    + `in; on a target whose parts are separated by a region of very low density, it instead `
    + `keeps whatever proportions it happened to start with, and the mixing measurement above is `
    + `the reason.`);

  set('jem-intro',
    `One more thing follows from the definition, and it is free. An ordinary classifier ends `
    + `with a vector of logits, and the logarithm of the sum of their exponentials is a number `
    + `per image that nobody trained and everyone has. Call minus that an energy and any `
    + `classifier is a density over images, up to the usual constant. The classifier here is the `
    + `one the whole module has used to score samples: ${(C.accuracy * 100).toFixed(2)}% on the `
    + `held out digits, trained on nothing but labels.`);

  set('jem-note',
    `So the claim is true and its content is not what it sounds like. This energy is a strong `
    + `detector of some things: it ranks a real digit above a shuffled one `
    + `${C.rows.find((r) => r.pool === 'shuffled').auc_energy} of the time and above the `
    + `ink matched noise ${matched.auc_energy} of the time. It also prefers uniform noise to `
    + `handwriting, ${noise.auc_energy}, which is not a rounding error, it is the wrong side of `
    + `a half. And plain softmax confidence, the thing the energy was supposed to improve on, `
    + `beats it on `
    + `${C.rows.filter((r) => r.auc_confidence !== null && r.auc_confidence !== undefined
      && r.auc_confidence > r.auc_energy).length} of the `
    + `${C.rows.filter((r) => r.auc_confidence !== null && r.auc_confidence !== undefined).length} `
    + `pools. The explanation is in the ink column and it is measurable: energy and total ink `
    + `correlate at ${C.ink_correlation} across all pools. A free energy model is a real thing, `
    + `and on this data most of what it has learned is brightness.`);

  set('verdict-intro',
    `An energy model is the least constrained generative model in this module and the least `
    + `usable, and every row below is a version of the same trade.`);
  set('verdict-shape',
    `No invertibility, no encoder, no architecture that has to be shaped around sampling: just a `
    + `function. It puts ${data.truth.holes ? '' : ''}the hole in the middle of the ring where `
    + `the truth puts it, which the flows article's model cannot.`);
  set('verdict-shape-warn',
    `Nothing it reports is a probability until you divide by an integral you cannot do, and its `
    + `training needs a sampler that has to work before the model does.`);
  set('verdict-number',
    `On this two dimensional problem the exactly trained arm sits ${exact.kl} nats from the `
    + `truth and its held out log density is ${exact.heldout}, both computable because the grid `
    + `gives the constant: ${MD.logz_train_grid}, stable to six decimals from a 64 by 64 grid to `
    + `a 1024 by 1024 one.`);
  set('verdict-number-warn',
    `In any real dimension that constant is unavailable, so two energy models cannot be compared `
    + `by likelihood at all, which is exactly what the next two articles can do.`);
  set('verdict-samples',
    `A chain gets there, and the correction makes it exact: ${malaBig.kl_ring} nats at a step of `
    + `${malaBig.eps} where the uncorrected chain scores ${ulaBig.kl_ring}.`);
  set('verdict-samples-warn',
    `It does not mix between separated parts of the distribution. At most `
    + `${(Math.max(...S.rows.filter((r) => r.ring_inflation < 1.5).map((r) => r.switch)) * 100).toFixed(1)}% `
    + `of walkers ever change component, so the proportions come from the initialisation.`);
  set('verdict-ood',
    `Free, from a classifier you already have, and genuinely informative on some pools: `
    + `${matched.auc_energy} against ink matched noise.`);
  set('verdict-ood-warn',
    `It scores ${noise.auc_energy} against uniform noise, meaning it prefers it, and it `
    + `correlates ${C.ink_correlation} with brightness. Check what your energy is actually `
    + `ranking before you deploy it as a detector.`);

  set('closing-note',
    `So the model that needs no concessions pays for it at the other end: the constant that `
    + `turns a score into a probability is exactly the thing nobody can compute, and every `
    + `practical training rule is an approximation whose error, measured here against a gradient `
    + `that is not an approximation, is ${A.cd1_bias} nats at one sampling step.`
    + `<br /><br />`
    + `There is one way out that has not been tried yet, and it is the opposite of this one. `
    + `Instead of writing any function and paying for the integral, write only functions whose `
    + `integral is one by construction: push a gaussian through a map that is invertible and `
    + `whose stretching factor you can compute, and the density comes out exact, with no `
    + `constant, no bound and no chain. <a href="../flows/">The next article</a> does that, and `
    + `measures what the restriction costs on the same ring this page just fitted.`);

  set('resources-note',
    `${n0(M.n_train)} points from the closed form density, ${n0(M.n_held)} held out, and an `
    + `energy of ${M.hidden} by ${M.hidden} softplus units plus an anchor term, trained for `
    + `${n0(M.updates)} Adam updates on batches of ${M.batch} at a step of ${M.lr}, over seeds `
    + `${M.seeds.join(', ')}. The exact arm normalises on a ${M.train_grid} by ${M.train_grid} `
    + `grid every update and every KL on the page is computed on a ${M.score_grid} by `
    + `${M.score_grid} one. The sampler study runs ${n0(S.chains)} chains with ${n0(S.burn)} `
    + `burn in steps and keeps ${n0(S.samples)} states. The classifier is the module's shared `
    + `judge, ${(C.accuracy * 100).toFixed(2)}% on the held out digits. The energy the page runs `
    + `is ${n0(MD.count)} numbers in float16: quantising moves its distance from the truth from `
    + `${MD.kl} to ${MD.kl_quantised} and the worst energy on the drawn grid by `
    + `${MD.quant_energy_shift}. Run on ${M.threads} threads with torch ${M.torch} and numpy `
    + `${M.numpy}. Cost is reported as energy evaluations rather than seconds, because that `
    + `number is the same on any machine.`);
}
