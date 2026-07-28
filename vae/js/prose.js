/* Every sentence on this page that carries a figure, composed from the file.
 *
 * Written this way for the reason the module keeps rediscovering: a number
 * typed into HTML survives the next time the generator runs, and goes on saying
 * what stopped being true. Every comparison below is a branch, so if a
 * re-measurement moves a result the sentence moves with it instead of quietly
 * lying about a direction.
 */
const n0 = (v) => v.toLocaleString('en-US');
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

/* how many times the noise floor a two sample statistic is, which is the only
 * scale it has */
const floors = (v, floor) => (v / floor).toFixed(v / floor < 10 ? 1 : 0);

export function initProse(data) {
  const M = data.meta;
  const H = data.hole;
  const C = data.compare;
  const E = data.estimators;
  const S = data.sweep.rows;
  const T = data.toy;
  const L = C.ladder;
  const two = L.find((r) => r.k === M.k_show);
  const wide = L[L.length - 1];
  const fl = H.floor.mmd2_max;
  const zero = S[0];
  const best = S.reduce((a, b) => (b.mmd2 < a.mmd2 ? b : a));
  const collapsed = S.filter((r) => r.active < M.k_collapse - 0.5);
  const k2 = E.rows.find((r) => r.k === M.k_show);
  const kBig = E.rows[E.rows.length - 1];
  const unit = H.arms.unit;

  set('opening-note',
    `It half works, and the half that fails is not the half anyone expects. Drawing `
    + `${n0(H.draws)} codes from a gaussian fitted to that network's own codes and decoding them `
    + `gives pictures that sit <span class="bold">${H.arms.gaussian.mmd2}</span> from real held `
    + `out digits in a kernel two sample statistic, where two pools of real digits sit `
    + `${H.floor.mmd2} apart and a classifier asked to separate the generated pool from a real `
    + `one gets ${H.arms.gaussian.c2st} instead of the ${H.floor.c2st} it manages on real against `
    + `real. So they are distinguishable, clearly. But that is `
    + `${H.arms.gaussian.mmd2 < H.reconstruction.mmd2
      ? `<span class="bold">closer</span> than the same network's own reconstructions of real `
        + `digits, which sit ${H.reconstruction.mmd2} away`
      : `worse than the same network's own reconstructions, at ${H.reconstruction.mmd2}`}, `
    + `and the failure only becomes total when the proposal is careless: draw from a standard `
    + `gaussian without fitting anything and the same decoder produces pictures ${unit.mmd2} `
    + `away, ${floors(unit.mmd2, fl)} times the floor, that a classifier separates `
    + `${pct(unit.c2st)} of the time. The lesson of the whole page is in that gap: what a plain `
    + `autoencoder is missing is not a good decoder, it is <span class="bold">a place to sample `
    + `from</span>, and it can be handed one after the fact with mixed results.`);

  set('scrolly-intro',
    `The data is the same ${n0(M.n)} MNIST digits the last four articles used, mean pooled from `
    + `28 by 28 to ${M.side} by ${M.side}, and the generator does not just say so: it rebuilds `
    + `them from the same seed and then runs the autoencoders article's own float16 encoder on `
    + `them, which reproduces that article's published codes to ${M.reuse.worst_code} where `
    + `quantising those weights alone accounts for ${M.reuse.floor}. ${n0(M.train)} are used for `
    + `fitting and ${n0(M.test)} are held out of everything here.`);

  set('scrolly-step-0',
    `An encoder that returns a point has nothing to say about where the points are. It puts `
    + `each digit somewhere and the somewheres form a cloud whose shape nobody chose and nobody `
    + `wrote down. The picture is that cloud, exactly as the <a href="../autoencoders/">bottleneck article</a> published it.`);
  set('scrolly-step-1',
    `Change one thing: the encoder returns a <span class="bold">distribution</span> instead of a `
    + `point, a mean and a width per digit, and training draws a sample from it before decoding. `
    + `The ellipses are those widths at one standard deviation, drawn for one digit in every `
    + `nine so the picture stays readable. The decoder now has to rebuild the digit from a `
    + `neighbourhood rather than from a point, which is already a change in what it has to be `
    + `good at.`);
  set('scrolly-step-2',
    `Then add the second term, which measures how far each of those clouds is from one fixed `
    + `standard gaussian and charges for it. That is the term that turns a cloud of codes into a `
    + `distribution you can name: not because it makes the codes tidy, but because it makes the `
    + `place you are going to sample from and the place the codes actually are into the same `
    + `place. At ${M.k_show} numbers this model pays ${C.kl_nats} nats for it.`);
  set('scrolly-step-3',
    `Which is what sampling is. Draw from the standard gaussian, decode, and look: the crosses `
    + `are draws from it, and they land where the digits are because that is what the `
    + `second term was buying. Nothing in the objective ever showed the model a picture it was `
    + `supposed to invent.`);
  set('scrolly-step-4',
    `The two terms pull against each other and the colours are the price each digit pays: `
    + `darker is more nats spent sitting where it sits. Turn the second term off and the model `
    + `is exactly the <a href="../autoencoders/">bottleneck article</a>'s; turn it up and the codes are squeezed until they carry `
    + `nothing at all. The sweep further down measures both ends.`);

  set('integral-note',
    `That integral is the whole problem. It has no closed form for any decoder worth using, so `
    + `nobody computes it, and the entire machinery below exists to avoid it. In two dimensions, `
    + `though, it is a sum over a grid, and the last section of this page does exactly that, `
    + `which is how the looseness of the bound stops being a matter of opinion.`);

  set('elbo-note',
    `The bound is the objective, and it is the sum of a term that wants the picture back and a `
    + `term that wants the code to be somewhere ordinary. Both are computable: the first by `
    + `decoding a sample, the second in closed form for gaussians. The gap between this bound `
    + `and the log likelihood it bounds is exactly the KL from the encoder's answer to the true `
    + `posterior, which is not computable either, and which this page measures anyway further `
    + `down: ${T.gap} nats on the two dimensional problem.`);

  set('plane-intro',
    `Two planes, the same ${n0(M.n)} digits, and two decoders that both run in this page. The `
    + `left one is the autoencoders article's network read out of that article's own data file; `
    + `the right one was trained here with the same widths, the same activations, the same `
    + `optimiser and one extra term. Drag the crosshair to compare what the two decoders make of `
    + `the same coordinates, and then press the button that draws a point instead of choosing `
    + `one.`);

  set('plane-note',
    `The dashed rings are what each model can offer as a proposal. The autoencoder has no prior `
    + `at all, so the honest thing to hand it is a gaussian fitted to its own codes, and that is `
    + `what is drawn: a tilted ellipse in the middle of its cloud. The variational one has the `
    + `standard gaussian it was trained towards, so its rings are circles at one and two, and `
    + `the cloud of codes sits inside them because ${C.kl_nats} nats of the objective were spent `
    + `putting it there.`);

  set('ladder-note',
    `And now the result that did not go the way the script said. Give both models the same `
    + `courtesy, a gaussian fitted to the codes each of them actually produces, and at `
    + `${M.k_show} numbers the <span class="bold">autoencoder is the one that samples `
    + `${two.ae.mmd2 < two.vae.mmd2 ? 'better' : 'worse'}</span>: ${two.ae.mmd2} against `
    + `${two.vae.mmd2}, with a floor of ${H.floor.mmd2} under both. The first thing to rule out `
    + `is that this is inherited from sharpness, since a model trained to rebuild a `
    + `neighbourhood makes blurrier pictures than one trained to rebuild a point. It is not: the `
    + `two models' reconstructions of held out digits sit ${two.ae.recon_mmd2} and `
    + `${two.vae.recon_mmd2} from real ones, a difference of `
    + `${Math.abs(C.decomposition.recon_gap).toExponential(1)} where the sampling difference is `
    + `${Math.abs(C.decomposition.sample_gap).toExponential(1)}.`
    + `<br /><br />`
    + `Two things are worth taking from the table. The first is that the variational model does `
    + `not sample best from its own prior: fitting a gaussian to its aggregate posterior instead `
    + `moves it from ${two.vae.prior_mmd2} to ${two.vae.mmd2}, which is the prior hole a trained `
    + `model still has, measured, even though its codes sit only ${two.vae.aggregate_kl} nats `
    + `from the standard gaussian against the autoencoder's ${two.ae.aggregate_kl}. The second `
    + `is what happens as the code widens: at ${wide.k} numbers the gap between the `
    + `autoencoder's fitted proposal and the variational model's own prior is `
    + `${Math.abs(wide.ae.mmd2 - wide.vae.prior_mmd2).toExponential(1)}, against `
    + `${Math.abs(two.ae.mmd2 - two.vae.prior_mmd2).toExponential(1)} at ${M.k_show}. What the `
    + `second term buys is not sharper pictures. It is that the sampler comes attached, and it `
    + `stays attached when the code stops being small enough to fit a gaussian to by eye.`);

  const ladderBody = L.map((r) => `
      <tr>
        <td><span class="bold">${r.k}</span></td>
        <td>${r.ae.mmd2} <span style="opacity:.7">(${floors(r.ae.mmd2, fl)}&times; floor)</span></td>
        <td>${r.vae.mmd2} <span style="opacity:.7">(${floors(r.vae.mmd2, fl)}&times; floor)</span></td>
        <td>${r.vae.prior_mmd2} <span style="opacity:.7">(${floors(r.vae.prior_mmd2, fl)}&times; floor)</span></td>
        <td>${r.ae.aggregate_kl} nats against ${r.vae.aggregate_kl}</td>
      </tr>`).join('');
  const tb = document.querySelector('#ladder-table');
  if (tb) tb.innerHTML = ladderBody;

  set('trick-intro',
    `There is a step in the paragraph above that should not have worked. The objective contains `
    + `an expectation over a random code, the code is drawn using parameters we are trying to `
    + `differentiate, and you cannot differentiate through a draw. There are two ways out, both `
    + `unbiased, and the difference between them is not theoretical.`);

  set('trick-note',
    `The first rewrites the draw as a fixed noise variable pushed through the parameters, so the `
    + `derivative goes through the sample and lands on the decoder. The second never `
    + `differentiates the objective at all: it weights the log derivative of the sampling `
    + `distribution by whatever the objective returned, which works for anything, including `
    + `things that are not differentiable. Below, both are run on the same problem, from the `
    + `same draws.`);

  set('variance-note',
    `On the real network the same measurement, over ${n0(k2.draws)} independent draws on a batch `
    + `of ${k2.rows} digits and the encoder's ${n0(k2.params)} weights, gives a total gradient `
    + `variance of ${k2.pathwise} for the pathwise rule against ${k2.score} for the score `
    + `function one: a factor of <span class="bold">${n0(k2.ratio)}</span> at ${k2.k} latent `
    + `numbers, rising to ${n0(kBig.ratio)} at ${kBig.k}. The two are unbiased for the same `
    + `thing and the file checks it rather than asserting it: their means differ by `
    + `${k2.mean_gap} against a standard error of ${k2.mean_gap_se}. `
    + `${k2.ratio_baseline < k2.ratio * 0.5
      ? `And the usual defence of the score function rule works: subtracting a running average `
        + `before weighting brings the factor down from ${n0(k2.ratio)} to `
        + `${n0(k2.ratio_baseline)}. It is still ${n0(Math.round(k2.ratio_baseline))} times `
        + `worse, which is why every variational autoencoder in use is written the other way.`
      : `Subtracting a running average before weighting barely helps here: the factor only `
        + `moves to ${n0(k2.ratio_baseline)}.`}`);

  set('beta-intro',
    `The two terms can be weighed against each other explicitly, and the weight has a name. At `
    + `zero, the second term is gone and the objective is squared error through a bottleneck: `
    + `the <a href="../autoencoders/">bottleneck article</a>'s model, exactly. Everything below is the same code, the same `
    + `${M.k_collapse} latent numbers, the same ${M.epochs} epochs and the same three seeds, `
    + `with one number changed.`);

  set('beta-note',
    `Three things move at once and they do not move together. The rate falls monotonically, from `
    + `${zero.kl} nats at zero to ${S[S.length - 1].kl} at ${S[S.length - 1].beta}. The held out `
    + `error is ${zero.mse} at zero, `
    + `${S.reduce((a, b) => (b.mse < a.mse ? b : a)).beta === 0
      ? 'and only gets worse from there'
      : `dips to ${S.reduce((a, b) => (b.mse < a.mse ? b : a)).mse} at beta = `
        + `${S.reduce((a, b) => (b.mse < a.mse ? b : a)).beta}, which is a mild surprise: a term `
        + `whose whole job is to constrain the code also regularises it`}, and then climbs to `
    + `${S[S.length - 1].mse}. And the samples are best at beta = ${best.beta}, at ${best.mmd2} `
    + `against ${zero.mmd2} at zero, a factor of ${(zero.mmd2 / best.mmd2).toFixed(0)}. `
    + `${collapsed.length
      ? `Past beta = ${collapsed[0].beta} the model starts abandoning dimensions: `
        + `${collapsed.map((r) => `${r.active} alive at ${r.beta}`).join(', ')}, out of `
        + `${M.k_collapse}. That is posterior collapse counted rather than described, and the `
        + `threshold is the usual one, ${M.active_threshold} nats in a dimension.`
      : `No dimension ever falls below ${M.active_threshold} nats in this sweep, so posterior `
        + `collapse does not appear at all at these settings.`} `
    + `The seed spread is in the table, and every claim here is bigger than it.`);

  set('bound-intro',
    `Everything so far was measured against real digits with a two sample statistic, because in `
    + `${M.pixels} dimensions there is no truth to compare against. So here is the same model on `
    + `a problem where there is one: ${n0(T.n_train)} points from a density written in closed `
    + `form, two gaussian blobs and a ring, fitted with the same objective and the same kind of `
    + `network. The point of two dimensions is not that it is easy. It is that the integral `
    + `defining the log likelihood is a sum over a grid, so the number the bound bounds can be `
    + `computed, and the generator computes it twice by unrelated routes: a quadrature per row `
    + `over the latent, and a splat of the whole prior through the decoder followed by a `
    + `convolution. They agree to ${T.guards.quadrature_gap}.`);

  set('bound-note',
    `The bound the model was trained on reports ${T.elbo} nats per held out row. The quantity it `
    + `is a bound on is ${T.exact}. The difference, <span class="bold">${T.gap} nats</span>, is `
    + `the KL from the encoder's answer to the true posterior, and it is not small: it is `
    + `${(T.gap / Math.abs(T.exact) * 100).toFixed(1)}% of the number being bounded. `
    + `${T.bounds[T.bounds.length - 1].gap < T.gap * 0.25
      ? `Importance weighting closes most of it, but slowly: ${n0(T.bounds[1].k)} samples get to `
        + `${T.bounds[1].bound}, ${n0(T.bounds[T.bounds.length - 1].k)} get to `
        + `${T.bounds[T.bounds.length - 1].bound}, still ${T.bounds[T.bounds.length - 1].gap} `
        + `short.`
      : `Importance weighting barely dents it: ${n0(T.bounds[T.bounds.length - 1].k)} samples `
        + `still leave ${T.bounds[T.bounds.length - 1].gap}.`} `
    + `And the exact number can finally be read against the only fixed point in the module: the `
    + `truth's own mean log density is ${T.ceiling} nats, so this fit is `
    + `${T.kl_from_ceiling} nats short of the best any model could be, which is the same thing `
    + `as the exact KL from the truth, ${T.kl_to_truth}. Every other article in this module gets `
    + `measured against those same two numbers.`);

  set('verdict-intro',
    `Four questions, and the answer is different for each of them, which is why "is a `
    + `variational autoencoder better than an autoencoder" has no answer.`);

  set('verdict-rebuild',
    `Squared error through a bottleneck is the objective that optimises rebuilding, and nothing `
    + `else competes with it at that. At ${M.k_show} numbers the two arms rebuild at `
    + `${two.ae.mse} and ${two.vae.mse}.`);
  set('verdict-rebuild-warn',
    `Its codes are a cloud nobody chose. Handed a careless proposal it produces ${unit.mmd2}, `
    + `${floors(unit.mmd2, fl)} times the floor.`);
  set('verdict-sample',
    `It comes with the place to sample from attached, and the wider the code the more that is `
    + `worth: from a standard gaussian at ${M.k_collapse} numbers, ${best.mmd2} against `
    + `${zero.mmd2} for the same network without the term.`);
  set('verdict-sample-warn',
    `At ${M.k_show} numbers a fitted gaussian on an autoencoder's codes `
    + `${two.ae.mmd2 < two.vae.prior_mmd2 ? 'beats it' : 'does not beat it'} `
    + `(${two.ae.mmd2} against ${two.vae.prior_mmd2}), so the argument is about what the model `
    + `guarantees, not about who wins today.`);
  set('verdict-bound',
    `The objective is a lower bound on a log likelihood, so it is at least comparable between `
    + `models of the same data, and it decomposes into two terms you can read separately.`);
  set('verdict-bound-warn',
    `It is a bound, and here it is loose by ${T.gap} nats out of ${T.exact}. Two models can be `
    + `ranked the wrong way round by it, and nothing in the training curve would say so.`);
  set('verdict-collapse',
    `The rate in nats is the diagnostic: it says how much the code actually carries, and how `
    + `many dimensions carry anything at all.`);
  set('verdict-collapse-warn',
    `${collapsed.length
      ? `Push the weight up and dimensions die: ${collapsed[collapsed.length - 1].active} of `
        + `${M.k_collapse} alive at beta = ${collapsed[collapsed.length - 1].beta}, and the `
        + `reconstruction error is ${collapsed[collapsed.length - 1].mse} by then.`
      : `Nothing collapsed in this sweep, but the rate is still the number to watch.`}`);

  set('closing-note',
    `So the promise the <a href="../autoencoders/">bottleneck article</a> made is kept, and it turned out to be worth less at two `
    + `numbers than the textbook suggests and much more at sixteen. What the variational `
    + `autoencoder really adds is not picture quality: it is that the model and the sampler are `
    + `the same object, that the code has a rate in nats you can read, and that the objective is `
    + `a bound on something with a name.`
    + `<br /><br />`
    + `The next thing to break is the code itself. Everything here is a point in a continuous `
    + `space, and the rate had to be measured in nats through a KL because a real number has no `
    + `natural size. Make the code an entry in a finite alphabet instead and the rate becomes `
    + `literal: so many cells, so many bits each, no estimate involved. That is `
    + `<a href="../vq-vae/">the next article</a>, where the gradient the encoder needs stops `
    + `existing and has to be replaced by one that does not exist either.`);

  set('resources-note',
    `${n0(M.n)} MNIST digits, seed ${M.seed}, mean pooled to ${M.side} by ${M.side} and split `
    + `${n0(M.train)} for fitting and ${n0(M.test)} held out, asserted identical to the `
    + `<a href="../autoencoders/">autoencoders article</a>'s by running that article's own `
    + `float16 encoder on them. The network is ${M.pixels} to ${M.hidden} to a mean and a `
    + `log variance and back, tanh hidden layers and a logistic output, with a gaussian `
    + `likelihood whose scale is learned (${C.sigma} here), trained with Adam for ${M.epochs} `
    + `epochs on batches of ${M.batch}. The sweep uses ${M.k_collapse} latent `
    + `numbers and seeds ${M.seeds.join(', ')}; the plane uses ${M.k_show}. Sample pools are `
    + `${n0(H.draws)} pictures scored against real held out digits with a radial basis kernel at `
    + `gamma ${H.floor.gamma}, whose floor was measured by splitting real digits in two `
    + `${H.floor.splits} times. The shared judge is a ${M.pixels} to ${M.judge_hidden} to 10 `
    + `network, ${pct(M.judge_accuracy, 2)} on the held out digits. The two dimensional section `
    + `uses ${n0(T.n_train)} points, a ${T.hidden} wide encoder and decoder, ${T.epochs} epochs, `
    + `and a quadrature grid of ${n0(T.grid)} by ${n0(T.grid)} on which the closed form density `
    + `integrates to one to ${T.quadrature[T.quadrature.length - 1].error}. The weights the page runs `
    + `are ${n0(data.net.count)} numbers in float16, and every figure quoted was measured on `
    + `those quantised weights: the difference is ${data.net.quant.code_shift} in the codes and `
    + `${data.net.quant.mse_full} against ${data.net.quant.mse_quantised} in the held out error. `
    + `Run on ${M.threads} threads with torch ${M.torch}.`);
}
