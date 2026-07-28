/* Every sentence on this page that carries a figure, written from the file. */
const n0 = (v) => v.toLocaleString('en-US');
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const L = data.linear;
  const C = data.curve.rows;
  const G = data.generalisation;
  const W = data.wide;
  const R = data.roll;
  const two = C.find((r) => r.k === M.k_show);
  const best = C.reduce((a, b) => (b.gain > a.gain ? b : a));
  const biggest = C[C.length - 1];

  set('opening-note',
    `Three numbers say what changes. With every activation function removed, the network below `
    + `finds a plane <span class="bold">${L.subspace_gap.toExponential(0)}</span> away from PCA's, `
    + `which is a theorem showing up, and its two directions align with the principal components at `
    + `between ${L.worst_alignment} and ${L.best_alignment}, which is the same theorem's limit: the `
    + `plane is determined, the axes in it are not. Put the activations back and the same two `
    + `numbers rebuild a held out digit <span class="bold">${two.gain} times</span> better than PCA `
    + `manages, and at ${best.k} numbers ${best.gain} times. And the part no other method in this `
    + `module offers at all: ${n0(M.test)} of these digits were kept out of the fit entirely, and `
    + `the encoder places them by evaluating a function, at an error of ${G.test_mse} against `
    + `${G.train_mse} on the ones it learned from.`);

  set('scrolly-intro',
    `The dataset is the same ${n0(M.n)} MNIST digits the last three articles used, mean pooled from `
    + `28 by 28 down to ${M.side} by ${M.side} so that a trained network fits in ${67} kilobytes and `
    + `can run in this page. ${n0(M.train)} of them are used for fitting and ${n0(M.test)} are held `
    + `out of everything on this page, including every curve.`);

  set('scrolly-step-0',
    `An autoencoder is two functions bolted together and trained as one: an encoder that takes `
    + `${M.pixels} pixels down to ${M.k_show} numbers and a decoder that takes those `
    + `${M.k_show} numbers back to ${M.pixels} pixels, with the loss being how far the round trip `
    + `misses. There is no other objective and no other supervision. Everything the middle layer `
    + `learns to carry, it learns because the second half needed it.`);
  set('scrolly-step-1',
    `Take out every activation function first, because that case has a known answer. With both `
    + `halves linear the round trip is just a rank-${M.k_show} matrix, and the best rank-${M.k_show} `
    + `matrix under squared error is the projection onto the top ${M.k_show} principal components. `
    + `Measured after ${n0(L.epochs)} epochs of gradient descent: `
    + `<span class="bold">${L.ae_mse}</span> against PCA's ${L.pca_mse}, a ratio of `
    + `${L.mse_ratio}, and the two planes differ by `
    + `<span class="bold">${L.subspace_gap.toExponential(0)}</span>.`);
  set('scrolly-step-2',
    `And here is the half that surprises people. The <i>plane</i> is PCA's; the `
    + `<i>axes in it</i> are not, and nothing in the loss ever asked them to be. The four numbers `
    + `are how much each of the network's two directions is each principal component: if they were `
    + `PCA's axes this would be the identity, and instead the alignments run from `
    + `${L.worst_alignment} to ${L.best_alignment}. The two directions are not even at right angles `
    + `to each other, ${L.axis_angle}° rather than 90. Squared error through a bottleneck fixes the `
    + `subspace and leaves the basis free, which is exactly the freedom the varimax rotation of the `
    + `<a href="../distances/">factor analysis article</a> exploited on purpose.`);
  set('scrolly-step-3',
    `Now put the activation functions back, so the encoder and the decoder can bend. The same two `
    + `numbers, the same held out digits, and the error goes from ${two.pca_test} to `
    + `<span class="bold">${two.deep_test}</span>, a factor of ${two.gain}. The colours are the true `
    + `digits and nothing in the training saw one: a network that is only trying to rebuild its `
    + `input has separated most of the classes on the way.`);
  set('scrolly-step-4',
    `The last panel is the one this whole module has not been able to draw. ${n0(M.test)} of these `
    + `points were never part of any fit. They are where they are because the encoder is a `
    + `<span class="bold">function</span> and it was evaluated on them, which took a matrix `
    + `multiply. Its error on them is ${G.test_mse} against ${G.train_mse} on the digits it learned `
    + `from, a ratio of ${G.overfit_ratio}. Ask t-SNE or Isomap or non-metric scaling to place a new `
    + `row and the only honest answer is to refit everything, at which point every old coordinate `
    + `has moved as well.`);

  set('theorem-note',
    `The theorem in the second panel is worth stating once properly, because it is the cleanest `
    + `link between this article and the first one in the module. With \\(f\\) and \\(g\\) linear, `
    + `the composition is a rank-\\(k\\) matrix, and Eckart and Young settled in 1936 which `
    + `rank-\\(k\\) matrix minimises squared error: the one built from the top \\(k\\) singular `
    + `vectors. So a linear autoencoder cannot beat PCA and, if it trains properly, cannot lose to `
    + `it either. Measured here: ${L.mse_ratio} times PCA's error and `
    + `${L.subspace_gap.toExponential(0)} from PCA's plane. What the theorem does not say is `
    + `anything about the basis, and the measurement fills that in: `
    + `<span class="bold">alignments of ${L.worst_alignment} to ${L.best_alignment} and an angle of `
    + `${L.axis_angle}°</span> between two directions that PCA would have made perpendicular. The `
    + `top two eigenvalues of this data are ${L.eigenvalues[0]} and ${L.eigenvalues[1]}, a ratio of `
    + `only ${(L.eigenvalues[0] / L.eigenvalues[1]).toFixed(3)} to the third at `
    + `${L.eigenvalues[2]}, so there was never much pinning them down.`);

  set('latent-intro',
    `Which makes the following possible, and it is the reason to have read this far. The network `
    + `below is the trained one, both halves, quantised to sixteen bits and running in this page. `
    + `The left panel is where the encoder put all ${n0(M.n)} digits. The right panel is what the `
    + `decoder says about whatever point you drag the crosshair to, <span class="bold">including `
    + `points where no digit ever was</span>.`);

  set('latent-note',
    `Two things are worth doing with that. Press <span class="bold">sweep the whole square</span> `
    + `and look at the grid: every tile is the decoder evaluated somewhere in the plane, most of `
    + `those somewheres are not any digit in the dataset, and the answers are still shapes that `
    + `change smoothly as you walk. That is what "a learned map" buys over "a solved layout": it `
    + `has a value everywhere. `
    + `<br /><br />`
    + `And then the honest limit. Five nearest neighbours on the ${M.k_show} numbers identifies `
    + `<span class="bold">${pct(G.code_accuracy)}</span> of the held out digits, against `
    + `${pct(G.pixel_accuracy)} on all ${M.pixels} pixels, and well below what `
    + `<a href="../neighbours/">t-SNE's two coordinates</a> managed on the same digits. `
    + `A bottleneck this narrow is not a good classifier of anything, and it was never asked `
    + `to be: its two numbers were chosen to rebuild an image, and rebuilding an image spends most `
    + `of its budget on ink, position and stroke width, which is why the map is organised by shape `
    + `rather than by identity. Quantising the weights to sixteen bits moved the two numbers by at `
    + `most ${data.net.quant.code_shift} and the held out error from ${data.net.quant.mse_full} to `
    + `${data.net.quant.mse_quantised}.`);

  set('curve-intro',
    `The obvious question is what the bending is worth, and it deserves a number rather than an `
    + `adjective. Three models at each bottleneck size on the same held out digits: PCA, an `
    + `autoencoder with every activation removed, and the same network with them put back.`);

  set('curve-note',
    `The gap is real and it is smaller than the folklore. At ${M.k_show} numbers the non-linear `
    + `network is <span class="bold">${two.gain} times</span> better than PCA on held out digits, `
    + `at ${best.k} it is ${best.gain} times, and at ${biggest.k} it is ${biggest.gain}. Meanwhile `
    + `the dashed line is on top of PCA everywhere, which is the theorem again. `
    + `<span class="bold">A factor of ${best.gain} is not why anybody uses these.</span> The reasons `
    + `are the two things the previous section showed and this chart cannot: that the encoder is a `
    + `function you can apply to a new row, and that the decoder is an inverse. Everything else in `
    + `this module gives you neither, and PCA gives you the first at a cost of ${two.gain} times the `
    + `error.`);

  set('wide-note',
    `One more thing, because "the bottleneck is what makes it learn" is the sentence everybody `
    + `takes away and it is only half true. Make the middle layer <span class="bold">${W.k}</span> `
    + `wide, which is wider than the ${W.p} pixels going in, and there is no bottleneck at all: the `
    + `network can and does learn to copy. Its reconstruction of a clean held out digit is `
    + `${W.plain_clean}, which is excellent and worth nothing. Now change one thing. Train the same `
    + `architecture on inputs with noise of ${W.noise} added and ask it for the clean digit. Not the `
    + `size of the layer, not the depth, not the training length: just what it is asked to do.`);

  const tbody = document.querySelector('#wide-table');
  if (tbody) {
    tbody.innerHTML = `
      <tr><td>trained to copy its input</td><td>${W.plain_clean}</td>
        <td><span class="bold">${W.plain_noisy}</span></td><td>${pct(W.plain_accuracy)}</td></tr>
      <tr><td>trained to remove noise of ${W.noise}</td><td>${W.denoise_clean}</td>
        <td><span class="bold">${W.denoise_noisy}</span></td><td>${pct(W.denoise_accuracy)}</td></tr>
      <tr><td>doing nothing at all</td><td>0</td><td>${W.identity_noisy}</td><td>-</td></tr>`;
  }

  set('roll-intro',
    `And a last measurement, which is the module closing its own loop. The `
    + `<a href="../manifolds/">manifold article</a> built a Swiss roll whose true two-dimensional `
    + `chart is known in closed form, and Isomap recovered it to ${R.isomap} where the best flat `
    + `projection managed ${R.pca}. The generator asserts that the ${R.n} points below are that `
    + `roll, coordinate for coordinate. So: give a two unit bottleneck the same points and the same `
    + `question.`);

  const bestTry = R.best;
  set('roll-note',
    `${R.disparity > 0.5
      ? `It does not unroll it. The best of ${R.attempts.length} attempts, at ${bestTry.hidden} `
        + `hidden units and ${n0(bestTry.epochs)} epochs, lands `
        + `<span class="bold">${R.disparity}</span> from the true chart against Isomap's `
        + `${R.isomap}, on a scale where 1 means carrying nothing. And it rebuilds the points `
        + `perfectly well while doing it, at ${R.mse} of squared error, which is the trap: `
        + `<span class="bold">a low reconstruction error is not evidence that the right two numbers `
        + `were found.</span> Nothing in the loss mentions distance along the sheet, so nothing in `
        + `the answer respects it, and the self-organising map in the previous article failed the `
        + `same test for the same reason while looking nothing like this. Two methods, one lesson: `
        + `a method optimises what it was given, and its own diagnostics measure whether it managed, `
        + `not whether what it was given was your question.`
      : `It recovers it, at ${R.disparity} against Isomap's ${R.isomap}.`}`);

  /* -------------------------------------------------------------- the verdict */
  set('verdict-encode',
    `Both are functions. The encoder placed ${n0(M.test)} unseen digits at an error of `
    + `${G.test_mse} against ${G.train_mse} on the fitted ones, a ratio of ${G.overfit_ratio}.`);
  set('verdict-encode-warn',
    `t-SNE, UMAP, Isomap, locally linear embedding and non-metric scaling have no encoder at all: `
    + `a new row means refitting, and every old coordinate moves.`);
  set('verdict-decode',
    `The decoder answers anywhere in the plane, including where no digit was, which is what the `
    + `sweep in the widget above is. PCA can also invert, but at ${two.gain} times the error at `
    + `k = ${M.k_show}.`);
  set('verdict-decode-warn',
    `Nothing constrains the code to be interpretable or the space between points to mean anything; `
    + `that is what module 3 adds, by putting a prior on the code.`);
  set('verdict-error',
    `${two.gain} times PCA at k = ${M.k_show} and ${best.gain} times at k = ${best.k}, on held out `
    + `digits.`);
  set('verdict-error-warn',
    `With no activations it is PCA exactly (${L.mse_ratio} times the error, `
    + `${L.subspace_gap.toExponential(0)} from the same plane), so the depth is doing the work and `
    + `an undertrained network buys nothing.`);
  set('verdict-roll',
    `On the same ${R.n} points, Isomap reaches ${R.isomap} from the true chart and the best of `
    + `${R.attempts.length} bottlenecks reaches ${R.disparity}.`);
  set('verdict-roll-warn',
    `And the bottleneck rebuilds those points to ${R.mse} while doing it, so the error it reports `
    + `is no warning at all.`);
  set('verdict-wide',
    `A layer of ${W.k} on ${W.p} pixels learns to copy: ${W.plain_clean} on a clean digit and `
    + `${W.plain_noisy} when asked to clean a noisy one. Trained to denoise instead, the same `
    + `architecture gets ${W.denoise_noisy}, against ${W.identity_noisy} for doing nothing.`);
  set('verdict-wide-warn',
    `Both codes classify about equally (${pct(W.plain_accuracy)} and `
    + `${pct(W.denoise_accuracy)}), because the copying one's code is the pixels; the difference `
    + `shows up only when something is asked of it that copying cannot do.`);

  set('closing-note',
    `This closes module 2.2, and the six methods in it divide more cleanly than they first appear. `
    + `Four of them (classical and non-metric scaling, Isomap, locally linear embedding) solve for `
    + `coordinates and hand you the coordinates. Two of them (t-SNE, UMAP) do the same and are `
    + `honest about caring only for neighbourhoods. Two of them (PCA, the autoencoder) hand you `
    + `<span class="bold">functions</span>, and that is a different kind of object: it can be `
    + `applied, inverted, saved, and used on data that did not exist when it was fitted. The price `
    + `of the last of those is that it has to be trained, that a low reconstruction error `
    + `(${R.mse} on the Swiss roll) is no evidence the right structure was found, and that nothing `
    + `constrains the code to mean anything.`
    + `<br /><br />`
    + `That last sentence is where this stops being a dimensionality reduction problem. If the `
    + `decoder is a function from ${M.k_show} numbers to an image, and it answers sensibly at points `
    + `where no digit ever was, then it is most of a way to <i>generate</i> digits: sample a point, `
    + `decode it, and look at what comes out. What is missing is any reason to believe a sampled `
    + `point will land somewhere the decoder handles well, which is a question about the `
    + `distribution of the code and not about reconstruction at all. Putting a distribution there `
    + `on purpose is what a variational autoencoder does, and that is module 3, where the objective `
    + `stops being "rebuild this" and starts being "produce something that could have been this". `
    + `Module 3 opens somewhere else, though: <a href="../gans/">Counterfeit</a> takes the harder `
    + `road of not writing the objective down at all and training a second network to be it, and `
    + `nearly everything awkward about generative models is visible there in miniature.`);

  set('resources-note',
    `${n0(M.n)} stratified MNIST digits, seed ${M.seed}, mean pooled to ${M.side} by ${M.side} and `
    + `split ${n0(M.train)} for fitting and ${n0(M.test)} held out. The network is `
    + `${M.pixels} to ${M.hidden} to ${M.k_show} and back, tanh in the hidden layers and a logistic `
    + `output, trained with Adam on squared error; the linear arm is the same thing with every `
    + `activation removed. PCA reference from scikit-learn ${M.sklearn}. The weights the page runs `
    + `are ${n0(data.net.count)} numbers in float16, and every figure quoted was measured on those `
    + `quantised weights rather than on the ones training produced: the difference is `
    + `${data.net.quant.code_shift} in the two codes and nothing at four decimal places in the `
    + `held out error. The Swiss roll is asserted by the generator to be identical to the one the `
    + `<a href="../manifolds/">manifold article</a> ships.`);
}
