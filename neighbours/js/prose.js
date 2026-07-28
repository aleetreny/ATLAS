/* Every sentence on this page that carries a figure, written from the file.
 *
 * Two of these paragraphs report that a well known demonstration did not
 * reproduce here. They are written as comparisons like everything else, so if a
 * rerun changes the answer the sentence changes with it rather than becoming a
 * confident claim about a number that moved.
 */
const n0 = (v) => v.toLocaleString('en-US');
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const C = data.crowding;
  const R = data.digits.runs;
  const perps = M.perplexities;
  const shown = String(M.perplexity_show);
  const N = data.noise;
  const B = data.blobs;
  const S = data.seeds;
  const U = data.umap;
  const board = data.board;
  const nudge = data.nudge;
  const bigSimplex = C.simplex[C.simplex.length - 1];
  const lowD = C.volume[0];
  const highD = C.volume[C.volume.length - 1];
  const counts = perps.map((p) => R[String(p)].clusters);
  const accs = perps.map((p) => R[String(p)].accuracy);
  const maxCount = Math.max(...counts);
  const minCount = Math.min(...counts);
  const pca = board.find((r) => r.method === 'pca');
  const tsne = board.find((r) => r.method === 't-sne');
  const umap = board.find((r) => r.method === 'umap');
  const uShown = U.runs[String(U.shown)];

  /* ------------------------------------------------------------- the opening */
  set('opening-note',
    `Three numbers set up everything below. ${bigSimplex.k} points that are all exactly the same `
    + `distance apart need ${bigSimplex.dims_needed} dimensions to exist; put them on a plane and `
    + `the best you can do leaves their distances spread by `
    + `<span class="bold">${pct(bigSimplex.spread, 2)}</span>, with the longest `
    + `${bigSimplex.ratio} times the shortest. Give up on distances and keep only neighbours, and `
    + `two coordinates classify a handwritten digit `
    + `<span class="bold">${(tsne.accuracy / pca.accuracy).toFixed(1)} times</span> better than `
    + `PCA's two (${pct(tsne.accuracy)} against ${pct(pca.accuracy)}). And the price, which is the `
    + `part nobody puts on the figure: on ${n0(M.n_live)} digits with exactly ten classes in them, `
    + `the number of separate clumps a counter finds in the map goes from `
    + `<span class="bold">${counts[0]}</span> at perplexity ${perps[0]} to `
    + `<span class="bold">${counts[counts.length - 1]}</span> at perplexity `
    + `${perps[perps.length - 1]}, and is never ten.`);

  /* ------------------------------------------------------------- the crowding */
  set('crowding-intro',
    `Before any algorithm, a fact about planes that decides the rest of the article. Three points `
    + `can be mutually equidistant on a page: an equilateral triangle. Four cannot, and no amount `
    + `of cleverness will make them, because four mutually equidistant points are a tetrahedron and `
    + `a tetrahedron needs three dimensions. In general k of them need k-1. So if your data has `
    + `${bigSimplex.k} groups that are all about equally different from each other, which is an `
    + `ordinary thing for data to have, then <span class="bold">no two-dimensional picture of it `
    + `exists</span>, and every method that tries to make one is choosing which distances to get `
    + `wrong. That is not a criticism of any of them. It is the situation.`);

  set('crowding-note',
    `The two panels are the same fact from opposite ends. From one end, a plane runs out of room `
    + `almost immediately: ${bigSimplex.k} equidistant points come out `
    + `${pct(bigSimplex.spread, 2)} apart in spread and ${bigSimplex.ratio} times apart in ratio. `
    + `From the other, high-dimensional data hardly has any distances left to preserve: points on a `
    + `sphere in ${lowD.d} dimensions have pairwise distances varying by ${pct(lowD.cv)} and in `
    + `${highD.d} dimensions by <span class="bold">${pct(highD.cv)}</span>. Everything is nearly `
    + `the same distance from everything else, so an objective that spends its budget on getting `
    + `those distances right is spending it on a quantity that has almost stopped varying. `
    + `<span class="bold">The way out is to stop asking for distances.</span>`);

  /* ---------------------------------------------------------------- the scrolly */
  const shownRef = data.digits.perplexity[shown];
  set('scrolly-step-0',
    `${n0(M.n_live)} handwritten digits, reduced to ${M.pcs} principal components first, which `
    + `keeps ${pct(M.variance_kept)} of the variance and is what every implementation does before `
    + `it starts. One point is marked. t-SNE never asks where it should go; it asks what the `
    + `probability is that this point would pick each other point as its neighbour.`);
  set('scrolly-step-1',
    `That probability is a gaussian centred on the point, normalised over the other `
    + `${n0(M.n_live - 1)} so it sums to one. Sorted, most of it lives in a handful of points and `
    + `the rest is nearly zero, which is the whole idea: a neighbour distribution is a statement `
    + `about a neighbourhood, not about the far side of the dataset.`);
  set('scrolly-step-2',
    `The width of that gaussian is not a setting. It is solved for, separately for every point, so `
    + `that the distribution it produces has a fixed <span class="bold">perplexity</span>: two to `
    + `the power of its entropy, which is to say the effective number of neighbours. At perplexity `
    + `${shown} the bandwidths this dataset needs run from ${shownRef.sigma_min} to `
    + `${shownRef.sigma_max}, a factor of `
    + `${(shownRef.sigma_max / shownRef.sigma_min).toFixed(1)}, and every one of them hits its `
    + `target to <span class="bold">${shownRef.worst_perplexity_error}</span>. That per-point `
    + `solve is what lets one number mean the same thing in a crowded region and an empty one, and `
    + `it is also the moment the map loses any notion of scale.`);
  set('scrolly-step-3',
    `The map gets a different distribution: a Student t with one degree of freedom instead of a `
    + `gaussian. This is the fix for the crowding problem above. A gaussian falls off so fast that `
    + `moderately distant points contribute nothing, so the optimiser has no force pushing them `
    + `apart and the map collapses inward. The t has a heavy tail, so a pair that ends up far apart `
    + `in the map still has a say, and the room they need gets made.`);
  set('scrolly-step-4',
    `Then move the map until the two distributions agree, by gradient descent on the divergence `
    + `between them. For the first ${M.exagg_iters} of ${n0(M.iters)} steps every neighbour `
    + `probability is multiplied by ${M.exagg}, which is a lie that gets told on purpose: it makes `
    + `neighbours grip much harder than the data says, groups form early and tightly, and the space `
    + `between them opens up. That space is not evidence of anything. It was manufactured, by a `
    + `constant, in the first quarter of the run.`);
  set('scrolly-step-5',
    `And that is all of it. Colour is the true digit, which nothing in the calculation ever saw. `
    + `A five nearest neighbour vote using only these two coordinates gets `
    + `<span class="bold">${pct(R[shown].accuracy)}</span> of them right, against `
    + `${pct(pca.accuracy)} for two principal components on the full set. Everything after this `
    + `section is about what that picture does and does not entitle you to say.`);

  set('asymmetry-note',
    `One thing worth reading off the last line rather than being told. The divergence is `
    + `\\(\\sum p \\log(p/q)\\), and it is weighted by \\(p\\), the data's probabilities. A pair `
    + `that is close in the data and far in the map has a large \\(p\\) and a small \\(q\\), so it `
    + `costs a great deal. A pair that is far in the data and close in the map has a small \\(p\\), `
    + `so it costs almost nothing. <span class="bold">The objective punishes tearing neighbours `
    + `apart and barely notices strangers being put together.</span> Everything the rest of this `
    + `page measures follows from that asymmetry: local structure is what the method protects, and `
    + `it protects it by spending everything else.`);

  /* ---------------------------------------------------------------- the live run */
  set('live-intro',
    `The whole algorithm runs below, in this page, on ${n0(M.n_live)} digits: the bandwidth search, `
    + `the joint distribution and all ${n0(M.iters)} gradient steps. Three datasets share the same `
    + `controls, and the reason they do is that the algorithm cannot tell them apart. One is the `
    + `digits. One is three gaussian blobs whose radii were built to be 1, 2 and 5 and whose two `
    + `gaps were built to differ by a factor of two. One is ${n0(M.n_live)} points of pure noise `
    + `with nothing in it at all.`);

  const bestAcc = Math.max(...accs);
  const worstAcc = Math.min(...accs);
  set('count-note',
    `Start with the count, because it is the thing everyone reads off a t-SNE map first and the `
    + `thing it is least able to tell you. These ${n0(M.n_live)} digits contain exactly ten classes `
    + `at every setting of that slider. The number of clumps the counter finds goes `
    + `<span class="bold">${counts.join(', ')}</span> as the perplexity goes ${perps.join(', ')}: `
    + `it moves by a factor of ${(maxCount / Math.max(minCount, 1)).toFixed(0)}, and it is never `
    + `ten. `
    + `<br /><br />`
    + `Meanwhile the thing the map is actually good at does not move at all. Over those same seven `
    + `settings, a five nearest neighbour vote on the two map coordinates scores between `
    + `${pct(worstAcc)} and ${pct(bestAcc)}, a range of `
    + `${((bestAcc - worstAcc) * 100).toFixed(0)} points. The neighbourhoods are stable and the `
    + `grouping is not, which is exactly what the asymmetric objective above predicts, and it is `
    + `the single most useful sentence to carry away: <span class="bold">a t-SNE map is a claim `
    + `about who is near whom, and not a claim about how many groups there are.</span> The counter `
    + `here is not tuned to produce that: its one parameter is pinned from both sides in the `
    + `generator, so that it calls a plain two-dimensional gaussian one clump and a t-SNE of three `
    + `separated blobs three.`);

  const blobShown = B.rows.find((r) => r.perplexity === M.perplexity_show);
  const radAll = B.rows.map((r) => r.radius_ratio);
  const sepAll = B.rows.map((r) => r.sep_ratio);
  set('size-note',
    `Then the sizes, and here the answer is not "unreliable", it is "no information". The three `
    + `blobs were built with radii ${B.radii.join(', ')}, so the widest is `
    + `<span class="bold">${B.true_radius_ratio}</span> times the narrowest in the data, and with `
    + `one gap ${B.true_sep_ratio} times the other. Across all ${perps.length} perplexities the map `
    + `reports radius ratios between ${Math.min(...radAll)} and ${Math.max(...radAll)} and gap `
    + `ratios between ${Math.min(...sepAll)} and ${Math.max(...sepAll)}. `
    + `<span class="bold">Every one of them is about 1.</span> A t-SNE clump is as big as the number `
    + `of points in it and no bigger, because the per-point bandwidth normalised the density away `
    + `before the optimiser ever saw it, and the gap between two clumps is whatever the exaggeration `
    + `and the heavy tail happened to leave there. UMAP does the same: at ${U.shown} neighbours it `
    + `reports ${uShown.radius_ratio} and ${uShown.sep_ratio}.`);

  const noiseShown = N.rows.find((r) => r.perplexity === M.perplexity_show);
  const noiseCounts = N.rows.map((r) => r.clusters);
  const noiseMax = Math.max(...noiseCounts);
  set('noise-note',
    `And then the famous one, which this page tried to reproduce and could not. The well known `
    + `demonstration is that t-SNE at a low perplexity turns pure noise into a picture full of `
    + `clusters. The noise button above is exactly that: ${n0(M.n_live)} points drawn independently `
    + `from a gaussian in ${M.pcs} dimensions, with no structure of any kind in them. The map does `
    + `look like something. But the counter, pinned on the two cases whose answers are known, finds `
    + `<span class="bold">${noiseCounts.join(', ')}</span> clumps across the seven perplexities, `
    + `against ${counts.join(', ')} on the real digits. `
    + `${noiseMax <= 2
      ? `On this dataset, at this size, it did not manufacture clusters that a counter can see.`
      : `On this dataset it did manufacture up to ${noiseMax} of them.`}`);

  const tbody = document.querySelector('#folklore-table');
  if (tbody) {
    tbody.innerHTML = N.folklore.map((f) => `<tr>`
      + `<td>${n0(f.n)} points in ${f.d} dimensions</td>`
      + `<td>${f.perplexity}</td>`
      + `<td><span class="bold">${f.clusters}</span></td>`
      + `<td>${pct(f.noise_share)}</td></tr>`).join('');
  }

  const anyFolklore = N.folklore.some((f) => f.clusters > 1);
  set('folklore-note',
    `Before reporting a failure to reproduce, it is worth trying properly, so the table above is `
    + `${N.folklore.length} more attempts at the settings the demonstration is usually run at: `
    + `fewer points, more dimensions, the lowest perplexity that still solves. `
    + `${anyFolklore
      ? `Some of them do produce extra clumps, so the effect is real and this article's main run was `
        + `simply too large to show it.`
      : `Every one of them comes back with one clump. So the honest statement is narrower than the `
        + `folklore and, I think, more useful: what the noise map has is <span class="bold">texture`
        + `</span>, and a reader's eye groups texture into clusters very readily, which is a fact `
        + `about readers and not about the algorithm. The caution survives; the mechanism it is `
        + `usually attributed to did not show up here. Note also what the clustering tendency `
        + `statistic does with this, because it is a good warning about that statistic: a plain `
        + `two-dimensional gaussian, which has no clusters in it whatsoever, scores `
        + `${N.hopkins_flat_2d} on it, against ${N.hopkins_uniform_2d} for a uniform square. It `
        + `measures departure from uniformity, and that is not the same question.`}`);

  set('seed-note',
    `There is a second source of instability, and it is larger than the perplexity. Run the same `
    + `data, the same perplexity and the same schedule from ${S.n} different starting `
    + `configurations and compare the ${(S.n * (S.n - 1)) / 2} pairs of maps: their shapes differ `
    + `by <span class="bold">${S.tsne_disparity}</span> on average and up to `
    + `${S.tsne_disparity_worst}, on a scale where 1 means carrying nothing about each other. And `
    + `yet <span class="bold">${pct(S.tsne_knn_overlap)}</span> of each point's ten nearest `
    + `neighbours are the same across runs. Two very different pictures of the same stable `
    + `neighbourhood structure, which is the same message a third time. UMAP is steadier on both `
    + `counts: ${S.umap_disparity} of shape difference and ${pct(S.umap_knn_overlap)} of `
    + `neighbour overlap.`
    + `<br /><br />`
    + `Worse, or more interesting: it does not take a different seed. Nudge the input by one part `
    + `in a million, ${nudge.input_change} on coordinates of order one, and the bandwidths move by `
    + `${nudge.beta_change} and the neighbour probabilities by ${nudge.p_change}, which is nothing. `
    + `The finished map moves by <span class="bold">${nudge.disparity}</span> in shape and its `
    + `divergence goes from ${nudge.kl_ref} to ${nudge.kl_nudged}, while `
    + `${pct(nudge.knn_overlap)} of each point's neighbours survive. That is why this page's own `
    + `run and the generator's do not land on identical divergences even though they run the same `
    + `code from the same starting point: two implementations cannot agree on a bandwidth to `
    + `better than about a part in ten million, one of them sums left to right and the other does `
    + `not, and a thousand steps of this descent is enough to turn that into a visibly different `
    + `picture with the same neighbours in it.`);

  /* ---------------------------------------------------------------------- UMAP */
  set('umap-intro',
    `UMAP arrives at something very similar from a different direction: a fuzzy neighbourhood graph `
    + `built with a local connectivity guarantee, and a cross entropy between edge probabilities `
    + `optimised by stochastic gradient descent with negative sampling. The differences that matter `
    + `in practice are that it is much faster, that its repulsive term is sampled rather than `
    + `summed, and that it is <span class="bold">steadier</span>: ${S.umap_disparity} of shape `
    + `difference between seeds against t-SNE's ${S.tsne_disparity}. What it is not is a fix for `
    + `any of the three problems above. At ${U.shown} neighbours it reports the blobs' radius ratio `
    + `as ${uShown.radius_ratio} where the truth is ${B.true_radius_ratio}, and its clump count on `
    + `the digits goes ${Object.values(U.runs).map((r) => r.clusters).join(', ')} as the neighbour `
    + `count goes ${Object.keys(U.runs).join(', ')}. Same three lessons, different package.`);

  set('board-note',
    `Which leaves a clean division of labour, and it is the last thing this module has to say about `
    + `two-dimensional pictures. If the question is "are these two things similar", a neighbour `
    + `embedding answers it well: ${pct(tsne.accuracy)} and ${pct(umap.accuracy)} against `
    + `${pct(pca.accuracy)} from a projection, with trustworthiness ${tsne.trustworthiness} and `
    + `${umap.trustworthiness} against ${pca.trustworthiness}. If the question is "how far apart `
    + `are these two things", it is the projection that answers, and by a wide margin: `
    + `${pca.global_corr} agreement with the original distances against ${tsne.global_corr} and `
    + `${umap.global_corr}. There is no method on this page that does both, and the reason is the `
    + `first section: <span class="bold">doing both is not available.</span>`);

  /* ------------------------------------------------------------- verdict table */
  set('verdict-local',
    `Two coordinates classify at ${pct(tsne.accuracy)} where two principal components manage `
    + `${pct(pca.accuracy)}, and ${pct(S.tsne_knn_overlap)} of each point's neighbours survive a `
    + `change of seed that moves the whole picture by ${S.tsne_disparity}.`);
  set('verdict-local-warn',
    `The objective is weighted by the data's probabilities, so it punishes separating neighbours `
    + `and barely notices strangers being brought together. Trustworthiness ${tsne.trustworthiness} `
    + `measures the first, and nothing on the map measures the second.`);
  set('verdict-count',
    `On ${n0(M.n_live)} digits with ten classes, the count goes ${counts.join(', ')} across `
    + `perplexities ${perps.join(', ')}, and UMAP's goes `
    + `${Object.values(U.runs).map((r) => r.clusters).join(', ')}. Neither is ever ten.`);
  set('verdict-count-warn',
    `The gaps that make clumps look separate are partly manufactured by the early exaggeration, `
    + `which multiplies every neighbour probability by ${M.exagg} for the first ${M.exagg_iters} of `
    + `${n0(M.iters)} steps.`);
  set('verdict-size',
    `Blobs built at radii ${B.radii.join(', ')}, a true ratio of ${B.true_radius_ratio}, come back `
    + `at ${blobShown.radius_ratio}. Gaps built at a ratio of ${B.true_sep_ratio} come back at `
    + `${blobShown.sep_ratio}.`);
  set('verdict-size-warn',
    `This is not noise that averaging would fix. The per-point bandwidth divides the density out `
    + `before the optimiser starts, so the information is gone rather than obscured.`);
  set('verdict-which',
    `UMAP is steadier (${S.umap_disparity} of shape difference between seeds against `
    + `${S.tsne_disparity}, ${pct(S.umap_knn_overlap)} neighbour overlap against `
    + `${pct(S.tsne_knn_overlap)}) and scores ${pct(umap.accuracy)} against ${pct(tsne.accuracy)}.`);
  set('verdict-which-warn',
    `Its repulsion is sampled rather than summed, so it has no exactly evaluable objective to `
    + `report, and its neighbour count moves the clump count as much as perplexity does `
    + `(${Object.values(U.runs).map((r) => r.clusters).join(', ')}).`);
  set('verdict-report',
    `A nudge of ${nudge.input_change} in the input moves the finished map by ${nudge.disparity} `
    + `while leaving ${pct(nudge.knn_overlap)} of the neighbourhoods intact, so one picture is not `
    + `a result.`);
  set('verdict-report-warn',
    `Perplexity, learning rate, initialisation, exaggeration and iteration count all change the `
    + `picture, and only the first is usually reported.`);

  /* --------------------------------------------------------------- the closing */
  set('closing-note',
    `The trade at the centre of this article is not a defect anybody could fix. Section one is a `
    + `theorem: ${bigSimplex.k} equidistant points do not fit on a plane, and in ${highD.d} `
    + `dimensions the distances that would have to be preserved have stopped varying anyway `
    + `(${pct(highD.cv)} of spread). Given that, a method has to choose what to keep, and these two `
    + `chose neighbourhoods and spent everything else: ${pct(tsne.accuracy)} classification against `
    + `PCA's ${pct(pca.accuracy)}, bought with ${tsne.global_corr} of distance agreement against `
    + `PCA's ${pca.global_corr}. That is a good trade for the question they were built for and a `
    + `terrible one for every other question, and the failures in between are not subtle: cluster `
    + `sizes at ${blobShown.radius_ratio} where the truth is ${B.true_radius_ratio}, a clump count `
    + `that never once equalled the ten classes present, and a picture that moves `
    + `${nudge.disparity} when the input moves ${nudge.input_change}.`
    + `<br /><br />`
    + `All four articles of this module have now produced a small number of coordinates by `
    + `<span class="bold">solving</span> for them: an eigenproblem, a stress, a graph, a `
    + `divergence. None of them left anything behind. Hand any of them a new row and there is no `
    + `map to put it on, only the whole fit to run again, and no way to go back from two `
    + `coordinates to a digit. The <a href="../autoencoders/">last article of the module</a> asks `
    + `for the one thing none of these can give: a reduction that is a pair of functions, learned, `
    + `that can be applied to a row it has never seen and inverted to get the row back.`);

  set('resources-note',
    `The digits are ${n0(M.n_big)} stratified MNIST images, the same draw the three previous `
    + `articles used, reduced to ${M.pcs} principal components (${pct(M.variance_kept)} of the `
    + `variance) before anything else; the page's own runs use ${n0(M.n_live)} of them. The `
    + `schedule is the published one: ${n0(M.iters)} iterations, exaggeration ${M.exagg} for the `
    + `first ${M.exagg_iters}, learning rate ${M.lr}, momentum 0.5 then 0.8. References from `
    + `scikit-learn ${M.sklearn} and umap-learn ${M.umap}; scikit-learn's own t-SNE at perplexity `
    + `${shown}, which uses a different initialisation and learning rate, reaches trustworthiness `
    + `${data.digits.sklearn_trustworthiness} against this file's ${R[shown].trustworthiness}. The `
    + `clump counter is DBSCAN with a radius of ${N.counter_multiple} times the median distance to `
    + `the tenth nearest neighbour, pinned so that it calls a two-dimensional gaussian one clump `
    + `and a t-SNE of three separated blobs three.`);
}
