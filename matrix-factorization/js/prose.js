/* Every number on this page, composed from the file it was measured into. */
const IDS = ['intro-1', 'intro-2', 'intro-3', 'hole-intro', 'hole-note', 'fit-intro', 'fit-math',
  'sweep-note', 'search-note', 'rot-intro', 'rot-verdict', 'imp-intro', 'imp-note',
  'verdict-intro', 'v-svd', 'v-svd-warn', 'v-als', 'v-als-warn', 'v-imp', 'v-imp-warn',
  'v-rot', 'v-rot-warn', 'closing-1', 'closing-2', 'closing-3', 'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initProse(data) {
  const M = data.meta;
  const D = data.digits;
  const E = data.ratings.rmse;
  const P = data.previous;
  const S = data.sweeps;
  const R = data.rotation;
  const I = data.implicit;

  const bestRank = S.rank.reduce((a, x) => (x.test < a.test ? x : a));
  const bestLam = S.lam.reduce((a, x) => (x.test < a.test ? x : a));
  const worstDigit = D.complete.reduce(
    (a, x) => Math.max(a, Math.abs(x.svd - x.published_pca)), 0);
  const bestHoled = D.holed.reduce((a, x) => (x.observed < a.observed ? x : a));
  const alsWins = E.als < E.svd_resid;
  const beatsKnn = E.als < P.item_knn_rmse;
  const impBeatsKnn = I.rank.hits['10'] > P.item_knn_hits10;

  set('intro-1',
    `<a href="../collaborative-filtering/">The previous article</a> never built a model. It kept the `
    + `whole table and, when asked about a cell, went and looked at the neighbouring columns. That `
    + `works, it is hard to beat with a small amount of effort, and it has one property that ends up `
    + `deciding everything: there is nothing to <i>store</i> except the table itself, and nothing to `
    + `<i>say</i> about a book except which other books its readers also read.`);

  set('intro-2',
    `The alternative is to insist that the table has a shape. Give every reader `
    + `${M.k} numbers and every book ${M.k} numbers, and let a rating be their dot product, `
    + `$$\\hat{r}_{ui} = \\mu + b_u + b_i + p_u^{\\top} q_i$$ `
    + `Then ${n0(M.users)} readers and ${n0(M.books)} books stop being `
    + `${n0(M.users * M.books)} unknown cells and become `
    + `${n0((M.users + M.books) * M.k)} free numbers, fitted to the `
    + `${n0(M.train_ratings)} cells that exist.`);

  set('intro-3',
    `Which is a factorisation, and this atlas has already spent an article on those. `
    + `<a href="../directions/">Directions</a> ran PCA, ICA and NMF on the same matrix and showed `
    + `they are one factorisation under three different constraints: orthogonal, independent, `
    + `non negative. This article adds a fourth constraint, and it is not of the same kind. The `
    + `others restrict what the factors may look like. This one restricts <span class="bold">which `
    + `entries the error is summed over</span>, and that single change breaks the closed form, `
    + `breaks the nesting, and turns a decomposition into a fitting problem.`);

  set('hole-intro',
    `That is a strong claim, so it is worth measuring on a matrix where both versions can be run. `
    + `These are the same ${n0(D.n)} handwritten digits that article factorised, `
    + `${D.pixels} pixels each, at the same five ranks. First with every pixel visible, which is `
    + `the case it studied, and then with `
    + `<span class="bold">${pc(D.hide, 0)}</span> of the pixels hidden at random and the error `
    + `measured on the hidden ones, which is the case a ratings table is always in.`);

  set('hole-note',
    `Two things are worth pulling out of that. The first is the identity: with nothing missing, a `
    + `truncated SVD of the centred matrix and the principal components of that earlier article `
    + `agree to <span class="bold">${worstDigit.toExponential(1)}</span>, which is the rounding of `
    + `its published file. Same objective, same answer, closed form, one line of linear algebra. `
    + `The second is what the hole costs: at rank ${bestHoled.rank}, filling the gaps with zero `
    + `scores ${bestHoled.zero.toFixed(4)}, filling them with the column average `
    + `${bestHoled.mean.toFixed(4)}, and summing the error only over what is actually there `
    + `<span class="bold">${bestHoled.observed.toFixed(4)}</span>. The first two are not worse `
    + `implementations of the third. They are answers to a different question, in which every `
    + `missing entry is a confident statement that the value is zero, or that it is average.`);

  set('fit-intro',
    `On the real table there is no complete version to fall back on, so the objective is written `
    + `over the observed cells and nothing else, with a penalty that keeps a reader with `
    + `${n0(M.k)} numbers and fourteen ratings from fitting all fourteen exactly.`);

  set('fit-math',
    `$$\\min_{p,q} \\sum_{(u,i) \\in \\text{observed}} \\left(r_{ui} - \\mu - b_u - b_i - `
    + `p_u^{\\top} q_i\\right)^2 + \\lambda \\left(\\|p_u\\|^2 + \\|q_i\\|^2\\right)$$ `
    + `That sum has no closed form solution, but it has a useful property: fix all the \\(q\\) and `
    + `it is a separate ridge regression per reader, fix all the \\(p\\) and it is a separate ridge `
    + `regression per book. Alternating between the two is the whole algorithm, and each half is `
    + `${n0(M.users)} independent ${M.k} by ${M.k} systems that a laptop solves at once.`);

  set('sweep-note',
    `The best rank measured is <span class="bold">${bestRank.k}</span> at `
    + `${bestRank.test.toFixed(4)}, and the best regularisation `
    + `<span class="bold">${bestLam.lam}</span> at ${bestLam.test.toFixed(4)}. Both sweeps run `
    + `${M.sweep_epochs} alternating passes rather than the ${M.epochs} of the headline fit, which `
    + `is why the same setting reads ${bestLam.test.toFixed(4)} there and `
    + `${E.als.toFixed(4)} in the table. For scale, the `
    + `neighbourhood method of the previous article scores `
    + `<span class="bold">${P.item_knn_rmse.toFixed(4)}</span> on exactly these cells and the bias `
    + `model alone ${P.bias_rmse.toFixed(4)}. `
    + `${beatsKnn
      ? `So the factorisation wins, by ${(P.item_knn_rmse - E.als).toFixed(4)}, and it does it `
        + `while storing ${n0((M.users + M.books) * M.k)} numbers instead of a `
        + `${n0(M.books)} by ${n0(M.books)} table of similarities.`
      : `So on this dataset the factorisation does not beat the neighbours at guessing the number, `
        + `and the interesting comparison is the one further down, which is about lists.`}`);

  set('search-note',
    `Two searches for the same objective are worth running, because the objective is what was `
    + `written down and the search is what actually happens. Alternating least squares solves each `
    + `half exactly; stochastic gradient descent walks towards it one batch at a time. They land at `
    + `${E.als.toFixed(4)} and ${E.sgd.toFixed(4)}, with their predictions correlated at `
    + `${data.ratings.agree.toFixed(4)}, and the gradient version takes `
    + `${data.ratings.seconds_ratio >= 1
      ? `${data.ratings.seconds_ratio.toFixed(1)} times as long as the alternating one`
      : `${(1 / data.ratings.seconds_ratio).toFixed(1)} times less than the alternating one`} `
    + `in this run, on the same machine at the same moment, which is the only way a wall clock `
    + `means anything. The reason to keep the slower one in the toolbox is that it never forms a `
    + `${M.k} by ${M.k} system at all, so it survives ranks and datasets where the exact half step `
    + `does not fit in memory.`);

  set('rot-intro',
    `Now the part that stops a lot of dashboards from being written. A fitted factorisation is `
    + `never unique: for any invertible \\(A\\), replacing \\(p_u\\) by \\(A^{\\top} p_u\\) and `
    + `\\(q_i\\) by \\(A^{-1} q_i\\) leaves every single prediction untouched, because `
    + `\\((A^{\\top}p)^{\\top}(A^{-1}q) = p^{\\top} q\\). Measured on `
    + `${n0(400 * 400)} predictions with a random invertible \\(A\\) of condition number `
    + `${R.cond.toFixed(1)}, the largest change in any of them is `
    + `${R.worst_abs === 0
      ? '<span class="bold">exactly zero</span>, to the last bit of a double'
      : `<span class="bold">${R.worst_abs.toExponential(2)}</span>`}, against scores of order `
    + `${R.scale.toFixed(2)}. Drag the slider: the cloud does not move, the axis does, and the list `
    + `of what the axis "means" changes with it.`);

  set('rot-verdict',
    `Under that same transformation, the five books at the top of each factor keep `
    + `<span class="bold">${pc(R.top_overlap)}</span> of their places. So a factor is not a genre `
    + `and it is not a taste. It is a coordinate, and coordinates are chosen by the initialisation `
    + `and the update order. If you want an axis that means something, you have to ask for it, which `
    + `is exactly what the constraints in <a href="../directions/">the directions article</a> were `
    + `for: non negativity gave parts, independence gave sources, and neither is on offer here for `
    + `free.`);

  set('imp-intro',
    `Everything above predicts the number in the box, which is the wrong task and the previous `
    + `article said so. The version of this method that makes lists changes the input: forget the `
    + `stars, take a one where a book was rated and a zero everywhere else, and weight the ones by `
    + `how much you believe them. That turns ${n0(M.train_ratings)} observations into `
    + `${n0(M.users * M.books)} cells, which sounds impossible and is not, because the sum over all `
    + `the zeros is the same for every reader and can be computed once.`);

  set('imp-note',
    `On the same held out books, the same readers and the same protocol as the previous article, `
    + `implicit factorisation lands the held out favourite in a list of ten `
    + `<span class="bold">${pc(I.rank.hits['10'])}</span> of the time, against `
    + `<span class="bold">${pc(P.item_knn_hits10)}</span> for item neighbours and `
    + `${pc(P.popular_hits10)} for recommending the most rated books to everybody. `
    + `${impBeatsKnn
      ? `That is the comparison the previous article could not make, and it goes to the `
        + `factorisation by ${(I.rank.hits['10'] / P.item_knn_hits10).toFixed(2)} times.`
      : `So on this catalogue the neighbourhood method is still ahead at ranking, by `
        + `${(P.item_knn_hits10 / I.rank.hits['10']).toFixed(2)} times, which is worth saying `
        + `plainly rather than tuning until it is not true.`} `
    + `It also reaches ${n0(I.rank.coverage)} distinct books across every reader, against `
    + `${n0(P.item_knn_coverage)} for the neighbours.`);

  set('verdict-intro',
    `Four uses of the same three letters, and they do not all mean the same thing.`);

  set('v-svd', `Exact, in closed form, and nested: rank ${D.complete[0].rank} is inside rank `
    + `${D.complete[D.complete.length - 1].rank}.`);
  set('v-svd-warn', `Only when nothing is missing. Handed a sparse matrix it silently answers a `
    + `different question, in which every empty cell is a measured zero.`);

  set('v-als', `${E.als.toFixed(4)} on ${n0(M.test_cells)} held out ratings, from `
    + `${n0((M.users + M.books) * M.k)} numbers.`);
  set('v-als-warn', `The rank and the penalty are not details: the sweep runs from `
    + `${S.rank[0].test.toFixed(4)} to ${bestRank.test.toFixed(4)} across rank alone.`);

  set('v-imp', `${pc(I.rank.hits['10'])} in a list of ten, which is the number a recommender is `
    + `actually judged on.`);
  set('v-imp-warn', `It needs the confidence weighting and the Gram trick; done naively it is a `
    + `sum over ${n0(M.users * M.books)} cells per pass.`);

  set('v-rot', `Nothing. Predictions are invariant `
    + `${R.worst_abs === 0 ? 'exactly' : `to ${R.worst_abs.toExponential(1)}`} under any `
    + `invertible change of coordinates.`);
  set('v-rot-warn', `A factor is a coordinate, not a genre. If a slide names one, the name came `
    + `from the person making the slide.`);

  set('closing-1',
    `The useful summary is that a factorisation buys three things and costs one. It buys size: `
    + `${n0((M.users + M.books) * M.k)} numbers instead of a table of `
    + `${n0(M.books)} squared similarities. It buys a shared representation, so the same vectors `
    + `serve prediction, retrieval and neighbours. And it buys transfer, in the weak sense that a `
    + `reader's vector summarises everything they have done. What it costs is interpretability, and `
    + `not by a little: the rotation slider is a proof, not an anecdote.`);

  set('closing-2',
    `What it does not fix is the hole the previous article ended on. A book with no ratings has no `
    + `column, so it gets no vector, so it cannot be recommended, so it gets no ratings. The `
    + `factorisation is still working from ids alone: nothing in this page knows that two books `
    + `share an author, a series or a language, and the file has all three.`);

  set('closing-3',
    `The next article puts those in. If a book's vector can be computed from its <i>features</i> `
    + `rather than looked up by its id, a book nobody has read still has one, and the same idea `
    + `applies to the reader. That is a <span class="bold">two tower</span> model, and its shape is `
    + `chosen for a reason that has nothing to do with accuracy: it is what lets you score a million `
    + `items without running a model a million times. `
    + `<a href="../two-tower/">Both sides of the dot product</a> is next.`);

  set('ref-note',
    `The ratings, the split and the held out cells are the previous article's, asserted rather than `
    + `declared: the fingerprint of the held out set is <code>${M.split_fingerprint}</code> and this `
    + `generator refuses to run if it does not match. The digits are the ${n0(D.n)} images `
    + `<a href="../directions/">the directions article</a> factorised, rebuilt from the same seed, `
    + `and the identity against its published PCA error is checked in the browser as well as here. `
    + `Alternating least squares runs ${M.epochs} passes at ${M.k} factors with regularisation `
    + `${M.lam}, both swept on the page; the implicit fit uses confidence weighting with the Gram `
    + `matrix trick, so a pass costs the observed cells and not the ${n0(M.users * M.books)} of `
    + `them. Every number here comes from <code>src/utils/generate_mf_data.py</code>.`);

  return IDS;
}
