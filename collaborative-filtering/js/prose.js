/* Every number on this page, composed from the file it was measured into.
 *
 * Nothing here is written by hand: each comparison is a branch, so that
 * rerunning the generator moves the prose with the measurement instead of
 * leaving a sentence quietly describing last week's numbers.
 */
const IDS = ['intro-shape', 'intro-plan', 'hole-intro', 'hole-note', 'sim-intro', 'sim-math',
  'sim-verdict', 'rank-intro', 'rank-note', 'rank-cost', 'slice-intro', 'slice-note',
  'verdict-intro', 'v-bias', 'v-bias-warn', 'v-item', 'v-item-warn', 'v-sim', 'v-sim-warn',
  'v-pop', 'v-pop-warn', 'closing-1', 'closing-2', 'closing-3', 'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const FLAVOUR_NAME = {
  shrunk: 'shrunk centred cosine', centred: 'centred cosine',
  cosine: 'cosine on the ones and zeros', jaccard: 'the overlap',
};
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initProse(data) {
  const M = data.meta;
  const S = data.shape;
  const E = data.rating.rmse;
  const R = data.rank;
  const RM = data.rank_meta;
  const F = data.flavours;
  const U = data.user_knn;
  const SL = data.slices;

  const bestK = data.rating.sweep_k.reduce((a, x) => (x.rmse < a.rmse ? x : a));
  const bestShrink = data.rating.sweep_shrink.reduce((a, x) => (x.rmse < a.rmse ? x : a));
  const noShrink = data.rating.sweep_shrink.find((x) => x.shrink === 0);
  /* the smallest k that is within a thousandth of the best, which is the
     honest way to say "and past here it stops mattering" without calling a
     plateau by eye */
  const plateauK = data.rating.sweep_k.find((x) => x.rmse <= bestK.rmse * 1.001);
  const lastK = data.rating.sweep_k[data.rating.sweep_k.length - 1];
  const knnWins = E.item_knn < E.bias;
  /* the two directions on the SAME cells: the headline item number is measured
     over every held out cell and the user one only over the sampled readers */
  const itemVsUser = U.rmse - U.item_rmse_same_cells;
  const popGap = R.item_knn.hits['10'] / R.popular.hits['10'];
  const flavourWorst = Object.entries(F).reduce((a, x) => (x[1].pop_pct > a[1].pop_pct ? x : a));
  const flavourBest = Object.entries(F).reduce((a, x) => (x[1].pop_pct < a[1].pop_pct ? x : a));

  set('intro-shape',
    `The table is ${n0(S.users)} readers by ${n0(S.books)} books, which is `
    + `${n0(S.cells)} cells, and ${n0(S.ratings)} of them have a number in them. That is `
    + `<span class="bold">${pc(S.density, 2)}</span> full. Everything this article does is an `
    + `attempt to fill in the rest of it, or at least to sort each row of it, using nothing but the `
    + `numbers already there: no titles, no authors, no genres, no cover art. The titles exist in `
    + `the file and they are used for exactly one thing, which is letting you check the answers by `
    + `eye.`);

  set('intro-plan',
    `The plan is the one that has held up since 1998. Two things are similar when the same people `
    + `like them; two people are similar when they like the same things; and a rating you have not `
    + `seen is a weighted average of the ones you have. It is a shorter idea than most of this `
    + `atlas and it needs one guard rail that most write-ups skip, which arrives in the third `
    + `section: before believing any of it, check what recommending the ${n0(M.top_k)} most rated `
    + `books to everybody scores.`);

  set('hole-intro',
    `A recommender's first decision is made before any modelling: which books are allowed to exist. `
    + `The catalogue here has a book with ${n0(S.per_book_max)} ratings and a book with `
    + `${S.per_book_min}, and the median is ${n0(Math.round(S.per_book_median))}. Half of all the `
    + `ratings in the file are spent on the top <span class="bold">${n0(S.half_books)}</span> books, `
    + `and nine tenths of them on the top ${n0(S.ninety_books)}. Drag the threshold and watch what `
    + `a cut costs.`);

  set('hole-note',
    `The reason this matters is the same reason every number below has two versions. A cut that `
    + `removes most of the catalogue removes almost none of the data, so it looks free on every `
    + `average you can compute, and it is not free at all for the reader who wanted the book you `
    + `just deleted. And this catalogue has already been cut once before it got here: the thinnest `
    + `book in it has ${S.per_book_min} ratings and the thinnest reader has ${S.per_user_min}, `
    + `because whoever assembled the file dropped everything below that. Every result on this page `
    + `is therefore an optimistic one, measured on a catalogue with the hardest part already `
    + `removed.`);

  set('sim-intro',
    `Similarity is the whole of this method and it is not one thing. A book, to this model, is a `
    + `column of ${n0(S.users)} numbers, mostly missing, and there are several honest ways to ask `
    + `how alike two of those columns are. The widget below computes four of them in the page, from `
    + `the same shared counts, for the ${data.widget.items.length} most rated books in the `
    + `catalogue.`);

  set('sim-math',
    `Two of them never look at the stars at all: cosine on the ones and zeros, `
    + `$$s_{ij} = \\frac{|U_i \\cap U_j|}{\\sqrt{|U_i|\\,|U_j|}}$$ and the same counts arranged as `
    + `an overlap, \\(|U_i \\cap U_j| / |U_i \\cup U_j|\\). The third uses the ratings as they come. `
    + `The fourth subtracts what the bias model already knows, `
    + `$$s_{ij} = \\frac{\\sum_{u \\in U_i \\cap U_j} e_{ui} e_{uj}}{\\|e_i\\| \\, \\|e_j\\|}, \\qquad `
    + `e_{ui} = r_{ui} - \\mu - b_u - b_i$$ so that two books count as similar when the same people `
    + `disagree about them in the same direction. The slider multiplies every similarity by `
    + `\\(n_{ij}/(n_{ij} + \\lambda)\\), which is how you say that a similarity computed from four `
    + `shared readers is not a small number, it is an unknown one.`);

  set('sim-verdict',
    `The column to watch is the last one. Averaged over the whole catalogue, the top eight `
    + `neighbours that <span class="bold">${FLAVOUR_NAME[flavourWorst[0]]}</span> returns sit at the `
    + `<span class="bold">${(100 * flavourWorst[1].pop_pct).toFixed(1)}th</span> percentile of `
    + `popularity, and the ones <span class="bold">${FLAVOUR_NAME[flavourBest[0]]}</span> returns at `
    + `the <span class="bold">${(100 * flavourBest[1].pop_pct).toFixed(1)}th</span>. Restricted to `
    + `the hundred most rated books, where the pull is strongest, those two sit at the `
    + `${(100 * flavourWorst[1].top_pop_pct).toFixed(1)}th and the `
    + `${(100 * flavourBest[1].top_pop_pct).toFixed(1)}th. A similarity that answers "the popular ones" to every `
    + `question is not wrong, exactly, and it is not personalisation either: the next section is `
    + `about how easy it is to score well while doing it.`
    + `<br /><br />`
    + `But the obvious moral of that comparison is the wrong one, and the second column says why. `
    + `The less popular neighbours are bought with evidence: `
    + `${pc(F.centred.co_under5)} of the ones centred cosine picks come from fewer than five shared `
    + `readers, against ${pc(F.shrunk.co_under5, 2)} once shrinkage is applied and `
    + `${pc(F.cosine.co_under5, 2)} for plain cosine on the ones and zeros. There are `
    + `${n0(data.rating.pairs.pairs)} pairs of books in this catalogue, `
    + `${pc(data.rating.pairs.under5)} of them share fewer than five readers and `
    + `${pc(data.rating.pairs.zero)} share none at all, so a rule that prefers rare books is mostly `
    + `a rule that prefers unmeasured ones. The honest version of "avoid the popularity bias" is `
    + `that you can, and it costs you evidence, and the sweep in the next section prices it.`);

  set('rank-intro',
    `Two tasks live under the word recommendation and they are scored differently. Filling in the `
    + `box asks for a number: hold out ${M.held_per_user} ratings from every reader, `
    + `${n0(M.test_cells)} in all, and measure how far off the guess is. Building the list asks for `
    + `an order: of the ${n0(RM.held_items)} held out books that these ${n0(RM.eval_users)} readers `
    + `went on to give four or five stars, how many turn up in a list of ten drawn from the whole `
    + `${n0(RM.catalogue)} book catalogue.`);

  set('rank-note',
    `The control is the point of the chart. Handing every reader the same ten most rated books, `
    + `with no personalisation of any kind, lands the held out book `
    + `<span class="bold">${pc(R.popular.hits['10'])}</span> of the time. Neighbours do `
    + `${popGap >= 1 ? `${popGap.toFixed(2)} times better` : `${(1 / popGap).toFixed(2)} times worse`}, `
    + `which is a real gap and a much smaller one than the phrase "personalised recommendations" `
    + `suggests. Sorting by average rating instead does ${pc(R.book_mean.hits['10'])}, `
    + `${R.book_mean.hits['10'] < R.popular.hits['10'] ? 'worse than popularity' : 'better than popularity'}, `
    + `and the reason is in its own numbers: the books it puts first sit at the `
    + `${(100 * R.book_mean.pop_pct).toFixed(0)}th percentile of popularity and it only ever names `
    + `${R.book_mean.coverage} of them across every reader, because a near perfect average is `
    + `something only a book almost nobody has rated can have. And part of what the popularity `
    + `baseline earns is built into the test set: `
    + `the held out books sit in the top ${pc(1 - RM.held_pop_pct)} of the catalogue by popularity, `
    + `because people mostly read what people mostly read.`);

  set('rank-cost',
    `The other direction of the same idea is to find people like you rather than books like the `
    + `book, and it costs differently. There are ${n0(S.books)} books and ${n0(S.users)} readers, so `
    + `the full book to book table is ${n0(U.sims_item)} similarities and the full reader to reader `
    + `table is ${n0(U.sims_user_full)}, a factor of ${U.ratio.toFixed(1)}. Measured on exactly the `
    + `same ${n0(U.n_test)} held out cells, with the same shrinkage and the same bias model `
    + `underneath, reader neighbours score <span class="bold">${U.rmse.toFixed(4)}</span> and book `
    + `neighbours <span class="bold">${U.item_rmse_same_cells.toFixed(4)}</span>, `
    + `${itemVsUser > 0 ? 'so the cheaper direction is also the better one'
      : 'so the more expensive direction is the better one here'}. `
    + `${U.rmse > U.bias_rmse_same_users
      ? `And there is a harder number in that comparison: on those cells the bias model alone scores `
        + `<span class="bold">${U.bias_rmse_same_users.toFixed(4)}</span>, so reader neighbours do `
        + `not beat three numbers added together, while book neighbours do.`
      : `Both beat the bias model's ${U.bias_rmse_same_users.toFixed(4)} on those cells.`} `
    + `There is a structural reason on top of the arithmetic: the book to book table changes when a `
    + `book gets new readers, which is slow, and the reader to reader table changes every time `
    + `anybody rates anything.`);

  set('slice-intro',
    `One error number for the whole site is an average over two populations that behave nothing `
    + `like each other. Split it, and the method's real shape shows up: on the cells it can answer, `
    + `it is ${knnWins ? 'a modest improvement' : 'not an improvement'}, and the thing that varies `
    + `most is not the error but whether it has an answer at all.`);

  set('slice-note',
    `That dashed line is the one to take away. Item neighbours only produce a number when the `
    + `reader has rated something that counts as a neighbour of the target book, and for the least `
    + `known tenth of held out books that happens ${pc(SL.book[0].covered)} of the time against `
    + `${pc(SL.book[SL.book.length - 1].covered)} for the best known tenth. Where they `
    + `do not, the answer falls back to the bias model, so the headline error is a blend of a `
    + `method and its fallback in a proportion nobody reports. Over all `
    + `${n0(M.test_cells)} held out cells, the method has an opinion `
    + `${pc(data.rating.covered)} of the time.`);

  set('verdict-intro',
    `Four decisions, and the last one is the one that decides whether the other three meant `
    + `anything.`);

  set('v-bias',
    `${E.bias.toFixed(4)} on held out ratings, from three numbers added together.`);
  set('v-bias-warn',
    `It cannot make a list. A rule for filling in a box has nothing to say about which ten of `
    + `${n0(S.books)} books to show.`);

  set('v-item',
    `${pc(R.item_knn.hits['10'])} of held out favourites in a list of ten, against `
    + `${pc(R.popular.hits['10'])} for the popularity control.`);
  set('v-item-warn',
    `It reaches ${n0(R.item_knn.coverage)} distinct books across every reader. `
    + `${SL.unreachable > 0 ? `${n0(SL.unreachable)} books are nobody's neighbour at all.`
      : 'Every book is somebody\'s neighbour, but most are nobody\'s top ten.'}`);

  set('v-sim',
    `Neighbours in the top ${pc(1 - F.shrunk.pop_pct)} of the catalogue by popularity, on a median `
    + `of ${n0(Math.round(F.shrunk.co_median))} shared readers.`);
  set('v-sim-warn',
    `Unshrunk, ${pc(F.centred.co_under5)} of the neighbours it picks come from fewer than five `
    + `shared readers, and the best measured shrinkage is `
    + `${bestShrink.shrink === 0 ? 'none' : bestShrink.shrink} `
    + `(${bestShrink.rmse.toFixed(4)} against ${noShrink.rmse.toFixed(4)} with none).`);

  set('v-pop',
    `${pc(R.popular.hits['10'])} at ten, from a model with no parameters and no reader in it.`);
  set('v-pop-warn',
    `Anything that does not beat this by a margin you can defend is not a recommender, and the held `
    + `out set is on its side: held out books sit in the top ${pc(1 - RM.held_pop_pct)} of the `
    + `catalogue.`);

  set('closing-1',
    `Collaborative filtering is one honest idea and a pile of decisions that each move the answer `
    + `more than the idea does. Which similarity: `
    + `${pc(F.cosine.pop_pct)} against ${pc(F.centred.pop_pct)} mean catalogue percentile between `
    + `the two extremes measured here. How much to distrust a similarity built from four readers: `
    + `the best shrinkage measured is <span class="bold">${bestShrink.shrink}</span>, which is not a `
    + `tweak of the usual twenty five, it is a different regime. And how many neighbours, where the `
    + `sweep says something the word neighbourhood does not prepare you for: the error falls all the `
    + `way from ${data.rating.sweep_k[0].rmse.toFixed(4)} at k = 1 to `
    + `<span class="bold">${bestK.rmse.toFixed(4)}</span> at k = ${n0(bestK.k)}, and it is already `
    + `within a thousandth of that by k = ${n0(plateauK.k)}. `
    + `${lastK.k >= S.books / 4
      ? `At ${n0(lastK.k)} of ${n0(S.books)} books this is not a neighbourhood any more: it is a `
        + `weighted average over most of the catalogue, wearing the name of one.`
      : 'The turn is inside the range measured.'} `
    + `None of those decisions is in the name of the method.`);

  set('closing-2',
    `And the ceiling is structural, not a matter of tuning. Every prediction here is a weighted `
    + `average of ratings the reader has already given, so a book with no ratings has no `
    + `neighbours, a reader with no ratings has no neighbours, and the table has to be scanned `
    + `again for every recommendation. What is missing is any notion of <i>why</i> two books go `
    + `together, which is the thing you would need in order to say anything about a book nobody has `
    + `read yet.`);

  set('closing-3',
    `The next article puts that notion in. Instead of comparing columns of the table, it `
    + `<span class="bold">factorises</span> the table: every reader and every book becomes a short `
    + `vector, and a rating becomes their dot product. That turns out to be the same factorisation `
    + `<a href="../directions/">the directions article</a> ran on complete data under three `
    + `different constraints, with a fourth one added that changes everything, which is that almost `
    + `every entry is missing. <a href="../matrix-factorization/">A few numbers per reader</a> is `
    + `where the module goes next.`);

  set('ref-note',
    `Everything above is measured on goodbooks-10k, ${n0(S.ratings)} ratings from ${n0(S.users)} `
    + `readers over ${n0(S.books)} books, kept outside the repository and checked by digest `
    + `(<code>${M.digest_ratings}</code>) so that a mirror changing under us shows up as a failure `
    + `rather than as a different article. The split holds out ${M.held_per_user} ratings per `
    + `reader, chosen at random with seed ${M.seed}, giving ${n0(M.test_cells)} held out cells; its `
    + `fingerprint is <code>${M.split_fingerprint}</code> and the next article asserts it. `
    + `Ranking is measured on ${n0(RM.eval_users)} readers sampled from the same split, scoring `
    + `every book they have not already rated. Neighbour methods use k = ${M.k} and shrinkage `
    + `${M.shrink}; both are swept on the page. The bias model is fitted by alternating updates `
    + `with regularisation 10 on each side. The browser recomputes the four similarities for the `
    + `${data.widget.items.length} most rated books from shared counts and dot products, and a `
    + `guard checks its cosine against this file's before anything is drawn, from `
    + `<code>src/utils/generate_cf_data.py</code>.`);

  return IDS;
}
