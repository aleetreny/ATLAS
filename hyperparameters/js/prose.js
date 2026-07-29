/* Every number in the running text, composed from the file at load.
 *
 * Same rule as the rest of the site: nothing here is typed, and every
 * comparison has both branches written, so rerunning the generator moves the
 * sentences instead of leaving them lying. Several of the sentences below
 * exist in two versions because the measurement went the other way from the
 * draft, and the honest reading is the one the file decides.
 */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const f3 = (v) => Number(v).toFixed(3);
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const signed = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(4)}`;

export const PROSE_IDS = [
  'opening-note', 'surface-intro', 'surface-note', 'search-intro', 'search-note',
  'bayes-intro', 'bayes-note', 'halving-intro', 'halving-note',
  'curse-intro', 'curse-note', 'verdict-intro',
  'verdict-grid', 'verdict-grid-warn', 'verdict-random', 'verdict-random-warn',
  'verdict-bayes', 'verdict-bayes-warn', 'verdict-halving', 'verdict-halving-warn',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const S = data.surface;
  const F = data.floor;
  const C = data.compare;
  const G = data.gp;
  const H = data.halving;
  const K = data.curse;
  const cells = M.grid * M.grid;

  const gridWins = C.svm.filter((r) => r.grid_test > r.rand_test).length;
  const inside = C.svm.filter((r) => Math.abs(r.grid_test - r.rand_test) <= F.test_sd).length;
  const flatInside = C.flattened.filter(
    (r) => Math.abs(r.grid_test - r.rand_test) <= F.test_sd).length;
  const flatRandWins = C.flattened.filter((r) => r.rand_test > r.grid_test).length;
  const big = C.svm[0];
  const gpBest = G.arms.noisy.test >= G.arms.noiseless.test ? 'noisy' : 'noiseless';
  const gpDiff = G.arms[gpBest].test - G.arms.random.test;
  const gpTie = Math.abs(gpDiff) <= G.arms.random.test_sd;
  const noiseTie = Math.abs(G.arms.noisy.test - G.arms.noiseless.test) < 1e-9;
  const cheap = H.rows[0];
  const wide = K.rows[K.rows.length - 1];
  const worstFloor = data.floor.rows.reduce((a, b) => (b.cv_sd > a.cv_sd ? b : a));

  set('opening-note',
    `The model is a kernel machine on the ${n0(M.n_train + M.n_test)} handwritten digits `
    + `that ship with scikit-learn, the same set the `
    + `<a href="../conformal/">conformal prediction article</a> used, split into `
    + `${n0(M.n_train)} rows to search on and ${n0(M.n_test)} that no search ever sees. `
    + `It has two dials, a penalty and a kernel width. Both were swept over `
    + `${M.grid} values, so there are <span class="bold">${n0(cells)} configurations</span>, `
    + `and every one of them was scored by ${M.repeats} repeats of ${M.folds} fold cross `
    + `validation and then again on the held out rows. That is ${n0(cells * (M.folds * M.repeats + 1))} `
    + `fits, and it buys the one thing an article about search needs: the answer.`);

  set('surface-intro',
    `Here is the whole thing. Bright is good, and the two axes are the two dials on a log `
    + `scale, which is how they are always searched and worth one sentence of justification: `
    + `these dials multiply, so the interesting question is never "plus one" but "times two". `
    + `The best cell by cross validation scores ${f4(S.best.cv)} and the best on held out `
    + `data scores ${f4(S.best.test)}; they are `
    + (S.best.at_cv[0] === S.best.at_test[0] && S.best.at_cv[1] === S.best.at_test[1]
      ? `the same cell.`
      : `not the same cell, which is the first hint of what the last section measures.`));

  set('surface-note',
    `The orange outline is the part of this picture that changes how the rest of the page `
    + `reads. Score one cell ${F.repeats} times with different fold draws and the answer `
    + `moves by ${f4(F.cv_sd)} on average, up to ${f4(worstFloor.cv_sd)} on the worst of the `
    + `three probed cells; resample the held out set and it moves ${f4(F.test_sd)}. Outlined `
    + `are the cells within that wobble of the best one, and there are `
    + `<span class="bold">${pct(G.near_optimal)}</span> of them. That is the real shape of `
    + `this problem: not a peak to be found but a plateau to be landed on, and a search `
    + `that lands anywhere on it has finished.`);

  set('search-intro',
    `The oldest argument in this subject is about the two pictures below. A lattice of `
    + `b points tries only the square root of b distinct values on each axis; b random `
    + `points try b distinct values on each. Bergstra and Bengio's paper is usually `
    + `remembered as "random beats grid", but the claim underneath is narrower and more `
    + `interesting: random beats grid <span class="bold">when only some of the dials `
    + `matter</span>, because then the lattice spends most of its budget re-measuring the `
    + `ones that do not.`);

  set('search-note',
    `Which makes it a claim about surfaces, not about search, and this page has surfaces. `
    + `On the kernel machine the two dials move the best reachable score by `
    + `${f3(S.importance.axis0)} and ${f3(S.importance.axis1)}: both matter, so the premise `
    + `does not hold and the prediction is that nothing much should separate the two `
    + `methods. It does not: the lattice wins ${gridWins} of the ${C.svm.length} budgets, `
    + `and ${inside} of the ${C.svm.length} differences are inside the ${f4(F.test_sd)} that `
    + `the held out set moves on its own. The forest is the reverse case, `
    + `${f3(data.forest.importance.axis0)} against `
    + `${f3(data.forest.importance.axis1)}. And the third button is the controlled version: `
    + `the same ${n0(cells)} measured scores with the first axis replaced by its own column `
    + `maximum, so that dial is switched off and the other keeps exactly the shape it was `
    + `measured to have. There the random draw wins ${flatRandWins} of `
    + `${C.flattened.length} budgets`
    + (flatInside === C.flattened.length
      ? `, and every difference is still inside the noise of the held out set, `
        + `which is the honest ending: at these budgets, on this plateau, the argument is `
        + `about a quantity too small to measure here.`
      : `, and ${C.flattened.length - flatInside} of those differences are larger than the `
        + `noise of the held out set, which is the mechanism showing up exactly where the `
        + `argument says it should.`)
    + ` What does separate the two is not the average but the spread: at ${big.budget} `
    + `evaluations a random draw has a spread of ${f4(big.rand_test_sd)} between draws and `
    + `the lattice has none, because the lattice is not a draw. Small budgets are a lottery `
    + `and the lattice is the ticket that always pays the same.`);

  set('bayes-intro',
    `Both methods so far are blind: neither looks at what it has already found. The `
    + `expensive alternative is to fit a model of the surface as you go and let it say `
    + `where to look next.`);

  set('bayes-note',
    `Over ${G.seeds} runs of ${G.budget} evaluations each, the surrogate ends on `
    + `${f4(G.arms[gpBest].test)} of held out accuracy and blind draws end on `
    + `${f4(G.arms.random.test)}. `
    + (gpTie
      ? `That difference, ${f4(Math.abs(gpDiff))}, is inside the ${f4(G.arms.random.test_sd)} `
        + `spread across runs, so <span class="bold">the model of the surface bought nothing `
        + `measurable here</span>. `
      : `That difference, ${f4(Math.abs(gpDiff))}, is larger than the `
        + `${f4(G.arms.random.test_sd)} spread across runs and favours `
        + `${gpDiff > 0 ? 'the surrogate' : 'the blind draw'}. `)
    + `The second view is the one that can tell them apart at all, because "best after `
    + `thirty" compresses every run into its last number: counting how many evaluations `
    + `each needs to reach within one wobble of the top, the surrogate takes a median of `
    + `${G.arms[gpBest].hits.median} and blind draws take `
    + `${G.arms.random.hits.median}. And the reason neither can shine is the plateau: with `
    + `${pct(G.near_optimal)} of cells already at the top, a blind draw expects to land on `
    + `one in ${G.expected_draws.toFixed(1)} tries. `
    + (noiseTie
      ? `Telling the surrogate that its observations carry noise changes nothing at all `
        + `here, to every decimal: the fold to fold wobble of ${f4(F.cv_sd)} is tiny next to `
        + `the ${f3(S.importance.best - Math.min(...S.cv.flat()))} the surface spans, so the `
        + `noise term is swamped. That is a fact about this surface, not about the idea.`
      : `Telling the surrogate that its observations carry noise moves its final score by `
        + `${f4(Math.abs(G.arms.noisy.test - G.arms.noiseless.test))}.`));

  set('halving-intro',
    `Everything above counts evaluations as if they cost the same, and they do not have to. `
    + `Successive halving trains every candidate briefly, throws away the worse half, `
    + `doubles the budget for the survivors, and repeats. It is the cheapest idea on this `
    + `page and it rests entirely on one assumption: that a candidate which looks bad on a `
    + `quarter of the data would have looked bad on all of it.`);

  set('halving-note',
    `That assumption is a rank correlation, so it can be drawn. Twenty four candidates from `
    + `across the surface, each scored at ${H.rows.map((r) => n0(r.n)).join(', ')} training `
    + `rows: at the smallest budget the ordering already agrees with the full one at `
    + `<span class="bold">${f3(cheap.spearman)}</span>, and `
    + `${pct(cheap.top_half_kept)} of the top half is still in the top half. `
    + `Halving costs the equivalent of ${H.cost_halving.toFixed(0)} full evaluations instead `
    + `of ${H.cost_full.toFixed(0)}, exactly ${H.saving.toFixed(2)} times less, and `
    + (H.halving_is_best
      ? `it keeps the candidate that turns out to be best.`
      : `it does not keep the best one. Whether that matters is a question about the size `
        + `of the loss and not about the algorithm, and the readout below does the `
        + `subtraction.`));

  set('curse-intro',
    `One thing is left, and it is the one that travels to every other article in this atlas. `
    + `A search returns two things: a configuration, and the score that made it win. The `
    + `configuration is fine. The score is not a score.`);

  set('curse-note',
    `The reason is that the maximum of b noisy numbers is biased upward, and the more you `
    + `look the worse it gets. Measuring it needs a second, independent cross validation of `
    + `the same cells, which this page has: the difference between the score that picked a `
    + `configuration and the score the same configuration gets when the folds are drawn `
    + `again is selection and nothing else. At ${wide.b} candidates it is `
    + `<span class="bold">${signed(wide.selection)}</span>. `
    + `The comparison everyone actually makes, cross validation against held out, gives `
    + `${signed(wide.gap)} instead, `
    + (Math.sign(wide.gap) !== Math.sign(wide.selection)
      ? `with the opposite sign, and the reason is worth the paragraph it takes. Cross `
        + `validation trains each fold on ${n0(M.n_train - Math.round(M.n_train / M.folds))} `
        + `rows while the final model is fitted on all ${n0(M.n_train)}, so it is `
        + `pessimistic by ${f4(Math.abs(K.honest_gap))} on an average cell. Two biases, `
        + `opposite directions, and on this problem the second is the larger. Anyone `
        + `reporting the second number alone would conclude there is no winner's curse `
        + `here, and would be wrong: it is there, it is just outweighed.`
      : `so both biases point the same way and the held out comparison sees their sum.`));

  set('verdict-intro',
    `Four methods, one surface, and the honest summary is that the surface decided most of `
    + `it. Two of the four rows below say "not measurably better", which is a result and `
    + `not a failure to measure: the noise floor is printed next to every one of them.`);

  set('verdict-grid',
    `Lands on ${f4(C.svm[C.svm.length - 1].grid_test)} at `
    + `${C.svm[C.svm.length - 1].budget} evaluations, with no run to run spread at all, `
    + `because it is not a draw.`);
  set('verdict-grid-warn',
    `Tries only ${C.svm[C.svm.length - 1].distinct_grid} distinct values per dial out of `
    + `${M.grid}. On a surface where one dial does nothing, that is most of the budget spent `
    + `on nothing.`);
  set('verdict-random',
    `Averages ${f4(C.svm[C.svm.length - 1].rand_test)} at the same budget and tries `
    + `${C.svm[C.svm.length - 1].distinct_rand} distinct values per dial.`);
  set('verdict-random-warn',
    `Spread of ${f4(big.rand_test_sd)} between draws at ${big.budget} evaluations. A single `
    + `small random search is a lottery ticket, and its result should be reported as one.`);
  set('verdict-bayes',
    `Ends on ${f4(G.arms[gpBest].test)} and reaches the target in a median of `
    + `${G.arms[gpBest].hits.median} evaluations against `
    + `${G.arms.random.hits.median} for blind draws.`);
  set('verdict-bayes-warn',
    gpTie
      ? `Bought ${f4(Math.abs(gpDiff))} over blind draws, inside a run to run spread of `
        + `${f4(G.arms.random.test_sd)}. It has to fit a model after every evaluation, so on `
        + `a plateau like this one it is paying for nothing.`
      : `Fits a model after every evaluation, which is only worth it when an evaluation is `
        + `expensive relative to that.`);
  set('verdict-halving',
    `Costs ${H.saving.toFixed(2)} times less than scoring everything at full size, on an `
    + `assumption that holds here at rank correlation ${f3(cheap.spearman)}.`);
  set('verdict-halving-warn',
    H.halving_is_best
      ? `The assumption is a property of the model and the data, not a theorem. Curves that `
        + `cross late, which is the usual failure, are invisible at small budgets.`
      : `It dropped the eventual winner here. The assumption is a property of the model and `
        + `the data, not a theorem.`);

  set('closing-1',
    `The through line of this page is that <span class="bold">the search matters less than `
    + `the shape of what is being searched</span>, and that the shape is measurable before `
    + `any method is chosen. On this surface ${pct(G.near_optimal)} of configurations are `
    + `already at the top, so four methods finish within a few thousandths of each other and `
    + `of the answer. On a surface with a genuine needle none of that would hold, and the `
    + `way to find out which one you have is the same either way: sweep coarsely, look at `
    + `the spread, and only then decide whether a clever search is worth its own cost.`);

  set('closing-2',
    `The last section leaves a debt. If the best score of ${wide.b} candidates is biased by `
    + `${signed(wide.selection)}, then a system that tries thousands of pipelines instead of `
    + `${n0(cells)} cells of one model has the same problem, larger. That is the next `
    + `article, <a href="../automl/">on what happens when the search chooses the model as `
    + `well as its dials</a>, and it is where the bias measured here stops being a curiosity `
    + `and starts being the thing that decides the leaderboard.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/utils/generate_hyperparameters_data.py</span>, seed `
    + `${M.seed}. The surface is ${M.grid} by ${M.grid} values of log2 C and log2 gamma for a `
    + `radial basis kernel machine on ${n0(M.n_train)} of the ${n0(M.n_train + M.n_test)} `
    + `digits, ${M.features} columns and ${M.classes} classes, scored by ${M.repeats} repeats `
    + `of ${M.folds} fold stratified cross validation and on the ${n0(M.n_test)} held out `
    + `rows. The whole grid was scored a second time with different fold seeds, which is what `
    + `makes the winner's curse separable from the training size bias. The noise floors come `
    + `from ${F.repeats} refits of three cells and from 400 bootstrap resamples of the held `
    + `out set. Grid against random is ${n0(M.search_seeds)} draws per budget; the Gaussian `
    + `process is written in the generator, not imported, with a squared exponential kernel `
    + `over grid indices and expected improvement, ${G.seeds} runs of ${G.budget} evaluations `
    + `starting from ${G.init} random ones. Nothing on this page is a wall clock: cost is `
    + `counted in fits and in evaluations, which is the only form that survives being read on `
    + `another machine.`);
}
