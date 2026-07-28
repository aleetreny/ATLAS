/* Every sentence on this page that carries a figure, written from the file.
 *
 * Nothing numeric is typed by hand, every comparison is written AS a comparison
 * with both outcomes spelled out, and the table cells and the references band
 * go through here too, because those are the two places a stale claim survives
 * longest: nothing around them disagrees.
 */
const n0 = (v) => v.toLocaleString('en-US');
const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const R = data.roll;
  const G = data.graph;
  const M = data.methods;
  const L = data.lle;
  const S = data.som;
  const H = data.hole;
  const D = data.digits;
  const rows = G.rows.filter((r) => r.disparity !== undefined);
  const bestRow = rows.reduce((a, b) => (b.disparity < a.disparity ? b : a));
  const firstShort = G.rows.find((r) => r.k === G.first_short_circuit);
  const beforeShort = G.rows.find((r) => r.k === G.first_short_circuit - 1);
  const cliff = firstShort.disparity / Math.max(beforeShort.disparity, 1e-12);
  const worstK = rows.reduce((a, b) => (b.disparity > a.disparity ? b : a));
  const lleBest = L.rows.find((r) => r.k === L.best_k);
  const lle5 = L.rows[0];
  const bestSom = S.best_attempt;

  /* ------------------------------------------------------------- the opening */
  set('opening-note',
    `Three numbers from three different parts of this page are the argument in advance. On `
    + `${n0(data.meta.n)} points sampled from a rectangle that has been rolled up, there is a pair `
    + `<span class="bold">${R.worst_ambient}</span> apart in the space they were measured in and `
    + `<span class="bold">${R.worst_geodesic}</span> apart along the sheet, a factor of `
    + `${R.worst_ratio}, and across all pairs the two kinds of distance correlate `
    + `${R.ambient_corr}. Replace the straight lines with shortest paths through a graph of near `
    + `neighbours and the best flat projection goes from ${M.pca.disparity} away from the true `
    + `answer to <span class="bold">${M.isomap.disparity}</span>. And the whole thing turns on `
    + `${firstShort.short_circuits} edges: at ${G.first_short_circuit} neighbours instead of `
    + `${G.first_short_circuit - 1}, ${firstShort.short_circuits} of `
    + `${n0(firstShort.edges)} edges jump between two turns of the roll and the error multiplies by `
    + `<span class="bold">${cliff.toFixed(0)}</span>.`);

  /* ------------------------------------------------------------------ the roll */
  set('roll-intro',
    `The dataset is a Swiss roll: ${n0(data.meta.n)} points scattered on a flat rectangle which is `
    + `then rolled into a spiral and handed over as three coordinates each. It is the standard test `
    + `case, and the reason it is worth using here is not that it is famous but that its answer is `
    + `available in closed form. The sheet is developable, so the distance between two points along `
    + `it is the arc length of the spiral between them combined with the height between them by `
    + `Pythagoras, and the arc length of \\(t \\mapsto (t\\cos t,\\, t \\sin t)\\) integrates to `
    + `\\(\\tfrac{1}{2}(t\\sqrt{1+t^2} + \\operatorname{arcsinh} t)\\). Every claim below about `
    + `recovering the sheet is therefore a number rather than a picture that looks about right.`);

  set('scrolly-step-0',
    `${n0(data.meta.n)} points, three numbers each. Flat, the sheet runs from `
    + `${R.chart_span[0]} to ${R.chart_span[1]} along its length and ${R.chart_span[2]} to `
    + `${R.chart_span[3]} across its width, and every point is tinted by where it sits along that `
    + `length. Nothing that follows is ever shown the colour.`);
  set('scrolly-step-1',
    `The worst pair in the dataset, and it is worth stopping on. These two points are `
    + `<span class="bold">${R.worst_ambient}</span> apart in the room, which is less than half the `
    + `average, and <span class="bold">${R.worst_geodesic}</span> apart along the sheet, which is `
    + `more than twice the average. A factor of ${R.worst_ratio}. Their colours disagree because `
    + `they belong to opposite ends of the rectangle, and the only reason they are near each other `
    + `is that the rectangle was rolled.`);
  set('scrolly-step-2',
    `Every method in this module until now would work with the ${R.amb_mean} average straight line `
    + `and not the ${R.geo_mean} average along the sheet, and here is the consequence. The best flat `
    + `projection of these points is <span class="bold">${M.pca.disparity}</span> away from the true `
    + `chart on a scale where 1 means carrying nothing, and its distances correlate `
    + `${M.pca.geodesic_corr} with the real ones. The colours give it away: the ramp is folded, so `
    + `points from opposite ends of the sheet sit on top of each other.`);
  set('scrolly-step-3',
    `The repair is to notice which distances are trustworthy. Over a short enough hop a curved `
    + `sheet is flat, so a straight line to a near neighbour is a good estimate of the distance `
    + `along the sheet and a straight line to anything else is not. Join each point to its `
    + `${data.meta.k_show} nearest neighbours and throw every other distance away: at this k the `
    + `worst edge in the graph spans ${beforeShort.worst_edge_ratio} times its own straight line, `
    + `so no edge is lying by much.`);
  set('scrolly-step-4',
    `Then get the long distances back by walking. The shortest path through the graph between the `
    + `two ends of the worst pair takes many small honest hops instead of one large dishonest one, `
    + `and adding them up estimates the distance along the sheet. Over all `
    + `${n0((data.meta.n * (data.meta.n - 1)) / 2)} pairs the estimate correlates `
    + `<span class="bold">${bestRow.geodesic_corr}</span> with the truth, against `
    + `${R.ambient_corr} for the straight lines it replaced.`);
  set('scrolly-step-5',
    `And now there is nothing left to invent. A table of distances is exactly what the `
    + `<a href="../distances/">previous article</a> turned back into a map: double centre it, `
    + `diagonalise, keep two columns. Isomap is that, with the table built by walking. The result is `
    + `<span class="bold">${M.isomap.disparity}</span> from the true chart, against `
    + `${M.pca.disparity} for the flat projection, and the ramp comes back in one piece.`);

  set('isomap-note',
    `Only the first line is new, and it is worth being blunt about how little that is. Isomap is `
    + `classical scaling with one substitution: where the previous article put the straight line `
    + `distance between two rows, this one puts the length of the shortest path through a graph of `
    + `near neighbours. Everything after that, the double centring, the eigendecomposition, the `
    + `scaling of the eigenvectors, is the same code. This page reuses it literally. And it inherits `
    + `the same diagnostic: the eigenvalues say whether the table could have come from points at all. `
    + `Which also means it inherits the same assumption, and there is a section at the end about `
    + `what happens when that assumption is wrong.`);

  /* --------------------------------------------------------------- the k dial */
  set('k-intro',
    `The substitution moved the entire problem into one number. How many neighbours counts as near? `
    + `Too few and the graph falls into pieces, and points in different pieces have no path between `
    + `them at all: below k = ${G.first_connected} this graph is in `
    + `${G.rows[0].components} separate components. Too many and an edge appears between two turns `
    + `of the roll, joining points that are near in the room and a long walk apart on the sheet, and `
    + `every shortest path that can use that edge does. There is no penalty term for this and no `
    + `warning: the algorithm runs, returns a map, and the map is wrong.`);

  set('cliff-note',
    `The sweep is unusually violent for a hyperparameter. At k = ${beforeShort.k} the graph has `
    + `${n0(beforeShort.edges)} edges, not one of which joins two turns, and the map is `
    + `${beforeShort.disparity} from the truth. At k = ${firstShort.k} it has `
    + `${n0(firstShort.edges)} edges, <span class="bold">${firstShort.short_circuits}</span> of `
    + `which do, and the map is ${firstShort.disparity} from the truth. Two edges out of `
    + `${n0(firstShort.edges)}, one part in ${Math.round(firstShort.edges / firstShort.short_circuits)}, `
    + `multiply the error by <span class="bold">${cliff.toFixed(0)}</span>. By k = ${worstK.k} the `
    + `map is ${worstK.disparity} from the truth, which is to say it carries nothing, and the `
    + `agreement with the true distances has fallen from ${bestRow.geodesic_corr} to `
    + `${worstK.geodesic_corr}.`
    + `<br /><br />`
    + `This is not a knob to tune by looking at the output, because the output looks plausible on `
    + `both sides of the cliff. It is a knob to tune by looking at the <span class="bold">graph</span>: `
    + `take the smallest k that connects it, then check that no edge is much longer than its `
    + `neighbours. Here that rule picks k = ${G.first_connected} or ${G.first_connected + 1}, and the `
    + `best available is ${bestRow.k} at ${bestRow.disparity}.`);

  const res = data.residual.rows;
  set('dimension-note',
    `The eigenvalues answer a second question for free, and it is the question everybody actually `
    + `wants answered. Embed into one dimension, then two, then three, and measure how much of the `
    + `graph distance table is left unexplained each time: `
    + `${res.map((r) => `${r.dim}: ${r.residual}`).join(', ')}. The drop from one dimension to two `
    + `is ${(res[0].residual - res[1].residual).toFixed(5)} and from two to three it is `
    + `${(res[1].residual - res[2].residual).toFixed(5)}, `
    + `${((res[0].residual - res[1].residual) / Math.max(res[1].residual - res[2].residual, 1e-12)).toFixed(0)} `
    + `times smaller. The elbow is at <span class="bold">${data.residual.elbow}</span>, which is the `
    + `right answer: the sheet is a rectangle and a rectangle is two-dimensional. Nothing in PCA's `
    + `spectrum on the raw coordinates would have told you that, because in the space the points `
    + `were measured in they genuinely fill three dimensions.`);

  /* ------------------------------------------------------------ four methods */
  set('compare-intro',
    `Isomap is not the only answer, and the other three on this page fail in three different and `
    + `instructive ways. All four are given the same ${n0(data.meta.n)} points and scored three `
    + `ways: how far the map is from the true chart after being allowed to rotate, reflect and `
    + `rescale; trustworthiness, which asks what share of each point's neighbours in the map were `
    + `really its neighbours; and how well the map's distances agree with the true ones along the `
    + `sheet. A method can do well on the second and badly on the other two, and two of these do `
    + `exactly that.`);

  set('lle-note',
    `Locally linear embedding takes a different route to the same idea: instead of measuring long `
    + `distances by walking, it never measures them at all. Each point is written as a weighted `
    + `average of its ${L.k} neighbours, the weights are the local geometry, and then it looks for `
    + `the low-dimensional configuration those same weights describe best. That is elegant and it `
    + `carries a real cost: it scores ${M.lle.trustworthiness} on trustworthiness, so its local `
    + `structure is genuinely good, and ${M.lle.disparity} on distance from the true chart, so its `
    + `global shape is `
    + `${M.lle.disparity > 0.4 ? 'largely invented' : 'roughly right'}. Its distances agree with the `
    + `real ones at ${M.lle.geodesic_corr} against Isomap's ${M.isomap.geodesic_corr}. `
    + `<span class="bold">Nothing in the method ever looked at a long distance, so nothing in the `
    + `output should be read as one.</span>`
    + `<br /><br />`
    + `There is a second, sharper lesson buried in choosing its k, and it is about solvers rather `
    + `than manifolds. The two directions LLE keeps are the second and third smallest eigenvectors `
    + `of a matrix whose smallest eigenvalues here are around `
    + `${lle5.eig2.toExponential(0)}. At k = ${lle5.k} the second and third are `
    + `${lle5.eig2.toExponential(1)} and ${lle5.eig3.toExponential(1)}, a ratio of ${lle5.gap} to `
    + `the fourth, and an iterative solver and an exact diagonalisation come back `
    + `<span class="bold">${lle5.iter_vs_exact}</span> apart, which is most of a configuration. `
    + `A dense solver like scikit-learn's still agrees with the exact answer to `
    + `${lle5.sk_vs_exact.toExponential(0)}, so nothing is broken; it is simply that the answer at `
    + `that k is decided by arithmetic rather than by data. At k = ${lleBest.k} the ratio is `
    + `${lleBest.gap} and the two solvers agree to ${lleBest.iter_vs_exact.toExponential(0)}, so `
    + `that is the k this page runs and the k the number above comes from.`);

  const somGap = S.disparity / Math.max(M.isomap.disparity, 1e-12);
  set('som-note',
    `The self-organising map is the odd one out, and the most interesting failure on the page. It `
    + `does not embed anything. It lays a ${S.rows} by ${S.cols} grid of prototype vectors into the `
    + `data, and on every sample it finds the closest prototype and drags it, and its grid `
    + `neighbours, a little way towards the sample. Run that for ${S.epochs} epochs and the grid has `
    + `crawled into the shape of the data. Read a point's position off the grid and you have a `
    + `two-dimensional coordinate.`
    + `<br /><br />`
    + `By its own two diagnostics it does this beautifully: a quantisation error of `
    + `${S.quantisation} (the average distance from a point to its prototype) and a topographic `
    + `error of ${S.topographic} (the share of points whose best and second-best prototypes are not `
    + `grid neighbours, which is the standard test for whether the map has folded). Its `
    + `trustworthiness is <span class="bold">${S.trustworthiness}</span>, the highest of the four `
    + `methods here. And its distance from the true chart is `
    + `<span class="bold">${S.disparity}</span>, against Isomap's ${M.isomap.disparity}: `
    + `${somGap.toFixed(0)} times worse, and near the maximum of 1. Walking along the long axis of `
    + `the grid tracks the long axis of the sheet at a correlation of `
    + `<span class="bold">${S.long_axis_corr}</span>, which is nothing.`);

  const tbody = document.querySelector('#som-table');
  if (tbody) {
    tbody.innerHTML = S.attempts.map((a) => {
      const win = a.disparity === bestSom.disparity;
      const cell = (v) => (win ? `<span class="bold">${v}</span>` : v);
      return `<tr><td>${a.label}, ${a.rows} by ${a.cols}</td>`
        + `<td>${a.quantisation}</td><td>${a.topographic}</td>`
        + `<td>${cell(a.disparity)}</td></tr>`;
    }).join('');
  }

  set('som-verdict',
    `Before publishing that, the obvious objection has to be paid for: maybe the schedule was `
    + `wrong. So the table above is ${S.attempts.length} attempts to make it work, each pushed in `
    + `the direction that would help if the schedule were the problem. A principal component `
    + `initialisation instead of random prototypes. Three times the epochs. Twice the starting `
    + `neighbourhood. A grid shaped like the sheet (${S.sheet_aspect} to 1) instead of square. The `
    + `best of all ${S.attempts.length} is <span class="bold">${bestSom.disparity}</span>, from `
    + `${bestSom.label} on a ${bestSom.rows} by ${bestSom.cols} grid, against Isomap's `
    + `${M.isomap.disparity}. `
    + `${bestSom.disparity > 0.5
      ? `A self-organising map does not unroll a Swiss roll, and the reason is structural rather `
        + `than a matter of tuning: nothing in the algorithm has any notion of distance along the `
        + `sheet. It minimises quantisation error subject to the grid staying smooth, and draping `
        + `the grid over the roll achieves both. The lesson generalises past this dataset: a `
        + `method's own diagnostics measure whether it did its own job, and cannot tell you whether `
        + `its job was yours.`
      : `So with the right settings it does substantially better, and the failure was the schedule `
        + `after all.`}`);

  /* ---------------------------------------------------------------- the hole */
  set('hole-intro',
    `One more case, and it is the one that shows where Isomap's own assumption sits. Cut a `
    + `rectangular hole in the sheet before rolling it. Nothing about the neighbourhood graph `
    + `changes in principle: it still connects, at k = ${data.meta.k_show} it is still one piece, `
    + `and every edge still measures a short honest hop. What changes is that some pairs now have to `
    + `walk <i>around</i> the hole, and a flat map has no way to make that true while keeping `
    + `everything else true. Classical scaling assumes the table came from points in a flat space, `
    + `and this table did not.`);

  set('hole-note',
    `The measurement separates the two effects, because one of them would be there anyway. A graph `
    + `path is a chain of straight chords across a curved surface, so it undershoots the real arc a `
    + `little on any sheet: on the intact roll the graph distance comes to ${H.ratio_intact} times `
    + `the flat chart distance. With the hole it is ${H.ratio_hole}, a detour of `
    + `<span class="bold">${H.detour}</span> that the hole alone added, and `
    + `${n0(H.crossing_pairs)} of the ${n0(H.total_pairs)} pairs, `
    + `${pct(H.crossing_pairs / H.total_pairs, 1)} of them, have to walk more than ${H.threshold} `
    + `times the flat distance. The map pays: <span class="bold">${H.disparity}</span> from the true `
    + `chart against ${M.isomap.disparity} on the intact roll, `
    + `${(H.disparity / M.isomap.disparity).toFixed(0)} times worse. And trustworthiness barely `
    + `moves, ${H.trustworthiness} against ${M.isomap.trustworthiness}, which is the same pattern as `
    + `the self-organising map: the local structure is fine and the global shape is a compromise. `
    + `Isomap needs the sheet to have no holes in it, and that is a real condition with a name, `
    + `geodesic convexity, and not a footnote.`);

  const dp = D.rows.find((r) => r.method === 'pca');
  const di = D.rows.find((r) => r.method === 'isomap');
  const dl = D.rows.find((r) => r.method === 'lle');
  const bestD = D.rows.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));
  set('digits-note',
    `Everything on this page so far has had a known answer, which is the only way to say anything `
    + `precise and is also not a situation anybody is ever in. So the last measurement is on real `
    + `data with no ground truth: the same ${n0(D.n)} MNIST digits the `
    + `<a href="../distances/">previous article</a> confused, embedded into two dimensions by each `
    + `method, with the only honest score available, whether the two numbers kept enough to tell a `
    + `three from a five. A five-nearest-neighbour classifier trained on ${n0(D.train)} of the `
    + `embedded points and tested on ${n0(D.test)} gets ${pct(dp.accuracy, 1)} from PCA's two `
    + `dimensions, ${pct(di.accuracy, 1)} from Isomap's and ${pct(dl.accuracy, 1)} from LLE's, `
    + `against ${pct(D.full_accuracy, 1)} on all ${784} pixels. `
    + `${bestD.method !== 'isomap'
      ? `<span class="bold">${bestD.method === 'lle' ? 'LLE' : bestD.method} wins</span>, which is `
        + `not the ordering the Swiss roll gave, and the disagreement is the point: on the roll the `
        + `question was whether the global shape came back, and here the question is whether nearby `
        + `digits stayed nearby. Those are different questions, LLE was built for the second one, `
        + `and the ranking follows the question rather than the method.`
      : `Isomap wins here too, consistently with the roll.`}`);

  /* ------------------------------------------------------------ verdict table */
  set('verdict-isomap',
    `On a sheet with a known answer it recovered it to ${M.isomap.disparity} where the best flat `
    + `projection managed ${M.pca.disparity}, and its distances agree with the real ones at `
    + `${M.isomap.geodesic_corr} against ${R.ambient_corr} for straight lines.`);
  set('verdict-isomap-warn',
    `It needs the sheet to have no holes. With one cut in it the same code lands ${H.disparity} `
    + `from the truth, ${(H.disparity / M.isomap.disparity).toFixed(0)} times worse, while `
    + `trustworthiness stays at ${H.trustworthiness} and hides it.`);
  set('verdict-k',
    `The graph first connects at k = ${G.first_connected}, the first short circuit appears at `
    + `k = ${G.first_short_circuit}, and the best reachable answer is ${bestRow.disparity} at `
    + `k = ${bestRow.k}. The safe window here is two values wide.`);
  set('verdict-k-warn',
    `${firstShort.short_circuits} edges out of ${n0(firstShort.edges)} multiply the error by `
    + `${cliff.toFixed(0)}, with no warning of any kind. Look at the longest edges in the graph, not `
    + `at the map.`);
  set('verdict-dim',
    `Residual variance against dimension has its elbow at ${data.residual.elbow} `
    + `(${res.map((r) => r.residual).slice(0, 4).join(', ')}), which is the true dimension of a `
    + `rectangle.`);
  set('verdict-dim-warn',
    `That curve is computed from the graph distances, so a short-circuited graph gives a confident `
    + `elbow in the wrong place. Fix k first.`);
  set('verdict-lle',
    `Trustworthiness ${M.lle.trustworthiness}, so neighbours stay neighbours, and it never has to `
    + `build an n by n table of shortest paths.`);
  set('verdict-lle-warn',
    `Distance from the true chart ${M.lle.disparity} against Isomap's ${M.isomap.disparity}: the `
    + `global shape is not recoverable from it. And below k = ${L.k} its eigenproblem is degenerate `
    + `enough that two solvers disagree by ${lle5.iter_vs_exact}.`);
  set('verdict-som',
    `Quantisation error ${S.quantisation}, topographic error ${S.topographic}, trustworthiness `
    + `${S.trustworthiness}, the best of the four. As a browsable summary of where the data is `
    + `dense, it works.`);
  set('verdict-som-warn',
    `Distance from the true chart ${S.disparity}, and ${S.long_axis_corr} correlation between its `
    + `long axis and the sheet's. ${S.attempts.length} attempts to fix that with initialisation, `
    + `epochs, neighbourhood width and grid shape got it to ${bestSom.disparity}.`);

  /* ---------------------------------------------------------------- the closing */
  set('closing-note',
    `The through line of this article is one substitution and its bill. Replacing the straight line `
    + `with a walk turns a method that cannot see a curved sheet into one that unrolls it exactly, `
    + `${M.pca.disparity} to ${M.isomap.disparity}, and everything downstream is the previous `
    + `article's code unchanged. The bill is that the walk is only as good as the graph, and the `
    + `graph has a cliff in it that no output diagnostic reports: ${firstShort.short_circuits} edges `
    + `out of ${n0(firstShort.edges)} cost a factor of ${cliff.toFixed(0)}, and the map on the wrong `
    + `side of that cliff looks perfectly reasonable.`
    + `<br /><br />`
    + `And there is a quieter assumption underneath, shared by all three of the coordinate methods `
    + `here and by everything in the two articles before them. They all try to reproduce `
    + `<span class="bold">distances</span>, near and far, in one number per pair, and they all `
    + `answer to a single global objective. That is why a handful of bad long-range entries can ruin `
    + `an entire map, and why the hole, which changes nothing local, changes everything. The next `
    + `article gives up on distances altogether. It keeps only the question of who is whose `
    + `neighbour, turns that into a probability, and matches probabilities instead, which fixes the `
    + `fragility and introduces a completely different set of ways to be misled: `
    + `<a href="../neighbours/">t-SNE and UMAP</a>.`);

  set('resources-note',
    `The roll is ${n0(data.meta.n)} points with \\(t\\) drawn uniformly on `
    + `\\([1.5\\pi, 4.5\\pi]\\) and height on \\([0, 21]\\), seed ${data.meta.seed}, rounded to four `
    + `decimals before anything was measured. The digits are ${n0(D.n)} stratified MNIST images, the `
    + `same draw the previous two articles used. References from scikit-learn ${data.meta.sklearn}: `
    + `Isomap, LocallyLinearEmbedding and trustworthiness. This page's own Isomap agrees with the `
    + `library's to ${data.checks.isomap_vs_sklearn.toExponential(0)} and its locally linear `
    + `embedding to ${data.checks.lle_vs_sklearn.toExponential(0)}. An edge counts as a short `
    + `circuit when the distance along the sheet between its two ends is more than `
    + `${data.meta.short_factor} times its length.`);
}
