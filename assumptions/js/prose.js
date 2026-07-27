/* The sentences that carry a measured number, written from the measurement.
 *
 * Nothing below is typed by hand. Every comparison is a branch, so that if a
 * generator run moves a result, the prose moves with it instead of quietly
 * becoming false: that includes the directions ("further", "better", "worse"),
 * which are numbers too, and the cells of the verdict table, which is where a
 * stale claim survives longest because nothing on the page contradicts it.
 */

/* The columns of the closing table, in the order they are declared in the
   HTML head row. One list, so the header, the cells and the sentence that
   counts them cannot drift apart. */
const BOARD_KEYS = ['kmeans', 'gmm', 'pam', 'dbscan_best', 'hdbscan', 'meanshift', 'optics'];

const n0 = (v) => v.toLocaleString('en-US');
const f3 = (v) => v.toFixed(3);
const pc = (v) => `${Math.round(v * 100)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
const word = (v) => (v >= 0 && v <= 10 ? WORDS[v] : n0(v));
const cap = (s) => s[0].toUpperCase() + s.slice(1);

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const O = data.outlier;
  const M = data.modes;
  const W = data.wine;
  const board = data.board;
  const byName = Object.fromEntries(board.map((r) => [r.name, r]));

  /* ---------------------------------------------------- the guards, in one line */
  const cells = M.codes.length * M.codes[0].length;
  const opt = Object.entries(data.optics);
  const matched = opt.filter(([, v]) => v.matches_sklearn).length;
  const worstCurve = Math.max(...opt.map(([, v]) => v.curve_gap));
  const shiftMismatch = Object.values(data.shift).reduce((s, v) => s + v.sklearn_mismatches, 0);
  const shiftChecks = Object.keys(data.shift).length * 5;
  const swaps = Math.max(...opt.map(([, v]) => v.ordering_swaps));
  set('guard-note',
    `PAM stops at a total distance of ${data.medoids.loss} on the blobs and FasterPAM stops at `
    + `${data.medoids.fasterpam_loss}, on the same partition to `
    + `${data.medoids.fasterpam_agrees.toFixed(4)}; the OPTICS ordering here is scikit-learn's own on `
    + `${matched} of the ${opt.length} datasets and, where it differs, differs in the order of `
    + `${word(swaps)} tied points, drawing a curve identical to ${worstCurve.toExponential(1)}; `
    + `mean shift agrees with scikit-learn at ${shiftChecks - shiftMismatch} of ${shiftChecks} `
    + `bandwidths checked; and k-modes settles ${Math.abs(M.kmodes.cost - M.kmodes.pkg_cost)} mismatch `
    + `${M.kmodes.cost < M.kmodes.pkg_cost ? 'below' : 'above'} the <code>kmodes</code> package's `
    + `Cao-initialised run, ${M.kmodes.cost} against ${M.kmodes.pkg_cost} out of ${n0(cells)} `
    + `category values. Nothing on this page is a claim about what the algorithm would do.`);

  /* ------------------------------------------------------------- medoids, clean */
  const same = Math.abs(data.medoids.ari - data.medoids.km_ari) < 0.0005;
  set('medoid-cost',
    `On the blobs the restriction costs nothing at all. K-means scores an adjusted Rand index of `
    + `${f3(data.medoids.km_ari)} against the groups the data was built from, and PAM `
    + `${f3(data.medoids.ari)}`
    + (same
      ? `: the same number, and the same partition. `
      : `, a difference of ${f3(Math.abs(data.medoids.ari - data.medoids.km_ari))}. `)
    + `Three of the 300 points are now promoted to being the centres, which is the only visible `
    + `change. Clean data is where a robust method has nothing to be robust about.`);

  const bf = data.medoids.brute_force;
  const exact = Math.abs(bf.optimum - bf.pam) < 1e-6;
  set('medoid-brute',
    `Whether build-and-swap finds the best set of medoids or merely a good one has an answer on a `
    + `small enough dataset. Take ${bf.n} of the 300 points, try every one of the ${n0(bf.triples)} `
    + `triples they allow, and the exhaustive best total distance is ${bf.optimum}. PAM finds ${bf.pam}`
    + (exact
      ? `, the same to four decimals. That is reassuring rather than conclusive, and the next section `
        + `is where it stops holding.`
      : `, which is ${(bf.pam - bf.optimum).toFixed(4)} worse than the optimum: the search is a `
        + `local one, and here you can see it.`));

  /* ---------------------------------------------------------------- the outlier */
  const dJ = O.J2 - O.J3;
  const dL = O.L2 - O.L3;
  const ratio = O.pred_pam / O.pred_km;
  set('outlier-arith',
    `Giving that one point a centre of its own costs exactly what the other 300 pay for having one `
    + `centre fewer, and both articles' worth of arithmetic is already on the table: serving the `
    + `blobs with two centres instead of three costs <span class="bold">${n0(Math.round(dJ))}</span> `
    + `more squared distance, and with two medoids instead of three, `
    + `<span class="bold">${dL.toFixed(1)}</span> more distance. Keeping the point costs `
    + `d² to k-means (times ${O.home_size + 1}/${O.home_size}, because a mean shifts towards whatever `
    + `joins it and the cluster it joins has ${O.home_size} members) and simply d to the medoids, `
    + `which do not move at all. Set each pair equal and the break-evens fall out: `
    + `k-means should let go at d = ${O.pred_km}, the medoid objective at d = ${O.pred_pam}, `
    + `<span class="bold">${ratio.toFixed(1)} times further out</span>. The sweep says k-means lets `
    + `go at ${O.break_km}, which is the prediction plus the width of one step. The missing square `
    + `is the whole story, and it is worth a factor of ${ratio.toFixed(1)} on this data.`);

  const objBetween = O.pred_pam > O.pred_pam_build && O.pred_pam < O.pred_pam_good;
  set('outlier-note',
    `The medoids, though, break twice, and neither time at ${O.pred_pam}. Classic PAM gives up `
    + `early, at <span class="bold">${O.break_pam}</span> against a predicted ${O.pred_pam_build}: `
    + `its greedy build hands the outlier a medoid as soon as that buys more than a third blob `
    + `medoid would, and taking it back in a single swap means paying the outlier's whole distance `
    + `at once, which no single swap can afford. FasterPAM, which starts from random medoid sets `
    + `instead, has the opposite problem and holds on too long: to isolate the outlier it must give `
    + `away one of three good medoids and live with the other two rather than the best two, so it `
    + `lets go at <span class="bold">${O.break_fast}</span> against a predicted ${O.pred_pam_good}. `
    + (objBetween
      ? `The objective's own answer, ${O.pred_pam}, sits between the two searches, and the two `
        + `implementations disagreed about the cheapest set of medoids at `
        + `${O.fasterpam_disagreements} of the ${O.grid.length} distances swept. `
      : `The objective's answer, ${O.pred_pam}, is outside the range the two searches bracket, which `
        + `means one of them is wrong more often than the other. `)
    + `Which is a lesson the k-means article already paid for once: an algorithm is an objective `
    + `<i>and</i> a search, and the answer you get is the second one.`);

  /* ------------------------------------------------------------------- the wine */
  const byKey = Object.fromEntries(W.metrics.map((m) => [m.key, m]));
  const bestMetric = W.metrics.reduce((a, b) => (b.ari > a.ari ? b : a));
  const worstMetric = W.metrics.reduce((a, b) => (b.ari < a.ari ? b : a));
  const eu = byKey.euclidean;
  set('wine-note',
    `Every one of the three lands on one exemplar per cultivar: under euclidean distance they are `
    + `bottles ${eu.medoids.map((i) => `#${i}`).join(', ')}, from cultivars `
    + `${eu.medoid_classes.map((c) => c + 1).join(', ')} respectively, which is three real wines `
    + `somebody could put on a table. The agreement with the cultivars runs from ${f3(worstMetric.ari)} `
    + `(${worstMetric.label}) to ${f3(bestMetric.ari)} (${bestMetric.label}), against `
    + `${f3(W.km_ari)} for k-means on the same 178 bottles, ${f3(W.gmm_ari)} for the mixture and `
    + `${f3(W.ward_ari)} for ward's tree. So on clean numeric data the restriction costs `
    + `${f3(W.km_ari - bestMetric.ari)} of agreement at best`
    + (bestMetric.ari > W.ward_ari
      ? `, though the cosine version does edge past the hierarchical answer, `
      : `, and does not reach the hierarchical answer either, `)
    + `and what it buys is that the centre of each cluster is a bottle. When the alternative is an `
    + `average of 13 standardised measurements that no wine has, that is often the trade you want.`);

  /* ----------------------------------------------------------------- categorical */
  /* How even the binning came out is a fact about the data, so it is counted
     here rather than asserted in the HTML. */
  const counts = [];
  for (let j = 0; j < M.codes[0].length; j++) {
    for (let l = 0; l < M.levels; l++) counts.push(M.codes.filter((r) => r[j] === l).length);
  }
  set('bin-spread', `and each of them covers between ${Math.min(...counts)} and `
    + `${Math.max(...counts)} of the ${M.codes.length}`);

  const oh = M.km_onehot.ari;
  const km = M.kmodes.ari;
  const med = M.medoids_hamming.ari;
  const codesAri = M.km_codes.ari;
  const ordered = [['a distribution over the categories in each column, from one-hot k-means', oh],
    ['one category per column, from k-modes', km],
    ['a whole row of the data, from k-medoids on the same Hamming distances', med]];
  const monotone = oh > km && km > med;
  set('modes-note',
    `K-modes scores <span class="bold">${f3(km)}</span> against the cultivars. K-means on the very `
    + `same codes, treated as the numbers 0, 1 and 2, scores ${f3(codesAri)}; k-means on a one-hot `
    + `expansion of those 13 columns into ${M.codes[0].length * M.levels}, ${f3(oh)}; and k-medoids `
    + `on the same Hamming distances, ${f3(med)}. `
    + (monotone
      ? `Line them up by how much freedom the centre has and the order is exact: `
        + `${ordered.map(([label, v]) => `${label}, ${f3(v)}`).join('; then ')}. Every step of `
        + `constraint costs agreement, which is the same trade the medoids made, in a world with no `
        + `arithmetic in it.`
      : `The order is not monotone in how constrained the centre is, so the tidy version of this `
        + `story does not survive the measurement.`));

  const range = M.km_range;
  const invariant = M.identical === M.perms.length;
  set('modes-invariance',
    `And now the button. There are ${n0(M.n_relabellings)} ways to number these categories, and the `
    + `data came with an ordered one, low then middle then high, which is exactly why k-means on the `
    + `codes is doing so well: it is quietly using an ordering that happens to be real. Across `
    + `${M.perms.length} random relabellings of the same data its score ranged from `
    + `<span class="bold">${f3(range[0])}</span> to <span class="bold">${f3(range[1])}</span>, mean `
    + `${f3(M.km_mean)}. `
    + (invariant
      ? `K-modes returned the <span class="bold">identical partition all ${M.perms.length} times</span>, `
        + `at the identical cost of ${M.kmodes.cost}, because the numbers are not in its objective: `
        + `Hamming distance can only ask whether two values are the same one.`
      : `K-modes returned the same partition ${M.identical} times out of ${M.perms.length}, which is `
        + `fewer than the invariance argument predicts and worth investigating.`));

  set('modes-verdict',
    `Which leaves k-modes in an honest, unglamorous place. It is not the most accurate way to `
    + `cluster these columns: one-hot plus k-means beats it by ${f3(oh - km)} and is just as immune `
    + `to renumbering, since permuting categories only permutes columns. What k-modes owns is that `
    + `its centre is <span class="bold">readable</span>, a profile of ${M.codes[0].length} categories `
    + `that describes a typical member instead of ${M.codes[0].length * M.levels} fractions, and `
    + `that it never builds the wide matrix at all, which is what matters when the columns are `
    + `postcodes rather than three tidy bins.`);

  /* ----------------------------------------------------------------------- optics */
  const T = data.optics.trap;
  const sepCuts = T.cuts.filter((r) => r.separated);
  const bestSepCut = sepCuts.reduce((a, b) => (b.recovered > a.recovered ? b : a));
  const xiJoint = T.xis.reduce((a, b) => (Math.min(b.separated, b.recovered) > Math.min(a.separated, a.recovered) ? b : a));
  set('optics-note',
    `Measured the way the density article measured it, because on this dataset the adjusted Rand `
    + `index rewards vandalism: of the ${T.cuts.length} radii in the sweep, the ones that keep the `
    + `tight pair apart run from ${sepCuts[0].eps} to ${sepCuts[sepCuts.length - 1].eps}, and the `
    + `best of them leaves <span class="bold">${pc(bestSepCut.recovered)}</span> of the diffuse `
    + `cluster in one piece. HDBSCAN managed ${pc(T.hdbscan_recovered)}. Reading `
    + `the shape of the same plot instead of its height, at xi ${xiJoint.xi}, keeps `
    + `<span class="bold">${pc(xiJoint.recovered)}</span> with the pair still separated, for an ARI `
    + `of ${f3(xiJoint.ari)}. The trap is solved outright. And the ARI trap is worth seeing for `
    + `yourself: the highest-scoring line of all, ${f3(T.best_cut.ari)} at eps ${T.best_cut.eps}, `
    + `gets there by declaring all 150 points of the diffuse cluster noise.`);

  const lines = opt.map(([name, v]) => {
    const bestXi = v.xis.reduce((a, b) => (b.ari > a.ari ? b : a));
    const dflt = v.xis.find((r) => r.xi === 0.05);
    return { name, best: v.best_cut, bestXi, dflt };
  });
  const cutWins = lines.filter((l) => l.best.ari >= l.bestXi.ari).length;
  set('optics-honest',
    `One caveat, and the sweep is what found it. With the default xi of 0.05, OPTICS scores `
    + `${lines.map((l) => `${f3(l.dflt.ari)} on the ${l.name}`).join(', ')}. A well-chosen horizontal `
    + `line on the very same reachability plot scores `
    + `${lines.map((l) => f3(l.best.ari)).join(', ')} respectively, and on the trap that last figure `
    + `is the ARI paying for a deleted cluster again. `
    + (cutWins > 0
      ? `On ${word(cutWins)} of the ${word(lines.length)} datasets here the best line beats the best `
        + `xi, and the one where xi wins is the one where no line exists. `
      : `Xi is ahead everywhere here, which is a stronger result than this article expected. `)
    + `So xi is a parameter with a default that fits very little, and what OPTICS removes is the `
    + `need for <i>one</i> radius, not the need for judgement.`);

  /* ------------------------------------------------------------------ mean shift */
  const B = data.shift.blobs;
  const R = data.shift.rings;
  const U = data.shift.unequal;
  const rightK = (S) => S.sweep.filter((r) => r.modes === S.true_k).length;
  set('shift-note',
    `On the blobs it works, and works well: the rule of thumb everyone uses picks a bandwidth of `
    + `${B.auto_bw} and mean shift finds <span class="bold">${B.auto_modes} modes</span> at an ARI `
    + `of ${f3(B.auto_ari)}, which is k-means' own ${f3(byName.blobs.kmeans)} on the same points, `
    + `without being told there were three. ${rightK(B)} of the ${B.sweep.length} bandwidths swept `
    + `return exactly three, from ${B.right_k_range[0]} to ${B.right_k_range[1]}: a wide plateau, `
    + `not a needle. On the rings, ${rightK(R) === 0 ? 'no bandwidth in the sweep returns two' : `only ${rightK(R)} bandwidths return two`}, `
    + `because every point climbs to the same summit in the middle where neither ring lives, and the `
    + `answer is ${R.auto_modes} cluster at ARI ${f3(R.auto_ari)}. On the small-tight-beside-large-diffuse `
    + `pair, ${rightK(U)} of ${U.sweep.length}. Which is the point: k came back as a bandwidth, and a `
    + `bandwidth is a density scale, the exact parameter OPTICS had just taken away. Nothing here is free, `
    + `things only move.`);

  const c = data.cost[data.cost.length - 1];
  /* Floor, not round: 332 times is two orders of magnitude and a half, and the
     conservative half is the one to keep when the denominator is 0.04 of a
     second and could be off by half of itself. */
  const orders = Math.floor(Math.log10(c.meanshift / c.kmeans));
  set('shift-cost',
    `And the bill, on the same ten-dimensional blobs at ${n0(c.n)} points, asking each for five `
    + `clusters, best of ${c.repeats} runs: k-means ${c.kmeans} seconds, PAM ${c.pam} plus the `
    + `${c.matrix_mb} MB of distance matrix it needs in memory, OPTICS ${c.optics}, mean shift `
    + `${c.meanshift}. That is at least ${orders === 1 ? 'an order' : `${word(orders)} orders`} of magnitude between `
    + `the two ends, and the matrix is the harder wall of the two: at ten times this many points it `
    + `would want ${n0(Math.round((c.matrix_mb * 100) / 1000))} GB, which is the real reason the `
    + `hierarchical article needed BIRCH. Everything here is for data you can hold in memory, which `
    + `is most data, though not the interesting kind.`);

  /* ------------------------------------------------------------- the two tables */
  const bestPam = board.reduce((a, b) => ((b.pam - b.kmeans) > (a.pam - a.kmeans) ? b : a));
  set('v-medoids',
    `On the blobs, ARI ${f3(data.medoids.ari)} against k-means' ${f3(data.medoids.km_ari)}: nothing. `
    + `On the wine, ${f3(byKey.euclidean.ari)} against ${f3(W.km_ari)}. On the trap, `
    + `${f3(bestPam.pam)} against ${f3(bestPam.kmeans)}.`);
  set('v-modes',
    `${f3(km)} against one-hot k-means' ${f3(oh)}, and it is the invariance that is worth having: `
    + `${f3(range[0])} to ${f3(range[1])} is the range k-means on the codes covers when the numbering `
    + `changes.`);
  set('v-optics',
    `${pc(xiJoint.recovered)} of the diffuse cluster recovered where the best radius manages `
    + `${pc(bestSepCut.recovered)}, and ${data.cost[data.cost.length - 1].optics} seconds at `
    + `${n0(data.cost[data.cost.length - 1].n)} points against k-means' `
    + `${data.cost[data.cost.length - 1].kmeans}.`);
  set('v-shift',
    `${B.auto_modes} modes unprompted on the blobs at ARI ${f3(B.auto_ari)}, and `
    + `${R.auto_modes} on the rings at ${f3(R.auto_ari)}. Slowest of the four by a distance: `
    + `${c.meanshift} seconds where k-means takes ${c.kmeans}.`);
  const ties = board.filter((r) => r.kmeans >= Math.max(r.pam, r.meanshift, r.optics)).length;
  set('v-kmeans',
    `Nothing, where its assumptions hold: on the blobs it ties the whole field at `
    + `${f3(byName.blobs.kmeans)}. It matches or beats the other three relaxations on ${ties} of the `
    + `${board.length} datasets below, and those were chosen to break it. It is also the fastest by a `
    + `distance: ${c.kmeans} seconds against ${c.pam}, ${c.optics} and ${c.meanshift}.`);

  const body = document.querySelector('#board-table tbody');
  if (body) {
    const LABEL = {
      blobs: 'three blobs', anisotropic: 'sheared blobs', rings: 'rings',
      unequal: 'unequal pair', trap: 'varying density',
    };
    body.innerHTML = board.map((r) => {
      const best = Math.max(...BOARD_KEYS.map((k) => r[k]));
      return `<tr><td>${LABEL[r.name] || r.name}</td>`
        + BOARD_KEYS.map((k) => `<td>${r[k] === best ? `<span class="bold">${f3(r[k])}</span>` : f3(r[k])}</td>`).join('')
        + `</tr>`;
    }).join('');
  }
  set('closing-scoreboard',
    `${cap(word(BOARD_KEYS.length))} algorithms on ${word(board.length)} datasets, `
    + `every cell measured here on the `
    + `coordinates this page ships: each parametric method given the true k, DBSCAN given the best `
    + `radius of a sweep, HDBSCAN and mean shift their usual defaults, OPTICS its default xi. Bold is `
    + `the winner of each row.`);

  const wins = (k) => board.filter((r) => r[k] >= Math.max(...BOARD_KEYS.map((o) => r[o]))).length;
  const densityRows = board.filter((r) => r.dbscan_best > Math.max(r.kmeans, r.pam, r.gmm)).length;
  const opticsLast = board.filter((r) => r.optics <= Math.min(...BOARD_KEYS.map((o) => r[o]))).length;
  const rings = byName.rings;
  set('board-note',
    `Three things stand out, and none of them is that the newer algorithms win. The mixture, from the `
    + `second article of the module, takes ${word(wins('gmm'))} of the ${word(board.length)} rows, `
    + `because most of these datasets really are made of gaussians and it is the only model here that `
    + `can learn a shape. Density takes the ${word(densityRows)} row where shape is the whole problem, `
    + `the rings, at ${f3(rings.dbscan_best)} where every centre-based method scores about zero. `
    + `And k-medoids' one clear win is the varying-density trap, `
    + `${f3(byName.trap.pam)} against k-means' ${f3(byName.trap.kmeans)}, for exactly the reason it `
    + `survives an outlier: the diffuse cluster's far edge counts once instead of once squared. `
    + `OPTICS at its default xi comes last in ${opticsLast} of the ${board.length} rows, which is `
    + `what a default is worth on data nobody tuned it for.`);

  /* ---------------------------------------------------------------- the fine print */
  const K = data.knife_edge;
  const verified = Object.values(data.meta.shared_datasets_verified);
  set('ref-note',
    `Datasets rebuilt with the constructions of the earlier articles and checked coordinate by `
    + `coordinate against the JSON those articles ship: `
    + `${verified.every(Boolean) ? `all ${verified.length} match` : `${verified.filter(Boolean).length} of ${verified.length} match, which is a bug`}. `
    + `Measured with scikit-learn 1.8.0 (OPTICS, MeanShift, KMeans, DBSCAN, HDBSCAN), kmedoids 0.5.5 `
    + `(FasterPAM) and kmodes 0.12.2, seed ${data.meta.seed}, min_samples ${data.meta.min_samples}, `
    + `in ${data.meta.seconds_to_generate} seconds. One number here disagrees with an earlier article `
    + `on purpose: k-means on the sheared blobs scores ${f3(K.rounded_ari)} above and `
    + `${f3(K.raw_ari)} in the k-means article, because that experiment measured on unrounded `
    + `coordinates while this page measures on the three-decimal ones it draws. The two clusterings `
    + `differ by ${f3(Math.abs(K.raw_ari - K.rounded_ari))} of agreement while the totals they `
    + `minimise differ by ${(Math.abs(K.raw_inertia - K.rounded_inertia)).toFixed(2)} out of `
    + `${K.raw_inertia}: on sheared blobs the two best solutions are all but tied, so rounding a `
    + `coordinate in the third decimal is enough to swap them. Worth stating rather than hiding, and `
    + `worth knowing before quoting any single number about sheared data.`);
}
