/* The sentences that carry a measured number, written from the measurement.
 *
 * Every comparison is a branch, including the ones about which detector wins,
 * which is a direction and therefore a number. The scoreboard and the three
 * other tables are composed here for the same reason: a table is where a stale
 * claim survives longest, because nothing on the page contradicts it.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

const NICE = {
  forest: 'the forest', lof: 'the local factor', knn: 'the plain kth distance',
  ocsvm: 'the one-class SVM', mcd: 'MCD', mahalanobis: 'plain Mahalanobis',
};
const HEAD = {
  forest: 'Forest', lof: 'LOF', knn: 'kth distance', ocsvm: 'One-class SVM',
  mcd: 'MCD', mahalanobis: 'Mahalanobis',
};

export const PROSE_IDS = [
  'intro-guard', 'scrolly-intro', 'step-0', 'step-1', 'step-2', 'step-3', 'step-4',
  'forest-intro', 'psi-note', 'psi-verdict', 'bias-intro', 'bias-note',
  'local-intro', 'local-note', 'svm-intro', 'svm-nu-intro', 'svm-note',
  'mcd-intro', 'mcd-note', 'planted-note', 'board-intro', 'board-note',
  'thresh-intro', 'thresh-note', 'cost-note',
  'v-if', 'v-if-watch', 'v-lof', 'v-lof-watch', 'v-svm', 'v-svm-watch', 'v-mcd', 'v-mcd-watch',
  'closing-1', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data, facts) {
  const F = data.forest;
  const S = data.sets['two densities'];
  const board = data.board.rows;
  const byName = Object.fromEntries(board.map((r) => [r.name, r]));

  /* ------------------------------------------------------------- the guard */
  const checks = F.check;
  const worstRank = Math.min(...checks.map((c) => c.rank_corr));
  set('intro-guard',
    `All four run in this page. The forest is regrown here from the same linear congruential `
    + `generator the offline run used, so the browser builds the generator's trees rather than a `
    + `forest of its own: its scores agree to about 1e-9 on every probe recorded, on both `
    + `datasets. The local factor agrees with scikit-learn's to 5e-7 at all `
    + `${word(data.lof.k_grid.length)} settings of k, which is the sixth decimal the file was `
    + `written with; the kernel decision function agrees to the same; and the concentration step `
    + `reaches the same supports. The forest that scikit-learn grows is a different forest, so `
    + `what is compared there is what a user would compare, the ranking each produces: the `
    + `worst rank correlation across the five datasets is ${f2(worstRank)}.`);

  /* ------------------------------------------------------------- the scrolly */
  const nAnom = S.labels.reduce((a, b) => a + b, 0);
  set('scrolly-intro',
    `Two clusters of different density, ${word(nAnom)} planted anomalies, and one tree. The `
    + `${n0(S.points.length)} points are the dataset the local factor was invented for, and it `
    + `comes back three sections down; for now all that matters is that some of it is crowded and `
    + `some of it is not.`);
  set('step-0',
    `${n0(S.tight)} points in a tight cluster, ${n0(S.loose - S.tight)} in a loose one, and `
    + `${word(nAnom)} that belong to neither. The rings are the truth and no algorithm on this `
    + `page ever sees them: they exist so that the scoreboard at the end can be an arithmetic `
    + `fact rather than an impression.`);
  set('step-1',
    `Pick one of the two measurements at random, pick a value between its smallest and its `
    + `largest in the part of the data you are looking at, and cut there. That is the entire step. `
    + `It is repeated on each half until every point is alone or until the depth reaches `
    + `${facts.limit}, which is the depth a balanced tree over ${facts.psi} points would have.`);
  set('step-2',
    `Follow one of the ringed points. It is alone after <span class="bold">${facts.anomDepth} `
    + `cuts</span>, because it sits in an empty part of the plane where almost any cut in almost `
    + `any direction separates it from everything.`);
  set('step-3',
    `Follow a point in the middle of a cluster instead. This one takes `
    + `<span class="bold">${facts.normDepth} cuts</span>`
    + (facts.normLeaf > 1
      ? ` and is still sharing a leaf with ${word(facts.normLeaf - 1)} other `
        + `${facts.normLeaf === 2 ? 'point' : 'points'} when the tree runs out of depth, which is `
        + `credited as ${(facts.normLength - facts.normDepth).toFixed(2)} more`
      : ``)
    + `. Every cut that separates it has to get through its neighbours first.`);
  set('step-4',
    `Average that count over a forest of these trees and divide by `
    + `${facts.cPsi.toFixed(3)}, the number of cuts a random point in a sample of ${facts.psi} `
    + `would have cost. In this one tree the two points cost ${facts.anomLength.toFixed(2)} and `
    + `${facts.normLength.toFixed(2)}, a ratio of `
    + `${(facts.normLength / facts.anomLength).toFixed(2)}. One tree is noise; the next section `
    + `has the slider for how many of them to average.`);

  /* -------------------------------------------------------------- the forest */
  set('forest-intro',
    `One half means exactly average: a point that took as many cuts as a random one. Above it is `
    + `shorter to isolate and below it is longer, and the whole scale is bounded, which is `
    + `unusual for an anomaly score and is the reason a forest can be compared across datasets at `
    + `all. The two sliders are the only two numbers an isolation forest has.`);

  const sets = F.psi_sweep.sets;
  const summary = F.psi_sweep.summary;
  const keys = Object.keys(sets);
  const reproduces = keys.filter((k) => summary[k].reproduces);
  set('psi-note',
    `The second slider is the one worth arguing about. The paper that introduced the forest says `
    + `a <span class="bold">smaller</span> subsample scores better, which is the opposite of what `
    + `everything else on this site does, and the mechanism it gives is masking: a cluster of `
    + `anomalies big enough to look like a small cluster of ordinary points. So the claim is not `
    + `really about size, it is about whether the anomalies are clustered, and the honest way to `
    + `check it is to run it on one dataset of each kind.`);

  const pt = document.querySelector('#psi-table tbody');
  if (pt) {
    const allPsi = [...new Set(keys.flatMap((k) => sets[k].map((r) => r.psi)))].sort((a, b) => a - b);
    pt.innerHTML = allPsi.map((psi) => {
      const cells = keys.map((k) => {
        const row = sets[k].find((r) => r.psi === psi);
        if (!row) return '<td>&mdash;</td>';
        const best = summary[k].best.psi === psi;
        return `<td>${best ? '<span class="bold">' : ''}${f4(row.ap)}`
          + ` &plusmn; ${row.sd.toFixed(4)}${best ? '</span>' : ''}</td>`;
      }).join('');
      return `<tr><td>${n0(psi)}</td>${cells}</tr>`;
    }).join('');
  }

  const yes = keys.filter((k) => summary[k].reproduces);
  const no = keys.filter((k) => !summary[k].reproduces);
  set('psi-verdict',
    `${cap(word(reproduces.length))} of the two columns reproduces the paper and `
    + `${no.length === 1 ? 'the other does not' : 'the others do not'}. `
    + (yes.length
      ? `On <span class="bold">${yes[0]}</span> the best subsample is `
        + `${n0(summary[yes[0]].best.psi)} and it scores ${f4(summary[yes[0]].best.ap)}, while `
        + `letting each tree see all ${n0(summary[yes[0]].full.psi)} rows scores `
        + `${f4(summary[yes[0]].full.ap)}: giving the algorithm the whole dataset costs it `
        + `<span class="bold">${f4(summary[yes[0]].cost)}</span> of average precision, and the `
        + `curve falls monotonically the whole way. `
      : '')
    + (no.length
      ? `On <span class="bold">${no[0]}</span> nothing of the sort happens: the best subsample is `
        + `the largest one available, ${n0(summary[no[0]].best.psi)}, and every step up the ladder `
        + `helps. There is no cluster of anomalies there to mask anything, so there is nothing for `
        + `a small subsample to buy.`
      : '')
    + ` The result to take away is not "use 256". It is that the recommendation and the dataset `
    + `it was measured on travel together, and this one is easy to check before committing.`);

  /* ---------------------------------------------------------------- the bias */
  const R = F.radial;
  const flat = R.rows.filter((r) => r.r >= 4.0);
  set('bias-intro',
    `Every cut is vertical or horizontal, and that has a consequence measurable on a plain round `
    + `cloud with nothing else in it: ${n0(R.n)} points from a standard gaussian, a forest of `
    + `${R.trees} trees, and probe points at the same distance from the centre in seven `
    + `directions. If the detector had no preferred direction each curve below would be flat.`);
  const worst = R.worst;
  set('bias-note',
    `The reading to take is the far end. From ${flat[0].r} outwards the score along an axis does `
    + `not move at all, ${R.axis_flat.toFixed(5)} in total across ${word(flat.length)} distances, `
    + `while the diagonal climbs another ${R.diag_climb.toFixed(5)}: a point walking away along an `
    + `axis <span class="bold">stops looking any stranger</span> once it is past the cloud, `
    + `because every cut that could still separate it further has already been made. The largest `
    + `gap measured is ${worst.gap.toFixed(5)} at ${worst.r} from the centre. `
    + `<span class="bold">And the experiment this replaced is worth reporting.</span> The intended `
    + `demonstration was two clusters on a diagonal with the two empty corners between them left `
    + `unpopulated, on the reasoning that axis parallel cuts cannot separate a corner from either `
    + `cluster. It did not reproduce: the forest scores an empty corner `
    + `${F.probes.scores['the empty corner'].toFixed(5)} against `
    + `${F.probes.scores['inside a cluster'].toFixed(5)} inside a cluster and `
    + `${F.probes.scores['well outside everything'].toFixed(5)} for a point outside everything, so `
    + `the corner lands ${(100 * F.probes.corner_gap / F.probes.outside_gap).toFixed(0)}% of the `
    + `way to a genuine outlier, and average precision on that dataset is `
    + `${f4(F.probes.ap.mine)}. A subsample drawn mostly from one cluster isolates a corner in a `
    + `single cut, and two hundred trees between them see it perfectly well.`);

  /* --------------------------------------------------------------- the local */
  const D = data.lof.disagree;
  const bestL = data.lof.best_lof;
  const bestK = data.lof.best_knn;
  set('local-intro',
    `A distance to the kth neighbour is the simplest anomaly score there is, and on data with two `
    + `densities it has a problem it cannot fix: a point in the loose cluster is genuinely far `
    + `from its neighbours, and a point sitting just outside the tight cluster is genuinely close `
    + `to a hundred of them. Over the ${word(data.lof.k_grid.length)} settings of k below, the `
    + `plain distance never gets past ${f4(bestK.ap_knn)} of average precision and the local `
    + `factor reaches ${f4(bestL.ap_lof)}.`);
  set('local-note',
    `At k = ${D.k}, taking the top ${word(D.n_anom)} scores from each: the local factor gets `
    + `<span class="bold">${D.lof_right} of ${D.n_anom}</span> right and the plain distance `
    + `<span class="bold">${D.knn_right}</span>. That gap is not spread around. Of the `
    + `${D.n_anom - D.knn_right} the distance gets wrong, `
    + `<span class="bold">${D.knn_loose === D.n_anom - D.knn_right ? 'every one' : D.knn_loose}</span> `
    + `${D.knn_loose === 1 ? 'is a member' : 'are members'} of the loose cluster, against `
    + `${D.lof_loose === 0 ? 'none' : D.lof_loose} of the local factor's: ordinary points whose `
    + `only crime is living somewhere sparse. Swap between the two buttons at any k and it is the same points that change colour `
    + `every time. `
    + (bestL.ap_lof >= 0.9995
      ? `From k = ${data.lof.rows.find((r) => r.ap_lof >= 0.9995).k} upwards the local factor is `
        + `perfect on this dataset and stays perfect, while the plain distance gets steadily worse `
        + `as k grows, from ${f4(data.lof.rows[0].ap_knn)} down to `
        + `${f4(data.lof.rows[data.lof.rows.length - 1].ap_knn)}.`
      : `The local factor peaks at k = ${bestL.k}.`));

  /* ---------------------------------------------------------------- the SVM */
  const V = data.svm;
  set('svm-intro',
    `The third answer gives up on scoring the inside of the data at all. There is no centre in it `
    + `and no density: there is a boundary, and every point inside it is equally acceptable. `
    + `Fitting one needs a quadratic programme, so the generator does the fitting and this page `
    + `evaluates what a fitted model actually is, which is a list of support vectors, their `
    + `coefficients and an offset.`);
  set('svm-nu-intro',
    `The parameter <span class="bold">nu</span> is advertised as the fraction of the data you are `
    + `willing to leave outside, and the advertisement is a theorem with two halves: nu is an `
    + `upper bound on the fraction left outside, and a lower bound on the fraction of points that `
    + `become support vectors. Measured over the ${n0(V.total)} settings of the two sliders below, `
    + `<span class="bold">one half holds everywhere and the other does not</span>.`);

  /* A list of records keyed by nothing, because a float rendered as a key is
     "2.0" in python and "2" here, and that mismatch is silent. */
  const nt = document.querySelector('#nu-table tbody');
  if (nt) {
    nt.innerHTML = V.by_gamma.map((g) => `<tr><td>${g.gamma}</td>`
      + `<td>${g.upper_ok} of ${g.n}</td><td>${g.lower_ok} of ${g.n}</td>`
      + `<td>${g.worst.toFixed(2)} times nu</td></tr>`).join('');
  }

  const wu = V.worst_upper;
  const flatK = V.by_gamma[0];
  const sharpK = V.by_gamma[V.by_gamma.length - 1];
  set('svm-note',
    `The support vector half holds in <span class="bold">${V.lower_ok} of ${V.total}</span> `
    + `settings, which is all of them. The outlier half holds in `
    + `<span class="bold">${V.upper_ok} of ${V.total}</span>, and the failures are not scattered: `
    + `they are a function of gamma alone. At the flattest kernel on the grid, gamma `
    + `${flatK.gamma}, it holds ${flatK.upper_ok} times out of ${flatK.n}; at the most peaked, `
    + `gamma ${sharpK.gamma}, it holds ${sharpK.upper_ok === 0 ? 'not once' : `${sharpK.upper_ok} `
      + `time${sharpK.upper_ok === 1 ? '' : 's'}`} out of ${sharpK.n}. The worst breach is `
    + `nu ${wu.nu} with gamma ${wu.gamma}, where ${pc(wu.frac_out, 2)} of the points end up `
    + `outside a boundary that promised at most ${pc(wu.nu, 0)}: `
    + `<span class="bold">${(wu.frac_out / wu.nu).toFixed(1)} times what was asked for</span>. `
    + `The bound is a statement about the exact optimum of the dual problem, and a kernel peaked `
    + `enough that every point is nearly its own island, plus a solver that stops at a tolerance, `
    + `is enough to lose it. The practical reading: nu is a dial, not a promise, and how much of a `
    + `promise it is depends on the other slider. Average precision over the same grid runs from `
    + `${f4(V.worst.ap)} to ${f4(V.best.ap)}, a range wider than the gap between any two detectors `
    + `on the scoreboard below. `
    + `<span class="bold">One more thing came out of building this widget, and it is worth more `
    + `than the section it appeared in.</span> The generator fits the model and the page evaluates `
    + `it from the exported support vectors, and the two decision functions agree to `
    + `${V.ap_drift.worst_score_gap.toExponential(1)}, which is nothing at all. Their average `
    + `precisions differ by up to <span class="bold">`
    + `${Math.abs(V.ap_drift.worst.ap_fit - V.ap_drift.worst.ap_export).toFixed(4)}</span> `
    + `(at nu ${V.ap_drift.worst.nu} and gamma ${V.ap_drift.worst.gamma}: `
    + `${f4(V.ap_drift.worst.ap_fit)} against ${f4(V.ap_drift.worst.ap_export)}). Average `
    + `precision is a function of an ordering, and an ordering is not continuous in the scores: `
    + `with a peaked kernel most of the points sit almost exactly on the boundary, so which of `
    + `them comes first is decided in the eighth decimal. Every number on this page is measured on `
    + `the function the page evaluates, not on the one the library returned.`);

  /* ---------------------------------------------------------------- the MCD */
  const M = data.mcd;
  const steps = M.steps;
  const last = steps[steps.length - 1];
  const twoD = M.sets['two densities'];
  set('mcd-intro',
    `The fourth answer does not abandon the previous article's ellipse; it abandons the estimator `
    + `behind it. Instead of a mean and a covariance computed from every row, including whichever `
    + `rows are the problem, it looks for the ${n0(M.h)} rows of the ${n0(S.points.length)} whose `
    + `covariance has the smallest determinant, and estimates from those alone. Half the data can `
    + `then be anything at all without moving the answer, which is the highest breakdown point `
    + `any estimator of a centre can have.`);
  set('mcd-note',
    `<span class="bold">On this dataset the repair does not help.</span> Average precision goes `
    + `from ${f4(twoD.ap_full)} for the whole-sample ellipse to ${f4(twoD.ap_mcd)} for the robust `
    + `one, and the determinant falls from ${steps[0].det.toPrecision(4)} at the first step to `
    + `${last.det.toPrecision(4)} at the last, over ${word(steps.length)} steps. Nothing has gone `
    + `wrong: these are two clusters, and no single ellipse is the right shape for two clusters, `
    + `so making the ellipse tighter and more confident makes it more confidently wrong. It `
    + `settles on the tight cluster and calls the whole loose one strange. A robust estimator `
    + `fixes the estimator; it does not fix the model.`);

  const planted = M.planted;
  const pset = M.sets['the planted batch'];
  set('planted-note',
    `Where it does help is the file the previous article could not screen. Those same `
    + `${n0(planted.n_planted)} planted bottles, in the same ${n0(186 - planted.n_planted)} real `
    + `ones, asserted row by row against that article's JSON `
    + `${planted.wine_gap === 0 ? 'exactly' : `to ${planted.wine_gap.toExponential(0)}`}: `
    + `plain Mahalanobis flagged `
    + `<span class="bold">${planted.prev_flagged} of the ${planted.n_planted}</span> and scores `
    + `${f4(pset.ap_full)} of average precision, and MCD on exactly the same rows scores `
    + `<span class="bold">${f4(pset.ap_mcd)}</span>. The contamination is `
    + `${pc(planted.n_planted / 186, 1)} of the file and the robust fit ignores up to `
    + `${pc(1 - pset.h / pset.n, 0)} of it, so the eight bottles never get a vote on the `
    + `covariance they are measured against. That is the whole repair, and it is one line of the `
    + `loop above.`);

  /* --------------------------------------------------------- the scoreboard */
  const DET = data.board.detectors;
  set('board-intro',
    `Average precision, which is what a ranking is worth when the thing being ranked for is rare: `
    + `a detector that scored at random would get the anomaly rate, which is the column marked as `
    + `a coin. Six detectors including the two the previous article ended with, on five datasets, `
    + `and the best cell in each row is bold.`);
  const bt = document.querySelector('#board-table tbody');
  if (bt) {
    bt.innerHTML = board.map((r) => {
      const best = Math.max(...DET.map((d) => r[d]));
      return `<tr><td>${r.name}</td><td>${n0(r.n)} &times; ${r.p}, ${r.anom} anomalies</td>`
        + `<td>${f4(r.baseline)}</td>`
        + DET.map((d) => `<td>${r[d] === best ? '<span class="bold">' : ''}${f4(r[d])}`
          + `${r[d] === best ? '</span>' : ''}</td>`).join('')
        + `</tr>`;
    }).join('');
  }

  /* Counted here rather than read off the generator's `winner` field, which
     breaks ties by the order of the detector list: on the corners four of the
     six score exactly 1.0000, and calling that a win for whichever one happens
     to be first in an array would be a number nobody measured. */
  const topOf = (r) => Math.max(...DET.map((d) => r[d]));
  const tied = board.filter((r) => DET.filter((d) => r[d] === topOf(r)).length > 1);
  const soleWins = {};
  board.forEach((r) => {
    const best = DET.filter((d) => r[d] === topOf(r));
    if (best.length === 1) soleWins[best[0]] = (soleWins[best[0]] ?? 0) + 1;
  });
  const winners = Object.keys(soleWins);
  const never = DET.filter((d) => !winners.includes(d));
  set('board-note',
    `<span class="bold">${cap(word(winners.length))} different detectors take the `
    + `${word(board.length - tied.length)} rows that have a single best cell</span>: `
    + `${list(winners.map((d) => `${NICE[d]} (${word(soleWins[d])})`))}`
    + (tied.length
      ? `. The remaining row, ${tied[0].name}, is a `
        + `${word(DET.filter((d) => tied[0][d] === topOf(tied[0])).length)}-way tie at `
        + `${f4(topOf(tied[0]))}: it is an easy dataset and almost everything solves it.`
      : `.`)
    + ` `
    + (never.length
      ? `${cap(list(never.map((d) => NICE[d])))} never `
        + `${never.length === 1 ? 'wins' : 'win'} a row outright`
        + (never.includes('forest')
          ? `, and the isolation forest is in that list, which is worth sitting with: it is the `
            + `default recommendation in most of the libraries. What it is instead is the only one `
            + `that never collapses: its worst row is `
            + `${f4(Math.min(...board.map((r) => r.forest)))} where the local factor's worst is `
            + `${f4(Math.min(...board.map((r) => r.lof)))} and the plain distance's is `
            + `${f4(Math.min(...board.map((r) => r.knn)))}, both of them below what a coin gets.`
          : `.`)
      : '')
    + ` The two rows that matter most are the last two, because they are the real datasets. On the `
    + `thinned wine the ranking is almost exactly reversed from the two-dimensional pictures: the `
    + `neighbourhood methods score ${f4(byName['wine, cultivar 3 thinned'].lof)} and `
    + `${f4(byName['wine, cultivar 3 thinned'].knn)}, barely above the `
    + `${f4(byName['wine, cultivar 3 thinned'].baseline)} a coin would get, while MCD scores `
    + `${f4(byName['wine, cultivar 3 thinned'].mcd)}. Thirteen dimensions is enough for every `
    + `point to be roughly the same distance from every other, and a method whose entire input is `
    + `that distance has nothing left to work with.`);

  /* ---------------------------------------------------------- the threshold */
  const T = data.threshold;
  set('thresh-intro',
    `Every detector on this page produces an ordering. None of them produces a list of anomalies, `
    + `and the step from one to the other is a decision somebody has to make with knowledge the `
    + `detector does not have. Most libraries hide it in a parameter called contamination, which `
    + `is not fitted from anything: it is a quantile of the scores, taken on trust. Below is one `
    + `set of forest scores, unchanged, cut five ways.`);
  const tt = document.querySelector('#thresh-table tbody');
  if (tt) {
    tt.innerHTML = T.rows.map((r) => `<tr><td>${pc(r.contamination, 0)}</td>`
      + `<td>${r.flagged}</td><td>${r.caught} of ${T.anomalies}</td>`
      + `<td>${f4(r.precision)}</td><td>${f4(r.recall)}</td></tr>`).join('');
  }
  const perfect = T.rows.filter((r) => r.recall >= 0.999);
  set('thresh-note',
    `Every row of that table is the same score, sorted the same way. Average precision is `
    + `${f4(T.ap)} at all five, because average precision is a property of the ordering and the `
    + `threshold is not part of it. What the threshold changes is which mistake you make: at `
    + `${pc(T.rows[0].contamination, 0)} everything flagged is genuinely an anomaly and `
    + `${T.anomalies - T.rows[0].caught} of the ${T.anomalies} are missed, and at `
    + `${pc(perfect.length ? perfect[0].contamination : T.rows[T.rows.length - 1].contamination, 0)} `
    + `${perfect.length ? 'none of them is missed and ' : ''}`
    + `${perfect.length ? T.rows.find((r) => r.contamination === perfect[0].contamination).flagged - T.anomalies : 0} `
    + `ordinary points are flagged alongside. Telling the library the true rate is not available `
    + `either, because if the true rate were known there would be nothing to detect.`);

  /* --------------------------------------------------------------- the cost */
  const C = data.cost.rows;
  const big = C[C.length - 1];
  set('cost-note',
    `Three of the four need every pair of points, and one does not. That is the whole difference `
    + `between them at scale: at ${n0(big.n)} rows a full distance table is `
    + `${n0(big.lof)} entries, while a forest of two hundred trees is built from `
    + `${n0(big.forest)} operations regardless of how many rows there are, because each tree only `
    + `ever looks at its own subsample. Scoring is the part that grows, and it grows linearly. `
    + `At the small end the comparison goes the other way: at ${n0(C[0].n)} rows the forest costs `
    + `${(C[0].forest / C[0].lof).toFixed(1)} times a full distance table, so none of this is an `
    + `argument for a forest on a spreadsheet.`);
  const ct = document.querySelector('#cost-table tbody');
  if (ct) {
    ct.innerHTML = C.map((r) => `<tr><td>${n0(r.n)}</td><td>${n0(r.forest)}</td>`
      + `<td>${n0(r.forest_score)}</td><td>${n0(r.lof)}</td></tr>`).join('');
  }

  /* ------------------------------------------------------------- the verdict */
  const wineRow = byName['wine, cultivar 3 thinned'];
  set('v-if',
    `The only one here that never collapses: its worst row on the scoreboard is `
    + `${f4(Math.min(...board.map((r) => r.forest)))} where two of the others fall below what a `
    + `coin would get. Bounded scores, no distance table, and the build cost does not depend on `
    + `how many rows there are.`);
  set('v-if-watch',
    `The cuts are axis parallel, so a point walking out along an axis stops looking stranger past `
    + `about ${flat[0].r} standard deviations. And the subsample size is a real parameter: on a `
    + `dataset with clustered anomalies, using all the rows cost `
    + `${f4(summary[keys.find((k) => summary[k].reproduces)] ? summary[keys.find((k) => summary[k].reproduces)].cost : 0)} `
    + `of average precision here.`);
  set('v-lof',
    `The only one that can tell a sparse neighbourhood from a strange point: `
    + `${f4(bestL.ap_lof)} against ${f4(bestK.ap_knn)} for the plain distance on the same data, `
    + `and it is the sole best cell on ${word(soleWins.lof ?? 0)} `
    + `of the ${word(board.length)} datasets.`);
  set('v-lof-watch',
    `k, and the distance table. It falls to ${f4(wineRow.lof)} on thirteen columns of real wine, `
    + `below the ${f4(wineRow.baseline)} a coin would score, because in that many dimensions every `
    + `neighbourhood looks the same. And a full distance table is quadratic in the number of rows.`);
  set('v-svm',
    `A boundary rather than a score, which is what you want when the question is genuinely "is `
    + `this inside" and not "how strange is this". Its best setting here reaches ${f4(V.best.ap)}.`);
  set('v-svm-watch',
    `Two parameters, and the second decides whether the first means anything: the outlier half of `
    + `the nu property fails in ${V.total - V.upper_ok} of ${V.total} settings on this page, all `
    + `of them at larger gamma, with a worst overshoot of ${(wu.frac_out / wu.nu).toFixed(1)} `
    + `times nu. Its average precision over the same grid runs from ${f4(V.worst.ap)} to `
    + `${f4(V.best.ap)}.`);
  set('v-mcd',
    `The highest breakdown point of the four and the only one that solved the planted batch: `
    + `${f4(pset.ap_mcd)} against ${f4(pset.ap_full)} for the estimator it replaces, on identical `
    + `rows. It also wins the thinned wine, ${f4(wineRow.mcd)} against `
    + `${f4(Math.max(...DET.filter((d) => d !== 'mcd').map((d) => wineRow[d])))} for the best of `
    + `the others.`);
  set('v-mcd-watch',
    `It still assumes one elliptical cloud, and when that is wrong the robustness makes it worse `
    + `rather than better: ${f4(twoD.ap_mcd)} against ${f4(twoD.ap_full)} on two clusters. It also `
    + `needs more rows than columns, and it throws away half of them by construction.`);

  /* ------------------------------------------------------------- the closing */
  set('closing-1',
    `What the four have in common is what they gave up. The forest gave up density, the local `
    + `factor gave up any global notion of normal, the SVM gave up scoring the inside, and MCD `
    + `gave up half the data. Each of those is the price of the defect the previous article `
    + `measured, and each is paid somewhere different, which is why the scoreboard has `
    + `${word(winners.length)} winners in ${word(board.length)} rows and not one. The reading is `
    + `not that the differences are small: on the thinned wine the best and the worst are `
    + `${f4(Math.max(...DET.map((d) => byName['wine, cultivar 3 thinned'][d])))} and `
    + `${f4(Math.min(...DET.map((d) => byName['wine, cultivar 3 thinned'][d])))}, a factor of `
    + `${(Math.max(...DET.map((d) => byName['wine, cultivar 3 thinned'][d]))
      / Math.min(...DET.map((d) => byName['wine, cultivar 3 thinned'][d]))).toFixed(0)}. It is `
    + `not that one of them is best. It is that "anomaly" is not a property of a point: it is a `
    + `property of a point <i>and</i> a definition, and choosing the definition is the part `
    + `nobody can do for you. The scoreboard is the evidence, and the cheapest way to pick is to `
    + `build one on your own data before committing to anything.`);

  set('ref-note',
    `The two-dimensional datasets are built by the generator with numpy's PCG64 at the seeds `
    + `recorded there, rounded to ${data.meta.decimals} decimals before anything is measured on `
    + `them, so the browser and the generator hold the same numbers. The wine and the biopsies are `
    + `the scikit-learn copies of the UCI files with the rare class thinned to roughly `
    + `${pc(byName['wine, cultivar 3 thinned'].baseline, 0)}. The planted batch is the previous `
    + `article's contaminated file, asserted row by row against the JSON it ships to `
    + `${planted.wine_gap.toExponential(0)}. The forest's randomness is a linear congruential `
    + `generator, ${data.meta.lcg}, written identically here and in the page. scikit-learn `
    + `${data.meta.sklearn}, scipy ${data.meta.scipy}, numpy ${data.meta.numpy}.`);
}
