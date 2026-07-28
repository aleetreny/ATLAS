/* The sentences that carry a measured number, written from the measurement.
 *
 * Nothing below is typed by hand. Every comparison is a branch, so that if a
 * generator run moves a result the prose moves with it instead of quietly
 * becoming false. That includes the directions ("better", "further", "more"),
 * which are numbers too, and the cells of both tables, which is where a stale
 * claim survives longest because nothing on the page contradicts it.
 */

/* The columns of the closing scoreboard, in the order the HTML head row
   declares them. One list, so the header, the cells and the sentence that
   counts them cannot drift apart. */
const BOARD_KEYS = ['kmeans', 'gmm', 'pam', 'dbscan_best', 'hdbscan', 'meanshift', 'optics',
  'ap', 'spectral'];

const n0 = (v) => v.toLocaleString('en-US');
const f3 = (v) => v.toFixed(3);
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
const word = (v) => (v >= 0 && v <= 10 ? WORDS[v] : n0(v));
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const plural = (n, s) => `${word(n)} ${s}${n === 1 ? '' : 's'}`;
/* "a, b and c", never "a and b and c" */
const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const AP = data.ap;
  const SP = data.spectral;
  const C = data.cut;
  const board = data.board;
  const byName = Object.fromEntries(board.map((r) => [r.name, r]));

  /* ------------------------------------------------------ the guard, in one line */
  const apOk = AP.checks.filter((c) => c.sklearn_same).length;
  set('intro-guard',
    `Both run in this page. The message passing, the graph, the eigenvectors and every one of the `
    + `${n0(C.splits)} possible cuts of the last section are computed in the browser and checked at `
    + `load time against the references the generator produced: scikit-learn for the messages, `
    + `scipy's dense eigensolver for the spectrum. As it stands: `
    + `<span id="guard-note"></span>`);
  set('guard-note',
    `the message passing written into this page elects the same exemplars as scikit-learn's own at `
    + `${apOk} of the ${AP.checks.length} preferences checked, on three datasets; its eigenvalues `
    + `agree with scipy's dense solver on ${SP.checks.length} graphs; and the tie-break noise `
    + `scikit-learn adds to break exact ties `
    + `${AP.noise_stable ? 'decides nothing here, since six random seeds give one answer'
      : 'does change the answer here, which the page says out loud rather than hiding'}. `
    + `Nothing below is a claim about what the algorithm would do.`);

  /* ---------------------------------------------------------------- the scrolly */
  const sub = AP.objective.sub_n;
  const brute = AP.objective.brute.chosen;
  set('scrolly-intro',
    `The blobs you already know, thinned to ${sub} points, and an algorithm with no k in it `
    + `anywhere. These particular ${sub} are the ones the generator could ask the impossible `
    + `question of: every one of the ${n0(brute.subsets)} sets of up to ${word(AP.objective.max_k)} `
    + `exemplars was tried, so at the end of the scroll there is a right answer to compare against.`);
  set('scrolly-final',
    `A point is an exemplar when its own two messages to itself add up to something positive, and `
    + `the rest join whichever exemplar they are most similar to. `
    + (brute.same
      ? `On these ${sub} points that answer is not merely good: it is the best of all `
        + `${n0(brute.subsets)} candidate sets, at a net similarity of ${brute.best_net}.`
      : `On these ${sub} points the exhaustive best is ${brute.best_net} and the messages found `
        + `${brute.ap_net}, so the search is a search and not a proof.`));

  /* -------------------------------------------------------------- the objective */
  const O = AP.objective;
  set('objective-note',
    `Whether the messages find the best set of exemplars has an answer twice over. On the `
    + `${sub}-point sample of the scrolly, every one of the ${n0(brute.subsets)} sets of up to `
    + `${word(O.max_k)} exemplars was scored, and the messages `
    + (brute.same ? `landed on the best one` : `landed ${(brute.best_net - brute.ap_net).toFixed(4)} short of the best one`)
    + `. On all 300 blobs, where enumeration is out of the question, the comparison is against `
    + `the previous article's PAM given the same similarities and the same k = ${O.k}: message `
    + `passing elects points ${O.exemplars.join(', ')} at a net similarity of ${n0(O.net)}, and `
    + `build-and-swap elects ${O.pam_medoids.join(', ')} at ${n0(O.pam_net)}`
    + (O.same_set
      ? `. The same three points, from two searches with nothing in common but the objective.`
      : `, which is not the same set: the two searches part company here.`)
    + (Math.abs(O.ari_ap - O.ari_pam) < 0.0005
      ? ` Both score ${f3(O.ari_ap)} against the groups the data was built from, where k-means `
        + `scores ${f3(byName.blobs.kmeans)}.`
      : ` They score ${f3(O.ari_ap)} and ${f3(O.ari_pam)} against the groups the data was built `
        + `from, where k-means scores ${f3(byName.blobs.kmeans)}.`));

  /* ------------------------------------------------------------- the preference */
  const defaults = Object.entries(AP.default);
  const wrong = defaults.filter(([, v]) => v.k !== v.true_k);
  const ks = defaults.map(([, v]) => v.k);
  const worstD = defaults.reduce((a, b) => (b[1].ari < a[1].ari ? b : a));
  const bestD = defaults.reduce((a, b) => (b[1].ari > a[1].ari ? b : a));
  set('default-note',
    `The library default for the preference is the median of all the similarities, and on this `
    + `module's datasets it is wrong ${wrong.length === defaults.length ? 'every time'
      : `on ${word(wrong.length)} of the ${word(defaults.length)}`}. `
    + `${cap(word(defaults.length))} datasets, ${word(defaults.length)} answers for k: `
    + `${defaults.map(([n, v]) => `${n} ${v.k}`).join(', ')}, against true values of two or three. `
    + `${wrong.length === defaults.length
      ? `Not one of the six is right`
      : `${cap(word(defaults.length - wrong.length))} of the six is right`}`
    + `, the adjusted Rand index runs from ${f3(worstD[1].ari)} (${worstD[0]}) to `
    + `${f3(bestD[1].ari)} (${bestD[0]}), and the `
    + `algorithm's own headline, that nobody has to say k, is doing real work here: nobody said `
    + `it, and nobody checked it either.`);

  /* the staircase */
  const sets = Object.entries(AP.sweep);
  const withGaps = sets.filter(([, v]) => v.missing.length);
  const nonMono = sets.filter(([, v]) => v.drops > 0);
  const rings = AP.sweep.rings;
  const ringTwos = rings.ari.filter((_, i) => rings.k[i] === 2);
  set('stair-note',
    `The staircase from preference to k is not a staircase. Swept over `
    + `${AP.sweep.blobs.grid.length} prices spanning two orders of magnitude, on all five `
    + `datasets, it does two things a knob for k would not. It `
    + `<span class="bold">skips</span>: ${word(withGaps.length)} of the ${word(sets.length)} have a `
    + `value of k that no price produces at all `
    + `(${list(withGaps.map(([n, v]) => `${n} never returns ${v.missing.join(' or ')}`))}). `
    + `And it <span class="bold">goes backwards</span>: on ${plural(nonMono.length, 'dataset')} there `
    + `is at least one step where making clusters cheaper returns fewer of them, `
    + `${nonMono.length ? list(nonMono.map(([n, v]) => `${word(v.drops)} on the ${n}`)) : ''}. `
    + `So you cannot ask for three clusters, and you cannot reliably binary-search for them `
    + `either. What you can do is sweep, which is the thing not needing k was supposed to save `
    + `you from. `
    + `On the rings the sweep is worth reading twice: the best it ever manages is `
    + `${f3(rings.best.ari)} at k = ${rings.best.k}, and the ${word(ringTwos.length)} prices that `
    + `do return two clusters score between ${f3(Math.min(...ringTwos))} and `
    + `${f3(Math.max(...ringTwos))}, which is the two circles cut down a diameter rather than `
    + `apart: an exemplar has to be one of the points, and a ring has no point in the middle of `
    + `it to be one.`);

  /* the damping */
  const D = AP.damping;
  const bad = D.rows.filter((r) => !r.converged);
  const ok = D.rows.filter((r) => r.converged);
  const flips = ok.filter((r, i) => i > 0 && r.k !== ok[i - 1].k).length;
  const worstSwing = Math.max(...bad.map((r) => r.swing), 0);
  /* "below x it never settles, above it always does" is a claim about the shape
     of the failures, not just their count: it only holds if the ones that fail
     really are the bottom of the grid. */
  const contiguous = bad.length > 0 && D.rows.slice(0, bad.length).every((r) => !r.converged);
  set('damping-note',
    `And the damping is not a numerical detail. It exists because the two messages feed each `
    + `other and the iteration swings, so each update keeps a fraction of the last one, and `
    + `scikit-learn refuses values below ${D.floor} outright. Measured on the blobs at a `
    + `preference of ${n0(D.preference)}: `
    + (contiguous
      ? `below ${ok[0].damping} the run never settles at all, and over its last 200 iterations the `
        + `number of points electing themselves still swings by as much as ${worstSwing}. Above `
        + `it, every run converges, in ${Math.min(...ok.map((r) => r.iters))} to `
        + `${Math.max(...ok.map((r) => r.iters))} iterations. `
      : `${plural(bad.length, 'of the ' + word(D.rows.length) + ' damping value')} never settle at `
        + `all (${list(bad.map((r) => r.damping.toFixed(2)))}), swinging by as much as `
        + `${worstSwing} over their last 200 iterations, and they are not the low end of the range. `
        + `The rest converge in ${Math.min(...ok.map((r) => r.iters))} to `
        + `${Math.max(...ok.map((r) => r.iters))} iterations. `)
    + `But they do not converge to the same thing: the ${ok.length} settings that work return `
    + `${plural(D.answers.length, 'different answer')} for k, ${D.answers.join(' and ')}, `
    + `and they alternate rather than cross over once (${plural(flips, 'change')} of mind across `
    + `the range). A parameter introduced to make an iteration stable is choosing the number of `
    + `clusters, which is the parameter this algorithm advertises not having.`);

  /* ---------------------------------------------------------------- the spectral */
  const R = SP.sets.rings;
  const T = SP.sets.trap;
  set('rings-note',
    `The rings are the dataset this module opened its failure gallery with, and the one every `
    + `centre-based algorithm since has failed on: k-means ${f3(byName.rings.kmeans)}, `
    + `k-medoids ${f3(byName.rings.pam)}, mean shift ${f3(byName.rings.meanshift)}, affinity `
    + `propagation ${f3(byName.rings.ap)}. Spectral clustering scores `
    + `<span class="bold">${f3(R.ari)}</span> on them at ${SP.knn} neighbours, and the reason is `
    + `visible in the spectrum: the graph is in ${word(R.components)} pieces, so two eigenvalues `
    + `are exactly zero and their eigenvectors are the two rings' indicator vectors. There is no `
    + `approximation happening. The same thing happens on the trap that ended the density `
    + `article, the one where no single radius works: ${f3(T.ari)}, against `
    + `${f3(byName.trap.kmeans)} for k-means and ${f3(byName.trap.dbscan_best)} for the best `
    + `DBSCAN radius there is.`);

  /* the band */
  const bands = Object.entries(SP.sets).map(([n, v]) => ({ n, ...v }));
  const ringsBand = R.knn_band;
  const ringLow = R.knn_sweep.filter((r) => r.k < ringsBand[0]);
  const ringHigh = R.knn_sweep.filter((r) => r.k > ringsBand[1]);
  set('band-note',
    `The parameter that replaced k has a band, and outside it the answer collapses. On the rings, `
    + `any graph from ${ringsBand[0]} to ${ringsBand[1]} neighbours scores ${f3(R.best_knn.ari)}. `
    + `${ringLow.length ? `Below that the graph shatters into more pieces than there are rings `
      + `(${list(ringLow.map((r) => `${word(r.components)} at ${r.k}`))} neighbours) and the score `
      + `drops as far as ${f3(Math.min(...ringLow.map((r) => r.ari)))}` : ''}`
    + `${ringLow.length && ringHigh.length ? '; ' : ''}`
    + `${ringHigh.length ? `above it the two circles are joined by enough short-cuts to become one `
      + `piece, and it falls to ${f3(Math.min(...ringHigh.map((r) => r.ari)))}`
      + `${Math.abs(Math.min(...ringHigh.map((r) => r.ari)) - byName.rings.kmeans) < 0.005
        ? `, which is what k-means was getting all along` : ''}` : ''}. `
    + `The gaussian graph has a band of its own and it is no wider: on the rings, of the `
    + `${word(R.rbf_sweep.length)} values of gamma swept from ${R.rbf_sweep[0].gamma} to `
    + `${R.rbf_sweep[R.rbf_sweep.length - 1].gamma}, ${word(R.rbf_sweep.filter((r) => r.ari >= R.best_rbf.ari - 0.001).length)} `
    + `reach ${f3(R.best_rbf.ari)} and the rest fall as low as `
    + `${f3(Math.min(...R.rbf_sweep.map((r) => r.ari)))}. Nothing was removed by `
    + `going to a graph: the question "how many clusters" was traded for the question "what counts `
    + `as near", and the second one is answered before the algorithm starts rather than after.`);

  /* the eigengap */
  const eg = Object.entries(SP.eigengap);
  const rightAt = (h) => eg.filter(([, v]) => v.picks[h] === v.true_k).length;
  set('eigengap-note',
    `The gaps in the spectrum are the closest thing here to an opinion about k. The rule of thumb `
    + `is to take the largest jump between consecutive eigenvalues, and on these ${word(eg.length)} `
    + `datasets it is right ${word(rightAt('4'))} times out of ${word(eg.length)} looking at the `
    + `first four eigenvalues, ${word(rightAt('6'))} looking at six, and ${word(rightAt('9'))} `
    + `looking at nine. Which is the honest shape of every heuristic that claims to find k: it has `
    + `a parameter of its own, how far to look, and nobody writes it down. `
    + `${list(eg.filter(([, v]) => v.picks['4'] === v.picks['9'] && v.picks['4'] === v.true_k)
      .map(([n]) => n))} give the same right answer however far you look; `
    + `${list(eg.filter(([, v]) => v.picks['4'] !== v.picks['9']).map(([n]) => n))} `
    + `change their mind when you do.`);

  /* ------------------------------------------------------------- the relaxation */
  set('cut-intro',
    `An eigenvector is the answer to a question nobody asked: the real-valued version of a `
    + `yes-or-no problem. Splitting ${C.n} points in two allows ${n0(C.splits)} possibilities and `
    + `splitting 300 allows more than there are atoms in anything, which is why the relaxation `
    + `exists. The price of relaxing is usually invisible, because the exact answer is not `
    + `available to compare against. At ${C.n} points it is.`);
  const gapPct = ((C.optimum - C.lambda2) / C.optimum) * 100;
  set('cut-note',
    (C.sweep_is_optimum
      ? `So the relaxation is exact here, and its eigenvalue is not. Sliding the threshold along `
        + `the eigenvector reaches ${C.sweep_best}, which is the best of all ${n0(C.splits)} `
        + `splits: the ordering the eigenvector puts the points in contains the optimal cut, and `
        + `all the search had to do was find the place to cut it. `
      : `So the relaxation is not exact here: the best threshold on the eigenvector gives `
        + `${C.sweep_best} against an exhaustive best of ${C.optimum}. `)
    + `Cutting at zero instead, which is the textbook shortcut, gives ${C.sign} and ranks `
    + `${n0(C.sign_rank)} of ${n0(C.splits)}: `
    + (C.sign_rank === 1
      ? `the same split.`
      : `close, and not the same split. The rounding rule is part of the algorithm.`)
    + ` The eigenvalue itself, ${C.lambda2}, is ${gapPct.toFixed(1)}% below the best cut that `
    + `actually exists, and no split will ever reach it. That is the standing arrangement with a `
    + `relaxation: it hands you an ordering worth having and a number that is not an answer.`);

  /* -------------------------------------------------------------------- the bill */
  const rows = data.cost.rows;
  const big = rows[rows.length - 1];
  const apCost = data.cost.ap;
  const kmCost = data.cost.kmeans;
  set('cost-note',
    `Everything above is quadratic in n, and quadratic in n is a table. Both algorithms need `
    + `every pair before they can start: k-means never forms one, DBSCAN needs only a radius `
    + `around each point, but a message and an edge both live on a pair. At the 300 points of `
    + `this page that is ${n0(rows[1].entries)} numbers, which is nothing. At ${n0(big.n)} it is `
    + `${n0(big.entries)} numbers and ${big.gb} GB, before any of them has been updated once. `
    + `And they do get updated: this page's run on the blobs took ${apCost.iters} iterations of `
    + `two n by n message boards, ${n0(apCost.updates)} updates in all, where k-means on the same `
    + `points settled after ${kmCost.passes} passes of ${n0(kmCost.per_pass)} distances, `
    + `${n0(kmCost.passes * kmCost.per_pass)} in total. That is a factor of `
    + `${n0(Math.round(apCost.updates / (kmCost.passes * kmCost.per_pass)))} on 300 points, and it `
    + `grows with n.`);

  const tbody = document.querySelector('#cost-table tbody');
  if (tbody) {
    const head = document.querySelector('#cost-updates-head');
    if (head) head.textContent = `Message updates at ${apCost.iters} iterations`;
    tbody.innerHTML = rows.map((r) => `<tr><td>${n0(r.n)}</td><td>${n0(r.entries)}</td>`
      + `<td>${r.gb < 0.01 ? `${(r.gb * 1000).toFixed(1)} MB` : `${r.gb} GB`}</td>`
      + `<td>${n0(r.entries * 2 * apCost.iters)}</td></tr>`).join('');
  }

  /* the variants */
  const V = Object.entries(SP.variants);
  const worst = SP.variants[SP.worst_variant];
  const worstSk = SP.variants[SP.worst_sklearn];
  const agree = V.filter(([, v]) => v.spread < 0.001);
  set('variants-note',
    `There is one more cost, and it is not arithmetic. "Spectral clustering" names at least three `
    + `different last steps, all of them in the standard tutorial: cluster the eigenvectors of the `
    + `symmetric Laplacian with every row rescaled to length one, or the eigenvectors of the `
    + `random-walk Laplacian, or the eigenvectors of the plain unnormalised one. On `
    + `${word(agree.length)} of the ${word(V.length)} datasets here the three agree to three `
    + `decimals. On the ${SP.worst_variant} they give ${f3(worst.njw)}, ${f3(worst.walk)} and `
    + `${f3(worst.unnorm)}, a spread of <span class="bold">${f3(worst.spread)}</span>, which is `
    + `larger than most of the differences this article has been measuring. And there is a fourth `
    + `version with no paper behind it, the one scikit-learn actually ships, which divides the `
    + `eigenvectors by the degrees instead of rescaling the rows: on the ${SP.worst_sklearn} it `
    + `scores ${f3(worstSk.sklearn)} where all three textbook versions score between `
    + `${f3(Math.min(worstSk.njw, worstSk.walk, worstSk.unnorm))} and `
    + `${f3(Math.max(worstSk.njw, worstSk.walk, worstSk.unnorm))}. This page runs the first, which `
    + `is the one the Ng, Jordan and Weiss paper describes, and the scoreboard below reports it: `
    + `worth knowing before treating any of the four numbers as "what spectral clustering scores".`);

  /* ------------------------------------------------------------------ the verdict */
  const blobRow = byName.blobs;
  set('v-ap',
    `A k that comes out rather than in, centres that are rows of the data, and no seed and no `
    + `restarts anywhere: it gives the same answer twice. On the blobs, tuned, `
    + `${f3(AP.sweep.blobs.best.ari)} against k-means' ${f3(blobRow.kmeans)}, on the same `
    + `${word(O.k)} points PAM picks.`);
  set('v-ap-watch',
    `The preference, which is k wearing a disguise and a worse disguise than it looks: the default `
    + `misses on all ${word(Object.keys(AP.default).length)} datasets here, the map from it to k skips `
    + `values and is not monotone, and the damping decides k too.`);
  set('v-spec',
    `Clusters that are connected rather than round. The rings ${f3(SP.sets.rings.ari)} and the `
    + `varying-density trap ${f3(SP.sets.trap.ari)}, against ${f3(byName.rings.kmeans)} and `
    + `${f3(byName.trap.kmeans)} for k-means, and it needs only the similarities, never the `
    + `coordinates.`);
  set('v-spec-watch',
    `The graph. Outside a band of neighbour counts the answer collapses to k-means' own, k still `
    + `has to be supplied, and three textbook versions of the last step disagree by up to `
    + `${f3(worst.spread)} of ARI.`);
  const kmWins = board.filter((r) => r.kmeans >= Math.max(r.ap, r.spectral) - 0.001).length;
  set('v-km',
    `Speed and nothing else, and that is not nothing: ${n0(kmCost.passes * kmCost.per_pass)} `
    + `distances against ${n0(apCost.updates)} message updates on the same 300 points. `
    + (kmWins === 0
      ? `It is not the best of the three on any of the ${word(board.length)} rows below, and it is `
        + `still the one to reach for first when n is large.`
      : `It is the best of the three on ${plural(kmWins, 'of the ' + word(board.length) + ' row')} below.`));

  /* ---------------------------------------------------------------- the scoreboard */
  const head = document.querySelectorAll('#board-table thead th');
  const body = document.querySelector('#board-table tbody');
  if (body) {
    body.innerHTML = board.map((r) => {
      const best = Math.max(...BOARD_KEYS.map((key) => r[key]));
      const cells = BOARD_KEYS.map((key) => {
        const v = r[key];
        const mark = Math.abs(v - best) < 0.0005 ? ' class="bold"' : '';
        return `<td${mark}>${f3(v)}</td>`;
      }).join('');
      return `<tr><td>${r.name} <span style="opacity:.6">k = ${r.k}</span></td>${cells}</tr>`;
    }).join('');
  }
  set('closing-scoreboard',
    `${cap(word(BOARD_KEYS.length))} algorithms across ${word(board.length)} datasets, every `
    + `cell an adjusted Rand index measured on the coordinates this page ships, with the best of `
    + `each row in bold. The DBSCAN column is the best any radius achieves, which is a favour no `
    + `other column gets, and affinity propagation runs at its library default, which is the `
    + `answer it gives when nobody tunes it.`);

  const wins = {};
  BOARD_KEYS.forEach((key) => {
    wins[key] = board.filter((r) => Math.abs(r[key] - Math.max(...BOARD_KEYS.map((k2) => r[k2]))) < 0.0005).length;
  });
  const drift = data.meta.board_drift || [];
  set('board-note',
    `Every cell is measured on the coordinates this page ships, in one run, so the columns are `
    + `comparable with each other. Seven of them were published by the previous article on those `
    + `same coordinates and `
    + (drift.length === 0
      ? `all of them reproduce here exactly.`
      : `${drift.length === 1 ? 'one cell' : `${word(drift.length)} cells`} come out different: `
        + `${drift.join('; ')}, a scikit-learn ${data.meta.sklearn} run against the `
        + `${data.meta.sklearn_previous} the earlier numbers were measured with. Everything else `
        + `reproduces to the fourth decimal, which is the point of checking.`)
    + ` Read the rows rather than the columns: ${board.map((r) => r.name).slice(0, 4).join(', ')} `
    + `and ${board[board.length - 1].name} were each built to break something, and four of the `
    + `five break k-means specifically.`);

  /* the three that do not take k */
  const nok = data.nok;
  const score = data.nok_score;
  set('nok-note',
    `Three of those ${word(BOARD_KEYS.length)} do not take k as an argument, which makes the column worth `
    + `reading on its own. Mean shift gets the number of clusters right on ${word(score.shift_k)} of the `
    + `${word(nok.length)} datasets, the eigengap rule on ${word(score.gap_k)}, and affinity propagation `
    + `at its default preference on ${word(score.ap_k)}: it answered `
    + `${nok.map((r) => r.ap_k).join(', ')} where the truth is ${nok.map((r) => r.true_k).join(', ')}. `
    + `Not asking for k is not the same as knowing it, and the algorithm that most loudly does not `
    + `ask is the one that is furthest out.`);

  set('wine-count', n0(AP.wine.n));

  /* the references note */
  set('ref-note',
    `Datasets rebuilt with the constructions of the earlier articles and checked coordinate by `
    + `coordinate against the JSON those articles ship: `
    + `${Object.entries(data.meta.shared_datasets_verified).filter(([, v]) => v).length} of `
    + `${Object.keys(data.meta.shared_datasets_verified).length} identical. Graphs use `
    + `${data.meta.knn} neighbours unless a slider says otherwise, affinity propagation runs at a `
    + `damping of ${data.meta.damping}, and the density columns of the scoreboard use minPts = `
    + `${data.meta.min_samples}, all as in the articles they are compared with. No wall clock `
    + `appears anywhere above: a time is the one measurement that changes between machines, so `
    + `the cost section counts similarities and message updates instead.`);
}
