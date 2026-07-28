/* Every sentence on this page that carries a figure, written from the file.
 *
 * Same discipline as the previous articles: nothing numeric is typed by hand,
 * and every comparison is written AS a comparison, with both outcomes spelled
 * out, so that rerunning the generator moves the sentence instead of leaving it
 * quietly wrong. Table cells too, and the references band at the bottom, which
 * is where a stale figure survives longest because nobody rereads it.
 */
const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const SHORT = {
  'od280/od315_of_diluted_wines': 'od280/od315',
  nonflavanoid_phenols: 'nonflavanoid phenols',
  color_intensity: 'colour intensity',
  alcalinity_of_ash: 'alcalinity of ash',
  malic_acid: 'malic acid',
  total_phenols: 'total phenols',
};
const label = (n) => SHORT[n] ?? n.replace(/_/g, ' ');

export function initProse(data) {
  const W = data.wine;
  const M = data.metrics.rows;
  const S = data.scaling;
  const F = data.factor;
  const G = data.digits;
  const euclid = M.find((r) => r.key === 'euclidean');
  const city = M.find((r) => r.key === 'cityblock');
  const sqcity = M.find((r) => r.key === 'sqrt cityblock');
  const cheb = M.find((r) => r.key === 'chebyshev');
  const cosine = M.find((r) => r.key === 'cosine');
  const worstMetric = M.slice(1).reduce((a, b) => (b.negative_mass > a.negative_mass ? b : a));
  const lowestStress = M.reduce((a, b) => (b.stress2 < a.stress2 ? b : a));
  const ctrl = F.control;
  const loud = F.sds[F.sds.length - 1];
  const top = F.sweep[F.sweep.length - 1];
  const quiet = F.sweep[0];
  const gammaRows = S.rows;
  const g1 = gammaRows.find((r) => r.gamma === 1);
  const gWorst = gammaRows.reduce((a, b) => (b.metric_disparity > a.metric_disparity ? b : a));
  const noiseWorst = S.noise_rows[S.noise_rows.length - 1];

  /* ------------------------------------------------------------- the opening */
  const nonEuclid = M.filter((r) => r.neg_count > 0).length;
  set('opening-note',
    `Three numbers say in advance what the rest of the page spends its time on. `
    + `Rebuilding the ${n0(W.n)} bottles from their distance table alone puts every bottle within `
    + `<span class="bold">${W.mds_pca_gap.toExponential(0)}</span> of where PCA put it, which is not `
    + `a resemblance but double precision arithmetic agreeing with itself. Change the definition of `
    + `distance and that stops being true in a way you can measure: of the six on this page, `
    + `<span class="bold">${nonEuclid}</span> produce eigenvalues that come back negative, and a `
    + `negative eigenvalue means no set of points anywhere has those distances. And on a dataset `
    + `built so the answer is known, giving one of eight variables a private noise of `
    + `${loud.toFixed(1)} puts PCA <span class="bold">${(top.pca_gap / top.fa_gap).toFixed(0)} times</span> `
    + `further from the truth than factor analysis, for the single reason that PCA has nowhere to `
    + `put the idea of noise.`);

  /* --------------------------------------------------------------- the scrolly */
  set('scrolly-step-0',
    `Two bottles from each cultivar, and every distance between them, in the thirteen standardised `
    + `measurements. That is ${(6 * 5) / 2} numbers for six bottles; for all ${n0(W.n)} it is `
    + `${n0((W.n * (W.n - 1)) / 2)}, and the largest of them is ${W.d_max} against a mean of `
    + `${W.d_mean}. The table is symmetric and its diagonal is zero, which is the only structure it `
    + `is guaranteed to have.`);
  set('scrolly-step-1',
    `The whole table. Nothing has been sorted: the bottles arrive in the order the UCI file lists `
    + `them, which happens to be by cultivar, and that is why the block structure is visible. The `
    + `thirteen measurements are gone. If you were handed only this, could you put the bottles back `
    + `on a page?`);
  set('scrolly-step-3',
    `Diagonalise it. The eigenvalues are the variances of the coordinates that come back, and a `
    + `variance cannot be negative, so a negative eigenvalue would mean the table was impossible. `
    + `Here the most negative of the ${n0(W.n)} is `
    + `<span class="bold">${Math.abs(euclid.most_negative) < 1e-9 ? '0' : euclid.most_negative}</span>, `
    + `and the share of eigenvalue mass arriving negative is ${W.negative_mass.toExponential(0)}, `
    + `which is not a small number, it is zero written by a computer.`);
  set('scrolly-step-4',
    `Take the two largest eigenvalues, scale their eigenvectors by the square roots, and that is a `
    + `map. These two directions carry <span class="bold">${pct(euclid.evr2, 1)}</span> of the `
    + `positive eigenvalue mass, so a good deal is missing, and the residual has a name: the stress `
    + `of this picture is ${euclid.stress2}. Nothing in the calculation was told about cultivars.`);
  set('scrolly-step-5',
    `Now the check that makes the section worth writing. Run PCA on the original ${W.features.length} `
    + `columns, which the scaling never saw, and plot its first two scores as rings. Worst `
    + `disagreement over all ${n0(W.n)} bottles and both coordinates: `
    + `<span class="bold">${W.mds_pca_gap.toExponential(0)}</span>. The eigenvalues match too, `
    + `${W.eig_variance_gap.toExponential(0)} apart after dividing by n. Classical scaling on `
    + `Euclidean distances <i>is</i> PCA. It is the same eigenproblem entered from the other end.`);

  set('identity-note',
    `The middle equality is the whole trick and it is worth saying slowly: double centring the `
    + `squared distances recovers the matrix of inner products \\(X_c X_c^{\\top}\\), and the `
    + `eigenvectors of \\(X_c X_c^{\\top}\\) are related to those of \\(X_c^{\\top} X_c\\), which is `
    + `\\(n\\) times the covariance matrix PCA diagonalises. Same non-zero eigenvalues, same `
    + `coordinates. Measured on these bottles the two agree to `
    + `<span class="bold">${W.mds_pca_gap.toExponential(0)}</span> per coordinate and `
    + `${W.eig_variance_gap.toExponential(0)} per eigenvalue. So when the data really is a matrix `
    + `of numbers and distance really means straight line, going through the distance table buys `
    + `nothing at all, and costs an \\(n \\times n\\) matrix where PCA needed a `
    + `\\(p \\times p\\) one. The reason to go through it anyway is that sometimes the numbers `
    + `handed to you are not distances between points.`);

  /* ----------------------------------------------------------- which distance */
  set('metric-note',
    `The results split cleanly in two, and the split is not the one intuition offers. `
    + `<span class="bold">Cityblock</span>, the one that feels most like a distance because it is `
    + `the one a taxi drives, is <span class="bold">not</span> Euclidean at all: `
    + `${pct(city.negative_mass)} of its eigenvalue mass arrives negative, over `
    + `${city.neg_count} of the ${n0(W.n)} eigenvalues. Take the square root of every one of those `
    + `same numbers and the negative mass falls to `
    + `<span class="bold">${sqcity.negative_mass.toExponential(0)}</span>, which is zero: the square `
    + `root of a cityblock distance <i>is</i> a Euclidean distance, for every dataset, and that is a `
    + `theorem rather than a property of these bottles. Chebyshev is the worst offender at `
    + `${pct(cheb.negative_mass)}, and cosine and correlation, the two everybody reaches for when the `
    + `magnitudes are not meaningful, sit at ${pct(cosine.negative_mass)} and `
    + `${pct(M.find((r) => r.key === 'correlation').negative_mass)}.`
    + `<br /><br />`
    + `That is not a reason to avoid them. It is a reason to know what the picture is: with `
    + `${worstMetric.neg_count} negative eigenvalues in the spectrum, the two-dimensional map is a `
    + `projection of something that was never a shape, and the coordinates it prints do not obey the `
    + `triangle inequality that the reader's eye is silently assuming. The measured repair is the `
    + `Lingoes constant, the smallest number you can add to every squared distance to make a real `
    + `configuration exist: ${city.lingoes} for cityblock against its mean distance of `
    + `${city.d_mean}, and ${cosine.lingoes} for cosine against ${cosine.d_mean}. It is not a small `
    + `adjustment, and having to make it is the finding.`);

  set('stress-note',
    `One trap in that widget deserves naming, because it is the trap the number invites. The lowest `
    + `two-dimensional stress of the six belongs to <span class="bold">${lowestStress.key}</span> at `
    + `${lowestStress.stress2}, comfortably below the Euclidean ${euclid.stress2}, and it would be `
    + `very easy to read that as "${lowestStress.key} is the better distance for this data". `
    + `${lowestStress.neg_count > 0
      ? `It is the opposite. Stress measures how well a flat picture reproduces the table, and the `
        + `${lowestStress.key} table has ${pct(lowestStress.negative_mass)} of its mass in dimensions `
        + `that do not exist. Something with less real structure is easier to draw. A goodness of fit `
        + `is only a goodness of fit to the thing you handed in, and if that thing is not a geometry `
        + `then fitting it well is not an achievement.`
      : `Here the lowest stress does belong to a genuinely Euclidean table, so on this run the number `
        + `can be read at face value.`}`);

  /* ------------------------------------------------------- metric vs non-metric */
  set('ordinal-note',
    `Everything so far took the numbers in the table literally: a 4 means twice as far as a 2. `
    + `Very often that is more than anyone can honestly claim. A panel rates similarity from 1 to 7; `
    + `a classifier reports confusion counts; a survey asks people to rank. In all of those the `
    + `<span class="bold">order</span> is meaningful and the spacing is a convention, and fitting a `
    + `model to the spacing is fitting a model to the convention. Kruskal's answer, in 1964, was to `
    + `ask for less: find a configuration whose distances are in the same order as the `
    + `dissimilarities, and do not care what function connects them as long as it only goes up.`);

  const flat = S.nonmetric_spread < 1e-9;
  set('shepard-note',
    `The nine settings of that slider produce nine different metric answers and `
    + `${flat ? 'one' : 'nine'} non-metric one${flat ? '' : 's'}. At an exponent of `
    + `${gWorst.gamma} the metric fit is ${gWorst.metric_disparity} away from the truth and its `
    + `stress has climbed to ${gWorst.metric_stress}, while the non-metric fit sits at `
    + `${gWorst.nonmetric_disparity} with a stress of ${gWorst.nonmetric_stress}. `
    + (flat
      ? `The non-metric column is not merely stable, it is the <span class="bold">same map</span> `
        + `every time: measured across all nine exponents the largest coordinate difference between `
        + `any two of them is ${S.nonmetric_spread.toExponential(0)}. A monotone distortion cannot `
        + `change an ordering, the algorithm reads nothing but the ordering, so the distortion is not `
        + `merely survivable, it is invisible.`
      : `The non-metric answers do move a little across the nine, by up to `
        + `${S.nonmetric_spread.toExponential(1)}.`)
    + ` For reference, the undistorted case is where metric scaling is unbeatable: at exponent 1 it `
    + `reaches a stress of ${g1.metric_stress} and a distance from the truth of `
    + `${g1.metric_disparity}, and scikit-learn's own SMACOF, started from eight random `
    + `configurations instead of this page's one, gets to ${S.sklearn_stress}.`);

  set('noise-note',
    `Which makes non-metric scaling sound free, and it is not, and the price is exactly where the `
    + `argument said it would be. Reading only the ordering is safe when the ordering is the `
    + `trustworthy part. Add noise to every dissimilarity instead of bending them monotonically, so `
    + `that the ordering itself starts to shuffle, and the two methods swap places: at a noise level `
    + `that leaves the rank correlation at ${noiseWorst.rank_corr}, metric scaling lands `
    + `${noiseWorst.metric_disparity} from the truth and non-metric `
    + `<span class="bold">${noiseWorst.nonmetric_disparity}</span>, about `
    + `${(noiseWorst.nonmetric_disparity / Math.max(noiseWorst.metric_disparity, 1e-9)).toFixed(0)} `
    + `times worse. Averaging is what makes the metric version robust, and refusing to average is `
    + `what makes the non-metric version faithful. Neither is the safe default.`);

  const tbody = document.querySelector('#noise-table');
  if (tbody) {
    tbody.innerHTML = S.noise_rows.map((r) => {
      const better = r.metric_disparity <= r.nonmetric_disparity ? 'metric' : 'nonmetric';
      const cell = (v, which) => (which === better && r.eps > 0
        ? `<span class="bold">${v.toFixed(5)}</span>` : v.toFixed(5));
      return `<tr><td>${r.eps === 0 ? 'none' : r.eps.toFixed(2)}</td>`
        + `<td>rank correlation ${r.rank_corr}</td>`
        + `<td>${cell(r.metric_disparity, 'metric')}</td>`
        + `<td>${cell(r.nonmetric_disparity, 'nonmetric')}</td></tr>`;
    }).join('');
  }

  /* ------------------------------------------------------------ factor analysis */
  set('factor-intro',
    `The second half of the article changes what is being distrusted. Suppose the data matrix does `
    + `exist and every number in it is real; there is still a question PCA cannot be asked. `
    + `Of the variance in column 5, how much is column 5 doing something the other columns also do, `
    + `and how much is column 5's own instrument being noisy? PCA has no opinion, because it does `
    + `not have a model: it rotates the axes so the first one catches the most variance, and if the `
    + `most variance belongs to the noisiest instrument then that is where the first axis goes.`);

  const eqRatio = ctrl.equal.pca_gap / ctrl.equal.fa_gap;
  const unRatio = ctrl.unequal.pca_gap / ctrl.unequal.fa_gap;
  set('control-note',
    `The slider is one experiment; the control is the other half of it. Turn the noise on variable 1 `
    + `up to ${ctrl.sd} and PCA sits ${ctrl.unequal.pca_gap} from the true plane against factor `
    + `analysis's ${ctrl.unequal.fa_gap}, a factor of ${unRatio.toFixed(0)}. Now change exactly one `
    + `thing, give <span class="bold">every</span> variable the same private noise instead, and the `
    + `gap closes: ${ctrl.equal.pca_gap} against ${ctrl.equal.fa_gap}, a factor of `
    + `${eqRatio.toFixed(2)}. `
    + (eqRatio < 1.6
      ? `That is not a coincidence and it is not a weak effect. With equal noise on every variable `
        + `the maximum likelihood factor model and PCA have the same solution: it is a theorem, and `
        + `the measurement is the theorem showing up. Everything factor analysis buys, it buys from `
        + `the <i>inequality</i> of the noise, and nowhere else.`
      : `Even with equal noise the two differ here by a factor of ${eqRatio.toFixed(2)}, which is `
        + `more than the theory leads you to expect from this sample size.`)
    + ` The other half of the model shows up in the off-diagonals: reproducing the covariances `
    + `between different variables, which is all a factor model ever tries to do, factor analysis `
    + `leaves a residual of ${ctrl.unequal.off_resid_fa} against PCA's `
    + `${ctrl.unequal.off_resid_pca}, and its estimates of the eight private variances correlate `
    + `${ctrl.unequal.psi_corr} with the true ones.`);

  set('scale-note',
    `The obvious objection is that nobody would run PCA on raw columns of wildly different `
    + `variance anyway: you would standardise first, and standardising would fix this. It helps `
    + `PCA a great deal, from ${ctrl.unequal.pca_gap} to ${ctrl.standardised.pca_gap}, and it is `
    + `worth understanding why that is not the same as fixing it. Standardising is a `
    + `<i>guess</i> that every variable deserves equal weight, made before looking; the factor model `
    + `<i>estimates</i> how much each variable deserves, from the data. Measured directly: the share `
    + `of each variable's variance that factor analysis calls private is the same number before and `
    + `after standardising, to <span class="bold">${ctrl.scale_invariance.toExponential(0)}</span>, `
    + `because maximum likelihood factor analysis is equivariant under rescaling and simply does not `
    + `notice. PCA's first direction, over the same change, moves by `
    + `<span class="bold">${ctrl.pca_direction_moves}</span> in its largest coordinate. One method `
    + `has a decision to make about units and the other does not.`);

  const wine = F.wine;
  set('rotation-note',
    `And now the strangest property of the model, which is not a defect but is very often sold as `
    + `one. Multiply the loadings by any rotation you like and \\(\\Lambda\\Lambda^{\\top}\\) is `
    + `unchanged, so the fit is unchanged, so the likelihood is unchanged: the model cannot tell you `
    + `which axes inside the factor space are the real ones, because as far as the data is concerned `
    + `there is no such thing. Measured on the wine with ${wine.k} factors, applying Kaiser's varimax `
    + `rotation moves individual loadings by up to <span class="bold">${wine.rot_load_change}</span>, `
    + `raises the simplicity of the picture from ${wine.simplicity_before} to `
    + `${wine.simplicity_after}, and changes the implied covariance matrix by `
    + `<span class="bold">${wine.rot_fit_gap.toExponential(0)}</span>. The story changes completely `
    + `and the model does not change at all. PCA does not have this freedom, because it pins the axes `
    + `by variance, and that is genuinely convenient and is also the reason people read meaning into `
    + `PCA's axes that the data never put there.`);

  const noisiest = wine.uniquenesses.indexOf(Math.max(...wine.uniquenesses));
  const bestK = wine.by_k.find((r) => r.k === wine.best_k);
  set('wine-factor-note',
    `On the actual wine the model earns something PCA cannot offer at all: it can be asked how many `
    + `factors there are and it can answer, because it has a likelihood. By BIC the answer is `
    + `<span class="bold">${wine.best_k}</span> (${wine.by_k.map((r) => `k=${r.k}: ${r.bic}`).join(', ')}), `
    + `and with ${wine.k} factors fitted the measurement it trusts least is `
    + `<span class="bold">${label(wine.noisiest)}</span>, ${pct(wine.noisiest_share, 1)} of whose `
    + `variance it attributes to that column alone and to nothing shared. The one it trusts most is `
    + `${label(wine.cleanest)} at ${pct(wine.cleanest_share, 1)}. Neither of those is a number PCA `
    + `has a slot for, and both are checkable claims about the world: if magnesium really is mostly `
    + `its own instrument, then dropping it should cost almost nothing downstream.`);

  /* ------------------------------------------------------------------ the digits */
  set('digits-intro',
    `One last table, and this one has no matrix behind it by construction. Take the same `
    + `${n0(G.n_train)} MNIST digits the <a href="../directions/">previous article</a> factorised, `
    + `train a plain ${G.k}-nearest-neighbour classifier on them, and run it on `
    + `${n0(G.n_test)} held out images: it gets ${pct(G.accuracy, 1)} right. The interesting part is `
    + `the ${pct(1 - G.accuracy, 1)} it gets wrong, because which digits it confuses with which is a `
    + `measurement of how alike they look to something that has never been told anything about `
    + `shapes. Turn those counts into dissimilarities the way Shepard did in 1962, and there is a `
    + `ten by ten table with no coordinates anywhere behind it. There is nothing here for PCA to `
    + `rotate. Scaling does not care.`);

  const pairText = G.top_pairs.slice(0, 3)
    .map((p) => `${p.a} and ${p.b} (${p.d})`).join(', ');
  set('digits-note',
    `The map is not a surprise, which is how you know the method worked: the closest pair in the `
    + `whole table is <span class="bold">${G.closest[0]} and ${G.closest[1]}</span> at `
    + `${G.closest_d}, then ${pairText.split(', ').slice(1).join(', ')}, and the furthest is `
    + `${G.furthest[0]} and ${G.furthest[1]}. `
    + `${G.neg_count > 0
      ? `But the spectrum has ${G.neg_count} negative eigenvalue carrying `
        + `${pct(G.negative_mass)} of the mass, so this table is not quite a geometry either, and `
        + `the smallest repair that would make it one is adding ${G.lingoes} to every squared `
        + `distance. That is the honest reason to prefer the non-metric button here: with `
        + `${G.never_confused} of the 45 pairs never confused even once, ${G.never_confused} `
        + `entries of the table are tied at the maximum, and only a method that reads the ordering `
        + `is allowed to say "at least this far" instead of inventing a number.`
      : `And the spectrum has nothing negative in it, so the table is a genuine geometry.`}`);

  /* -------------------------------------------------------------- verdict table */
  set('verdict-pca',
    `Classical scaling on Euclidean distances gives the identical answer (${W.mds_pca_gap.toExponential(0)} `
    + `per coordinate on these ${n0(W.n)} bottles) for more work: an n by n eigenproblem instead of `
    + `a p by p one.`);
  set('verdict-pca-warn',
    `It has no model of noise, so a variable that varies because its instrument is bad gets the same `
    + `respect as one that varies because something is happening. Measured cost here: `
    + `${ctrl.unequal.pca_gap} against ${ctrl.unequal.fa_gap}.`);
  set('verdict-classical',
    `It is the only method here that never needed the data matrix. And its eigenvalues report, for `
    + `free, whether the table could have come from points at all.`);
  set('verdict-classical-warn',
    `Check the negative end of the spectrum before believing the picture: ${pct(city.negative_mass)} `
    + `of the mass for cityblock, ${pct(cheb.negative_mass)} for chebyshev. A low stress on a `
    + `non-Euclidean table is easy and means nothing.`);
  set('verdict-nonmetric',
    `It fits any increasing function of the dissimilarities, so a monotone distortion costs it `
    + `${flat ? 'exactly nothing' : 'almost nothing'}: nine different distortions of the same data `
    + `gave maps ${S.nonmetric_spread.toExponential(0)} apart, while metric scaling drifted to `
    + `${gWorst.metric_disparity}.`);
  set('verdict-nonmetric-warn',
    `That robustness is bought by throwing away the averaging that noise needs. With noise big `
    + `enough to move the rank correlation to ${noiseWorst.rank_corr}, it lands `
    + `${noiseWorst.nonmetric_disparity} from the truth against metric scaling's `
    + `${noiseWorst.metric_disparity}.`);
  set('verdict-fa',
    `It is a model, so it has a likelihood, so k can be chosen by BIC (${wine.best_k} for the wine) `
    + `and the fit can be rejected. With unequal noise it was ${unRatio.toFixed(0)} times closer to `
    + `the truth than PCA, and it is unmoved by whether you standardised `
    + `(${ctrl.scale_invariance.toExponential(0)}).`);
  set('verdict-fa-warn',
    `With equal noise on every variable it is PCA (${ctrl.equal.pca_gap} against `
    + `${ctrl.equal.fa_gap}), so the extra machinery buys nothing there. And it is an iterative fit: `
    + `on the widget's data the EM took between ${Math.min(...F.sweep.map((r) => r.fa_iters))} and `
    + `${Math.max(...F.sweep.map((r) => r.fa_iters))} iterations.`);
  set('verdict-rotation',
    `Rotating the loadings changes the implied covariance by ${wine.rot_fit_gap.toExponential(0)}, `
    + `which is to say not at all, so choosing readable axes costs nothing in fit. Varimax raised `
    + `simplicity from ${wine.simplicity_before} to ${wine.simplicity_after} here.`);
  set('verdict-rotation-warn',
    `The other side of the same coin: individual loadings moved by up to ${wine.rot_load_change}, so `
    + `any story told about a particular factor is a story about a rotation somebody chose, not about `
    + `the data.`);

  /* ----------------------------------------------------------------- the closing */
  set('closing-note',
    `Both halves of this article are about the same thing seen twice, which is that the answer `
    + `depends on what you were willing to assume before you started. Classical scaling assumes the `
    + `dissimilarities came from points in a Euclidean space, and when they did it returns PCA `
    + `exactly, and when they did not it says so in the spectrum rather than failing quietly. `
    + `Non-metric scaling assumes only the ordering, which makes a distortion invisible to it `
    + `(${S.nonmetric_spread.toExponential(0)} across nine of them) and noise expensive `
    + `(${noiseWorst.nonmetric_disparity} against ${noiseWorst.metric_disparity}). Factor analysis `
    + `assumes each variable has a private variance, which is worth a factor of `
    + `${unRatio.toFixed(0)} when that is true and nothing at all when it is not `
    + `(${ctrl.equal.pca_gap} against ${ctrl.equal.fa_gap}).`
    + `<br /><br />`
    + `And every method on this page and the last one still shares an assumption that has not been `
    + `questioned yet. Classical scaling looks for coordinates whose <span class="bold">straight `
    + `line</span> distances match the table. Metric scaling minimises a residual against `
    + `<span class="bold">straight line</span> distances in the map. Both of them, and PCA, and ICA, `
    + `and NMF, take it as given that the distance between two rows is the distance through the `
    + `space they were measured in. If the data lies on a curved sheet, that is the distance a bird `
    + `takes and not the distance along the sheet, and the two can differ by an enormous factor `
    + `while the numbers stay perfectly well behaved. The <a href="../manifolds/">next article</a> `
    + `builds a dataset where the distance along the sheet is known exactly, in closed form, and `
    + `measures what every method here does with it.`);

  set('resources-note',
    `The ${n0(W.n)} wines are the UCI wine recognition set, standardised, and the raw matrix is `
    + `asserted by the generator to be identical to the one the `
    + `<a href="../directions/">directions article</a> ships, coordinate for coordinate. The digits `
    + `are ${n0(G.n_train)} stratified MNIST images for the fit and ${n0(G.n_test)} for the test, `
    + `seed ${data.meta.seed}. References from scikit-learn ${data.meta.sklearn}: PCA, `
    + `FactorAnalysis, SMACOF and IsotonicRegression. This page's own pool adjacent violators agrees `
    + `with the library's to ${S.pav_gap.toExponential(0)}, and its EM factor analysis agrees with `
    + `FactorAnalysis on the private variances to `
    + `${Math.max(...F.sweep.map((r) => r.sk_psi_gap)).toExponential(0)} across all `
    + `${F.sds.length} settings of the slider.`);
}
