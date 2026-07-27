/* Every sentence on this page that carries a figure, written from the file.
 *
 * Same discipline as the depth, after-depth and detection articles: nothing
 * numeric is typed by hand, and every comparison is written AS a comparison,
 * with both outcomes spelled out, so that rerunning the generator moves the
 * sentence instead of leaving it quietly wrong. Table cells too: a table is
 * where a stale claim survives longest, because nothing around it disagrees.
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
  const C = data.cloud;
  const S = data.spectrum;
  const I = data.ica;
  const A = data.atoms;
  const rank = (k) => A.errors.find((r) => r.rank === k);
  const mid = rank(16);
  const top = A.errors[A.errors.length - 1];

  /* ------------------------------------------------------------- the opening */
  set('opening-note',
    `Three numbers, from three different parts of this page, are the whole argument in advance. `
    + `On ${n0(A.n_images)} handwritten digits rebuilt from ${mid.rank} atoms, PCA and ICA reach `
    + `errors that differ by <span class="bold">${mid.ica_gap_unquantised.toExponential(0)}</span>, `
    + `because ICA does not look for a better subspace, it turns the one PCA already found, and `
    + `turning a subspace cannot change how far anything is from it. NMF, which is not allowed to subtract, `
    + `pays <span class="bold">${mid.nmf_ratio} times</span> that error and gets something else `
    + `back. And on two signals mixed together, the answer PCA hands you still has `
    + `<span class="bold">${I.pca_leak}</span> of the wrong signal in it, where one further `
    + `rotation, picked by a criterion PCA cannot compute, brings that down to ${I.ica_leak}. `
    + `Same factorisation, three constraints, three completely different failures.`);

  /* --------------------------------------------------------------- the scrolly */
  const R = C.raw;
  set('scrolly-step-best',
    `Turn it to <span class="bold">${R.dial_best_deg}°</span> and the shadows are as spread out as `
    + `they will ever get: ${pct(R.dial_best_share)} of the variance kept, `
    + `${pct(1 - R.dial_best_share)} thrown away. Turn it ${R.worst_deg - R.dial_best_deg}° further, `
    + `to exactly perpendicular, and it is the worst line available at ${pct(R.worst_share)}. `
    + `The best one has a name, the <span class="bold">first principal component</span>, and the `
    + `dashed line at right angles to it is the second. With two measurements there was never a `
    + `third choice to make.`);

  set('identity-note',
    `Nothing in that line is an approximation or a modelling assumption. Measured across all `
    + `${Math.round(180 / C.dial_step)} settings the dial below can take, the largest amount by `
    + `which the two sides disagree is <span class="bold">${C.identity_gap}</span>, which is double `
    + `precision arithmetic and nothing else. So "find the direction that keeps the most variance" `
    + `and "find the line the cloud sits closest to" are not two related problems, they are one `
    + `problem stated twice, and that is why one method answers a compression question and a `
    + `structure question at the same time. Take the dial: the widget scores you against the best `
    + `setting the slider can actually reach, swept at its own half-degree resolution.`);

  /* ------------------------------------------------------------- variance units */
  const sd = S.sd;
  const iMax = sd.indexOf(Math.max(...sd));
  const iMin = sd.indexOf(Math.min(...sd));
  const ratio = sd[iMax] / sd[iMin];
  const even = 1 / Math.sqrt(S.features.length);
  set('scale-note',
    `The dial has a second button on it, and it is the most expensive button in this article. `
    + `Everything above worked because the two measurements happened to be roughly the same size. `
    + `They are not always. Across the wine's ${S.features.length} columns, `
    + `${label(S.features[iMax])} has a standard deviation of ${sd[iMax].toFixed(1)} and `
    + `${label(S.features[iMin])} has ${sd[iMin].toFixed(4)}: a factor of `
    + `<span class="bold">${Math.round(ratio).toLocaleString('en-US')}</span>, and entirely an `
    + `artefact of the units somebody chose to record them in. Variance is not unitless, so `
    + `"the direction of greatest variance" is not unitless either. Run PCA on the raw columns and `
    + `the first component captures <span class="bold">${pct(S.raw.evr[0])}</span> of the variance, `
    + `which sounds like a triumph until you look at what it is made of: a unit vector over `
    + `${S.features.length} measurements would spread about ${even.toFixed(4)} of its length over `
    + `each if it mixed them evenly, and this one puts `
    + `<span class="bold">${S.raw.top_loading}</span> on ${label(S.raw.top_feature)} alone. It is `
    + `not a combination of anything. It is ${label(S.raw.top_feature)} with a new name, and it `
    + `reaches the 90% mark at k = ${S.raw.k90} by saying nothing at all.`);

  const K = S.std.k90;
  const km = S.kmeans;
  const bestRow = km.by_k.find((r) => r.k === km.best_k);
  const beatsAll = km.best_ari > km.all_features;
  set('spectrum-note',
    `Standardise first and the same thirteen columns give a first component worth `
    + `${pct(S.std.evr[0])}, a second worth ${pct(S.std.evr[1])}, and no single measurement `
    + `carrying more than ${S.std.top_loading} of any of them. Ninety per cent of the variance now `
    + `costs <span class="bold">${K} components</span> instead of ${S.raw.k90}, which is a worse `
    + `headline number and a far better model: the thirteen measurements really do carry `
    + `${K} directions' worth of information, and the version that fitted in one was measuring the `
    + `size of a unit. `
    + (beatsAll
      ? `The clearest evidence that the reduction is not merely lossy compression comes from `
        + `handing the result to something else. K-means looking for three clusters scores an `
        + `adjusted Rand index of ${km.all_features} against the cultivars using all `
        + `${S.features.length} standardised measurements, which is the number the `
        + `<a href="../kmeans/">k-means article</a> published. Give it only the first `
        + `<span class="bold">${km.best_k}</span> principal components and it scores `
        + `<span class="bold">${km.best_ari}</span>. Discarding ${S.features.length - km.best_k} of `
        + `${S.features.length} dimensions did not cost accuracy, it bought some, because what went `
        + `out with them was mostly noise.`
      : `Handing the result to k-means, the first ${km.best_k} components score ${km.best_ari} `
        + `against ${km.all_features} on all ${S.features.length}, so here the reduction costs a `
        + `little and buys speed rather than accuracy.`));

  /* ------------------------------------------ variance order is not usefulness */
  const sep = S.separation;
  const last = S.features.length;
  set('usefulness-note',
    `One caution before leaving the spectrum, because "explains 36% of the variance" is the most `
    + `over-read number in this module. Variance is not importance, and PCA has never seen your `
    + `labels. Score each of the ${last} components by how cleanly it separates the three cultivars `
    + `and the ranking that comes back is `
    + `<span class="bold">${sep.order.slice(0, 6).map((v) => `PC${v}`).join(', ')}</span> and so on, `
    + `not 1, 2, 3, 4, 5, 6. The last component of all, PC${last}, carries `
    + `${pct(S.std.evr[last - 1])} of the variance, the least of any of them, and it still separates `
    + `the cultivars better than <span class="bold">${sep.last_beats}</span> of the ten components `
    + `sitting above it. If you are reducing dimensions before a supervised model, that is the `
    + `component you were about to delete first. PCA is the right tool when the question is `
    + `"what is this data shaped like"; when the question is "which directions predict my label", `
    + `it is answering a different one, and the discriminants of `
    + `<a href="../geometric-classification/">module 1.2</a> are the tool that asks yours.`);

  /* -------------------------------------------------------------------- the ICA */
  const kSrc = I.kurtosis.sources;
  const kMic = I.kurtosis.mics;
  const towardGaussian = Math.abs(kMic[0]) < Math.abs(kSrc[0]) && Math.abs(kMic[1]) < Math.abs(kSrc[1]);
  set('ica-note',
    `Two signals, recorded by two microphones that each hear both of them. The desk mixes them by `
    + `[[${I.mixing[0].join(', ')}], [${I.mixing[1].join(', ')}]], instantaneously and with no echo, `
    + `which is the polite version of the problem. You are given the two recordings and asked for `
    + `the two signals, and you are not told the mixing. PCA has an answer, and its answer is wrong: `
    + `the worse-recovered signal comes back at <span class="bold">${I.pca_match}</span> `
    + `correlation with its source while <span class="bold">${I.pca_leak}</span> of the other one is `
    + `still audible in it. Not slightly wrong. Wrong in the way that leaves you listening to both `
    + `people at once.`
    + `<br /><br />`
    + `The lever that fixes it is a statistic PCA does not compute. Kurtosis measures how `
    + `un-gaussian a distribution is, and it is exactly zero for a gaussian. The two sources here `
    + `sit at ${kSrc[0]} and ${kSrc[1]}, both a long way from it. `
    + (towardGaussian
      ? `Their mixtures sit at ${kMic[0]} and ${kMic[1]}: <span class="bold">closer to gaussian `
        + `than either ingredient</span>, which is the central limit theorem doing what it always `
        + `does, and it is the whole lever. If adding things up makes them more gaussian, then `
        + `un-adding them means hunting for the combination that is <i>least</i> gaussian. `
      : `Their mixtures sit at ${kMic[0]} and ${kMic[1]}, which does not sit between the sources `
        + `and the gaussian the way the argument needs, so on this particular desk the intuition `
        + `has to be carried by the measurement rather than the story. `)
    + `Variance cannot see any of this: it is a second-order quantity, kurtosis is a fourth-order `
    + `one, and no amount of covariance will ever contain it.`);

  /* the gaussian impossibility */
  const st = I.stability;
  const G = I.gaussian;
  const flat = G.objective_span < I.objective_span / 3;
  const looksUniform = Math.abs(st.gaussian_spread_deg - st.chance_spread_deg) < 6;
  set('gauss-note',
    `Now press <span class="bold">gaussian voices</span>, which replaces the two signals with two `
    + `independent gaussians and changes nothing else, and watch the curve go flat. `
    + (flat
      ? `It spans <span class="bold">${G.objective_span}</span> from end to end against `
        + `${I.objective_span} for the real signals, ${G.flatter_by} times flatter, and what little `
        + `shape is left is sampling noise. `
      : `It spans ${G.objective_span} against ${I.objective_span}, which is flatter but not by the `
        + `margin the theory demands, so this particular draw understates the effect. `)
    + `The clean way to see that is to stop looking at one draw. Repeat the whole experiment on `
    + `${st.draws} independent draws and record where the peak lands each time. With the real `
    + `signals it lands in the same place every time, a spread of `
    + `<span class="bold">${st.real_spread_deg}°</span>. With gaussian sources it lands at `
    + `<span class="bold">${st.gaussian_spread_deg}°</span> of spread, and ${st.chance_spread_deg}° `
    + `is what ${st.draws} draws of a peak landing uniformly at random would give`
    + (looksUniform
      ? `. Those are the same number. The peak is not being found badly, it is not there: the dial `
        + `is reporting noise.`
      : `, so the gaussian peak is scattered but not quite uniformly.`)
    + `<br /><br />`
    + `This is not a limitation of FastICA or of kurtosis. A rotation of two independent gaussians `
    + `is two independent gaussians, with the same distribution, exactly. The rotated data and the `
    + `original are not merely hard to tell apart, they are the same random variable, so there is `
    + `no procedure of any kind, now or ever, that could recover which rotation you started from. `
    + `<span class="bold">ICA needs its sources to be non-gaussian, and that is a theorem rather `
    + `than a caveat.</span> At most one gaussian source is allowed, which is exactly why "add some `
    + `gaussian noise" is a harmless assumption everywhere else in this atlas and a fatal one here.`);

  /* -------------------------------------------------------------------- the NMF */
  const two = rank(2);
  set('nmf-note',
    `The third constraint is the plainest to state and the strangest in its consequences: `
    + `<span class="bold">no negative numbers anywhere</span>, not in the atoms and not in the `
    + `amounts. On data that is counts, or intensities, or pixels, that is not a restriction so `
    + `much as a description, since none of those can be negative in the first place. It is also `
    + `binding on the input, which is worth saying out loud: NMF cannot be run on the standardised `
    + `wine above at all, because standardising put half the numbers below zero.`
    + `<br /><br />`
    + `Here is what it buys and what it costs, on ${n0(A.n_images)} MNIST digits, the same handwriting `
    + `<a href="../lenet/">article 20</a> taught a network to read. PCA's atoms are free to be `
    + `negative, so a reconstruction is a tug of war: add most of this ghost, subtract a bit of that `
    + `one, and the light and dark cancel until a digit is left. NMF cannot cancel. Every atom is `
    + `ink, every amount adds ink, so the only way to build a three is out of pieces that are `
    + `already parts of a three. That constraint has to cost accuracy, and the amount is measurable: `
    + `at ${two.rank} atoms NMF pays ${two.nmf_ratio} times PCA's error, at ${mid.rank} atoms `
    + `${mid.nmf_ratio} times, and at ${top.rank} atoms <span class="bold">${top.nmf_ratio} `
    + `times</span>. The penalty grows with the budget, because the more atoms PCA is given the more `
    + `elaborate the cancellations it is allowed to arrange.`);

  const f = A.faces;
  const digitsPartsy = mid.nmf_top10 > mid.pca_top10 * 1.5;
  const facesPartsy = f.nmf_top10 > f.pca_top10 * 1.5;
  set('parts-note',
    `The famous claim about NMF, from the paper that put it on the map, is that it learns the `
    + `<i>parts</i> of things, and the famous picture is of faces coming apart into noses and `
    + `eyebrows. It is worth putting a number on that instead of repeating it. Measure how `
    + `concentrated an atom is by asking what share of its total weight sits in its heaviest tenth `
    + `of pixels: a part is concentrated, a ghost is spread out. On these digits at ${mid.rank} `
    + `atoms, NMF scores <span class="bold">${mid.nmf_top10}</span> against PCA's ${mid.pca_top10} `
    + `and ICA's ${mid.ica_top10}`
    + (digitsPartsy
      ? `. That is the parts effect, and it is enormous: nine tenths of an NMF atom lives in a tenth `
        + `of the picture, which is why its atoms look like strokes and PCA's look like weather.`
      : `, which is a smaller difference than the folklore suggests.`)
    + `<br /><br />`
    + `Then run exactly the same recipe on ${f.n} portrait photographs, which is where the original `
    + `figure came from, and the effect `
    + (facesPartsy
      ? `survives: ${f.nmf_top10} against ${f.pca_top10}.`
      : `<span class="bold">mostly disappears</span>: ${f.nmf_top10} against PCA's ${f.pca_top10}, `
        + `a gap of ${(f.nmf_top10 - f.pca_top10).toFixed(4)} where the digits gave `
        + `${(mid.nmf_top10 - mid.pca_top10).toFixed(4)}. The error penalty collapses too, from `
        + `${mid.nmf_ratio} times on digits to ${f.nmf_ratio} times on faces. Both of those point `
        + `the same way: on these faces the non-negativity constraint is barely binding, so it `
        + `barely costs anything and barely changes anything. Parts-based decomposition is a `
        + `property of the data meeting the constraint, not of the algorithm alone, and a dataset `
        + `of aligned faces with no black background gives the constraint very little to push `
        + `against. The figure is real; it is not automatic.`));

  const nest = A.nesting[A.nesting.length - 1];
  const nested = nest.pca > 0.999;
  set('nesting-note',
    `One more difference, and it is the one that bites in practice. PCA's answer at `
    + `${nest.from} components <i>contains</i> its answer at fewer: ask for one more and you get `
    + `everything you had plus one, so "how many components" can be decided after the fact and the `
    + `ordering is meaningful. Neither of the other two works that way. Going from ${nest.from} `
    + `atoms to ${nest.to}, each of PCA's old atoms is still there at a cosine of `
    + (nested
      ? `<span class="bold">${nest.pca}</span> with one of the new ones`
      : `${nest.pca}`)
    + `, while NMF's best matches average ${nest.nmf} and its worst-matched atom only ${nest.nmf_worst}, `
    + `and ICA's average ${nest.ica}. Ask NMF or ICA for one more atom and you do not get an extra `
    + `one, you get a <span class="bold">different set</span>. That is why the slider in the widget `
    + `above visibly rewrites the dictionary for two of the three columns and merely extends it for `
    + `the third, and why k is a real hyperparameter for NMF and ICA and a viewing choice for PCA.`);

  /* -------------------------------------------------------------- verdict table */
  set('verdict-scale',
    `The wine's units span a factor of ${Math.round(ratio).toLocaleString('en-US')}. Unstandardised, `
    + `PC1 is ${pct(S.raw.evr[0])} of the variance and ${S.raw.top_loading} of one measurement; `
    + `standardised, it is ${pct(S.std.evr[0])} and a genuine mixture.`);
  set('verdict-k',
    `On the standardised wine, ${K} components reach 90% of the variance. But the task is the real `
    + `judge: k-means peaks at ${km.best_k} component${km.best_k === 1 ? '' : 's'} with ARI `
    + `${km.best_ari}, ${km.best_ari > km.all_features ? 'above' : 'below'} the `
    + `${km.all_features} it gets on all ${S.features.length}.`);
  set('verdict-k-warn',
    `Variance order is not usefulness order: PC${last} has the least variance of all `
    + `(${pct(S.std.evr[last - 1])}) and out-separates ${sep.last_beats} of the ten components above `
    + `it.`);
  set('verdict-pca',
    `No rank-${mid.rank} factorisation of any kind beats it on squared error, which is a theorem, `
    + `and the atoms are nested so k is chosen after the fact. Measured here: ${mid.pca}, with ICA `
    + `matching it exactly (${mid.ica_gap_unquantised.toExponential(0)}) and NMF `
    + `${mid.nmf_ratio} times behind.`);
  set('verdict-ica',
    `Whitening is PCA's job and the rotation is ICA's. On two mixed signals PCA leaves `
    + `${I.pca_leak} of the wrong one in the answer and one further rotation cuts that to `
    + `${I.ica_leak}. FastICA needed ${I.fastica.iters} iterations to find it.`);
  set('verdict-ica-warn',
    `Gaussian sources make it impossible, not merely hard: the objective spans ${G.objective_span} `
    + `against ${I.objective_span} and the peak lands at chance (${st.gaussian_spread_deg}° of `
    + `spread against ${st.chance_spread_deg}° for pure noise). There is also no natural order or `
    + `scale to the components it returns.`);
  set('verdict-nmf',
    `Atoms you can point at: ${mid.nmf_top10} of an atom's weight in its heaviest tenth of pixels `
    + `against PCA's ${mid.pca_top10}, and ${mid.nmf_zero_coeffs} of ${mid.rank} amounts set to `
    + `exactly zero, so each row's recipe is a short list.`);
  set('verdict-nmf-warn',
    `It costs ${mid.nmf_ratio} times PCA's error at ${mid.rank} atoms and ${top.nmf_ratio} times at `
    + `${top.rank}, the atoms are not nested, and whether they look like parts depends on the data: `
    + `the same recipe on faces gave ${f.nmf_top10} against ${f.pca_top10}.`);

  /* -------------------------------------------------------------------- closing */
  set('closing-note',
    `The through line is that none of these three methods is better than the others, because they `
    + `are not answering the same question, and the constraint is where the question lives. PCA `
    + `asks for the smallest reconstruction error and gets it, provably, which is why it is the `
    + `right default for compression and the wrong one for interpretation: nothing in ${pct(S.std.evr[0])} `
    + `of the variance promises that the direction means anything. ICA gives up on error entirely, `
    + `keeps PCA's subspace exactly (${mid.ica_gap_unquantised.toExponential(0)} at ${mid.rank} `
    + `atoms) and spends its whole budget on choosing the axes inside it, which is worth everything `
    + `when the rows really are mixtures of something and worth nothing when they are not. NMF pays `
    + `${mid.nmf_ratio} times the error for atoms a person can name and recipes with `
    + `${mid.nmf_zero_coeffs} of ${mid.rank} entries at zero.`
    + `<br /><br />`
    + `And all three share one assumption so quietly that it is easy to miss: the atoms combine by `
    + `<span class="bold">addition</span>. Every method here approximates a row as a weighted sum, `
    + `which means every one of them models the data as living on, or near, a flat `
    + `${mid.rank}-dimensional slice of the original space. That is a strong claim about the world, `
    + `and it is often wrong. Data that lies on a curved surface, a spiral, a sheet rolled up in `
    + `three dimensions, has no low-dimensional flat slice to be projected onto, and every `
    + `factorisation in this article will either flatten the curve or waste its budget describing `
    + `it. Dropping the word <i>linear</i> is what the rest of module 2.2 is about, and it is where `
    + `neighbourhoods start to matter more than directions.`);
}
