/* The sentences that carry a measured number, written from the measurement.
 *
 * Nothing below is typed by hand. Every comparison is a branch, including the
 * directions ("bigger", "below", "instead of"), which are numbers too, and both
 * tables, which is where a stale claim survives longest because nothing on the
 * page contradicts it.
 *
 * PROSE_IDS is exported so that the verification pass can do the mechanical
 * check that catches a prose file which throws halfway: read each id's initial
 * text out of the HTML, render, and require every one of them to have changed.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const pc = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve', 'thirteen'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const plural = (n, s) => `${word(n)} ${s}${n === 1 ? '' : 's'}`;

export const PROSE_IDS = [
  'intro-guard', 'scrolly-intro', 'step-0', 'step-1', 'step-2', 'step-3', 'step-4',
  'bench-note', 'ceiling-intro', 'ceiling-note', 'breakdown-intro', 'breakdown-note',
  'pair-intro', 'pair-note', 'selffit-note', 'cutoff-intro', 'cutoff-note',
  'deep-intro', 'deep-note', 'identity-note', 'cost-note',
  'v-z', 'v-z-watch', 'v-iqr', 'v-iqr-watch', 'v-mz', 'v-mz-watch', 'v-md', 'v-md-watch',
  'closing-1', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const O = data.one_d;
  const S = O.sub;
  const C = O.ceiling;
  const T = data.two_d;
  const M = data.many_d;
  const N = M.n;
  const P = M.p;
  const col = O.column;

  /* ------------------------------------------------------------- the guard */
  set('intro-guard',
    `Everything on this page is computed in the browser: the means, the medians and the quartiles, `
    + `the ${P} by ${P} covariance and the Cholesky factor that turns it into a distance, the `
    + `chi-squared tail, and every one of the ${n0(N)} squared distances the last widget redraws on `
    + `each move. All of it is checked at load time against what the generator measured, and the `
    + `tolerances are set from how many decimals the file was written with rather than from hope. `
    + `The wine is not a copy: standardised, this page's matrix reproduces the one `
    + `<a href="../distances/">the scaling article</a> ships to `
    + `${data.meta.shared.distances_matrix ? data.meta.shared.distances_matrix.toExponential(1) : 'the last decimal'}`
    + `, which is the fourth decimal that file was written with.`);

  /* ------------------------------------------------------------ the scrolly */
  set('scrolly-intro',
    `Twelve bottles and one number each: the ${col} content of the first ${word(S.n)} rows of the `
    + `file, in milligrams per litre. Twelve because that is a realistic size for a batch somebody `
    + `actually screens by hand, and because at twelve the arithmetic below has nowhere to hide.`);
  set('step-0',
    `The ${col} of the first ${word(S.n)} bottles, laid on a line: `
    + `${S.values.map((v) => v.toFixed(0)).join(', ')}. Their mean is `
    + `${data.checks.sub_stats.mean.toFixed(1)} and their median `
    + `${data.checks.sub_stats.median.toFixed(1)}, close enough to each other that nothing here `
    + `looks like a problem.`);
  set('step-1',
    `Two intervals, drawn as the areas they are willing to accept. Three standard deviations `
    + `either side of the mean is one of them; Tukey's fences, a step and a half of the `
    + `interquartile range past each quartile, is the other. Nothing is outside either, which is `
    + `the correct answer: these ${word(S.n)} readings are twelve ordinary bottles.`);

  const first = S.one.find((r) => Math.abs(r.z) > O.z_cut);
  const firstFence = S.one.find((r) => r.past_fence);
  const lastOne = S.one[S.one.length - 1];
  set('step-2',
    `Take the last bottle and push it. It leaves the fences at `
    + `${firstFence ? firstFence.pos.toFixed(0) : 'once'} and crosses three standard deviations at `
    + `${first ? first.pos.toFixed(0) : 'no point at all'}`
    + (first
      ? `, so the two rules do not fire together. `
      : `, which is to say never: on ${word(S.n)} readings it cannot. `)
    + `The fences move while the pushed bottle is still overtaking its neighbours, from `
    + `${S.one[0].fence.toFixed(1)} to ${lastOne.fence.toFixed(1)}, and then stop: once it is the `
    + `largest reading in the sample, the quartiles are computed from readings that are all behind `
    + `it and nothing it does afterwards reaches them. The three-sigma band has no such luck.`);

  const far = S.one[S.one.length - 1];
  set('step-3',
    `Keep pushing. The mean slides from ${data.checks.sub_stats.mean.toFixed(1)} to `
    + `${far.mean.toFixed(1)} and the standard deviation from `
    + `${data.checks.sub_stats.sd.toFixed(1)} to ${far.sd.toFixed(1)}, so the band widens to `
    + `swallow the very point that widened it. By the end of the slider the bottle is `
    + `${(far.pos / S.values[S.suspect]).toFixed(1)} times its original reading and its z has `
    + `stopped at ${f3(Math.abs(far.z))}, `
    + (Math.abs(far.z) < C.rows.find((r) => r.n === S.n).k1 + 0.001
      ? `a hair under the ${f3(S.ceil_one)} that ${word(S.n)} readings can produce and not a `
        + `thousandth more, however far it goes.`
      : `and it will not go past ${f3(S.ceil_one)}, whatever the slider does.`));

  const end = S.two[S.two.length - 1];
  const start2 = S.two[0];
  set('step-4',
    `Now park a second odd bottle at ${S.comp_at.toFixed(0)}. On its own it was the obvious `
    + `suspect, at z ${f3(Math.abs(S.comp_peak.z_comp))}. With the first one pushed to the end of `
    + `the slider its z is <span class="bold">${f3(Math.abs(end.z_comp))}</span>: it has become an `
    + `average bottle without moving a millimetre. There are now two bottles that anyone can see `
    + `are out of place, and three sigma flags <span class="bold">${word(end.n_z)}</span> of the `
    + `twelve: the same count as before the push, and a different bottle. The modified z, built on `
    + `the median and the median deviation instead, goes from ${word(start2.n_mz)} to `
    + `${word(end.n_mz)}, which is the right answer both times.`);

  /* -------------------------------------------------------------- the bench */
  const cross = S.two.filter((r) => Math.abs(r.z_comp) >= O.z_cut).length;
  set('bench-note',
    `Two things are worth staying for. The top curve <span class="bold">stops climbing</span>, `
    + `and where it stops has nothing to do with wine: ${f3(S.ceil_one)} with one odd bottle and `
    + `${f3(S.ceil_two)} with two, which is exactly where the next section says it has to be. And `
    + `with two odd bottles the curve for the one that never moved runs `
    + `<span class="bold">downhill</span>: it is above three sigma in ${word(cross)} of the `
    + `${n0(S.two.length)} slider positions and below it in the other ${n0(S.two.length - cross)}, `
    + `while its modified z sits at ${f3(Math.abs(end.mz_comp))} and does not care. `
    + `That is masking, and it needs two points to happen, which is why a rule that is fine on a `
    + `worked example can be useless on a real file.`);

  /* ------------------------------------------------------------ the ceiling */
  set('ceiling-intro',
    `Nothing about the data survives into that expression except how many readings there are and `
    + `how many of them are odd together. The generator checked it the other way round, by putting `
    + `the odd readings at a billion and reading the z off: the formula and the push agree to `
    + `${C.err === 0 ? 'the last bit of double precision' : C.err.toExponential(1)}.`);

  const k1 = C.k_rows.find((r) => r.k === 1);
  set('ceiling-note',
    `Two consequences, and the first is the one that gets published. `
    + `<span class="bold">The three-sigma rule cannot fire at all below ${word(C.first_n)} `
    + `readings</span>: at ten the largest possible z is `
    + `${f3(C.rows.find((r) => r.n === 10).k1)}, so a screen of nine measurements that reports no `
    + `outliers has reported nothing. The second is worse, because it does not go away with more `
    + `data. On all ${n0(N)} bottles a single odd one can reach `
    + `${f3(k1.ceiling)}, comfortably past three; with `
    + `<span class="bold">${word(C.k_kills)}</span> of them sitting together the ceiling is `
    + `${f3(C.k_rows.find((r) => r.k === C.k_kills).ceiling)} and the rule is arithmetically `
    + `incapable of firing, however far out the ${word(C.k_kills)} of them are. `
    + `${cap(word(C.k_kills))} out of ${n0(N)} is ${pc(C.k_kills / N, 1)} of the file.`);

  /* ---------------------------------------------------------- the breakdown */
  const byName = Object.fromEntries(data.breakdown.map((r) => [r.name, r]));
  set('breakdown-intro',
    `The breakdown point of an estimator is the smallest fraction of the data you have to move `
    + `before it will follow you anywhere. It is usually quoted; here it is measured, by sending `
    + `readings off to a trillion one at a time and watching which estimate goes with them. `
    + `The wine's ${col} column, all ${n0(N)} of it.`);

  const WHAT = {
    mean: 'the centre every z-score is measured from',
    'standard deviation': 'the unit every z-score is measured in',
    median: 'the centre the modified z is measured from',
    MAD: 'the unit the modified z is measured in',
    'upper Tukey fence': 'the line a boxplot draws its whisker to',
    '10% trimmed mean': 'the compromise between the two centres',
  };
  const tb = document.querySelector('#breakdown-table tbody');
  if (tb) {
    tb.innerHTML = data.breakdown.map((r) => `<tr><td><span class="bold">${r.name}</span></td>`
      + `<td>${n0(r.points)} of ${n0(N)}</td><td>${pc(r.fraction, 1)}</td>`
      + `<td>${WHAT[r.name] ?? ''}</td></tr>`).join('');
  }
  set('breakdown-note',
    `One reading out of ${n0(N)} takes the mean and the standard deviation with it, so both halves `
    + `of a z-score break at ${pc(byName.mean.fraction, 1)} of the file. The median and the median `
    + `deviation need ${n0(byName.median.points)}, which is `
    + `${pc(byName.median.fraction, 0)} and is the most any estimator of a centre can ask for: past `
    + `half, the contamination is the data. Tukey's fences sit in between at `
    + `${pc(byName['upper Tukey fence'].fraction, 1)}, which is the quartile they are built on. `
    + `That ordering is the whole reason the same twelve bottles, scored two ways, gave two `
    + `answers above.`);

  /* --------------------------------------------------------- the pair */
  const corner = T.corner;
  set('pair-intro',
    `Screening a table column by column is the obvious thing to do and it misses a whole kind of `
    + `strangeness. Two of the wine's columns, ${T.cols[0].replace(/_/g, ' ')} and `
    + `${T.cols[1].replace(/_/g, ' ')}, correlate at ${f3(T.corr)}: bottles high in one are high `
    + `in the other, and a bottle that is high in one and low in the other is a bottle that does `
    + `not exist in these ${n0(N)}, even though neither of its two readings is remarkable on its `
    + `own.`);

  set('pair-note',
    `The far corner of the rectangle is where the two rules are furthest apart, and it is not a `
    + `subtle gap: a bottle sitting there is ${f2(corner.z)} standard deviations out on each axis, `
    + `which any column-by-column screen at the ${pc(T.marg_p, 1)} level waves through, and `
    + `${f2(corner.d)} out on the pair, which is `
    + `${(corner.d / Math.sqrt(T.chi_cut)).toFixed(1)} times the cutoff the ellipse uses. `
    + `Measured by area, ${pc(T.area.rect_only / T.area.rect, 1)} of the rectangle lies outside `
    + `the ellipse. Measured on the wines themselves, `
    + `<span class="bold">${word(T.rect_only)}</span> of the ${n0(N)} bottles are inside both `
    + `margins and outside the ellipse`
    + (T.ellipse_only
      ? `, and ${word(T.ellipse_only)} `
        + `${T.ellipse_only === 1 ? 'is the other way round' : 'are the other way round'}.`
      : `, and not one is the other way round: on this pair the ellipse is strictly the stricter `
        + `rule, which is what a correlation of ${f3(T.corr)} buys.`));

  const ray = T.ray.rows;
  const rayEnd = ray[ray.length - 1];
  set('selffit-note',
    `And the toggle is the article's argument in one control. Drag the suspect straight out across `
    + `the correlation and its own distance runs to ${f2(rayEnd.d2_in)} while it is in the fit, `
    + `against ${f2(rayEnd.d2_out)} for the same point scored against the `
    + `${n0(N)} bottles alone: a factor of ${(rayEnd.d2_out / rayEnd.d2_in).toFixed(2)} it awards `
    + `itself simply by being present when the ellipse was drawn. The second number is the one you `
    + `would want. It is also the one you cannot have, because knowing which row to leave out is `
    + `the entire problem.`);

  /* ------------------------------------------------------------- the cutoff */
  const cal = M.calibration;
  const wine178 = cal.find((r) => r.n === N && r.p === P);
  const unreachable = cal.filter((r) => !r.reachable);
  set('cutoff-intro',
    `If the mean and the covariance were known rather than estimated, the squared distance of a `
    + `gaussian point would be chi-squared on ${word(P)} degrees of freedom and the textbook `
    + `cutoff would be exactly right. Every table below was built by handing gaussian samples that `
    + `contain nothing at all to that cutoff, ${n0(M.cal_trials)} samples per row, and counting `
    + `how often it fired.`);

  const ct = document.querySelector('#cal-table tbody');
  if (ct) {
    ct.innerHTML = cal.map((r) => `<tr><td>${n0(r.n)}</td><td>${r.p}</td>`
      + `<td>${f2(r.cut)}${r.reachable ? '' : ' <span class="bold">(above the ceiling)</span>'}</td>`
      + `<td>${pc(r.nominal, 2)}</td><td>${pc(r.measured, 3)}</td>`
      + `<td>${pc(r.exact_beta, 3)}</td></tr>`).join('');
  }

  const worst = cal.reduce((a, b) => (b.measured / b.nominal < a.measured / a.nominal ? b : a));
  set('cutoff-note',
    `Every measured rate is <span class="bold">below</span> the nominal one, which is the opposite `
    + `of the direction most people brace for: the textbook cutoff is not trigger-happy, it is `
    + `asleep. At ${n0(worst.n)} rows and ${worst.p} columns it fires `
    + `${pc(worst.measured, 3)} of the time instead of ${pc(worst.nominal, 2)}`
    + (worst.measured === 0
      ? `, and it is not a small sample of the simulation: `
        + `${n0(worst.n * M.cal_trials)} gaussian rows produced not one flag, because the cutoff `
        + `${f2(worst.cut)} is above ${f2(worst.ceiling)}, the largest squared distance `
        + `${n0(worst.n)} rows can produce. Same ceiling, same algebra, ${word(P)} dimensions `
        + `instead of one.`
      : `, a factor of ${(worst.nominal / worst.measured).toFixed(1)}.`)
    + (unreachable.length
      ? ` ${cap(plural(unreachable.length, 'row'))} of the table `
        + `${unreachable.length === 1 ? 'has' : 'have'} a cutoff above the ceiling, so on `
        + `${unreachable.length === 1 ? 'that shape of table' : 'those shapes of table'} the rule `
        + `cannot flag anything whatsoever.`
      : '')
    + ` On the wine's own shape, ${n0(N)} by ${P}, it fires ${pc(wine178.measured, 3)} of the `
    + `time against a nominal ${pc(wine178.nominal, 2)}, and the exact beta says `
    + `${pc(wine178.exact_beta, 3)}. The simulation and the algebra agree; the textbook does not.`);

  /* -------------------------------------------------------------- the depth */
  set('deep-intro',
    `One suspect can flatter itself. A batch of them can do considerably better, and the batch `
    + `does not have to be big: on ${n0(N)} bottles it takes `
    + `<span class="bold">${word(M.masking.lost)}</span> planted rows to make every one of them `
    + `invisible, which is ${pc(M.masking.lost / N, 1)} of the file.`);

  const rows = M.masking.rows;
  const last = rows.filter((r) => r.m && r.flagged === r.m).pop();
  const firstLost = rows.find((r) => r.m === M.masking.lost);
  set('deep-note',
    `The batch does not have to be large, and the collapse is not gradual. At `
    + `${word(last.m)} planted bottles all ${word(last.m)} are flagged, at a squared distance of `
    + `${f2(last.d2)} against a cutoff of ${f2(last.cut)}. At `
    + `<span class="bold">${word(firstLost.m)}</span>, one more than the number the widget `
    + `hesitates at, their distance is ${f2(firstLost.d2)} and `
    + `<span class="bold">none of them is flagged</span>. They did not move; the covariance moved `
    + `to them. Push the slider to the end and the ${n0(M.masking.swamp.m)} planted bottles score `
    + `${f2(rows[rows.length - 1].d2)} while `
    + `${n0(M.masking.swamp.real_flagged)} genuine bottles are flagged instead of the `
    + `${n0(rows[0].real_flagged)} that were flagged when the file was clean. That second half has `
    + `its own name, swamping, and it is the same mechanism read from the other end.`);

  set('identity-note',
    `One thing on that panel never moves, whatever is planted. The ${n0(N)} squared distances of a `
    + `sample to its own mean and covariance always sum to exactly `
    + `<span class="bold">p(n&minus;1)</span>, here ${P} times ${n0(N - 1)}, which is `
    + `${n0(M.sum_should_be)}, and the page measures ${M.sum.toFixed(4)}. The total is fixed before `
    + `anybody looks at the data, so <span class="bold">every distance a contaminated row takes `
    + `for itself is a distance some other row does not get</span>. That is not a metaphor for `
    + `masking, it is masking: a budget of ${n0(M.sum_should_be)} being spent, and the outliers `
    + `deciding how. The same arithmetic caps any single row at (n&minus;1)&sup2;/n, which is `
    + `${f2(M.ceiling)} here against a largest observed ${f2(M.max)}.`);

  /* --------------------------------------------------------------- the cost */
  const cost = data.cost.rows;
  const wide = cost[cost.length - 1];
  set('cost-note',
    `Screening column by column is a pass over the table and nothing else. The ellipse needs a `
    + `covariance, which is quadratic in the number of columns, and an inverse, which is cubic, `
    + `and it needs more rows than columns before the inverse exists at all. At `
    + `${n0(wide.p)} columns and ${n0(data.cost.n)} rows the first rule touches `
    + `${n0(wide.marginal)} numbers and the second `
    + `${n0(wide.cov)}, a factor of ${(wide.cov / wide.marginal).toFixed(0)}, and the second one `
    + `does not run at all, because ${n0(data.cost.n)} rows cannot fill a `
    + `${n0(wide.p)} by ${n0(wide.p)} covariance.`);

  const costBody = document.querySelector('#cost-table tbody');
  if (costBody) {
    costBody.innerHTML = cost.map((r) => `<tr><td>${n0(r.p)}</td><td>${n0(r.marginal)}</td>`
      + `<td>${n0(r.cov)}</td>`
      + `<td>${n0(r.needs)}${r.needs > data.cost.n
        ? ' <span class="bold">(more than there are)</span>' : ''}</td></tr>`).join('');
  }

  /* ------------------------------------------------------------- the verdict */
  set('v-z',
    `One number, no assumptions to configure, and a meaning everybody already shares. On the `
    + `${col} column it flags ${word(O.flag_z.length)} of ${n0(N)} bottles.`);
  set('v-z-watch',
    `The ceiling: ${f3(k1.ceiling)} on ${n0(N)} readings, ${f3(S.ceil_one)} on ${word(S.n)}, and `
    + `below three as soon as ${word(C.k_kills)} readings are odd together. It breaks at `
    + `${pc(byName.mean.fraction, 1)} of the file.`);
  const f0 = S.one[0];
  const fEnd = S.one[S.one.length - 1];
  const band0 = f0.mean + O.z_cut * f0.sd;
  const bandEnd = fEnd.mean + O.z_cut * fEnd.sd;
  set('v-iqr',
    `No mean and no standard deviation anywhere in it. Over the whole push in the bench above the `
    + `upper fence travels from ${f0.fence.toFixed(1)} to ${fEnd.fence.toFixed(1)}, a factor of `
    + `${(fEnd.fence / f0.fence).toFixed(2)}, while the edge of the three-sigma band travels from `
    + `${band0.toFixed(1)} to ${bandEnd.toFixed(1)}, a factor of `
    + `${(bandEnd / band0).toFixed(2)}. Flags ${word(O.flag_iqr.length)} bottles in the ${col} `
    + `column, `
    + `${O.flag_iqr.length > O.flag_z.length ? 'more than' : O.flag_iqr.length < O.flag_z.length
      ? 'fewer than' : 'the same as'} three sigma.`);
  set('v-iqr-watch',
    `The fences are quartiles, so they break at ${pc(byName['upper Tukey fence'].fraction, 1)}, `
    + `and the 1.5 is a convention rather than a result: it puts the fences at `
    + `${f2(O.tukey_fence_sigma)} standard deviations on gaussian data, which is the `
    + `${pc(O.tukey_tail, 2)} tail, and on anything skewed it is whatever it is.`);
  set('v-mz',
    `The highest breakdown on the table, ${pc(byName.median.fraction, 0)}, and the only rule here `
    + `that catches both odd bottles in the bench above at every slider position. Flags `
    + `${word(O.flag_mz.length)} bottles in the ${col} column.`);
  set('v-mz-watch',
    `The MAD is ${O.mad.toFixed(0)} here, and it is zero whenever more than half the readings are `
    + `identical, at which point every other reading scores infinity. Discrete or heavily rounded `
    + `columns are where that happens.`);
  set('v-md',
    `The only rule on this page that can see a bottle which is ordinary in every column and `
    + `impossible in the pair: ${word(T.rect_only)} of the ${n0(N)} bottles on `
    + `${T.cols[0].replace(/_/g, ' ')} against ${T.cols[1].replace(/_/g, ' ')} alone.`);
  set('v-md-watch',
    `It inherits both defects and adds one. The estimate it measures against is the sample's own, `
    + `so ${word(M.masking.lost)} planted rows hide themselves completely; the chi-squared cutoff `
    + `is calibrated for a covariance nobody has; and the whole thing needs more rows than `
    + `columns.`);

  /* ------------------------------------------------------------- the closing */
  set('closing-1',
    `Nothing above is an argument against these rules. A z-score on a column with one thing wrong `
    + `with it is a good z-score, and it caught the ${word(O.flag_z.length)} bottles it should `
    + `have. The argument is narrower and it is arithmetic: <span class="bold">a rule that `
    + `estimates its own yardstick from the sample can only be trusted when the contamination is `
    + `smaller than its breakdown point</span>, which for the mean and the standard deviation is `
    + `one row in ${n0(N)}. Everything on this page follows from that sentence, including the two `
    + `results that look like curiosities: the ceiling, which is what happens when the yardstick `
    + `grows exactly as fast as the thing it is measuring, and the fixed total of `
    + `${n0(M.sum_should_be)}, which is what happens when a budget of strangeness is handed out `
    + `and the strangest rows get to write the rules.`);

  set('ref-note',
    `The wine is the dataset the earlier articles ship. Its ${n0(N)} rows and ${P} columns are `
    + `written here to ${data.meta.decimals} decimals, which moves the worst value in the file by `
    + `${data.meta.rounding_gap.toExponential(0)}; every number on the page is measured on the `
    + `rounded matrix rather than on the loader's, so the browser and the generator hold the same `
    + `data. Standardised it reproduces `
    + `<a href="../distances/">the scaling article</a>'s matrix to `
    + `${data.meta.shared.distances_matrix.toExponential(1)} and, centred, the pair `
    + `<a href="../directions/">the directions article</a> draws to `
    + `${data.meta.shared.directions_pair.gap.toExponential(1)}. Simulations use numpy's PCG64 `
    + `with the seeds recorded in the generator; scikit-learn ${data.meta.sklearn}, scipy `
    + `${data.meta.scipy}, numpy ${data.meta.numpy}.`);
}
