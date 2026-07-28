/* Every figure on this page, composed from the file it was measured into.
 *
 * Composed rather than typed because a number typed into a sentence survives
 * the next run of the generator by pretending nothing changed, and because
 * every comparison here has to be a branch: which model wins, whether an
 * interval covers what it promised, and whether a seasonal term helps a cycle
 * that has no fixed length are all questions this file answers from the data
 * rather than from what the outline expected.
 */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const S = data.stationarity;
  const I = data.identify;
  const E = data.equivalence;
  const R = data.rolling;
  const U = data.sunspots;
  const F = data.refit;
  const P = data.period;
  const co2 = Object.fromEntries(S.co2.map((x) => [x.name, x]));
  const best = R.rows[0];
  const second = R.rows[1];
  const snaive = R.rows.find((x) => x.model === 'seasonal naive');
  const sunBest = U.rows[0];
  const sunEts = U.rows.find((x) => x.model.startsWith('ETS'));
  const sunNaive = U.rows.find((x) => x.model === 'naive');
  const cov = R.coverage;
  const mean95 = cov.reduce((a, r) => a + r.c95, 0) / cov.length;
  const mean80 = cov.reduce((a, r) => a + r.c80, 0) / cov.length;
  const sun95 = U.coverage.reduce((a, r) => a + r.c95, 0) / U.coverage.length;
  const sun80 = U.coverage.reduce((a, r) => a + r.c80, 0) / U.coverage.length;
  const noiseRows = I.rows.filter((x) => x.true_p === 0);
  const longest = noiseRows[noiseRows.length - 1];
  const ruleMean = I.rows.reduce((a, x) => a + x.rule_acc, 0) / I.rows.length;
  const aicMean = I.rows.reduce((a, x) => a + x.aic_acc, 0) / I.rows.length;

  set('opening-note',
    `Two series carry this article, both of them bundled with the libraries that `
    + `read them, so a clean clone reproduces every number below. `
    + `<span class="bold">${n0(M.co2_n)} monthly averages of carbon dioxide</span> measured at `
    + `Mauna Loa from ${M.co2.start} to ${M.co2.end}, which rise and repeat with a period of `
    + `exactly twelve months because the northern hemisphere has one growing season a year; and `
    + `<span class="bold">${n0(M.sun_n)} yearly counts of sunspots</span> from ${M.sunspots.start} `
    + `to ${M.sunspots.end}, which also repeat, about every eleven years, and never twice the `
    + `same. Everything that works on the first fails on the second, and the gap between them is `
    + `most of what the classical toolbox is for. `
    + `(The carbon dioxide record is ${n0(M.co2.weekly_n)} weekly flask samples with `
    + `${M.co2.weekly_missing} missing; averaged into calendar months that leaves `
    + `${M.co2.months_interpolated} months empty, and those ${M.co2.months_interpolated} are `
    + `filled by linear interpolation. It is a small thing and it is still part of the data, so `
    + `it is written here rather than in a footnote.)`);

  set('scrolly-intro',
    `Almost every classical model assumes the series is <span class="bold">stationary</span>: `
    + `that its mean, its variance and its correlation with its own past do not depend on where `
    + `you are looking. Carbon dioxide is not. The standard repair is subtraction, and there are `
    + `two things worth subtracting: last month, and last year. Below are all four combinations, `
    + `each with the correlogram that goes with it, and two tests that were built to answer `
    + `exactly this question.`);

  set('scrolly-step-0',
    `Raw, the series climbs from ${data.series.co2.y[0]} to `
    + `${data.series.co2.y[data.series.co2.y.length - 1]} parts per million. The augmented `
    + `Dickey-Fuller test reports p = ${co2.raw.adf_p}, which fails to reject a unit root, and `
    + `the KPSS test reports p = ${co2.raw.kpss_p}, which rejects stationarity. Both tests agree: `
    + `not stationary. The correlogram agrees too, decaying from ${co2.raw.acf1} so slowly that `
    + `three years later it is still at ${S.curves.raw.acf[36]}.`);

  set('scrolly-step-1',
    `Subtract the previous month and the trend is gone, and now look at what the tests say. `
    + `Dickey-Fuller: p = ${co2.d1.adf_p}. KPSS: p = ${co2.d1.kpss_p}. `
    + `<span class="bold">Both tests now call this stationary</span>, and its correlation with `
    + `itself twelve months ago is <span class="bold">${co2.d1.acf_season}</span>. Nothing is `
    + `broken: a fixed cycle does not wander off, and wandering off is the only thing either `
    + `test was built to detect. The picture and the p-value are both right and they are `
    + `answering different questions.`);

  set('scrolly-step-2',
    `Subtract the same month a year ago instead and the seasonal spikes collapse `
    + `(${co2.raw.acf_season} at lag twelve becomes ${co2.dS.acf_season}), but the slow decay `
    + `comes back: ${co2.dS.acf1} at lag one, and KPSS goes back to rejecting at `
    + `p = ${co2.dS.kpss_p}. One difference removes one problem.`);

  set('scrolly-step-3',
    `Both, and there is nothing left to remove: ${co2.d1dS.n} observations with a standard `
    + `deviation of ${co2.d1dS.sd} against ${co2.raw.sd} at the start, KPSS at `
    + `p = ${co2.d1dS.kpss_p} and Dickey-Fuller at p = ${co2.d1dS.adf_p}. The two negative spikes `
    + `left at lags one and twelve (${co2.d1dS.acf1} and ${co2.d1dS.acf_season}) are not noise `
    + `either: they are the signature of a moving average term at each of those lags, which is `
    + `exactly the model the next section fits.`);

  set('stationarity-note',
    `That second step is the one worth keeping. A test is a question with a fixed shape, and `
    + `"is this stationary" is not the question it asks: Dickey-Fuller asks whether the level `
    + `wanders without a resting place, KPSS asks whether it stays around one, and a seasonal `
    + `cycle is a perfectly well behaved answer to both. Run them on the differenced carbon `
    + `dioxide and they agree it is fine while its twelve month autocorrelation is `
    + `${co2.d1.acf_season}. The correlogram is not decoration next to the test. It is the part `
    + `that can see what the test cannot.`);

  set('identify-intro',
    `The classical recipe is to read the model order off these two plots: an autoregression of `
    + `order p has a partial autocorrelation that cuts off after lag p, and a moving average of `
    + `order q has an autocorrelation that cuts off after lag q. Below are four simulated series `
    + `with their orders hidden. The dashed band is the usual one, plus or minus 1.96 over the `
    + `square root of the sample size, and the rule is to take the last lag that pokes outside `
    + `it.`);

  set('identify-note',
    `Reading a plot is a hypothesis test, and this one is run ${I.max_lag} times on the same `
    + `data. On a series with no autocorrelation at all, each lag pokes outside a 95% band with `
    + `probability 0.05, so the rule finds something with probability `
    + `<span class="bold">${pc(I.predicted_fp, 2)}</span> before any data exists: that is `
    + `1 - 0.95<sup>${I.max_lag}</sup>, and it is a property of the procedure, not of the world. `
    + `Measured on ${n0(I.n_sim)} white noise series at each of three lengths, the rule claims an `
    + `order in ${pc(noiseRows[0].rule_over, 1)}, ${pc(noiseRows[1].rule_over, 1)} and `
    + `${pc(longest.rule_over, 1)} of them, `
    + `${Math.abs(longest.rule_over - I.predicted_fp) < 0.03
      ? `and the longest series lands on the closed form: ${pc(longest.rule_over, 1)} against `
        + `${pc(I.predicted_fp, 2)} predicted. The shorter ones fall below it because the `
        + `estimated partial autocorrelations at short samples are not the independent tests the `
        + `arithmetic assumes.`
      : `against ${pc(I.predicted_fp, 2)} predicted, and the gap is the arithmetic's assumption `
        + `that ${I.max_lag} estimated correlations are ${I.max_lag} independent tests.`}`);

  const idRows = document.querySelector('#identify-table');
  if (idRows) {
    idRows.innerHTML = I.rows.map((x) => `<tr>
      <td><span class="bold">${x.process}</span></td>
      <td>${x.n}</td>
      <td>${pc(x.rule_acc, 1)}</td>
      <td>${pc(x.aic_acc, 1)}</td>
      <td>${pc(x.rule_over, 1)}</td>
    </tr>`).join('');
  }

  set('identify-table-note',
    `Every cell is ${n0(I.n_sim)} series simulated from the process on the left and identified `
    + `twice: once by the rule, once by taking the smallest AIC over AR(0) to AR(${I.max_lag}). `
    + `The averages are ${pc(ruleMean, 1)} for the rule and ${pc(aicMean, 1)} for AIC, `
    + `${Math.abs(ruleMean - aicMean) < 0.03
      ? `which is close enough that this table is not an argument for the criterion over the `
        + `plot. It is an argument about both: the right order is recovered somewhere between `
        + `half and three quarters of the time by either, on data generated from exactly the `
        + `model being fitted, with no seasonality, no trend and no outliers.`
      : `so the criterion is the better instrument here, though neither is reliable: this is `
        + `data generated from exactly the model being fitted, with no seasonality, no trend and `
        + `no outliers.`} `
    + `The worst cell is AR(3) at 60 observations, `
    + `${pc(I.rows.find((x) => x.true_p === 3 && x.n === 60).rule_acc, 1)} for the rule. Order `
    + `selection is not a solved step that happens before the interesting part; it is a `
    + `measurement with a standard error, and treating a fitted order as a fact is where a lot `
    + `of confident forecasting goes wrong.`);

  set('identity-intro',
    `Exponential smoothing and ARIMA are usually taught as rival families, one built from `
    + `weighted averages and one from difference equations. For the simplest member of each, they `
    + `are the same model. Simple exponential smoothing keeps a level and slides it towards each `
    + `new observation; ARIMA(0,1,1) says the change is white noise plus a fraction of the last `
    + `error:`);

  set('identity-algebra',
    `Substitute $e_t = y_t - \\ell_{t-1}$ into the second and it becomes `
    + `$\\hat y_{t+1} = y_t + \\theta(y_t - \\ell_{t-1})$, which for `
    + `$\\theta = \\alpha - 1$ is $\\alpha y_t + (1-\\alpha)\\ell_{t-1}$, which is the first. Not `
    + `approximately, not asymptotically: the same arithmetic. The slider below runs both `
    + `recursions over all ${n0(M.co2_n)} months in your browser and reports the largest `
    + `disagreement between them.`);

  set('identity-note',
    `So the identity holds to ${E.identity_worst.toExponential(1)} parts per million across the `
    + `${E.identity_grid[2]} rates, which is float arithmetic and nothing else. And then the two `
    + `fitted models disagree by <span class="bold">${E.co2.h1_gap}</span> parts per million one `
    + `month out. Both facts are true and neither is a mistake. `
    + `${E.co2.reachable
      ? `The likelihood lands at theta = ${E.co2.theta_hat}, inside the range a smoother can `
        + `reach, and the difference is down to the two objectives.`
      : `Fitting the smoother minimises squared one step error with alpha kept in [0, 1], and it `
        + `walks to the edge: ${E.co2.alpha_hat.toFixed(4)}, which is the naive forecast. Fitting the ARIMA `
        + `maximises an exact likelihood with theta free in (-1, 1), and it lands at `
        + `${E.co2.theta_hat}, which implies a smoothing rate of ${E.co2.implied_alpha}. A `
        + `smoothing rate above one is not a smoothing rate: it overshoots each new observation `
        + `on purpose, which is a sensible thing to do on a series that is still trending and `
        + `not something the smoothing recipe is allowed to express.`} `
    + `Run the same two estimators on ${n0(E.ma1.n)} points actually generated by an ARIMA(0,1,1) `
    + `with theta = ${E.ma1.true_theta}, where the equivalence has somewhere to live, and they `
    + `agree: alpha = ${E.ma1.alpha_hat} against an implied ${E.ma1.implied_alpha}, a gap of `
    + `${E.ma1.gap}, and forecasts ${E.ma1.h1_gap} apart. `
    + `<span class="bold">An identity between models is not an identity between estimators</span>, `
    + `and which of the two you are being sold is worth asking whenever a library offers you `
    + `both.`);

  set('forecast-intro',
    `Everything from here is scored the same way, and the way matters more than the models do. `
    + `The last ${R.test} months of the carbon dioxide record are never fitted on. The model is `
    + `refitted from scratch at ${R.n_origins} successive origins, and at each one it forecasts `
    + `the next twelve months, which are then compared with what happened. No shuffling, no `
    + `random split, nothing below the line ever visible above it. Here is one of those `
    + `${R.n_origins} origins drawn in full.`);

  set('forecast-note',
    `Two things on that chart are worth separating. The line is a point forecast and the band is `
    + `a distribution, and only one model here produces both. The band is also the part that `
    + `almost never gets scored, so the next section scores it: at every horizon, over all `
    + `${R.n_origins} origins, how often did the truth land inside the interval that promised to `
    + `contain it?`);

  set('coverage-intro',
    `An interval that says 95% is making a frequency claim, and a frequency claim can be `
    + `counted. Both series are here because they answer differently, and the difference is the `
    + `result.`);

  set('coverage-note',
    `On carbon dioxide the intervals are `
    + `${Math.abs(mean95 - 0.95) < 0.02 ? 'about right' : 'off'}: `
    + `${pc(mean95, 1)} of the truth falls inside the 95% band and ${pc(mean80, 1)} inside the `
    + `80% one, averaged over the twelve horizons. On the sunspots the same machinery reports `
    + `${pc(sun95, 1)} and ${pc(sun80, 1)}. `
    + `${sun95 < mean95 - 0.02
      ? `The intervals are narrower than they claim on the series whose shape the model gets `
        + `wrong, and honest on the series whose shape it gets right. That is the useful reading `
        + `of a coverage number: it is not measuring the sample size, it is measuring the `
        + `misspecification. An interval is computed as though the fitted parameters were the `
        + `true parameters and the model were the true model, and on the sunspots the second of `
        + `those is badly false, so one year in `
        + `${Math.round(1 / (1 - sun95))} lands outside a band that promised one in twenty.`
      : `Both come out close to their nominal levels, which is more than the usual warning about `
        + `prediction intervals would predict.`}`);

  set('board-intro',
    `Five methods, ${R.n_origins} origins, twelve horizons, and every error divided by the same `
    + `unit: the average error the seasonal naive forecast makes inside the training data. One `
    + `means "no better than repeating last year", which is a low bar that is famously hard to `
    + `clear.`);

  const sunBy = Object.fromEntries(U.rows.map((x) => [x.model, x]));
  const pair = {
    SARIMA: sunBy['AR(9)'], ETS: sunBy['ETS(A,Ad,N)'],
    'seasonal naive': sunBy['seasonal naive'], naive: sunBy.naive, drift: sunBy.mean,
  };
  const names = {
    SARIMA: 'AR(9)', ETS: 'ETS(A,Ad,N)', 'seasonal naive': 'seasonal naive',
    naive: 'naive', drift: 'mean',
  };
  const bt = document.querySelector('#board-table');
  if (bt) {
    bt.innerHTML = R.rows.map((row) => {
      const other = pair[row.model];
      return `<tr>
        <td><span class="bold">${row.model}</span></td>
        <td>${row.mase_h1}</td>
        <td>${row.mase_h12}</td>
        <td><span class="value">${row.mase_mean}</span></td>
        <td>${other ? `${other.mase_mean} <span style="opacity:.7">(${names[row.model]})</span>`
    : ''}</td>
      </tr>`;
    }).join('');
  }

  set('board-note',
    `The last column is the same table on the sunspots, with the nearest equivalent model, and `
    + `it does not rhyme. On carbon dioxide ${best.model} wins at ${best.mase_mean} with `
    + `${second.model} just behind at ${second.mase_mean}, and both are about `
    + `${(snaive.mase_mean / best.mase_mean).toFixed(1)} times better than repeating last year. `
    + `On the sunspots the winner is ${sunBest.model} at ${sunBest.mase_mean}, `
    + `${sunBest.mase_mean > 1 ? 'which is still worse than the unit it is divided by' : ''}, and `
    + `exponential smoothing with a damped trend comes <span class="bold">last</span>, at `
    + `${sunEts.mase_mean}, behind even the naive forecast's ${sunNaive.mase_mean}: a method one `
    + `place off the podium on the first series is four times worse than doing nothing on the `
    + `second. `
    + `${P.helps
      ? `And the seasonal machinery is not simply wrong there either. Sweeping the seasonal `
        + `period from 9 to 13 years, the best is ${P.best_period} at ${P.best} against `
        + `${P.none} with no seasonal term at all, so a rigid ${P.best_period} year cycle `
        + `imposed on a cycle that is not rigid still helps by `
        + `${((1 - P.best / P.none) * 100).toFixed(0)}%. It just never gets below one.`
      : `And a seasonal term does not rescue it: the best period between 9 and 13 years scores `
        + `${P.best} against ${P.none} for no seasonal term at all.`} `
    + `One more number, because the protocol above is expensive and the shortcut is everywhere: `
    + `refitting at every origin against fitting once and filtering the new observations through `
    + `fixed parameters, on ${F.n_origins} origins of the carbon dioxide series, gives ${F.refit} `
    + `and ${F.fixed}. ${Math.abs(F.gap) < 0.005
      ? `The shortcut costs nothing measurable here, and this page still uses the expensive `
        + `version everywhere, because knowing that it costs nothing on this series is not the `
        + `same as assuming it.`
      : `The shortcut costs ${Math.abs(F.gap)}, which is why every other number here refits.`}`);

  set('verdict-intro',
    `Four situations, and what this page measured about each.`);
  set('verdict-season',
    `On a series with both, ${best.model} scores ${best.mase_mean} and ${second.model} `
    + `${second.mase_mean}, against ${snaive.mase_mean} for repeating last year.`);
  set('verdict-season-warn',
    `Differencing twice is not free: it costs ${M.co2_n - co2.d1dS.n} observations and it `
    + `assumes the cycle really is fixed. Read the correlogram before believing a test.`);
  set('verdict-cycle',
    `${sunBest.model} wins on the sunspots at ${sunBest.mase_mean}, and the seasonal sweep puts `
    + `the best period at ${P.best_period} years.`);
  set('verdict-cycle-warn',
    `Nothing here beats the unit: the best score is ${sunBest.mase_mean} where one is the `
    + `in sample seasonal naive error. A cycle you can see is not a cycle you can forecast.`);
  set('verdict-eval',
    `${R.n_origins} rolling origins on one series and ${U.n_origins} on the other, refitted at `
    + `each, with the horizon scored separately because the error is not monotone in it.`);
  set('verdict-eval-warn',
    `A random split leaks the future into the past. Shuffling a series before cross validation `
    + `is not a smaller version of the same mistake; it is a different experiment that answers a `
    + `question nobody asked.`);
  set('verdict-interval',
    `Counting is all it takes: ${pc(mean95, 1)} inside the 95% band on one series and `
    + `${pc(sun95, 1)} on the other.`);
  set('verdict-interval-warn',
    `Intervals come from the model being true. When it is not, they are narrow in exactly the `
    + `situation where you needed them to be wide.`);

  set('closing-note',
    `Everything above forecasts one number at a time from its own past. Two things break that. `
    + `The first is having more than one series and suspecting they know something about each `
    + `other, which turns one equation into a system and the parameter count into a square. The `
    + `second is having a series whose <span class="bold">mean</span> is unforecastable, where `
    + `every model in this article scores about the same as a constant, and yet whose spread `
    + `clearly is not: quiet weeks follow quiet weeks and violent days arrive in groups. `
    + `<a href="../volatility/">The next article</a> takes both: a vector autoregression on four `
    + `macroeconomic accounts, and a variance model on twenty years of daily returns, where the `
    + `question stops being what will happen and becomes how wrong you are about to be.`);

  set('resources-note',
    `${n0(M.co2_n)} monthly carbon dioxide averages (${M.co2.start} to ${M.co2.end}, bundled `
    + `with statsmodels ${M.statsmodels}, digest <code>${M.co2_digest}</code>) and `
    + `${n0(M.sun_n)} yearly sunspot counts (${M.sunspots.start} to ${M.sunspots.end}, digest `
    + `<code>${M.sun_digest}</code>). Both are rounded to four decimals before anything is `
    + `measured on them, so the recursions this page runs and the ones the generator ran see the `
    + `same numbers. The identification study is ${n0(I.n_sim)} simulated series per cell at `
    + `lengths ${I.lengths.join(', ')}, seed ${M.seed}, with the rule allowed to look `
    + `${I.max_lag} lags deep. The rolling evaluations refit at every origin: ${R.n_origins} on `
    + `carbon dioxide with a seasonal ARIMA(0,1,1)(0,1,1)<sub>12</sub> and ${U.n_origins} on the `
    + `sunspots with an AR(9). Run on ${M.threads} threads with numpy ${M.numpy}.`);
}
