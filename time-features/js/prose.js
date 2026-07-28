/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const F = data.features;
  const L = data.leakage;
  const E = data.extrapolate;
  const D = data.direct;
  const P = data.prophet;
  const C = data.decomposition;
  const B = data.board;
  const bestSet = F.rows.reduce((a, b) => (b.mase_mean < a.mase_mean ? b : a), F.rows[0]);
  const lagsOnly = F.rows[0];
  const fullSet = F.rows[F.rows.length - 1];
  const bestTau = P.sweep.reduce((a, b) => (b.mase < a.mase ? b : a), P.sweep[0]);
  const worstAgree = P.sweep.reduce((a, b) => (b.agree > a.agree ? b : a), P.sweep[0]);
  const flat = P.sweep.filter((s) => s.active === 0);
  const easy = C.rows[0];
  const hard = C.rows[C.rows.length - 1];

  set('opening-note',
    `Everything below is measured on the same series as `
    + `<a href="../arima/">the first article of this branch</a>, and that is a check rather than `
    + `a claim: the digest of the ${n0(M.co2_n)} monthly carbon dioxide averages is `
    + `<code>${M.co2_digest}</code> on both pages, and this generator refuses to run if the other `
    + `file disagrees. The evaluation is the same too, ${B.available ? B.n_origins : ''} rolling `
    + `origins over the last ${M.test} months, twelve months ahead, errors in the same units. So `
    + `the table at the end can put a gradient boosted tree next to a seasonal ARIMA and mean it.`);

  set('scrolly-intro',
    `The recipe is mechanical. One row per month. The target is that month's value. Every column `
    + `is something computable from strictly earlier months: the value one month ago, twelve `
    + `months ago, a rolling average, the month number, a sine and a cosine of the year. Then it `
    + `is a regression problem and time has been removed from it.`);

  set('scrolly-step-0',
    `Start with the rows. Each one is a month, and the last column is what we are trying to `
    + `produce. On its own this is a table with a target and no features, which predicts the `
    + `average of the column and nothing else.`);

  set('scrolly-step-1',
    `The obvious columns are earlier values of the same series: one, two, three and twelve months `
    + `back. Twelve is there because the cycle is twelve, and putting it in as a column is what `
    + `replaces the seasonal difference the classical model took. On their own, these four score `
    + `${lagsOnly.mase_mean} over the ${F.n_origins} origins.`);

  set('scrolly-step-2',
    `The calendar is free information: the month index, and the first two harmonics of the year `
    + `as a sine and cosine pair, which is the same Fourier trick the trend model further down `
    + `uses. Adding those five takes the error from ${lagsOnly.mase_mean} to `
    + `${F.rows[1].mase_mean}, the only step in this table that pays for itself.`);

  set('scrolly-step-3',
    `And then anything else: rolling means over three and twelve months, a rolling standard `
    + `deviation, the last change, the change a year ago, and a raw index of how far into the `
    + `record the row is. Fifteen columns. Every one of them is a value the reader could compute `
    + `standing at that month with no knowledge of the future, which is the one rule; a rolling `
    + `mean that accidentally includes its own row turns a forecast into a lookup, and the `
    + `resulting score will be beautiful.`);

  set('table-note',
    `That is the whole method, and it is why it is popular: the columns can be anything. `
    + `Holidays, promotions, a competitor's price, the weather forecast, another series entirely. `
    + `A classical model has to be extended to accept each of those; a table just gets another `
    + `column. What follows is the three ways the trade goes wrong, in the order they usually `
    + `bite.`);

  set('leak-intro',
    `The first one is the evaluation, and it happens before any modelling. A table has rows, `
    + `and a table with rows invites the reflex of every other supervised problem: shuffle, split `
    + `five ways, average the folds. Here is what those splits actually do, on the `
    + `${n0(L.n_rows)} rows this table has.`);

  set('leak-note',
    `Two protocols, both wrong, failing in opposite directions. The shuffled split reports `
    + `${L.shuffled_mase} where the honest experiment says ${L.rolling_mase}: it is `
    + `${((L.rolling_mase / L.shuffled_mase - 1) * 100).toFixed(0)}% too good, because every row `
    + `it scores has neighbours in the training set and the model is being asked to interpolate `
    + `rather than to forecast. The contiguous version reports ${L.blocked_mase}, which is `
    + `${(L.blocked_mase / L.rolling_mase).toFixed(1)} times too <span class="bold">bad</span>, `
    + `and the reason is worth having: its first block is forecast by a model trained entirely on `
    + `later data, and this series trends upwards, so that model is being asked for values below `
    + `anything it ever saw. Which is exactly the failure the next section is about, arriving `
    + `early and by accident.`);

  set('ceiling-intro',
    `A tree ends in a leaf, and a leaf predicts the average of the training targets that fell `
    + `into it. Every value a forest or a boosted ensemble can print is therefore a weighted `
    + `average of values it has already seen, which means it has a ceiling and a floor. On a `
    + `series that goes up, that is not a caveat about the tails: it is the forecast.`);

  set('ceiling-note',
    `The repair is one line and it costs nothing: train on the change instead of the level, then `
    + `add the change back. Now the quantity being predicted is a monthly difference, which is not `
    + `trending, so the ceiling stops mattering, and the same model with the same columns goes `
    + `from ${E.levels.mase_mean} to ${E.diff.mase_mean} over ${E.n_origins} origins. This is the `
    + `most common single mistake in feature based forecasting and it is completely invisible in `
    + `a shuffled cross validation, because in a shuffled split the highest values of the series `
    + `are in the training folds. The two failures of this article compound: the wrong split `
    + `hides the wrong target.`);

  set('features-intro',
    `With that fixed, which columns were worth having? Same model, same ${F.n_origins} origins, `
    + `four nested feature sets, forecasting twelve months ahead by feeding the model its own `
    + `output back in.`);

  const ft = document.querySelector('#features-table');
  if (ft) {
    ft.innerHTML = F.rows.map((row) => `<tr>
      <td><span class="bold">${row.label}</span></td>
      <td>${row.n_features}</td>
      <td>${row.mase_h1}</td>
      <td>${row.mase_h12}</td>
      <td><span class="value">${row.mase_mean}</span></td>
    </tr>`).join('');
  }

  set('features-note',
    `The calendar columns are the only ones that clearly pay: ${lagsOnly.mase_mean} to `
    + `${F.rows[1].mase_mean}. After that, six more columns move the average by `
    + `${Math.abs(fullSet.mase_mean - F.rows[1].mase_mean).toFixed(4)}, which on this series is `
    + `nothing, and the trend index makes it very slightly worse at the long horizons `
    + `(${F.rows[1].mase_h12} to ${fullSet.mase_h12}) for the reason the section above is about: `
    + `a column that only ever increases is a column whose future values are all outside the `
    + `training range. `
    + `One more choice that the table hides. Reaching twelve months has two strategies: fit one `
    + `model and feed it its own output back in, or fit twelve models, one per horizon. The first `
    + `scores ${D.recursive_mean} here and the second ${D.direct_mean} over ${D.n_origins} `
    + `origins, so the twelve model version costs twelve times the training and loses `
    + `${(D.direct_mean / D.recursive_mean).toFixed(2)} to one, `
    + `${D.direct[11] < D.recursive[11]
      ? `except at the far end, where it wins ${D.direct[11]} against ${D.recursive[11]}: `
        + `feeding a forecast back in compounds an error, and forecasting the twelfth month `
        + `directly does not.`
      : `at every horizon.`}`);

  set('prophet-intro',
    `The other popular way to forecast without writing down a state space model is to write down `
    + `a curve instead. Prophet is a regression on time itself: a straight line whose slope is `
    + `allowed to change at a fixed set of candidate points, plus a Fourier series for the `
    + `season, with the slope changes penalised in absolute value so that most of them come out `
    + `at exactly zero.`);

  set('prophet-equation-note',
    `That penalty is a lasso, which the atlas has met before in `
    + `<a href="../regularization/">regularization</a>, and it does the same thing here: with `
    + `${P.n_changepoints} candidate bends offered, the fit uses `
    + `${P.sweep.find((s) => s.tau === P.tau).active} of them at the library's default price. `
    + `This page implements the model rather than importing it, and solves it twice by unrelated `
    + `methods, accelerated proximal gradient and coordinate descent from scikit-learn. They `
    + `agree on the fitted trend to ${P.agree.toExponential(1)} parts per million at that `
    + `default, and to ${worstAgree.agree.toExponential(1)} at the setting where they agree `
    + `least, which is the one with the most surviving bends (${worstAgree.active} of them at a `
    + `price of ${worstAgree.tau}). That is not a detail of the plumbing: at 1500 iterations the `
    + `two solvers disagreed by 0.018 and the fitted trend was still moving, so the iteration `
    + `count is part of the measurement.`);

  set('prophet-note',
    `The slider is the parameter people reach for when a Prophet forecast looks wrong, and it is `
    + `worth knowing what it moves. Across its range the fit to the history stays plausible and `
    + `the out of sample error changes by a factor of `
    + `${(P.sweep[P.sweep.length - 1].mase / bestTau.mase).toFixed(2)}: `
    + `${bestTau.mase} at a price of ${bestTau.tau}, ${P.sweep[P.sweep.length - 1].mase} at `
    + `${P.sweep[P.sweep.length - 1].tau}, where ${flat.length} of the seven settings leave `
    + `<span class="bold">no bends at all</span> and the trend is a straight line through forty `
    + `three years of a curve. The library's own default of ${P.tau} sits between the two at `
    + `${P.sweep_default} on the sweep's origins, and ${P.mase_mean} on all ${P.n_origins} of `
    + `them, so the shortcut of sweeping on every fourth origin costs `
    + `${Math.abs(P.sweep_default - P.mase_mean).toFixed(4)} and the two can be read together. `
    + `Rerun at the best price on all ${P.best_full.n_origins} origins it scores `
    + `${P.best_full.mase_mean}.`);

  set('decomp-intro',
    `A decomposition is the other reason to use this model: it hands back a trend and a season `
    + `as separate curves, and those get put in slides. But on real data nobody ever observed the `
    + `trend of the carbon dioxide record separately from its season, so there is no way to know `
    + `whether the split it produced is right. On the written down collection there is: `
    + `${C.n_series} series of ${C.length} points, drawn from four families, with the parts `
    + `recorded before the series existed.`);

  const dt = document.querySelector('#decomp-table');
  if (dt) {
    dt.innerHTML = C.rows.map((row) => `<tr>
      <td><span class="bold">${row.family.replace(/_/g, ' ')}</span></td>
      <td><span class="value">${row.trend_err}</span></td>
      <td>${row.season_err}</td>
      <td>${row.fit_err}</td>
    </tr>`).join('');
  }

  set('decomp-note',
    `Errors are in standard deviations of the series, so they can be compared across families. `
    + `When the truth is inside the model class the recovery is nearly exact: `
    + `${easy.trend_err} on ${easy.family.replace(/_/g, ' ')}. When it is not, it is `
    + `${(hard.trend_err / easy.trend_err).toFixed(0)} times worse `
    + `(${hard.trend_err} on ${hard.family.replace(/_/g, ' ')}), and a series with a sudden jump `
    + `in level lands at ${C.rows.find((x) => x.family === 'level_shift').trend_err}, because a `
    + `continuous piecewise linear trend cannot contain a discontinuity and has to smear it over `
    + `several years. None of that is visible in the fitted curve, which looks equally convincing `
    + `in all four cases. <span class="bold">A decomposition is a claim about unobservable `
    + `quantities, and the only place it can be scored is where they were written down first.</span>`);

  set('board-intro',
    `Everything from this article and the first one, on the same ${B.n_origins} origins of the `
    + `same series, scored the same way.`);

  const rows = [
    ...(B.available ? B.rows.map((x) => ({
      model: x.model, from: '4.2a', h1: x.mase[0], h12: x.mase[11], mean: x.mase_mean,
    })) : []),
    { model: 'boosted tree on the change', from: 'here', h1: E.diff.mase[0],
      h12: E.diff.mase[11], mean: E.diff.mase_mean },
    { model: 'boosted tree on the level', from: 'here', h1: E.levels.mase[0],
      h12: E.levels.mase[11], mean: E.levels.mase_mean },
    { model: `hinge trend, price ${P.best_full.tau}`, from: 'here', h1: P.best_full.mase[0],
      h12: P.best_full.mase[11], mean: P.best_full.mase_mean },
    { model: `hinge trend, price ${P.tau}`, from: 'here', h1: P.mase[0], h12: P.mase[11],
      mean: P.mase_mean },
  ].sort((a, b) => a.mean - b.mean);
  const bt = document.querySelector('#board-table');
  if (bt) {
    bt.innerHTML = rows.map((row) => `<tr>
      <td><span class="bold">${row.model}</span></td>
      <td>${row.from}</td>
      <td>${row.h1}</td>
      <td>${row.h12}</td>
      <td><span class="value">${row.mean}</span></td>
    </tr>`).join('');
  }

  set('board-note',
    `The winner is ${rows[0].model} at ${rows[0].mean}, from ${rows[0].from === '4.2a'
      ? 'the classical article' : 'this one'}. `
    + `${rows[0].from === '4.2a'
      ? `That is worth sitting with. This series is a smooth trend with an exactly periodic `
        + `season and no covariates, which is precisely what a seasonal ARIMA was designed for in `
        + `1970, and none of the machinery in this article gets near it: the best thing here is `
        + `${rows.find((x) => x.from === 'here').model} at `
        + `${rows.find((x) => x.from === 'here').mean}, ${(rows.find((x) => x.from === 'here').mean / rows[0].mean).toFixed(2)} `
        + `times the winner's error.`
      : `The table method wins here, which is the outcome the competitions report on collections `
        + `of many messy series.`} `
    + `The honest reading is not that one family is better. It is that the table method buys `
    + `<span class="bold">flexibility</span>, and flexibility is worth nothing on a series that `
    + `needs none. What it is worth on a thousand series that each need a little, and what has to `
    + `change for a single model to be fitted to all of them at once, is the next article.`);

  set('verdict-intro', `Four situations, and what this page measured about each.`);
  set('verdict-diff',
    `Trained on the level the same model scores ${E.levels.mase_mean}; trained on the change, `
    + `${E.diff.mase_mean}. ${E.over_ceiling} of the twelve months at the drawn origin are above `
    + `anything in the training data.`);
  set('verdict-diff-warn',
    `The symptom is a forecast that flattens rather than one that is noisy, and it does not show `
    + `up at one step ahead, which is where most people look.`);
  set('verdict-table',
    `Fifteen columns, any regressor, and the calendar alone took ${lagsOnly.mase_mean} to `
    + `${F.rows[1].mase_mean}.`);
  set('verdict-table-warn',
    `Every column must be computable standing at its own row. The rest of the columns here bought `
    + `${Math.abs(fullSet.mase_mean - F.rows[1].mase_mean).toFixed(4)}.`);
  set('verdict-cv',
    `A rolling origin over ${L.n_origins} origins reports ${L.rolling_mase}, and it is the only `
    + `one of the three protocols that never trains on data later than what it scores.`);
  set('verdict-cv-warn',
    `Shuffled folds report ${L.shuffled_mase} and contiguous folds ${L.blocked_mase}. Being wrong `
    + `in the pessimistic direction is not safer: it will make you discard a model that works.`);
  set('verdict-decomp',
    `On series whose parts were written down, the trend comes back to ${easy.trend_err} of a `
    + `standard deviation when the truth is in the model class.`);
  set('verdict-decomp-warn',
    `And to ${hard.trend_err} when it is not, with a fitted curve that looks just as good. The `
    + `price of a bend moves the forecast by a factor of `
    + `${(P.sweep[P.sweep.length - 1].mase / bestTau.mase).toFixed(2)}.`);

  set('closing-note',
    `Everything in this article fits one series at a time. Ask for a forecast of a thousand `
    + `series and it fits a thousand models, each of which sees only its own history, and each of `
    + `which has to discover on its own that Decembers are different. The alternative is a single `
    + `model trained across all of them, which is the idea behind every deep forecaster of the `
    + `last decade and is not obviously a good one: the series have different scales, different `
    + `seasons and different lengths. <a href="../deep-forecast/">The next article</a> builds two `
    + `of those, a stack of residual blocks that learns a basis and an attention model that reads `
    + `covariates, and asks the question the fashion never asks out loud, which is how many `
    + `series it takes before sharing beats fitting each one alone.`);

  set('resources-note',
    `${n0(M.co2_n)} monthly carbon dioxide averages, the same series and the same rounding as `
    + `4.2a (digest <code>${M.co2_digest}</code>, asserted by the generator). The table is `
    + `${n0(L.n_rows)} rows and up to ${fullSet.n_features} columns; the regressor is XGBoost `
    + `${M.xgboost} with 300 trees of depth 3 at a learning rate of 0.06, seed ${M.seed}, refit `
    + `at every one of the ${F.n_origins} origins. The hinge trend offers ${P.n_changepoints} `
    + `candidate bends over the first 80% of the record and ${P.n_fourier} Fourier pairs, solved `
    + `by 8000 accelerated proximal gradient steps and checked against coordinate descent from `
    + `scikit-learn ${M.sklearn} (agreement ${P.agree.toExponential(1)} on the fitted trend). The `
    + `written down collection is ${C.n_series} series of ${C.length} points from four families, `
    + `seed ${M.seed}. Run on ${M.threads} threads with numpy ${M.numpy}.`);
}
