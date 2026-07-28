/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const S = data.scaling;
  const B = data.bases;
  const A = data.attention;
  const N = data.normalisation;
  const R = data.real;
  const last = S.rows[S.rows.length - 1];
  const first = S.rows[0];
  const beatsLocal = S.rows.filter((x) => x.mase < S.local_nbeats);
  const beatsEts = S.rows.filter((x) => x.mase < S.ets);
  const dec = B.decomposition;
  const bestDec = dec[0];
  const worstDec = dec[dec.length - 1];
  const noiseRow = dec.find((x) => x.family === 'noise');
  const driver = A.rows[0];
  const copy = A.rows[1];
  const echo = A.rows[2];
  const noiseCov = A.rows[3];
  const gap = Math.abs(B.generic.mase - B.interpretable.mase);
  const prevRows = (data.board && data.board.rows) || [];
  const bestPrev = prevRows.length
    ? prevRows.reduce((a, b) => (b.mase_mean < a.mase_mean ? b : a), prevRows[0]) : null;
  const snaivePrev = prevRows.find((x) => x.model === 'seasonal naive') || null;
  const floor = Math.max(B.generic.spread, B.interpretable.spread);

  set('opening-note',
    `The collection is written down rather than collected: ${n0(M.n_train + M.n_test)} series of `
    + `${M.length} points from ${M.families.length} families whose trend, season and noise were `
    + `recorded before the series existed. ${n0(M.n_train)} of them are for training and `
    + `${M.n_test} are held out, and nothing here is ever fitted to a held out series. That is a `
    + `deliberate choice and it buys two things this branch has not had before: an answer key for `
    + `a decomposition, and a clean way to ask what a model knows about a series it has never `
    + `seen. The real carbon dioxide record turns up once at the end, and no model on this page `
    + `is trained on it either.`);

  set('scrolly-intro',
    `N-BEATS is a stack of blocks and one idea. Each block looks at what is left of the input, `
    + `produces two things, a <span class="bold">backcast</span> that claims to explain part of `
    + `the lookback and a <span class="bold">forecast</span> that contributes to the answer, and `
    + `then subtracts its own backcast so the next block only sees the residue.`);

  set('stack-equation-note',
    `The interpretable version constrains what a block can say: the first blocks may only emit `
    + `low order polynomials and the later ones only sums of sines and cosines at the seasonal `
    + `frequency. That is the whole difference, and it costs `
    + `${B.generic.mase < B.interpretable.mase
      ? `${(B.generic.mase - B.interpretable.mase).toFixed(4)} of accuracy`
      : `nothing measurable: ${B.interpretable.mase} against ${B.generic.mase}`}. `
    + `Below is one held out window through the constrained stack, with the components the series `
    + `was actually built from drawn underneath.`);

  set('scrolly-step-0',
    `${M.lookback} months in, ${M.horizon} months out, and the window has been centred and scaled `
    + `by its own mean and standard deviation before the network sees it, which is the only reason `
    + `one network can be shown series that live at different levels.`);

  set('scrolly-step-1',
    `The first three blocks can only produce cubic polynomials in time. Whatever the window `
    + `contains, what comes out of them is smooth, and what is left over is passed on.`);

  set('scrolly-step-2',
    `The last three blocks can only produce sums of sines and cosines at the seasonal frequency `
    + `and its first three harmonics. They never see the original window: they see what the trend `
    + `blocks could not explain.`);

  set('scrolly-step-3',
    `The forecast is the sum of what every block contributed. The dashed curves are the trend and `
    + `the season this series was generated from, and the comparison is the only honest way to `
    + `ask whether the constrained blocks mean what their names say. Over ${n0(dec.reduce((a, x) => a + x.n, 0))} `
    + `held out windows the answer depends entirely on the family, which is the table two sections `
    + `down.`);

  set('stack-note',
    `Nothing in that construction is specific to one series, which is the point of it: the same `
    + `weights are asked about every series in the collection, and the scaling in the previous `
    + `step is what makes that possible. So the first question is whether sharing helps at all.`);

  set('scaling-intro',
    `One network, trained on ${first.series}, ${S.rows[1].series}, ${S.rows[2].series} and `
    + `${last.series} series, always scored on the same ${S.n_test} held out ones over the last `
    + `stretch of each. Against three things that need no sharing: the same architecture fitted to `
    + `each held out series alone using only the part of it before the scored stretch, `
    + `exponential smoothing refitted at every origin, and repeating last year.`);

  set('scaling-note',
    `${beatsLocal.length
      ? `Sharing overtakes the same architecture fitted to one series alone at `
        + `${beatsLocal[0].series} series (${beatsLocal[0].mase} against ${S.local_nbeats}), `
        + `and the curve keeps falling to ${last.mase}.`
      : `On this collection sharing never overtakes the same architecture fitted to each series `
        + `alone (${last.mase} against ${S.local_nbeats}), which is the honest result at this `
        + `size and worth stating plainly: ${last.series} series is not many, and the published `
        + `wins are reported on tens of thousands.`} `
    + `${beatsEts.length
      ? `Against exponential smoothing refitted at every origin, ${beatsEts[0].series} series is `
        + `enough (${beatsEts[0].mase} against ${S.ets}).`
      : `Exponential smoothing refitted at every origin scores ${S.ets} and none of these sizes `
        + `reaches it. A classical local model with five parameters is still the thing to beat.`} `
    + `Two cautions about reading this curve. Each point is one training run, so the `
    + `non-monotone step from ${first.series} to ${S.rows[1].series} series `
    + `(${first.mase} then ${S.rows[1].mase}) is what a single seed looks like, not a real dip. `
    + `And every series here is ${M.length} points long, so more series means more variety, never `
    + `more history: this axis is about breadth.`);

  set('bases-intro',
    `The two bases score ${B.generic.mase} for the generic blocks and ${B.interpretable.mase} for `
    + `the constrained ones, a difference of ${gap.toFixed(4)} against a spread across three seeds `
    + `of ${floor.toFixed(4)}, so `
    + `${gap < floor
      ? `<span class="bold">this comparison does not resolve</span>: the constraint is free at `
        + `this size.`
      : `the difference is larger than the seed noise, which makes the constraint worth its cost.`} `
    + `Accuracy is not what the constrained version is for, though. It is for the two curves the `
    + `scrolly drew, and those can be scored, because the parts were written down. Errors below `
    + `are in standard deviations of the series.`);

  const bt = document.querySelector('#bases-table');
  if (bt) {
    bt.innerHTML = dec.map((row) => `<tr>
      <td><span class="bold">${row.family.replace(/_/g, ' ')}</span></td>
      <td><span class="value">${row.trend_err}</span></td>
      <td>${row.season_err}</td>
      <td>${row.n}</td>
    </tr>`).join('');
  }

  set('bases-note',
    `The order of that table is the whole message. On ${bestDec.family.replace(/_/g, ' ')} the `
    + `trend blocks recover the real trend to ${bestDec.trend_err} of a standard deviation. On `
    + `${worstDec.family.replace(/_/g, ' ')} they are at ${worstDec.trend_err}, `
    + `${(worstDec.trend_err / bestDec.trend_err).toFixed(0)} times worse. `
    + `${noiseRow
      ? `And the seasonal blocks are the ones to watch: on series that are pure noise with no `
        + `season at all, they still produce one, and it is wrong by ${noiseRow.season_err} of a `
        + `standard deviation. A block whose only vocabulary is sines will answer in sines `
        + `whatever it is shown.`
      : ''} `
    + `This is the same lesson the previous article measured on a hinge trend, arrived at from a `
    + `different direction: a decomposition is a claim about quantities nobody observed, and `
    + `naming a block "trend" does not make its output one.`);

  set('attention-intro',
    `The other family of deep forecaster is built on attention, and its selling point is the same `
    + `word: interpretable. A temporal fusion transformer publishes a weight per input, and those `
    + `weights get read as importances. Here is that model shape on ${n0(A.n_windows)} windows of `
    + `series driven by a promotion, with four inputs: the promotion itself, `
    + `<span class="bold">an exact duplicate of it</span>, yesterday's promotion, and a column of `
    + `noise.`);

  set('attention-note',
    `Three readings, in order of how much they cost you. `
    + `First, the weights are not nonsense: noise gets ${noiseCov.weight} and breaking it costs `
    + `${noiseCov.ratio} times the loss, which is nothing, and the two agree. `
    + `Second, a weight of zero does not mean no effect. ${echo.name} is given ${echo.weight} and `
    + `breaking it still costs ${echo.ratio} times, because the weights are themselves computed `
    + `from all four inputs, so destroying an input the model says it ignores changes what it `
    + `decides to attend to. `
    + `Third, and worst: the promotion and its exact duplicate carry identical information. The `
    + `model gives them ${driver.weight} and ${copy.weight}, which is an arbitrary split of one `
    + `signal, and breaking `
    + `${Math.min(driver.ratio, copy.ratio) < 1.1 ? 'the less favoured one costs almost nothing '
      + `(${Math.min(driver.ratio, copy.ratio)} times)` : 'either costs a lot'} `
    + `while breaking both costs ${A.both_copies.ratio} times. Neither number is the importance of `
    + `the promotion. The weight is a description of one fitted model's internal traffic, and the `
    + `permutation is a statement about that model given everything else it still has.`);

  set('norm-intro',
    `One more thing has to be right before any of the above is, and it is the least interesting `
    + `sentence in every paper: each window is centred and scaled by its own mean and standard `
    + `deviation. The levels in this collection run from ${N.level_min} to ${N.level_max}, so a `
    + `shared network sees the same shape at wildly different heights. Removing that step and `
    + `retraining gives ${N.raw} against ${N.normalised}, across three seeds each with spreads of `
    + `${N.raw_spread} and ${N.normalised_spread}, `
    + `${N.raw < N.normalised - Math.max(N.raw_spread, N.normalised_spread)
      ? `so on this collection <span class="bold">not normalising is better</span>, by more than `
        + `the seed noise. That is not the textbook answer and it is what was measured. The `
        + `per family split says where it comes from: `
        + `${(() => {
          const fams = N.families.map((f) => ({
            f, d: N.per_family.raw[f] - N.per_family.normalised[f],
          })).sort((a, b) => a.d - b.d);
          return `${fams[0].f.replace(/_/g, ' ')} gains most from leaving the level alone `
            + `(${N.per_family.raw[fams[0].f]} against ${N.per_family.normalised[fams[0].f]}) and `
            + `${fams[fams.length - 1].f.replace(/_/g, ' ')} loses most `
            + `(${N.per_family.raw[fams[fams.length - 1].f]} against `
            + `${N.per_family.normalised[fams[fams.length - 1].f]})`;
        })()}. Scaling a window by its own standard deviation throws away how big it is, and for `
        + `some of these families that is the informative part.`
      : `so normalising earns its place, by ${(N.raw - N.normalised).toFixed(4)} against a seed `
        + `spread of ${Math.max(N.raw_spread, N.normalised_spread)}.`}`);

  const rows = [
    { model: `one network, ${last.series} series`, on: `${M.n_test} held out synthetic series`,
      err: last.mase },
    { model: 'the same architecture, one series each', on: 'each held out series alone',
      err: S.local_nbeats },
    { model: 'exponential smoothing', on: 'each held out series alone', err: S.ets },
    { model: 'seasonal naive', on: 'nothing', err: S.seasonal_naive },
  ].sort((a, b) => a.err - b.err);
  const boardT = document.querySelector('#board-table');
  if (boardT) {
    boardT.innerHTML = rows.map((row) => `<tr>
      <td><span class="bold">${row.model}</span></td>
      <td>${row.on}</td>
      <td><span class="value">${row.err}</span></td>
      <td>${(row.err / S.seasonal_naive).toFixed(2)}x</td>
    </tr>`).join('');
  }

  set('board-note',
    `And then the number this article exists to set up. The network trained on `
    + `${last.series} written down series, which has never seen a real measurement of anything, `
    + `was handed the carbon dioxide record from `
    + `<a href="../arima/">the first article of this branch</a> and asked for twelve months at `
    + `each of its ${R.n_origins} origins. It scores <span class="value">${R.mase}</span> in the `
    + `same units as that article's table, where ${bestPrev ? `${bestPrev.model} fitted to the `
      + `series itself scores ${bestPrev.mase_mean}` : 'a fitted model does better'} and `
    + `${snaivePrev ? `repeating last year scores ${snaivePrev.mase_mean}` : 'the naive rules do worse'}. `
    + `It is not the best forecast of that series. It is a forecast of that series produced by a `
    + `model that was never fitted to it, and it beats several things that were.`);

  set('verdict-intro', `Four situations, and what this page measured about each.`);
  set('verdict-one',
    `${S.local_nbeats < last.mase
      ? `The same architecture fitted to one series scores ${S.local_nbeats} where the shared one `
        + `scores ${last.mase}.`
      : `A shared model reaches ${last.mase} where a single series model gets ${S.local_nbeats}.`} `
    + `Exponential smoothing, with five parameters, is at ${S.ets}.`);
  set('verdict-one-warn',
    `A deep model on one series is mostly a way to overfit slowly. Watch what it is trained on: `
    + `training on windows that overlap the ones being scored gives a beautiful number and no `
    + `forecast.`);
  set('verdict-many',
    `The curve falls from ${first.mase} to ${last.mase} between ${first.series} and `
    + `${last.series} series, and every one of them is a series the model will never see again.`);
  set('verdict-many-warn',
    `Breadth, not history: these are all ${M.length} points long. And the scaling step is not a `
    + `detail, it is the thing that lets one network see all of them.`);
  set('verdict-interp',
    `The constrained basis costs ${gap.toFixed(4)} in accuracy against a seed spread of `
    + `${floor.toFixed(4)}, and recovers the real trend to ${bestDec.trend_err} where the truth is `
    + `in its vocabulary.`);
  set('verdict-interp-warn',
    `And to ${worstDec.trend_err} where it is not. On pure noise the seasonal blocks still emit a `
    + `season, wrong by ${noiseRow ? noiseRow.season_err : ''} of a standard deviation.`);
  set('verdict-attn',
    `Permuting an input and measuring the loss gives a number with a unit: `
    + `${driver.ratio} times for the promotion, ${noiseCov.ratio} for noise.`);
  set('verdict-attn-warn',
    `With duplicated inputs even that misleads: either copy alone costs `
    + `${Math.min(driver.ratio, copy.ratio)} times, both together ${A.both_copies.ratio}.`);

  set('closing-note',
    `Every model in this branch so far has been fitted to the thing it forecasts, or trained on a `
    + `collection that the thing it forecasts belongs to. The last idea in time series drops that `
    + `too: pretrain on an enormous pile of series, then forecast something new with no fitting at `
    + `all, the way a language model completes a sentence it has never read. The last number above `
    + `is a hint that this can work; it is not yet an argument. `
    + `<a href="../zero-shot/">The next article</a> builds it properly, and the awkward question `
    + `underneath it is one that a public benchmark cannot answer and a written down corpus can: `
    + `whether the model has seen your series before.`);

  set('resources-note',
    `${n0(M.n_train + M.n_test)} series of ${M.length} points from ${M.families.length} families `
    + `(${M.families.map((f) => f.replace(/_/g, ' ')).join(', ')}), seed ${M.seed}, digest of the `
    + `first twenty training series <code>${M.corpus_digest}</code>. The stack is `
    + `${B.generic.params.toLocaleString('en-US')} parameters for the generic basis and `
    + `${B.interpretable.params.toLocaleString('en-US')} for the constrained one, six blocks each, `
    + `hidden width ${M.hidden}, lookback ${M.lookback}, horizon ${M.horizon}, trained for `
    + `${M.epochs} epochs on the absolute error with Adam. Accuracy numbers are means over three `
    + `seeds where a seed floor was needed and single runs where the point is a curve. The `
    + `attention model is ${n0(A.n_windows)} windows from ${A.n_train} series with a promotion `
    + `effect averaging ${A.effect}. Run on ${M.threads} threads with torch ${M.torch}.`);
}
