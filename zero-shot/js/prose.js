/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const Q = data.quantisation;
  const P = data.pretrain;
  const S = data.corpus_size;
  const U = data.unseen;
  const X = data.context;
  const B = data.board;
  const floor = Q.rows.find((r) => r.bins === M.bins);
  const cov = P.co2.coverage;
  const m95 = cov.reduce((a, r) => a + r.c95, 0) / cov.length;
  const m80 = cov.reduce((a, r) => a + r.c80, 0) / cov.length;
  const small = S.rows[0];
  const big = S.rows[S.rows.length - 1];
  const bestCtx = X.rows.reduce((a, b) => (b.mase < a.mase ? b : a), X.rows[0]);
  const prevBest = B ? B.rows.reduce((a, b) => (b.mase_mean < a.mase_mean ? b : a), B.rows[0])
    : null;
  const prevNaive = B ? B.rows.find((r) => r.model === 'seasonal naive') : null;
  const beaten = B ? B.rows.filter((r) => r.mase_mean > P.co2.mase) : [];

  set('opening-note',
    `The model on this page is ${n0(M.params)} parameters, trained for ${M.epochs} passes over `
    + `${n0(M.n_windows)} windows drawn from ${n0(M.n_series)} series, none of which is real. `
    + `Every one of them came out of the same written down generator this branch has been using: `
    + `trends, seasons, random walks, level shifts, noise. Then it is handed the `
    + `${n0(B ? B.n_origins : P.co2.n_origins)} origins of the carbon dioxide record from `
    + `<a href="../arima/">the first article</a>, the same series, the same origins, the same `
    + `units, and asked to continue. <span class="bold">It cannot have seen that series, because `
    + `its training data contains no real series at all</span>, and that sentence is the reason `
    + `this article exists: every published foundation forecaster has to argue it, and none of `
    + `them can prove it.`);

  set('scrolly-intro',
    `The trick that makes a language model swallow a series is not a neural network at all. It is `
    + `three lines of arithmetic that turn any window of any series, in any units, into a string `
    + `of symbols from a fixed alphabet of ${M.bins}.`);

  set('scrolly-step-0',
    `Here is ${M.context} months of carbon dioxide, in parts per million. Nothing about those `
    + `numbers is portable: another series would be in dollars, or degrees, or beats per minute, `
    + `and would live at a completely different level.`);

  set('scrolly-step-1',
    `Subtract this window's own mean and divide by its own standard deviation. Every window from `
    + `every series in the world now lives in the same place, and a model trained on one can be `
    + `shown another. This is the same normalisation the previous article ablated, doing a `
    + `different job: there it was so that one network could see many series, here it is so that `
    + `one alphabet can.`);

  set('scrolly-step-2',
    `Clip at plus or minus ${M.clip} standard deviations and round into ${M.bins} bins. The `
    + `information that survives is exactly what the bin edges allow, which is the subject of the `
    + `widget below.`);

  set('scrolly-step-3',
    `And that is it. The model sees ${M.context} integers and is asked for ${M.horizon} more, `
    + `exactly as a language model sees tokens and is asked for the next ones. It has no idea `
    + `there is a series behind them, no notion of a trend or a season, and no parameter that `
    + `belongs to your data.`);

  set('token-note',
    `That rounding is a real loss and it can be measured before any model exists. Give an oracle `
    + `the true next twelve months and ask it only to write them down in this alphabet and read `
    + `them back: the error it makes is a floor under everything a model reading these letters `
    + `could ever achieve.`);

  set('alphabet-note',
    `At ${M.bins} letters that floor is ${floor.mase}, in the same units as every other error in `
    + `this branch, and the model that reads them scores ${P.co2.mase}, `
    + `${(P.co2.mase / floor.mase).toFixed(0)} times higher. So the tokeniser is nowhere near the `
    + `limiting factor here, which is worth knowing because it is the part people find suspicious. `
    + `The bins would start to matter around ${Q.rows[0].bins} letters, where the floor is `
    + `${Q.rows[0].mase} and the step between neighbouring letters is ${Q.rows[0].step_ppm} parts `
    + `per million.`);

  set('forecast-intro',
    `Now the forecast itself. The model produces a distribution over the next letter, so a `
    + `forecast is a <span class="bold">sample</span>, and a hundred samples are a hundred `
    + `possible futures. The point forecast is their median and the interval is their percentiles: `
    + `both fall out of the same hundred draws.`);

  set('forecast-note',
    `<span class="value">${P.co2.mase}</span> on ${P.co2.n_origins} origins of a series it was `
    + `never fitted to. `
    + `${beaten.length
      ? `That is better than ${beaten.map((b) => b.model).join(', ')} from the first article of `
        + `this branch, all of which were fitted to this series, and worse than `
        + `${B.rows.filter((r) => r.mase_mean <= P.co2.mase).map((r) => r.model).join(' and ')}.`
      : `That is worse than everything the first article fitted to this series.`} `
    + `On the sunspots, where nothing in its corpus resembles an eleven year cycle that is never `
    + `the same length twice, it scores ${P.sunspots.mase} over ${P.sunspots.n_origins} origins. `
    + `The interval is the part that comes free: ${pc(m95, 1)} coverage against a promised 95% and `
    + `${pc(m80, 1)} against 80%, with no distributional assumption anywhere in the pipeline.`);

  set('corpus-intro',
    `A foundation model is a claim about the corpus, so the corpus is the thing to vary. Same `
    + `architecture, same training length, same evaluation, three sizes of pile.`);

  const ct = document.querySelector('#corpus-table');
  if (ct) {
    ct.innerHTML = S.rows.map((row) => `<tr>
      <td><span class="bold">${row.series} series</span> (${pc(row.share, 0)} of the pile)</td>
      <td>${n0(row.windows)}</td>
      <td><span class="value">${row.mase}</span></td>
      <td>${(row.mase / big.mase).toFixed(3)}x</td>
    </tr>`).join('');
  }

  set('corpus-note',
    `${small.mase > big.mase
      ? `More series is better, and the gap between the smallest and the largest pile is `
        + `${(small.mase / big.mase).toFixed(2)} times. `
      : `More series did not help here: ${small.series} series scored ${small.mase} and `
        + `${big.series} scored ${big.mase}. `}`
    + `Each row is one training run, so small differences between neighbouring rows are one seed `
    + `apiece and should not be read as a trend; the two ends are far enough apart to mean `
    + `something. Note also what this axis is not: every series in the pile is `
    + `${M.length} points long, so a bigger pile is more <i>variety</i> and never more history. `
    + `And how much of the history does it use at all? Feeding it `
    + `${X.rows.map((r) => r.context).join(', ')} months of context and asking for the same twelve `
    + `gives ${X.rows.map((r) => r.mase).join(', ')}, so the best is `
    + `${bestCtx.context} months` + `${bestCtx.context === X.trained_on
      ? `, which is exactly the length it was trained on: a context window is not a suggestion, `
        + `it is part of the model.`
      : `, against the ${X.trained_on} it was trained on, which is a reminder that a context `
        + `window is part of the model and not a knob.`}`);

  set('unseen-intro',
    `Here is the experiment a public benchmark cannot run. Take the corpus, remove an entire `
    + `family of series from it, ${U.held_out.replace(/_/g, ' ')}, retrain from scratch, and then `
    + `forecast series from exactly that family. One model has seen the shape before and the other `
    + `has never seen it in its life.`);

  const ut = document.querySelector('#unseen-table');
  if (ut) {
    const rows = Object.entries(U.rows);
    ut.innerHTML = rows.map(([name, row]) => `<tr>
      <td><span class="bold">${name}</span></td>
      <td>${row.families}</td>
      <td><span class="value">${row.mase}</span></td>
      <td>${(row.mase / U.rows['saw it'].mase).toFixed(3)}x</td>
    </tr>`).join('');
  }

  set('unseen-note',
    `${U.gap > 0
      ? `Removing the family costs ${U.gap} in error, a factor of ${U.ratio}. The model that never `
        + `saw the shape still forecasts it, because a trend with a season is not unrelated to a `
        + `damped trend or a bare season, and what it lost is the part that was specific.`
      : `Removing the family cost nothing measurable (${U.rows['never saw it'].mase} against `
        + `${U.rows['saw it'].mase}), which says the other families already contain everything `
        + `this one needed.`} `
    + `Either way the number exists, and that is the whole point. When a foundation forecaster `
    + `reports a benchmark result, this is the number nobody can compute: the corpora are `
    + `enormous, assembled from public repositories, and the benchmarks are public too. `
    + `<span class="bold">The claim "it had never seen this series" is usually an assertion about `
    + `a pile nobody has read.</span> Here the pile was written down by a generator in this `
    + `repository, so the claim is a fact.`);

  set('board-intro',
    `Everything this branch has fitted to the carbon dioxide record, on the same `
    + `${B ? B.n_origins : P.co2.n_origins} origins and in the same units.`);

  const bt = document.querySelector('#board-table');
  if (bt && B) {
    const rows = B.rows.map((x) => ({
      model: x.model, fitted: 'yes', h1: x.mase[0], mean: x.mase_mean,
    })).concat([{
      model: 'this page, pretrained on written down series',
      fitted: 'no', h1: P.co2.per_h[0], mean: P.co2.mase,
    }]).sort((a, b) => a.mean - b.mean);
    bt.innerHTML = rows.map((row) => `<tr>
      <td><span class="bold">${row.model}</span></td>
      <td>${row.fitted}</td>
      <td>${row.h1}</td>
      <td><span class="value">${row.mean}</span></td>
    </tr>`).join('');
  }

  const fittedStillWins = prevBest && prevBest.mase_mean < P.co2.mase;
  set('board-note',
    `${prevBest ? (fittedStillWins
      ? `${prevBest.model} still wins at ${prevBest.mase_mean}, and it should: it is a `
        + `model whose shape was designed for exactly this kind of series and whose parameters were `
        + `estimated from this series at every origin. `
      : `Nothing fitted to this series beats the pretrained model here: the best of them is `
        + `${prevBest.model} at ${prevBest.mase_mean}. `) : ''}`
    + `The row worth arguing about is the last one to be added. A model that has never been shown `
    + `a single real measurement of anything scores ${P.co2.mase}`
    + `${prevNaive ? `, which is ${(prevNaive.mase_mean / P.co2.mase).toFixed(1)} times better `
      + `than repeating last year` : ''}, and it did it in one forward pass with nothing fitted. `
    + `That is not the end of forecasting and it is not a toy either: it is a different trade. `
    + `Everything before it in this branch spends estimation on your series; this spends training `
    + `on somebody else's and then spends nothing on yours.`);

  set('verdict-intro', `Four situations, and what this page measured about each.`);
  set('verdict-quick',
    `${P.co2.mase} on a series it never saw, in one pass, against `
    + `${prevNaive ? prevNaive.mase_mean : ''} for repeating last year.`);
  set('verdict-quick-warn',
    `It has no idea what your series is. On the sunspots, whose cycle nothing in the corpus `
    + `resembles, it scores ${P.sunspots.mase}.`);
  set('verdict-fit',
    `${prevBest ? `${prevBest.model} scores ${prevBest.mase_mean} on the same origins, which is `
      + `${fittedStillWins
        ? `${(P.co2.mase / prevBest.mase_mean).toFixed(1)} times better than the pretrained model.`
        : `${(prevBest.mase_mean / P.co2.mase).toFixed(1)} times worse than the pretrained model, `
          + `so on this series fitting bought nothing.`}` : ''}`);
  set('verdict-fit-warn',
    `${fittedStillWins
      ? `That advantage is real on one well behaved series and is exactly what disappears when you `
        + `have ten thousand of them and no time to look at any.`
      : `One series is one series, and a fitted model that loses here is not a fitted model that `
        + `loses everywhere.`}`);
  set('verdict-interval',
    `A hundred sampled continuations give ${pc(m95, 1)} coverage against a promised 95%, with no `
    + `distributional assumption.`);
  set('verdict-interval-warn',
    `The samples are only as good as the alphabet: the floor at ${M.bins} letters is `
    + `${floor.mase}, and a narrow interval cannot be narrower than a bin.`);
  set('verdict-leak',
    `Removing one family from the corpus and retraining measures what having seen the shape was `
    + `worth: ${U.ratio} times here.`);
  set('verdict-leak-warn',
    `You can only run that experiment if you know what is in the corpus. For every published `
    + `foundation forecaster, you do not.`);

  set('closing-note',
    `That is the branch: from a model with three parameters fitted to one series, to a model with `
    + `${n0(M.params)} fitted to none of them. What has not changed once in five articles is the `
    + `shape of the data, a single number arriving at regular intervals, and the shape of the `
    + `question, what comes next. `
    + `<a href="../spectrograms/">The next article</a> keeps both and changes the rate: `
    + `eight thousand numbers a second instead of one a month. At that rate nothing in this branch `
    + `works, because the interesting structure is not in the values at all, it is in how fast `
    + `they wiggle, and getting at it requires turning the sound into a picture first.`);

  set('resources-note',
    `The corpus is ${n0(M.n_series)} series of ${M.length} points from seven written down `
    + `families, seed ${M.seed}, tokenised at ${M.bins} letters over plus or minus ${M.clip} `
    + `window standard deviations, giving ${n0(M.n_windows)} training windows of `
    + `${M.context} + ${M.horizon}. The model is a ${n0(M.params)} parameter decoder: two masked `
    + `self attention blocks of width 96 with four heads, trained ${M.epochs} epochs with AdamW on `
    + `the next letter. Forecasts are the median of 100 sampled continuations, and intervals are `
    + `their percentiles. The real series are the same carbon dioxide record (digest `
    + `<code>${M.co2_digest}</code>) and sunspot counts (digest <code>${M.sun_digest}</code>) as `
    + `the first article of this branch, which this generator asserts before it runs. Trained on `
    + `${M.threads} threads with torch ${M.torch}.`);
}
