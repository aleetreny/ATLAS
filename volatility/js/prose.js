/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.two_r2;
  const G = data.garch;
  const F = data.forecast;
  const V = data.vix;
  const R = data.risk;
  const A = data.var;
  const I = data.irf;
  const N = data.granger;
  const best = A.table[0];
  const bestAr = A.table.find((r) => r.model.startsWith('AR'));
  const worst = A.table[A.table.length - 1];
  const meanRow = A.table.find((r) => r.model === 'mean');
  const garchT = R.rows.find((r) => r.model === 'GARCH t');
  const constant = R.rows.find((r) => r.model === 'normal constant');
  const fBest = F.rows[0];
  const fConst = F.rows.find((r) => r.model === 'constant');

  set('opening-note',
    `The data is the same size as the last article's and nothing else about it is. `
    + `<span class="bold">${n0(M.ret_n)} daily returns of the S&amp;P 500</span>, `
    + `${M.sp500.start} to ${M.sp500.end}, which includes the whole of 2008; and `
    + `<span class="bold">${n0(M.macro_n)} quarters of US national accounts</span>, four series `
    + `at once, ${M.macro.start} to ${M.macro.end}. On the first, forecasting the next value is `
    + `close to hopeless and everyone in the field agrees: the mean of a daily return is `
    + `${G.mean} percent and the useful part of the model is somewhere else entirely. On the `
    + `second, four series that obviously move together turn out to be `
    + `${meanRow && best ? `barely worth modelling jointly at all` : 'worth modelling jointly'}, `
    + `which is a duller result than the textbook promises and a more useful one.`);

  set('scrolly-intro',
    `Take the daily return of an index, the log of today's close over yesterday's. Ask it two `
    + `questions with the same machinery: can the past predict the sign of the next one, and can `
    + `the past predict its size?`);

  set('scrolly-step-0',
    `Twenty years of days. The worst is ${G.worst_day.ret} percent on `
    + `${G.worst_day.label}, and there is no need for a statistic to see the structure: the `
    + `violent days arrive together. The distribution underneath is the giveaway. A normal with `
    + `the same standard deviation is drawn over it, and the real thing is taller in the middle `
    + `and far heavier at the edges, which is what "fat tails" means when it is a picture.`);

  set('scrolly-step-1',
    `Regress the return on its own last ${T.lags} days: the R-squared is `
    + `<span class="bold">${T.r2_return}</span>. The correlogram says the same thing: the largest `
    + `autocorrelation in forty lags is ${Math.max(...T.acf_return.slice(1).map(Math.abs)).toFixed(4)} `
    + `against a band of ${T.band}. There is essentially nothing there, and there is a reason `
    + `there is nothing there: anything a regression this simple could find would be free money, `
    + `and free money does not stay on the floor.`);

  set('scrolly-step-2',
    `Now the same regression on the absolute return: R-squared `
    + `<span class="bold">${T.r2_abs}</span>, ${(T.r2_abs / T.r2_return).toFixed(0)} times the `
    + `first. The correlogram of the absolute return is positive at every one of the forty lags, `
    + `${T.acf_abs[1]} at one day and still ${T.acf_abs[40]} two months later. The size of a `
    + `surprise has a memory that lasts for months.`);

  set('scrolly-step-3',
    `Both correlograms, same axes. This is the whole reason the next section exists: the series `
    + `has almost no forecastable mean and a strongly forecastable spread, so a model of the mean `
    + `is the wrong instrument and a model of the variance is the right one. And the variance is `
    + `<span class="bold">never observed</span>. Every day gives one number; the spread it was `
    + `drawn from does not leave a record.`);

  set('returns-note',
    `That last point is what makes this different from every fitting problem in the atlas so far. `
    + `There is no column of targets. A day whose variance was high can easily deliver a small `
    + `return, and a day whose variance was low can deliver a large one, so no error can be `
    + `computed against a truth: the truth was never written down. What can be computed is how `
    + `likely the observed returns are under a proposed path of variances, and that is the only `
    + `thing holding the next section up.`);

  set('garch-intro',
    `GARCH is one line: today's variance is a constant, plus a slice of yesterday's squared `
    + `surprise, plus a slice of yesterday's variance. Three numbers, and everything the model can `
    + `express is in them. The likelihood underneath is what scores a proposed path, since the `
    + `path itself is invisible.`);

  set('garch-equation-note',
    `The sliders below run that recursion over all ${n0(M.ret_n)} days in your browser and report `
    + `the likelihood as you go. Try to beat the fitted model: the answer is omega ${G.omega}, `
    + `alpha ${G.alpha}, beta ${G.beta}, and the two buttons beside "fit it" put the model into `
    + `the two states worth seeing at least once, one with no memory and one whose memory never `
    + `fades.`);

  set('garch-note',
    `The number to take away is <span class="bold">${G.persistence}</span>: alpha plus beta, the `
    + `fraction of today's variance that is still there tomorrow. It implies a half life of `
    + `${G.half_life} trading days, about ${(G.half_life / 21).toFixed(1)} months, and it is close `
    + `enough to one that the model is almost a random walk in the variance. Beta at ${G.beta} is `
    + `doing most of that: what the model mostly says is "tomorrow is like today", with a small `
    + `correction ${G.alpha} for what just happened. The peak of the fitted path is `
    + `${G.peak_vol.vol_annual}% annualised on ${G.peak_vol.label}, against a long run level of `
    + `${G.uncond_vol_annual}%. `
    + `And it beats its rivals out of sample on ${n0(F.n_scored)} days: by the QLIKE loss, `
    + `${fBest.model} at ${fBest.qlike} and a constant variance at ${fConst.qlike}, `
    + `with an exponentially weighted average, the recursion GARCH contains as a special case, `
    + `${(F.rows[1].qlike - fBest.qlike).toFixed(5)} behind the winner.`);

  set('vix-intro',
    `A likelihood scores a model against itself. There is a second forecast of the same quantity `
    + `published every trading day, made by a different mechanism entirely: the VIX, which is what `
    + `option prices imply about the next thirty <i>calendar</i> days of volatility. Thirty `
    + `calendar days is about ${V.horizon} trading days, so that is the window the model is asked `
    + `for as well, and there are three numbers for it: what the model said, what the market said, `
    + `and what happened.`);

  set('vix-note',
    `Two readings, and the second one is the one nobody teaches. First, the market's number is `
    + `${V.corr_vix_real > V.corr_garch_real ? 'the better forecast' : 'the worse forecast'} `
    + `(${V.corr_vix_real} against ${V.corr_garch_real} correlation with what actually happened), `
    + `which is not surprising given it is priced by people who can also see the news. Second, and `
    + `more interesting: the VIX averages ${V.mean_vix} while the realised volatility of the `
    + `following month averages ${V.mean_real}, a gap of ${V.premium} points that does not close. `
    + `That is not the market being wrong ${n0(V.n)} days in a row. It is the price of insurance `
    + `sitting above its expected cost, which is what insurance does, and it means a volatility `
    + `forecast and a volatility price are two different objects that happen to be quoted in the `
    + `same units.`);

  set('risk-intro',
    `Everything so far has been scored by likelihood or by correlation. Here is the score that `
    + `pays people's salaries: draw a line such that the loss should only go past it one day in a `
    + `hundred, and then count. Four ways of drawing it, ${n0(R.n_days)} days.`);

  set('risk-note',
    `The one percent line is missed by every model here, and the direction is always the same: `
    + `more days past the line than promised. The best is ${garchT.model} at `
    + `${pc(garchT.obs1, 2)} where ${pc(0.01, 0)} was claimed, and it is still rejected by the `
    + `count (p = ${garchT.kupiec1}). Modelling the variance is not the same as modelling the `
    + `tail, and a Student t innovation with ${G.nu} degrees of freedom gets closer without `
    + `getting there. `
    + `The row worth reading twice is the constant variance one: at one percent it is too `
    + `optimistic (${pc(constant.obs1, 2)} against 1%) and at five percent it is too `
    + `pessimistic (${pc(constant.obs5, 2)} against 5%). One number for the spread cannot be `
    + `both calm enough for a calm year and wide enough for 2008, so it is wrong in both `
    + `directions at once, and only counting at two levels shows it.`);

  set('var-intro',
    `Now the other question. Four macroeconomic series that plainly move together: output, `
    + `consumption, investment and unemployment, as quarterly growth rates. A vector `
    + `autoregression writes every series as a linear function of the recent past of `
    + `<i>all</i> of them, which is the natural thing to want and has a price written on the `
    + `front of it.`);

  set('var-count',
    `With k series and p lags that is k<sup>2</sup>p coefficients plus k constants: `
    + `${A.counts.map((c) => `${c.var_params} at p = ${c.p}`).join(', ')}, against `
    + `${A.counts.map((c) => c.ar_params).join(', ')} for the same p if each series is modelled `
    + `on its own. The table below is ${A.n_origins} rolling origins, ${A.horizon} quarters `
    + `ahead, every error divided by the in sample mean absolute change of its own column so the `
    + `four series can share a number.`);

  const vt = document.querySelector('#var-table');
  const params = Object.fromEntries(A.counts.map((c) => [c.p, c]));
  if (vt) {
    vt.innerHTML = A.table.map((row) => {
      const p = +(row.model.match(/\((\d)\)/) || [])[1];
      const isVar = row.model.startsWith('VAR');
      const np = row.model === 'mean' ? 4
        : (isVar ? params[p].var_params : params[p].ar_params);
      return `<tr>
        <td><span class="bold">${row.model}</span></td>
        <td>${np}</td>
        <td><span class="value">${row.overall}</span></td>
        <td>${row.model === best.model ? 'the best' : `${row.t} (gap ${row.gap_vs_best})`}</td>
        <td>${row.model === best.model ? ''
    : (row.resolves ? 'yes, it is worse' : 'no')}</td>
      </tr>`;
    }).join('');
  }

  set('var-note',
    `${best.model} is on top, and the row under it is what the table is about: `
    + `${bestAr.model} costs ${bestAr.gap_vs_best} more, which is ${bestAr.t} standard errors of `
    + `the paired difference across origins. `
    + `${bestAr.resolves
      ? `That resolves, so the joint model does earn its parameters here.`
      : `<span class="bold">That does not resolve.</span> Four separate univariate models are `
        + `indistinguishable from the joint one that has ${params[1].var_params} parameters `
        + `instead of ${params[1].ar_params}, on the series that are supposed to be the textbook `
        + `case for modelling them together.`} `
    + `Two more rows say the rest. Forecasting each series by its own historical mean scores `
    + `${meanRow.overall}, which is ${(meanRow.overall / best.overall).toFixed(3)} times the best `
    + `model in the table, and ${worst.model} with ${params[4].var_params} parameters is the worst `
    + `thing here at ${worst.overall}. Quarterly growth rates are mostly noise, the vector model `
    + `spends its parameters discovering that, and the honest summary of this table is that the `
    + `interesting thing a vector autoregression does is not forecasting.`);

  set('irf-intro',
    `What it does instead is answer questions of the form "if investment is shocked, what happens `
    + `to everything else, and for how long". That is an impulse response, it is the reason these `
    + `models are in every central bank, and it needs one more assumption than the forecast does.`);

  set('irf-note',
    `The two buttons are the same fitted model. The ordering is a claim about which variable can `
    + `react to which within a single quarter, it cannot be read off the data, and it moves the `
    + `answer by up to ${I.max_gap} percent on ${I.orders['gdp first'].order.length} series. The `
    + `response of output to an investment shock peaks ${I.peak.realgdp.ratio} times higher under `
    + `one ordering than the other. Any paper that prints one of these charts has made that `
    + `choice; the good ones say so and show the other one.`);

  set('granger-intro',
    `And then the phrase. "X Granger causes Y" means only that the past of X improves a forecast `
    + `of Y beyond Y's own past, which is a statement about forecasting and not about causing, and `
    + `Granger said so. The famous failure is what happens when both series wander. Below, two `
    + `<span class="bold">independent</span> random walks, ${n0(N.n)} pairs of them, `
    + `${N.length} points each, with nothing whatsoever connecting them.`);

  const gt = document.querySelector('#granger-table');
  if (gt) {
    gt.innerHTML = `
      <tr>
        <td><span class="bold">Regress one on the other, no lags, read the t statistic</span></td>
        <td><span class="value">${pc(N.classic_levels_rate, 1)}</span> called significant</td>
        <td>${pc(N.classic_diffs_rate, 1)}</td>
        <td>5%</td>
      </tr>
      <tr>
        <td><span class="bold">The Granger test, ${N.lag} lags of both</span></td>
        <td><span class="value">${pc(N.levels_rate, 1)}</span> called significant</td>
        <td>${pc(N.diffs_rate, 1)}</td>
        <td>5%</td>
      </tr>`;
  }

  set('granger-note',
    `Two procedures, two very different answers, and the difference is the point. The plain `
    + `regression is the classic spurious result: ${pc(N.classic_levels_rate, 1)} of `
    + `${n0(N.n)} independent pairs come out significant, with a median absolute t statistic of `
    + `${N.classic_median_t} and a median R-squared of ${N.classic_median_r2}, all of it from `
    + `nothing. The Granger test on the same pairs rejects `
    + `${pc(N.levels_rate, 1)} of the time, `
    + `${N.levels_rate > 2 * N.nominal
      ? `which is still ${(N.levels_rate / N.nominal).toFixed(1)} times what it promised`
      : `which is close to the ${pc(N.nominal, 0)} it promised`}, because it already includes the `
    + `target's own lags and those soak up most of the wandering. So the folklore is attached to `
    + `the wrong test: it is the naive regression that manufactures ${pc(N.classic_levels_rate, 0)} `
    + `false discoveries, not the Granger test. Difference both series first and both procedures `
    + `come back to ${pc(N.classic_diffs_rate, 1)} and ${pc(N.diffs_rate, 1)}, right where they `
    + `should be. On the real accounts, in growth rates, the strongest pair is `
    + `${data.granger.real[0].cause} to ${data.granger.real[0].effect}, and it means what it says `
    + `on the tin: it helps a forecast.`);

  set('verdict-intro', `Four things you might want from a series like these, and what each cost.`);
  set('verdict-mean',
    `Five lags of a daily return explain ${T.r2_return} of the next one. On quarterly growth `
    + `rates, the historical mean scores ${meanRow.overall} against ${best.overall} for the best `
    + `fitted model.`);
  set('verdict-mean-warn',
    `A backtest that looks profitable on an R-squared of ${T.r2_return} is measuring transaction `
    + `costs and luck.`);
  set('verdict-vol',
    `The same five lags explain ${T.r2_square} of the next squared return, and the fitted `
    + `recursion beats a constant on ${n0(F.n_scored)} held out days (${fBest.qlike} against `
    + `${fConst.qlike}).`);
  set('verdict-vol-warn',
    `Persistence of ${G.persistence} is close enough to one that a long horizon forecast is `
    + `mostly assumption, and the market's own number correlates better with the outcome `
    + `(${V.corr_vix_real}) than the model does (${V.corr_garch_real}).`);
  set('verdict-risk',
    `At the five percent line the two GARCH models are indistinguishable from their promise `
    + `(p = ${R.rows.find((r) => r.model === 'GARCH normal').kupiec5} and ${garchT.kupiec5}).`);
  set('verdict-risk-warn',
    `At the one percent line all four are rejected, the best of them at ${pc(garchT.obs1, 2)} `
    + `against ${pc(0.01, 0)}. A conditional variance is not a tail model.`);
  set('verdict-var',
    `An impulse response and a Granger test are what the joint model is really for, and both are `
    + `readable off ${params[1].var_params} coefficients.`);
  set('verdict-var-warn',
    `Not the forecast: ${bestAr.t} standard errors separate the joint model from four separate `
    + `ones, and ${worst.model} with ${params[4].var_params} parameters is the worst row in the `
    + `table.`);

  set('closing-note',
    `Both halves of this article fitted a shape written down in advance and estimated a handful of `
    + `numbers inside it: three for the variance, ${params[1].var_params} for the vector model. `
    + `That is the whole classical bargain, and it has been the bargain for fifty years. The other `
    + `way to forecast a series is to stop writing the shape down: turn the series into a table of `
    + `features, one row per time point, and hand it to any of the regressors from `
    + `<a href="../gradient-boosting/">module 1</a>, which is what most working forecasters `
    + `actually do now. It works, and it fails in ways the classical models cannot fail, starting `
    + `with the fact that a tree cannot produce a number it has never seen. `
    + `<a href="../time-features/">The next article</a> builds that table.`);

  set('resources-note',
    `${n0(M.ret_n)} daily log returns of the S&amp;P 500 in percent (${M.sp500.start} to `
    + `${M.sp500.end}, bundled with arch ${M.arch}, digest <code>${M.ret_digest}</code>), the `
    + `daily VIX over ${V.start} to ${V.end}, and ${n0(M.macro_n)} quarters of US accounts `
    + `(bundled with statsmodels ${M.statsmodels}, digest <code>${M.macro_digest}</code>). The `
    + `variance model is fitted by exact Gaussian likelihood on all ${n0(G.n)} days; the recursion `
    + `starts at omega + (alpha + beta) times the backcast ${G.backcast}, which is where the `
    + `library starts it, and starting it at the sample variance instead moves the path by `
    + `${G.start_gap}. This page's own filter agrees with the library's to `
    + `${G.agreement.toExponential(1)}. The out of sample comparison refits every `
    + `${F.refit_every} days from day ${n0(F.start)} onwards and filters in between, ${n0(F.n_scored)} `
    + `scored days. The spurious causality study is ${n0(N.n)} pairs of independent random walks `
    + `of ${N.length} points, seed ${M.seed}. Run on ${M.threads} threads with numpy ${M.numpy}.`);
}
