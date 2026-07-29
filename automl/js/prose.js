/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const f3 = (v) => Number(v).toFixed(3);
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const signed = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(4)}`;

export const PROSE_IDS = [
  'opening-note', 'gain-intro', 'gain-note', 'axes-intro', 'axes-note',
  'curse-intro', 'curse-note', 'port-intro', 'port-note', 'ens-intro', 'ens-note',
  'verdict-intro', 'verdict-search', 'verdict-search-warn', 'verdict-port',
  'verdict-port-warn', 'verdict-ens', 'verdict-ens-warn', 'verdict-nested',
  'verdict-nested-warn', 'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.tables;
  const P = data.portfolio;
  const E = data.ensemble;
  const N = data.nested;
  const names = Object.keys(T);

  const gains = names.map((n) => ({
    name: n, gain: T[n].best.test - T[n].default.test,
    sd: T[n].best.test_sd + T[n].default.test_sd,
  }));
  const real = gains.filter((x) => Math.abs(x.gain) > x.sd);
  const bigGain = gains.reduce((a, b) => (b.gain > a.gain ? b : a));
  const spans = names.map((n) => {
    const a = T[n].axes;
    return { name: n, ranked: Object.entries(a).map(([k, v]) => [k, v.span])
      .sort((p, q) => q[1] - p[1]) };
  });
  const winners = {};
  spans.forEach((s) => { winners[s.ranked[0][0]] = (winners[s.ranked[0][0]] || 0) + 1; });
  const topAxis = Object.entries(winners).sort((a, b) => b[1] - a[1])[0];
  const curses = names.map((n) => {
    const c = T[n].curse;
    return { name: n, one: c[0].gap, many: c[c.length - 1].gap };
  });
  const worstCurse = curses.reduce((a, b) => ((b.many - b.one) > (a.many - a.one) ? b : a));
  const pf = P.rows.find((r) => r.arm === 'portfolio');
  const rd = P.rows.find((r) => r.arm === 'random');
  const full = P.rows.find((r) => r.arm === 'full');
  const ensGains = names.map((n) => {
    const rows = E[n].rows;
    return { name: n, gain: rows[rows.length - 1].test - T[n].best.test,
      sd: T[n].best.test_sd };
  });
  const ensReal = ensGains.filter((x) => x.gain > x.sd);

  set('opening-note',
    `The catalogue is four decisions multiplied together: ${M.scalers.length} ways to scale `
    + `the columns, ${M.reducers.length} ways to reduce them, ${M.weights.length} ways to `
    + `weight the classes and ${M.models.length} model settings, which is `
    + `<span class="bold">${n0(M.configs)} pipelines</span>. Each one was fitted `
    + `${M.folds} times for cross validation and once more on everything, on each of four `
    + `tables: ${names.join(', ')}. Three of them are the tables scikit-learn ships and the `
    + `fourth is ${n0(M.r8_docs)} Reuters newswires turned into ${M.r8_dims} dense columns by `
    + `the pipeline the <a href="../tf-idf/">TF-IDF article</a> measured, which is here `
    + `because a text table behaves differently from a table of measurements and one of the `
    + `claims on this page is about what transfers between tables.`);

  set('gain-intro',
    `Start with the question anyone paying for a search wants answered: how much better is `
    + `the winner than what you would have got by not searching at all. The comparison needs `
    + `a defined starting point, and the honest one is the pipeline that comes out of the `
    + `box: no scaling, no reduction, no weighting, logistic regression at the library's `
    + `default penalty, which is <span class="mono">${M.default}</span>.`);

  set('gain-note',
    `Searching all ${n0(M.configs)} pipelines buys `
    + `${gains.map((x) => `${signed(x.gain)} on ${x.name}`).join(', ')}. `
    + (real.length === 0
      ? `<span class="bold">None of those four gains is larger than the wobble of the held `
        + `out set it was measured on</span>, which is the result and not an excuse: on `
        + `tables this size, the catalogue is worth less than the uncertainty of the number `
        + `used to judge it.`
      : `${real.length} of the four are larger than the wobble of the held out set they were `
        + `measured on; the largest is ${signed(bigGain.gain)} on ${bigGain.name}. `)
    + ` The faint tick on each row is the best held out score in the whole catalogue, and `
    + `the gap between it and the ring is unavoidable: choosing by a noisy score cannot `
    + `land on the best held out pipeline except by luck.`);

  set('axes-intro',
    `The next question is where any gain came from, and it is usually answered with a `
    + `preference rather than a measurement. Here all four decisions can be put on one axis, `
    + `because for each one there is a quantity that means the same thing: fix that decision `
    + `at a level, and ask what the best score still reachable is once everything else is `
    + `free to adapt.`);

  set('axes-note',
    `That number is a maximum over the rest of the pipeline, so a short bar does not mean `
    + `the decision is irrelevant. It means the rest of the pipeline can recover from it, `
    + `which is a different and more useful thing to know. Across the four tables, `
    + `<span class="bold">${topAxis[0]}</span> is the widest of the four on `
    + `${topAxis[1]} of them. ` + spans.map((s) => `On ${s.name} the order is `
      + `${s.ranked.map(([k, v]) => `${k} ${f3(v)}`).join(', ')}`).join('. ') + `.`);

  set('curse-intro',
    `The <a href="../hyperparameters/">previous article</a> measured the winner's curse over `
    + `a grid of one model and found it real but small. Here the catalogue is a different `
    + `shape: fewer candidates, `
    + `but candidates that differ from each other far more, and tables that are much smaller `
    + `than a thousand rows. Both of those make the same effect bigger.`);

  set('curse-note',
    `Try one pipeline and the difference between its cross validated score and its held out `
    + `score averages ${curses.map((c) => signed(c.one)).join(', ')} on the four tables. Try `
    + `all ${n0(M.configs)} and keep the winner, and the same difference becomes `
    + `${curses.map((c) => signed(c.many)).join(', ')}. The worst of the four is `
    + `<span class="bold">${worstCurse.name}</span>, where looking through the catalogue adds `
    + `${signed(worstCurse.many - worstCurse.one)} to the reported score without adding `
    + `anything to the model. The fix is not to search less. The fix is that the number you `
    + `report has to come from somewhere the search never touched, and that is what nested `
    + `cross validation is: the whole search repeated inside every outer fold, `
    + `${n0(N.fits)} fits for one honest number.`);

  set('port-intro',
    `Everything so far treats each table as if it were the first table anyone had ever seen. `
    + `An automated system does not: it arrives having been run on other people's data, and `
    + `the cheapest form of that memory is a short list of configurations that between them `
    + `cover a lot of tables.`);

  set('port-note',
    `The list is built greedily on ${P.learnt_on.join(', ')} and spent, cold, on `
    + `${P.cold}. Greedily matters here and it is worth one sentence: each entry is chosen `
    + `to maximise the <span class="bold">best of the list</span> across tables, not its `
    + `average, so the second entry is picked to cover the tables where the first one fails. `
    + `That is why the list is not just the top ${P.size} pipelines.`);

  set('ens-intro',
    `One thing is left, and it is the piece that actually wins competitions. A search fits `
    + `hundreds of models and then throws away all but one of them. They are already trained, `
    + `their predictions are already computed, and the second best is not usually wrong in `
    + `the same places as the best.`);

  set('ens-note',
    `The selection is greedy with replacement on the out of fold predictions, which is the `
    + `only set that can choose an ensemble without looking at held out rows. Averaging up to `
    + `${E[names[0]].rows.length} picks buys `
    + `${ensGains.map((x) => `${signed(x.gain)} on ${x.name}`).join(', ')}, `
    + (ensReal.length === 0
      ? `and none of those clears the wobble of its own held out set either. On tables this `
        + `size, the honest summary of this page is that everything after the first fit is `
        + `inside the noise.`
      : `and ${ensReal.length} of the four clear the wobble of their own held out set, which `
        + `is more than the search itself managed on ${real.length}.`)
    + ` The cost is the one number in its favour: nothing new is fitted.`);

  set('verdict-intro',
    `Four pieces, and they are not equally defensible. The column on the right is the one `
    + `that decides whether any of this is worth doing, and it is written in the same units `
    + `as the column in the middle.`);

  set('verdict-search',
    `${signed(bigGain.gain)} at best across the four tables, from `
    + `${n0(M.configs)} pipelines each fitted ${M.folds + 1} times.`);
  set('verdict-search-warn',
    real.length === 0
      ? `Every gain was inside the bootstrap spread of the set used to judge it. On tables of `
        + `a few hundred rows the search is measuring its own noise.`
      : `${M.configs * (M.folds + 1)} fits per table, and ${gains.length - real.length} of `
        + `the four gains were still inside the noise of the held out set.`);
  set('verdict-port',
    `${f4(pf.test)} in ${P.size} evaluations on a table it had never seen, against `
    + `${f4(rd.test)} for the same number of blind draws.`);
  set('verdict-port-warn',
    `${pct(rd.beats)} of blind draws did at least as well, so on one cold table this is a `
    + `tendency and not a guarantee. It also needs the other tables to exist first.`);
  set('verdict-ens',
    `${ensGains.map((x) => signed(x.gain)).join(', ')} over the single best pipeline, on the `
    + `four tables in order.`);
  set('verdict-ens-warn',
    `Nothing is refitted, so the cost is a few array averages. The risk is the opposite one: `
    + `the curve that chooses the size is monotone by construction, so it always looks like `
    + `more is better.`);
  set('verdict-nested',
    `${f4(N.mean)} with a spread of ${f4(N.sd)} across ${N.outer} outer folds on wine, `
    + `against ${f4(T.wine.best.cv)} from the search's own cross validation.`);
  set('verdict-nested-warn',
    `${n0(N.fits)} fits instead of ${n0(M.configs * (M.folds + 1))}. It is the only number `
    + `on this page that nobody chose anything with, and it costs the search over again `
    + `inside every fold.`);

  set('closing-1',
    `Two things on this page point the same way and it is worth saying them together. The `
    + `search bought ${real.length === 0 ? 'nothing that clears' : 'little that clears'} the `
    + `noise of the sets used to judge it, and the score it reports overstates by up to `
    + `${signed(worstCurse.many)}. Neither of those is an argument against automated `
    + `machine learning; they are an argument about what it is for. On a table of a few `
    + `hundred rows the catalogue is mostly a way of spending compute to move a number that `
    + `cannot be measured to that precision. What survives is the cheap part: a warm start `
    + `worth ${signed(pf.test - rd.test)} over blind draws for ${P.size} evaluations, and an `
    + `average over models that were fitted anyway.`);

  set('closing-2',
    `And there is a third way to arrive at a table already knowing something, which is `
    + `bigger than a short list of pipelines: start from a model somebody else fitted, on `
    + `somebody else's data. That is <a href="../distillation/">the next article</a>, where `
    + `what gets reused is not a configuration but the weights themselves, and where the `
    + `control that decides whether any of it is real turns out to be a network that was `
    + `never trained at all.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/utils/generate_automl_data.py</span>, seed ${M.seed}. The `
    + `catalogue is ${M.scalers.join(', ')} crossed with ${M.reducers.join(', ')} crossed `
    + `with ${M.weights.join(', ')} crossed with ${M.models.length} model settings, `
    + `${n0(M.configs)} pipelines, each scored by ${M.folds} fold stratified cross `
    + `validation on ${pct(1 - M.test_frac)} of the rows and once on the `
    + `${pct(M.test_frac)} held out. Forests are ${M.trees} trees, boosting is `
    + `${M.boost_iters} iterations, and the digits table is cut to ${n0(M.digits_n)} rows: `
    + `all three are budget decisions and all three are in the cache key, so a rerun with `
    + `different ones cannot silently mix with these. Held out spreads are ${M.boot} `
    + `bootstrap resamples. The curse curves are 4,000 draws per budget. Nested cross `
    + `validation is ${N.outer} outer folds, ${n0(N.fits)} fits. No wall clocks: cost is in `
    + `fits.`);
}
