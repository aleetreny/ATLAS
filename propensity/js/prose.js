/* Every sentence with a measured number, composed from the measurement.
 *
 * The scoreboard paragraph is written as a branch on who actually won rather
 * than as a description of who is supposed to win, because on this world the
 * winner is not the one the literature advertises and a sentence typed from
 * memory would have said the wrong thing.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sgn = (v, d = 4) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-problem', 'intro-theorem',
  'balance-intro', 'balance-note',
  'weights-intro', 'board-verdict',
  'dr-intro', 'dr-note',
  'overlap-intro', 'overlap-trim',
  'hidden-intro', 'hidden-result',
  'v-reg', 'v-reg-watch', 'v-ipw', 'v-ipw-watch',
  'v-match', 'v-match-watch', 'v-aipw', 'v-aipw-watch',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

function rows(sel, data, cells) {
  const body = d3.select(sel).select('tbody');
  if (body.empty()) return;
  body.html('');
  data.forEach((d) => {
    const tr = body.append('tr');
    cells(d).forEach((c) => {
      const td = tr.append('td');
      if (c && c.html !== undefined) td.html(c.html);
      else td.text(c);
    });
  });
}

export function initProse(data) {
  const M = data.meta;
  const B = data.balance;
  const S = data.scoreboard;
  const D = data.double_robust;
  const O = data.overlap;
  const T = data.trimming;
  const H = data.hidden;

  const by = Object.fromEntries(S.rows.map((r) => [r.key, r]));
  const ranked = [...S.rows].filter((r) => r.key !== 'naive')
    .sort((a, b) => a.rmse - b.rmse);
  const winner = ranked[0];
  const worstBalanceBefore = Math.max(...B.before.map(Math.abs));
  const worstBalanceAfter = Math.max(...B.after_true.map(Math.abs));
  const wrongHidden = Math.max(Math.abs(B.after_wrong[1]), Math.abs(B.after_wrong[3]));
  const tightest = O.reduce((p, c) => (c.min_tail < p.min_tail ? c : p), O[0]);
  const worstOverlap = O.reduce((p, c) => (Math.abs(c.bias) > Math.abs(p.bias) ? c : p), O[0]);
  const both = T.filter((r) => r.side !== 'upper');
  const upper = T.filter((r) => r.side === 'upper');
  const noTrim = both[0];
  const someTrim = both.find((r) => r.trim === 0.05) ?? both[2];
  const estimandMove = Math.abs(someTrim.truth_kept - noTrim.truth_all);
  const upperEnd = upper.length ? upper[upper.length - 1] : null;
  const upperMove = upperEnd ? Math.abs(upperEnd.truth_kept - noTrim.truth_all) : 0;

  set('intro-problem',
    `The previous article ended with a recipe that works and a problem that stops it. `
    + `Comparing treated with untreated inside groups of people who share the confounder `
    + `removes the bias, and it needs people who share the confounder. With `
    + `${word(M.theta.length)} continuous covariates there are no two rows alike anywhere in `
    + `the data, so the strategy has no subjects left. Cutting each covariate into `
    + `${word(5)} bins gives ${n0(5 ** M.theta.length)} cells for ${n0(M.n)} rows, and almost `
    + `every cell is either empty or all one treatment.`);

  set('intro-theorem',
    `The result that rescues it is one line. If treatment assignment depends only on the `
    + `covariates, then it is enough to compare people who share a single number: the `
    + `<span class="bold">probability of being treated</span>, \\(e(V) = P(X = 1 \\mid V)\\). `
    + `Not the covariates. The one number computed from them. Everything else in this `
    + `article is a consequence of that, including the ways it goes wrong, and the world `
    + `below is written out in full so that the effect being estimated is `
    + `<span class="bold">${f2(M.tau0)}</span> by construction rather than by estimation.`);

  set('balance-intro',
    `A theorem about balance is a claim that can be checked, so it gets checked before `
    + `anything is built on it. Across the whole sample the treated and the untreated are `
    + `different people: the standardised difference reaches `
    + `<span class="bold">${f3(worstBalanceBefore)}</span> on the covariate that matters `
    + `most, where anything above 0.1 is conventionally called imbalance. Inside thin strata `
    + `of the true score, the worst of the four falls to `
    + `<span class="bold">${f3(worstBalanceAfter)}</span>. The score was computed from the `
    + `covariates and it makes the covariates comparable, which is the part that sounds like `
    + `a trick and is not.`);

  set('balance-note',
    `The control is what makes that mean something. The third view uses a `
    + `<span class="bold">deliberately wrong score</span>, built from two of the four `
    + `covariates. It balances the two it looks at, to ${f3(Math.abs(B.after_wrong[0]))} and `
    + `${f3(Math.abs(B.after_wrong[2]))}, and leaves the other two at up to `
    + `<span class="bold">${f3(wrongHidden)}</span>, which is where they started. A `
    + `propensity score is not a device that removes confounding. It is a device that `
    + `removes confounding by the variables you put into it, and a balance table computed on `
    + `those same variables cannot tell the difference.`);

  set('weights-intro',
    `Once the score exists there are four things people do with it, and they are not `
    + `variations on a theme: weight every unit by the inverse of the probability of the `
    + `treatment it received, match each unit to its nearest opposite, cut the score into `
    + `strata and average within them, or put the score into the outcome model. The widget `
    + `runs all of them on one sample of ${n0(data.cloud.X.length)} rows in the browser. The `
    + `table underneath is the same comparison repeated over ${n0(S.reps)} independent `
    + `samples of ${n0(S.n)}, which is the only way the word "better" means anything.`);

  set('board-verdict',
    `Two things in that table are worth stopping on. The first is that the difference in `
    + `means is off by <span class="bold">${sgn(by.naive.bias)}</span> on an effect of `
    + `${f2(M.tau0)}, and every method that uses the score removes essentially all of it. `
    + `The second is the ranking, which is not the one the propensity literature advertises: `
    + `by root mean squared error the winner here is `
    + `<span class="bold">${winner.label}</span> at ${f4(winner.rmse)}, ahead of `
    + `${f4(by.ipw.rmse)} for weighting and ${f4(by.matching.rmse)} for matching. The reason `
    + `is in the setup rather than in the methods: the outcome model in this world is `
    + `linear and the regression is given the right form, so it is being handed the best `
    + `possible case. The weighting estimators pay for their generality in variance, `
    + `${f3(by.ipw.sd)} against ${f3(by.regression.sd)}, and the unstabilised version pays `
    + `${f3(by.ipw_raw.sd)}.`);

  set('dr-intro',
    `Every estimator above needs a model and the model can be wrong. The regression needs `
    + `the outcome model; the weighting needs the propensity model; and there is a third `
    + `option that uses both and needs only one of them to be right. That claim is exactly `
    + `checkable: fit the two by two table of right and wrong models and read off which `
    + `cells survive. "Wrong" here means something specific rather than sloppy, and it is `
    + `written into the generator: both models see the squares of the covariates instead of `
    + `the covariates, which preserves the scale and destroys the linearity.`);

  set('dr-note',
    `Three cells are the advertisement and they hold. With the propensity model right and `
    + `the outcome model wrong, the regression is off by `
    + `<span class="bold">${sgn(D.cells['10'].regression.bias)}</span> and the doubly robust `
    + `estimator by ${sgn(D.cells['10'].aipw.bias)}. With the outcome model right and the `
    + `propensity model wrong, weighting is off by `
    + `<span class="bold">${sgn(D.cells['01'].ipw.bias)}</span> and the doubly robust `
    + `estimator by ${sgn(D.cells['01'].aipw.bias)}. The fourth cell is the one worth `
    + `remembering: with both models wrong all three land on `
    + `<span class="bold">${sgn(D.cells['00'].aipw.bias)}</span>, which is the bias of doing `
    + `nothing at all. Doubly robust means two chances. It does not mean a guarantee, and `
    + `the name has been read as one often enough to be worth a table.`);

  set('overlap-intro',
    `Everything so far assumed that for every kind of person there exist both treated and `
    + `untreated examples. That assumption has a name, positivity, and unlike the previous `
    + `article's unmeasured confounding it leaves fingerprints in the data. The slider makes `
    + `the two groups more separable without changing which covariate matters. At the far `
    + `end the smallest distance to a certain outcome is `
    + `<span class="bold">${tightest.min_tail.toExponential(1)}</span>, the largest single `
    + `weight carries ${pc(worstOverlap.max_share, 2)} of the total, and the bias reaches `
    + `<span class="bold">${sgn(worstOverlap.bias, 3)}</span> with a spread between samples `
    + `of ${f3(worstOverlap.sd)}. What is being estimated has stopped existing in the data, `
    + `and the estimator keeps returning a number.`);

  set('overlap-trim',
    `The standard repair is to throw the extreme scores away, and it works: trimming at `
    + `${pc(someTrim.trim, 0)} takes the error from ${sgn(noTrim.error_vs_all, 3)} to `
    + `${sgn(someTrim.error_vs_kept, 3)} while keeping ${pc(someTrim.kept)} of the rows. `
    + `The warning usually attached to that is that trimming changes what is being `
    + `estimated, because the survivors are a different population and the effect is not `
    + `the same for everybody. Here that did <span class="bold">not</span> happen: the true `
    + `effect among the survivors is ${f4(someTrim.truth_kept)} against `
    + `${f4(noTrim.truth_all)} for everybody, a shift of ${f4(estimandMove)}, which is `
    + `nothing. The reason is worth more than the warning would have been. Trimming both `
    + `tails removes the people most and least likely to be treated, the effect here rises `
    + `with the same covariate that raises that probability, and the two removals cancel. `
    + `${upperEnd
      ? `Cutting only the upper tail breaks the symmetry and the estimand moves as `
        + `advertised: at ${pc(upperEnd.trim, 0)} the true effect among the survivors is `
        + `<span class="bold">${f4(upperEnd.truth_kept)}</span>, a shift of `
        + `${f4(upperMove)}. So the warning is real and it is about which tail, which is `
        + `not what it usually says.`
      : ''}`);

  set('hidden-intro',
    `Every diagnostic in this article has been about balance, and balance is measured on the `
    + `covariates that went into the score. That makes the whole apparatus silent about the `
    + `one thing that actually decides whether any of it is valid.`);

  set('hidden-result',
    `Here is the same machinery with one covariate hidden from the analyst. The score is `
    + `estimated from the ${word(3)} that remain, weighting is applied, and the balance `
    + `table comes out clean: the standardised differences on the visible covariates average `
    + `<span class="bold">${f3(H.smd_seen)}</span>, well inside anything anybody would call `
    + `balanced. The hidden one sits at <span class="bold">${f3(H.smd_hidden)}</span> and `
    + `nobody can see it. The estimate is off by `
    + `<span class="bold">${sgn(H.bias)}</span>, which is ${f2(Math.abs(H.bias) / H.se)} `
    + `times its own standard error and ${pc(Math.abs(H.bias) / M.tau0)} of the effect. `
    + `There is no diagnostic in this article, or in the field, that fires here.`);

  rows('#board-table', S.rows, (d) => [
    { html: `<span class="bold">${d.label}</span>` },
    sgn(d.bias),
    f4(d.sd),
    f4(d.rmse),
  ]);

  set('v-reg', `bias ${sgn(by.regression.bias)}, spread ${f4(by.regression.sd)}, and the best `
    + `error in the table at ${f4(by.regression.rmse)}`);
  set('v-reg-watch',
    `It wins here because it was handed the right functional form. In the two by two table, `
    + `with the outcome model wrong, it is off by ${sgn(D.cells['10'].regression.bias)}.`);
  set('v-ipw', `bias ${sgn(by.ipw.bias)} with a spread of ${f4(by.ipw.sd)}, against `
    + `${f4(by.ipw_raw.sd)} without stabilising`);
  set('v-ipw-watch',
    `At the tightest overlap measured, one unit carries ${pc(worstOverlap.max_share, 2)} of `
    + `the weight and the effective sample size falls to `
    + `${pc(tightest.ess_frac)} of the rows.`);
  set('v-match', `matching ${sgn(by.matching.bias)}, five strata ${sgn(by.strata.bias)}`);
  set('v-match-watch',
    `The strata estimator keeps ${sgn(by.strata.bias)} of bias for the same reason the `
    + `previous article measured: inside a stratum the score still varies.`);
  set('v-aipw', `bias ${sgn(by.aipw.bias)}, and correct in ${word(3)} of the four cells of `
    + `the model table`);
  set('v-aipw-watch',
    `Both models wrong gives ${sgn(D.cells['00'].aipw.bias)}, the same as doing nothing.`);

  set('closing-1',
    `The propensity score is the most useful idea in this branch and the easiest to oversell. `
    + `What it does is real and was checked above: one number carries all the information `
    + `about assignment that the covariates contain, and comparing people who share it is `
    + `enough. What it does not do is tell you whether the covariates were the right ones. `
    + `Every table in this article looks equally healthy in the run where the answer is `
    + `${sgn(H.bias)} away from the truth, and the only difference between that run and a `
    + `correct one is a column that was never collected.`);

  set('closing-2',
    `Which is the same wall the previous article hit, now with better instruments in front `
    + `of it. Both articles have assumed that somebody wrote the confounder down. The `
    + `<a href="../instruments/">next one</a> gives that up: no confounder measured, no `
    + `balance to check, no score to estimate. What it uses instead is a lever from outside `
    + `the system, and the interesting part is not that it works. It is how much of the `
    + `answer is decided by an assumption that no amount of data can test.`);

  set('ref-note',
    `The world is written out in the generator: ${word(M.theta.length)} independent `
    + `covariates, two of which decide both treatment and outcome, one only the outcome and `
    + `one only the treatment; ${n0(M.n)} rows per sample and ${n0(S.reps)} samples per row `
    + `of the scoreboard; seed ${M.seed}. The true effect is `
    + `${f2(M.tau0)} plus a term that varies across people, which is what makes the trimming `
    + `section measurable, and it was confirmed by simulating the intervened world at `
    + `${f4(M.truth)}.`);
}
