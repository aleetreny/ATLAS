/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 2) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const R = data.regret;
  const S = data.slope;
  const C = data.coverage;
  const K = data.calibration;
  const B = data.constant;
  const D = data.drift;
  const N = data.snapshots;
  const bestC = B.rows.reduce((a, b) => (b.regret < a.regret ? b : a), B.rows[0]);
  const proved = B.rows.find((d) => d.c === B.proved_c);
  const plain = D.rows.filter((r) => !r.policy.includes('window'));
  const worstDrift = plain.reduce((a, b) => (b.after > a.after ? b : a), plain[0]);
  const otherDrift = plain.find((r) => r !== worstDrift);
  const winTs = D.rows.find((r) => r.policy === 'thompson_window');
  const ts = S.rows.find((d) => d.policy === 'thompson');
  const ucb = S.rows.find((d) => d.policy === 'ucb');

  set('opening-note',
    `The five machines are the ones <a href="../bandits/">the previous article</a> used, read out `
    + `of its file, so every regret here can be put next to every regret there. Two policies, and `
    + `the same idea underneath both: an arm that has been pulled twice is not worth the same as `
    + `an arm that has been pulled two hundred times, even when their averages agree. One turns `
    + `that into a number added to the estimate, the other into a distribution over what the `
    + `estimate could have been.`);

  set('idea-intro',
    `The first keeps an interval. Take the average so far and add a term that grows with how long `
    + `the run has been going and shrinks with how many pulls this arm has had, then pull whichever `
    + `arm has the highest total. An arm gets attention either because it looks good or because `
    + `nobody has checked it lately, and the same number decides both. The second keeps a `
    + `distribution: after ${'s'} wins in ${'n'} pulls, what the machine could be is a Beta, and `
    + `each round you draw one number from each arm's distribution and pull the winner of the `
    + `draw.`);

  set('idea-note',
    `Neither of them has a rate to tune or a schedule to set, and neither needs to be told the gap `
    + `between the machines. Both are single lines of code. Below, the two internal states are `
    + `drawn side by side at four moments of the same run, recomputed in this page from the counts `
    + `the generator recorded. That run is <span class="bold">chosen rather than drawn</span>: of `
    + `${N.candidates} seeds, it is the one whose share of pulls on the best machine `
    + `(${pc(N.share_best, 1)}) is closest to the median of all of them, because a single run can `
    + `be lucky and the first one tried was.`);

  set('scrolly-step-0',
    `After ${N.marks[0]} pulls the bound is enormous on everything: nothing has been seen often `
    + `enough to be ruled out. The posteriors say the same thing in another language, spreading `
    + `across most of the interval. At this point both policies are effectively pulling at random, `
    + `which is correct.`);

  set('scrolly-step-1',
    `By ${N.marks[1]} pulls the two worst machines have narrower distributions sitting low, and `
    + `their bounds have come down with them. Nothing was decided: the interval shrank because `
    + `they were pulled, and they were pulled because the interval was wide.`);

  set('scrolly-step-2',
    `At ${N.marks[2]} the best machine has most of the pulls and the tightest posterior. The one `
    + `just below it still has a distribution overlapping the leader, which is exactly the `
    + `condition under which Thompson keeps sampling it: overlap is the probability of being `
    + `pulled.`);

  set('scrolly-step-3',
    `At ${N.marks[3]} the bad machines have not been forgotten. Their bounds are still above their `
    + `estimates by a wide margin, because the logarithm on top keeps growing while their pull `
    + `counts do not, so every so often one of them becomes the highest index again and gets `
    + `checked. That is the mechanism, and it is why neither policy ever locks on.`);

  set('scrolly-note',
    `Both of those are the same picture of the same idea, so the question is whether they behave `
    + `the same. Over ${n0(M.runs)} runs of ${n0(M.horizon)} pulls, they do not.`);

  set('regret-note',
    `Thompson sampling regrets <span class="bold">${R.thompson.final}</span> and the confidence `
    + `bound ${R.ucb.final}, a factor of ${(R.ucb.final / R.thompson.final).toFixed(1)}, with the `
    + `blind rate from the last article at ${R.eps.final} in between. The dashed line is the floor `
    + `Lai and Robbins put under this problem: ${M.bound_constant} times the logarithm of the pull `
    + `count, computed from these five probabilities rather than quoted. Fitted over the second `
    + `half of the run, Thompson's regret grows at ${ts.slope} per unit of logarithm, `
    + `${ts.ratio} times that constant, and the confidence bound's at ${ucb.slope}, `
    + `${ucb.ratio} times. <span class="bold">A slope below the floor is not a contradiction and `
    + `it is worth being careful about it:</span> the bound says what the regret must eventually `
    + `grow like, and ${n0(M.horizon)} pulls is not eventually. What the comparison does settle is `
    + `the other one: ${ucb.ratio} against ${ts.ratio} is a real difference between two policies `
    + `measured the same way.`);

  set('honest-intro',
    `So why does the one with the theorem lose? The answer is measurable and it is not about the `
    + `theorem being wrong. It is about what the theorem has to promise.`);

  set('honest-note',
    `Three views, one argument. The bound is <span class="bold">never wrong</span>: over `
    + `${n0(C.runs)} runs of ${C.horizon} pulls the true mean of an arm was above that arm's own `
    + `bound in ${pc(C.share_outside, 4)} of the checks, against the ${pc(C.promised, 2)} the `
    + `inequality allows on average. The posterior is nearly honest instead: the truth sits inside `
    + `the middle ${K.level * 100}% of it ${pc(K.inside)} of the time, `
    + `${((K.level - K.inside) * 100).toFixed(2)} points optimistic, which is what happens when the `
    + `data is not an independent sample because the policy chose it. And the price of never being `
    + `wrong is the third view: the constant the proof needs is ${B.proved_c}, the constant that `
    + `wins here is <span class="bold">${bestC.c}</span>, and using the proved one costs `
    + `${(proved.regret / bestC.regret).toFixed(1)} times the regret over ${n0(B.horizon)} pulls. `
    + `An interval built to be right on every problem is too wide for the one in front of you, and `
    + `too wide means pulls spent on arms that did not need them.`);

  set('drift-intro',
    `Everything above assumes the machines stand still. One experiment breaks that: at the halfway `
    + `mark, the best and the worst swap places, and nothing tells the policies.`);

  set('drift-note',
    `<span class="bold">The policy that won the stationary problem is the one that suffers most.</span> `
    + `In the second half Thompson regrets ${worstDrift.after === D.rows.find((r) => r.policy === 'thompson').after ? worstDrift.after : D.rows.find((r) => r.policy === 'thompson').after} `
    + `and the confidence bound ${otherDrift.after === D.rows.find((r) => r.policy === 'ucb').after ? otherDrift.after : D.rows.find((r) => r.policy === 'ucb').after}, `
    + `the reverse of their order before the swap. The cause is the thing that made it win: a `
    + `posterior that has become confident is slow to be argued out of it, while a bound with a `
    + `growing logarithm on top keeps going back to check. Keeping only the last ${D.window} pulls `
    + `fixes both, and the fix has a price that is visible in the first column: `
    + `${winTs.before} against ${D.rows.find((r) => r.policy === 'thompson').before} before the `
    + `swap, where nothing was changing and the forgetting bought nothing. Which half of that `
    + `trade you are in is not something any of these algorithms can tell you.`);

  set('verdict-intro',
    `Two lines of code, no rate to tune, no schedule, no need to know the gap. The table is what `
    + `${n0(M.runs)} runs say about which of them, and about the one place where both of them are `
    + `the wrong answer.`);

  set('verdict-ts',
    `${R.thompson.final} against ${R.ucb.final} over ${n0(M.horizon)} pulls, and it spends `
    + `${pc(R.thompson.share_best, 1)} of them on the best machine`);
  set('verdict-ts-warn',
    `it commits: after the machines swapped it regretted `
    + `${D.rows.find((r) => r.policy === 'thompson').after} in the second half against `
    + `${D.rows.find((r) => r.policy === 'ucb').after} for the bound`);
  set('verdict-ucb',
    `every choice is a number you can print: this arm has the highest estimate plus bonus, and the `
    + `bonus is why`);
  set('verdict-ucb-warn',
    `the proved constant is ${(B.proved_c / bestC.c).toFixed(0)} times the one that wins here, and `
    + `costs ${(proved.regret / bestC.regret).toFixed(1)} times the regret`);
  set('verdict-drift',
    `only the last ${D.window} pulls counted: it cuts Thompson's second half from `
    + `${D.rows.find((r) => r.policy === 'thompson').after} to ${winTs.after}`);
  set('verdict-drift-warn',
    `and costs ${(winTs.before / D.rows.find((r) => r.policy === 'thompson').before).toFixed(1)} `
    + `times more in a first half where nothing changed`);
  set('verdict-const',
    `the sweep found ${bestC.c} where the proof needs ${B.proved_c}, for `
    + `${(proved.regret / bestC.regret).toFixed(1)} times less regret`);
  set('verdict-const-warn',
    `a constant that works everywhere is too big for here, and the measurement of that is the `
    + `${pc(C.share_outside, 4)} of checks where the bound was actually wrong`);

  set('closing-note',
    `The bandit is the whole of the exploration problem and none of the rest of it. There is no `
    + `state: the machines do not care what you pulled last, and pulling one does not change what `
    + `the others will pay. Put that back in, so that an action moves you somewhere and where you `
    + `are decides what is available next, and almost everything on this page has to be rebuilt. `
    + `<a href="../q-learning/">The next article</a> starts that: the same problem of acting under `
    + `uncertainty, with a world that remembers.`);

  set('resources-note',
    `Everything was measured by <code>src/<wbr>utils/<wbr>generate_<wbr>ucb_<wbr>data.py</code> with seed ${M.seed}: `
    + `${n0(M.runs)} runs of ${n0(M.horizon)} pulls for the regret comparison, ${n0(C.runs)} runs `
    + `of ${C.horizon} for the coverage count, ${K.runs} runs for the calibration check, and 600 `
    + `runs of ${n0(D.horizon)} for the swap. The machines pay one or nothing with probabilities `
    + `${M.arms.join(', ')}, read from <code>${M.shared_with}</code>. The confidence bound uses `
    + `the square root of ${B.proved_c} times the logarithm of the pull count over the pulls given `
    + `to the arm; the posterior is a Beta starting from one and one.`);
}
