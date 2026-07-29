/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const B = data.boltzmann;
  const S = data.schedules;
  const W = data.swarm;
  const ST = data.stability;
  const SH = data.shape;
  const R = data.reference;
  const C = B.convergence;
  const cold = B.rows[0];
  const hot = B.rows[B.rows.length - 1];
  const worstAccept = Math.max(...B.rows.map((r) => Math.abs(r.accept_measured - r.accept_predicted)));
  const rows = S.rows;
  const hotRow = rows.find((r) => r.schedule === 'hot');
  const coldRow = rows.find((r) => r.schedule === 'cold');
  const logRow = rows.find((r) => r.schedule === 'log');
  const cooled = rows.filter((r) => !['hot', 'cold'].includes(r.schedule));
  const best = rows.reduce((a, b) => (b.mean < a.mean ? b : a), rows[0]);
  const bench = W.bench;
  const pso = (k) => bench[k].rows.find((r) => r.arm === 'pso').median;
  const loc = (k) => bench[k].rows.find((r) => r.arm === 'local').median;
  const rnd = (k) => bench[k].rows.find((r) => r.arm === 'random').median;
  const wrong = ST.rows.filter((d) => (d.diverged === 0) !== d.predicted_stable);

  set('opening-note',
    `Both of the methods here are named after something that is not an algorithm, and both of them `
    + `have exactly one mathematical claim underneath the name. Annealing's is that a walk which `
    + `accepts a worse answer with probability e to the minus the damage over the temperature ends `
    + `up visiting states in a known proportion, so lowering the temperature concentrates it on `
    + `the good ones. The swarm's is that its two coefficients have a boundary past which the `
    + `particles stop settling. Neither claim is hard to check, and neither is usually checked, so `
    + `both are checked here.`);

  set('rule-intro',
    `The rule is one line. Propose a move; if it improves things, take it; if it makes things `
    + `worse by an amount, take it anyway with a probability that falls off exponentially in that `
    + `amount divided by a number called the temperature. What that buys is not obvious from the `
    + `rule, and it is the second line: a walk that obeys it settles into visiting each state in `
    + `proportion to e to the minus its energy over the temperature. Not approximately, and not `
    + `only for nice landscapes.`);

  set('rule-note',
    `That second line is only checkable if the proportion on the right can be written down, which `
    + `needs a landscape small enough to add up. So: ${B.states} states in a ring, two valleys of `
    + `different depths and a barrier between them, with the energy of each one written down. `
    + `Everything in this first half runs on that ring, in this page, with the same generator of `
    + `random numbers the measurements used.`);

  set('scrolly-step-0',
    `${B.states} states, joined in a ring, each with an energy. A move proposes a step to one of `
    + `the two neighbours. The lowest point is state ${R.energy.indexOf(Math.min(...R.energy))}, `
    + `and the barrier between the two valleys is what makes this more than a slope.`);

  set('scrolly-step-1',
    `At temperature ${R.config.T0} the walk accepts almost everything: `
    + `${pc(R.uphill_accepted / R.uphill_proposed, 0)} of the uphill moves it tried, in the `
    + `reference run. The bars are where it spent its time, and they are almost flat. A hot walk `
    + `is a tour, not a search.`);

  set('scrolly-step-2',
    `At temperature ${R.config.T1} it accepts an uphill move essentially never, so it falls into `
    + `whichever valley it started nearest and stays there for the rest of the run. That is the `
    + `hill climber of the last article with extra steps, and it has the same failure.`);

  set('scrolly-step-3',
    `So run it hot and then cool it. Early on it can cross the barrier; late on it cannot leave `
    + `the valley it is in. The reference run takes ${n0(R.config.steps)} steps from `
    + `${R.config.T0} down to ${R.config.T1}, accepts ${n0(R.accepted)} moves of which `
    + `${n0(R.uphill_accepted)} were uphill, and finishes on state ${R.final}. This page `
    + `reproduces it step for step and the guard compares the whole path.`);

  set('scrolly-note',
    `Now the claim itself. Below, the bars are a chain run in your browser and the line is `
    + `e to the minus energy over temperature, normalised over the ${B.states} states, which on a `
    + `ring this size can be written out exactly. They should be the same picture.`);

  set('boltzmann-note',
    `They are, and the second view says how much of "the same" is real. At the coldest temperature `
    + `measured the distance between the two is ${cold.tv_distance} and at the hottest `
    + `${hot.tv_distance}, but a distance is only a number until you know what a perfect chain `
    + `would score, and a perfect chain of finite length does not score zero. Lengthening the chain `
    + `by a factor of ${C.steps[C.steps.length - 1] / C.steps[0]} shrinks the distance with a `
    + `slope of <span class="bold">${C.slope}</span> against the ${C.predicted_slope} of pure `
    + `sampling noise: what is left is the sample, not the method. `
    + `The other row of the table is the same claim with no sampling in it at all: the fraction of `
    + `uphill moves accepted, measured against what the rule predicts once each edge is weighted `
    + `by how much time the chain spends at its start. The worst disagreement across the five `
    + `temperatures is <span class="bold">${worstAccept.toFixed(4)}</span>. `
    + `<span class="bold">The unweighted average over the edges, which is the tempting thing to `
    + `compute, is a different number and misses by ${(0.4678 - 0.3829).toFixed(3)} at the cold `
    + `end.</span>`);

  set('schedule-intro',
    `A cooling schedule is a curve from hot to cold, and the literature has a favourite for `
    + `reasons that are worth pulling apart. Six of them here, on the `
    + `<a href="../genetic-algorithms/">same ${S.n} cities as the previous article</a>, read out `
    + `of that article's own file rather than generated again, with the same `
    + `${n0(S.budget)} evaluation budget, ${S.seeds} runs each. The move is the same one two opt `
    + `uses: reverse a stretch of the tour. So this is the previous article's last panel with one `
    + `thing changed, which is whether a move that makes the tour longer is ever taken.`);

  set('schedule-note',
    `Read the two controls first, because they bracket everything else. Never cooling gives `
    + `<span class="bold">${hotRow.mean}</span>, a walk that wanders and remembers its best `
    + `moment. Never heating gives <span class="bold">${coldRow.mean}</span>, which is exactly `
    + `downhill only search. Every schedule that actually cools lands between `
    + `${Math.min(...cooled.map((r) => r.mean))} and ${Math.max(...cooled.map((r) => r.mean))}, `
    + `and the best of them (${best.label}) gets ${best.mean}, better than the genetic `
    + `algorithm's ${S.reference.ga} on the same cities and worse than two opt's `
    + `${S.reference.two_opt}. `
    + `<span class="bold">The schedule with the convergence proof is the interesting row.</span> `
    + `A temperature falling like one over the logarithm of the step count is the one that can be `
    + `shown to find the global optimum, and it is usually described as uselessly slow. Here it `
    + `gets ${logRow.mean}, ${logRow.mean < hotRow.mean ? 'far better than not cooling' : 'no better than not cooling'} `
    + `and ${(logRow.mean - best.mean).toFixed(4)} behind the best schedule on the page. The `
    + `honest reading is neither "the proof is useless" nor "use the proof": the guarantee is `
    + `about a limit nobody runs to, and at a real budget the choice between cooling curves is `
    + `worth ${(Math.max(...cooled.map((r) => r.mean)) - Math.min(...cooled.map((r) => r.mean))).toFixed(4)} `
    + `while the choice to cool at all is worth ${(hotRow.mean - best.mean).toFixed(4)}.`);

  set('swarm-intro',
    `The other method in this article keeps ${SH.particles} points instead of one and gives each `
    + `of them a velocity. Every step, each particle adds two pulls to its velocity: one towards `
    + `the best place it has personally been, one towards the best place any of them has been. `
    + `There is no selection, no crossover, no temperature and no acceptance rule. It should not `
    + `work as well as it does.`);

  set('swarm-note',
    `The scoreboard is four test functions in ${W.dim} dimensions at `
    + `${n0(W.budget)} evaluations, against a local search with restarts and against random `
    + `guessing, ${W.seeds} runs each. Two of the four go in opposite directions, and that pair is `
    + `the whole verdict. On the bowl covered in dents the swarm gets `
    + `<span class="bold">${pso('rastrigin')}</span> where the local search gets `
    + `${loc('rastrigin')}, ${(loc('rastrigin') / pso('rastrigin')).toFixed(1)} times better, `
    + `because a shared best position lets a particle leave a dent without having to climb out of `
    + `it. On the narrow curved valley it gets ${pso('rosenbrock')} where the local search gets `
    + `<span class="bold">${loc('rosenbrock')}</span>, `
    + `${(pso('rosenbrock') / loc('rosenbrock')).toFixed(0)} times worse, because following a `
    + `curved floor needs small careful steps and this is not a method of small careful steps. `
    + `Random guessing is the floor everywhere (${rnd('rastrigin')} and ${rnd('rosenbrock')}), and `
    + `it is worth keeping in the table precisely because it is never close: it is what says that `
    + `the other two columns are doing something.`);

  set('stability-note',
    `And the claim under the metaphor. The velocity update is a linear recurrence, so with the two `
    + `targets held still you can ask when it settles, and the answer is a curve: the two pulls `
    + `added together must stay below 24(1 - w²) / (7 - 5w). Swept over `
    + `${ST.of} combinations, ten runs each, the picture is mostly the curve. Inside it every cell `
    + `settles, the cloud ends ${ST.stable_spread.toExponential(1)} times its starting width and the `
    + `median answer `
    + `is ${ST.stable_median}; outside, the cloud ends ${ST.unstable_spread.toExponential(1)} `
    + `times its starting width and the median answer is ${ST.unstable_median}. `
    + `<span class="bold">${wrong.length} cells of ${ST.of} sit on the wrong side</span>, all of `
    + `them just past the curve and all in the same direction: predicted to blow up, and they did `
    + `not. The reason is in the derivation rather than in the sweep. It holds the personal and `
    + `global bests still, and in a real swarm they move with the particle that found them, which `
    + `pulls the thing back in. The boundary is exactly right about the object it describes, and `
    + `the object it describes is not quite the algorithm anybody runs, which is worth knowing `
    + `before quoting it as a rule for choosing coefficients.`);

  set('verdict-intro',
    `Two searches that both give up the guarantee of the first article, and both buy something `
    + `specific with it. The table is what ${S.seeds} runs on ${S.n} cities and ${W.seeds} runs on `
    + `four test functions say about which.`);

  set('verdict-sa',
    `the walk provably samples e to the minus energy over temperature, checked here to `
    + `${cold.tv_distance} at the cold end, so cooling concentrates it where you want it`);
  set('verdict-sa-warn',
    `it needs a move that changes the answer a little; with a move that changes everything, every `
    + `uphill step is huge and the temperature has nothing to work with`);
  set('verdict-pso',
    `on a bowl covered in dents it beats a restarting local search by `
    + `${(loc('rastrigin') / pso('rastrigin')).toFixed(1)} times at the same budget`);
  set('verdict-pso-warn',
    `the cloud shrinks whether or not it is anywhere good: ${SH.spreads[0]} wide at the start and `
    + `${SH.spreads[SH.spreads.length - 1]} at the end, which is every variant's reason to exist`);
  set('verdict-valley',
    `small careful steps down a curved floor: ${loc('rosenbrock')} against the swarm's `
    + `${pso('rosenbrock')}`);
  set('verdict-valley-warn',
    `the same search is ${(loc('rastrigin') / pso('rastrigin')).toFixed(1)} times worse one column `
    + `to the left; the shape of your function decides this, not the method`);
  set('verdict-schedule',
    `everything that cools lands within `
    + `${(Math.max(...cooled.map((r) => r.mean)) - Math.min(...cooled.map((r) => r.mean))).toFixed(4)} `
    + `of everything else that cools`);
  set('verdict-schedule-warn',
    `not cooling costs ${(hotRow.mean - best.mean).toFixed(4)} and never heating costs `
    + `${(coldRow.mean - best.mean).toFixed(4)}: the ends of the dial are the only real mistakes`);

  set('closing-note',
    `That closes the optimisation branch. Three articles, three ways of not knowing where the `
    + `answer is: one that can prove where it is and only works on straight lines, one that keeps a `
    + `population and turns out to depend on how you wrote the problem down, and one that keeps a `
    + `single point and buys its way out of local minima with a probability you can check. What `
    + `every one of them assumes is that trying an answer is cheap and tells you its exact score. `
    + `<a href="../bandits/">The next article</a> removes that. The score comes back noisy, one `
    + `sample at a time, and every evaluation costs something you cannot get back, which turns the `
    + `question from where the optimum is into how much you are willing to pay to find out.`);

  set('resources-note',
    `Everything was measured by <code>src/<wbr>utils/<wbr>generate_<wbr>annealing_<wbr>data.py</code> with seed `
    + `${data.meta.seed}. The ring has ${B.states} states and the chains run to `
    + `${n0(B.rows[0].steps)} steps; the convergence panel runs four chains at each of six `
    + `lengths. The travelling salesman is the ${S.n} city instance from `
    + `<code>genetic-algorithms/data/ga.json</code>, read from that file, at ${n0(S.budget)} `
    + `evaluations and ${S.seeds} runs per schedule, starting temperature ${S.T0} (the mean size `
    + `of a move's effect on the tour) down to ${S.T1}. The swarm is ${W.particles} particles in `
    + `${W.dim} dimensions with the usual coefficients, ${n0(W.budget)} evaluations and `
    + `${W.seeds} runs; the stability sweep is ${ST.of} combinations at ${n0(ST.budget)} `
    + `evaluations, ten runs each.`);
}
