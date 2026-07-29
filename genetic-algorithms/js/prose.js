/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const P = data.probe;
  const A = data.arms;
  const pop = data.population;
  const take = data.takeover;
  const T = data.tsp;
  const R = data.reference;
  const arm = (land, key) => A[land].rows.find((r) => r.arm === key);
  const tightOne = arm('trap_tight', 'one_point');
  const looseOne = arm('trap_loose', 'one_point');
  const tightMut = arm('trap_tight', 'mutation_only');
  const looseMut = arm('trap_loose', 'mutation_only');
  const tightUni = arm('trap_tight', 'uniform');
  const climbOne = arm('onemax', 'hill_climber');
  const gaOne = arm('onemax', 'one_point');
  const climbRoyal = arm('royal', 'hill_climber');
  const rnd = arm('onemax', 'random');
  const t1 = take.rows[0];
  const t2 = take.rows[1];
  const tLast = take.rows[take.rows.length - 1];
  const firstFull = pop.rows.find((d) => d.success === M.seeds);
  const ratio = Math.round(gaOne.median_evals / climbOne.median_evals);

  set('opening-note',
    `A genetic algorithm is four rules on top of a population of guesses: score them, let the `
    + `better ones have more children, build each child by mixing two parents, and flip a few bits `
    + `on the way out. Three of those four are shared with every other stochastic search ever `
    + `written. The one that is not is the mixing, and it is the one nobody measures, because `
    + `measuring it means running the same algorithm without it and admitting what comes back. `
    + `That is the experiment on this page, on landscapes whose answers are known in advance so `
    + `there is nothing to argue about.`);

  set('landscape-intro',
    `Four landscapes over ${M.bits} bits, each with its optimum written down rather than searched `
    + `for. <span class="bold">Count the ones</span>: fitness is how many bits are set, so every `
    + `single flip either helps or hurts and the way up is never in doubt. `
    + `<span class="bold">Blocks, all or nothing</span>: the bits are read in groups of `
    + `${M.block}, and a group pays ${M.block} only when every bit in it is set, otherwise `
    + `nothing. And the two that matter, built from a <span class="bold">trap</span>.`);

  set('landscape-note',
    `Inside a block of ${M.block} bits, the optimum is all ones and is worth ${M.block}, but every `
    + `block that is not complete is paid for its <span class="bold">zeros</span>: all zeros is `
    + `worth ${M.block - 1}, one bit set is worth ${M.block - 2}, and so on. So the local gradient `
    + `points exactly away from the answer. String ${M.blocks} of those together and the all zeros `
    + `chromosome scores <span class="bold">${P.deceptive_attractor}</span> out of a possible `
    + `${P.optima.trap_tight}, and every single bit flip away from it looks like a mistake. `
    + `The fourth landscape is that same function with the ${M.bits} bit positions shuffled once, `
    + `so a block's ${M.block} bits are scattered across the chromosome instead of sitting next to `
    + `each other. Not a harder problem: the same problem, with the parts moved. Remember that.`);

  set('scrolly-intro',
    `Here is one generation of it, drawn on the real thing. This is not an illustration: the `
    + `population below is the first generation of a run that the file also contains, produced by `
    + `the same code with the same random number generator, so the cut points and the flipped bits `
    + `on screen are the ones that happened. Each row is a chromosome, dark cells are ones, and `
    + `the grey lines mark the blocks of ${M.block} the trap is built from.`);

  set('scrolly-step-0',
    `${R.config.pop} chromosomes of ${M.bits} random bits, and the bar on the right is what each `
    + `one scores. The best of them is ${R.history[0].best} out of ${P.optima.trap_tight}, and the `
    + `average is ${R.history[0].mean.toFixed(2)}, which is roughly what you get by guessing.`);

  set('scrolly-step-1',
    `To fill each of the ${R.config.pop} slots in the next generation, draw ${R.config.tour} `
    + `chromosomes at random and keep the best of them. That is all selection is. Good rows appear `
    + `several times, bad rows disappear, and the number on the left says which parent each row `
    + `came from. Nothing has been invented yet: this step only deletes.`);

  set('scrolly-step-2',
    `Now the only step that is particular to this method. Take the rows in pairs, choose one `
    + `position, and swap everything after it. If the parts of a good answer sit next to each `
    + `other, a cut has a decent chance of keeping each part intact and putting two of them in the `
    + `same child. If they do not, the same cut chops through them.`);

  set('scrolly-step-3',
    `Finally, flip each bit with probability one in ${M.bits}, which is the usual choice and the `
    + `one measured here. Outlined cells changed. This is the only step that can invent a bit `
    + `value the population had already lost, which is why a run with it switched off stops dead `
    + `once every row agrees.`);

  set('scrolly-note',
    `Run that for ${R.config.gens} generations and you get the widget below, which is the same `
    + `code again, executing while you read. It opens on the settings the file recorded, and at `
    + `those settings it <span class="bold">fails</span>: it finishes at ${R.final_best} out of `
    + `${P.optima.trap_tight}, with the population averaging `
    + `${R.history[R.history.length - 1].mean.toFixed(2)} against the `
    + `${P.deceptive_attractor} the trap pays for getting every block exactly wrong. The `
    + `population has agreed on the answer that every single bit flip recommends. Every control `
    + `here can be moved, and one of them fixes it.`);

  set('ga-note',
    `The population slider is the one that does it. With ${R.config.pop} guesses the run above `
    + `converges on the trap; the table further down measures that over ${M.seeds} seeds rather `
    + `than one, and the crossover arm goes from ${pop.rows.find((d) => d.pop === 100).success} `
    + `successes out of ${M.seeds} at a population of 100 to `
    + `${pop.rows.find((d) => d.pop === M.pop).success} at ${M.pop}, on the same budget of `
    + `${n0(M.budget)} evaluations. Switching the landscape to the shuffled version, which changes `
    + `nothing except where the bits live, takes it back down again.`);

  set('arms-intro',
    `One run is a run. Here are five searchers on each of the four landscapes, ${M.seeds} seeds `
    + `each, every one stopped at the same ${n0(M.budget)} evaluations, all with a population of `
    + `${M.pop} where there is a population at all. Two of them are not genetic algorithms: a hill `
    + `climber that flips single bits and restarts from somewhere new when it gets stuck, and `
    + `${n0(M.budget)} random guesses, which is the floor.`);

  set('arms-note',
    `Start with the landscape a genetic algorithm is usually demonstrated on. Everything solves `
    + `counting the ones, so the interesting column is the cost: the hill climber gets there in `
    + `<span class="bold">${climbOne.median_evals} evaluations</span> and the genetic algorithm in `
    + `${n0(gaOne.median_evals)}, about ${ratio} times as many, while random guessing never `
    + `arrives at all and stalls at ${rnd.mean_best} of ${rnd.best_possible}. On a smooth `
    + `landscape a population is an expensive way to walk uphill. Change to the all or nothing `
    + `blocks and the hill climber collapses to `
    + `<span class="bold">${climbRoyal.success} of ${climbRoyal.of}</span>: a single flip never `
    + `improves a block, so there is no hill to climb and everything it reaches `
    + `(${climbRoyal.mean_best} of ${climbRoyal.best_possible}) comes from its restarts.`);

  set('linkage-note',
    `Then the pair this page was built for. On the traps with the blocks side by side, one point `
    + `crossover finds the optimum <span class="bold">${tightOne.success} times out of `
    + `${tightOne.of}</span>; uniform crossover, which decides each bit independently and so cuts `
    + `blocks apart, finds it ${tightUni.success}; mutation alone finds it ${tightMut.success}. `
    + `Now shuffle the bit positions. Same function, same optimum, same budget, same population, `
    + `and one point crossover drops to <span class="bold">${looseOne.success} of `
    + `${looseOne.of}</span>. The control is mutation alone, which cannot see where a bit lives: `
    + `it scores ${tightMut.success} and ${looseMut.success}, the same to within a run and `
    + `${tightMut.per_seed.join() === looseMut.per_seed.join() ? 'on the same seeds' : 'not on the same seeds'}, `
    + `so the problem did not get harder, only the representation changed. `
    + `<span class="bold">That is the building block idea stated as something falsifiable, and it `
    + `survived.</span> It also comes with its bill: crossover is worth a great deal exactly when `
    + `the parts of an answer are adjacent in your encoding, and worth nothing when they are not, `
    + `and which of those you have is a fact about how you chose to write the problem down.`);

  set('dials-intro',
    `Two settings decide whether any of this happens, and both are usually copied from whatever `
    + `paper was open at the time. Neither of them is about biology.`);

  set('dials-note',
    `Read the population panel as a budget, not as a size: everything in it got the same `
    + `${n0(M.budget)} evaluations, so ${pop.rows[0].pop} guesses buy `
    + `${Math.floor(M.budget / pop.rows[0].pop)} generations and ${pop.rows[pop.rows.length - 1].pop} `
    + `buy ${Math.floor(M.budget / pop.rows[pop.rows.length - 1].pop)}. Small populations agree `
    + `with themselves before they have found anything worth agreeing on: `
    + `${pop.rows[0].success} of ${M.seeds} at ${pop.rows[0].pop}, against `
    + `${firstFull ? `${firstFull.success} of ${M.seeds} at ${firstFull.pop}` : 'no clean sweep at any size here'}. `
    + `The other panel is selection with everything else switched off, which is the only way to `
    + `see it. With contests of ${t1.tournament} there is no selection at all and the best `
    + `individual was lost in <span class="bold">${t1.best_lost} of ${t1.of}</span> runs; from `
    + `${t2.tournament} up, the population converges in ${t2.median} generations down to `
    + `${tLast.median}, against the textbook log(population) over log(contest), which predicts `
    + `${t2.predicted} down to ${tLast.predicted}. The right shape, consistently low. And even at `
    + `${t2.tournament} the best individual is thrown away in ${t2.best_lost} runs of `
    + `${t2.of}, which is why every implementation quietly copies it forward by hand.`);

  set('tsp-intro',
    `All of the above is on landscapes built to have the structure this method is supposed to `
    + `exploit. The honest test is a problem somebody actually has, with a searcher that is not a `
    + `metaphor. ${T.n} cities, ${n0(T.budget)} tour lengths evaluated by each run, ${T.seeds} `
    + `runs each: a genetic algorithm with the standard crossover for permutations against two `
    + `opt, which repeatedly reverses a stretch of the tour whenever that makes it shorter.`);

  set('tsp-note',
    `Two opt averages ${T.two_opt.mean} and the genetic algorithm ${T.ga.mean}, `
    + `${(Math.abs(T.gap) * 100).toFixed(2)}% ${T.gap > 0 ? 'worse' : 'better'}, at the same `
    + `number of evaluations. But the gap is not the story, the spread is: two opt varies by `
    + `${T.two_opt.spread} between runs and the genetic algorithm by ${T.ga.spread}, `
    + `${(T.ga.spread / T.two_opt.spread).toFixed(1)} times as much. Its best run of ${T.seeds} `
    + `(${T.ga.best}) is within ${((T.ga.best / T.two_opt.best - 1) * 100).toFixed(2)}% of two `
    + `opt's best (${T.two_opt.best}); its typical run is not. That is the trade in one line: a `
    + `search that knows a good local move for your problem beats a search that only knows how to `
    + `mix two answers, and it beats it reliably. What the genetic algorithm has instead is that `
    + `it needed to be told nothing about tours at all beyond how to score one.`);

  set('verdict-intro',
    `The method is easy to write, hard to argue with, and easy to be fooled by, because it always `
    + `returns something and the something is always better than where it started. The table is `
    + `what ${M.seeds} seeds on four landscapes and one real problem say about when to reach for `
    + `it.`);

  set('verdict-smooth',
    `single flips point the way, so climbing is ${ratio} times cheaper than breeding: `
    + `${climbOne.median_evals} evaluations against ${n0(gaOne.median_evals)}`);
  set('verdict-smooth-warn',
    `check that it is smooth before believing this: on the all or nothing blocks the same climber `
    + `finds the optimum ${climbRoyal.success} times in ${climbRoyal.of}`);
  set('verdict-blocks',
    `crossover keeps a good stretch intact and puts two of them in one child: `
    + `${tightOne.success} of ${tightOne.of} where mutation alone gets ${tightMut.success}`);
  set('verdict-blocks-warn',
    `it is the adjacency that does it, not the crossover: shuffle the positions and the same arm `
    + `goes to ${looseOne.success} of ${looseOne.of}`);
  set('verdict-local',
    `two opt beats the population on the travelling salesman by `
    + `${(Math.abs(T.gap) * 100).toFixed(2)}% and with `
    + `${(T.ga.spread / T.two_opt.spread).toFixed(1)} times less spread between runs`);
  set('verdict-local-warn',
    `a local move is a piece of knowledge about your problem; if you do not have one, this row is `
    + `not available to you`);
  set('verdict-blind',
    `it needs only a score, and it recovers from a deceptive gradient that stops a climber dead: `
    + `${tightOne.success} of ${tightOne.of} against ${arm('trap_tight', 'hill_climber').success}`);
  set('verdict-blind-warn',
    `the population has to be big enough to hold the parts: ${pop.rows[0].success} of ${M.seeds} `
    + `at ${pop.rows[0].pop} guesses and ${pop.rows.find((d) => d.pop === M.pop).success} at ${M.pop}, `
    + `on the same budget`);

  set('closing-note',
    `What survives the measuring is smaller than the metaphor and more useful. Crossover is worth `
    + `something specific: it assembles parts of an answer that already exist in different `
    + `individuals, and it only manages that when your encoding puts each part in one piece. `
    + `Everything else about the method, the population, the selection, the mutation, is a way of `
    + `keeping enough different guesses alive long enough for that to happen, which is why the `
    + `population slider mattered more than the crossover buttons. `
    + `<a href="../annealing-swarm/">The next article</a> keeps the randomness and throws away the `
    + `population: one point wandering a landscape, allowed to go downhill on purpose, with the `
    + `probability of that written down and measured against what actually gets accepted.`);

  set('resources-note',
    `Everything was measured by <code>src/utils/generate_ga_data.py</code> with seed ${M.seed}, `
    + `${M.seeds} seeds per cell, a budget of ${n0(M.budget)} fitness evaluations, chromosomes of `
    + `${M.bits} bits in blocks of ${M.block}, populations of ${M.pop}, contests of `
    + `${R.config.tour} and a mutation rate of one in ${M.bits}. The travelling salesman is `
    + `${T.n} cities drawn uniformly in the unit square, ${n0(T.budget)} evaluations and `
    + `${T.seeds} runs per searcher. The reference run in the page is `
    + `${R.config.pop} chromosomes for ${R.config.gens} generations from seed ${R.config.seed}, `
    + `reproduced here bit for bit.`);
}
