/* Every number in the running text, composed from the file at load.
 *
 * Same rule as the rest of the site: nothing here is typed, and every
 * comparison has both branches written, so rerunning the generator moves the
 * sentences instead of leaving them lying.
 */
const n0 = (v) => Number(v).toLocaleString('en-US');
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const pts = (v) => `${(v * 100).toFixed(1)} points`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const S = data.solve;
  const A = data.accuracy;
  const C = data.constant;
  const NZ = data.noise;
  const V = data.versus;
  const T = data.tree;
  const MO = data.model;

  const plain = A.rows.filter((d) => !d.smart);
  const lo = plain[0];
  const hi = plain[plain.length - 1];
  const floor = NZ.rows.find((d) => d.sims === C.sims && !d.smart);
  const floorSmart = NZ.rows.find((d) => d.sims === C.sims && d.smart);
  const book = C.rows.find((d) => d.c === C.textbook_c);
  const bestC = C.rows.reduce((a, b) => (b.share > a.share ? b : a), C.rows[0]);
  const worstC = C.rows.reduce((a, b) => (b.share < a.share ? b : a), C.rows[0]);
  const lastVs = V.minimax[V.minimax.length - 1];
  const lastTree = V.tree[V.tree.length - 1];
  const firstM = MO.rows[0];
  const lastM = MO.rows[MO.rows.length - 1];
  const far = (r) => r.horizon[r.horizon.length - 1];
  const draws = S.value === 0;

  set('opening-note',
    `The game on this page is connect four on a board ${M.rows} squares tall and ${M.cols} `
    + `wide, which is small enough to be finished off completely: every position reachable `
    + `from an empty board, ${n0(S.positions)} of them counting whose turn it is, has been `
    + `solved by working backwards from the ends. `
    + (draws
      ? `The answer is that the first player cannot win. With both sides perfect the game is a `
        + `draw, and ${S.best_moves.length} of the ${M.cols} opening columns hold it: `
        + `${S.best_moves.join(', ')}. `
      : `The first player wins with perfect play, opening in ${S.best_moves.join(' or ')}. `)
    + `That is what makes the rest of the page possible. When the search picks a column, the `
    + `page is not comparing it to another search or to a rating: it knows.`);

  set('solve-intro',
    `Here is a position from that game, and a search that runs when you press a button. It has `
    + `no idea what connect four is. It knows the rules, it can tell when a game has ended, and `
    + `it has a budget: play the rest of the game out at random this many times, keep count, and `
    + `then move where the counting went. Under each column is what that move is really worth, `
    + `which the search never sees.`);

  set('board-note',
    `Two things are worth watching. The first is that the counts are not the point and the `
    + `column is: two searches on the same position, in two languages, drawing different random `
    + `numbers, will hand you different counts and the same move, once the budget is large `
    + `enough. The second is what "large enough" means here, which is not what it sounds like: `
    + `on this position the generator's own run picked a losing column at `
    + `${T.frames.filter((f) => !f.correct).map((f) => f.sims).join(' and ')} simulations and the `
    + `right one at ${T.frames.filter((f) => f.correct).map((f) => f.sims).join(' and ')}. `
    + `More thinking is better on average and not on every position, which is a distinction that `
    + `only exists because there is an exact answer to check against.`);

  set('steps-intro',
    `The whole method is four steps repeated until the budget runs out. Walk down the tree you `
    + `have, always taking the child with the largest score; add one node; play the rest of the `
    + `game out at random from there; carry the result back up. Everything interesting lives in `
    + `the score, which adds two quantities that have nothing to do with each other.`);

  set('steps-note',
    `The first term is what that move has been worth so far. The second is an apology for not `
    + `having tried it much: it is large when the child has few visits and shrinks as the count `
    + `grows, so a move that has looked bad twice keeps getting looked at, and a move that has `
    + `looked bad two hundred times does not. It is the same trade the bandit articles measured, `
    + `<a href="../bandits/">explore then commit</a> and `
    + `<a href="../thompson-ucb/">the upper confidence bound</a>, applied at every node of a `
    + `tree instead of once at a row of machines. The constant in front of it decides the `
    + `exchange rate, and it is swept further down.`);

  set('scrolly-note',
    `Nothing in those four steps mentions connect four, and that is the selling point: hand the `
    + `same code a different set of rules and it plays that instead. It is also the weakness. The `
    + `play out is random, so the value it reports is the value of a position under two players `
    + `who are not trying, and a position that is winning only if you find the one right `
    + `continuation looks, to a random play out, like a coin flip.`);

  set('dials-intro',
    `Three knobs, one standard to judge them by. How long it thinks, how much the formula pays `
    + `for a move being untried, and what the play out does; and every answer read against the `
    + `same floor, which is what the same configuration does when the only thing that changes is `
    + `the search's own random numbers.`);

  set('dials-note',
    `The middle panel is the one to argue with. The constant matters: the sweep covers `
    + `${pts(bestC.share - worstC.share)} between its best and worst against `
    + `${pts(floor.spread)} of movement from reseeding alone. But the textbook value `
    + `${C.textbook_c} is not the top of the curve here, it scores ${pct(book.share)} against `
    + `${pct(bestC.share)} at ${bestC.c}, and it lands on exactly the same score as turning the `
    + `exploration term off. That is not a scandal: ${C.textbook_c} comes from a bound that `
    + `assumes rewards in a range this game does not use, and a board this small has a shallow `
    + `tree in which almost anything gets visited eventually. It is worth saying out loud `
    + `because a constant copied from a paper is the most common untested number in a search.`);

  set('model-intro',
    `Everything so far assumed a simulator: something that can be asked what happens if, as many `
    + `times as you like, for free. Take that away and you have to learn one, and then the plan `
    + `is made inside a model rather than inside the world. The question that decides whether `
    + `that works is not how good the model is. It is how fast being slightly wrong compounds.`);

  set('model-note',
    `That table is the argument for planning short and looking often. A model whose one step `
    + `error you cannot improve, because it is already at the noise floor of the world it copies, `
    + `is still ${(firstM.weight_error / lastM.weight_error).toFixed(0)} times wrong in its `
    + `parameters at the small sample size, and the plan feels it: over ${far(firstM).k} imagined `
    + `steps the error is ${far(firstM).error} against ${far(lastM).error}. Every system that `
    + `plans in a learnt model does something about this. Plan a short horizon and replan from `
    + `the real state; learn a value at the end of the horizon so the plan does not have to be `
    + `long; or keep the model honest about its own uncertainty and stop trusting it when it `
    + `admits to not knowing.`);

  set('verdict-intro',
    `Two families in one article, joined by a single assumption: search needs something it can `
    + `ask what happens if, and everything below is about where that something comes from and `
    + `how far you can push it.`);

  set('verdict-mcts',
    `${pct(hi.share)} of the best moves found on a solved game, knowing nothing but the rules, `
    + `and it improves with thinking time without retraining anything`);
  set('verdict-mcts-warn',
    `it plateaus: ${hi.correct - lo.correct} more positions of ${hi.of} bought by going from `
    + `${lo.sims} simulations to ${n0(hi.sims)}, and random play outs misjudge positions that `
    + `need one exact continuation`);
  set('verdict-minimax',
    `at ${n0(V.minimax[1].nodes)} positions it matches the tree search, and it is exact rather `
    + `than statistical: no seed, no variance, the same answer every time`);
  set('verdict-minimax-warn',
    `without an evaluation function it stops improving with depth, ${pct(lastVs.share)} at `
    + `${n0(lastVs.nodes)} positions against ${pct(lastTree.share)} for the tree on the same budget`);
  set('verdict-rollout',
    `cheap to add, and the thing every strong engine eventually replaces with a learnt evaluation`);
  set('verdict-rollout-warn',
    `measure it before believing it: the hint here is worth `
    + `${pts(Math.abs(floorSmart.mean - floor.mean))} at ${C.sims} simulations against a seed `
    + `spread of ${pts(floor.spread)}, which is nothing`);
  set('verdict-model',
    `the only option when experience is expensive, and a model good enough for a short plan is `
    + `much cheaper than one good enough for a long one`);
  set('verdict-model-warn',
    `one step accuracy does not tell you the plan is any good: it moved by `
    + `${(Math.abs(firstM.one_step - lastM.one_step)).toFixed(6)} across the whole sweep while `
    + `the ${far(firstM).k} step error moved by ${(far(firstM).error - far(lastM).error).toFixed(5)}`);

  set('closing-note',
    `That is the end of the branch, and of a run that started with `
    + `<a href="../linear-programming/">a problem where the answer is at a corner</a> and ends `
    + `with one where the answer is at the end of a game nobody can see the end of. Look back `
    + `along it and the same shape keeps coming up. `
    + `<a href="../bandits/">A bandit</a> is this article's formula with a tree one node deep. `
    + `<a href="../q-learning/">Q-learning</a> is what you do when you cannot afford to think at `
    + `the moment of the decision and have to have thought beforehand. `
    + `<a href="../policy-gradient/">A policy gradient</a> is what you do when there is no `
    + `maximum you can take. And the strongest game playing systems built so far are all three `
    + `at once: a learnt value to end the play out early, a learnt policy to say where to look `
    + `first, and this tree in the middle deciding where the next simulation goes. `
    + `<a href="../confounding/">The next module</a> goes back to the question underneath all of `
    + `it, which every article in this branch was allowed to take for granted because it had a `
    + `simulator: how anybody knows that doing something caused anything at all.`);

  set('resources-note',
    `Everything was measured by <code>src/utils/generate_mcts_data.py</code> with seed `
    + `${M.seed}. The game is connect ${M.connect} on ${M.rows} by ${M.cols}, solved in full by `
    + `negamax with a memo (${n0(S.positions)} position and turn pairs). The search is plain `
    + `UCT: no priors, no learnt evaluation, random play outs unless the panel says otherwise. `
    + `Accuracy is measured on ${A.positions} reachable non terminal positions, the constant on `
    + `${C.positions}, the comparison against minimax on ${V.positions}, and the floor by `
    + `rerunning the same configuration with ${NZ.rows[0].seeds} search seeds. The world model `
    + `panel fits a linear model to a linear world with noise ${MO.noise} per step, so what is `
    + `swept is how much experience it saw and not what shape it has. Nothing on this page is a `
    + `wall clock: the costs are counted in positions looked at.`);

  const steps = [
    `This is the state of the tree after a couple of dozen simulations, and the figure is the `
      + `real one: it was warmed up here and then one more simulation was run with its trace `
      + `recorded. Each child carries what it has been worth and how often it has been tried, `
      + `and the formula reads both: the number on top is the average, the grey one below is `
      + `the bonus for being untried. The search takes the largest sum.`,
    `The child it picked has children of its own that have never been visited. It takes one of `
      + `them, at random, and that node joins the tree. This is the only place the tree grows, `
      + `one node per simulation, which is why a search with ${n0(hi.sims)} simulations has at `
      + `most ${n0(hi.sims)} nodes and usually far fewer that matter.`,
    `From the new node the game is played out with no method at all: legal moves at random until `
      + `somebody wins or the board fills. This is the part that sounds like it cannot possibly `
      + `work, and the reason it does is that it is not trying to evaluate the position, it is `
      + `sampling it, and the average of enough samples is a value.`,
    `The result goes back up the path that produced it, and the sign flips at every step, `
      + `because a game that is good for me is bad for you. That is the whole update: one visit `
      + `and one result added to every node on the path. Then it starts again.`,
  ];
  steps.forEach((t, i) => set(`scrolly-step-${i}`, t));
}
