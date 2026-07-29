/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const R = data.representations;
  const A = data.ablation;
  const O = data.overestimate;
  const Wt = data.weights;
  const both = A.rows.find((d) => d.replay && d.target);
  const neither = A.rows.find((d) => !d.replay && !d.target);
  const replayOnly = A.rows.find((d) => d.replay && !d.target);
  const targetOnly = A.rows.find((d) => !d.replay && d.target);
  const single = O.rows.find((d) => !d.double);
  const double = O.rows.find((d) => d.double);
  const nWeights = Object.values(Wt.weights)
    .reduce((a, m) => a + (Array.isArray(m[0]) ? m.length * m[0].length : m.length), 0);

  set('opening-note',
    `The same cliff as <a href="../q-learning/">the previous article</a>, and one thing changed. `
    + `The learner no longer gets a slot per square: it gets a function of two numbers, where it `
    + `is across and how far down, and has to produce four values from that. `
    + `${M.rows * M.cols * 4} numbers become ${nWeights}. Everything else, the world, the rewards, `
    + `the update, is what it was, so anything that moves is the representation moving it.`);

  set('setup-intro',
    `Three things stop being true at once when the table goes. The update was a write to one cell `
    + `and is now a nudge to a function, so touching one square moves its neighbours. The target `
    + `is computed from the same function being trained, so it moves while you aim at it. And the `
    + `data arrives in the order the agent walks it, so it is not a sample of anything.`);

  set('setup-note',
    `Two smaller things had to change with it, and they are worth stating because they are usually `
    + `left out. The rewards are divided by ${M.scale} before the network sees them: with a `
    + `hundred going straight in, the first target is a hundred and the network never recovers. `
    + `And values are discounted at ${M.gamma}, where the previous article did not discount at `
    + `all. A table on an episodic task converges without a discount; a function that bootstraps `
    + `off its own output has no such guarantee, so the exact answer everything is scored against `
    + `is recomputed here with the same ${M.gamma}. The best route is still ${M.truth_return} `
    + `undiscounted, worth ${M.truth_discounted} once discounted.`);

  set('scrolly-step-0',
    `The exact answer again, from the same value iteration, now discounted at ${M.gamma}. `
    + `${M.rows * M.cols} squares, four actions, ${M.rows * M.cols * 4} numbers, every one of them `
    + `stored and none of them guessed.`);

  set('scrolly-step-1',
    `And the network's version of the same picture, computed in this page by running its `
    + `${nWeights} weights over all ${M.rows * M.cols} squares. From a distance it is the same `
    + `shape: low near the drop, rising towards the goal. Nothing forced that. It is what fitting `
    + `a smooth function to values that are themselves smooth in most places will do.`);

  set('scrolly-step-2',
    `Where the two disagree. The worst square is off by ${Wt.gap}, and the squares that are worst `
    + `are the ones nobody stands on: walking into the drop teleports you, so those cells are `
    + `never a starting point for anything and the network is extrapolating there with no data at `
    + `all. On the squares its own route visits it is off by ${Wt.gap_on_path}.`);

  set('scrolly-step-3',
    `And it still walks the right way. Its route is ${Wt.path.length - 1} steps worth `
    + `${Wt.return}, against ${M.truth_return} for the exact answer. Being wrong about a value and `
    + `being wrong about which action is best are different failures, and only the second one `
    + `costs anything.`);

  set('scrolly-note',
    `That is one network. Over ${R.network.seeds} seeds, and against two other ways of holding the `
    + `same values, the picture is less flattering.`);

  set('rep-note',
    `Three columns and one honest headline: <span class="bold">the table needs `
    + `${n0(M.table_episodes)} episodes and gets ${R.table.greedy} in `
    + `${R.table.reaches_goal} of ${R.table.seeds} seeds; the network needs ${n0(M.episodes)} and `
    + `gets ${R.network.greedy} in ${R.network.reaches_goal} of ${R.network.seeds}</span>. Six `
    + `times the experience for a worse hit rate, on a world the table can hold exactly. That is `
    + `the price of generalising, paid in full, on a problem where generalising buys nothing. `
    + `The middle column is the control that keeps this honest: a straight line in the same two `
    + `inputs `
    + `${R.linear.greedy === null ? 'never reaches the goal in any seed' : `gets ${R.linear.greedy}`}, `
    + `because a plane cannot bend around a cliff. So the failure is not "it generalised", it is `
    + `what it generalised with, and the network is the smallest thing on this list that can `
    + `represent the answer at all.`);

  set('ablation-intro',
    `Both crutches that made deep Q-networks work are usually described together. Removing them `
    + `one at a time says which does what, and the answer here is that only one of them is doing `
    + `the heavy lifting.`);

  set('ablation-note',
    `With both, the policy is worth ${both.greedy} and reaches the goal in `
    + `${both.reaches_goal} of ${both.of} seeds. Take the replay buffer away and it is `
    + `${targetOnly.greedy === null ? 'never able to reach the goal' : targetOnly.greedy} in `
    + `${targetOnly.reaches_goal} of ${targetOnly.of}; take the frozen target away and keep the `
    + `buffer and it is ${replayOnly.greedy === null ? 'never able to reach it' : replayOnly.greedy} `
    + `in ${replayOnly.reaches_goal}. With neither, ${neither.reaches_goal} of ${neither.of}. `
    + `<span class="bold">The buffer is the one that matters here</span>, and the reason is the `
    + `third thing in the list above: without it the network is trained on a walk, which is `
    + `${' '}forty correlated steps in a row through the same corner of the board, and a function `
    + `fitted to that forgets everywhere else. Read all of it against the whisker, `
    + `${both.gap_spread}, which is how far apart two seeds of the same cell land: with `
    + `${A.seeds} seeds a cell is only different from another cell if it is further away than `
    + `that, and two of these four are not.`);

  set('over-intro',
    `One more thing the table did not do. A network's estimate of what a square is worth can be `
    + `checked against what its own policy actually collects from that square, because those are `
    + `the same quantity, so the difference is a bias rather than a disagreement.`);

  set('over-note',
    `Two numbers, same question, and the comparison only works if both are discounted the same `
    + `way, which is worth saying because getting that wrong turns ${(-9.733).toFixed(3)} into `
    + `${M.truth_return} and invents a bias of ${(Math.abs(-13 + 9.733)).toFixed(2)} out of `
    + `nothing. Done properly: one network believes the start is worth `
    + `<span class="bold">${single.believes}</span> and its own policy collects ${single.gets}, so `
    + `it is optimistic by ${Math.abs(single.bias).toFixed(3)}. Two networks, one picking and the `
    + `other scoring, believe ${double.believes} and collect ${double.gets}: `
    + `<span class="bold">pessimistic by ${Math.abs(double.bias).toFixed(3)}</span>. The fix `
    + `overshoots, which is the known behaviour and not a surprise, and it is worth publishing `
    + `because "double fixes the overestimate" is usually said as if the residue were zero. The `
    + `exact value of that square is ${O.truth_at_start}.`);

  set('verdict-intro',
    `A world with ${M.rows * M.cols} squares does not need any of this, which is exactly why it is `
    + `the right place to measure what it costs. The table is what ${R.network.seeds} seeds say.`);

  set('verdict-table',
    `${R.table.greedy} in ${R.table.reaches_goal} of ${R.table.seeds} seeds after `
    + `${n0(M.table_episodes)} episodes, and its values land ${R.table.gap} from the exact answer`);
  set('verdict-table-warn',
    `it needs one slot per state, which is why this is the last article where it is an option`);
  set('verdict-net',
    `${nWeights} numbers instead of ${M.rows * M.cols * 4}, and it still finds the best route: `
    + `${R.network.greedy} when it arrives`);
  set('verdict-net-warn',
    `${n0(M.episodes)} episodes against the table's ${n0(M.table_episodes)}, and it arrives in `
    + `${R.network.reaches_goal} of ${R.network.seeds} seeds`);
  set('verdict-value',
    `its own estimate of the start is off by ${Math.abs(single.bias).toFixed(3)} with one network `
    + `and ${Math.abs(double.bias).toFixed(3)} with two, in opposite directions`);
  set('verdict-value-warn',
    `and the two numbers are only comparable if both are discounted the same way`);
  set('verdict-simple',
    `a straight line in the same inputs `
    + `${R.linear.greedy === null ? 'never arrives' : `gets ${R.linear.greedy}`}, so simpler is `
    + `not automatically safer either`);
  set('verdict-simple-warn',
    `the question is whether your function can represent the answer, not how many parameters it `
    + `has`);

  set('closing-note',
    `Everything in this branch so far learns a value and then acts greedily on it, which means the `
    + `policy is a maximum over the value function and inherits every error in it. `
    + `<a href="../policy-gradient/">The next article</a> stops doing that: it keeps the policy `
    + `itself as the thing being learnt and pushes it directly, which removes the maximum and `
    + `replaces it with a new problem, because the direction to push in has to be estimated from `
    + `samples and that estimate turns out to be very noisy. On a world small enough to compute `
    + `the exact gradient, that noise can be measured rather than described.`);

  set('resources-note',
    `Everything was measured by <code>src/<wbr>utils/<wbr>generate_<wbr>dqn_<wbr>data.py</code> with seed ${M.seed}. `
    + `The world is the cliff walk of the previous article, deterministic, ${M.rows} by `
    + `${M.cols}. The network is two inputs, one hidden layer of ${Wt.hidden} rectified units and `
    + `four outputs, trained with Adam at 1e-3, batches of 32 from a buffer of 10,000, target `
    + `copied every 200 steps, exploration falling from 0.3 to 0.05, rewards divided by `
    + `${M.scale}, discount ${M.gamma}, ${n0(M.episodes)} episodes and ${R.network.seeds} seeds. `
    + `The table gets ${n0(M.table_episodes)} episodes. The exact answer is value iteration to a `
    + `change below 1e-12 with the same discount.`);
}
