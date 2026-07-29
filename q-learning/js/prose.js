/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const T = data.truth;
  const L = data.learners;
  const E = data.epsilon;
  const A = data.alpha;
  const B = data.bias;
  const zero = E.rows[E.rows.length - 1];
  const tenth = E.rows.find((d) => d.eps === M.eps) || E.rows[2];

  set('opening-note',
    `Two updates, one character apart, and they learn different things. Both look at where they `
    + `landed and ask what that place is worth; one asks what the <span class="bold">best</span> `
    + `action there would be worth, the other asks what the action it is <span class="bold">going `
    + `to take</span> is worth. On a world with a cliff in it, that difference is visible from `
    + `across the room, and it is measurable because this world is small enough to solve exactly.`);

  set('world-intro',
    `${T.rows} rows, ${T.cols} columns, four moves, and a step costs ${1}. Along the bottom, `
    + `between the start and the goal, there is a ${T.cliff.length} square drop: walk into it and `
    + `you pay ${100} and wake up back at the start. Nothing here is random. That matters, because `
    + `it means the answer can be computed rather than estimated, and everything the two learners `
    + `come back with can be scored against it instead of against each other.`);

  set('world-note',
    `Value iteration solves it in <span class="bold">${T.sweeps} sweeps</span>, with the largest `
    + `change in the last one being ${T.delta}, which is zero to the last bit a double has. The `
    + `best route is ${T.path.length - 1} steps and is worth ${T.return}. Both learners get the `
    + `same ${n0(M.episodes)} episodes, the same step size of ${M.alpha}, and take a random move `
    + `${pc(M.eps, 0)} of the time.`);

  set('scrolly-step-0',
    `The board. The dark band is the drop, and the two labelled squares are where every episode `
    + `starts and where it ends. A learner is told none of this: it finds out by walking into `
    + `things.`);

  set('scrolly-step-1',
    `Solved exactly. The shade of a square is what standing on it is worth if you play well from `
    + `there, and the arrows are what playing well means: where two moves are equally good, both `
    + `are drawn, which on this board is most squares. Notice the row above the drop, where they `
    + `point along it: the shortest way is the dangerous way, and nothing here is random, so `
    + `danger costs nothing.`);

  set('scrolly-step-2',
    `Q-learning comes back with exactly that route: ${L.qlearning.path.length - 1} steps along the `
    + `edge, worth ${L.qlearning.path_return}, in ${pc(L.qlearning.path_share, 0)} of the seeds. It `
    + `asks what the best next action is worth, so it is learning about a walker who never slips, `
    + `even while it is slipping.`);

  set('scrolly-step-3',
    `SARSA comes back one row higher: ${L.sarsa.path.length - 1} steps, worth `
    + `${L.sarsa.path_return}. It asks what the action it is <span class="bold">about to `
    + `take</span> is worth, and ${pc(M.eps, 0)} of the time that action is random. A walker who `
    + `sometimes steps sideways at random should not walk along a cliff, and SARSA is the only one `
    + `of the two that knows it is that walker.`);

  set('scrolly-note',
    `Both tables are drawn below over the same board, next to the exact answer, which the page `
    + `recomputes on load rather than reading.`);

  set('grid-note',
    `Two details in that table are worth stopping on. The first is that Q-learning's route is `
    + `${pc(L.qlearning.path_share, 0)} of its seeds and SARSA's is `
    + `${pc(L.sarsa.path_share, 0)}: the safe route is a family of routes, not one, because `
    + `anywhere above the drop is safe and the values there are close together. The second is the `
    + `last column. A greedy reading of SARSA's table reaches the goal in `
    + `<span class="bold">${L.sarsa.reaches_goal} of ${L.sarsa.seeds}</span> seeds against `
    + `${L.qlearning.reaches_goal} of ${L.qlearning.seeds}, and the reason is not that SARSA is `
    + `worse. Every reward here is negative and the table starts at zero, so a square nobody ever `
    + `tried looks better than anywhere they have been, and a policy read without care walks `
    + `straight into the part of the map it knows nothing about. The paths above are read only `
    + `over actions that were actually tried.`);

  set('curves-intro',
    `Which one is better depends on a question that is usually left unasked: better at what? There `
    + `are two numbers here and they point different ways.`);

  set('curves-note',
    `While learning, SARSA earns <span class="bold">${L.sarsa.online_last100}</span> an episode `
    + `and Q-learning ${L.qlearning.online_last100}, a difference of `
    + `${(L.sarsa.online_last100 - L.qlearning.online_last100).toFixed(1)} per episode, all of it `
    + `paid in falls. Once the exploring stops, Q-learning's policy is worth `
    + `<span class="bold">${L.qlearning.greedy_return}</span> and SARSA's ${L.sarsa.greedy_return}. `
    + `Neither column is the real one. If the cliff is a simulator, only the second matters and `
    + `Q-learning wins by ${(L.qlearning.greedy_return - L.sarsa.greedy_return).toFixed(1)}. If it `
    + `is a machine that breaks, the first is a bill and SARSA wins by `
    + `${(L.sarsa.online_last100 - L.qlearning.online_last100).toFixed(1)} an episode. The last `
    + `column is how far each table is from the exact answer on the squares its own route visits: `
    + `${L.qlearning.q_gap} for Q-learning and ${L.sarsa.q_gap} for SARSA, which is another way of `
    + `saying they are not estimating the same quantity.`);

  set('dials-intro',
    `Three sweeps: one that makes the difference between the two vanish, one that turns out not to `
    + `matter much, and one that is not about either of them but about the word "maximum".`);

  set('dials-note',
    `The first sweep settles what the difference actually is. At every level of exploring, `
    + `Q-learning's policy is worth ${tenth.qlearning.greedy} and SARSA's ${tenth.sarsa.greedy}, `
    + `until exploring reaches zero, where <span class="bold">both come back with `
    + `${zero.sarsa.greedy} in ${zero.sarsa.optimal_paths} of 20 seeds</span> and the two `
    + `algorithms are the same algorithm. SARSA is not more cautious by temperament: it is `
    + `estimating the value of the policy it is actually running, and when that policy stops `
    + `taking random steps there is nothing left to be cautious about. The step size sweep is the `
    + `boring one and it is worth publishing as boring. The third is different in kind. `
    + `A wrong turn leads to ${B.n_actions} actions that each pay ${B.mu} on average, so it is `
    + `worth ${B.truth_left} against ${B.truth_right} for going the other way, and a table that `
    + `updates towards the largest of ${B.n_actions} noisy estimates is updating towards the `
    + `luckiest of them. One table takes the wrong turn ${pc(B.final_single)} of the time at the `
    + `end of the run; two tables, each checking the other's pick, take it ${pc(B.final_double)}, `
    + `against a floor of ${pc(B.eps / 2, 0)} that exploring forces on both. That is `
    + `${((B.final_single - B.eps / 2) / Math.max(B.final_double - B.eps / 2, 1e-9)).toFixed(1)} `
    + `times as much error above the floor, from a single word in the update rule.`);

  set('verdict-intro',
    `One world, solved exactly, and two updates that differ by which action they ask about. The `
    + `table is what ${L.sarsa.seeds} seeds say about when that difference is worth having.`);

  set('verdict-q',
    `its policy is worth ${L.qlearning.greedy_return} against SARSA's ${L.sarsa.greedy_return}, `
    + `and it is the exact answer value iteration found`);
  set('verdict-q-warn',
    `it earns ${L.qlearning.online_last100} an episode getting there against `
    + `${L.sarsa.online_last100}, all of it paid in falls`);
  set('verdict-sarsa',
    `it learns the value of the policy you are running, exploration included, so its route stays `
    + `${L.sarsa.path.length - L.qlearning.path.length} steps away from the drop`);
  set('verdict-sarsa-warn',
    `turn the exploring off and it becomes Q-learning: both find ${zero.sarsa.greedy} in `
    + `${zero.sarsa.optimal_paths} of 20 seeds`);
  set('verdict-double',
    `a maximum over ${B.n_actions} noisy estimates is a maximum over the luckiest, and two tables `
    + `cut the wrong turns from ${pc(B.final_single)} to ${pc(B.final_double)}`);
  set('verdict-double-warn',
    `it costs twice the memory and half the data per table, and on a world with no noise in it `
    + `there is nothing to fix`);
  set('verdict-vi',
    `${T.sweeps} sweeps and the answer is exact, with the last sweep moving nothing by more than `
    + `${T.delta}`);
  set('verdict-vi-warn',
    `it needs the rules written down, and ${T.rows * T.cols} squares to sweep; the next article is `
    + `about what happens when there are too many to store`);

  set('closing-note',
    `Everything here rests on one thing the cliff has and almost nothing else does: `
    + `${T.rows * T.cols} squares and ${T.actions.length} actions, so a value can be kept for every `
    + `one of them and the answer can be looked up. A board game has more positions than there are `
    + `atoms nearby, and a camera has more pictures than that. `
    + `<a href="../dqn/">The next article</a> replaces the table with a function that guesses the `
    + `values it was never shown, and measures what that costs: the guarantees on this page all `
    + `assumed the table.`);

  set('resources-note',
    `Everything was measured by <code>src/utils/generate_qlearning_data.py</code> with seed `
    + `${M.seed}. The world is the cliff walk of Sutton and Barto: ${T.rows} by ${T.cols}, `
    + `deterministic, minus one a step and minus a hundred into the drop. Value iteration runs to `
    + `a change below 1e-12. Both learners get ${n0(M.episodes)} episodes, step size ${M.alpha}, `
    + `exploring ${pc(M.eps, 0)} of the time, over ${L.sarsa.seeds} seeds; the sweeps use 20 seeds `
    + `per point. The maximisation bias runs ${n0(B.seeds)} independent copies of the two state `
    + `problem for ${B.episodes} episodes each.`);
}
