/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const G = data.gradient;
  const T = data.training;
  const E = data.entropy;
  const O = data.optimal;
  const row = (e, k) => G.rows.find((d) => d.episodes === e && d.kind === k);
  const one = row(1, 'plain');
  const oneB = row(1, 'baseline');
  const big = G.rows.filter((d) => d.kind === 'plain').slice(-1)[0];
  const bigB = G.rows.filter((d) => d.kind === 'baseline').slice(-1)[0];
  const bestEnt = E.rows.reduce((a, b) => (b.J > a.J ? b : a), E.rows[0]);

  set('opening-note',
    `The world is ${M.size} by ${M.size} squares with a goal in one corner and a square in the `
    + `middle that pays ${O.trap_reward}, and the policy is one number per square and action, `
    + `turned into probabilities. Small enough that the exact gradient can be computed: for a `
    + `fixed policy the Bellman equation is linear and so is the discounted visit distribution, so `
    + `both get solved and the direction is written down rather than estimated. Everything after `
    + `that is a comparison against a known answer.`);

  set('grad-intro',
    `The rule is one line and it is not obvious that it works. Play some episodes; for each thing `
    + `you did, multiply how much better than average it turned out by the direction that would `
    + `have made it more likely; add all of that up. On average that sum points along the gradient `
    + `of what the policy is worth. On average.`);

  set('grad-note',
    `The two words at the end are the article. Averaged over ${n0(G.bias.plain.samples)} single `
    + `episode estimates, the sum lands ${G.bias.plain.distance} from the exact gradient with a `
    + `cosine of ${G.bias.plain.cosine}: unbiased, as promised. Any single one of those estimates `
    + `is <span class="bold">${one.error}</span> away from a gradient whose entire length is `
    + `${G.exact_norm}, pointing in a direction whose cosine with the truth is `
    + `<span class="bold">${one.cosine}</span>. It is not a noisy version of the answer. It is `
    + `nearly perpendicular to it.`);

  set('scrolly-step-0',
    `The exact gradient: one bar for each of the ${M.states} squares times ${M.actions} actions, `
    + `${G.exact_norm} long. It is what the policy gradient theorem says, computed rather than `
    + `sampled, and the grey bars stay behind everything that follows.`);

  set('scrolly-step-1',
    `One episode's estimate over the top of it, drawn with the distance the generator measured. `
    + `${one.error} away, ${(one.error / G.exact_norm).toFixed(1)} times the length of the thing `
    + `it is estimating. A step in that direction is a step in almost any direction.`);

  set('scrolly-step-2',
    `Sixteen episodes averaged: ${row(16, 'plain').error} away, cosine `
    + `${row(16, 'plain').cosine}. Better, and still further from the gradient than the gradient `
    + `is from nothing. Averaging cuts the error by the square root of the count, which is a slow `
    + `way to buy accuracy.`);

  set('scrolly-step-3',
    `Now the same sixteen episodes with one change: subtract from each return what that state was `
    + `already worth. ${row(16, 'baseline').error} away, cosine ${row(16, 'baseline').cosine}. `
    + `Nothing was added and nothing was approximated. A number that does not depend on the action `
    + `cannot change what the sum averages to, which the generator confirms `
    + `(${G.bias.baseline.distance} from the exact over ${n0(G.bias.baseline.samples)} draws), and `
    + `it can change how much the sum jumps around.`);

  set('scrolly-note',
    `Swept over both, that is the widget below: ${G.rows[0].repeats} estimates at each point, `
    + `measured against the exact gradient two ways.`);

  set('noise-note',
    `Two things worth taking away. First, the scale: at one episode the estimate is `
    + `${(one.error / G.exact_norm).toFixed(1)} times the gradient's own length away from it, and `
    + `it takes ${big.episodes} episodes to get the error below ${big.error}. Every practical `
    + `method in this family is a way of not paying that. Second, what the baseline is worth, said `
    + `as episodes rather than as "variance reduction": with it, ${bigB.episodes} episodes give a `
    + `cosine of ${bigB.cosine}; without it, the same ${big.episodes} episodes give ${big.cosine}, `
    + `and the plain estimator does not reach ${bigB.cosine} anywhere in this sweep. And it is `
    + `free in the only sense that matters: it does not bend the answer, it only stops the answer `
    + `from wobbling.`);

  set('train-intro',
    `So what does a real method add? Four runs on the same world, each differing from the one `
    + `above it by exactly one thing, with the value of the policy computed exactly at every point `
    + `rather than sampled.`);

  set('train-note',
    `The jump is in one place and it is not where the names suggest. A baseline moves the value `
    + `from ${T.reinforce.J} to ${T.baseline.J} and cuts the spread between seeds from `
    + `${T.reinforce.J_spread} to ${T.baseline.J_spread}. Reusing each batch eight times instead `
    + `of throwing it away moves it to <span class="bold">${T.repeat.J}</span> against a ceiling `
    + `of ${O.J}: at this budget that is the whole difference between failing and arriving. `
    + `And clipping, which is the part of PPO everyone can name, touches `
    + `<span class="bold">${(T.ppo.clipped * 100).toFixed(1)}%</span> of the samples, costs `
    + `${(T.repeat.J - T.ppo.J).toFixed(5)} of value here, and cuts the worst single policy jump `
    + `from ${T.repeat.kl_max} to ${T.ppo.kl_max}, ${(T.repeat.kl_max / T.ppo.kl_max).toFixed(1)} `
    + `times smaller. That is exactly what it is for: it is insurance on the reuse, and on a world `
    + `this forgiving the insurance is a small loss. The entropy sweep goes the same way. The best `
    + `bonus measured is ${bestEnt.beta} and every step up from there costs value, down to `
    + `${E.rows[E.rows.length - 1].J} at ${E.rows[E.rows.length - 1].beta}, while the policy stays `
    + `${E.rows[E.rows.length - 1].entropy} spread against ${E.max_entropy} for choosing at `
    + `random. One goal, no local optimum worth escaping, nothing for the insurance to insure.`);

  set('verdict-intro',
    `Everything above is one world with ${M.states} squares, which is what made the comparison `
    + `possible and is also why three of the four rows below come with a caveat about scale.`);

  set('verdict-reinforce',
    `unbiased, checked: ${n0(G.bias.baseline.samples)} estimates average to `
    + `${G.bias.baseline.distance} from the exact gradient`);
  set('verdict-reinforce-warn',
    `one episode lands ${one.error} away from a gradient of length ${G.exact_norm}, at cosine `
    + `${one.cosine}`);
  set('verdict-reuse',
    `eight passes over each batch takes the value from ${T.baseline.J} to ${T.repeat.J} at the `
    + `same number of episodes`);
  set('verdict-reuse-warn',
    `the policy that collected the data is not the one being updated any more, which is what the `
    + `next row is about`);
  set('verdict-clip',
    `it cuts the worst policy jump by ${(T.repeat.kl_max / T.ppo.kl_max).toFixed(1)} times while `
    + `touching ${(T.ppo.clipped * 100).toFixed(1)}% of the samples`);
  set('verdict-clip-warn',
    `it is not free: ${(T.repeat.J - T.ppo.J).toFixed(5)} of value here, and with one pass over `
    + `each batch there is nothing for it to clip`);
  set('verdict-entropy',
    `it keeps the policy spread: ${E.rows[E.rows.length - 1].entropy} against `
    + `${E.rows[0].entropy} with no bonus at all`);
  set('verdict-entropy-warn',
    `on this world it only costs: ${bestEnt.J} at ${bestEnt.beta} down to `
    + `${E.rows[E.rows.length - 1].J} at ${E.rows[E.rows.length - 1].beta}`);

  set('closing-note',
    `Both halves of this branch now have a shape. Learn a value and act greedily on it, and the `
    + `policy inherits every error in the value; learn the policy directly, and the direction to `
    + `move it in has to be guessed from returns that are nearly perpendicular to it. Everything `
    + `with a name is one of a small number of answers to those two problems. `
    + `<a href="../mcts/">The last article of this branch</a> stops learning altogether and does `
    + `the other thing you can do with a model of the world: search inside it before acting, and `
    + `then measure what happens when the model is one you learnt rather than one you were given.`);

  set('resources-note',
    `Everything was measured by <code>src/<wbr>utils/<wbr>generate_<wbr>pg_<wbr>data.py</code> with seed ${M.seed}. `
    + `The world is ${M.size} by ${M.size} with a discount of ${M.gamma}, a step costing `
    + `${O.step_cost}, a goal paying ${O.goal_reward} and a square in the middle paying `
    + `${O.trap_reward}. The policy is a softmax over one parameter per state and action. The `
    + `exact gradient comes from the policy gradient theorem with both the values and the `
    + `discounted visit distribution solved as linear systems; every point of the sweep is `
    + `${G.rows[0].repeats} independent estimates. The four runs are ${T.reinforce.seeds} seeds `
    + `each at batches of ${T.reinforce.batch}, and the entropy sweep ${E.rows[0].seeds}.`);
}
