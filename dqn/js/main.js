/* Entry point.
 *
 * The reference run is replayed here from the same generator of random numbers
 * the measurement used, so the guard compares arms and rewards pull by pull with
 * no tolerance. The lower bound and the explore then commit formula are both
 * recomputed rather than read.
 */
import { initNetScrolly } from './scrolly-net.js';
import { initRepWidget } from './rep-widget.js';
import { initAblationWidget } from './ablation-widget.js';
import { initOverWidget } from './over-widget.js';
import { initProse } from './prose.js';
import { makeWorld, valueIteration, greedyPath } from './gridkit.js';
import { forward, observe } from './netkit.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
}

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(label, ok, detail) {
  if (!ok) console.warn(`[dqn] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const M = data.meta;
  const world = makeWorld({ rows: M.rows, cols: M.cols, start: M.start, goal: M.goal });
  world.cliffCells = M.cliff;

  /* 1. the exact answer, recomputed with this article's discount */
  const solved = valueIteration(world, { gamma: M.gamma });
  let worst = 0;
  M.truth_Q.forEach((row, s) => row.forEach((v, a) => {
    worst = Math.max(worst, Math.abs(v - solved.Q[s][a]));
  }));
  check('value iteration', worst <= 5e-4,
    `worst entry off by ${worst}`);

  /* 2. the network, run here, against the values the file recorded.
   *
   * The weights travel with six decimals and pass through a hidden layer, so
   * the budget is a few times that rather than machine precision. */
  let worstNet = 0;
  data.weights.Q.forEach((row, s) => {
    const rc = [Math.floor(s / M.cols), s % M.cols];
    const mine = forward(data.weights.weights, observe(rc, M.rows, M.cols))
      .map((v) => v * data.weights.scale);
    row.forEach((v, a) => { worstNet = Math.max(worstNet, Math.abs(v - mine[a])); });
  });
  check('the network', worstNet < 1e-2,
    `running the weights here moves a value by ${worstNet}`);

  /* 3. the route the network takes, read from the network rather than the file */
  const qfun = (rc) => forward(data.weights.weights, observe(rc, M.rows, M.cols));
  let rc = M.start.slice();
  let total = 0;
  let reached = false;
  const mine = [rc.slice()];
  for (let i = 0; i < 200; i++) {
    const q = qfun(rc);
    let a = 0;
    for (let k = 1; k < q.length; k++) if (q[k] > q[a]) a = k;
    const res = world.step(rc, a);
    total += res.reward;
    rc = res.nxt;
    mine.push(rc.slice());
    if (res.done) { reached = true; break; }
  }
  check('the route', reached && Math.abs(total - data.weights.return) < 1e-9
    && mine.length === data.weights.path.length,
  `${mine.length - 1} steps worth ${total} against ${data.weights.path.length - 1} `
    + `and ${data.weights.return}`);

  /* 4. the table beats the network on sample efficiency, which is the headline */
  check('the headline', data.representations.table.reaches_goal
    >= data.representations.network.reaches_goal
    && M.table_episodes < M.episodes,
  `table ${data.representations.table.reaches_goal} of `
    + `${data.representations.table.seeds} in ${M.table_episodes} episodes, network `
    + `${data.representations.network.reaches_goal} in ${M.episodes}`);

  /* 5. the two crutches: the buffer is the one that matters here */
  if (deep) {
    const both = data.ablation.rows.find((d) => d.replay && d.target);
    const noReplay = data.ablation.rows.find((d) => !d.replay && d.target);
    check('the ablation', both.reaches_goal >= noReplay.reaches_goal,
      `with the buffer ${both.reaches_goal} of ${both.of}, without it ${noReplay.reaches_goal}`);
  }

  /* 6. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info(`[dqn] load-time checks complete: value iteration within ${worst.toExponential(1)}, `
    + `network within ${worstNet.toExponential(1)}, route ${total}`);
  return { worst, worstNet };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/dqn.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the network on the board', () => initNetScrolly(data));
  safely('three representations', () => initRepWidget(data));
  safely('the two crutches', () => initAblationWidget(data));
  safely('what it believes', () => initOverWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
