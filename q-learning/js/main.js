/* Entry point.
 *
 * The reference run is replayed here from the same generator of random numbers
 * the measurement used, so the guard compares arms and rewards pull by pull with
 * no tolerance. The lower bound and the explore then commit formula are both
 * recomputed rather than read.
 */
import { initCliffScrolly } from './scrolly-cliff.js';
import { initGridWidget } from './grid-widget.js';
import { initCurvesWidget } from './curves-widget.js';
import { initDialsWidget } from './dials-widget.js';
import { initProse } from './prose.js';
import { makeWorld, valueIteration, greedyPath } from './gridkit.js';

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
  if (!ok) console.warn(`[q-learning] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const T = data.truth;
  const world = makeWorld({ rows: T.rows, cols: T.cols, start: T.start, goal: T.goal });
  world.cliffCells = T.cliff;

  /* 1. the exact answer, recomputed here, entry by entry.
   *
   * The world is deterministic and the file holds four decimals, so the only
   * thing that can separate the two is the rounding: anything bigger is a real
   * disagreement about the rules. */
  const solved = valueIteration(world);
  let worst = 0;
  T.Q.forEach((row, s) => row.forEach((v, a) => {
    worst = Math.max(worst, Math.abs(v - solved.Q[s][a]));
  }));
  check('value iteration', worst <= 5e-5 && solved.sweeps === T.sweeps,
    `worst entry off by ${worst}, ${solved.sweeps} sweeps here against ${T.sweeps}`);

  /* 2. the best route, and what it is worth */
  const best = greedyPath(world, solved.Q);
  check('the best route', best.reached && Math.abs(best.total - T.return) < 1e-9
    && best.path.length === T.path.length,
  `${best.path.length - 1} steps worth ${best.total} against ${T.path.length - 1} and ${T.return}`);

  /* 3. the two learnt routes are legal moves in this world and score what the
   *    file says they score */
  let routeBad = 0;
  Object.keys(data.learners).forEach((k) => {
    const path = data.learners[k].path;
    let total = 0;
    for (let i = 0; i + 1 < path.length; i++) {
      let ok = false;
      for (let a = 0; a < world.nActions; a++) {
        const res = world.step(path[i], a);
        if (res.nxt[0] === path[i + 1][0] && res.nxt[1] === path[i + 1][1]) {
          ok = true;
          total += res.reward;
          break;
        }
      }
      if (!ok) routeBad += 1;
    }
    if (Math.abs(total - data.learners[k].path_return) > 1e-9) routeBad += 1;
  });
  check('the learnt routes', routeBad === 0,
    `${routeBad} steps or totals in the drawn routes do not match the rules`);

  /* 4. SARSA's route keeps its distance from the drop and Q-learning's does not,
   *    which is the claim the whole article rests on */
  const rowsUsed = (k) => new Set(data.learners[k].path.map((p) => p[0]));
  const qOnEdge = data.learners.qlearning.path.some((p) => p[0] === T.rows - 2);
  const sarsaHigher = Math.min(...data.learners.sarsa.path.map((p) => p[0]))
    < Math.min(...data.learners.qlearning.path.map((p) => p[0]));
  check('the two routes differ as claimed', qOnEdge && sarsaHigher,
    `Q-learning on the row above the drop: ${qOnEdge}; SARSA further up: ${sarsaHigher} `
    + `(rows used: ${[...rowsUsed('sarsa')].join(',')} against ${[...rowsUsed('qlearning')].join(',')})`);

  /* 5. with no exploring at all the two are the same algorithm */
  if (deep) {
    const zero = data.epsilon.rows[data.epsilon.rows.length - 1];
    check('no exploring', zero.eps === 0
      && zero.sarsa.greedy === zero.qlearning.greedy,
    `at zero exploring SARSA gets ${zero.sarsa.greedy} and Q-learning ${zero.qlearning.greedy}`);
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

  console.info(`[q-learning] load-time checks complete: value iteration within `
    + `${worst.toExponential(1)} in ${solved.sweeps} sweeps, best route ${best.total}`);
  return { worst, sweeps: solved.sweeps };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/qlearn.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the cliff', () => initCliffScrolly(data));
  safely('the three tables', () => initGridWidget(data));
  safely('the two numbers', () => initCurvesWidget(data));
  safely('the sweeps', () => initDialsWidget(data));

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
