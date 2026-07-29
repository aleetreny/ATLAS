/* The world and its exact answer, without any DOM.
 *
 * Forty eight squares and four actions is small enough that the page can solve
 * it rather than read the solution, so the guard in main.js reruns value
 * iteration here and compares it with the file entry by entry. Everything is
 * deterministic, so that comparison has no tolerance in it beyond the rounding
 * the file was written with.
 */
export const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
export const ACTION_NAMES = ['up', 'down', 'left', 'right'];

export function makeWorld(spec) {
  const { rows, cols, start, goal } = spec;
  const isCliff = ([r, c]) => r === rows - 1 && c > 0 && c < cols - 1;
  const sid = ([r, c]) => r * cols + c;
  const step = (rc, a) => {
    const nr = Math.min(Math.max(rc[0] + ACTIONS[a][0], 0), rows - 1);
    const nc = Math.min(Math.max(rc[1] + ACTIONS[a][1], 0), cols - 1);
    const nxt = [nr, nc];
    if (isCliff(nxt)) return { nxt: start.slice(), reward: -100, done: false };
    return { nxt, reward: -1, done: nr === goal[0] && nc === goal[1] };
  };
  return { rows, cols, start, goal, isCliff, sid, step,
    nStates: rows * cols, nActions: ACTIONS.length };
}

/* Sweep until nothing moves. No discounting: the episode ends, so there is
 * nothing to discount, and the stopping rule is the precision of a double
 * rather than a round number somebody liked. */
export function valueIteration(world, { tol = 1e-12, maxSweeps = 100000, gamma = 1 } = {}) {
  const Q = [];
  for (let s = 0; s < world.nStates; s++) Q.push(new Array(world.nActions).fill(0));
  let sweeps = 0;
  let delta = 0;
  for (let it = 0; it < maxSweeps; it++) {
    delta = 0;
    for (let s = 0; s < world.nStates; s++) {
      const rc = [Math.floor(s / world.cols), s % world.cols];
      if (rc[0] === world.goal[0] && rc[1] === world.goal[1]) continue;
      for (let a = 0; a < world.nActions; a++) {
        const { nxt, reward, done } = world.step(rc, a);
        const target = reward + (done ? 0 : gamma * Math.max(...Q[world.sid(nxt)]));
        delta = Math.max(delta, Math.abs(target - Q[s][a]));
        Q[s][a] = target;
      }
    }
    sweeps += 1;
    if (delta < tol) break;
  }
  return { Q, sweeps, delta };
}

/* The path a table produces once the exploring stops. Actions that were never
 * tried are excluded when counts are supplied: every reward here is negative
 * and the table starts at zero, so an untouched entry looks better than
 * anything learnt and a careless reading walks straight into it. */
export function greedyPath(world, Q, counts = null, limit = 200) {
  let rc = world.start.slice();
  const path = [rc.slice()];
  let total = 0;
  for (let i = 0; i < limit; i++) {
    const s = world.sid(rc);
    let best = -1;
    let bestV = -Infinity;
    for (let a = 0; a < world.nActions; a++) {
      const seen = !counts || counts[s].some((n) => n > 0) === false || counts[s][a] > 0;
      const v = seen ? Q[s][a] : -Infinity;
      if (v > bestV) { bestV = v; best = a; }
    }
    const { nxt, reward, done } = world.step(rc, best);
    total += reward;
    rc = nxt;
    path.push(rc.slice());
    if (done) return { path, total, reached: true };
  }
  return { path, total, reached: false };
}
