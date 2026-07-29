/* The world, its exact values and its exact gradient, without any DOM.
 *
 * All three come out of linear algebra rather than sampling: for a fixed policy
 * the Bellman equation is linear, and so is the discounted visit distribution,
 * so both can be solved and the gradient written down. That is what turns the
 * rest of the page into a comparison instead of two estimates arguing.
 */
export function makeWorld(spec) {
  const { size, gamma, goal, trap, stepCost, goalReward, trapReward } = spec;
  const nS = size * size;
  const nA = 4;
  const deltas = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const move = (s, a) => {
    const row = Math.floor(s / size);
    const col = s % size;
    const nr = Math.min(Math.max(row + deltas[a][0], 0), size - 1);
    const nc = Math.min(Math.max(col + deltas[a][1], 0), size - 1);
    return nr * size + nc;
  };
  const R = [];
  for (let s = 0; s < nS; s++) {
    R.push([]);
    for (let a = 0; a < nA; a++) {
      if (s === goal) { R[s].push(0); continue; }
      const n = move(s, a);
      R[s].push(stepCost + (n === goal ? goalReward : 0) + (n === trap ? trapReward : 0));
    }
  }
  return { size, gamma, goal, trap, nS, nA, move, R };
}

export function softmaxRows(theta, nS, nA) {
  const pi = [];
  for (let s = 0; s < nS; s++) {
    const row = theta.slice(s * nA, (s + 1) * nA);
    const m = Math.max(...row);
    const e = row.map((v) => Math.exp(v - m));
    const z = e.reduce((a, b) => a + b, 0);
    pi.push(e.map((v) => v / z));
  }
  return pi;
}

/* Gaussian elimination: the systems here are twenty five by twenty five, so
 * there is no reason to be clever about it. */
export function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    const tmp = M[c]; M[c] = M[piv]; M[piv] = tmp;
    const d = M[c][c];
    for (let j = c; j <= n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c || Math.abs(M[r][c]) < 1e-15) continue;
      const f = M[r][c];
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row) => row[n]);
}

export function exactValues(world, pi) {
  const { nS, nA, gamma, goal, move, R } = world;
  const A = [];
  const b = [];
  for (let s = 0; s < nS; s++) {
    const row = new Array(nS).fill(0);
    row[s] = 1;
    let rew = 0;
    if (s !== goal) {
      for (let a = 0; a < nA; a++) {
        row[move(s, a)] -= gamma * pi[s][a];
        rew += pi[s][a] * R[s][a];
      }
    }
    A.push(row);
    b.push(s === goal ? 0 : rew);
  }
  const V = solve(A, b);
  const Q = [];
  for (let s = 0; s < nS; s++) {
    Q.push([]);
    for (let a = 0; a < nA; a++) Q[s].push(R[s][a] + gamma * V[move(s, a)]);
  }
  return { V, Q };
}

export function exactGradient(world, pi, start) {
  const { nS, nA, gamma, goal, move } = world;
  const { V, Q } = exactValues(world, pi);
  /* the discounted visit distribution, from the transposed system */
  const A = [];
  const b = [];
  for (let s = 0; s < nS; s++) {
    const row = new Array(nS).fill(0);
    row[s] = 1;
    A.push(row);
    b.push(s === start ? 1 : 0);
  }
  for (let s = 0; s < nS; s++) {
    if (s === goal) continue;
    for (let a = 0; a < nA; a++) A[move(s, a)][s] -= gamma * pi[s][a];
  }
  const d = solve(A, b);
  d[goal] = 0;
  const grad = new Array(nS * nA).fill(0);
  for (let s = 0; s < nS; s++) {
    if (s === goal) continue;
    for (let a = 0; a < nA; a++) {
      for (let bIdx = 0; bIdx < nA; bIdx++) {
        grad[s * nA + bIdx] += d[s] * pi[s][a] * Q[s][a]
          * ((a === bIdx ? 1 : 0) - pi[s][bIdx]);
      }
    }
  }
  return { grad, V, Q, d };
}

/* One episode under the policy, and the estimator built from it. The browser
 * needs its own sampler for the guard: the whole article rests on the average
 * of many of these landing on the exact gradient, and a claim that big should
 * be rerun here rather than read out of the file that made it. */
export function rollout(world, pi, start, rand, maxSteps = 60) {
  const { goal, move } = world;
  let s = start;
  const traj = [];
  for (let t = 0; t < maxSteps; t++) {
    const u = rand();
    let acc = 0;
    let a = pi[s].length - 1;
    for (let k = 0; k < pi[s].length; k++) {
      acc += pi[s][k];
      if (u < acc) { a = k; break; }
    }
    traj.push([s, a, world.R[s][a]]);
    s = move(s, a);
    if (s === goal) break;
  }
  return traj;
}

export function sampledGradient(world, pi, start, rand, episodes, baseline) {
  const { nS, nA, gamma } = world;
  const grad = new Array(nS * nA).fill(0);
  for (let e = 0; e < episodes; e++) {
    const traj = rollout(world, pi, start, rand);
    const returns = new Array(traj.length).fill(0);
    let G = 0;
    for (let t = traj.length - 1; t >= 0; t--) { G = traj[t][2] + gamma * G; returns[t] = G; }
    traj.forEach(([s, a], t) => {
      const adv = returns[t] - (baseline ? baseline[s] : 0);
      for (let b = 0; b < nA; b++) {
        grad[s * nA + b] += (gamma ** t) * adv * ((a === b ? 1 : 0) - pi[s][b]);
      }
    });
  }
  return grad.map((v) => v / episodes);
}

export const norm = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
export const cosine = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0)
  / (norm(a) * norm(b) + 1e-12);
