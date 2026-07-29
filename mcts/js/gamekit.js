/* The game and the search, in the browser.
 *
 * Both are written again here rather than replayed from the file, because the
 * whole point of the page is that a search is something you can watch run: the
 * board widget runs a real tree search on a real position and the guard checks
 * it against the exact answer the generator solved for. The two languages will
 * never draw the same random numbers, so what is compared is what a search is
 * supposed to get right, not the sequence it took to get there.
 */
export function makeGame(rows, cols, connect) {
  const lines = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      [[0, 1], [1, 0], [1, 1], [1, -1]].forEach(([dr, dc]) => {
        const cells = [];
        let rr = r;
        let cc = c;
        for (let k = 0; k < connect; k++) {
          if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return;
          cells.push(rr * cols + cc);
          rr += dr;
          cc += dc;
        }
        cells.length === connect && lines.push(cells);
      });
    }
  }

  /* a state is one array per column, filled from the bottom up */
  const moves = (s) => s.map((col, i) => (col.length < rows ? i : -1)).filter((i) => i >= 0);
  const play = (s, c, who) => s.map((col, i) => (i === c ? col.concat([who]) : col));
  const key = (s, who) => `${who}|${s.map((col) => col.join('')).join(',')}`;

  function winner(s) {
    const g = new Array(rows * cols).fill(0);
    s.forEach((col, c) => col.forEach((v, r) => { g[r * cols + c] = v; }));
    for (let i = 0; i < lines.length; i++) {
      const [a, b, c2, d] = lines[i];
      const v = g[a];
      if (v && v === g[b] && v === g[c2] && v === g[d]) return v;
    }
    return 0;
  }

  /* null while the game is alive, 0.5 for a draw, otherwise the winner */
  function terminal(s) {
    const w = winner(s);
    if (w) return w;
    return moves(s).length ? null : 0.5;
  }

  return { rows, cols, connect, moves, play, key, winner, terminal };
}

export function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function rolloutRandom(game, s0, who0, rand, log) {
  let s = s0;
  let who = who0;
  for (;;) {
    const t = game.terminal(s);
    if (t !== null) { if (log) log.end = t; return t === 0.5 ? 0 : (t === who ? 1 : -1); }
    const ms = game.moves(s);
    s = game.play(s, ms[Math.floor(rand() * ms.length)], who);
    who = 3 - who;
    if (log) log.plies += 1;
  }
}

/* the same hint the generator gives it: take a win if there is one, block one
   if there is one, otherwise play at random */
function rolloutSmart(game, s0, who0, rand, log) {
  let s = s0;
  let who = who0;
  for (;;) {
    const t = game.terminal(s);
    if (t !== null) { if (log) log.end = t; return t === 0.5 ? 0 : (t === who ? 1 : -1); }
    const ms = game.moves(s);
    let pick = ms.find((c) => game.winner(game.play(s, c, who)) === who);
    if (pick === undefined) pick = ms.find((c) => game.winner(game.play(s, c, 3 - who)) === 3 - who);
    if (pick === undefined) pick = ms[Math.floor(rand() * ms.length)];
    s = game.play(s, pick, who);
    who = 3 - who;
    if (log) log.plies += 1;
  }
}

export function makeTree(game, state, who) {
  const rootKey = game.key(state, who);
  return { rootKey, N: { [rootKey]: 0 }, W: { [rootKey]: 0 },
    kids: {}, nodes: { [rootKey]: [state, who] }, move: {} };
}

/* One simulation: the four steps, in order, on a tree that persists between
 * calls. W is stored from the point of view of whoever moves AT that node,
 * which is the only thing to get right: the sign flips on every step up the
 * path, and again when a parent reads one of its children.
 *
 * With `trace` it also returns what it did, which is what the scrollytelling
 * figure draws: a real simulation rather than an illustration of one. */
export function simulate(game, tree, rand, { c = 1.4, smart = false, trace = false } = {}) {
  const { N, W, kids, nodes, move } = tree;
  const rollout = smart ? rolloutSmart : rolloutRandom;
  const path = [tree.rootKey];
  const scored = [];
  let k = tree.rootKey;
  let value = 0;
  let expanded = null;
  let plies = 0;
  let ended = null;
  for (;;) {
    const [s, w] = nodes[k];
    const t = game.terminal(s);
    if (t !== null) { value = t === 0.5 ? 0 : (t === w ? 1 : -1); ended = t; break; }
    if (!kids[k]) {
      kids[k] = game.moves(s).map((c2) => {
        const child = game.play(s, c2, w);
        const ck = game.key(child, 3 - w);
        nodes[ck] = [child, 3 - w];
        move[ck] = c2;
        if (!(ck in N)) { N[ck] = 0; W[ck] = 0; }
        return ck;
      });
    }
    const unseen = kids[k].filter((ck) => N[ck] === 0);
    if (unseen.length) {
      k = unseen[Math.floor(rand() * unseen.length)];
      expanded = k;
      path.push(k);
      const log = trace ? { plies: 0, end: null } : null;
      value = rollout(game, nodes[k][0], nodes[k][1], rand, log);
      if (log) { plies = log.plies; ended = log.end; }
      break;
    }
    const total = kids[k].reduce((a, ck) => a + N[ck], 0);
    let best = -Infinity;
    let bestk = kids[k][0];
    if (trace) scored.push({ parent: k, options: kids[k].map((ck) => ({
      key: ck, q: -W[ck] / N[ck], u: c * Math.sqrt(Math.log(total + 1) / N[ck]) })) });
    kids[k].forEach((ck) => {
      const q = -W[ck] / N[ck];
      const u = q + c * Math.sqrt(Math.log(total + 1) / N[ck]);
      if (u > best) { best = u; bestk = ck; }
    });
    k = bestk;
    path.push(k);
  }
  for (let j = 0; j < path.length; j++) {
    N[path[j]] += 1;
    W[path[j]] += (path.length - 1 - j) % 2 === 0 ? value : -value;
  }
  return { path, expanded, value, plies, ended, scored };
}

export function search(game, state, who, sims, { c = 1.4, smart = false, rand } = {}) {
  const tree = makeTree(game, state, who);
  for (let i = 0; i < sims; i++) simulate(game, tree, rand, { c, smart });
  return { ...summarise(game, tree, state), tree };
}

export function summarise(game, tree, state) {
  const counts = {};
  const values = {};
  game.moves(state).forEach((c2, i) => {
    const ck = tree.kids[tree.rootKey][i];
    counts[c2] = tree.N[ck];
    values[c2] = tree.N[ck] ? -tree.W[ck] / tree.N[ck] : 0;
  });
  const order = Object.keys(counts).map(Number);
  const pick = order.reduce((a, b) => (counts[b] > counts[a] ? b : a), order[0]);
  return { counts, values, pick };
}

/* The whole game below a position, solved. 260,000 nodes from the position this
 * page draws, which is about a second: too slow for load, fast enough for the
 * deep check, and it is the standard every number here is scored against. */
export function solveExact(game, state, who, memo = new Map()) {
  const k = game.key(state, who);
  const hit = memo.get(k);
  if (hit !== undefined) return hit;
  const t = game.terminal(state);
  if (t !== null) {
    const v = t === 0.5 ? 0 : (t === who ? 1 : -1);
    memo.set(k, v);
    return v;
  }
  let best = -2;
  game.moves(state).forEach((c) => {
    best = Math.max(best, -solveExact(game, game.play(state, c, who), 3 - who, memo));
  });
  memo.set(k, best);
  return best;
}

/* the grid the file publishes is written top row first, which is how a board is
   drawn; the search wants columns filled from the bottom */
export function stateFromGrid(grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  const s = [];
  for (let c = 0; c < cols; c++) {
    const col = [];
    for (let r = rows - 1; r >= 0; r--) if (grid[r][c]) col.push(grid[r][c]);
    s.push(col);
  }
  return s;
}
