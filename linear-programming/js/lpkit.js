/* The solvers this page runs, written without touching the DOM.
 *
 * Everything here also exists in src/utils/generate_lp_data.py, and the guards
 * in main.js check the two against each other: the corners of the polytope, the
 * pivots of the simplex, the pivot count on the Klee and Minty cube, and the
 * node count of the branch and bound. A page about exact answers should not be
 * reading its own claims out of a file.
 */

/* Every pair of constraints, solved, and kept if no other constraint is
 * violated. That is the definition of a corner, and enumerating it is the whole
 * of the first section. */
export function enumerateVertices(A, b, tol = 1e-9) {
  const out = [];
  let pairs = 0;
  let singular = 0;
  for (let i = 0; i < b.length; i++) {
    for (let j = i + 1; j < b.length; j++) {
      pairs += 1;
      const det = A[i][0] * A[j][1] - A[i][1] * A[j][0];
      if (Math.abs(det) < 1e-12) { singular += 1; continue; }
      const x = (b[i] * A[j][1] - b[j] * A[i][1]) / det;
      const y = (A[i][0] * b[j] - A[j][0] * b[i]) / det;
      let ok = true;
      for (let k = 0; k < b.length; k++) {
        if (A[k][0] * x + A[k][1] * y > b[k] + tol) { ok = false; break; }
      }
      if (!ok) continue;
      if (out.some((p) => Math.abs(p.x - x) < 1e-9 && Math.abs(p.y - y) < 1e-9)) continue;
      out.push({ x, y, active: [i, j] });
    }
  }
  return { vertices: out, pairs, singular };
}

/* Counter-clockwise, so the ring can be drawn as one path. */
export function ring(vertices) {
  /* no d3 in here on purpose: this module is imported from Node to check it
     against the generator before any browser is involved */
  const cx = vertices.reduce((s, p) => s + p.x, 0) / vertices.length;
  const cy = vertices.reduce((s, p) => s + p.y, 0) / vertices.length;
  return vertices.slice().sort((p, q) =>
    Math.atan2(p.y - cy, p.x - cx) - Math.atan2(q.y - cy, q.x - cx));
}

/* max c'x subject to Ax <= b, x >= 0, with b >= 0 so the slack basis is a
 * feasible starting corner. Returns the path of corners it walked, which is
 * what the scrolly draws.
 *
 * `rule` picks the entering column: "dantzig" takes the most negative reduced
 * cost (the textbook one, and the one the cube below punishes), "bland" takes
 * the first, which cannot cycle and is slower on everything else. */
export function simplex(c, A, b, { rule = 'dantzig', maxIter = 100000 } = {}) {
  const m = A.length;
  const n = c.length;
  const T = [];
  for (let i = 0; i < m; i++) {
    const row = new Float64Array(n + m + 1);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1;
    row[n + m] = b[i];
    T.push(row);
  }
  const obj = new Float64Array(n + m + 1);
  for (let j = 0; j < n; j++) obj[j] = -c[j];
  T.push(obj);
  const basis = [];
  for (let i = 0; i < m; i++) basis.push(n + i);

  const point = () => {
    const x = new Array(n).fill(0);
    for (let i = 0; i < m; i++) if (basis[i] < n) x[basis[i]] = T[i][n + m];
    return x;
  };

  const path = [point()];
  let pivots = 0;
  for (;;) {
    let col = -1;
    if (rule === 'bland') {
      for (let j = 0; j < n + m; j++) if (T[m][j] < -1e-9) { col = j; break; }
    } else {
      let best = -1e-9;
      for (let j = 0; j < n + m; j++) if (T[m][j] < best) { best = T[m][j]; col = j; }
    }
    if (col < 0) break;
    let row = -1;
    let bestRatio = Infinity;
    for (let i = 0; i < m; i++) {
      if (T[i][col] > 1e-9) {
        const ratio = T[i][n + m] / T[i][col];
        if (ratio < bestRatio - 1e-12) { bestRatio = ratio; row = i; }
      }
    }
    if (row < 0) return { x: null, value: Infinity, pivots, path, unbounded: true };
    const piv = T[row][col];
    for (let j = 0; j <= n + m; j++) T[row][j] /= piv;
    for (let k = 0; k <= m; k++) {
      if (k === row || Math.abs(T[k][col]) < 1e-12) continue;
      const f = T[k][col];
      for (let j = 0; j <= n + m; j++) T[k][j] -= f * T[row][j];
    }
    basis[row] = col;
    pivots += 1;
    path.push(point());
    if (pivots > maxIter) break;
  }
  return { x: point(), value: T[m][n + m], pivots, path, unbounded: false };
}

/* The deformed cube. Every corner is within a whisker of the one above it, so
 * the steepest-looking direction is a trap at every step. */
export function kleeMinty(n) {
  const c = [];
  for (let j = 1; j <= n; j++) c.push(10 ** (n - j));
  const A = [];
  const b = [];
  for (let i = 1; i <= n; i++) {
    const row = new Array(n).fill(0);
    for (let j = 1; j < i; j++) row[j - 1] = 2 * 10 ** (i - j);
    row[i - 1] = 1;
    A.push(row);
    b.push(100 ** (i - 1));
  }
  return { c, A, b };
}

/* Densest first, and by index when two are equally dense. The tie break is
 * written out on both sides on purpose: numpy's argsort is not stable, so
 * leaving it implicit gave the generator one search order and the browser
 * another, and the two node counts stopped being the same number. */
export function densityOrder(w, v) {
  return w.map((_, i) => i).sort((a, c) => (v[c] / w[c] - v[a] / w[a]) || (a - c));
}

/* The relaxation of a knapsack has a closed form: sort by value per kilo and
 * fill until the bag is full, cutting the last item in half. That is both the
 * relaxation the article measures and the bound the search below prunes with. */
export function knapsackLP(w, v, cap, from = 0, taken = 0, order = null) {
  const idx = order || densityOrder(w, v);
  let total = taken;
  let left = cap;
  const x = new Array(w.length).fill(0);
  for (let k = from; k < idx.length && left > 1e-12; k++) {
    const i = idx[k];
    const take = Math.min(1, left / w[i]);
    x[i] = take;
    total += take * v[i];
    left -= take * w[i];
  }
  return { value: total, x, order: idx };
}

/* Depth first, taking the densest item first, and recording every node so the
 * widget can replay the search one node at a time. */
export function branchAndBound(w, v, cap, { useBound = true, nodeCap = 5e6 } = {}) {
  const order = densityOrder(w, v);
  const ws = order.map((i) => w[i]);
  const vs = order.map((i) => v[i]);
  const n = w.length;
  const bound = (i, room, taken) => {
    let total = taken;
    let left = room;
    for (let k = i; k < n && left > 1e-12; k++) {
      const take = Math.min(1, left / ws[k]);
      total += take * vs[k];
      left -= take * ws[k];
    }
    return total;
  };
  let best = 0;
  let nodes = 0;
  const trace = [];
  const stack = [{ i: 0, room: cap, taken: 0, parent: -1, took: null }];
  while (stack.length) {
    const node = stack.pop();
    const id = nodes;
    nodes += 1;
    if (nodes > nodeCap) return { best, nodes, trace, complete: false, order };
    let why = 'open';
    /* strictly greater, with no slack, exactly as the generator writes it: a
       tolerance here would prune a different set of nodes and the two node
       counts would stop being the same number */
    if (node.taken > best) { best = node.taken; why = 'incumbent'; }
    const bnd = bound(node.i, node.room, node.taken);
    if (node.i === n) why = why === 'incumbent' ? 'incumbent' : 'leaf';
    else if (useBound && bnd <= best + 1e-9) why = 'pruned';
    trace.push({ id, depth: node.i, taken: node.taken, room: node.room,
      bound: bnd, best, why, parent: node.parent, took: node.took });
    if (node.i === n || why === 'pruned') continue;
    if (ws[node.i] <= node.room) {
      stack.push({ i: node.i + 1, room: node.room - ws[node.i],
        taken: node.taken + vs[node.i], parent: id, took: true });
    }
    stack.push({ i: node.i + 1, room: node.room, taken: node.taken,
      parent: id, took: false });
  }
  return { best, nodes, trace, complete: true, order };
}
