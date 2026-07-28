/* Everything this article computes about the Swiss roll, in one place.
 *
 * The five hundred points, the sorted neighbour lists, the graph at any k, the
 * shortest paths, Isomap, locally linear embedding and the training loop of the
 * self-organising map. All of it mirrors, step for step, what
 * generate_manifolds_data.py runs in numpy, including both starting points that
 * would otherwise be random: the eigensolver's block (a linear congruential
 * generator written out identically at both ends) and the map's initial
 * prototypes and sample order (drawn by the generator and shipped in the file).
 * A training run that starts somewhere else is a different training run, and
 * "looks about the same" is not a check.
 */
import {
  pairwise, doubleCentre, topEigen, bottomEigen, allPairsShortest, components,
  procrustes, corr, trustworthiness, toRows, zeros,
} from '../../assets/js/embed.js';

/* ------------------------------------------------------------- the projection */
/* One fixed viewpoint, chosen so the turns of the spiral are all visible and
 * the height of the sheet reads as height. A rotation control would be nice and
 * would also be a second thing for the reader to get wrong. */
const VIEW = { a: 0.62, b: 0.42 };
export function project3(P) {
  const ca = Math.cos(VIEW.a);
  const sa = Math.sin(VIEW.a);
  const cb = Math.cos(VIEW.b);
  const sb = Math.sin(VIEW.b);
  return P.map(([X, Y, Z]) => {
    const x = X * ca + Z * sa;
    const zz = -X * sa + Z * ca;
    return [x, -(Y * cb) + zz * sb, zz];       // third entry is depth, for sorting
  });
}

/* -------------------------------------------------------------- the neighbours */
/* The sorted neighbour list of every point, computed once. Every value of k is
 * then a prefix of it, which is what makes the slider cheap. */
export function neighbourOrder(D) {
  const n = D.length;
  return Array.from({ length: n }, (_, i) => {
    const idx = [];
    for (let j = 0; j < n; j++) if (j !== i) idx.push(j);
    idx.sort((a, b) => D[i][a] - D[i][b]);
    return Int32Array.from(idx);
  });
}

export function graphAt(order, D, k) {
  const n = order.length;
  const adj = Array.from({ length: n }, () => new Map());
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const j = order[i][t];
      if (!adj[i].has(j)) edges.push([i, j]);
      adj[i].set(j, D[i][j]);
      adj[j].set(i, D[i][j]);
    }
  }
  /* An edge added from both ends is one edge; the list above can hold it twice
     when i is in j's list and j is in i's. */
  const seen = new Set();
  const uniq = [];
  edges.forEach(([i, j]) => {
    const key = i < j ? `${i},${j}` : `${j},${i}`;
    if (seen.has(key)) return;
    seen.add(key);
    uniq.push(i < j ? [i, j] : [j, i]);
  });
  return { adj, edges: uniq };
}

/* ------------------------------------------------------------------- Isomap */
export function isomapAt(state, k) {
  const { adj, edges } = graphAt(state.order, state.D, k);
  const comp = components(adj);
  const out = { k, edges, adj, components: comp.count, label: comp.label };
  /* An edge is a short circuit when the sheet has to be walked far to get
     between its two ends: near in the room, distant along the wall. */
  out.shorts = edges.filter(([i, j]) => state.GEO[i][j] > state.short * state.D[i][j]);
  if (comp.count > 1) return out;
  const Dg = allPairsShortest(adj);
  out.graphD = Dg;
  const flatG = [];
  const flatT = [];
  for (let i = 0; i < Dg.length; i++) {
    for (let j = i + 1; j < Dg.length; j++) {
      flatG.push(Dg[i][j]);
      flatT.push(state.GEO[i][j]);
    }
  }
  out.geoCorr = corr(flatG, flatT);
  const { values, vectors } = topEigen(doubleCentre(Dg), 2, { iters: 120, seed: 7 });
  const Z = zeros(Dg.length, 2);
  for (let j = 0; j < 2; j++) {
    const s = Math.sqrt(Math.max(values[j], 0));
    for (let i = 0; i < Dg.length; i++) Z[i][j] = vectors[i][j] * s;
  }
  out.coords = Z;
  out.disparity = procrustes(state.chart, Z).disparity;
  out.aligned = procrustes(state.chart, Z);
  return out;
}

/* ---------------------------------------------------------------------- LLE */
export function lleAt(state, k, reg = 1e-3) {
  const n = state.points.length;
  const W = zeros(n, n);
  const dim = state.points[0].length;
  for (let i = 0; i < n; i++) {
    const nb = state.order[i].subarray(0, k);
    /* the local Gram matrix of the neighbours, taken from i */
    const Zl = zeros(k, dim);
    for (let t = 0; t < k; t++) {
      for (let d = 0; d < dim; d++) Zl[t][d] = state.points[nb[t]][d] - state.points[i][d];
    }
    const C = zeros(k, k);
    let tr = 0;
    for (let a = 0; a < k; a++) {
      for (let b = a; b < k; b++) {
        let s = 0;
        for (let d = 0; d < dim; d++) s += Zl[a][d] * Zl[b][d];
        C[a][b] = s;
        C[b][a] = s;
      }
      tr += C[a][a];
    }
    for (let a = 0; a < k; a++) C[a][a] += reg * tr;
    const w = solveSym(C, new Float64Array(k).fill(1));
    let sum = 0;
    for (let a = 0; a < k; a++) sum += w[a];
    for (let a = 0; a < k; a++) W[i][nb[a]] = w[a] / sum;
  }
  /* M = (I - W)'(I - W) */
  const IW = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) IW[i][j] = (i === j ? 1 : 0) - W[i][j];
  }
  const M = zeros(n, n);
  for (let a = 0; a < n; a++) {
    for (let b = a; b < n; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += IW[i][a] * IW[i][b];
      M[a][b] = s;
      M[b][a] = s;
    }
  }
  const { values, vectors } = bottomEigen(M, 4, { iters: 80, shift: 1e-6, seed: 7 });
  const Z = zeros(n, 2);
  const scale = Math.sqrt(n);
  for (let i = 0; i < n; i++) {
    Z[i][0] = vectors[i][1] * scale;
    Z[i][1] = vectors[i][2] * scale;
  }
  return { coords: Z, values, aligned: procrustes(state.chart, Z) };
}

/* Cholesky solve for a small symmetric positive definite system. */
function solveSym(A, b) {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let t = 0; t < j; t++) s -= L[i][t] * L[j][t];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let t = 0; t < i; t++) s -= L[i][t] * y[t];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let t = i + 1; t < n; t++) s -= L[t][i] * x[t];
    x[i] = s / L[i][i];
  }
  return x;
}

/* --------------------------------------------------------------------- PCA */
export function pca2(points) {
  const n = points.length;
  const p = points[0].length;
  const m = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) m[j] += points[i][j] / n;
  const S = zeros(p, p);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) S[a][b] += (points[i][a] - m[a]) * (points[i][b] - m[b]) / n;
    }
  }
  const { vectors } = topEigen(S, 2, { iters: 200, seed: 7 });
  const Z = zeros(n, 2);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 2; j++) {
      let s = 0;
      for (let t = 0; t < p; t++) s += (points[i][t] - m[t]) * vectors[t][j];
      Z[i][j] = s;
    }
  }
  return Z;
}

/* ------------------------------------------------ the self-organising map */
/* Kohonen's loop, written out, with both draws taken from the file: which
 * points became the starting prototypes, and the order the samples arrive in
 * on each epoch. `upto` runs only the first so many epochs, which is what makes
 * the widget's step button possible. */
export function somTrain(X, cfg, upto = null) {
  const { rows, cols, epochs, orders, initIndex, sigma0, lr0 } = cfg;
  const n = X.length;
  const p = X[0].length;
  const cells = rows * cols;
  const P = zeros(cells, p);
  for (let c = 0; c < cells; c++) {
    for (let d = 0; d < p; d++) P[c][d] = X[initIndex[c]][d];
  }
  const grid = zeros(cells, 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      grid[r * cols + c][0] = r;
      grid[r * cols + c][1] = c;
    }
  }
  const G2 = zeros(cells, cells);
  for (let a = 0; a < cells; a++) {
    for (let b = 0; b < cells; b++) {
      G2[a][b] = (grid[a][0] - grid[b][0]) ** 2 + (grid[a][1] - grid[b][1]) ** 2;
    }
  }
  const total = epochs * n;
  let step = 0;
  const run = upto === null ? epochs : upto;
  for (let e = 0; e < run; e++) {
    const ord = orders[e];
    for (let t = 0; t < n; t++) {
      const i = ord[t];
      const frac = step / total;
      const sig = sigma0 * (0.5 / sigma0) ** frac;
      const lr = lr0 * (0.01 / lr0) ** frac;
      let best = 0;
      let bd = Infinity;
      for (let c = 0; c < cells; c++) {
        let s = 0;
        for (let d = 0; d < p; d++) s += (P[c][d] - X[i][d]) ** 2;
        if (s < bd) {
          bd = s;
          best = c;
        }
      }
      const denom = 2 * sig * sig;
      for (let c = 0; c < cells; c++) {
        const hc = lr * Math.exp(-G2[best][c] / denom);
        if (hc < 1e-6) continue;
        for (let d = 0; d < p; d++) P[c][d] += hc * (X[i][d] - P[c][d]);
      }
      step += 1;
    }
  }
  return { P, grid, rows, cols };
}

export function somScores(X, som) {
  const { P, grid, rows, cols } = som;
  const n = X.length;
  const best = new Int32Array(n);
  const second = new Int32Array(n);
  let qe = 0;
  for (let i = 0; i < n; i++) {
    let b0 = -1;
    let b1 = -1;
    let d0 = Infinity;
    let d1 = Infinity;
    for (let c = 0; c < P.length; c++) {
      let s = 0;
      for (let d = 0; d < X[0].length; d++) s += (P[c][d] - X[i][d]) ** 2;
      if (s < d0) {
        d1 = d0; b1 = b0; d0 = s; b0 = c;
      } else if (s < d1) {
        d1 = s; b1 = c;
      }
    }
    best[i] = b0;
    second[i] = b1;
    qe += Math.sqrt(d0);
  }
  let adjacent = 0;
  for (let i = 0; i < n; i++) {
    const a = grid[best[i]];
    const b = grid[second[i]];
    if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) <= 1.0001) adjacent += 1;
  }
  const cellsUsed = new Set(Array.from(best)).size;
  const Z = zeros(n, 2);
  for (let i = 0; i < n; i++) {
    Z[i][0] = grid[best[i]][0];
    Z[i][1] = grid[best[i]][1];
  }
  return {
    best, quantisation: qe / n, topographic: 1 - adjacent / n, coords: Z,
    ties: n - cellsUsed, rows, cols,
  };
}

/* -------------------------------------------------------------- the fixtures */
export function buildState(data) {
  const points = toRows(data.roll.points);
  const chart = toRows(data.roll.chart);
  const D = pairwise(points);
  const GEO = pairwise(chart);
  return {
    points,
    chart,
    scaled: toRows(data.roll.scaled),
    D,
    GEO,
    order: neighbourOrder(D),
    short: data.meta.short_factor,
    proj: project3(points),
  };
}

export const trust = (D, Z, k = 12) => trustworthiness(D, pairwise(Z), k);
