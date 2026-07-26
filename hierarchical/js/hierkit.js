/* Agglomerative clustering from scratch: the full Lance-Williams family
 * (single, complete, average, ward), a scipy-style linkage matrix as output,
 * cluster extraction at any k, and a dendrogram layout. The browser checks
 * its own ward tree of the wine against scipy's at load time.
 */

export function distanceMatrix(points) {
  const n = points.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let f = 0; f < points[i].length; f++) {
        const d = points[i][f] - points[j][f];
        s += d * d;
      }
      const d = Math.sqrt(s);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
}

/* Lance-Williams update of d(merged, k) from d(i,k), d(j,k), d(i,j). */
function lw(method, dik, djk, dij, ni, nj, nk) {
  switch (method) {
    case 'single': return Math.min(dik, djk);
    case 'complete': return Math.max(dik, djk);
    case 'average': return (ni * dik + nj * djk) / (ni + nj);
    default: {   // ward, on euclidean distances, exactly as scipy
      const t = ni + nj + nk;
      return Math.sqrt(((ni + nk) * dik * dik + (nj + nk) * djk * djk - nk * dij * dij) / t);
    }
  }
}

/* Naive O(n^3) agglomeration: fine at the 178 points this article uses.
 * Returns a scipy-style linkage matrix: rows [a, b, distance, size], with
 * original points 0..n-1 and the merge of step s getting id n+s. */
export function agglomerate(points, method) {
  const n = points.length;
  const D = distanceMatrix(points);
  const active = new Set(Array.from({ length: n }, (_, i) => i));
  const size = new Float64Array(n).fill(1);
  const id = Array.from({ length: n }, (_, i) => i);
  const Z = [];
  for (let step = 0; step < n - 1; step++) {
    let bi = -1;
    let bj = -1;
    let bd = Infinity;
    for (const i of active) {
      for (const j of active) {
        if (j <= i) continue;
        if (D[i][j] < bd) { bd = D[i][j]; bi = i; bj = j; }
      }
    }
    Z.push([id[bi], id[bj], bd, size[bi] + size[bj]]);
    for (const k of active) {
      if (k === bi || k === bj) continue;
      const d = lw(method, D[bi][k], D[bj][k], bd, size[bi], size[bj], size[k]);
      D[bi][k] = d;
      D[k][bi] = d;
    }
    size[bi] += size[bj];
    id[bi] = n + step;
    active.delete(bj);
  }
  return Z;
}

/* Cut a linkage matrix into k clusters: replay merges until k remain.
 * Returns labels 0..k-1 in leaf order of appearance. */
export function cutTree(Z, n, k) {
  const parent = new Int32Array(2 * n - 1).fill(-1);
  const stop = n - k;
  for (let s = 0; s < stop; s++) {
    parent[Z[s][0]] = n + s;
    parent[Z[s][1]] = n + s;
  }
  const root = (v) => {
    while (parent[v] !== -1) v = parent[v];
    return v;
  };
  const labels = new Array(n);
  const seen = new Map();
  for (let i = 0; i < n; i++) {
    const r = root(i);
    if (!seen.has(r)) seen.set(r, seen.size);
    labels[i] = seen.get(r);
  }
  return labels;
}

/* Dendrogram layout from a linkage matrix: leaf order by tree traversal,
 * every merge as {x, y (its two children's x), h (height), a, b}. */
export function dendrogram(Z, n) {
  const children = new Map();
  Z.forEach((row, s) => children.set(n + s, [row[0], row[1]]));
  const leaves = [];
  (function walk(v) {
    if (v < n) { leaves.push(v); return; }
    const [a, b] = children.get(v);
    walk(a);
    walk(b);
  })(2 * n - 2);
  const leafX = new Map(leaves.map((v, i) => [v, i]));
  const x = new Map(leafX);
  const merges = Z.map((row, s) => {
    const [a, b, h] = row;
    const xa = x.get(a);
    const xb = x.get(b);
    const xm = (xa + xb) / 2;
    x.set(n + s, xm);
    return { a, b, h, xa, xb, xm, ha: a < n ? 0 : Z[a - n][2], hb: b < n ? 0 : Z[b - n][2] };
  });
  return { leaves, merges };
}

/* Adjusted Rand index (no noise convention needed here). */
export function ari(a, b) {
  const ka = Math.max(...a) + 1;
  const kb = Math.max(...b) + 1;
  const M = Array.from({ length: ka }, () => new Array(kb).fill(0));
  for (let i = 0; i < a.length; i++) M[a[i]][b[i]] += 1;
  const comb2 = (x) => x * (x - 1) / 2;
  let sumIJ = 0;
  const rows = new Array(ka).fill(0);
  const cols = new Array(kb).fill(0);
  for (let i = 0; i < ka; i++) {
    for (let j = 0; j < kb; j++) {
      sumIJ += comb2(M[i][j]);
      rows[i] += M[i][j];
      cols[j] += M[i][j];
    }
  }
  const sumA = rows.reduce((s, v) => s + comb2(v), 0);
  const sumB = cols.reduce((s, v) => s + comb2(v), 0);
  const total = comb2(a.length);
  const expected = sumA * sumB / total;
  const max = (sumA + sumB) / 2;
  return max === expected ? 1 : (sumIJ - expected) / (max - expected);
}
