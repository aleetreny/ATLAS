/* DBSCAN from scratch, in the browser, plus the small measurement toolkit
 * the article needs (k-distance curves and the adjusted Rand index). All of
 * it is checked at load time against scikit-learn reference answers shipped
 * in data/density.json.
 */

const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;

/* Neighbour lists within eps, O(n^2): fine at this article's few hundred
 * points, and honest about what the algorithm actually computes. */
export function neighbours(points, eps) {
  const e2 = eps * eps;
  const n = points.length;
  const out = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (d2(points[i], points[j]) <= e2) {
        out[i].push(j);
        out[j].push(i);
      }
    }
  }
  return out;
}

/* The algorithm. A point is core if its eps-ball holds minPts points
 * (itself included). Clusters are the connected components of core points,
 * grown by BFS; non-core points in reach of a core become border points of
 * that cluster; everything else is noise (-1).
 * Returns labels plus the per-point role ('core' | 'border' | 'noise') and
 * the BFS discovery order of the first cluster, which the scrolly animates. */
export function dbscan(points, eps, minPts) {
  const nb = neighbours(points, eps);
  const n = points.length;
  const core = nb.map((list) => list.length + 1 >= minPts);
  const labels = new Array(n).fill(-1);
  const role = points.map((_, i) => (core[i] ? 'core' : 'noise'));
  let c = -1;
  let firstOrder = null;
  for (let i = 0; i < n; i++) {
    if (!core[i] || labels[i] !== -1) continue;
    c += 1;
    const order = [];
    const queue = [i];
    labels[i] = c;
    while (queue.length) {
      const p = queue.shift();
      order.push(p);
      for (const q of nb[p]) {
        if (labels[q] !== -1) continue;
        labels[q] = c;
        if (core[q]) {
          queue.push(q);
        } else {
          role[q] = 'border';
          order.push(q);
        }
      }
    }
    if (firstOrder === null) firstOrder = order;
  }
  return { labels, role, core, nClusters: c + 1, nNoise: labels.filter((l) => l === -1).length, firstOrder };
}

/* Sorted distance to the k-th nearest neighbour of every point: the curve
 * whose knee is the classic way to read eps off the data. */
export function kdist(points, k) {
  const n = points.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const ds = [];
    for (let j = 0; j < n; j++) {
      if (j !== i) ds.push(d2(points[i], points[j]));
    }
    ds.sort((a, b) => a - b);
    out[i] = Math.sqrt(ds[k - 1]);
  }
  return out.sort((a, b) => a - b);
}

/* Adjusted Rand index between two labelings, noise (-1) treated as its own
 * class: calling a real point noise is a disagreement and counts as one. */
export function ari(a, b) {
  const map = (v) => v + 2;   // shift so -1 becomes a valid index
  const ka = Math.max(...a) + 3;
  const kb = Math.max(...b) + 3;
  const M = Array.from({ length: ka }, () => new Array(kb).fill(0));
  for (let i = 0; i < a.length; i++) M[map(a[i])][map(b[i])] += 1;
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
