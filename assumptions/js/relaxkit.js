/* The four algorithms of this article, written out in the browser, plus the
 * two things they are judged with (k-means, for the comparison, and the
 * adjusted Rand index). Every one of them is checked at load time against the
 * reference answers in data/relaxations.json: PAM against the Rust FasterPAM,
 * k-modes against the kmodes package, OPTICS and mean shift against
 * scikit-learn.
 *
 * They are written to match those references exactly rather than to be
 * elegant, because a page that says "this is what the algorithm does" and then
 * does something slightly different is worse than no page at all. Where a
 * detail exists only to reproduce a library's tie-breaking, it says so.
 */

/* ------------------------------------------------------------------ distance */

export const euclid = (p, q) => Math.hypot(...p.map((v, i) => v - q[i]));

/* Full pairwise distance matrix under one of the three metrics the wine
 * section offers. Medoids never need coordinates, only distances, which is
 * exactly why the metric is free to be anything at all. */
export function distanceMatrix(X, metric = 'euclidean') {
  const n = X.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  const norms = metric === 'cosine' ? X.map((r) => Math.hypot(...r)) : null;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d = 0;
      if (metric === 'manhattan') {
        for (let c = 0; c < X[i].length; c++) d += Math.abs(X[i][c] - X[j][c]);
      } else if (metric === 'cosine') {
        let dot = 0;
        for (let c = 0; c < X[i].length; c++) dot += X[i][c] * X[j][c];
        d = 1 - dot / (norms[i] * norms[j]);
      } else {
        for (let c = 0; c < X[i].length; c++) d += (X[i][c] - X[j][c]) ** 2;
        d = Math.sqrt(d);
      }
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
}

/* ------------------------------------------------------- k-medoids, classic PAM
 * Kaufman and Rousseeuw's two phases: BUILD greedily collects k medoids, then
 * SWAP repeatedly applies the single (medoid, non-medoid) exchange that lowers
 * the total distance most, until none does. No randomness anywhere, so the
 * page and the generator cannot disagree by luck. */

function lossOf(D, med) {
  let s = 0;
  for (let i = 0; i < D.length; i++) {
    let best = Infinity;
    for (const m of med) if (D[m][i] < best) best = D[m][i];
    s += best;
  }
  return s;
}

export function pam(D, k, maxIter = 100, { trace = false } = {}) {
  const n = D.length;
  const steps = trace ? [] : null;
  /* BUILD: the most central point, then whichever point buys the largest total
   * reduction in distance given the medoids chosen so far. */
  let bestTotal = Infinity;
  let first = 0;
  for (let c = 0; c < n; c++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += D[c][i];
    if (s < bestTotal) {
      bestTotal = s;
      first = c;
    }
  }
  const med = [first];
  const isMed = new Uint8Array(n);
  isMed[first] = 1;
  if (steps) steps.push({ phase: 'build', medoids: [first], loss: lossOf(D, med), added: first });
  while (med.length < k) {
    const near = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let b = Infinity;
      for (const m of med) if (D[m][i] < b) b = D[m][i];
      near[i] = b;
    }
    let bestGain = -1;
    let pick = 0;
    for (let c = 0; c < n; c++) {
      if (isMed[c]) continue;
      let g = 0;
      for (let i = 0; i < n; i++) {
        const diff = near[i] - D[c][i];
        if (diff > 0) g += diff;
      }
      if (g > bestGain) {
        bestGain = g;
        pick = c;
      }
    }
    med.push(pick);
    isMed[pick] = 1;
    if (steps) steps.push({ phase: 'build', medoids: med.slice(), loss: lossOf(D, med), added: pick });
  }

  /* SWAP */
  for (let it = 0; it < maxIter; it++) {
    const cost = lossOf(D, med);
    let best = null;
    for (let i = 0; i < k; i++) {
      const others = med.filter((_, j) => j !== i);
      const dOther = new Float64Array(n);
      for (let p = 0; p < n; p++) {
        let b = Infinity;
        for (const m of others) if (D[m][p] < b) b = D[m][p];
        dOther[p] = b;
      }
      for (let h = 0; h < n; h++) {
        if (isMed[h]) continue;
        let s = 0;
        for (let p = 0; p < n; p++) s += Math.min(dOther[p], D[h][p]);
        if (best === null || s < best.cost) best = { cost: s, i, h };
      }
    }
    if (!best || best.cost >= cost - 1e-12) {
      if (steps) steps.push({ phase: 'done', medoids: med.slice(), loss: cost, rejected: best });
      break;
    }
    const leaving = med[best.i];
    isMed[leaving] = 0;
    med[best.i] = best.h;
    isMed[best.h] = 1;
    if (steps) steps.push({ phase: 'swap', medoids: med.slice(), loss: best.cost, from: leaving, to: best.h, was: cost });
  }

  const labels = new Array(n);
  for (let i = 0; i < n; i++) {
    let b = Infinity;
    let a = 0;
    for (let c = 0; c < k; c++) {
      if (D[med[c]][i] < b) {
        b = D[med[c]][i];
        a = c;
      }
    }
    labels[i] = a;
  }
  return { medoids: med.slice(), labels, loss: lossOf(D, med), steps };
}

/* --------------------------------------------------------------------- k-modes
 * Hamming distance, and a centre that is the most common category in each
 * column. Cao's initialisation: the densest row first, then the row maximising
 * density times distance to the nearest mode so far. Both the initialisation
 * and the update depend on the categories only through how often they occur
 * and whether two of them are equal, never through the numbers they were
 * given, which is the entire point of the section. */

const hamming = (a, b) => {
  let d = 0;
  for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) d++;
  return d;
};

export function kmodes(A, k, maxIter = 100) {
  const n = A.length;
  const m = A[0].length;
  const freq = [];
  for (let j = 0; j < m; j++) {
    const f = new Map();
    for (let i = 0; i < n; i++) f.set(A[i][j], (f.get(A[i][j]) || 0) + 1);
    freq.push(f);
  }
  const dens = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += freq[j].get(A[i][j]) / n;
    dens[i] = s / m;
  }
  let start = 0;
  for (let i = 1; i < n; i++) if (dens[i] > dens[start]) start = i;
  const modes = [A[start].slice()];
  while (modes.length < k) {
    let bestScore = -1;
    let pick = 0;
    for (let i = 0; i < n; i++) {
      let d = Infinity;
      for (const mo of modes) d = Math.min(d, hamming(A[i], mo));
      const s = d * dens[i];
      if (s > bestScore) {
        bestScore = s;
        pick = i;
      }
    }
    modes.push(A[pick].slice());
  }

  let labels = new Array(n).fill(-1);
  for (let it = 0; it < maxIter; it++) {
    const next = new Array(n);
    for (let i = 0; i < n; i++) {
      let b = Infinity;
      let a = 0;
      for (let c = 0; c < k; c++) {
        const d = hamming(A[i], modes[c]);
        if (d < b) {
          b = d;
          a = c;
        }
      }
      next[i] = a;
    }
    if (next.every((v, i) => v === labels[i])) break;
    labels = next;
    for (let c = 0; c < k; c++) {
      const rows = [];
      for (let i = 0; i < n; i++) if (labels[i] === c) rows.push(A[i]);
      if (!rows.length) continue;
      for (let j = 0; j < m; j++) {
        const count = new Map();
        for (const r of rows) count.set(r[j], (count.get(r[j]) || 0) + 1);
        let top = 0;
        count.forEach((v) => { if (v > top) top = v; });
        /* A tie goes to whichever of the tied values turns up first inside the
         * cluster. Breaking it by the smallest category number instead would
         * make the answer depend on the numbering, which is the one thing
         * k-modes is here to avoid. */
        for (const r of rows) {
          if (count.get(r[j]) === top) {
            modes[c][j] = r[j];
            break;
          }
        }
      }
    }
  }
  let cost = 0;
  for (let i = 0; i < n; i++) cost += hamming(A[i], modes[labels[i]]);
  return { modes, labels, cost };
}

/* ---------------------------------------------------------------------- OPTICS
 * The ordering scikit-learn produces, step for step: take the unprocessed
 * point with the smallest reachability (ties to the lowest index), then relax
 * its unprocessed neighbours against its core distance. What comes out is not
 * a clustering but a walk through the data and, for each step, the density at
 * which that point became reachable. */

export function optics(points, minSamples) {
  const n = points.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = euclid(points[i], points[j]);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  const core = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = Array.from(D[i]).sort((a, b) => a - b);
    core[i] = row[minSamples - 1];       // its own zero is included, as in sklearn
  }
  const reach = new Float64Array(n).fill(Infinity);
  const done = new Uint8Array(n);
  const order = new Array(n);
  for (let t = 0; t < n; t++) {
    let p = -1;
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      if (!done[i] && reach[i] < best) {
        best = reach[i];
        p = i;
      }
    }
    if (p === -1) for (let i = 0; i < n; i++) if (!done[i]) { p = i; break; }
    done[p] = 1;
    order[t] = p;
    for (let i = 0; i < n; i++) {
      if (done[i]) continue;
      const rd = Math.max(D[p][i], core[p]);
      if (rd < reach[i]) reach[i] = rd;
    }
  }
  return { order, reach: Array.from(reach), core: Array.from(core) };
}

/* A horizontal line across the reachability plot IS a DBSCAN radius: a new
 * cluster starts wherever the curve rises above the line at a point that is
 * still dense enough to be a core point, and points that are both above the
 * line and not dense enough are noise. (This is scikit-learn's
 * cluster_optics_dbscan, and it is the reason the plot is worth drawing.) */
export function extractDbscan(order, reach, core, eps) {
  const labels = new Array(reach.length).fill(0);
  let c = -1;
  for (const i of order) {
    const far = reach[i] > eps;
    if (far && core[i] <= eps) c += 1;
    labels[i] = far && core[i] > eps ? -1 : c;
  }
  return labels.map((v) => (v < 0 ? -1 : v));
}

/* ------------------------------------------------------------------ mean shift
 * Flat kernel, every point a seed, following scikit-learn including the two
 * details that decide close calls: the intensity ranking a peak is the size of
 * the last window measured (the one around the previous position), and equal
 * intensities are broken by the coordinates, largest first. */

export function meanShift(points, bw, { trails = 0, maxIter = 300 } = {}) {
  const stop = 1e-3 * bw;
  const peaks = new Map();
  const paths = [];
  for (let s = 0; s < points.length; s++) {
    let p = points[s].slice();
    const path = trails ? [p.slice()] : null;
    let inside = 0;
    let it = 0;
    for (;;) {
      let sx = 0;
      let sy = 0;
      inside = 0;
      for (let i = 0; i < points.length; i++) {
        if (euclid(points[i], p) <= bw) {
          sx += points[i][0];
          sy += points[i][1];
          inside++;
        }
      }
      if (!inside) break;
      const old = p;
      p = [sx / inside, sy / inside];
      if (path && path.length < 60) path.push(p.slice());
      if (Math.hypot(p[0] - old[0], p[1] - old[1]) <= stop || it === maxIter) break;
      it++;
    }
    if (inside) peaks.set(`${p[0]},${p[1]}`, { p, n: inside });
    if (path && s % Math.max(1, Math.round(points.length / trails)) === 0) paths.push(path);
  }
  const ranked = [...peaks.values()].sort((a, b) => (b.n - a.n) || (b.p[0] - a.p[0]) || (b.p[1] - a.p[1]));
  const centers = [];
  for (const { p } of ranked) {
    if (centers.every((q) => euclid(p, q) > bw)) centers.push(p);
  }
  const labels = points.map((pt) => {
    let b = Infinity;
    let a = 0;
    centers.forEach((c, i) => {
      const d = euclid(pt, c);
      if (d < b) {
        b = d;
        a = i;
      }
    });
    return a;
  });
  return { centers, labels, paths };
}

/* ---------------------------------------------------------------- for comparison
 * k-means, seeded so that a reader who moves a slider back and forth gets the
 * same answer twice: k-means++ starts, the best of several restarts kept. */

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function kmeans(points, k, { restarts = 10, seed = 19, maxIter = 100 } = {}) {
  const rand = rng(seed);
  const dim = points[0].length;
  const d2 = (p, q) => {
    let s = 0;
    for (let c = 0; c < dim; c++) s += (p[c] - q[c]) ** 2;
    return s;
  };
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const centers = [points[Math.floor(rand() * points.length)].slice()];
    while (centers.length < k) {
      const w = points.map((p) => Math.min(...centers.map((c) => d2(p, c))));
      const total = w.reduce((a, b) => a + b, 0);
      let u = rand() * total;
      let i = 0;
      while (u > w[i] && i < w.length - 1) {
        u -= w[i];
        i++;
      }
      centers.push(points[i].slice());
    }
    let labels = new Array(points.length).fill(-1);
    let inertia = Infinity;
    for (let it = 0; it < maxIter; it++) {
      let J = 0;
      const next = points.map((p) => {
        let b = Infinity;
        let a = 0;
        centers.forEach((c, i) => {
          const d = d2(p, c);
          if (d < b) {
            b = d;
            a = i;
          }
        });
        J += b;
        return a;
      });
      const stable = next.every((v, i) => v === labels[i]);
      labels = next;
      inertia = J;
      const sums = centers.map(() => new Float64Array(dim + 1));
      points.forEach((p, i) => {
        const s = sums[labels[i]];
        for (let c = 0; c < dim; c++) s[c] += p[c];
        s[dim] += 1;
      });
      sums.forEach((s, i) => {
        if (s[dim]) centers[i] = Array.from({ length: dim }, (_, c) => s[c] / s[dim]);
      });
      if (stable) break;
    }
    if (!best || inertia < best.inertia) best = { centers: centers.map((c) => c.slice()), labels, inertia };
  }
  return best;
}

/* ------------------------------------------------------------------------ ARI
 * Noise (-1) counts as a class of its own: calling a real point noise is a
 * disagreement and is scored as one. Careful with what that rewards, though:
 * on a dataset with a diffuse cluster, declaring the whole cluster noise can
 * score 1.000, which is why this article measures the trap with two other
 * numbers instead. */
export function ari(a, b) {
  const map = (v) => v + 2;
  const ka = Math.max(...a) + 3;
  const kb = Math.max(...b) + 3;
  const M = Array.from({ length: ka }, () => new Array(kb).fill(0));
  for (let i = 0; i < a.length; i++) M[map(a[i])][map(b[i])] += 1;
  const comb2 = (x) => (x * (x - 1)) / 2;
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
  const expected = (sumA * sumB) / total;
  const max = (sumA + sumB) / 2;
  return max === expected ? 1 : (sumIJ - expected) / (max - expected);
}
