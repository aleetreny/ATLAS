/* K-means, from scratch, in the browser.
 *
 * This is the article's actual subject, so it runs here rather than being
 * precomputed: Lloyd's loop, k-means++ seeding, inertia and the silhouette.
 * Everything is checked at load time against the scikit-learn reference values
 * shipped in data/clusters.json.
 *
 * Randomness is seeded (mulberry32) so a reader pressing "new random start"
 * gets variety while the article's own described runs are reproducible.
 */

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

const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;

/* Nearest-centre assignment. Returns labels and the resulting inertia. */
export function assign(points, centers) {
  const labels = new Array(points.length);
  let J = 0;
  for (let i = 0; i < points.length; i++) {
    let best = 0;
    let bd = Infinity;
    for (let c = 0; c < centers.length; c++) {
      const d = d2(points[i], centers[c]);
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    labels[i] = best;
    J += bd;
  }
  return { labels, inertia: J };
}

/* Mean of each cluster. An emptied centre stays where it was rather than
 * jumping to NaN, which matches what a careful implementation does and keeps a
 * degenerate random start recoverable. */
export function update(points, labels, k) {
  const sums = Array.from({ length: k }, () => [0, 0, 0]);
  for (let i = 0; i < points.length; i++) {
    const s = sums[labels[i]];
    s[0] += points[i][0];
    s[1] += points[i][1];
    s[2] += 1;
  }
  return sums.map((s) => (s[2] ? [s[0] / s[2], s[1] / s[2]] : null)); // caller substitutes previous centre for null
}

/* Classic random init: k distinct data points. */
export function initRandom(points, k, rand) {
  const idx = new Set();
  while (idx.size < k) idx.add(Math.floor(rand() * points.length));
  return [...idx].map((i) => [...points[i]]);
}

/* k-means++: each next centre drawn with probability proportional to squared
 * distance from the nearest centre already chosen. Returns the centres AND the
 * D^2 weights of each round, so the article can draw the choosing itself. */
export function initPP(points, k, rand, trace = null) {
  const centers = [[...points[Math.floor(rand() * points.length)]]];
  while (centers.length < k) {
    const w = points.map((p) => Math.min(...centers.map((c) => d2(p, c))));
    if (trace) trace.push({ centers: centers.map((c) => [...c]), weights: [...w] });
    const total = w.reduce((a, b) => a + b, 0);
    let r = rand() * total;
    let i = 0;
    while (r > w[i]) {
      r -= w[i];
      i++;
    }
    centers.push([...points[i]]);
  }
  if (trace) trace.push({ centers: centers.map((c) => [...c]), weights: null });
  return centers;
}

/* Full Lloyd run. Returns the trajectory so a caller can animate it. */
export function lloyd(points, centers0, maxIter = 60) {
  let centers = centers0.map((c) => [...c]);
  const path = [];
  let prev = Infinity;
  for (let it = 0; it < maxIter; it++) {
    const { labels, inertia } = assign(points, centers);
    path.push({ centers: centers.map((c) => [...c]), labels, inertia });
    const means = update(points, labels, centers.length);
    centers = means.map((m, c) => (m === null ? centers[c] : m));
    if (Math.abs(prev - inertia) < 1e-9) break;
    prev = inertia;
  }
  const { labels, inertia } = assign(points, centers);
  path.push({ centers, labels, inertia });
  return path;
}

/* Mean silhouette over all points: (b - a) / max(a, b), where a is the mean
 * distance to my own cluster and b the mean distance to the nearest other.
 * O(n^2), fine at the few hundred points this article uses. */
export function silhouette(points, labels, k) {
  const n = points.length;
  if (k < 2) return 0;
  const byC = Array.from({ length: k }, () => []);
  points.forEach((p, i) => byC[labels[i]].push(i));
  let total = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const own = byC[labels[i]];
    if (own.length <= 1) {
      counted++; // a singleton contributes s(i) = 0 to the mean, per Rousseeuw
      continue;
    }
    let a = 0;
    for (const j of own) if (j !== i) a += Math.sqrt(d2(points[i], points[j]));
    a /= own.length - 1;
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === labels[i] || !byC[c].length) continue;
      let s = 0;
      for (const j of byC[c]) s += Math.sqrt(d2(points[i], points[j]));
      b = Math.min(b, s / byC[c].length);
    }
    total += (b - a) / Math.max(a, b);
    counted++;
  }
  return counted ? total / counted : 0;
}
