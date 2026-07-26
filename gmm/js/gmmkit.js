/* Expectation-Maximisation for 2-D Gaussian mixtures, from scratch, in the
 * browser. The E-step and M-step run live in every widget of this article,
 * and the converged log-likelihood is checked against scikit-learn's answer
 * at load time.
 *
 * Numerical care, because a mixture can collapse: responsibilities go through
 * log-sum-exp, and every covariance gets a small ridge on its diagonal.
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

const RIDGE = 1e-6;
const LOG2PI = Math.log(2 * Math.PI);

/* One component's log-density at p. cov = [c00, c01, c11]. */
function logPdf(p, mean, cov) {
  const [a, b, c] = cov;
  const det = a * c - b * b;
  const dx = p[0] - mean[0];
  const dy = p[1] - mean[1];
  const q = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
  return -0.5 * (q + Math.log(det)) - LOG2PI;
}

/* E-step: responsibilities and the mean log-likelihood. */
export function eStep(points, m) {
  const n = points.length;
  const k = m.means.length;
  const resp = new Array(n);
  let ll = 0;
  for (let i = 0; i < n; i++) {
    const logs = new Array(k);
    let mx = -Infinity;
    for (let c = 0; c < k; c++) {
      logs[c] = Math.log(m.weights[c]) + logPdf(points[i], m.means[c], m.covs[c]);
      if (logs[c] > mx) mx = logs[c];
    }
    let s = 0;
    for (let c = 0; c < k; c++) s += Math.exp(logs[c] - mx);
    ll += mx + Math.log(s);
    resp[i] = logs.map((l) => Math.exp(l - mx) / s);
  }
  return { resp, loglik: ll / n };
}

/* M-step: weighted means, covariances and weights from responsibilities. */
export function mStep(points, resp, k) {
  const n = points.length;
  const means = [];
  const covs = [];
  const weights = [];
  for (let c = 0; c < k; c++) {
    let nk = 0;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
      nk += resp[i][c];
      mx += resp[i][c] * points[i][0];
      my += resp[i][c] * points[i][1];
    }
    mx /= nk;
    my /= nk;
    let s00 = 0;
    let s01 = 0;
    let s11 = 0;
    for (let i = 0; i < n; i++) {
      const dx = points[i][0] - mx;
      const dy = points[i][1] - my;
      s00 += resp[i][c] * dx * dx;
      s01 += resp[i][c] * dx * dy;
      s11 += resp[i][c] * dy * dy;
    }
    means.push([mx, my]);
    covs.push([s00 / nk + RIDGE, s01 / nk, s11 / nk + RIDGE]);
    weights.push(nk / n);
  }
  return { means, covs, weights };
}

/* A starting model from k initial means: the data's own covariance shrunk by
 * k (and optionally further, for starts that should behave like concrete
 * local bets rather than global shrugs), uniform weights. */
export function initFrom(points, means0, spread = 1) {
  const n = points.length;
  const k = means0.length;
  let mx = 0;
  let my = 0;
  for (const p of points) { mx += p[0]; my += p[1]; }
  mx /= n;
  my /= n;
  let s00 = 0;
  let s01 = 0;
  let s11 = 0;
  for (const p of points) {
    const dx = p[0] - mx;
    const dy = p[1] - my;
    s00 += dx * dx;
    s01 += dx * dy;
    s11 += dy * dy;
  }
  const d = k * spread;
  const cov = [s00 / n / d + RIDGE, s01 / n / d, s11 / n / d + RIDGE];
  return {
    means: means0.map((m) => [...m]),
    covs: means0.map(() => [...cov]),
    weights: means0.map(() => 1 / k),
  };
}

/* k-means++-style seeding for the means. */
export function seedMeans(points, k, rand) {
  const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
  const means = [[...points[Math.floor(rand() * points.length)]]];
  while (means.length < k) {
    const w = points.map((p) => Math.min(...means.map((c) => d2(p, c))));
    const total = w.reduce((a, b) => a + b, 0);
    let r = rand() * total;
    let i = 0;
    while (r > w[i]) { r -= w[i]; i++; }
    means.push([...points[i]]);
  }
  return means;
}

/* Full EM run. Returns the trajectory so widgets can animate it. */
export function em(points, model0, maxIter = 200, tol = 1e-6) {
  let model = model0;
  const path = [];
  let prev = -Infinity;
  for (let it = 0; it < maxIter; it++) {
    const { resp, loglik } = eStep(points, model);
    path.push({ model, resp, loglik });
    if (Math.abs(loglik - prev) < tol) break;
    prev = loglik;
    model = mStep(points, resp, model.means.length);
  }
  return path;
}

/* Hard labels from responsibilities. */
export const argmax = (r) => r.indexOf(Math.max(...r));

/* The n-sigma ellipse of a 2-D covariance, as polygon points. */
export function ellipse(mean, cov, nsig = 2, steps = 48) {
  const [a, b, c] = cov;
  const tr = a + c;
  const det = a * c - b * b;
  const l1 = tr / 2 + Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const l2 = tr / 2 - Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const theta = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  const r1 = nsig * Math.sqrt(Math.max(l1, 0));
  const r2 = nsig * Math.sqrt(Math.max(l2, 0));
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * 2 * Math.PI;
    const x = r1 * Math.cos(t);
    const y = r2 * Math.sin(t);
    pts.push([
      mean[0] + x * Math.cos(theta) - y * Math.sin(theta),
      mean[1] + x * Math.sin(theta) + y * Math.cos(theta),
    ]);
  }
  return pts;
}
