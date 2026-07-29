/* The two indices, and the arithmetic behind the bound, without any DOM.
 *
 * These are recomputed in the page from the counts the generator recorded
 * rather than read out of the file, which is what makes the pictures of the
 * bound and of the posterior claims rather than decoration.
 */

/* The upper confidence bound: the estimate, plus a term that shrinks with the
 * pulls given to this arm and grows with the pulls given to everything else. */
export const ucbIndex = (mean, n, t, c = 2) =>
  mean + Math.sqrt((c * Math.log(t)) / Math.max(n, 1));

/* log of the beta function, from a Lanczos approximation of log gamma: enough
 * to draw a posterior and to normalise it, which is all the page asks. */
function logGamma(z) {
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  const x = z - 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < g.length; i++) a += g[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export const logBeta = (a, b) => logGamma(a) + logGamma(b) - logGamma(a + b);

export function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta(a, b));
}

/* The divergence between two coins and the constant it produces: the floor the
 * regret slopes in this article are measured against. */
export function kl(p, q) {
  const e = 1e-12;
  const a = Math.min(Math.max(p, e), 1 - e);
  const bq = Math.min(Math.max(q, e), 1 - e);
  return a * Math.log(a / bq) + (1 - a) * Math.log((1 - a) / (1 - bq));
}

export function boundConstant(arms) {
  const best = Math.max(...arms);
  let total = 0;
  arms.forEach((a) => { if (a < best) total += (best - a) / kl(a, best); });
  return total;
}
