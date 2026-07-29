/* The whole method, in two functions.
 *
 * That is not a joke about brevity: the reason this article can have a guard
 * that demands exact agreement is that the procedure is a rank and a lookup,
 * with no fitting and no optimisation anywhere. Both the widgets and the guard
 * call these, so a disagreement between the page and its export cannot be two
 * different implementations of the same idea.
 */

/* The rank to take: ceil((n+1)(1-alpha)).
 *
 * Written with a rounding guard because ceil of a float that should be an
 * integer is a classic way to be off by one: at n = 19 and alpha = 0.1,
 * (n+1)(1-alpha) is 17.999999999999996 in binary floating point, and the naive
 * ceil returns 18 while the arithmetic says 18 as well by luck. The nudge
 * makes it not luck.
 */
export function rankFor(n, alpha) {
  const exact = (n + 1) * (1 - alpha);
  const rounded = Math.round(exact);
  return Math.abs(exact - rounded) < 1e-9 ? rounded : Math.ceil(exact);
}

/* The interval half-width for a level, from a sorted array of calibration
 * scores. Infinity when the rank exceeds what the calibration set can supply,
 * which is the honest answer and not an error: with 20 points there is no
 * ninety-nine percent interval to be had. */
export function quantileAt(sortedScores, alpha) {
  const n = sortedScores.length;
  const k = rankFor(n, alpha);
  return k <= n ? sortedScores[k - 1] : Infinity;
}

/* The density of a Beta distribution, for drawing the spread of coverage
 * across calibration sets. Written with log gamma so the normalising constant
 * survives a with a = 1801: the naive product overflows well before that. */
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

export function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  const logB = logGamma(a) + logGamma(b) - logGamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}
