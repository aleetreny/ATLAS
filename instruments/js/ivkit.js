/* The three quantities this article is made of, written once.
 *
 * The same three functions serve the widgets and the guard, so that when the
 * page prints a number under a cloud and the guard compares it with the
 * generator's, the two sides of that comparison are not two different
 * implementations of the same idea.
 */

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

/* Least squares slope of y on x. */
export function ols(x, y) {
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let den = 0;
  for (let i = 0; i < x.length; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  return num / den;
}

/* The ratio: how much the outcome moved with the lever, over how much the
 * treatment moved with it. Written as two covariances rather than as two
 * chained regressions because it is the same number and this way the reader
 * can see where the trouble lives: the denominator. */
export function wald(z, x, y) {
  const mz = mean(z);
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let den = 0;
  for (let i = 0; i < z.length; i++) {
    num += (z[i] - mz) * (y[i] - my);
    den += (z[i] - mz) * (x[i] - mx);
  }
  return num / den;
}

/* First stage F with a single instrument, which is the square of the t on the
 * lever. It is the one diagnostic that gets reported for an instrument, so the
 * page computes it rather than quoting it. */
export function firstStageF(z, x) {
  const n = z.length;
  const mz = mean(z);
  const mx = mean(x);
  let szz = 0;
  let szx = 0;
  for (let i = 0; i < n; i++) {
    szz += (z[i] - mz) ** 2;
    szx += (z[i] - mz) * (x[i] - mx);
  }
  const b = szx / szz;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += ((x[i] - mx) - b * (z[i] - mz)) ** 2;
  return (b * b * szz) / (ss / (n - 2));
}

/* The fitted line through a cloud, as two endpoints. */
export function fitLine(x, y) {
  const b = ols(x, y);
  const mx = mean(x);
  const my = mean(y);
  const lo = Math.min(...x);
  const hi = Math.max(...x);
  return [[lo, my + b * (lo - mx)], [hi, my + b * (hi - mx)]];
}
