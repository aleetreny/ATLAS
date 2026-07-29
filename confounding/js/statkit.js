/* The four pieces of arithmetic this article needs, written once.
 *
 * They live here rather than inside a widget because the guard in main.js uses
 * the same ones to check the page against its own export: two implementations
 * of a slope would make a disagreement ambiguous, which is the trap the wine
 * article fell into with two different Theil-Sen estimators under one name.
 */

/* Least squares slope of y on x. */
export function slope(x, y) {
  const n = x.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  return den > 0 ? num / den : 0;
}

/* Residuals of y after removing the least squares fit on z. */
export function residuals(z, y) {
  const b = slope(z, y);
  const mz = z.reduce((a, v) => a + v, 0) / z.length;
  const my = y.reduce((a, v) => a + v, 0) / y.length;
  return y.map((v, i) => v - (my + b * (z[i] - mz)));
}

/* Coefficient on x in the regression of y on (x, z), with an intercept.
 *
 * Computed by partialling z out of both sides rather than by solving normal
 * equations, for two reasons. The cheap one is that this atlas has a standing
 * rule against forming X'X. The better one is that partialling out IS what
 * this article means by adjusting: take the part of the treatment that the
 * confounder does not explain, take the part of the outcome it does not
 * explain, and relate what is left.
 */
export function ols2(x, z, y) {
  return slope(residuals(z, x), residuals(z, y));
}

/* The site's linear congruential generator plus Box-Muller, so the browser can
 * draw its own sample from the same world the generator wrote down. Written
 * identically on both sides everywhere in this atlas; numpy's generator does
 * not reproduce in a browser and a sample nobody can reproduce is a drawing.
 */
export function rng(seed) {
  let s = seed >>> 0;
  const unit = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  let spare = null;
  return {
    unit,
    normal() {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0;
      let v = 0;
      while (u === 0) u = unit();
      while (v === 0) v = unit();
      const r = Math.sqrt(-2 * Math.log(u));
      const t = 2 * Math.PI * v;
      spare = r * Math.sin(t);
      return r * Math.cos(t);
    },
  };
}

/* One draw from the world the article writes out, at whatever confounding
 * strength the reader has asked for. */
export function draw(n, alpha, meta, seed) {
  const g = rng(seed);
  const u = new Array(n);
  const x = new Array(n);
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    u[i] = g.normal();
    x[i] = alpha * u[i] + g.normal() > 0 ? 1 : 0;
    y[i] = meta.base + meta.beta * x[i] + meta.gamma * u[i] + meta.noise * g.normal();
  }
  return { u, x, y };
}

/* The closed form the whole first section is about. */
export function millsGap(alpha) {
  return (2 * alpha) / Math.sqrt(1 + alpha * alpha) * Math.sqrt(2 / Math.PI);
}
