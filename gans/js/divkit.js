/* The game on the line, in closed form.
 *
 * Everything here is a trapezoid integral on the grid the generator declares
 * in game.json, written the same way it is written in
 * src/utils/generate_gan_data.py, so the numbers this page shows are the
 * numbers that file measured rather than a copy of them. The article checks
 * that at load: four (mean, sd) pairs travel in the JSON with the value of
 * every divergence at them, and this module has to reproduce all of them.
 */

export function makeGrid(spec) {
  const { x0, x1, n } = spec;
  const dx = (x1 - x0) / (n - 1);
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = x0 + i * dx;
  return { x, dx, n };
}

export function normalPdf(x, mu, sd) {
  const out = new Float64Array(x.length);
  const k = 1 / (sd * Math.sqrt(2 * Math.PI));
  for (let i = 0; i < x.length; i++) {
    const t = (x[i] - mu) / sd;
    out[i] = k * Math.exp(-0.5 * t * t);
  }
  return out;
}

export function mixturePdf(x, comps) {
  const out = new Float64Array(x.length);
  comps.forEach((c) => {
    const p = normalPdf(x, c.mu, c.sd);
    for (let i = 0; i < out.length; i++) out[i] += p[i] / comps.length;
  });
  return out;
}

export function trapz(y, dx) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += y[i];
  return (s - 0.5 * (y[0] + y[y.length - 1])) * dx;
}

function cumtrapz(y, dx) {
  const out = new Float64Array(y.length);
  for (let i = 1; i < y.length; i++) out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * dx;
  return out;
}

const TINY = 1e-300;
const safeLog = (v) => Math.log(v > TINY ? v : TINY);

/* Forward KL, reverse KL, Jensen-Shannon and Wasserstein-1 between two
 * densities sampled on the same grid. One function, one quadrature rule, so
 * the four are a choice of objective rather than four implementations. */
export function divergences(p, q, dx) {
  const n = p.length;
  const a = new Float64Array(n);
  const b = new Float64Array(n);
  const c = new Float64Array(n);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const m = 0.5 * (p[i] + q[i]);
    a[i] = p[i] > 0 ? p[i] * (safeLog(p[i]) - safeLog(q[i])) : 0;
    b[i] = q[i] > 0 ? q[i] * (safeLog(q[i]) - safeLog(p[i])) : 0;
    c[i] = p[i] > 0 ? p[i] * (safeLog(p[i]) - safeLog(m)) : 0;
    d[i] = q[i] > 0 ? q[i] * (safeLog(q[i]) - safeLog(m)) : 0;
  }
  const cp = cumtrapz(p, dx);
  const cq = cumtrapz(q, dx);
  const gap = new Float64Array(n);
  for (let i = 0; i < n; i++) gap[i] = Math.abs(cp[i] - cq[i]);
  return {
    fwd: trapz(a, dx),
    rev: trapz(b, dx),
    js: 0.5 * trapz(c, dx) + 0.5 * trapz(d, dx),
    w1: trapz(gap, dx),
  };
}

/* The optimal discriminator, which is not a network here but a ratio, and the
 * value of the game at it. */
export function optimalD(p, q) {
  const out = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) {
    const s = p[i] + q[i];
    out[i] = s > 0 ? p[i] / s : 0.5;
  }
  return out;
}

export function valueAtOptimum(p, q, dx) {
  const d = optimalD(p, q);
  const a = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) {
    a[i] = (p[i] > 0 ? p[i] * safeLog(d[i]) : 0) + (q[i] > 0 ? q[i] * safeLog(1 - d[i]) : 0);
  }
  return trapz(a, dx);
}

/* How well the best possible discriminator can tell the two apart: the
 * accuracy of the likelihood ratio test, which is what a perfectly trained
 * classifier converges to. */
export function bestAccuracy(p, q, dx) {
  const a = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) {
    a[i] = 0.5 * (p[i] > q[i] ? p[i] : 0) + 0.5 * (q[i] >= p[i] ? q[i] : 0);
  }
  return trapz(a, dx);
}

/* The gradient the generator receives, with the discriminator frozen at the
 * optimum, for both of the two losses that get called "the GAN loss".
 * d/dmu of an expectation over N(mu, sd) is an integral against
 * q(x) (x - mu) / sd^2, so no sampling is involved and no noise either. */
export function generatorGradients(p, q, x, mu, sd, dx) {
  const d = optimalD(p, q);
  const sat = new Float64Array(x.length);
  const ns = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const dq = q[i] * (x[i] - mu) / (sd * sd);
    sat[i] = safeLog(1 - d[i]) * dq;
    ns[i] = -safeLog(d[i]) * dq;
  }
  return { sat: trapz(sat, dx), ns: trapz(ns, dx) };
}
