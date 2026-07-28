/* The trained energy, and the sampler that walks on it, both run here.
 *
 * The energy is a two layer softplus network plus an anchor term that keeps the
 * density integrable, exactly as the generator wrote it, and the gradient is
 * differentiated by hand rather than approximated: a Langevin step needs it
 * thousands of times a second and a central difference would cost four forward
 * passes for the same thing.
 */
import { decodeWeights, dense, byName, lcg } from '../../assets/js/halfweights.js';

const softplus = (t) => (t > 20 ? t : Math.log1p(Math.exp(t)));
const sigmoid = (t) => 1 / (1 + Math.exp(-t));

export function makeEnergy(data, arm = 'exact') {
  const M = data.model;
  const A = M.arms[arm];
  const w = decodeWeights(A.weights_b64, A.count);
  const L = byName(A.layers);
  const tau2 = M.tau * M.tau;
  const [n1] = [L.l1.shape[1]];
  const [n2] = [L.l2.shape[1]];

  function forward(p) {
    const z1 = dense(w, L.l1, p);
    const h1 = new Float64Array(n1);
    const a1 = new Float64Array(n1);
    for (let i = 0; i < n1; i++) { h1[i] = softplus(z1[i]); a1[i] = sigmoid(z1[i]); }
    const z2 = dense(w, L.l2, h1);
    const h2 = new Float64Array(n2);
    const a2 = new Float64Array(n2);
    for (let i = 0; i < n2; i++) { h2[i] = softplus(z2[i]); a2[i] = sigmoid(z2[i]); }
    const out = dense(w, L.l3, h2)[0];
    const anchor = (p[0] * p[0] + p[1] * p[1]) / (2 * tau2);
    return { energy: out + anchor, a1, a2 };
  }

  /* the derivative of the energy, written out, because the sampler needs it
     once per walker per step */
  function grad(p) {
    const { a1, a2 } = forward(p);
    const d2 = new Float64Array(n2);
    for (let o = 0; o < n2; o++) d2[o] = a2[o] * w[L.l3.w_offset + o];
    const d1 = new Float64Array(n1);
    for (let i = 0; i < n1; i++) {
      let s = 0;
      for (let o = 0; o < n2; o++) s += w[L.l2.w_offset + o * n1 + i] * d2[o];
      d1[i] = a1[i] * s;
    }
    const g = [p[0] / tau2, p[1] / tau2];
    for (let i = 0; i < 2; i++) {
      let s = 0;
      for (let o = 0; o < n1; o++) s += w[L.l1.w_offset + o * 2 + i] * d1[o];
      g[i] += s;
    }
    return g;
  }

  return {
    energy: (p) => forward(p).energy,
    grad,
    logz: A.logz_train_grid,
    logProb: (p) => -forward(p).energy - A.logz_train_grid,
  };
}

/* One step of unadjusted Langevin, or of the Metropolis adjusted version, or of
 * plain descent with the noise switched off, on the same walkers. */
export function makeSampler(E, { n = 600, seed = 4242, half = 4.2 } = {}) {
  const rng = lcg(seed);
  const pts = [];
  for (let i = 0; i < n; i++) {
    pts.push([(rng.uniform() * 2 - 1) * half, (rng.uniform() * 2 - 1) * half]);
  }
  let accepted = 0;
  let proposed = 0;

  const logq = (from, to, eps) => {
    const g = E.grad(from);
    const mx = from[0] - eps * g[0];
    const my = from[1] - eps * g[1];
    return -((to[0] - mx) ** 2 + (to[1] - my) ** 2) / (4 * eps);
  };

  return {
    points: pts,
    get acceptance() { return proposed ? accepted / proposed : 1; },
    reset(s) {
      const r = lcg(s);
      for (let i = 0; i < pts.length; i++) {
        pts[i][0] = (r.uniform() * 2 - 1) * half;
        pts[i][1] = (r.uniform() * 2 - 1) * half;
      }
      accepted = 0;
      proposed = 0;
    },
    step(eps, mode) {
      const sd = Math.sqrt(2 * eps);
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const g = E.grad(p);
        const noise = mode === 'nonoise' ? [0, 0] : [rng.normal() * sd, rng.normal() * sd];
        const q = [p[0] - eps * g[0] + noise[0], p[1] - eps * g[1] + noise[1]];
        if (mode === 'mala') {
          proposed += 1;
          const logA = (E.energy(p) - E.energy(q)) + logq(q, p, eps) - logq(p, q, eps);
          if (Math.log(Math.max(rng.uniform(), 1e-12)) < logA) {
            accepted += 1;
            p[0] = q[0];
            p[1] = q[1];
          }
        } else {
          p[0] = Math.max(-half * 1.6, Math.min(half * 1.6, q[0]));
          p[1] = Math.max(-half * 1.6, Math.min(half * 1.6, q[1]));
        }
      }
    },
  };
}

export function unpack(b64, n) {
  return decodeWeights(b64, n);
}
