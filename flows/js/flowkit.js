/* The flow, run here, in both directions.
 *
 * A coupling layer is cheap enough that the page can do what no other article
 * in this module can: push a whole grid through the model at interactive rates,
 * invert it exactly, and read the stretching factor at a point. The weights are
 * the float16 the generator exported and the generator measured the page's own
 * log densities on them: quantising moves a log density by the number in the
 * readout and nothing more.
 */
import { decodeWeights, dense, tanh, lcg } from '../../assets/js/halfweights.js';

export function makeFlow(data) {
  const N = data.net;
  const w = decodeWeights(N.weights_b64, N.count);
  const spec = N.layers;
  const parity = N.parity;
  const sMax = N.s_max;
  const depth = parity.length;

  /* the scale and shift a layer applies, given the coordinate that passes through */
  function st(i, xa) {
    const h1 = tanh(dense(w, spec[3 * i], [xa]));
    const h2 = tanh(dense(w, spec[3 * i + 1], h1));
    const o = dense(w, spec[3 * i + 2], h2);
    return [Math.tanh(o[0]) * sMax, o[1]];
  }

  function forwardTo(p, upto) {
    let x = [p[0], p[1]];
    let ld = 0;
    for (let i = 0; i < Math.min(upto, depth); i++) {
      const a = parity[i] === 0 ? 0 : 1;
      const b = 1 - a;
      const [s, t] = st(i, x[a]);
      const y = [0, 0];
      y[a] = x[a];
      y[b] = x[b] * Math.exp(s) + t;
      x = y;
      ld += s;
    }
    return { z: x, logdet: ld };
  }

  function inverseFrom(z, from) {
    let y = [z[0], z[1]];
    for (let i = Math.min(from, depth) - 1; i >= 0; i--) {
      const a = parity[i] === 0 ? 0 : 1;
      const b = 1 - a;
      const [s, t] = st(i, y[a]);
      const x = [0, 0];
      x[a] = y[a];
      x[b] = (y[b] - t) * Math.exp(-s);
      y = x;
    }
    return y;
  }

  return {
    depth,
    forward: (p) => forwardTo(p, depth),
    forwardTo,
    inverse: (z) => inverseFrom(z, depth),
    inverseFrom,
    logProb(p) {
      const { z, logdet } = forwardTo(p, depth);
      return -0.5 * (z[0] * z[0] + z[1] * z[1]) - Math.log(2 * Math.PI) + logdet;
    },
    /* |det J| by the analytic formula, and by a central difference, so the page
       can show that the two agree the way the generator says they do */
    detAt(p, h = 1e-4) {
      const { logdet } = forwardTo(p, depth);
      const cols = [];
      for (let j = 0; j < 2; j++) {
        const up = [p[0], p[1]];
        const dn = [p[0], p[1]];
        up[j] += h;
        dn[j] -= h;
        const a = forwardTo(up, depth).z;
        const b = forwardTo(dn, depth).z;
        cols.push([(a[0] - b[0]) / (2 * h), (a[1] - b[1]) / (2 * h)]);
      }
      const det = cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0];
      return { analytic: Math.exp(logdet), numeric: Math.abs(det) };
    },
    sample(n, seed, temperature = 1) {
      const rng = lcg(seed);
      const out = [];
      for (let i = 0; i < n; i++) {
        out.push(inverseFrom([rng.normal() * temperature, rng.normal() * temperature], depth));
      }
      return out;
    },
  };
}

export function unpack(b64, n) {
  return decodeWeights(b64, n);
}
