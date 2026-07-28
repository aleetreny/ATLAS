/* The two networks this page runs.
 *
 * One of them was trained by this article's generator. The other one is not
 * ours at all: it is the network the autoencoders article published, fetched
 * from that article's own data file and run here, because the comparison the
 * page opens with is only worth anything if the thing being sampled is
 * literally that model and not a reimplementation of it.
 */
import { decodeWeights, dense, tanh, sigmoid, byName, lcg } from '../../assets/js/halfweights.js';

export function makeVae(data) {
  const N = data.net;
  const w = decodeWeights(N.weights_b64, N.count);
  const L = byName(N.layers);
  return {
    k: L.emu.shape[1],
    encode(x) {
      const h = tanh(dense(w, L.enc1, x));
      const mu = dense(w, L.emu, h);
      const lv = dense(w, L.elv, h).map((v) => Math.min(4, Math.max(-9, v)));
      return { mu, lv };
    },
    decode(z) {
      return sigmoid(dense(w, L.dec2, tanh(dense(w, L.dec1, z))));
    },
  };
}

/* The previous article's network, from the previous article's file. Its layers
 * are positional there (encoder, code, decoder, pixels) rather than named. */
export async function loadAutoencoder(url = '../autoencoders/data/autoencoders.json') {
  const prev = await fetch(url).then((r) => r.json());
  const N = prev.net;
  const w = decodeWeights(N.weights_b64, N.count);
  const [e1, e2, d1, d2] = N.layers;
  return {
    count: N.count,
    encode(x) { return dense(w, e2, tanh(dense(w, e1, x))); },
    decode(z) { return sigmoid(dense(w, d2, tanh(dense(w, d1, z)))); },
  };
}

/* The gaussian a practitioner would fit to a cloud of codes, and draws from it.
 * Two dimensions, so the square root of the covariance is written out rather
 * than factorised by a library. */
export function fitGaussian(rows) {
  const n = rows.length;
  const d = rows[0].length;
  const m = new Array(d).fill(0);
  rows.forEach((r) => { for (let i = 0; i < d; i++) m[i] += r[i] / n; });
  const S = Array.from({ length: d }, () => new Array(d).fill(0));
  rows.forEach((r) => {
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) S[i][j] += ((r[i] - m[i]) * (r[j] - m[j])) / n;
    }
  });
  return { mean: m, cov: S };
}

export function gaussianDraws(fit, n, seed) {
  const rng = lcg(seed);
  const [a, b, , c] = [fit.cov[0][0], fit.cov[0][1], fit.cov[1][0], fit.cov[1][1]];
  const l11 = Math.sqrt(Math.max(a, 1e-12));
  const l21 = b / l11;
  const l22 = Math.sqrt(Math.max(c - l21 * l21, 1e-12));
  const out = [];
  for (let i = 0; i < n; i++) {
    const u = rng.normal();
    const v = rng.normal();
    out.push([fit.mean[0] + l11 * u, fit.mean[1] + l21 * u + l22 * v]);
  }
  return out;
}

export function unitDraws(k, n, seed) {
  const rng = lcg(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < k; j++) row.push(rng.normal());
    out.push(row);
  }
  return out;
}

/* float16 payloads that are not layers: the sample strips and the two density
 * pictures of the two dimensional section. */
export function unpack(b64, n) {
  return decodeWeights(b64, n);
}

export function nearest(point, rows) {
  let best = Infinity;
  for (let i = 0; i < rows.length; i++) {
    let s = 0;
    for (let j = 0; j < point.length; j++) s += (point[j] - rows[i][j]) ** 2;
    if (s < best) best = s;
  }
  return Math.sqrt(best);
}
