/* The whole pipeline, run in the page: sample in the latent, then decode.
 *
 * Three networks travel in one payload and they do different jobs. The score
 * network is tiny because its input is eight numbers rather than a hundred and
 * ninety six, which is the entire argument of the article. The decoder is the
 * thing that turns those eight numbers back into a picture, and it is also the
 * thing that fixes the ceiling: whatever the sampler produces, the reader is
 * looking at the decoder's output. The encoder rides along so the page can show
 * the round trip that measures that ceiling, on real digits, live.
 */
import { byName, decodeWeights, dense, lcg } from '../../assets/js/halfweights.js';

export function makeLatent(data) {
  const N = data.net;
  const W = decodeWeights(N.b64, N.count);
  const L = byName(N.spec);
  const K = N.k;
  const abar = N.abar;
  const steps = N.steps;

  const silu = (v) => v.map((t) => t / (1 + Math.exp(-t)));
  const tanh = (v) => v.map(Math.tanh);
  const sigmoid = (v) => v.map((t) => 1 / (1 + Math.exp(-t)));

  /* how many dense layers the score network has, read off the spec rather than
     assumed, so a generator that changes its depth does not silently break */
  const diffLayers = N.spec.filter((s) => s.name.startsWith('d') && !s.name.startsWith('dec'))
    .map((s) => s.name);
  const decLayers = N.spec.filter((s) => s.name.startsWith('dec')).map((s) => s.name);
  const encLayers = N.spec.filter((s) => s.name.startsWith('enc')).map((s) => s.name);

  function embed(t) {
    const out = new Float64Array(32);
    for (let i = 0; i < 16; i++) {
      const f = Math.exp((Math.log(200) * i) / 15);
      const a = (t / steps) * f * Math.PI;
      out[i] = Math.sin(a);
      out[i + 16] = Math.cos(a);
    }
    return out;
  }

  /** eps_theta(z, t) for one latent point. */
  function epsOne(z, t) {
    const e = embed(t);
    const inp = new Float64Array(K + 32);
    for (let i = 0; i < K; i++) inp[i] = z[i];
    for (let i = 0; i < 32; i++) inp[K + i] = e[i];
    let h = inp;
    diffLayers.forEach((name, i) => {
      h = dense(W, L[name], h);
      if (i < diffLayers.length - 1) h = silu(h);
    });
    return h;
  }

  /** The decoder: tanh inside, logistic out, which is the bottleneck article's
   *  shape and therefore the shape whose ceiling this page measures. */
  function decode(z) {
    let h = z;
    decLayers.forEach((name, i) => {
      h = dense(W, L[name], h);
      h = i < decLayers.length - 1 ? tanh(h) : sigmoid(h);
    });
    return h;
  }

  function encode(x) {
    let h = x;
    encLayers.forEach((name, i) => {
      h = dense(W, L[name], h);
      if (i < encLayers.length - 1) h = tanh(h);
    });
    return h;
  }

  /** Standardised latent back to the decoder's own scale. The forward process
   *  pushes towards a standard normal, so the generator trained on a
   *  standardised copy and stored the shift and the scale. */
  function unstandardise(z) {
    const out = new Float64Array(K);
    for (let i = 0; i < K; i++) out[i] = z[i] * N.sd[i] + N.mu[i];
    return out;
  }

  /**
   * Sample `n` pictures: run the reverse chain in the latent, decode at the end.
   * The clamp on the predicted clean latent forces the noise to be recomputed
   * from it, exactly as in the pixel article: the update is built on an
   * identity, and changing one side of it without the other breaks the chain.
   */
  function sample(n, stepsUsed, seed) {
    const rnd = lcg(seed);
    const zs = [];
    for (let i = 0; i < n; i++) {
      const z = new Float64Array(K);
      for (let j = 0; j < K; j++) z[j] = rnd.normal();
      zs.push(z);
    }
    const idx = [];
    for (let j = 0; j < stepsUsed; j++) {
      idx.push(Math.round(steps - 1 - (j * (steps - 1)) / (stepsUsed - 1)));
    }
    for (let j = 0; j < idx.length; j++) {
      const t = idx[j];
      const aT = abar[t];
      const aPrev = j + 1 < idx.length ? abar[idx[j + 1]] : 1.0;
      const sT = Math.sqrt(aT);
      const sN = Math.sqrt(Math.max(1 - aT, 0));
      const sP = Math.sqrt(aPrev);
      const sPN = Math.sqrt(Math.max(1 - aPrev, 0));
      for (let i = 0; i < n; i++) {
        const z = zs[i];
        const eps = epsOne(z, t);
        const nz = new Float64Array(K);
        for (let d = 0; d < K; d++) {
          let z0 = (z[d] - sN * eps[d]) / sT;
          z0 = Math.max(-4, Math.min(4, z0));
          const e2 = (z[d] - sT * z0) / Math.max(sN, 1e-12);
          nz[d] = sP * z0 + sPN * e2;
        }
        zs[i] = nz;
      }
    }
    return { latents: zs, pictures: zs.map((z) => decode(unstandardise(z))) };
  }

  /** The round trip that fixes the ceiling: a real picture in, the same picture
   *  as the decoder can manage out. */
  function roundTrip(x) {
    return decode(encode(x));
  }

  return { epsOne, decode, encode, unstandardise, sample, roundTrip, K, steps };
}
