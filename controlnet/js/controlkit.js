/* The conditional sampler, run in the page.
 *
 * The whole trick of classifier free guidance is that ONE network answers two
 * questions, so the page runs one set of weights twice per step: once with the
 * label it was asked for and once with the null token the training used when it
 * hid the label. The difference between those two answers is the direction of
 * "more of this label", and the guidance scale is how far along it to step.
 * At a scale of zero the second pass is skipped and this is the unconditional
 * model from the previous article.
 */
import { byName, decodeWeights, dense, lcg } from '../../assets/js/halfweights.js';

export function makeControl(data) {
  const N = data.net;
  const W = decodeWeights(N.b64, N.count);
  const L = byName(N.spec);
  const K = N.k;
  const abar = N.abar;
  const steps = N.steps;
  const NULL = N.null_token;

  const silu = (v) => v.map((t) => t / (1 + Math.exp(-t)));
  const tanh = (v) => v.map(Math.tanh);
  const sigmoid = (v) => v.map((t) => 1 / (1 + Math.exp(-t)));

  const scoreLayers = N.spec.filter((s) => /^c\d+$/.test(s.name)).map((s) => s.name);
  const decLayers = N.spec.filter((s) => s.name.startsWith('dec')).map((s) => s.name);
  const embSpec = L.emb;

  /* the label embedding is a lookup, not a matrix product: row y of a table */
  function labelVec(y) {
    const [rows, cols] = embSpec.shape;
    const out = new Float64Array(cols);
    /* The export contract writes every matrix row major over OUTPUT units, so
       a table of `rows` classes by `cols` features is stored as `cols` blocks
       of `rows`. Reading it the other way round returns a different class's
       vector and still produces plausible digits, which is why the probe
       below checks the network's output rather than the reading. */
    for (let i = 0; i < cols; i++) out[i] = W[embSpec.w_offset + i * rows + y];
    return out;
  }

  function tembed(t) {
    const out = new Float64Array(32);
    for (let i = 0; i < 16; i++) {
      const f = Math.exp((Math.log(200) * i) / 15);
      const a = (t / steps) * f * Math.PI;
      out[i] = Math.sin(a);
      out[i + 16] = Math.cos(a);
    }
    return out;
  }

  /** eps_theta(z, t, y) for one latent point and one label. */
  function epsOne(z, t, y) {
    const te = tembed(t);
    const le = labelVec(y);
    const inp = new Float64Array(K + 32 + le.length);
    for (let i = 0; i < K; i++) inp[i] = z[i];
    for (let i = 0; i < 32; i++) inp[K + i] = te[i];
    for (let i = 0; i < le.length; i++) inp[K + 32 + i] = le[i];
    let h = inp;
    scoreLayers.forEach((name, i) => {
      h = dense(W, L[name], h);
      if (i < scoreLayers.length - 1) h = silu(h);
    });
    return h;
  }

  function decode(z) {
    let h = z;
    decLayers.forEach((name, i) => {
      h = dense(W, L[name], h);
      h = i < decLayers.length - 1 ? tanh(h) : sigmoid(h);
    });
    return h;
  }

  function unstandardise(z) {
    const out = new Float64Array(K);
    for (let i = 0; i < K; i++) out[i] = z[i] * N.sd[i] + N.mu[i];
    return out;
  }

  /** Sample one picture per entry of `labels`, at guidance `scale`. */
  function sample(labels, scale, stepsUsed, seed) {
    const n = labels.length;
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
        const un = epsOne(z, t, NULL);
        let eps = un;
        if (scale !== 0) {
          const co = epsOne(z, t, labels[i]);
          eps = new Float64Array(K);
          for (let d = 0; d < K; d++) eps[d] = un[d] + scale * (co[d] - un[d]);
        }
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

  return { epsOne, decode, unstandardise, sample, K, steps, NULL };
}
