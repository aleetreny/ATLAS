/* The encoder, the codebook, the decoder and the prior over codes, run here.
 *
 * Everything is the float16 the generator exported, and the generator measured
 * every figure this page quotes on those same quantised numbers: quantising
 * moved two of 2,400 code assignments and nothing at five decimals of the held
 * out error, which is in the readout rather than in a comment.
 */
import { decodeWeights, dense, relu, sigmoid, byName, lcg } from '../../assets/js/halfweights.js';

export function makeVq(data) {
  const N = data.net;
  const w = decodeWeights(N.weights_b64, N.count);
  const L = byName(N.layers);
  const book = decodeWeights(N.codebook_b64, N.codebook_count);
  const { cells, dim, k, pixels } = N.arch;

  const entry = (j) => book.subarray(j * dim, (j + 1) * dim);

  return {
    cells,
    dim,
    k,
    pixels,
    entry,
    /* 196 pixels to `cells` vectors of `dim` numbers */
    encode(x) {
      return dense(w, L.enc2, relu(dense(w, L.enc1, x)));
    },
    /* nearest entry, and the distance to the runner up, which is the number
       the gradient section is about */
    quantise(z) {
      const out = [];
      for (let c = 0; c < cells; c++) {
        let best = -1;
        let bd = Infinity;
        let second = Infinity;
        for (let j = 0; j < k; j++) {
          const e = entry(j);
          let s = 0;
          for (let i = 0; i < dim; i++) s += (z[c * dim + i] - e[i]) ** 2;
          if (s < bd) { second = bd; bd = s; best = j; } else if (s < second) second = s;
        }
        out.push({ code: best, d: Math.sqrt(bd), margin: (Math.sqrt(second) - Math.sqrt(bd)) / 2 });
      }
      return out;
    },
    /* code indices back to a picture */
    decode(codes) {
      const z = new Float64Array(cells * dim);
      codes.forEach((j, c) => {
        const e = entry(j);
        for (let i = 0; i < dim; i++) z[c * dim + i] = e[i];
      });
      return sigmoid(dense(w, L.dec2, relu(dense(w, L.dec1, z))));
    },
  };
}

/* The autoregressive prior: 256 masked one-hot entries then four position
 * flags in, 64 logits out. Cells are filled left to right, and a cell that has
 * not been filled yet contributes zeros, which is what "masked" means here. */
export function makePrior(data) {
  const P = data.net.prior;
  const w = decodeWeights(P.weights_b64, P.count);
  const L = byName(P.layers);
  const { cells, k } = data.net.arch;

  const logitsFor = (codes, pos) => {
    const x = new Float64Array(cells * k + cells);
    for (let c = 0; c < pos; c++) x[c * k + codes[c]] = 1;
    x[cells * k + pos] = 1;
    return dense(w, L.p2, relu(dense(w, L.p1, x)));
  };

  return {
    logitsFor,
    sample(seed) {
      const rng = lcg(seed);
      const codes = new Array(cells).fill(0);
      for (let c = 0; c < cells; c++) {
        const lg = logitsFor(codes, c);
        let m = -Infinity;
        for (let j = 0; j < k; j++) m = Math.max(m, lg[j]);
        let tot = 0;
        const p = new Float64Array(k);
        for (let j = 0; j < k; j++) { p[j] = Math.exp(lg[j] - m); tot += p[j]; }
        let u = rng.uniform() * tot;
        let pick = k - 1;
        for (let j = 0; j < k; j++) { u -= p[j]; if (u <= 0) { pick = j; break; } }
        codes[c] = pick;
      }
      return codes;
    },
  };
}

export function uniformCodes(k, cells, seed) {
  const rng = lcg(seed);
  return Array.from({ length: cells }, () => Math.min(k - 1, Math.floor(rng.uniform() * k)));
}
