/* The trained pair of functions, run here.
 *
 * The weights arrive as float16 in base64, the same export contract the
 * detection article introduced, because 25,670 numbers at four decimals of JSON
 * is a quarter of a megabyte and at two bytes each it is sixty-seven kilobytes.
 * The generator measures every figure this page quotes on the QUANTISED weights,
 * because those are the numbers the browser will have.
 */

/* float16 to float64. The naive version, which ignores the exponent-zero case,
 * flushes every subnormal to zero: about a fifth of the small weights in a
 * trained network, and it moves the output visibly. */
function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

export function decodeWeights(b64, n) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const u16 = new Uint16Array(buf);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

/* A layer's weights are written row major over OUTPUT units, so reading them
 * back is a stride of `inputs` per output and no transpose is needed here. */
function layer(w, spec, x) {
  const [nin, nout] = spec.shape;
  const out = new Float64Array(nout);
  for (let o = 0; o < nout; o++) {
    let s = w[spec.b_offset + o];
    const base = spec.w_offset + o * nin;
    for (let i = 0; i < nin; i++) s += w[base + i] * x[i];
    out[o] = s;
  }
  return out;
}

const tanh = (v) => v.map(Math.tanh);
const sigmoid = (v) => v.map((t) => 1 / (1 + Math.exp(-t)));

export function makeNet(data) {
  const N = data.net;
  const w = decodeWeights(N.weights_b64, N.count);
  const [e1, e2, d1, d2] = N.layers;
  return {
    encode(x) {
      return layer(w, e2, tanh(layer(w, e1, x)));
    },
    decode(z) {
      return sigmoid(layer(w, d2, tanh(layer(w, d1, z))));
    },
    layers: N.layers,
    weights: w,
  };
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}
