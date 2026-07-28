/* Trained weights, in float16, run in the page.
 *
 * The export contract is the one the detection article introduced and the
 * autoencoders article reused: a flat float16 payload in base64, plus a list of
 * layer specifications saying where each layer's weights and biases start. Each
 * layer is written row major over OUTPUT units, so reading it back is a stride
 * and never a transpose.
 *
 * Every article in module 3.2 runs at least one trained model here, so the
 * decoder lives once, in this file, instead of four times. The one subtle part
 * is `halfToFloat`: a version that ignores the zero exponent flushes every
 * subnormal to zero, which is about a fifth of the small weights of a trained
 * network and moves the output visibly.
 */

export function halfToFloat(h) {
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
  const count = n === undefined ? u16.length : n;
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

/* One dense layer applied to one vector. `spec` carries shape [inputs, outputs],
 * the offsets, and a bias size that may be zero. */
export function dense(w, spec, x) {
  const [nin, nout] = spec.shape;
  const out = new Float64Array(nout);
  for (let o = 0; o < nout; o++) {
    let s = spec.b_size ? w[spec.b_offset + o] : 0;
    const base = spec.w_offset + o * nin;
    for (let i = 0; i < nin; i++) s += w[base + i] * x[i];
    out[o] = s;
  }
  return out;
}

export const tanh = (v) => v.map(Math.tanh);
export const sigmoid = (v) => v.map((t) => 1 / (1 + Math.exp(-t)));
export const relu = (v) => v.map((t) => (t > 0 ? t : 0));
export const softplus = (v) => v.map((t) => (t > 20 ? t : Math.log1p(Math.exp(t))));

/* A layer specification list, by name, so a caller can say what it wants
 * instead of counting positions. */
export function byName(layers) {
  const out = {};
  layers.forEach((s) => { out[s.name] = s; });
  return out;
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}

/* The linear congruential generator plus Box-Muller that the generators write
 * twice, once in python and once here, so that anything the page has to sample
 * lands in the same place the measurement did. numpy's generator does not
 * reproduce in a browser; this one does. */
export function lcg(seed) {
  let s = seed >>> 0;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  return {
    uniform: next,
    normal() {
      const u1 = Math.max(next(), 1e-12);
      const u2 = next();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    },
  };
}
