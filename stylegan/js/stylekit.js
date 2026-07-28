/* The trained style-based generator, run here.
 *
 * Same contract as the detection and bottleneck articles: float16 weights in
 * base64, a table of offsets and shapes, and every figure the article quotes
 * measured on these quantised weights rather than on the ones the optimiser
 * produced. What is new is that the whole point of this generator is where the
 * latent enters, so this file exposes the styles per layer rather than only a
 * forward pass: the widgets hand it a list of five w vectors, one per block,
 * and mixing two samples is passing two different vectors in that list.
 *
 * One detail that is easy to get wrong and impossible to see: torch's `std`
 * is the unbiased one, dividing by n-1. The instance normalisation below does
 * the same, because a browser that divides by n produces a slightly different
 * picture from the one the generator measured and nothing on screen says so.
 */

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
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

const leaky = (v) => (v > 0 ? v : 0.2 * v);
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

export class StyleGen {
  constructor(data) {
    const N = data.net;
    this.w = decodeWeights(N.weights_b64, N.count);
    this.layers = N.layers;
    this.meta = data.meta;
    this.wMean = N.w_mean;
    this.blocks = data.meta.blocks;
    this.zDim = data.meta.z;
    this.wDim = data.meta.w;
  }

  take(name) {
    const spec = this.layers[name];
    if (!spec) throw new Error(`the export has no tensor called ${name}`);
    return { data: this.w.subarray(spec.offset, spec.offset + spec.shape.reduce((a, b) => a * b, 1)), shape: spec.shape };
  }

  /* z -> w. Four affine layers with leaky rectifiers between them, which is
     the mapping network at this scale. */
  map(z) {
    let x = z;
    [0, 2, 4, 6].forEach((li, k) => {
      const W = this.take(`mapping.net.${li}.weight`);
      const b = this.take(`mapping.net.${li}.bias`);
      const [nout, nin] = W.shape;
      const out = new Float32Array(nout);
      for (let o = 0; o < nout; o++) {
        let acc = b.data[o];
        for (let i = 0; i < nin; i++) acc += W.data[o * nin + i] * x[i];
        out[o] = k < 3 ? leaky(acc) : acc;
      }
      x = out;
    });
    return x;
  }

  /* The average w over the prior, which is what truncation pulls towards. */
  meanW() {
    return Float32Array.from(this.wMean);
  }

  truncate(w, psi) {
    const m = this.wMean;
    const out = new Float32Array(w.length);
    for (let i = 0; i < w.length; i++) out[i] = m[i] + psi * (w[i] - m[i]);
    return out;
  }

  /* ws: an array of five style vectors, one per block. noises: an array of
     five Float32Arrays (or null for zeros). Returns a 32x32x3 Float32Array. */
  synth(ws, noises = null) {
    const c0 = this.take('const');
    let ch = c0.shape[1];
    let res = c0.shape[2];
    let x = Float32Array.from(c0.data);

    this.blocks.forEach((spec, bi) => {
      const W = this.take(`blocks.${bi}.conv.weight`);
      const bias = this.take(`blocks.${bi}.conv.bias`);
      const [cout, cin] = W.shape;
      const up = spec.res > res;
      if (up) {
        const nres = res * 2;
        const nx = new Float32Array(ch * nres * nres);
        for (let c = 0; c < ch; c++) {
          for (let y = 0; y < nres; y++) {
            for (let xx = 0; xx < nres; xx++) {
              nx[c * nres * nres + y * nres + xx] =
                x[c * res * res + (y >> 1) * res + (xx >> 1)];
            }
          }
        }
        x = nx;
        res = nres;
      }
      /* 3x3 convolution, zero padded, which is what torch does with padding 1 */
      const out = new Float32Array(cout * res * res);
      for (let o = 0; o < cout; o++) {
        const ob = o * res * res;
        out.fill(bias.data[o], ob, ob + res * res);
        for (let i = 0; i < cin; i++) {
          const wb = (o * cin + i) * 9;
          const ib = i * res * res;
          for (let y = 0; y < res; y++) {
            for (let xx = 0; xx < res; xx++) {
              let acc = 0;
              for (let ky = -1; ky <= 1; ky++) {
                const sy = y + ky;
                if (sy < 0 || sy >= res) continue;
                for (let kx = -1; kx <= 1; kx++) {
                  const sx = xx + kx;
                  if (sx < 0 || sx >= res) continue;
                  acc += W.data[wb + (ky + 1) * 3 + (kx + 1)] * x[ib + sy * res + sx];
                }
              }
              out[ob + y * res + xx] += acc;
            }
          }
        }
      }
      /* per-pixel noise, one map shared across channels, scaled per channel */
      const nscale = this.take(`blocks.${bi}.noise`);
      const noise = noises ? noises[bi] : null;
      if (noise) {
        for (let o = 0; o < cout; o++) {
          const s = nscale.data[o];
          for (let p = 0; p < res * res; p++) out[o * res * res + p] += s * noise[p];
        }
      }
      /* adaptive instance normalisation: forget this sample's own statistics,
         then adopt the ones the style asks for */
      const A = this.take(`blocks.${bi}.affine.weight`);
      const Ab = this.take(`blocks.${bi}.affine.bias`);
      const wv = ws[bi];
      const style = new Float32Array(cout * 2);
      for (let o = 0; o < cout * 2; o++) {
        let acc = Ab.data[o];
        for (let i = 0; i < this.wDim; i++) acc += A.data[o * this.wDim + i] * wv[i];
        style[o] = acc;
      }
      const np = res * res;
      for (let o = 0; o < cout; o++) {
        const ob = o * np;
        let mu = 0;
        for (let p = 0; p < np; p++) mu += out[ob + p];
        mu /= np;
        let va = 0;
        for (let p = 0; p < np; p++) va += (out[ob + p] - mu) ** 2;
        const sd = Math.sqrt(va / (np - 1)) + 1e-5;      // unbiased, like torch
        const gamma = 1 + style[o];
        const beta = style[cout + o];
        for (let p = 0; p < np; p++) {
          out[ob + p] = leaky(((out[ob + p] - mu) / sd) * gamma + beta);
        }
      }
      x = out;
      ch = cout;
    });

    const R = this.take('to_rgb.weight');
    const Rb = this.take('to_rgb.bias');
    const img = new Float32Array(res * res * 3);
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < res * res; p++) {
        let acc = Rb.data[c];
        for (let i = 0; i < ch; i++) acc += R.data[c * ch + i] * x[i * res * res + p];
        img[p * 3 + c] = sigmoid(acc);
      }
    }
    return img;
  }

  /* Convenience: one latent, one image, optionally truncated. */
  fromZ(z, { psi = 1, noises = null } = {}) {
    let w = this.map(z);
    if (psi !== 1) w = this.truncate(w, psi);
    return this.synth(new Array(this.blocks.length).fill(w), noises);
  }
}

/* One noise map per block, from a seed, so that a widget can hold the noise
 * fixed while it varies the style. Without that, two pictures differ in two
 * things at once and the comparison the widget exists for is not the one on
 * screen. */
export function makeNoise(blocks, seed) {
  let s = seed >>> 0;
  const u = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  return blocks.map((b, i) => {
    const res = [4, 8, 8, 16, 32][i];
    const out = new Float32Array(res * res);
    for (let k = 0; k < res * res; k++) {
      const u1 = Math.max(u(), 1e-12);
      const u2 = u();
      out[k] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return out;
  });
}

/* ------------------------------------------------------------------------ */
/* The measurement functions, written the same way as in the generator, so the
 * factors this page reports off a generated image are the same quantities the
 * article's tables were built from. */
export function rgbToHue(r, g, b) {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const d = Math.max(mx - mn, 1e-9);
  let h;
  if (mx === r) h = ((g - b) / d) % 6;
  else if (mx === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h /= 6;
  return ((h % 1) + 1) % 1;
}

export function hueDist(a, b) {
  const d = Math.abs(a - b) % 1;
  return Math.min(d, 1 - d);
}

export function measure(img, res, thresh = 0.14) {
  const border = [];
  for (let i = 0; i < res; i++) {
    [[0, i], [res - 1, i], [i, 0], [i, res - 1]].forEach(([y, x]) => {
      const p = (y * res + x) * 3;
      border.push([img[p], img[p + 1], img[p + 2]]);
    });
  }
  const med = [0, 1, 2].map((c) => {
    const v = border.map((p) => p[c]).sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)];
  });
  let area = 0;
  let sx = 0;
  let sy = 0;
  let rr = 0;
  let gg = 0;
  let bb = 0;
  for (let y = 0; y < res; y++) {
    for (let x = 0; x < res; x++) {
      const p = (y * res + x) * 3;
      const d = Math.abs(img[p] - med[0]) + Math.abs(img[p + 1] - med[1])
        + Math.abs(img[p + 2] - med[2]);
      if (d > thresh) {
        area += 1;
        sx += x + 0.5;
        sy += y + 0.5;
        rr += img[p];
        gg += img[p + 1];
        bb += img[p + 2];
      }
    }
  }
  const n = Math.max(area, 1);
  return {
    area,
    size: Math.sqrt(area / Math.PI),
    cx: sx / n,
    cy: sy / n,
    hue: area >= 4 ? rgbToHue(rr / n, gg / n, bb / n) : NaN,
    coverage: area / (res * res),
  };
}
