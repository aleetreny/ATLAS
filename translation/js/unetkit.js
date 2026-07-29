/* The two trained translators, running here.
 *
 * Same export contract as the previous two articles: float16 in base64, a
 * table of offsets and shapes, and every figure the article quotes measured on
 * these quantised weights. Two networks travel rather than one, because the
 * whole first half of this article is a comparison between them and reading it
 * on your own drawing is the point.
 *
 * Two details that are easy to get wrong and invisible if you do. Group
 * normalisation uses the biased variance, dividing by n, which is what torch
 * does and is not what a hand written standard deviation usually does. And the
 * decoder concatenates the encoder's feature map BEFORE its convolution, in
 * that channel order, so getting the order backwards produces a picture that
 * looks plausible and is not the model's.
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
const relu = (v) => (v > 0 ? v : 0);
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/* A tensor is {data, c, h, w}, channels first, which is torch's order and
 * therefore the order the exported weights expect. */
function conv(x, W, b, kernel, stride, pad, cout) {
  const oh = Math.floor((x.h + 2 * pad - kernel) / stride) + 1;
  const ow = Math.floor((x.w + 2 * pad - kernel) / stride) + 1;
  const out = new Float32Array(cout * oh * ow);
  for (let o = 0; o < cout; o++) {
    const ob = o * oh * ow;
    out.fill(b[o], ob, ob + oh * ow);
    for (let i = 0; i < x.c; i++) {
      const wb = (o * x.c + i) * kernel * kernel;
      const ib = i * x.h * x.w;
      for (let y = 0; y < oh; y++) {
        for (let xx = 0; xx < ow; xx++) {
          let acc = 0;
          for (let ky = 0; ky < kernel; ky++) {
            const sy = y * stride + ky - pad;
            if (sy < 0 || sy >= x.h) continue;
            for (let kx = 0; kx < kernel; kx++) {
              const sx = xx * stride + kx - pad;
              if (sx < 0 || sx >= x.w) continue;
              acc += W[wb + ky * kernel + kx] * x.data[ib + sy * x.w + sx];
            }
          }
          out[ob + y * ow + xx] += acc;
        }
      }
    }
  }
  return { data: out, c: cout, h: oh, w: ow };
}

/* torch's GroupNorm: biased variance over (channels in the group) x h x w. */
function groupNorm(x, gamma, beta, groups = 4, eps = 1e-5) {
  const per = x.c / groups;
  const np = x.h * x.w;
  const out = new Float32Array(x.data.length);
  for (let g = 0; g < groups; g++) {
    let mu = 0;
    for (let c = g * per; c < (g + 1) * per; c++) {
      for (let p = 0; p < np; p++) mu += x.data[c * np + p];
    }
    mu /= per * np;
    let va = 0;
    for (let c = g * per; c < (g + 1) * per; c++) {
      for (let p = 0; p < np; p++) va += (x.data[c * np + p] - mu) ** 2;
    }
    va /= per * np;
    const inv = 1 / Math.sqrt(va + eps);
    for (let c = g * per; c < (g + 1) * per; c++) {
      for (let p = 0; p < np; p++) {
        out[c * np + p] = (x.data[c * np + p] - mu) * inv * gamma[c] + beta[c];
      }
    }
  }
  return { data: out, c: x.c, h: x.h, w: x.w };
}

function act(x, fn) {
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) out[i] = fn(x.data[i]);
  return { data: out, c: x.c, h: x.h, w: x.w };
}

function upsample(x) {
  const h = x.h * 2;
  const w = x.w * 2;
  const out = new Float32Array(x.c * h * w);
  for (let c = 0; c < x.c; c++) {
    for (let y = 0; y < h; y++) {
      for (let xx = 0; xx < w; xx++) {
        out[c * h * w + y * w + xx] = x.data[c * x.h * x.w + (y >> 1) * x.w + (xx >> 1)];
      }
    }
  }
  return { data: out, c: x.c, h, w };
}

function concat(a, b) {
  const out = new Float32Array(a.data.length + b.data.length);
  out.set(a.data, 0);
  out.set(b.data, a.data.length);
  return { data: out, c: a.c + b.c, h: a.h, w: a.w };
}

export class UNet {
  constructor(spec) {
    this.w = decodeWeights(spec.weights_b64, spec.count);
    this.layers = spec.layers;
  }

  take(name) {
    const s = this.layers[name];
    if (!s) throw new Error(`the export has no tensor called ${name}`);
    const n = s.shape.reduce((a, b) => a * b, 1);
    return this.w.subarray(s.offset, s.offset + n);
  }

  shapeOf(name) {
    return this.layers[name].shape;
  }

  /* img: a res x res x 3 Float32Array, the way a canvas hands it over.
     Returns the same layout, so a caller never sees channels first. */
  run(img, res) {
    const x = { data: new Float32Array(3 * res * res), c: 3, h: res, w: res };
    for (let p = 0; p < res * res; p++) {
      for (let c = 0; c < 3; c++) x.data[c * res * res + p] = img[p * 3 + c];
    }
    let h1 = conv(x, this.take('e1.0.weight'), this.take('e1.0.bias'), 4, 2, 1,
      this.shapeOf('e1.0.weight')[0]);
    h1 = act(h1, leaky);
    let h2 = conv(h1, this.take('e2.0.weight'), this.take('e2.0.bias'), 4, 2, 1,
      this.shapeOf('e2.0.weight')[0]);
    h2 = act(groupNorm(h2, this.take('e2.1.weight'), this.take('e2.1.bias')), leaky);
    let h3 = conv(h2, this.take('e3.0.weight'), this.take('e3.0.bias'), 4, 2, 1,
      this.shapeOf('e3.0.weight')[0]);
    h3 = act(groupNorm(h3, this.take('e3.1.weight'), this.take('e3.1.bias')), relu);

    let u3 = conv(upsample(h3), this.take('d3.1.weight'), this.take('d3.1.bias'), 3, 1, 1,
      this.shapeOf('d3.1.weight')[0]);
    u3 = act(groupNorm(u3, this.take('d3.2.weight'), this.take('d3.2.bias')), relu);
    let u2 = conv(upsample(concat(u3, h2)), this.take('d2.1.weight'), this.take('d2.1.bias'),
      3, 1, 1, this.shapeOf('d2.1.weight')[0]);
    u2 = act(groupNorm(u2, this.take('d2.2.weight'), this.take('d2.2.bias')), relu);
    const out = conv(upsample(concat(u2, h1)), this.take('d1.1.weight'), this.take('d1.1.bias'),
      3, 1, 1, 3);

    const rgb = new Float32Array(res * res * 3);
    for (let p = 0; p < res * res; p++) {
      for (let c = 0; c < 3; c++) rgb[p * 3 + c] = sigmoid(out.data[c * res * res + p]);
    }
    return rgb;
  }
}

/* Mean absolute detail left after a 3x3 box blur: the same quantity the
 * generator measures, so a number produced on the reader's own drawing is
 * comparable with the table. */
export function highFreq(img, res) {
  let acc = 0;
  for (let y = 0; y < res; y++) {
    for (let x = 0; x < res; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const sy = Math.min(Math.max(y + dy, 0), res - 1);
            const sx = Math.min(Math.max(x + dx, 0), res - 1);
            s += img[(sy * res + sx) * 3 + c];
          }
        }
        acc += Math.abs(img[(y * res + x) * 3 + c] - s / 9);
      }
    }
  }
  return acc / (res * res * 3);
}
