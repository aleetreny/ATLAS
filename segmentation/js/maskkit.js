/* The promptable network, run in the browser.
 *
 * Same contract as the detector in the previous article: the generator folds
 * batch normalisation into the convolutions, quantises to float16, writes the
 * weights as base64 and records exactly what has to happen to them. This file
 * implements that and nothing else, and the widget checks its own masks
 * against reference masks the generator computed in float64 with the same
 * quantised weights before believing them.
 *
 * The shape of the network is the U-Net's: down to a sixteenth of the canvas,
 * back up, with the encoder's two feature maps concatenated on the way. What
 * is different is the input, which carries a second channel saying where the
 * reader pointed, and the output, which is one number per pixel and no class
 * anywhere.
 */

/* base64 to Float32Array, reading the float16 the generator wrote. */
export function decodeWeights(b64, n) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

/* IEEE 754 binary16 to binary32, subnormals included: the weights are small
 * and a naive version flushes the tail of them to zero in silence. */
function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/* A tensor is a flat Float32Array plus its shape, channel-major: index
 * ((c * h) + y) * w + x, which is torch's layout for one image. */
export const tensor = (c, h, w) => ({ c, h, w, d: new Float32Array(c * h * w) });

export function conv2d(inp, weight, bias, cout, k, pad) {
  const out = tensor(cout, inp.h, inp.w);
  for (let o = 0; o < cout; o++) {
    const base = o * inp.h * inp.w;
    const b = bias[o];
    for (let y = 0; y < inp.h; y++) {
      for (let x = 0; x < inp.w; x++) {
        let acc = b;
        for (let ci = 0; ci < inp.c; ci++) {
          const wOff = ((o * inp.c) + ci) * k * k;
          const iOff = ci * inp.h * inp.w;
          for (let ky = 0; ky < k; ky++) {
            const sy = y + ky - pad;
            if (sy < 0 || sy >= inp.h) continue;
            for (let kx = 0; kx < k; kx++) {
              const sx = x + kx - pad;
              if (sx < 0 || sx >= inp.w) continue;
              acc += weight[wOff + ky * k + kx] * inp.d[iOff + sy * inp.w + sx];
            }
          }
        }
        out.d[base + y * inp.w + x] = acc;
      }
    }
  }
  return out;
}

export function relu(t) {
  for (let i = 0; i < t.d.length; i++) if (t.d[i] < 0) t.d[i] = 0;
  return t;
}

export function maxPool2(inp) {
  const h = inp.h >> 1;
  const w = inp.w >> 1;
  const out = tensor(inp.c, h, w);
  for (let c = 0; c < inp.c; c++) {
    const si = c * inp.h * inp.w;
    const di = c * h * w;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = inp.d[si + (2 * y) * inp.w + 2 * x];
        const b = inp.d[si + (2 * y) * inp.w + 2 * x + 1];
        const c2 = inp.d[si + (2 * y + 1) * inp.w + 2 * x];
        const d2 = inp.d[si + (2 * y + 1) * inp.w + 2 * x + 1];
        out.d[di + y * w + x] = Math.max(a, b, c2, d2);
      }
    }
  }
  return out;
}

/* Nearest-neighbour upsampling by two, which is what the export says the
 * decoder does. Bilinear here would be a different network. */
export function upsample2(inp) {
  const h = inp.h * 2;
  const w = inp.w * 2;
  const out = tensor(inp.c, h, w);
  for (let c = 0; c < inp.c; c++) {
    const si = c * inp.h * inp.w;
    const di = c * h * w;
    for (let y = 0; y < h; y++) {
      const sy = y >> 1;
      for (let x = 0; x < w; x++) {
        out.d[di + y * w + x] = inp.d[si + sy * inp.w + (x >> 1)];
      }
    }
  }
  return out;
}

/* Concatenate along channels; both tensors must already agree on h and w. */
export function concat(a, b) {
  const out = tensor(a.c + b.c, a.h, a.w);
  out.d.set(a.d, 0);
  out.d.set(b.d, a.d.length);
  return out;
}

export const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/* The prompt channel: a bump at the point the reader clicked, with the width
 * the generator trained with. Read sigma from the export rather than assuming
 * it, because it is a property of the trained weights and not a drawing
 * choice. */
export function promptChannel(px, py, side, sigma) {
  const out = new Float32Array(side * side);
  const denom = 2 * sigma * sigma;
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      const dx = x - px;
      const dy = y - py;
      out[y * side + x] = Math.exp(-(dx * dx + dy * dy) / denom);
    }
  }
  return out;
}

/* The whole network. The layer list and its shapes come from the export, so
 * the order below is the order the generator recorded rather than one assumed
 * here; the encoder maps are kept because the decoder concatenates them. */
export function forward(model, image, prompt) {
  const { layers, arch } = model;
  const w = model.weights;
  const side = arch.canvas;
  const take = (i) => {
    const L = layers[i];
    return {
      weight: w.subarray(L.w_offset, L.w_offset + L.w_size),
      bias: w.subarray(L.b_offset, L.b_offset + L.b_size),
      cout: L.shape[0],
      k: L.shape[3],
    };
  };
  const inp = tensor(2, side, side);
  inp.d.set(image, 0);
  inp.d.set(prompt, side * side);

  const l0 = take(0);
  const s1 = relu(conv2d(inp, l0.weight, l0.bias, l0.cout, l0.k, arch.pad));
  const l1 = take(1);
  const s2 = relu(conv2d(maxPool2(s1), l1.weight, l1.bias, l1.cout, l1.k, arch.pad));
  const l2 = take(2);
  const z = relu(conv2d(maxPool2(s2), l2.weight, l2.bias, l2.cout, l2.k, arch.pad));
  const l3 = take(3);
  const u2 = relu(conv2d(concat(upsample2(z), s2), l3.weight, l3.bias, l3.cout, l3.k, arch.pad));
  const l4 = take(4);
  const u1 = relu(conv2d(concat(upsample2(u2), s1), l4.weight, l4.bias, l4.cout, l4.k, arch.pad));
  const l5 = take(5);
  /* the head is 1x1, so it carries no padding of its own */
  return conv2d(u1, l5.weight, l5.bias, l5.cout, l5.k, 0);
}

/* Counting, not estimating: these are the same two ratios the generator
 * computed on the same integer masks. */
export function maskIou(a, b) {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter += 1;
    if (a[i] || b[i]) union += 1;
  }
  return union > 0 ? inter / union : 0;
}

export function maskDice(a, b) {
  let inter = 0;
  let tot = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter += 1;
    if (a[i]) tot += 1;
    if (b[i]) tot += 1;
  }
  return tot > 0 ? (2 * inter) / tot : 0;
}
