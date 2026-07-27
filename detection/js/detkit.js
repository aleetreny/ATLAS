/* The detector, run in the browser.
 *
 * The generator folds batch normalisation into the convolutions, quantises the
 * weights to float16 and writes them as base64, together with a contract that
 * says exactly what has to happen to them: convolution with padding 1, bias,
 * ReLU except on the head, a 2x2 max pool after the listed layers, then a
 * decode from grid offsets to corners. Everything below implements that
 * contract and nothing else, and the widget checks its own boxes against
 * reference boxes the generator computed in float64 before believing them.
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

/* IEEE 754 binary16 to binary32, including subnormals and the specials: the
 * weights are small and a naive version silently flushes the tail to zero. */
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
  const half = pad;
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
            const sy = y + ky - half;
            if (sy < 0 || sy >= inp.h) continue;
            for (let kx = 0; kx < k; kx++) {
              const sx = x + kx - half;
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

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/* The whole network: the layer list comes from the export, so the order here
 * is the order the generator recorded rather than one assumed. */
export function forward(model, image) {
  const { layers, arch } = model;
  const w = model.weights;
  let x = { c: 1, h: arch.grid * arch.stride, w: arch.grid * arch.stride, d: image };
  layers.forEach((L, i) => {
    /* shape is [cout, cin, k, k] as torch stores it, and the offsets are the
       names the generator writes: w_offset/w_size, b_offset/b_size. Reading
       them from the export rather than assuming is the whole point of the
       export having a contract. */
    const weight = w.subarray(L.w_offset, L.w_offset + L.w_size);
    const bias = w.subarray(L.b_offset, L.b_offset + L.b_size);
    const cout = L.shape[0];
    const k = L.shape[3];
    const pad = k === 1 ? 0 : arch.pad;
    x = conv2d(x, weight, bias, cout, k, pad);
    if (i < layers.length - 1) {
      relu(x);
      if (arch.pool_after.includes(i)) x = maxPool2(x);
    }
  });
  return x;
}

/* head channel a*(5+C)+t: t 0..3 box, 4 objectness, 5.. class logits */
export function decode(head, arch) {
  const C = arch.n_classes;
  const A = arch.anchors.length;
  const G = head.h;
  const S = arch.stride;
  const per = 5 + C;
  const out = [];
  for (let a = 0; a < A; a++) {
    const [aw, ah] = arch.anchors[a];
    for (let row = 0; row < G; row++) {
      for (let col = 0; col < G; col++) {
        const at = (t) => head.d[((a * per + t) * G + row) * G + col];
        const cx = (col + sigmoid(at(0))) * S;
        const cy = (row + sigmoid(at(1))) * S;
        const bw = aw * Math.exp(Math.max(-4, Math.min(4, at(2))));
        const bh = ah * Math.exp(Math.max(-4, Math.min(4, at(3))));
        const obj = sigmoid(at(4));
        let best = -Infinity;
        let bi = 0;
        let sum = 0;
        const logits = [];
        for (let c = 0; c < C; c++) {
          const v = at(5 + c);
          logits.push(v);
          if (v > best) { best = v; bi = c; }
        }
        for (let c = 0; c < C; c++) sum += Math.exp(logits[c] - best);
        const score = obj * (1 / sum);
        const lim = arch.grid * S;
        out.push({
          cls: bi,
          score,
          box: [
            Math.max(0, Math.min(lim, cx - bw / 2)),
            Math.max(0, Math.min(lim, cy - bh / 2)),
            Math.max(0, Math.min(lim, cx + bw / 2)),
            Math.max(0, Math.min(lim, cy + bh / 2)),
          ],
        });
      }
    }
  }
  return out;
}

export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ar = (a[2] - a[0]) * (a[3] - a[1]);
  const br = (b[2] - b[0]) * (b[3] - b[1]);
  const uni = ar + br - inter;
  return uni > 0 ? inter / uni : 0;
}

/* Greedy non-maximum suppression, per class, highest score first. */
export function nms(dets, threshold, classwise = true) {
  const kept = [];
  const order = [...dets].sort((p, q) => q.score - p.score);
  const dead = new Array(order.length).fill(false);
  for (let i = 0; i < order.length; i++) {
    if (dead[i]) continue;
    kept.push(order[i]);
    for (let j = i + 1; j < order.length; j++) {
      if (dead[j]) continue;
      if (classwise && order[j].cls !== order[i].cls) continue;
      if (iou(order[i].box, order[j].box) > threshold) dead[j] = true;
    }
  }
  return kept;
}
