/* Glow, run in the page, in both directions.
 *
 * Six steps of actnorm, a learned four by four mixing of the channels, and an
 * affine coupling layer, on a 14 by 14 image squeezed once into 7 by 7 by 4.
 * Every one of those is invertible by construction, so the same weights that
 * turn noise into a picture turn a picture back into the exact noise it came
 * from, which is what the round trip in the readout is checking.
 *
 * The tensors arrive as one float16 payload with a list of shapes, in the
 * layout torch stores them: a convolution is [out, in, 3, 3] row major.
 */
import { decodeWeights } from '../../assets/js/halfweights.js';

const relu = (v) => (v > 0 ? v : 0);

export function makeGlow(data) {
  const N = data.net;
  const flat = decodeWeights(N.weights_b64, N.count);
  const T = {};
  N.layers.forEach((s) => {
    T[s.name] = { data: flat.subarray(s.offset, s.offset + s.size), shape: s.shape };
  });
  const K = N.k;
  const H = data.meta.side / 2;
  const C = 4;
  const HW = H * H;

  /* a 3x3 (or 1x1) convolution with zero padding, on [cin][H][W] */
  function conv(x, wt, bt, cin, cout, k) {
    const out = new Float64Array(cout * HW);
    const pad = (k - 1) / 2;
    for (let o = 0; o < cout; o++) {
      const bias = bt.data[o];
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          let s = bias;
          for (let c = 0; c < cin; c++) {
            for (let a = 0; a < k; a++) {
              const ii = i + a - pad;
              if (ii < 0 || ii >= H) continue;
              for (let b = 0; b < k; b++) {
                const jj = j + b - pad;
                if (jj < 0 || jj >= H) continue;
                s += wt.data[((o * cin + c) * k + a) * k + b] * x[c * HW + ii * H + jj];
              }
            }
          }
          out[o * HW + i * H + j] = s;
        }
      }
    }
    return out;
  }

  function couplingST(step, xa) {
    const w0 = T[`cpl0_w_${step}`];
    const hid = w0.shape[0];
    let h = conv(xa, w0, T[`cpl0_b_${step}`], 2, hid, 3);
    for (let i = 0; i < h.length; i++) h[i] = relu(h[i]);
    h = conv(h, T[`cpl1_w_${step}`], T[`cpl1_b_${step}`], hid, hid, 1);
    for (let i = 0; i < h.length; i++) h[i] = relu(h[i]);
    const o = conv(h, T[`cpl2_w_${step}`], T[`cpl2_b_${step}`], hid, 4, 3);
    const s = new Float64Array(2 * HW);
    const t = new Float64Array(2 * HW);
    for (let i = 0; i < 2 * HW; i++) {
      s[i] = Math.tanh(o[i]) * 2.0;
      t[i] = o[2 * HW + i];
    }
    return { s, t };
  }

  function mix(x, step, inverse) {
    const W = T[`mix_w${step}`].data;
    /* four by four, so the inverse is written out rather than factorised */
    const M = inverse ? invert4(W) : W;
    const out = new Float64Array(C * HW);
    for (let o = 0; o < C; o++) {
      for (let p = 0; p < HW; p++) {
        let s = 0;
        for (let c = 0; c < C; c++) s += M[o * C + c] * x[c * HW + p];
        out[o * HW + p] = s;
      }
    }
    return out;
  }

  function invert4(W) {
    /* Gauss-Jordan on a 4 by 4, which is exact enough and short enough to read */
    const a = [];
    for (let i = 0; i < C; i++) {
      a.push([]);
      for (let j = 0; j < C; j++) a[i].push(W[i * C + j]);
      for (let j = 0; j < C; j++) a[i].push(i === j ? 1 : 0);
    }
    for (let col = 0; col < C; col++) {
      let piv = col;
      for (let r = col + 1; r < C; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
      const tmp = a[col]; a[col] = a[piv]; a[piv] = tmp;
      const d = a[col][col];
      for (let j = 0; j < 2 * C; j++) a[col][j] /= d;
      for (let r = 0; r < C; r++) {
        if (r === col) continue;
        const f = a[r][col];
        for (let j = 0; j < 2 * C; j++) a[r][j] -= f * a[col][j];
      }
    }
    const out = new Float64Array(C * C);
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) out[i * C + j] = a[i][C + j];
    return out;
  }

  function squeeze(img) {
    const side = data.meta.side;
    const out = new Float64Array(C * HW);
    for (let di = 0; di < 2; di++) {
      for (let dj = 0; dj < 2; dj++) {
        const oc = di * 2 + dj;
        for (let i = 0; i < H; i++) {
          for (let j = 0; j < H; j++) {
            out[oc * HW + i * H + j] = img[(2 * i + di) * side + 2 * j + dj];
          }
        }
      }
    }
    return out;
  }

  function unsqueeze(x) {
    const side = data.meta.side;
    const out = new Float64Array(side * side);
    for (let di = 0; di < 2; di++) {
      for (let dj = 0; dj < 2; dj++) {
        const oc = di * 2 + dj;
        for (let i = 0; i < H; i++) {
          for (let j = 0; j < H; j++) {
            out[(2 * i + di) * side + 2 * j + dj] = x[oc * HW + i * H + j];
          }
        }
      }
    }
    return out;
  }

  return {
    steps: K,
    forward(img) {
      let x = squeeze(img);
      let ld = 0;
      for (let i = 0; i < K; i++) {
        const ls = T[`act_logs${i}`].data;
        const b = T[`act_b${i}`].data;
        for (let c = 0; c < C; c++) {
          const e = Math.exp(ls[c]);
          for (let p = 0; p < HW; p++) x[c * HW + p] = x[c * HW + p] * e + b[c];
          ld += ls[c] * HW;
        }
        x = mix(x, i, false);
        ld += Math.log(Math.abs(det4(T[`mix_w${i}`].data))) * HW;
        const xa = x.subarray(0, 2 * HW);
        const { s, t } = couplingST(i, xa);
        const y = new Float64Array(C * HW);
        y.set(xa, 0);
        for (let p = 0; p < 2 * HW; p++) {
          y[2 * HW + p] = x[2 * HW + p] * Math.exp(s[p]) + t[p];
          ld += s[p];
        }
        x = y;
      }
      return { z: x, logdet: ld };
    },
    inverse(z) {
      let x = Float64Array.from(z);
      for (let i = K - 1; i >= 0; i--) {
        const xa = x.subarray(0, 2 * HW);
        const { s, t } = couplingST(i, xa);
        const y = new Float64Array(C * HW);
        y.set(xa, 0);
        for (let p = 0; p < 2 * HW; p++) {
          y[2 * HW + p] = (x[2 * HW + p] - t[p]) * Math.exp(-s[p]);
        }
        x = mix(y, i, true);
        const ls = T[`act_logs${i}`].data;
        const b = T[`act_b${i}`].data;
        for (let c = 0; c < C; c++) {
          const e = Math.exp(-ls[c]);
          for (let p = 0; p < HW; p++) x[c * HW + p] = (x[c * HW + p] - b[c]) * e;
        }
      }
      return unsqueeze(x);
    },
    logProb(img) {
      const { z, logdet } = this.forward(img);
      let lp = 0;
      for (let i = 0; i < z.length; i++) lp += -0.5 * z[i] * z[i] - 0.5 * Math.log(2 * Math.PI);
      return lp + logdet;
    },
    latentSize: C * HW,
  };
}

function det4(W) {
  const m = (i, j) => W[i * 4 + j];
  /* expansion by the first row, on the three by three minors */
  const minor = (r0, r1, r2, c0, c1, c2) => (
    m(r0, c0) * (m(r1, c1) * m(r2, c2) - m(r1, c2) * m(r2, c1))
    - m(r0, c1) * (m(r1, c0) * m(r2, c2) - m(r1, c2) * m(r2, c0))
    + m(r0, c2) * (m(r1, c0) * m(r2, c1) - m(r1, c1) * m(r2, c0)));
  return m(0, 0) * minor(1, 2, 3, 1, 2, 3)
    - m(0, 1) * minor(1, 2, 3, 0, 2, 3)
    + m(0, 2) * minor(1, 2, 3, 0, 1, 3)
    - m(0, 3) * minor(1, 2, 3, 0, 1, 2);
}
