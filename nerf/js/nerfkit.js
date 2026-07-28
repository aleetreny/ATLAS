/* The radiance field, run in the browser.
 *
 * The generator quantises every weight to float16 and writes them as base64
 * together with a contract that says what has to happen to them: a Fourier
 * encoding of the position, a stack of ReLU layers, a softplus density head,
 * and a colour head that is handed the viewing direction. Everything below
 * implements that contract and nothing else, and the widget checks its own
 * pixels against references the generator computed with the very same
 * quantised weights before believing them.
 *
 * The only reason this is fast enough to drag is empty space skipping: a
 * conservative occupancy grid says which cells the fitted density is nowhere
 * near zero in, and a sample that lands anywhere else never reaches the
 * network. The generator measures what that costs (it is not free, and it is
 * not zero) and the number travels in the JSON.
 */

/* base64 to Float32Array, reading the float16 the generator wrote. */
export function decodeF16(b64, n) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

/* IEEE 754 binary16 to binary32, subnormals included: a naive version flushes
   the small weights to zero and the field quietly loses its faint parts. */
export function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

export function decodeBits(b64, n) {
  const bin = atob(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  }
  return out;
}

const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

/* Y = W X^T + b for a batch of n rows. W is row major [dout][din], which is
   how torch stores a Linear, so both the weight row and the input row are
   contiguous in the inner loop. */
function linear(X, n, din, W, b, dout, Y) {
  for (let p = 0; p < n; p++) {
    const xo = p * din;
    const yo = p * dout;
    for (let o = 0; o < dout; o++) {
      const wo = o * din;
      let acc = b[o];
      for (let i = 0; i < din; i++) acc += W[wo + i] * X[xo + i];
      Y[yo + o] = acc;
    }
  }
}

function reluInPlace(A, n) {
  for (let i = 0; i < n; i++) if (A[i] < 0) A[i] = 0;
}

export class Field {
  constructor(model) {
    this.arch = model.arch;
    this.w = decodeF16(model.weights, model.n_weights);
    this.layers = {};
    model.layers.forEach((L) => {
      this.layers[L.name] = {
        W: this.w.subarray(L.w_offset, L.w_offset + L.w_size),
        b: this.w.subarray(L.b_offset, L.b_offset + L.b_size),
        out: L.shape[0],
        in: L.shape[1],
      };
    });
    this.trunk = model.layers.filter((L) => L.name.startsWith('trunk')).map((L) => this.layers[L.name]);
    this.inX = 3 + 6 * this.arch.Lx;
    this.inD = 3 + 6 * this.arch.Ld;
    this.grid = model.grid ? decodeBits(model.grid.bits, model.grid.res ** 3) : null;
    this.gridRes = model.grid ? model.grid.res : 0;
    this.evals = 0;
    this._buf = {};
  }

  buf(name, size) {
    let b = this._buf[name];
    if (!b || b.length < size) { b = new Float32Array(size); this._buf[name] = b; }
    return b;
  }

  occupied(x, y, z) {
    if (!this.grid) return true;
    const g = this.gridRes;
    const lo = this.arch.aabb.lo;
    const hi = this.arch.aabb.hi;
    let i = Math.floor(((x - lo[0]) / (hi[0] - lo[0])) * g);
    let j = Math.floor(((y - lo[1]) / (hi[1] - lo[1])) * g);
    let k = Math.floor(((z - lo[2]) / (hi[2] - lo[2])) * g);
    i = i < 0 ? 0 : i >= g ? g - 1 : i;
    j = j < 0 ? 0 : j >= g ? g - 1 : j;
    k = k < 0 ? 0 : k >= g ? g - 1 : k;
    return this.grid[(i * g + j) * g + k] === 1;
  }

  /* Density and colour for n points. pts and dirs are flat xyz triples.
     sigmaOut is length n, colOut length 3n. */
  evalPoints(n, pts, dirs, sigmaOut, colOut) {
    if (n === 0) return;
    this.evals += n;
    const { Lx, Ld, sig_scale: SS, centre, half } = this.arch;
    const inX = this.inX;
    const X = this.buf('X', n * inX);
    for (let p = 0; p < n; p++) {
      const o = p * inX;
      for (let k = 0; k < 3; k++) {
        X[o + k] = (pts[p * 3 + k] - centre[k]) / half[k];
      }
      for (let l = 0; l < Lx; l++) {
        const f = (2 ** l) * Math.PI;
        for (let k = 0; k < 3; k++) {
          X[o + 3 + l * 6 + k] = Math.sin(f * X[o + k]);
          X[o + 3 + l * 6 + 3 + k] = Math.cos(f * X[o + k]);
        }
      }
    }
    let H = this.buf('H0', n * this.trunk[0].out);
    let src = X;
    let din = inX;
    this.trunk.forEach((L, i) => {
      const dst = this.buf(`H${i % 2}`, n * L.out);
      linear(src, n, din, L.W, L.b, L.out, dst);
      reluInPlace(dst, n * L.out);
      src = dst;
      din = L.out;
      H = dst;
    });
    const sg = this.layers.sigma;
    const S = this.buf('S', n);
    linear(H, n, din, sg.W, sg.b, 1, S);
    for (let p = 0; p < n; p++) sigmaOut[p] = SS * softplus(S[p]);

    const ft = this.layers.feat;
    const F = this.buf('F', n * ft.out);
    linear(H, n, din, ft.W, ft.b, ft.out, F);

    const c1 = this.layers.col1;
    const cin = c1.in;
    const C0 = this.buf('C0', n * cin);
    for (let p = 0; p < n; p++) {
      const o = p * cin;
      for (let k = 0; k < ft.out; k++) C0[o + k] = F[p * ft.out + k];
      if (this.arch.viewdep) {
        const base = o + ft.out;
        for (let k = 0; k < 3; k++) C0[base + k] = dirs[p * 3 + k];
        for (let l = 0; l < Ld; l++) {
          const f = (2 ** l) * Math.PI;
          for (let k = 0; k < 3; k++) {
            C0[base + 3 + l * 6 + k] = Math.sin(f * dirs[p * 3 + k]);
            C0[base + 3 + l * 6 + 3 + k] = Math.cos(f * dirs[p * 3 + k]);
          }
        }
      }
    }
    const C1 = this.buf('C1', n * c1.out);
    linear(C0, n, cin, c1.W, c1.b, c1.out, C1);
    reluInPlace(C1, n * c1.out);
    const c2 = this.layers.col2;
    const C2 = this.buf('C2', n * 3);
    linear(C1, n, c1.out, c2.W, c2.b, 3, C2);
    for (let i = 0; i < n * 3; i++) colOut[i] = 1 / (1 + Math.exp(-C2[i]));
  }
}

/* Slab test against the field's own scene box. */
function rayBox(lo, hi, o, d) {
  let tn = -Infinity;
  let tf = Infinity;
  for (let k = 0; k < 3; k++) {
    const inv = 1 / (Math.abs(d[k]) < 1e-9 ? 1e-9 : d[k]);
    let a = (lo[k] - o[k]) * inv;
    let b = (hi[k] - o[k]) * inv;
    if (a > b) { const t = a; a = b; b = t; }
    if (a > tn) tn = a;
    if (b < tf) tf = b;
  }
  return [Math.max(tn, 0), tf];
}

/* Render a frame of the fitted field.
 *
 * `skip` uses the occupancy grid. `importance` spends half the sample budget
 * on a first uniform pass and redraws the other half from the weights that
 * pass produced, which is the coarse-to-fine idea from the paper reduced to
 * one network and one pixel's worth of bookkeeping. */
export function renderField(field, cam, res, steps, dirFn, {
  skip = true, importance = false, bg = [0.945, 0.953, 0.953], block = 512,
} = {}) {
  const t0 = performance.now();
  field.evals = 0;
  const out = new Float32Array(res * res * 3);
  /* the expected depth and the total weight, so the page can ask whether the
     fit put the surface in the right place and not only whether the picture
     came out the right colour */
  const depth = new Float32Array(res * res);
  const alpha = new Float32Array(res * res);
  const lo = field.arch.aabb.lo;
  const hi = field.arch.aabb.hi;
  const N = res * res;
  const S1 = importance ? Math.max(4, steps >> 1) : steps;
  const S2 = importance ? steps - S1 : 0;
  const total = S1 + S2;

  const pts = new Float32Array(block * total * 3);
  const dirs = new Float32Array(block * total * 3);
  const sig = new Float32Array(block * total);
  const col = new Float32Array(block * total * 3);
  const ts = new Float32Array(block * total);
  const live = new Int32Array(block * total);
  const rayDirs = new Float32Array(block * 3);
  const tnArr = new Float32Array(block);
  const tfArr = new Float32Array(block);

  for (let start = 0; start < N; start += block) {
    const nb = Math.min(block, N - start);
    let nHit = 0;
    const hitIdx = [];
    for (let b = 0; b < nb; b++) {
      const pix = start + b;
      const d = dirFn(Math.floor(pix / res), pix % res);
      rayDirs[b * 3] = d[0];
      rayDirs[b * 3 + 1] = d[1];
      rayDirs[b * 3 + 2] = d[2];
      const [tn, tf] = rayBox(lo, hi, cam.pos, d);
      tnArr[b] = tn;
      tfArr[b] = tf;
      if (tf > tn) { hitIdx.push(b); nHit++; }
      else {
        out[pix * 3] = bg[0];
        out[pix * 3 + 1] = bg[1];
        out[pix * 3 + 2] = bg[2];
      }
    }
    if (!nHit) continue;

    /* --- pass one: equal steps, exactly the quadrature the generator used */
    let nLive = 0;
    for (let h = 0; h < nHit; h++) {
      const b = hitIdx[h];
      const tn = tnArr[b];
      const dt = (tfArr[b] - tn) / S1;
      for (let s = 0; s < S1; s++) {
        const t = tn + (s + 0.5) * dt;
        const slot = h * total + s;
        ts[slot] = t;
        const x = cam.pos[0] + t * rayDirs[b * 3];
        const y = cam.pos[1] + t * rayDirs[b * 3 + 1];
        const z = cam.pos[2] + t * rayDirs[b * 3 + 2];
        sig[slot] = 0;
        if (!skip || field.occupied(x, y, z)) {
          pts[nLive * 3] = x;
          pts[nLive * 3 + 1] = y;
          pts[nLive * 3 + 2] = z;
          dirs[nLive * 3] = rayDirs[b * 3];
          dirs[nLive * 3 + 1] = rayDirs[b * 3 + 1];
          dirs[nLive * 3 + 2] = rayDirs[b * 3 + 2];
          live[nLive] = slot;
          nLive++;
        }
      }
    }
    const sigTmp = new Float32Array(nLive);
    const colTmp = new Float32Array(nLive * 3);
    field.evalPoints(nLive, pts, dirs, sigTmp, colTmp);
    for (let i = 0; i < nLive; i++) {
      const slot = live[i];
      sig[slot] = sigTmp[i];
      col[slot * 3] = colTmp[i * 3];
      col[slot * 3 + 1] = colTmp[i * 3 + 1];
      col[slot * 3 + 2] = colTmp[i * 3 + 2];
    }

    /* --- pass two: redraw from the weights the first pass produced */
    let counts = null;
    if (importance) {
      counts = new Int32Array(nHit).fill(S1);
      let n2 = 0;
      for (let h = 0; h < nHit; h++) {
        const b = hitIdx[h];
        const tn = tnArr[b];
        const dt = (tfArr[b] - tn) / S1;
        const w = new Float32Array(S1);
        let T = 1;
        let sum = 0;
        for (let s = 0; s < S1; s++) {
          const a = 1 - Math.exp(-sig[h * total + s] * dt);
          w[s] = T * a;
          sum += w[s];
          T *= (1 - a + 1e-12);
        }
        if (sum < 1e-6) continue;
        let acc = 0;
        let s = 0;
        for (let k = 0; k < S2; k++) {
          const u = ((k + 0.5) / S2) * sum;
          while (s < S1 - 1 && acc + w[s] < u) { acc += w[s]; s++; }
          const frac = w[s] > 1e-12 ? (u - acc) / w[s] : 0.5;
          const t = tn + (s + frac) * dt;
          const slot = h * total + counts[h];
          counts[h]++;
          ts[slot] = t;
          const x = cam.pos[0] + t * rayDirs[b * 3];
          const y = cam.pos[1] + t * rayDirs[b * 3 + 1];
          const z = cam.pos[2] + t * rayDirs[b * 3 + 2];
          sig[slot] = 0;
          if (!skip || field.occupied(x, y, z)) {
            pts[n2 * 3] = x;
            pts[n2 * 3 + 1] = y;
            pts[n2 * 3 + 2] = z;
            dirs[n2 * 3] = rayDirs[b * 3];
            dirs[n2 * 3 + 1] = rayDirs[b * 3 + 1];
            dirs[n2 * 3 + 2] = rayDirs[b * 3 + 2];
            live[n2] = slot;
            n2++;
          }
        }
      }
      const s2 = new Float32Array(n2);
      const c2 = new Float32Array(n2 * 3);
      field.evalPoints(n2, pts, dirs, s2, c2);
      for (let i = 0; i < n2; i++) {
        const slot = live[i];
        sig[slot] = s2[i];
        col[slot * 3] = c2[i * 3];
        col[slot * 3 + 1] = c2[i * 3 + 1];
        col[slot * 3 + 2] = c2[i * 3 + 2];
      }
    }

    /* --- composite */
    for (let h = 0; h < nHit; h++) {
      const b = hitIdx[h];
      const pix = start + b;
      const tn = tnArr[b];
      const tf = tfArr[b];
      const m = importance ? counts[h] : S1;
      let order = null;
      if (importance) {
        order = Array.from({ length: m }, (_, i) => i).sort((p, q) => ts[h * total + p] - ts[h * total + q]);
      }
      let T = 1;
      let r = 0;
      let g = 0;
      let bl = 0;
      let acc = 0;
      let dsum = 0;
      const dtEven = (tf - tn) / S1;
      for (let s = 0; s < m; s++) {
        const slot = h * total + (order ? order[s] : s);
        let dt = dtEven;
        if (order) {
          const tCur = ts[slot];
          const tNext = s + 1 < m ? ts[h * total + order[s + 1]] : tf;
          dt = Math.max(tNext - tCur, 1e-6);
        }
        const a = 1 - Math.exp(-sig[slot] * dt);
        const wgt = T * a;
        r += wgt * col[slot * 3];
        g += wgt * col[slot * 3 + 1];
        bl += wgt * col[slot * 3 + 2];
        acc += wgt;
        dsum += wgt * ts[slot];
        T *= (1 - a + 1e-10);
      }
      /* the background gets whatever weight the field did not claim, written
         the same way the generator writes it so the two agree bit for bit */
      out[pix * 3] = r + (1 - acc) * bg[0];
      out[pix * 3 + 1] = g + (1 - acc) * bg[1];
      out[pix * 3 + 2] = bl + (1 - acc) * bg[2];
      alpha[pix] = acc;
      depth[pix] = acc > 1e-6 ? dsum / acc : 0;
    }
  }
  return { rgb: out, depth, alpha, ms: performance.now() - t0, evals: field.evals };
}
