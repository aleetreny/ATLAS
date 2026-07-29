/* A GAN small enough to train in a browser tab, written to be the same
 * arithmetic as the numpy version in src/utils/generate_gan_data.py.
 *
 * Same random number generator (a linear congruential stream plus Box-Muller,
 * because numpy's generator does not exist here), same initialisation order,
 * same layer sizes, same Adam, same loop. The generator exports a frozen
 * forward pass and a two hundred step reference trajectory, and main.js checks
 * both, so "watch a GAN train" on this page is the same experiment the
 * article measured rather than a lookalike.
 */

export class LCG {
  constructor(seed) {
    this.s = seed >>> 0;
  }

  u() {
    this.s = (Math.imul(this.s, 1664525) + 1013904223) >>> 0;
    return this.s / 4294967296;
  }

  uniform(n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.u();
    return out;
  }

  normal(n) {
    const out = new Float64Array(n);
    let i = 0;
    while (i < n) {
      const u1 = Math.max(this.u(), 1e-12);
      const u2 = this.u();
      const r = Math.sqrt(-2 * Math.log(u1));
      out[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
      i += 2;
    }
    return out;
  }
}

/* Weights row major over outputs, inner loop over inputs: the same order the
 * numpy side draws them in, which is the only reason the two initialisations
 * are the same numbers. */
function heInit(rng, nin, nout) {
  const lim = Math.sqrt(6 / (nin + nout));
  const w = new Float64Array(nout * nin);
  for (let o = 0; o < nout; o++) {
    for (let i = 0; i < nin; i++) w[o * nin + i] = (rng.u() * 2 - 1) * lim;
  }
  return { w, b: new Float64Array(nout) };
}

export class MiniMLP {
  constructor(rng, sizes, act) {
    this.sizes = sizes;
    this.act = act;
    this.W = [];
    this.b = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const { w, b } = heInit(rng, sizes[i], sizes[i + 1]);
      this.W.push(w);
      this.b.push(b);
    }
    this.m = this.W.map((w) => new Float64Array(w.length))
      .concat(this.b.map((b) => new Float64Array(b.length)));
    this.v = this.W.map((w) => new Float64Array(w.length))
      .concat(this.b.map((b) => new Float64Array(b.length)));
    this.t = 0;
  }

  actF(z) {
    return this.act === 'tanh' ? Math.tanh(z) : (z > 0 ? z : 0.2 * z);
  }

  actD(z) {
    if (this.act === 'tanh') {
      const t = Math.tanh(z);
      return 1 - t * t;
    }
    return z > 0 ? 1 : 0.2;
  }

  /* x: (n, sizes[0]) flat. Returns the activations and pre-activations of
     every layer, because the backward pass needs both. */
  forward(x, n) {
    const a = [x];
    const zs = [];
    for (let l = 0; l < this.W.length; l++) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      const z = new Float64Array(n * nout);
      const prev = a[a.length - 1];
      for (let s = 0; s < n; s++) {
        for (let o = 0; o < nout; o++) {
          let acc = this.b[l][o];
          const base = o * nin;
          const row = s * nin;
          for (let i = 0; i < nin; i++) acc += prev[row + i] * this.W[l][base + i];
          z[s * nout + o] = acc;
        }
      }
      zs.push(z);
      if (l < this.W.length - 1) {
        const h = new Float64Array(z.length);
        for (let i = 0; i < z.length; i++) h[i] = this.actF(z[i]);
        a.push(h);
      } else {
        a.push(z);
      }
    }
    return { a, zs };
  }

  /* dout: (n, last) flat. Returns per layer gradients and the gradient with
     respect to the INPUT, which is the only channel through which a
     discriminator can say anything to a generator. */
  backward(a, zs, dout, n) {
    const gw = this.W.map((w) => new Float64Array(w.length));
    const gb = this.b.map((b) => new Float64Array(b.length));
    let d = dout;
    for (let l = this.W.length - 1; l >= 0; l--) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      for (let o = 0; o < nout; o++) {
        let sb = 0;
        for (let s = 0; s < n; s++) {
          const dv = d[s * nout + o];
          sb += dv;
          const base = o * nin;
          const row = s * nin;
          for (let i = 0; i < nin; i++) gw[l][base + i] += dv * a[l][row + i];
        }
        gb[l][o] = sb;
      }
      const nxt = new Float64Array(n * nin);
      for (let s = 0; s < n; s++) {
        for (let i = 0; i < nin; i++) {
          let acc = 0;
          for (let o = 0; o < nout; o++) acc += d[s * nout + o] * this.W[l][o * nin + i];
          nxt[s * nin + i] = l > 0 ? acc * this.actD(zs[l - 1][s * nin + i]) : acc;
        }
      }
      d = nxt;
    }
    return { gw, gb, dx: d };
  }

  step(gw, gb, lr, b1 = 0.5, b2 = 0.999, eps = 1e-8) {
    this.t += 1;
    const gs = gw.concat(gb);
    const ps = this.W.concat(this.b);
    const bc1 = 1 - b1 ** this.t;
    const bc2 = 1 - b2 ** this.t;
    for (let k = 0; k < gs.length; k++) {
      const g = gs[k];
      const p = ps[k];
      for (let i = 0; i < g.length; i++) {
        this.m[k][i] = b1 * this.m[k][i] + (1 - b1) * g[i];
        this.v[k][i] = b2 * this.v[k][i] + (1 - b2) * g[i] * g[i];
        p[i] -= lr * (this.m[k][i] / bc1) / (Math.sqrt(this.v[k][i] / bc2) + eps);
      }
    }
  }
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function ringCentres(k, r) {
  const out = [];
  for (let i = 0; i < k; i++) {
    const a = (i * 2 * Math.PI) / k;
    out.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return out;
}

/* The mode counter, with the same two thresholds the generator calibrated:
 * a point belongs to a mode if it is within four standard deviations of the
 * centre, and a mode is covered if half a percent of the sample belongs to it.
 * Real samples must score eight and a point mass must score one, which is
 * checked in the generator and checked again here on the exported clouds. */
export function modeReport(pts, centres, sd, { tolSd = 4, minShare = 0.005 } = {}) {
  const counts = new Array(centres.length).fill(0);
  let good = 0;
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    let bestD = Infinity;
    let who = 0;
    for (let c = 0; c < centres.length; c++) {
      const dx = pts[i][0] - centres[c][0];
      const dy = pts[i][1] - centres[c][1];
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < bestD) { bestD = d; who = c; }
    }
    if (bestD <= tolSd * sd) { counts[who] += 1; good += 1; }
  }
  const covered = counts.filter((c) => c >= minShare * n).length;
  const total = counts.reduce((s, c) => s + c, 0);
  return { covered, quality: good / n, counts, maxShare: total ? Math.max(...counts) / total : 0 };
}

/* One GAN, trainable a chunk of steps at a time so the page stays responsive
 * without hanging anything on an animation frame (a frame is not composed when
 * the browser panel is hidden, and a control that waits for one looks broken
 * to everything that checks this page). */
export class LiveGAN {
  constructor({ seed = 7, h = 24, batch = 64, arm = 'ns', k = 8, r = 2, sd = 0.05, lr = 2e-3 }) {
    this.rng = new LCG(seed);
    this.G = new MiniMLP(this.rng, [2, h, h, 2], 'tanh');
    this.D = new MiniMLP(this.rng, [2, h, h, 1], 'leaky');
    this.centres = ringCentres(k, r);
    Object.assign(this, { batch, arm, k, r, sd, lr });
    this.it = 0;
    this.losses = [];
  }

  sampleReal(n) {
    const u = this.rng.uniform(n);
    const g = this.rng.normal(n * 2);
    const out = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const c = this.centres[Math.floor(u[i] * this.k) % this.k];
      out[i * 2] = c[0] + this.sd * g[i * 2];
      out[i * 2 + 1] = c[1] + this.sd * g[i * 2 + 1];
    }
    return out;
  }

  stepOnce() {
    const B = this.batch;
    const real = this.sampleReal(B);
    const z = this.rng.normal(B * 2);
    const gf = this.G.forward(z, B);
    const fake = gf.a[gf.a.length - 1];

    const rf = this.D.forward(real, B);
    let ff = this.D.forward(fake, B);
    const pr = new Float64Array(B);
    const pf = new Float64Array(B);
    let ld = 0;
    for (let i = 0; i < B; i++) {
      pr[i] = sigmoid(rf.a[rf.a.length - 1][i]);
      pf[i] = sigmoid(ff.a[ff.a.length - 1][i]);
      ld -= (Math.log(Math.max(pr[i], 1e-12)) + Math.log(Math.max(1 - pf[i], 1e-12))) / B;
    }
    const dr = new Float64Array(B);
    const df = new Float64Array(B);
    for (let i = 0; i < B; i++) {
      dr[i] = (pr[i] - 1) / B;
      df[i] = pf[i] / B;
    }
    const g1 = this.D.backward(rf.a, rf.zs, dr, B);
    const g2 = this.D.backward(ff.a, ff.zs, df, B);
    const gw = g1.gw.map((w, i) => { const o = new Float64Array(w.length); for (let k = 0; k < w.length; k++) o[k] = w[k] + g2.gw[i][k]; return o; });
    const gb = g1.gb.map((b, i) => { const o = new Float64Array(b.length); for (let k = 0; k < b.length; k++) o[k] = b[k] + g2.gb[i][k]; return o; });
    this.D.step(gw, gb, this.lr);

    ff = this.D.forward(fake, B);
    const seed = new Float64Array(B);
    let lg = 0;
    for (let i = 0; i < B; i++) {
      const p = sigmoid(ff.a[ff.a.length - 1][i]);
      if (this.arm === 'sat') {
        lg += Math.log(Math.max(1 - p, 1e-12)) / B;
        seed[i] = -p / B;
      } else {
        lg -= Math.log(Math.max(p, 1e-12)) / B;
        seed[i] = (p - 1) / B;
      }
    }
    const back = this.D.backward(ff.a, ff.zs, seed, B);
    const gg = this.G.backward(gf.a, gf.zs, back.dx, B);
    this.G.step(gg.gw, gg.gb, this.lr);

    this.it += 1;
    this.losses.push([ld, lg]);
    return { ld, lg };
  }

  run(n) {
    for (let i = 0; i < n; i++) this.stepOnce();
  }

  /* A fresh cloud of samples. Uses its own stream so that drawing the picture
     does not change the training trajectory. */
  sample(n, seed = 99) {
    const rng = new LCG(seed);
    const z = rng.normal(n * 2);
    const out = this.G.forward(z, n).a;
    const flat = out[out.length - 1];
    const pts = [];
    for (let i = 0; i < n; i++) pts.push([flat[i * 2], flat[i * 2 + 1]]);
    return pts;
  }

  /* What the discriminator currently believes, on a grid, which is the only
     way to see the surface the generator is climbing. */
  surface(lo, hi, n) {
    const xs = new Float64Array(n * n * 2);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        xs[(r * n + c) * 2] = lo + ((hi - lo) * c) / (n - 1);
        xs[(r * n + c) * 2 + 1] = lo + ((hi - lo) * r) / (n - 1);
      }
    }
    const out = this.D.forward(xs, n * n).a;
    const flat = out[out.length - 1];
    const grid = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) grid[i] = sigmoid(flat[i]);
    return grid;
  }
}
