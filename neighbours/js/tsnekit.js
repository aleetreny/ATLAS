/* t-SNE in the browser, and the two statistics the article uses to interrogate
 * its output.
 *
 * Everything here mirrors generate_neighbours_data.py line for line, and every
 * place the two could have drifted has been pinned:
 *
 * - the starting configuration is drawn by a linear congruential generator
 *   written out identically at both ends, and shipped in the file as well;
 * - the two synthetic datasets are drawn by the same generator through a Box
 *   and Muller transform, so a quarter of a megabyte of noise does not have to
 *   be downloaded in order to be told there is nothing in it;
 * - the bandwidth search, the gains rule, the two momenta and the iteration at
 *   which early exaggeration stops are the published ones, and none of them is
 *   decided by a convergence test.
 *
 * The result is that the browser and the generator take the same trajectory and
 * can be compared to eight decimals rather than to within a shrug.
 */
import { zeros, pairwise } from '../../assets/js/embed.js';

/* ------------------------------------------------------------ the generator */
export function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function seededBlock(n, k, seed) {
  const next = lcg(seed);
  const Q = zeros(n, k);
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) Q[i][j] = next() - 0.5;
  return Q;
}

/* Standard normals by Box and Muller, two uniforms per entry, row major. */
export function seededNormal(n, d, seed, scale = 1) {
  const next = lcg(seed);
  const out = zeros(n, d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const u1 = Math.max(next(), 1e-12);
      const u2 = next();
      out[i][j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
    }
  }
  return out;
}

/* --------------------------------------------------------- squared distances */
export function sqDists(X) {
  const n = X.length;
  const p = X[0].length;
  const norms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < p; j++) s += X[i][j] * X[i][j];
    norms[i] = s;
  }
  const D = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dot = 0;
      for (let t = 0; t < p; t++) dot += X[i][t] * X[j][t];
      const v = Math.max(norms[i] + norms[j] - 2 * dot, 0);
      D[i][j] = v;
      D[j][i] = v;
    }
  }
  return D;
}

/* -------------------------------------------------------- the neighbour graph */
/* One bandwidth per point, solved so that the neighbour distribution it induces
 * has exactly the requested perplexity. Perplexity is two to the power of the
 * entropy, which is to say the effective number of neighbours, and solving for
 * it point by point is what lets one number mean the same thing in a dense
 * region and a sparse one. */
export function conditionalP(D2, perplexity, { tol = 1e-10, iters = 60 } = {}) {
  const n = D2.length;
  const P = zeros(n, n);
  const betas = new Float64Array(n);
  const logU = Math.log(perplexity);
  const row = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let lo = -Infinity;
    let hi = Infinity;
    let beta = 1;
    for (let it = 0; it < iters; it++) {
      let ssum = 0;
      let dsum = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) { row[j] = 0; continue; }
        const v = Math.exp(-D2[i][j] * beta);
        row[j] = v;
        ssum += v;
        dsum += D2[i][j] * v;
      }
      let H;
      if (ssum <= 1e-300) {
        H = 0;
      } else {
        H = Math.log(ssum) + (beta * dsum) / ssum;
      }
      const diff = H - logU;
      if (Math.abs(diff) < tol) break;
      if (diff > 0) {
        lo = beta;
        beta = hi === Infinity ? beta * 2 : (beta + hi) / 2;
      } else {
        hi = beta;
        beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2;
      }
    }
    let ssum = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) { row[j] = 0; continue; }
      row[j] = Math.exp(-D2[i][j] * beta);
      ssum += row[j];
    }
    betas[i] = beta;
    for (let j = 0; j < n; j++) P[i][j] = ssum > 1e-300 ? row[j] / ssum : 0;
  }
  return { P, betas };
}

export function jointP(X, perplexity) {
  const D2 = sqDists(X);
  const { P, betas } = conditionalP(D2, perplexity);
  const n = P.length;
  const J = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) J[i][j] = Math.max((P[i][j] + P[j][i]) / (2 * n), 1e-12);
  }
  return { P: J, betas, D2 };
}

/* ----------------------------------------------------------------- the descent */
export function makeRun(P, Y0, cfg) {
  const n = P.length;
  const Y = Y0.map((r) => Float64Array.from(r));
  const dY = zeros(n, 2);
  const gains = zeros(n, 2);
  for (let i = 0; i < n; i++) { gains[i][0] = 1; gains[i][1] = 1; }
  return { P, Y, dY, gains, it: 0, cfg, n, kl: NaN };
}

/* One iteration, written so that a widget can run fifty of them between two
 * frames without the page going away. */
export function stepRun(run, count = 1) {
  const { P, cfg } = run;
  const n = run.n;
  const gx = new Float64Array(n);
  const gy = new Float64Array(n);
  const num = new Float64Array(n * n);
  for (let t = 0; t < count && run.it < cfg.iters; t++) {
    /* One pass over the upper triangle, accumulating into both ends. The
       obvious version walks all n^2 pairs twice, once to build the kernel and
       once for the gradient, and at five hundred points that is the difference
       between a widget that animates and one that freezes the page. */
    let total = 0;
    for (let i = 0; i < n; i++) {
      const yi0 = run.Y[i][0];
      const yi1 = run.Y[i][1];
      const base = i * n;
      for (let j = i + 1; j < n; j++) {
        const dx = yi0 - run.Y[j][0];
        const dy = yi1 - run.Y[j][1];
        const v = 1 / (1 + dx * dx + dy * dy);
        num[base + j] = v;
        total += v;
      }
    }
    total *= 2;
    const exagg = run.it < cfg.exaggIters ? cfg.exagg : 1;
    const mom = run.it < cfg.exaggIters ? 0.5 : 0.8;
    gx.fill(0);
    gy.fill(0);
    let kl = 0;
    for (let i = 0; i < n; i++) {
      const yi0 = run.Y[i][0];
      const yi1 = run.Y[i][1];
      const base = i * n;
      const Pi = P[i];
      for (let j = i + 1; j < n; j++) {
        const v = num[base + j];
        const q = v / total > 1e-12 ? v / total : 1e-12;
        const pq = (Pi[j] * exagg - q) * v;
        const dx = yi0 - run.Y[j][0];
        const dy = yi1 - run.Y[j][1];
        gx[i] += pq * dx;
        gy[i] += pq * dy;
        gx[j] -= pq * dx;
        gy[j] -= pq * dy;
        kl += 2 * Pi[j] * Math.log(Pi[j] / q);
      }
    }
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
      const g0 = 4 * gx[i];
      const g1 = 4 * gy[i];
      run.gains[i][0] = Math.max(
        Math.sign(g0) !== Math.sign(run.dY[i][0]) ? run.gains[i][0] + 0.2 : run.gains[i][0] * 0.8,
        0.01);
      run.gains[i][1] = Math.max(
        Math.sign(g1) !== Math.sign(run.dY[i][1]) ? run.gains[i][1] + 0.2 : run.gains[i][1] * 0.8,
        0.01);
      run.dY[i][0] = mom * run.dY[i][0] - cfg.lr * run.gains[i][0] * g0;
      run.dY[i][1] = mom * run.dY[i][1] - cfg.lr * run.gains[i][1] * g1;
      run.Y[i][0] += run.dY[i][0];
      run.Y[i][1] += run.dY[i][1];
      mx += run.Y[i][0] / n;
      my += run.Y[i][1] / n;
    }
    for (let i = 0; i < n; i++) {
      run.Y[i][0] -= mx;
      run.Y[i][1] -= my;
    }
    run.kl = kl;
    run.it += 1;
  }
  return run;
}

/* The divergence at the current configuration, without the exaggeration, which
 * is the number the article quotes. */
export function klNow(P, Y) {
  const n = P.length;
  let total = 0;
  const num = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = Y[i][0] - Y[j][0];
      const dy = Y[i][1] - Y[j][1];
      const v = 1 / (1 + dx * dx + dy * dy);
      num[i][j] = v;
      num[j][i] = v;
      total += 2 * v;
    }
  }
  let kl = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const q = Math.max(num[i][j] / total, 1e-12);
      kl += P[i][j] * Math.log(P[i][j] / q);
    }
  }
  return kl;
}

export function runToEnd(P, Y0, cfg) {
  const run = makeRun(P, Y0, cfg);
  stepRun(run, cfg.iters);
  return run;
}

/* -------------------------------------------------------- reading the picture */
/* A small DBSCAN, used only to make "how many clumps would a reader count"
 * mechanical. Its one parameter is pinned in the generator from both sides: it
 * must call a plain two-dimensional gaussian one clump and a t-SNE of three
 * separated blobs three. */
export function dbscan(Y, eps, minPts) {
  const n = Y.length;
  const label = new Int32Array(n).fill(-2);         // -2 unvisited, -1 noise
  const eps2 = eps * eps;
  const near = (i) => {
    const out = [];
    for (let j = 0; j < n; j++) {
      const dx = Y[i][0] - Y[j][0];
      const dy = Y[i][1] - Y[j][1];
      if (dx * dx + dy * dy <= eps2) out.push(j);
    }
    return out;
  };
  let c = 0;
  for (let i = 0; i < n; i++) {
    if (label[i] !== -2) continue;
    const seeds = near(i);
    if (seeds.length < minPts) {
      label[i] = -1;
      continue;
    }
    label[i] = c;
    for (let t = 0; t < seeds.length; t++) {
      const q = seeds[t];
      if (label[q] === -1) label[q] = c;
      if (label[q] !== -2) continue;
      label[q] = c;
      const more = near(q);
      if (more.length >= minPts) for (const m of more) seeds.push(m);
    }
    c += 1;
  }
  return { label, count: c };
}

export function countBlobs(Y, mult, minPts = 8) {
  const D = pairwise(Y);
  const n = Y.length;
  const tenth = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = Array.from(D[i]);
    row.splice(i, 1);
    row.sort((a, b) => a - b);
    tenth[i] = row[9];
  }
  const sorted = Array.from(tenth).sort((a, b) => a - b);
  const median = sorted[Math.floor(n / 2)];
  const { label, count } = dbscan(Y, mult * median, minPts);
  let noise = 0;
  for (let i = 0; i < n; i++) if (label[i] === -1) noise += 1;
  return { count, noise: noise / n, label, eps: mult * median };
}

/* Clustering tendency is quoted on this page but not computed here: the
 * generator's version draws its uniform sample with numpy's generator, which a
 * browser cannot reproduce, and a statistic that cannot be checked has no
 * business being recomputed in two places. The article cites the file's number
 * and says where it came from. */
