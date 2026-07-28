/* ATLAS shared maths for module 2.2: everything the dimensionality reduction
 * articles have to run in the browser rather than download.
 *
 * The rule these articles work under is that a figure the reader can see is a
 * figure the reader's own machine computed, checked at load time against what
 * scikit-learn said. That needs a real symmetric eigensolver, real shortest
 * paths and a real isotonic regression, none of which a chart library provides.
 *
 * Three conventions, all of which have cost time somewhere in this repository:
 *
 * - Everything statistical is POPULATION form (divide by n, never n-1). numpy
 *   defaults to n-1 and a browser has no such notion, so mixing the two puts a
 *   whitening or a covariance a few parts in a thousand away from its reference,
 *   which looks exactly like an implementation bug and is not.
 * - Matrices are arrays of Float64Array rows. Plain nested arrays work but are
 *   two to four times slower on the inner loops, and these run inside slider
 *   callbacks.
 * - Any iterative solver here is deterministic given its starting point, so the
 *   generator can run the identical algorithm and the two can be compared to
 *   eight decimals instead of to within some hand-waved tolerance.
 */

/* ------------------------------------------------------------------ matrices */
export function toRows(A) {
  return A.map((r) => (r instanceof Float64Array ? r : Float64Array.from(r)));
}

export function zeros(n, m) {
  return Array.from({ length: n }, () => new Float64Array(m));
}

export function colMeans(X) {
  const n = X.length;
  const p = X[0].length;
  const m = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) m[j] += X[i][j];
  for (let j = 0; j < p; j++) m[j] /= n;
  return m;
}

export function centre(X) {
  const m = colMeans(X);
  return X.map((row) => {
    const out = new Float64Array(row.length);
    for (let j = 0; j < row.length; j++) out[j] = row[j] - m[j];
    return out;
  });
}

/* Population covariance of an n x p matrix, already centred or not. */
export function covariance(X) {
  const Xc = centre(X);
  const n = Xc.length;
  const p = Xc[0].length;
  const S = zeros(p, p);
  for (let i = 0; i < n; i++) {
    const r = Xc[i];
    for (let a = 0; a < p; a++) {
      const va = r[a];
      if (va === 0) continue;
      for (let b = a; b < p; b++) S[a][b] += va * r[b];
    }
  }
  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      S[a][b] /= n;
      S[b][a] = S[a][b];
    }
  }
  return S;
}

/* ------------------------------------------------------------ eigendecomposition */
/* Householder reduction to tridiagonal form followed by the implicit QL
 * algorithm: the EISPACK pair, which is what LAPACK's dsyev does underneath for
 * a symmetric matrix. Cyclic Jacobi is shorter to write and about ten times
 * slower, which matters when a 178 by 178 matrix has to come back inside a
 * button press. Returns eigenvalues DESCENDING with matching eigenvector
 * columns, which is the order every article here draws them in. */
export function symEigen(Ain) {
  const n = Ain.length;
  const V = Ain.map((r) => Float64Array.from(r));
  const d = new Float64Array(n);
  const e = new Float64Array(n);

  for (let i = n - 1; i > 0; i--) {
    const l = i - 1;
    let h = 0;
    let scale = 0;
    if (l > 0) {
      for (let k = 0; k <= l; k++) scale += Math.abs(V[i][k]);
      if (scale === 0) {
        e[i] = V[i][l];
      } else {
        for (let k = 0; k <= l; k++) {
          V[i][k] /= scale;
          h += V[i][k] * V[i][k];
        }
        let f = V[i][l];
        let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
        e[i] = scale * g;
        h -= f * g;
        V[i][l] = f - g;
        f = 0;
        for (let j = 0; j <= l; j++) {
          V[j][i] = V[i][j] / h;
          g = 0;
          for (let k = 0; k <= j; k++) g += V[j][k] * V[i][k];
          for (let k = j + 1; k <= l; k++) g += V[k][j] * V[i][k];
          e[j] = g / h;
          f += e[j] * V[i][j];
        }
        const hh = f / (h + h);
        for (let j = 0; j <= l; j++) {
          f = V[i][j];
          e[j] = g = e[j] - hh * f;
          for (let k = 0; k <= j; k++) V[j][k] -= f * e[k] + g * V[i][k];
        }
      }
    } else {
      e[i] = V[i][l];
    }
    d[i] = h;
  }
  d[0] = 0;
  e[0] = 0;
  for (let i = 0; i < n; i++) {
    if (d[i] !== 0) {
      for (let j = 0; j < i; j++) {
        let g = 0;
        for (let k = 0; k < i; k++) g += V[i][k] * V[k][j];
        for (let k = 0; k < i; k++) V[k][j] -= g * V[k][i];
      }
    }
    d[i] = V[i][i];
    V[i][i] = 1;
    for (let j = 0; j < i; j++) {
      V[j][i] = 0;
      V[i][j] = 0;
    }
  }

  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  for (let l = 0; l < n; l++) {
    let iter = 0;
    let m;
    do {
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= Number.EPSILON * dd) break;
      }
      if (m !== l) {
        if (iter++ === 60) break;
        let g = (d[l + 1] - d[l]) / (2 * e[l]);
        let r = Math.hypot(g, 1);
        g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1;
        let c = 1;
        let p = 0;
        for (let i = m - 1; i >= l; i--) {
          let f = s * e[i];
          const b = c * e[i];
          r = Math.hypot(f, g);
          e[i + 1] = r;
          if (r === 0) {
            d[i + 1] -= p;
            e[m] = 0;
            break;
          }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          for (let k = 0; k < n; k++) {
            f = V[k][i + 1];
            V[k][i + 1] = s * V[k][i] + c * f;
            V[k][i] = c * V[k][i] - s * f;
          }
        }
        if (r === 0 && m - 1 >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0;
      }
    } while (m !== l);
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => d[b] - d[a]);
  const values = Float64Array.from(order, (i) => d[i]);
  const vectors = Array.from({ length: n }, (_, r) =>
    Float64Array.from(order, (i) => V[r][i]));
  return { values, vectors };
}

/* Sign convention: an eigenvector is only defined up to a sign, so pin it by
 * the entry of largest magnitude. Without this two runs of the same code draw
 * mirror images of each other and nothing is wrong. */
export function fixSigns(cols) {
  const n = cols.length;
  const k = cols[0].length;
  for (let j = 0; j < k; j++) {
    let best = 0;
    let bi = 0;
    for (let i = 0; i < n; i++) {
      if (Math.abs(cols[i][j]) > best) {
        best = Math.abs(cols[i][j]);
        bi = i;
      }
    }
    if (cols[bi][j] < 0) for (let i = 0; i < n; i++) cols[i][j] = -cols[i][j];
  }
  return cols;
}

/* ------------------------------------------------------------------ distances */
const METRICS = {
  euclidean: (a, b) => {
    let s = 0;
    for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2;
    return Math.sqrt(s);
  },
  cityblock: (a, b) => {
    let s = 0;
    for (let j = 0; j < a.length; j++) s += Math.abs(a[j] - b[j]);
    return s;
  },
  chebyshev: (a, b) => {
    let s = 0;
    for (let j = 0; j < a.length; j++) s = Math.max(s, Math.abs(a[j] - b[j]));
    return s;
  },
  cosine: (a, b) => {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let j = 0; j < a.length; j++) {
      dot += a[j] * b[j];
      na += a[j] * a[j];
      nb += b[j] * b[j];
    }
    const den = Math.sqrt(na * nb);
    return den === 0 ? 0 : 1 - dot / den;
  },
};
/* Correlation distance is cosine on the row-centred vectors, and writing it
 * that way rather than as its own loop is the reason the two come out with the
 * same rounding. */
METRICS.correlation = (a, b) => {
  const ca = new Float64Array(a.length);
  const cb = new Float64Array(b.length);
  let ma = 0;
  let mb = 0;
  for (let j = 0; j < a.length; j++) {
    ma += a[j];
    mb += b[j];
  }
  ma /= a.length;
  mb /= b.length;
  for (let j = 0; j < a.length; j++) {
    ca[j] = a[j] - ma;
    cb[j] = b[j] - mb;
  }
  return METRICS.cosine(ca, cb);
};
METRICS['sqrt cityblock'] = (a, b) => Math.sqrt(METRICS.cityblock(a, b));

export function pairwise(X, metric = 'euclidean') {
  const f = METRICS[metric];
  if (!f) throw new Error(`unknown metric ${metric}`);
  const n = X.length;
  const D = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = f(X[i], X[j]);
      D[i][j] = v;
      D[j][i] = v;
    }
  }
  return D;
}

/* B = -0.5 J D^2 J. Three lines, and the whole of classical scaling. */
export function doubleCentre(D) {
  const n = D.length;
  const rowm = new Float64Array(n);
  let grand = 0;
  const D2 = zeros(n, n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      const v = D[i][j] * D[i][j];
      D2[i][j] = v;
      s += v;
    }
    rowm[i] = s / n;
    grand += s;
  }
  grand /= n * n;
  const B = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) B[i][j] = -0.5 * (D2[i][j] - rowm[i] - rowm[j] + grand);
  }
  return B;
}

/* Classical scaling: coordinates and the whole spectrum, because the negative
 * end of that spectrum is the only thing that says whether the dissimilarity
 * came from any Euclidean configuration at all. */
export function classicalMDS(D, k = 2) {
  const { values, vectors } = symEigen(doubleCentre(D));
  const n = D.length;
  const Z = zeros(n, k);
  for (let j = 0; j < k; j++) {
    const s = Math.sqrt(Math.max(values[j], 0));
    for (let i = 0; i < n; i++) Z[i][j] = vectors[i][j] * s;
  }
  return { coords: fixSigns(Z), values };
}

export function negativeMass(values) {
  let neg = 0;
  let tot = 0;
  for (let i = 0; i < values.length; i++) {
    tot += Math.abs(values[i]);
    if (values[i] < 0) neg += -values[i];
  }
  return tot === 0 ? 0 : neg / tot;
}

/* Kruskal's stress-1 between a target dissimilarity and a configuration. */
export function stress1(D, Z) {
  const d = pairwise(Z);
  let num = 0;
  let den = 0;
  for (let i = 0; i < D.length; i++) {
    for (let j = i + 1; j < D.length; j++) {
      num += (D[i][j] - d[i][j]) ** 2;
      den += d[i][j] * d[i][j];
    }
  }
  return Math.sqrt(num / den);
}

/* ------------------------------------------------------------------ procrustes */
/* Rotate, reflect, scale and shift B onto A and report what is left, as a share
 * of A's own scatter. Two dimensions only, which is all these articles need. */
export function procrustes(A, B) {
  const n = A.length;
  const ac = [0, 0];
  const bc = [0, 0];
  for (let i = 0; i < n; i++) {
    ac[0] += A[i][0];
    ac[1] += A[i][1];
    bc[0] += B[i][0];
    bc[1] += B[i][1];
  }
  ac[0] /= n; ac[1] /= n; bc[0] /= n; bc[1] /= n;
  let na = 0;
  let nb = 0;
  const Ac = zeros(n, 2);
  const Bc = zeros(n, 2);
  for (let i = 0; i < n; i++) {
    Ac[i][0] = A[i][0] - ac[0];
    Ac[i][1] = A[i][1] - ac[1];
    Bc[i][0] = B[i][0] - bc[0];
    Bc[i][1] = B[i][1] - bc[1];
    na += Ac[i][0] ** 2 + Ac[i][1] ** 2;
    nb += Bc[i][0] ** 2 + Bc[i][1] ** 2;
  }
  na = Math.sqrt(na);
  nb = Math.sqrt(nb);
  if (na === 0 || nb === 0) return { disparity: 1, coords: Bc };
  for (let i = 0; i < n; i++) {
    Ac[i][0] /= na; Ac[i][1] /= na; Bc[i][0] /= nb; Bc[i][1] /= nb;
  }
  /* M = Bc' Ac, a 2x2, so its singular values are available in closed form
     through the eigenvalues of M'M and no iterative SVD is needed. */
  let m00 = 0; let m01 = 0; let m10 = 0; let m11 = 0;
  for (let i = 0; i < n; i++) {
    m00 += Bc[i][0] * Ac[i][0];
    m01 += Bc[i][0] * Ac[i][1];
    m10 += Bc[i][1] * Ac[i][0];
    m11 += Bc[i][1] * Ac[i][1];
  }
  const { values, vectors } = symEigen([
    Float64Array.from([m00 * m00 + m10 * m10, m00 * m01 + m10 * m11]),
    Float64Array.from([m00 * m01 + m10 * m11, m01 * m01 + m11 * m11]),
  ]);
  const s0 = Math.sqrt(Math.max(values[0], 0));
  const s1 = Math.sqrt(Math.max(values[1], 0));
  const scale = s0 + s1;
  /* R = U V' with M = U S V'. From the eigenvectors of M'M (which are V) and
     U = M V S^-1, R comes out as M V S^-1 V'. */
  const v = [[vectors[0][0], vectors[0][1]], [vectors[1][0], vectors[1][1]]];
  const inv = [s0 > 1e-12 ? 1 / s0 : 0, s1 > 1e-12 ? 1 / s1 : 0];
  const mv = [
    [m00 * v[0][0] + m01 * v[1][0], m00 * v[0][1] + m01 * v[1][1]],
    [m10 * v[0][0] + m11 * v[1][0], m10 * v[0][1] + m11 * v[1][1]],
  ];
  const R = [[0, 0], [0, 0]];
  for (let a = 0; a < 2; a++) {
    for (let b = 0; b < 2; b++) {
      R[a][b] = mv[a][0] * inv[0] * v[b][0] + mv[a][1] * inv[1] * v[b][1];
    }
  }
  let disp = 0;
  const out = zeros(n, 2);
  for (let i = 0; i < n; i++) {
    out[i][0] = scale * (Bc[i][0] * R[0][0] + Bc[i][1] * R[1][0]);
    out[i][1] = scale * (Bc[i][0] * R[0][1] + Bc[i][1] * R[1][1]);
    disp += (Ac[i][0] - out[i][0]) ** 2 + (Ac[i][1] - out[i][1]) ** 2;
  }
  return { disparity: disp, coords: out, reference: Ac };
}

/* ------------------------------------------------------------------- isotonic */
/* Pool adjacent violators. The same algorithm the calibration article ran on
 * probabilities; here it runs on distances, and it is the single line that
 * separates non-metric scaling from metric scaling. Input must already be in
 * the order the monotone constraint applies to. */
export function pav(y) {
  const n = y.length;
  const vals = new Float64Array(n);
  const wts = new Float64Array(n);
  const size = new Int32Array(n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    vals[k] = y[i];
    wts[k] = 1;
    size[k] = 1;
    k++;
    while (k > 1 && vals[k - 2] > vals[k - 1]) {
      const tot = wts[k - 2] + wts[k - 1];
      vals[k - 2] = (wts[k - 2] * vals[k - 2] + wts[k - 1] * vals[k - 1]) / tot;
      wts[k - 2] = tot;
      size[k - 2] += size[k - 1];
      k--;
    }
  }
  const out = new Float64Array(n);
  let j = 0;
  for (let i = 0; i < k; i++) {
    for (let t = 0; t < size[i]; t++) out[j++] = vals[i];
  }
  return out;
}

/* --------------------------------------------------------------------- SMACOF */
/* One Guttman transform per iteration. `order` is the permutation that sorts
 * the upper triangle of the target by size; passing it switches the algorithm
 * from metric (fit the dissimilarities) to non-metric (fit any monotone
 * function of them, found by pool adjacent violators at every step). */
export function smacof(D, Z0, iters, { nonmetric = false, order = null, trace = false } = {}) {
  const n = D.length;
  let Z = Z0.map((r) => Float64Array.from(r));
  let target = D;
  const hist = [];
  const pairs = order ? order.length : 0;
  const dv = new Float64Array(pairs);
  for (let it = 0; it < iters; it++) {
    const d = pairwise(Z);
    if (nonmetric) {
      let idx = 0;
      let sum2 = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) sum2 += d[i][j] * d[i][j];
      }
      for (let t = 0; t < pairs; t++) {
        const o = order[t];
        const i = Math.floor(o / n);
        const j = o % n;
        dv[t] = d[i][j];
      }
      const hat = pav(dv);
      let hs = 0;
      for (let t = 0; t < pairs; t++) hs += hat[t] * hat[t];
      const scale = Math.sqrt(sum2 / Math.max(hs, 1e-12));
      const T = zeros(n, n);
      for (let t = 0; t < pairs; t++) {
        const o = order[t];
        const i = Math.floor(o / n);
        const j = o % n;
        T[i][j] = hat[t] * scale;
        T[j][i] = T[i][j];
      }
      target = T;
    }
    const Znew = zeros(n, 2);
    for (let i = 0; i < n; i++) {
      let diag = 0;
      let x = 0;
      let y = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dij = d[i][j];
        const b = dij > 1e-12 ? -target[i][j] / dij : 0;
        diag -= b;
        x += b * Z[j][0];
        y += b * Z[j][1];
      }
      Znew[i][0] = (x + diag * Z[i][0]) / n;
      Znew[i][1] = (y + diag * Z[i][1]) / n;
    }
    Z = Znew;
    if (trace) hist.push(stress1(target, Z));
  }
  return { coords: Z, trace: hist, target };
}

/* The permutation that sorts the upper triangle of D, encoded as i*n+j so one
 * integer names a pair. Stable, because Python's argsort is and the two have to
 * break ties the same way. */
export function upperOrder(D) {
  const n = D.length;
  const idx = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) idx.push(i * n + j);
  return idx
    .map((o, t) => [D[Math.floor(o / n)][o % n], t, o])
    .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]))
    .map((r) => r[2]);
}

/* ----------------------------------------------------------------- neighbours */
/* Symmetric k-nearest-neighbour graph as adjacency lists. Symmetric because an
 * asymmetric one makes the shortest path between two points depend on which end
 * you start from, which is not something a distance is allowed to do. */
export function knnGraph(D, k) {
  const n = D.length;
  const adj = Array.from({ length: n }, () => new Map());
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < n; i++) {
    const row = idx.filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    for (let t = 0; t < k && t < row.length; t++) {
      const j = row[t];
      adj[i].set(j, D[i][j]);
      adj[j].set(i, D[i][j]);
    }
  }
  return adj;
}

export function components(adj) {
  const n = adj.length;
  const label = new Int32Array(n).fill(-1);
  let c = 0;
  for (let s = 0; s < n; s++) {
    if (label[s] !== -1) continue;
    const stack = [s];
    label[s] = c;
    while (stack.length) {
      const u = stack.pop();
      for (const v of adj[u].keys()) {
        if (label[v] === -1) {
          label[v] = c;
          stack.push(v);
        }
      }
    }
    c++;
  }
  return { count: c, label };
}

/* Dijkstra from every source, with a binary heap. Floyd-Warshall would be four
 * lines and n^3; at n = 500 that is 125 million and this is about 30. */
export function allPairsShortest(adj) {
  const n = adj.length;
  const out = zeros(n, n);
  for (let s = 0; s < n; s++) {
    const dist = out[s];
    dist.fill(Infinity);
    dist[s] = 0;
    const heap = [[0, s]];
    const done = new Uint8Array(n);
    while (heap.length) {
      let bi = 0;
      for (let i = 1; i < heap.length; i++) if (heap[i][0] < heap[bi][0]) bi = i;
      const [du, u] = heap.splice(bi, 1)[0];
      if (done[u]) continue;
      done[u] = 1;
      for (const [v, w] of adj[u]) {
        const nd = du + w;
        if (nd < dist[v]) {
          dist[v] = nd;
          heap.push([nd, v]);
        }
      }
    }
  }
  return out;
}

/* ---------------------------------------------------------------- statistics */
export function corr(a, b) {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    sab += x * y;
    sa += x * x;
    sb += y * y;
  }
  return sab / Math.sqrt(sa * sb);
}

/* Trustworthiness: how many of a point's neighbours in the MAP were not its
 * neighbours in the data, weighted by how far down the original ranking they
 * sat. One minus it is the share of the map's local structure that is invented.
 * Same definition and same normalisation as scikit-learn's. */
export function trustworthiness(D, Dz, k = 12) {
  const n = D.length;
  const rank = zeros(n, n);
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i][a] - D[i][b]);
    for (let r = 0; r < order.length; r++) rank[i][order[r]] = r + 1;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => Dz[i][a] - Dz[i][b])
      .slice(0, k);
    for (const j of order) {
      const r = rank[i][j];
      if (r > k) sum += r - k;
    }
  }
  return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * sum;
}
