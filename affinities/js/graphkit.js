/* The two algorithms of this article, written out in the browser, plus the
 * pieces they are built from: a symmetric eigensolver, the normalised cut, and
 * the k-means and adjusted Rand index everything else is judged with.
 *
 * Both algorithms here start from an n by n table and never look at a
 * coordinate again, which is what the article is about, so every routine below
 * takes a matrix rather than points wherever it can.
 *
 * All of it is checked at load time against the references in
 * data/affinity.json: affinity propagation against scikit-learn's own
 * implementation exemplar for exemplar, the eigenvalues against scipy's
 * `eigh`, and the normalised cuts against an exhaustive enumeration.
 *
 * Matrices are flat Float64Arrays in row-major order, because n squared of
 * anything is the recurring cost of this article and an array of arrays pays
 * a pointer per row for nothing.
 */

/* ------------------------------------------------------------------ matrices */

export const at = (M, n, i, j) => M[i * n + j];

/* Negative squared euclidean distance, which is the similarity affinity
 * propagation is normally handed and the one scikit-learn defaults to. The
 * diagonal is left at zero here and filled with the preference later, because
 * the preference is the dial the reader moves. */
export function similarity(X) {
  const n = X.length;
  const d = X[0].length;
  const S = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let c = 0; c < d; c++) s += (X[i][c] - X[j][c]) ** 2;
      S[i * n + j] = -s;
      S[j * n + i] = -s;
    }
  }
  return S;
}

/* The median of the n(n-1)/2 off-diagonal similarities: scikit-learn's default
 * preference, and the one the article measures against. */
export function medianSimilarity(S, n) {
  const vals = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) vals.push(S[i * n + j]);
  vals.sort((a, b) => a - b);
  const m = vals.length;
  return m % 2 ? vals[(m - 1) / 2] : (vals[m / 2 - 1] + vals[m / 2]) / 2;
}

export function minSimilarity(S, n) {
  let lo = Infinity;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (S[i * n + j] < lo) lo = S[i * n + j];
  return lo;
}

/* ------------------------------------------------------- affinity propagation
 * Two messages per ordered pair, updated in lockstep until the set of points
 * that elect themselves stops changing:
 *
 *   r(i,k) <- s(i,k) - max over k' != k of [ a(i,k') + s(i,k') ]
 *   a(i,k) <- min( 0, r(k,k) + sum over i' not in {i,k} of max(0, r(i',k)) )
 *   a(k,k) <- sum over i' != k of max(0, r(i',k))
 *
 * and point k is an exemplar when r(k,k) + a(k,k) > 0.
 *
 * This follows scikit-learn's loop line for line, including three details that
 * decide the answer rather than the speed:
 *
 *   - convergence is a rolling window: every point's exemplar flag has to hold
 *     steady for `convIter` iterations, not just the set as a whole for one.
 *   - after the loop, each cluster re-elects as its exemplar the member with
 *     the largest within-cluster similarity sum, and the assignment is redone.
 *   - an exemplar belongs to its own cluster by decree, not by argmax. Without
 *     that line an exemplar can be assigned to a neighbouring exemplar (its own
 *     preference is often lower than its similarity to the next one over) and
 *     the partition drifts away from the reference.
 *
 * scikit-learn also adds a random perturbation of the order of machine epsilon
 * to break exact ties. It is not reproduced here, and the generator checks that
 * it changes nothing on this data: six random seeds give the same exemplars.
 */
export function affinityPropagation(S0, n, {
  preference, damping = 0.9, maxIter = 2000, convIter = 50, trace = false, snapshots = null,
} = {}) {
  const S = Float64Array.from(S0);
  for (let i = 0; i < n; i++) S[i * n + i] = preference;
  const A = new Float64Array(n * n);
  const R = new Float64Array(n * n);
  const T = new Float64Array(n * n);
  const col = new Float64Array(n);
  const flags = new Uint8Array(n * convIter);      // rolling window of exemplar flags
  const counts = trace ? [] : null;
  const shots = snapshots ? {} : null;
  let iters = maxIter;
  let converged = false;

  for (let it = 0; it < maxIter; it++) {
    /* responsibilities: the best alternative to k, seen from i */
    for (let i = 0; i < n; i++) {
      const off = i * n;
      let b1 = -Infinity;
      let b2 = -Infinity;
      let arg = 0;
      for (let j = 0; j < n; j++) {
        const v = A[off + j] + S[off + j];
        if (v > b1) {
          b2 = b1;
          b1 = v;
          arg = j;
        } else if (v > b2) {
          b2 = v;
        }
      }
      for (let j = 0; j < n; j++) {
        const rn = S[off + j] - (j === arg ? b2 : b1);
        R[off + j] = damping * R[off + j] + (1 - damping) * rn;
      }
    }

    /* availabilities: how much support k has from everybody other than i */
    col.fill(0);
    for (let i = 0; i < n; i++) {
      const off = i * n;
      for (let j = 0; j < n; j++) {
        const rp = i === j ? R[off + j] : Math.max(R[off + j], 0);
        T[off + j] = rp;
        col[j] += rp;
      }
    }
    for (let i = 0; i < n; i++) {
      const off = i * n;
      for (let j = 0; j < n; j++) {
        let an = col[j] - T[off + j];
        if (i !== j && an > 0) an = 0;              // off-diagonal availabilities are never positive
        A[off + j] = damping * A[off + j] + (1 - damping) * an;
      }
    }

    /* convergence: the flag of every point steady across the whole window */
    let k = 0;
    const slot = it % convIter;
    for (let i = 0; i < n; i++) {
      const e = A[i * n + i] + R[i * n + i] > 0 ? 1 : 0;
      flags[i * convIter + slot] = e;
      k += e;
    }
    if (counts) counts.push(k);
    /* A copy of both message boards at named iterations, for the scrolly: the
       story of this algorithm is the messages themselves, and they are gone by
       the time it finishes. */
    if (shots && snapshots.includes(it + 1)) {
      shots[it + 1] = { R: Float64Array.from(R), A: Float64Array.from(A), k };
    }
    if (it >= convIter) {
      let unconverged = false;
      for (let i = 0; i < n && !unconverged; i++) {
        let s = 0;
        for (let t = 0; t < convIter; t++) s += flags[i * convIter + t];
        if (s !== 0 && s !== convIter) unconverged = true;
      }
      if (!unconverged && k > 0) {
        iters = it + 1;
        converged = true;
        break;
      }
    }
  }

  const ex = [];
  for (let i = 0; i < n; i++) if (A[i * n + i] + R[i * n + i] > 0) ex.push(i);
  const self = new Float64Array(n);
  for (let i = 0; i < n; i++) self[i] = A[i * n + i] + R[i * n + i];
  if (!ex.length) {
    return { exemplars: [], labels: new Array(n).fill(0), iters, converged, counts, shots, self };
  }

  const assign = (exs) => {
    const lab = new Array(n);
    for (let i = 0; i < n; i++) {
      let b = -Infinity;
      let a = 0;
      for (let c = 0; c < exs.length; c++) {
        const v = S[i * n + exs[c]];
        if (v > b) {
          b = v;
          a = c;
        }
      }
      lab[i] = a;
    }
    exs.forEach((e, c) => { lab[e] = c; });        // an exemplar is its own cluster
    return lab;
  };

  let labels = assign(ex);
  for (let c = 0; c < ex.length; c++) {
    const members = [];
    for (let i = 0; i < n; i++) if (labels[i] === c) members.push(i);
    let best = -Infinity;
    let pick = ex[c];
    for (const j of members) {
      let s = 0;
      for (const i of members) s += S[i * n + j];
      if (s > best) {
        best = s;
        pick = j;
      }
    }
    ex[c] = pick;
  }
  ex.sort((a, b) => a - b);
  labels = assign(ex);
  return { exemplars: ex, labels, iters, converged, counts, shots, self };
}

/* What affinity propagation is maximising: the similarity of every point to
 * its exemplar, plus the preference paid once per exemplar. Writing it down is
 * what makes the comparison with k-medoids a comparison and not an analogy. */
export function netSimilarity(S, n, exemplars, preference) {
  let total = preference * exemplars.length;
  for (let i = 0; i < n; i++) {
    if (exemplars.includes(i)) continue;
    let b = -Infinity;
    for (const e of exemplars) if (S[i * n + e] > b) b = S[i * n + e];
    total += b;
  }
  return total;
}

/* ---------------------------------------------------------------- the graphs */

/* k nearest neighbours, symmetrised with an OR: i and j are joined if either
 * one is in the other's list. Unweighted, which is scikit-learn's
 * `affinity='nearest_neighbors'` and the version the article draws. */
export function knnGraph(X, k) {
  const n = X.length;
  const W = new Float64Array(n * n);
  const idx = new Array(n);
  const dist = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let c = 0; c < X[i].length; c++) s += (X[i][c] - X[j][c]) ** 2;
      dist[j] = s;
      idx[j] = j;
    }
    const order = idx.slice().sort((a, b) => dist[a] - dist[b] || a - b);
    for (let t = 1; t <= k && t < n; t++) {
      W[i * n + order[t]] = 1;
      W[order[t] * n + i] = 1;
    }
  }
  return W;
}

/* A gaussian on every pair, with no zeros anywhere: the fully connected graph
 * whose bandwidth is the only thing deciding what counts as near. */
export function rbfGraph(X, gamma) {
  const n = X.length;
  const W = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let c = 0; c < X[i].length; c++) s += (X[i][c] - X[j][c]) ** 2;
      const w = Math.exp(-gamma * s);
      W[i * n + j] = w;
      W[j * n + i] = w;
    }
  }
  return W;
}

/* Connected components by breadth-first search. Worth having on its own: the
 * number of components is exactly the multiplicity of the eigenvalue zero, so
 * this is the same quantity the spectrum reports, arrived at without any
 * arithmetic at all. */
export function components(W, n, tol = 0) {
  const comp = new Int32Array(n).fill(-1);
  let c = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] >= 0) continue;
    const queue = [s];
    comp[s] = c;
    while (queue.length) {
      const p = queue.pop();
      for (let j = 0; j < n; j++) {
        if (comp[j] < 0 && W[p * n + j] > tol) {
          comp[j] = c;
          queue.push(j);
        }
      }
    }
    c += 1;
  }
  return { comp: Array.from(comp), count: c };
}

/* ------------------------------------------------------------- eigenvectors
 * The top q eigenpairs of a symmetric matrix by Lanczos with full
 * reorthogonalisation, deflated against any eigenvectors already known.
 *
 * Why not the plain power method: the gap that separates the eigenvalues this
 * article cares about is tiny (on the blobs the third and fourth normalised
 * Laplacian eigenvalues are 0.006 and 0.058, which is a ratio of 0.97 in the
 * matrix actually iterated), and the power method needs about one matrix
 * multiply per digit per that ratio, which is hundreds. A Krylov space of 80
 * gets the same accuracy, because it converges with the square root of the gap
 * rather than the gap.
 *
 * Why the caller passes in known vectors: a Krylov space built from one
 * starting vector can only ever produce ONE vector out of a repeated
 * eigenvalue's subspace, and a graph in two pieces has the eigenvalue zero
 * twice. The rings are exactly that case, so this is not a hypothetical.
 * `spectral` below hands over the indicator of every connected component,
 * which is that whole subspace written down exactly rather than iterated
 * towards, and Lanczos then works in what is left.
 */
function orthogonalise(v, basis) {
  for (let pass = 0; pass < 2; pass++) {          // twice: once is not enough in floating point
    for (const u of basis) {
      let d = 0;
      for (let i = 0; i < v.length; i++) d += v[i] * u[i];
      for (let i = 0; i < v.length; i++) v[i] -= d * u[i];
    }
  }
  return v;
}

const norm2 = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

export function topEigen(M, n, q, { steps = 80, known = [] } = {}) {
  /* A graph of k neighbours has about 2k non-zeros per row out of n, so the
     dense product below would spend 95% of its time multiplying by zero. One
     pass to find out which it is pays for itself many times over. */
  let nnz = 0;
  for (let i = 0; i < M.length; i++) if (M[i] !== 0) nnz++;
  let mv;
  if (nnz * 6 < n * n) {
    const ptr = new Int32Array(n + 1);
    const col = new Int32Array(nnz);
    const val = new Float64Array(nnz);
    let t = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = M[i * n + j];
        if (v !== 0) {
          col[t] = j;
          val[t] = v;
          t++;
        }
      }
      ptr[i + 1] = t;
    }
    mv = (v) => {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let p = ptr[i]; p < ptr[i + 1]; p++) s += val[p] * v[col[p]];
        out[i] = s;
      }
      return out;
    };
  } else {
    mv = (v) => {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const off = i * n;
        let s = 0;
        for (let j = 0; j < n; j++) s += M[off + j] * v[j];
        out[i] = s;
      }
      return out;
    };
  }

  const rayleigh = (x) => {
    const Mx = mv(x);
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * Mx[i];
    return s;
  };
  const found = known.map((v) => Float64Array.from(v));
  const values = found.map(rayleigh);
  if (values.length >= q) return { values: values.slice(0, q), vectors: found.slice(0, q) };

  let seed = 12345;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff - 0.5;
  };
  const v0 = new Float64Array(n);
  for (let i = 0; i < n; i++) v0[i] = rand();
  orthogonalise(v0, found);
  const nv = norm2(v0);
  if (nv < 1e-10) return { values, vectors: found };
  for (let i = 0; i < n; i++) v0[i] /= nv;

  const V = [v0];
  const alpha = [];
  const beta = [];
  for (let it = 0; it < Math.min(steps, n - found.length); it++) {
    const w = mv(V[it]);
    orthogonalise(w, found);
    let a = 0;
    for (let i = 0; i < n; i++) a += w[i] * V[it][i];
    alpha.push(a);
    orthogonalise(w, V);
    const b = norm2(w);
    if (b < 1e-10) break;
    beta.push(b);
    for (let i = 0; i < n; i++) w[i] /= b;
    V.push(w);
  }

  const m = alpha.length;
  const T = Array.from({ length: m }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) {
    T[i][i] = alpha[i];
    if (i + 1 < m) {
      T[i][i + 1] = beta[i];
      T[i + 1][i] = beta[i];
    }
  }
  const { vectors: tvec } = jacobiEigen(T);
  for (let t = 0; t < tvec.length && values.length < q; t++) {
    const y = tvec[t];
    const x = new Float64Array(n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) x[j] += y[i] * V[i][j];
    orthogonalise(x, found);
    const nx = norm2(x);
    if (nx < 1e-8) continue;
    for (let i = 0; i < n; i++) x[i] /= nx;
    /* Rayleigh quotient of the vector actually returned, rather than the Ritz
       value from the small matrix: they differ in the last digits, and the one
       the guard compares against scipy has to be the honest one. */
    found.push(x);
    values.push(rayleigh(x));
  }
  return { values, vectors: found };
}

/* Cyclic Jacobi for a small dense symmetric matrix, descending. Used on the
 * Lanczos tridiagonal (at most 60 by 60) and on the 16 by 16 Laplacian of the
 * exhaustive section, where the whole spectrum is wanted and n is tiny. */
export function jacobiEigen(Ain, { sweeps = 60, tol = 1e-13 } = {}) {
  const p = Ain.length;
  const A = Ain.map((row) => Float64Array.from(row));
  const V = Array.from({ length: p }, (_, i) => {
    const r = new Float64Array(p);
    r[i] = 1;
    return r;
  });
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += A[i][j] ** 2;
    if (Math.sqrt(off) < tol) break;
    for (let i = 0; i < p; i++) {
      for (let j = i + 1; j < p; j++) {
        if (Math.abs(A[i][j]) < 1e-300) continue;
        const theta = (A[j][j] - A[i][i]) / (2 * A[i][j]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < p; k++) {
          const aik = A[i][k];
          const ajk = A[j][k];
          A[i][k] = c * aik - s * ajk;
          A[j][k] = s * aik + c * ajk;
        }
        for (let k = 0; k < p; k++) {
          const aki = A[k][i];
          const akj = A[k][j];
          A[k][i] = c * aki - s * akj;
          A[k][j] = s * aki + c * akj;
        }
        for (let k = 0; k < p; k++) {
          const vki = V[k][i];
          const vkj = V[k][j];
          V[k][i] = c * vki - s * vkj;
          V[k][j] = s * vki + c * vkj;
        }
      }
    }
  }
  const order = Array.from({ length: p }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  return {
    values: order.map((i) => A[i][i]),
    vectors: order.map((i) => Float64Array.from({ length: p }, (_, k) => V[k][i])),
  };
}

/* ------------------------------------------------------- spectral clustering
 * Ng, Jordan and Weiss: the q smallest eigenvectors of the symmetric
 * normalised Laplacian, every row rescaled to unit length, and k-means on
 * those rows. The eigenvalues come back too, because their gaps are the only
 * thing on this page that has an opinion about k. */
export function spectral(W, n, k, { extra = 4, restarts = 30, seed = 19, steps = 140 } = {}) {
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += W[i * n + j];
    deg[i] = s;
  }
  const dinv = new Float64Array(n);
  for (let i = 0; i < n; i++) dinv[i] = deg[i] > 1e-12 ? 1 / Math.sqrt(deg[i]) : 0;
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) M[i * n + j] = dinv[i] * W[i * n + j] * dinv[j];
  }
  /* The eigenvectors with eigenvalue zero, written down instead of iterated
     towards: on a graph in c pieces they are the c indicators of those pieces,
     scaled by the square root of the degree. This is the identity the article
     draws bars for, used here to make the solver exact where it matters most
     and to keep it out of trouble where a repeated eigenvalue would otherwise
     defeat a Krylov method. */
  const { comp, count } = components(W, n);
  const known = [];
  for (let c = 0; c < count; c++) {
    const v = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) {
      if (comp[i] === c) {
        v[i] = Math.sqrt(deg[i]);
        s += deg[i];
      }
    }
    if (s <= 0) continue;
    const inv = 1 / Math.sqrt(s);
    for (let i = 0; i < n; i++) v[i] *= inv;
    known.push(v);
  }
  const q = Math.min(n, k + extra);
  const { values, vectors } = topEigen(M, n, q, { known: known.slice(0, q), steps });
  const eigenvalues = values.map((v) => 1 - v);          // of the Laplacian, ascending
  const rows = [];
  for (let i = 0; i < n; i++) {
    const r = new Array(k);
    for (let c = 0; c < k; c++) r[c] = vectors[c] ? vectors[c][i] : 0;
    const nr = Math.hypot(...r);
    rows.push(nr > 1e-12 ? r.map((v) => v / nr) : r);
  }
  const km = kmeans(rows, k, { restarts, seed });
  return { labels: km.labels, eigenvalues, embedding: rows, degrees: Array.from(deg) };
}

/* Shi and Malik's normalised cut of a bipartition, the quantity the eigenvector
 * is a relaxation of: the weight crossing the boundary, charged against the
 * total weight on each side, so that shaving off one lonely point is expensive
 * rather than free. */
export function ncut(W, n, mask) {
  let cut = 0;
  let assocA = 0;
  let assocB = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const w = W[i * n + j];
      if (mask[i]) assocA += w; else assocB += w;
      if (mask[i] !== mask[j]) cut += w;
    }
  }
  cut /= 2;                                        // every crossing edge counted twice above
  if (!assocA || !assocB) return Infinity;
  return cut / assocA + cut / assocB;
}

/* ------------------------------------------------------------ for comparison */

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function kmeans(points, k, { restarts = 10, seed = 19, maxIter = 100 } = {}) {
  const rand = rng(seed);
  const dim = points[0].length;
  const d2 = (p, q) => {
    let s = 0;
    for (let c = 0; c < dim; c++) s += (p[c] - q[c]) ** 2;
    return s;
  };
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const centers = [points[Math.floor(rand() * points.length)].slice()];
    while (centers.length < k) {
      const w = points.map((p) => Math.min(...centers.map((c) => d2(p, c))));
      const total = w.reduce((a, b) => a + b, 0);
      let u = rand() * total;
      let i = 0;
      while (u > w[i] && i < w.length - 1) {
        u -= w[i];
        i++;
      }
      centers.push(points[i].slice());
    }
    let labels = new Array(points.length).fill(-1);
    let inertia = Infinity;
    for (let it = 0; it < maxIter; it++) {
      let J = 0;
      const next = points.map((p) => {
        let b = Infinity;
        let a = 0;
        centers.forEach((c, i) => {
          const d = d2(p, c);
          if (d < b) {
            b = d;
            a = i;
          }
        });
        J += b;
        return a;
      });
      const stable = next.every((v, i) => v === labels[i]);
      labels = next;
      inertia = J;
      const sums = centers.map(() => new Float64Array(dim + 1));
      points.forEach((p, i) => {
        const s = sums[labels[i]];
        for (let c = 0; c < dim; c++) s[c] += p[c];
        s[dim] += 1;
      });
      sums.forEach((s, i) => {
        if (s[dim]) centers[i] = Array.from({ length: dim }, (_, c) => s[c] / s[dim]);
      });
      if (stable) break;
    }
    if (!best || inertia < best.inertia) best = { centers: centers.map((c) => c.slice()), labels, inertia };
  }
  return best;
}

/* Adjusted Rand index, with -1 treated as a class of its own, as in every
 * other article of this module. */
export function ari(a, b) {
  const map = (v) => v + 2;
  const ka = Math.max(...a) + 3;
  const kb = Math.max(...b) + 3;
  const M = Array.from({ length: ka }, () => new Array(kb).fill(0));
  for (let i = 0; i < a.length; i++) M[map(a[i])][map(b[i])] += 1;
  const comb2 = (x) => (x * (x - 1)) / 2;
  let sumIJ = 0;
  const rows = new Array(ka).fill(0);
  const cols = new Array(kb).fill(0);
  for (let i = 0; i < ka; i++) {
    for (let j = 0; j < kb; j++) {
      sumIJ += comb2(M[i][j]);
      rows[i] += M[i][j];
      cols[j] += M[i][j];
    }
  }
  const sumA = rows.reduce((s, v) => s + comb2(v), 0);
  const sumB = cols.reduce((s, v) => s + comb2(v), 0);
  const total = comb2(a.length);
  const expected = (sumA * sumB) / total;
  const max = (sumA + sumB) / 2;
  return max === expected ? 1 : (sumIJ - expected) / (max - expected);
}
