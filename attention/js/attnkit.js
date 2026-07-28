/* The trained encoder, running in the browser, one sentence at a time.
 *
 * The weights that matter travel in full: every projection of every head, both
 * layer norms of both blocks, the feed-forward, the output. Only the embedding
 * table is trimmed, to the rows the example sentences actually use, because the
 * full one is thirty thousand rows and the page cannot ask about words that are
 * not on it anyway.
 *
 * That distinction is the point of doing it this way. The attention matrices
 * below are not pictures the generator drew and shipped: they are computed
 * here, from the weights, which is why the reader can switch the scaling off or
 * knock out a head and watch the same numbers change.
 */
import { decodeHalf, softmax, entropy } from '../../assets/js/textkit.js';

export function loadModel(M) {
  const blob = decodeHalf(M.blob, M.size);
  const get = (name) => {
    const s = M.spec[name];
    if (!s) throw new Error(`the export has no tensor called ${name}`);
    const n = s.shape.reduce((a, b) => a * b, 1);
    return { data: blob.subarray(s.offset, s.offset + n), shape: s.shape };
  };
  const blocks = [];
  for (let l = 0; l < M.layers; l++) {
    blocks.push({
      q: lin(get(`blocks.${l}.att.q.weight`), get(`blocks.${l}.att.q.bias`)),
      k: lin(get(`blocks.${l}.att.k.weight`), get(`blocks.${l}.att.k.bias`)),
      v: lin(get(`blocks.${l}.att.v.weight`), get(`blocks.${l}.att.v.bias`)),
      o: lin(get(`blocks.${l}.att.o.weight`), get(`blocks.${l}.att.o.bias`)),
      n1: norm(get(`blocks.${l}.n1.weight`), get(`blocks.${l}.n1.bias`)),
      n2: norm(get(`blocks.${l}.n2.weight`), get(`blocks.${l}.n2.bias`)),
      ff1: lin(get(`blocks.${l}.ff.0.weight`), get(`blocks.${l}.ff.0.bias`)),
      ff2: lin(get(`blocks.${l}.ff.2.weight`), get(`blocks.${l}.ff.2.bias`)),
    });
  }
  return {
    d: M.d_model,
    heads: M.heads,
    layers: M.layers,
    dk: M.d_model / M.heads,
    emb: get('emb.weight'),
    pos: get('pos.weight'),
    blocks,
    normF: norm(get('norm.weight'), get('norm.bias')),
    head: lin(get('head.weight'), get('head.bias')),
    words: M.words,
    tags: M.tags,
    sentences: M.sentences,
  };
}

const lin = (w, b) => ({ w: w.data, b: b.data, out: w.shape[0], in: w.shape[1] });
const norm = (w, b) => ({ w: w.data, b: b.data, n: w.shape[0] });

/* y = W x + b, with W stored row-major as torch writes it. */
function matvec(L, x, off = 0) {
  const out = new Float64Array(L.out);
  for (let o = 0; o < L.out; o++) {
    let s = L.b[o];
    const base = o * L.in;
    for (let i = 0; i < L.in; i++) s += L.w[base + i] * x[off + i];
    out[o] = s;
  }
  return out;
}

/* Layer norm, population variance, matching torch's default eps. */
function layerNorm(N, x) {
  const n = N.n;
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * N.w[i] + N.b[i];
  return out;
}

const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

/**
 * One forward pass, returning the tags and every attention matrix on the way.
 *
 * @param opts.scale       divide the scores by the square root of the head size
 * @param opts.positional  add the position embedding at all
 * @param opts.drop        [layer, head] to zero out, for ablation
 */
export function forward(m, ids, { scale = true, positional = true, drop = null } = {}) {
  const L = ids.length;
  const d = m.d;
  let h = [];
  for (let t = 0; t < L; t++) {
    const row = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      row[i] = m.emb.data[ids[t] * d + i] + (positional ? m.pos.data[t * d + i] : 0);
    }
    h.push(row);
  }

  const maps = [];
  m.blocks.forEach((B, l) => {
    const normed = h.map((row) => layerNorm(B.n1, row));
    const q = normed.map((row) => matvec(B.q, row));
    const k = normed.map((row) => matvec(B.k, row));
    const v = normed.map((row) => matvec(B.v, row));
    const layerMaps = [];
    const ctx = Array.from({ length: L }, () => new Float64Array(d));
    for (let hd = 0; hd < m.heads; hd++) {
      const off = hd * m.dk;
      const map = [];
      for (let i = 0; i < L; i++) {
        const scores = new Float64Array(L);
        for (let j = 0; j < L; j++) {
          let s = 0;
          for (let a = 0; a < m.dk; a++) s += q[i][off + a] * k[j][off + a];
          scores[j] = scale ? s / Math.sqrt(m.dk) : s;
        }
        softmax(scores);
        map.push(Array.from(scores));
        const zero = drop && drop[0] === l && drop[1] === hd;
        if (!zero) {
          for (let j = 0; j < L; j++) {
            const wgt = scores[j];
            if (wgt === 0) continue;
            for (let a = 0; a < m.dk; a++) ctx[i][off + a] += wgt * v[j][off + a];
          }
        }
      }
      layerMaps.push(map);
    }
    maps.push(layerMaps);
    h = h.map((row, i) => {
      const projected = matvec(B.o, ctx[i]);
      const res = new Float64Array(d);
      for (let a = 0; a < d; a++) res[a] = row[a] + projected[a];
      const n2 = layerNorm(B.n2, res);
      const hidden = matvec(B.ff1, n2);
      for (let a = 0; a < hidden.length; a++) hidden[a] = gelu(hidden[a]);
      const back = matvec(B.ff2, hidden);
      const out = new Float64Array(d);
      for (let a = 0; a < d; a++) out[a] = res[a] + back[a];
      return out;
    });
  });

  const tags = h.map((row) => {
    const logits = matvec(m.head, layerNorm(m.normF, row));
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  });
  return { tags, maps, hidden: h };
}

/** Mean entropy of every attention distribution in one forward pass. */
export function meanEntropy(maps) {
  let acc = 0;
  let n = 0;
  maps.forEach((layer) => layer.forEach((head) => head.forEach((row) => {
    acc += entropy(Float64Array.from(row));
    n += 1;
  })));
  return n ? acc / n : 0;
}

/** How concentrated the attention is: the mean of each row's largest weight. */
export function meanPeak(maps) {
  let acc = 0;
  let n = 0;
  maps.forEach((layer) => layer.forEach((head) => head.forEach((row) => {
    acc += Math.max(...row);
    n += 1;
  })));
  return n ? acc / n : 0;
}
