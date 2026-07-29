/* The weighting, in the browser, from the same integers the generator used.
 *
 * What travels in the JSON is a term count and a document frequency, both
 * whole numbers, and nothing else. Every weight on this page is built here
 * from those, which is why the guard in main.js can demand agreement to 1e-12
 * with the offline reference instead of settling for "close enough": there is
 * no rounding anywhere between the two sides to argue about.
 *
 * The formula is deliberately written as the three independent decisions it
 * is, because that is the article: what you do with the count, whether you
 * weight by rarity, and whether you divide by the length.
 */
import { decodeHalf, decodeU16, decodeU32, decodeU8 } from '../../assets/js/textkit.js';

export { decodeHalf };

/** Unpack the shipped index into typed arrays once. */
export function makeIndex(S) {
  return {
    words: S.words,
    df: decodeU32(S.df, S.words.length),
    nTrain: S.n_train,
    idx: decodeU16(S.idx, S.nnz),
    cnt: decodeU8(S.cnt),
    ptr: decodeU32(S.ptr, S.docs + 1),
    docs: S.docs,
    labels: S.labels,
    lens: decodeU16(S.lens, S.docs),
    titles: S.titles,
    labelNames: S.label_names,
    minDf: S.min_df,
  };
}

/** The smoothed inverse document frequency, ln((1+N)/(1+df)) + 1. */
export function idfOf(df, nTrain) {
  return Math.log((1 + nTrain) / (1 + df)) + 1;
}

export function idfTable(ix) {
  const out = new Float64Array(ix.df.length);
  for (let i = 0; i < out.length; i++) out[i] = idfOf(ix.df[i], ix.nTrain);
  return out;
}

/* The three things anybody does with a raw count. `binary` throws the count
 * away and keeps only that the word occurred; `raw` keeps it as it is;
 * `sublinear` is 1 + ln(c), which says the second occurrence of a word matters
 * much less than the first and the tenth barely at all. */
export function tfOf(c, mode) {
  if (mode === 'binary') return 1;
  if (mode === 'raw') return c;
  if (mode === 'sublinear') return 1 + Math.log(c);
  throw new Error(`unknown tf mode ${mode}`);
}

/**
 * One document as a sparse weighted vector.
 * @param {object} ix   the unpacked index
 * @param {number} d    document number
 * @param {object} opts {tf, useIdf, normalise, idf}
 */
export function vectorOf(ix, d, { tf = 'sublinear', useIdf = true, normalise = true,
  idf = null } = {}) {
  const a = ix.ptr[d];
  const b = ix.ptr[d + 1];
  const n = b - a;
  const index = new Int32Array(n);
  const val = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const j = ix.idx[a + k];
    index[k] = j;
    let v = tfOf(ix.cnt[a + k], tf);
    if (useIdf) v *= idf ? idf[j] : idfOf(ix.df[j], ix.nTrain);
    val[k] = v;
  }
  let norm = 0;
  for (let k = 0; k < n; k++) norm += val[k] * val[k];
  norm = Math.sqrt(norm);
  if (normalise && norm > 0) {
    for (let k = 0; k < n; k++) val[k] /= norm;
    norm = 1;
  }
  return { idx: index, val, norm, d };
}

/**
 * The k heaviest terms of a document.
 *
 * Ties are broken by vocabulary index, which is alphabetical, so the order is
 * total and the browser and the generator cannot disagree about it. With the
 * binary weighting every term in a document ties with every other, and without
 * a stated tie-break the two sides would print different lists from identical
 * numbers.
 */
export function topTerms(ix, vec, k) {
  const order = Array.from(vec.val.keys())
    .sort((p, q) => (vec.val[q] - vec.val[p]) || (vec.idx[p] - vec.idx[q]));
  return order.slice(0, k).map((o) => ({
    word: ix.words[vec.idx[o]], value: vec.val[o], j: vec.idx[o], df: ix.df[vec.idx[o]],
  }));
}

/* ------------------------------------------------------------------- search */

/* The whole test split, weighted once and kept as one flat pair of arrays.
 * 110,031 non-zeros is small enough that a full scan against every document is
 * a couple of milliseconds, so the page can afford to answer a query with the
 * exact answer rather than an approximate index. */
export function weightAll(ix, opts) {
  const idf = opts.useIdf ? idfTable(ix) : null;
  const val = new Float64Array(ix.idx.length);
  const norms = new Float64Array(ix.docs);
  for (let d = 0; d < ix.docs; d++) {
    const a = ix.ptr[d];
    const b = ix.ptr[d + 1];
    let s = 0;
    for (let k = a; k < b; k++) {
      let v = tfOf(ix.cnt[k], opts.tf);
      if (idf) v *= idf[ix.idx[k]];
      val[k] = v;
      s += v * v;
    }
    s = Math.sqrt(s);
    norms[d] = s;
    if (opts.normalise && s > 0) {
      for (let k = a; k < b; k++) val[k] /= s;
      norms[d] = 1;
    }
  }
  return { val, norms };
}

/**
 * Every document scored against one query document, exactly.
 *
 * `cosine` scores the L2-normalised vectors, which is a dot product once they
 * are normalised. `euclid` scores minus the squared distance between the
 * vectors as they come, which is the same dot product plus the two lengths,
 * so the two rankings differ only in what they do about length and nothing
 * else: that is the comparison the section is making, and doing it any other
 * way would confound it with a second change.
 */
export function rank(ix, W, q, metric) {
  const scores = new Float64Array(ix.docs);
  const acc = new Float64Array(ix.words.length);
  const qa = ix.ptr[q];
  const qb = ix.ptr[q + 1];
  for (let k = qa; k < qb; k++) acc[ix.idx[k]] = W.val[k];
  const qn2 = W.norms[q] * W.norms[q];
  for (let d = 0; d < ix.docs; d++) {
    const a = ix.ptr[d];
    const b = ix.ptr[d + 1];
    let s = 0;
    for (let k = a; k < b; k++) {
      const w = acc[ix.idx[k]];
      if (w !== 0) s += w * W.val[k];
    }
    scores[d] = metric === 'cosine' ? s : -(qn2 + W.norms[d] * W.norms[d] - 2 * s);
  }
  for (let k = qa; k < qb; k++) acc[ix.idx[k]] = 0;
  scores[q] = -Infinity;
  return scores;
}

export function topK(scores, k) {
  const order = Array.from(scores.keys()).sort((p, q) => scores[q] - scores[p]);
  return order.slice(0, k);
}
