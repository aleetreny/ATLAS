/* The three trained models, unpacked, with neighbours and analogies computed here.
 *
 * The whole point of shipping the vectors rather than a list of neighbours is
 * that the reader can ask about any of the 900 words, in any of the three
 * models, and get the real answer rather than one somebody chose in advance.
 * 900 by 64 is 57,600 numbers per model, so an exhaustive cosine against every
 * word is well under a millisecond and there is no index to be approximate
 * about.
 *
 * Vectors arrive as float16 and are renormalised after decoding, because the
 * rounding to half precision is part of the experiment: the generator measured
 * everything it publishes on these same rounded values, not on the ones the
 * training produced.
 */
import { decodeHalf } from '../../assets/js/textkit.js';

export function makeSpace(V) {
  const n = V.words.length;
  const d = V.dim;
  const models = {};
  Object.entries(V.models).forEach(([name, m]) => {
    const flat = decodeHalf(m.vec, n * d);
    /* Renormalise: half precision moves a unit vector off the sphere by up to
     * about 1e-3, and a cosine is only a cosine on the sphere. */
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += flat[i * d + j] * flat[i * d + j];
      s = Math.sqrt(s) || 1;
      for (let j = 0; j < d; j++) flat[i * d + j] /= s;
    }
    models[name] = { vec: flat, proj: m.proj.xy, explained: m.proj.explained };
  });
  const index = new Map();
  V.words.forEach((w, i) => index.set(w, i));
  return {
    words: V.words,
    counts: V.counts,
    tags: V.tags,
    tagPurity: V.tag_purity,
    dim: d,
    n,
    models,
    index,
    names: Object.keys(models),
  };
}

export function vectorOf(space, model, i) {
  const d = space.dim;
  return space.models[model].vec.subarray(i * d, (i + 1) * d);
}

export function cosine(space, model, i, j) {
  const d = space.dim;
  const v = space.models[model].vec;
  let s = 0;
  for (let k = 0; k < d; k++) s += v[i * d + k] * v[j * d + k];
  return s;
}

/** The k nearest words to word i, excluding itself. */
export function neighbours(space, model, i, k = 10) {
  const { dim: d, n } = space;
  const v = space.models[model].vec;
  const out = [];
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    let s = 0;
    for (let q = 0; q < d; q++) s += v[i * d + q] * v[j * d + q];
    out.push({ j, s });
  }
  /* Ties broken by index, which is frequency order, so the two sides of this
   * article cannot print different lists from identical numbers. */
  out.sort((a, b) => (b.s - a.s) || (a.j - b.j));
  return out.slice(0, k).map((o) => ({
    word: space.words[o.j], cos: o.s, tag: space.tags[o.j], i: o.j,
  }));
}

/**
 * a is to b as c is to what.
 *
 * `exclude` is not a detail. Without dropping a, b and c from the search the
 * answer is very often c itself, because c is the nearest thing to something
 * that is almost c, and every reported accuracy quietly depends on which
 * convention was used. Both are available here so the page can show the gap.
 */
export function analogy(space, model, a, b, c, { exclude = true, k = 5 } = {}) {
  const { dim: d, n, index } = space;
  const ia = index.get(a);
  const ib = index.get(b);
  const ic = index.get(c);
  if (ia === undefined || ib === undefined || ic === undefined) return null;
  const v = space.models[model].vec;
  const q = new Float64Array(d);
  let norm = 0;
  for (let t = 0; t < d; t++) {
    q[t] = v[ib * d + t] - v[ia * d + t] + v[ic * d + t];
    norm += q[t] * q[t];
  }
  norm = Math.sqrt(norm) || 1;
  for (let t = 0; t < d; t++) q[t] /= norm;
  const skip = exclude ? new Set([ia, ib, ic]) : new Set();
  const out = [];
  for (let j = 0; j < n; j++) {
    if (skip.has(j)) continue;
    let s = 0;
    for (let t = 0; t < d; t++) s += q[t] * v[j * d + t];
    out.push({ j, s });
  }
  out.sort((p, r) => (r.s - p.s) || (p.j - r.j));
  return out.slice(0, k).map((o) => ({ word: space.words[o.j], cos: o.s, i: o.j }));
}

/** How many of a word's neighbours carry its own part-of-speech tag. */
export function posAgreement(space, model, i, k = 10) {
  const want = space.tags[i];
  const nb = neighbours(space, model, i, k);
  const hits = nb.filter((x) => x.tag === want).length;
  return { hits, of: nb.length, want };
}
