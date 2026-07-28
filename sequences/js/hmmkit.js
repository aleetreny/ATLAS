/* The hidden Markov model, running in the browser, including Viterbi.
 *
 * The whole model travels: three tables of log probabilities, which is all a
 * hidden Markov model is. That makes it the smallest complete model on this
 * site by a wide margin, and it is worth saying so, because the article's
 * argument is that a great deal of what a tagger needs is already in tables
 * this size.
 *
 * Everything is in logarithms rather than probabilities. That is not an
 * optimisation: the product of two hundred probabilities underflows to zero
 * long before a sentence ends, so in probability space the algorithm returns
 * whatever the first zero decided and gives no sign that it did.
 */
import { decodeHalf } from '../../assets/js/textkit.js';

export function makeHmm(M) {
  const T = M.n_tags;
  const V = M.n_words;
  return {
    tags: M.tags,
    words: M.words,
    index: new Map(M.words.map((w, i) => [w, i])),
    T,
    V,
    logStart: Float64Array.from(M.log_start),
    logTrans: decodeHalf(M.log_trans, T * T),
    logEmit: decodeHalf(M.log_emit, T * V),
  };
}

export function wordId(hmm, w) {
  const i = hmm.index.get(w.toLowerCase());
  return i === undefined ? 0 : i;
}

/**
 * Viterbi, with the trellis kept so it can be drawn.
 *
 * Returns the best path and the full table of partial scores, because the
 * point of the widget is that the reader can see the algorithm considering the
 * paths it eventually throws away. A version that returned only the answer
 * would be shorter and would not be worth drawing.
 */
export function viterbi(hmm, ids) {
  const { T } = hmm;
  const n = ids.length;
  const delta = new Float64Array(n * T);
  const back = new Int32Array(n * T);
  for (let t = 0; t < T; t++) {
    delta[t] = hmm.logStart[t] + hmm.logEmit[t * hmm.V + ids[0]];
    back[t] = -1;
  }
  for (let i = 1; i < n; i++) {
    for (let t = 0; t < T; t++) {
      let best = -Infinity;
      let arg = 0;
      for (let p = 0; p < T; p++) {
        const v = delta[(i - 1) * T + p] + hmm.logTrans[p * T + t];
        if (v > best) { best = v; arg = p; }
      }
      delta[i * T + t] = best + hmm.logEmit[t * hmm.V + ids[i]];
      back[i * T + t] = arg;
    }
  }
  const path = new Int32Array(n);
  let best = -Infinity;
  for (let t = 0; t < T; t++) {
    if (delta[(n - 1) * T + t] > best) { best = delta[(n - 1) * T + t]; path[n - 1] = t; }
  }
  for (let i = n - 1; i > 0; i--) path[i - 1] = back[i * T + path[i]];
  return { path, delta, back, score: best, n };
}

/**
 * The greedy path: at each step take the tag that looks best right there.
 *
 * This exists so the widget can show the two disagreeing. Greedy is what most
 * people picture when they picture tagging, and the gap between it and Viterbi
 * is exactly the value of planning ahead, on this sentence, in this model.
 */
export function greedy(hmm, ids) {
  const { T } = hmm;
  const n = ids.length;
  const path = new Int32Array(n);
  let prev = -1;
  for (let i = 0; i < n; i++) {
    let best = -Infinity;
    let arg = 0;
    for (let t = 0; t < T; t++) {
      const v = (prev < 0 ? hmm.logStart[t] : hmm.logTrans[prev * T + t])
        + hmm.logEmit[t * hmm.V + ids[i]];
      if (v > best) { best = v; arg = t; }
    }
    path[i] = arg;
    prev = arg;
  }
  return path;
}

/** The log probability the model assigns to one specific tagging. */
export function scorePath(hmm, ids, path) {
  const { T } = hmm;
  let s = hmm.logStart[path[0]] + hmm.logEmit[path[0] * hmm.V + ids[0]];
  for (let i = 1; i < ids.length; i++) {
    s += hmm.logTrans[path[i - 1] * T + path[i]] + hmm.logEmit[path[i] * hmm.V + ids[i]];
  }
  return s;
}

/** Emission log probabilities for one word, over every tag. */
export function emissionsFor(hmm, id) {
  const out = new Float64Array(hmm.T);
  for (let t = 0; t < hmm.T; t++) out[t] = hmm.logEmit[t * hmm.V + id];
  return out;
}
