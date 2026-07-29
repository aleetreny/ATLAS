/* Collapsed Gibbs sampling for LDA, in the browser, token by token.
 *
 * This is the same loop the generator runs, written a second time, and the
 * word "same" is doing real work: the random number generator is a linear
 * congruential one written identically on both sides, the counts are integers,
 * and the normalising sum is a sequential loop rather than a library call.
 *
 * That last detail is the whole reason this file looks the way it does.
 * numpy's `sum` over eight numbers is a tree, not a left-to-right loop, and a
 * tree and a loop disagree in the final bit. That bit decides which side of
 * the sampled uniform a topic boundary falls on, so every few thousand tokens
 * the two implementations would pick different topics and the chains would
 * separate for good, with nothing actually wrong anywhere. Written as a loop
 * on both sides, the two produce identical integers forever, and the guard can
 * demand exactly that instead of a tolerance.
 */
import { decodeU16, decodeU32 } from '../../assets/js/textkit.js';

export function makeCorpus(S) {
  const idx = decodeU16(S.idx, S.nnz);
  const cnt = decodeU16(S.cnt, S.nnz);
  const ptr = decodeU32(S.ptr, S.docs + 1);
  /* Unfold to one entry per occurrence: LDA gives a topic to each occurrence
   * of a word, not to each distinct word. */
  const docOf = new Int32Array(S.tokens);
  const wordOf = new Int32Array(S.tokens);
  let n = 0;
  for (let d = 0; d < S.docs; d++) {
    for (let k = ptr[d]; k < ptr[d + 1]; k++) {
      for (let c = 0; c < cnt[k]; c++) {
        docOf[n] = d;
        wordOf[n] = idx[k];
        n++;
      }
    }
  }
  return {
    docs: S.docs,
    words: S.words,
    nWords: S.words.length,
    labels: S.labels,
    titles: S.titles,
    docOf,
    wordOf,
    nTok: n,
  };
}

/** A sampler in its initial state, ready to be swept. */
export function newSampler(corpus, { k = 8, alpha = 0.5, beta = 0.01, seed = 19 } = {}) {
  let state = seed >>> 0;
  const rnd = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };

  const { nTok, nWords, docs, docOf, wordOf } = corpus;
  const z = new Int32Array(nTok);
  const ndk = new Int32Array(docs * k);
  const nwk = new Int32Array(nWords * k);
  const nk = new Int32Array(k);
  for (let i = 0; i < nTok; i++) {
    let t = Math.floor(rnd() * k);
    if (t >= k) t = k - 1;
    z[i] = t;
    ndk[docOf[i] * k + t] += 1;
    nwk[wordOf[i] * k + t] += 1;
    nk[t] += 1;
  }

  return {
    k,
    alpha,
    beta,
    corpus,
    z,
    ndk,
    nwk,
    nk,
    sweeps: 0,
    wbeta: nWords * beta,
    p: new Float64Array(k),
    rnd,
    getState: () => state,
  };
}

/** One full pass over every token, in the order the generator uses. */
export function sweep(S) {
  const { k, alpha, beta, wbeta, z, ndk, nwk, nk, p, rnd } = S;
  const { nTok, docOf, wordOf } = S.corpus;
  for (let i = 0; i < nTok; i++) {
    const d = docOf[i] * k;
    const w = wordOf[i] * k;
    let t = z[i];
    ndk[d + t] -= 1;
    nwk[w + t] -= 1;
    nk[t] -= 1;
    let tot = 0;
    for (let c = 0; c < k; c++) {
      const v = (ndk[d + c] + alpha) * (nwk[w + c] + beta) / (nk[c] + wbeta);
      p[c] = v;
      tot += v;
    }
    const u = rnd() * tot;
    let acc = 0;
    t = k - 1;
    for (let c = 0; c < k; c++) {
      acc += p[c];
      if (u < acc) { t = c; break; }
    }
    z[i] = t;
    ndk[d + t] += 1;
    nwk[w + t] += 1;
    nk[t] += 1;
  }
  S.sweeps += 1;
  return S;
}

/** Topic-word distributions, derived from the counts the same way Python does. */
export function phiOf(S) {
  const { k, beta, nwk } = S;
  const n = S.corpus.nWords;
  const out = [];
  for (let t = 0; t < k; t++) {
    const row = new Float64Array(n);
    let tot = 0;
    for (let w = 0; w < n; w++) { const v = nwk[w * k + t] + beta; row[w] = v; tot += v; }
    for (let w = 0; w < n; w++) row[w] /= tot;
    out.push(row);
  }
  return out;
}

/** Document-topic mixtures. */
export function thetaOf(S) {
  const { k, alpha, ndk } = S;
  const out = [];
  for (let d = 0; d < S.corpus.docs; d++) {
    const row = new Float64Array(k);
    let tot = 0;
    for (let t = 0; t < k; t++) { const v = ndk[d * k + t] + alpha; row[t] = v; tot += v; }
    for (let t = 0; t < k; t++) row[t] /= tot;
    out.push(row);
  }
  return out;
}

export function topWords(phi, words, n = 8) {
  return phi.map((row) => Array.from(row.keys())
    .sort((a, b) => (row[b] - row[a]) || (a - b))
    .slice(0, n)
    .map((j) => ({ word: words[j], p: row[j] })));
}

/**
 * log p(w, z | alpha, beta), the quantity the sampler climbs.
 *
 * Written out because the honest progress number for a Gibbs chain is the
 * joint, not a training perplexity: a training perplexity also falls when the
 * model is doing nothing but memorising.
 */
export function jointLoglik(S) {
  const { k, alpha, beta, ndk, nwk, nk } = S;
  const n = S.corpus.nWords;
  const D = S.corpus.docs;
  let ll = D * (lgamma(k * alpha) - k * lgamma(alpha));
  for (let d = 0; d < D; d++) {
    let tot = 0;
    for (let t = 0; t < k; t++) { ll += lgamma(ndk[d * k + t] + alpha); tot += ndk[d * k + t]; }
    ll -= lgamma(tot + k * alpha);
  }
  ll += k * (lgamma(n * beta) - n * lgamma(beta));
  for (let w = 0; w < n; w++) {
    for (let t = 0; t < k; t++) ll += lgamma(nwk[w * k + t] + beta);
  }
  for (let t = 0; t < k; t++) ll -= lgamma(nk[t] + n * beta);
  return ll;
}

/* Lanczos approximation to log gamma. Good to about 1e-13 in the range this
 * uses, which is far tighter than anything the page prints, and the loglik is
 * a progress curve rather than a published constant. */
const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7];

export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  const z = x - 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < LANCZOS.length; i++) a += LANCZOS[i] / (z + i + 1);
  const t = z + LANCZOS.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * How well the mixtures recover a set of known labels.
 *
 * Three numbers because they answer three different questions and no one of
 * them is enough: the adjusted Rand index of assigning each document to its
 * heaviest topic, the purity of that same partition, and the accuracy of the
 * best one-to-one matching between topics and categories, which is the one
 * that answers "is there one topic per category".
 */
export function alignScore(theta, labels, catNames) {
  const k = theta[0].length;
  const cats = catNames || [...new Set(labels)].sort();
  const y = labels.map((l) => cats.indexOf(l));
  const hard = theta.map((row) => {
    let b = 0;
    for (let t = 1; t < row.length; t++) if (row[t] > row[b]) b = t;
    return b;
  });
  const M = Array.from({ length: k }, () => new Array(cats.length).fill(0));
  hard.forEach((t, i) => { M[t][y[i]] += 1; });
  const n = labels.length;
  const purity = M.reduce((a, r) => a + Math.max(...r), 0) / n;

  /* Adjusted Rand index from the contingency table. */
  const c2 = (v) => (v * (v - 1)) / 2;
  let sumIJ = 0;
  M.forEach((r) => r.forEach((v) => { sumIJ += c2(v); }));
  const a = M.map((r) => r.reduce((p, q) => p + q, 0));
  const b = cats.map((_, j) => M.reduce((p, r) => p + r[j], 0));
  const sumA = a.reduce((p, v) => p + c2(v), 0);
  const sumB = b.reduce((p, v) => p + c2(v), 0);
  const exp = (sumA * sumB) / c2(n);
  const mx = (sumA + sumB) / 2;
  const ari = mx === exp ? 0 : (sumIJ - exp) / (mx - exp);

  /* Best one-to-one matching. k is 8 here, so the exact assignment is a small
   * search rather than an approximation: greedy would be a different number
   * and calling it "the best matching" would be false. */
  const used = new Array(cats.length).fill(false);
  let bestSum = 0;
  const walk = (t, acc) => {
    if (t === k) { bestSum = Math.max(bestSum, acc); return; }
    let any = false;
    for (let j = 0; j < cats.length; j++) {
      if (used[j]) continue;
      any = true;
      used[j] = true;
      walk(t + 1, acc + M[t][j]);
      used[j] = false;
    }
    if (!any) { bestSum = Math.max(bestSum, acc); }
  };
  if (k <= 8 && cats.length <= 8) walk(0, 0);
  return { ari, purity, oneToOne: bestSum / n, table: M, cats };
}
