/* The pieces every article in module 4.1 needs, written once.
 *
 * Three of them are contracts with Python rather than conveniences, and they
 * are the reason this file exists instead of seven copies:
 *
 *  - `tokenize` is a word-for-word mirror of `TOKEN` in `src/utils/nlp_data.py`.
 *    Every count, every vocabulary size and every co-occurrence in this module
 *    is downstream of it, so a difference of one character here moves all of
 *    them at once and no single guard would name the cause.
 *  - `decodeHalf` reads the float16 the generators export. The naive version of
 *    it flushes subnormals to zero; this site already paid for that once, in
 *    the detection article, and embeddings have plenty of small components.
 *  - population statistics, not sample statistics. numpy divides by n-1 by
 *    default and JavaScript divides by whatever you write, and a standard
 *    deviation that misses in the third decimal is almost always this.
 */

/* Lowercase, runs of letters, nothing else. Deliberately blunt: "don't" gives
 * ["don", "t"] and "1987" gives nothing at all. That is a decision the articles
 * report rather than hide, because a tokenizer is the first modelling choice
 * anybody makes about language and almost nobody writes it down. */
const TOKEN = /[a-z]+/g;

export function tokenize(text) {
  return text.toLowerCase().match(TOKEN) || [];
}

/* ------------------------------------------------------------ binary decoding */

/* IEEE 754 binary16 to binary32, subnormals and specials included. */
export function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

function bytesOf(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

/** base64 of little-endian float16 to a Float32Array of `n` values. */
export function decodeHalf(b64, n) {
  const u8 = bytesOf(b64);
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

/** base64 of little-endian uint16 to a Uint16Array: token ids, counts, indices. */
export function decodeU16(b64, n) {
  const u8 = bytesOf(b64);
  return new Uint16Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + 2 * n));
}

/** base64 of little-endian uint32. */
export function decodeU32(b64, n) {
  const u8 = bytesOf(b64);
  return new Uint32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + 4 * n));
}

/** base64 of uint8. */
export function decodeU8(b64) {
  return bytesOf(b64);
}

/* ------------------------------------------------------------- sparse vectors */

/* A document is {idx, val} with idx strictly increasing. Every similarity in
 * the TF-IDF article is a merge over two of these, which is what makes a live
 * search over ten thousand documents possible in a page. */

export function sparseDot(a, b) {
  let i = 0;
  let j = 0;
  let acc = 0;
  while (i < a.idx.length && j < b.idx.length) {
    const d = a.idx[i] - b.idx[j];
    if (d === 0) { acc += a.val[i] * b.val[j]; i++; j++; } else if (d < 0) i++; else j++;
  }
  return acc;
}

export function sparseNorm(a) {
  let acc = 0;
  for (let i = 0; i < a.val.length; i++) acc += a.val[i] * a.val[i];
  return Math.sqrt(acc);
}

export function cosine(a, b) {
  const na = a.norm !== undefined ? a.norm : sparseNorm(a);
  const nb = b.norm !== undefined ? b.norm : sparseNorm(b);
  if (na === 0 || nb === 0) return 0;
  return sparseDot(a, b) / (na * nb);
}

/** Squared euclidean distance between two sparse vectors, merge and tails. */
export function sparseDist2(a, b) {
  let i = 0;
  let j = 0;
  let acc = 0;
  while (i < a.idx.length || j < b.idx.length) {
    const ai = i < a.idx.length ? a.idx[i] : Infinity;
    const bj = j < b.idx.length ? b.idx[j] : Infinity;
    if (ai === bj) { const d = a.val[i] - b.val[j]; acc += d * d; i++; j++; } else if (ai < bj) { acc += a.val[i] * a.val[i]; i++; } else { acc += b.val[j] * b.val[j]; j++; }
  }
  return acc;
}

/* ------------------------------------------------------------ dense numerics */

export function dot(a, b, off = 0, n = a.length) {
  let acc = 0;
  for (let i = 0; i < n; i++) acc += a[i] * b[off + i];
  return acc;
}

export function norm(a) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * a[i];
  return Math.sqrt(acc);
}

/** Softmax in place over a slice, shifted by the max so it cannot overflow. */
export function softmax(v, off = 0, n = v.length) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (v[off + i] > m) m = v[off + i];
  let s = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(v[off + i] - m); v[off + i] = e; s += e; }
  for (let i = 0; i < n; i++) v[off + i] /= s;
  return v;
}

/** Shannon entropy in nats of a distribution that already sums to one. */
export function entropy(p, off = 0, n = p.length) {
  let h = 0;
  for (let i = 0; i < n; i++) { const q = p[off + i]; if (q > 0) h -= q * Math.log(q); }
  return h;
}

/** Mean and population standard deviation: numpy's ddof=0, written on purpose. */
export function meanSd(xs) {
  const n = xs.length;
  if (!n) return { mean: 0, sd: 0 };
  let m = 0;
  for (let i = 0; i < n; i++) m += xs[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = xs[i] - m; v += d * d; }
  return { mean: m, sd: Math.sqrt(v / n) };
}

/** Pearson correlation, population form on both sides. */
export function pearson(xs, ys) {
  const a = meanSd(xs);
  const b = meanSd(ys);
  if (a.sd === 0 || b.sd === 0) return NaN;
  let c = 0;
  for (let i = 0; i < xs.length; i++) c += (xs[i] - a.mean) * (ys[i] - b.mean);
  return c / xs.length / (a.sd * b.sd);
}

/* ----------------------------------------------------------------------- rng */

/* The same linear congruential generator the Python side writes, so a page can
 * reproduce a shuffle or an initialisation instead of shipping its result.
 * numpy's Mersenne Twister does not exist in a browser; this does, on both
 * sides, and it is checked against its Python twin at load time. */
export function lcg(seed) {
  let s = seed >>> 0;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  next.int = (n) => Math.floor(next() * n);
  next.normal = () => {
    const u = Math.max(next(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  return next;
}

/** Fisher-Yates with an explicit source, because d3.shuffle ignores one. */
export function shuffle(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rand.int(i + 1);
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
  return arr;
}

/* ------------------------------------------------------------------ formatting */

export const fmt = {
  n0: (v) => v.toLocaleString('en-US'),
  n1: (v) => v.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
  pc: (v, d = 1) => `${(100 * v).toFixed(d)}%`,
  f2: (v) => v.toFixed(2),
  f3: (v) => v.toFixed(3),
  f4: (v) => v.toFixed(4),
};
