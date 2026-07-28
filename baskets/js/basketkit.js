/* Baskets, as the generator holds them: one bitset per item, with bit t set
 * when transaction t contains that item.
 *
 * That layout is the whole reason this page can enumerate a lattice live. The
 * support of any set of items is the population count of the AND of their
 * bitsets, which is a few hundred machine words whatever the set is, so a
 * thousand subsets cost about as much as one pass over the database.
 *
 * No DOM in here, so node can check it against the generator without a browser.
 */

/* base64 to a Uint32Array, little endian in both directions. The generator
 * writes `int.to_bytes(n, "little")`, so byte 0 holds transactions 0 to 7 with
 * transaction 0 in the low bit, and this reads it back the same way. */
export function decodeBitset(b64, bytes) {
  const bin = atob(b64);
  const words = new Uint32Array(Math.ceil(bytes / 4));
  for (let i = 0; i < bin.length; i++) {
    words[i >> 2] |= bin.charCodeAt(i) << ((i & 3) * 8);
  }
  return words;
}

/* Hamming weight of a 32 bit word, the standard SWAR trick. */
function popcount32(v) {
  let x = v - ((v >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >> 24) & 0x3f;
}

export function popcount(words) {
  let s = 0;
  for (let i = 0; i < words.length; i++) s += popcount32(words[i]);
  return s;
}

export function andInto(out, a, b) {
  for (let i = 0; i < out.length; i++) out[i] = a[i] & b[i];
  return out;
}

export function andCount(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += popcount32(a[i] & b[i]);
  return s;
}

/* A database: the item names, their bitsets, and how many transactions there
 * are. Everything else on this page is a function of these three. */
export function makeDb(names, b64, bytes, n) {
  const sets = b64.map((s) => decodeBitset(s, bytes));
  /* the last byte can hold bits past the end of the database; the generator
     never sets them, but clearing them here means a stray one could not
     silently inflate a support */
  const spare = words(bytes) * 32 - n;
  if (spare > 0) {
    sets.forEach((w) => {
      w[w.length - 1] &= 0xffffffff >>> spare;
    });
  }
  return { names, sets, n, bytes, counts: sets.map(popcount) };
}

function words(bytes) {
  return Math.ceil(bytes / 4);
}

/* Support of a set given as item indices. */
export function supportOf(db, idx) {
  if (idx.length === 0) return db.n;
  if (idx.length === 1) return db.counts[idx[0]];
  let acc = db.sets[idx[0]].slice();
  for (let i = 1; i < idx.length; i++) andInto(acc, acc, db.sets[idx[i]]);
  return popcount(acc);
}

/* Every non-empty subset of the first `k` items, with its exact support.
 *
 * Depth first, carrying the intersection down, so each node costs one AND and
 * one population count rather than a fresh pass. 2^14 is about sixteen thousand
 * nodes, which is nothing; the point of doing it at all is that the pruning can
 * then be checked against the truth instead of trusted. */
export function enumerateAll(db, k, { onNode = null } = {}) {
  const out = new Map();
  const full = new Uint32Array(words(db.bytes)).fill(0xffffffff);
  const spare = words(db.bytes) * 32 - db.n;
  if (spare > 0) full[full.length - 1] &= 0xffffffff >>> spare;
  const walk = (start, mask, chosen) => {
    for (let j = start; j < k; j++) {
      const m = andInto(new Uint32Array(mask.length), mask, db.sets[j]);
      const key = chosen.concat([j]);
      const c = popcount(m);
      out.set(key.join(','), c);
      if (onNode) onNode(key, c);
      walk(j + 1, m, key);
    }
  };
  walk(0, full, []);
  return out;
}

/* Apriori, levelwise, with the prune step written out so it can be counted.
 *
 * The work tally is the reason this is here rather than a library call: what
 * the article is about is how many candidates never get counted. */
export function apriori(db, k, minCount, { maxLen = null } = {}) {
  const work = { generated: 0, pruned: 0, counted: 0, frequent: 0, passes: 0 };
  const levels = [];
  const freq = new Set();
  let level = [];
  for (let j = 0; j < k; j++) {
    work.generated += 1;
    work.counted += 1;
    if (db.counts[j] >= minCount) level.push({ key: [j], mask: db.sets[j], count: db.counts[j] });
  }
  work.passes += 1;
  while (level.length) {
    work.frequent += level.length;
    levels.push(level.map((e) => ({ key: e.key.slice(), count: e.count })));
    level.forEach((e) => freq.add(e.key.join(',')));
    if (maxLen && level[0].key.length >= maxLen) break;
    const next = [];
    for (let a = 0; a < level.length; a++) {
      for (let b = a + 1; b < level.length; b++) {
        const ka = level[a].key;
        const kb = level[b].key;
        let sharedPrefix = true;
        for (let i = 0; i < ka.length - 1; i++) if (ka[i] !== kb[i]) { sharedPrefix = false; break; }
        if (!sharedPrefix) continue;
        const cand = ka.concat([kb[kb.length - 1]]);
        work.generated += 1;
        let ok = true;
        for (let drop = 0; drop < cand.length && ok; drop++) {
          const sub = cand.filter((_, i) => i !== drop);
          if (!freq.has(sub.join(','))) ok = false;
        }
        if (!ok) {
          work.pruned += 1;
          continue;
        }
        work.counted += 1;
        const mask = andInto(new Uint32Array(level[a].mask.length), level[a].mask,
          db.sets[cand[cand.length - 1]]);
        const count = popcount(mask);
        if (count >= minCount) next.push({ key: cand, mask, count });
      }
    }
    work.passes += 1;
    level = next;
  }
  return { levels, work, freq };
}

/* The rules a set of frequent itemsets supports, with the four measures the
 * article compares. `counts` maps a sorted key to its support. */
export function rulesFrom(counts, n, names, { minConf = 0, minSup = 0 } = {}) {
  const out = [];
  counts.forEach((c, key) => {
    const k = key.split(',').map(Number);
    if (k.length < 2) return;
    if (c / n < minSup) return;
    /* every way of splitting the set into an antecedent and a consequent */
    for (let bits = 1; bits < 2 ** k.length - 1; bits++) {
      const ante = k.filter((_, i) => bits & (1 << i));
      const cons = k.filter((_, i) => !(bits & (1 << i)));
      const sa = counts.get(ante.join(','));
      const sc = counts.get(cons.join(','));
      if (sa === undefined || sc === undefined) continue;
      const confidence = c / sa;
      if (confidence < minConf) continue;
      const lift = confidence / (sc / n);
      out.push({
        a: ante.map((i) => names[i]),
        c: cons.map((i) => names[i]),
        aIdx: ante,
        cIdx: cons,
        count: c,
        support: c / n,
        confidence,
        lift,
        leverage: c / n - (sa / n) * (sc / n),
        conviction: confidence < 1 ? (1 - sc / n) / (1 - confidence) : Infinity,
      });
    }
  });
  return out;
}

/* ------------------------------------------------------------------ FP-tree */

/* Rebuild the horizontal transactions from the vertical bitsets, restricted to
 * the items shipped. The tree needs them in that direction and shipping both
 * layouts would be shipping the same data twice. */
export function transactions(db) {
  const out = Array.from({ length: db.n }, () => []);
  db.sets.forEach((w, j) => {
    for (let t = 0; t < db.n; t++) {
      if ((w[t >>> 5] >>> (t & 31)) & 1) out[t].push(j);
    }
  });
  return out;
}

/* Insert every transaction into a prefix tree, its items sorted by `rank`.
 * The node count is what the compression buys, and it depends entirely on the
 * order the items are given in. */
export function fpTree(txs, rank, { limit = null } = {}) {
  const root = { item: -1, count: 0, children: new Map(), depth: 0 };
  let nodes = 1;
  let inserted = 0;
  const use = limit === null ? txs.length : Math.min(limit, txs.length);
  for (let t = 0; t < use; t++) {
    const items = txs[t].filter((i) => rank[i] !== undefined).sort((a, b) => rank[a] - rank[b]);
    if (!items.length) continue;
    inserted += 1;
    let node = root;
    root.count += 1;
    for (const i of items) {
      if (!node.children.has(i)) {
        node.children.set(i, { item: i, count: 0, children: new Map(), depth: node.depth + 1 });
        nodes += 1;
      }
      node = node.children.get(i);
      node.count += 1;
    }
  }
  return { root, nodes, inserted };
}
