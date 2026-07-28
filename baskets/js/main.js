/* Entry point: decode the baskets, boot every widget in isolation, and then
 * check this page's counting against the generator's.
 *
 * The guards here are unusually strong for this site, because the objects are
 * integers rather than floating point: a support is a count of baskets, so
 * "agrees to within rounding" is not a thing that can happen. Anything other
 * than exact equality is a bug.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initScrollyLattice } from './scrolly-lattice.js';
import { initLatticeWidget } from './lattice-widget.js';
import { initTreeWidget } from './tree-widget.js';
import { initRulesWidget } from './rules-widget.js';
import {
  makeDb, enumerateAll, apriori, rulesFrom, supportOf, transactions, fpTree,
} from './basketkit.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false,
  });
}

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(data, db, deep = false) {
  const problems = [];
  const G = data.groceries;

  /* ---- the bitsets decoded to the supports the generator counted */
  let wrong = 0;
  G.counts.forEach((c, i) => {
    if (c !== db.counts[i]) wrong += 1;
  });
  if (wrong) {
    problems.push(`${wrong} of the ${G.counts.length} item supports decoded wrong, so the `
      + 'bitsets are not being read the way they were written');
  }
  const pairHere = supportOf(db, [0, 1]);
  if (pairHere !== data.rules.pair_count) {
    problems.push(`the two commonest items share ${pairHere} baskets here against `
      + `${data.rules.pair_count} offline`);
  }

  /* ---- Apriori against exhaustion, on as many items as the check can afford.
     This is the guard that matters: it does not compare the page against the
     generator, it compares the page against the truth. */
  const k = deep ? 14 : 11;
  const all = enumerateAll(db, k);
  if (all.size !== 2 ** k - 1) {
    problems.push(`enumerating ${k} items gave ${all.size} subsets, not ${2 ** k - 1}`);
  }
  let closure = 0;
  all.forEach((c, key) => {
    const idx = key.split(',').map(Number);
    if (idx.length < 2) return;
    for (let d = 0; d < idx.length; d++) {
      const sub = idx.filter((_, i) => i !== d).join(',');
      if (all.get(sub) < c) closure += 1;
    }
  });
  if (closure) {
    problems.push(`downward closure fails ${closure} times, which cannot happen and means the `
      + 'supports are wrong');
  }
  const minCount = data.apriori.min_count;
  const ap = apriori(db, k, minCount);
  const found = new Set(ap.levels.flat().map((e) => e.key.join(',')));
  const brute = new Set([...all.entries()].filter(([, c]) => c >= minCount).map(([q]) => q));
  if (found.size !== brute.size || ![...found].every((q) => brute.has(q))) {
    problems.push(`on ${k} items Apriori found ${found.size} frequent sets and exhaustion found `
      + `${brute.size}: the pruning is losing sets`);
  }

  /* ---- the tallies the widget prints, against the generator's own */
  const refs = deep ? G.ship_ref : G.ship_ref.slice(0, 3);
  refs.forEach((r) => {
    const res = apriori(db, db.names.length, r.count);
    const w = res.work;
    ['frequent', 'counted', 'pruned', 'generated', 'passes'].forEach((field) => {
      if (w[field] !== r.work[field]) {
        problems.push(`at support ${r.support}: ${field} is ${w[field]} here against `
          + `${r.work[field]} offline`);
      }
    });
    const counts = new Map();
    res.levels.forEach((lv) => lv.forEach((e) => counts.set(e.key.join(','), e.count)));
    const rules = rulesFrom(counts, db.n, db.names, { minConf: data.apriori.min_conf });
    if (rules.length !== r.rules) {
      problems.push(`at support ${r.support}: ${rules.length} rules here against ${r.rules}`);
    }
    if (rules.filter((q) => q.lift < 1).length !== r.below_one) {
      problems.push(`at support ${r.support}: the count of rules with lift below one differs`);
    }
  });

  /* ---- the trap rule, item for item */
  const trap = data.rules.trap;
  if (trap) {
    const a = trap.a.map((nm) => db.names.indexOf(nm));
    const c = trap.c.map((nm) => db.names.indexOf(nm));
    if (a.every((v) => v >= 0) && c.every((v) => v >= 0)) {
      const both = supportOf(db, a.concat(c).slice().sort((p, q) => p - q));
      const conf = both / supportOf(db, a);
      const lift = conf / (supportOf(db, c) / db.n);
      if (Math.abs(conf - trap.confidence) > 1e-6 || Math.abs(lift - trap.lift) > 1e-6) {
        problems.push(`the rule the article calls the worst one: confidence ${conf.toFixed(6)} `
          + `and lift ${lift.toFixed(6)} here against ${trap.confidence} and ${trap.lift}`);
      }
    }
  }

  /* ---- the tree: the ordering effect, which is the claim the section makes */
  const txs = transactions(db);
  const rankFreq = {};
  const rankRev = {};
  db.names.forEach((nm, i) => {
    rankFreq[i] = i;
    rankRev[i] = db.names.length - 1 - i;
  });
  const nFreq = fpTree(txs, rankFreq).nodes;
  const nRev = fpTree(txs, rankRev).nodes;
  if (!(nRev > nFreq)) {
    problems.push(`the prefix tree is not smaller with the commonest items first `
      + `(${nFreq} against ${nRev}), which is what the section claims`);
  }

  /* ---- the wine, whose numbers the generator publishes in full */
  const W = data.wine;
  const wdb = makeDb(W.items, W.bitsets, W.bytes, W.n);
  const slots = wdb.counts.reduce((a, b) => a + b, 0);
  if (slots !== W.n * W.basket_size) {
    problems.push(`the wine baskets hold ${slots} item slots here against `
      + `${W.n * W.basket_size}`);
  }
  const wap = apriori(wdb, W.items.length, Math.ceil(W.min_support * W.n), { maxLen: 4 });
  ['frequent', 'counted', 'pruned', 'generated'].forEach((field) => {
    const ref = { frequent: W.frequent, counted: W.counted, pruned: W.pruned,
      generated: W.generated }[field];
    if (wap.work[field] !== ref) {
      problems.push(`the wine's ${field} is ${wap.work[field]} here against ${ref} offline`);
    }
  });

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every count on this page matches the reference that owns it'
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/baskets.json').then((r) => r.json());
  const G = data.groceries;
  const db = makeDb(G.names, G.bitsets, G.bytes, G.n);
  const scrolly = safely('the lattice', () => initScrollyLattice(data, db));
  safely('the numbered sentences', () => initProse(data, scrolly ? scrolly.facts : {
    k: 5, subsets: 31, frequent: 0, cut: 0, minCount: Math.ceil(0.02 * G.n),
    names: G.names.slice(0, 5), singles: G.counts.slice(0, 5),
  }));
  safely('counted and never counted', () => initLatticeWidget(data, db));
  safely('the prefix tree', () => initTreeWidget(data, db));
  safely('the rules', () => initRulesWidget(data, db));
  renderMath();
  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data, db));
  } else {
    setTimeout(() => check(data, db), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, db, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
