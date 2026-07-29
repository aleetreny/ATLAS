/* Entry point: unpack the counts, boot each widget in isolation, then make the
 * page prove it rebuilt the same numbers the generator did.
 *
 * The guard here can be unusually strict. Nothing floating point travels: the
 * file holds a term count and a document frequency, both integers, and both
 * sides compute ln((1+N)/(1+df)) + 1 and 1 + ln(c) from those. So the only
 * discrepancy available is the nine decimal places the reference was rounded
 * to when it was written out, and the tolerance is set from that number rather
 * than from an intuition about what seems reasonable.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initScrollyZipf } from './scrolly-zipf.js';
import { initWeightsWidget } from './weights-widget.js';
import { initFactorialWidget } from './factorial-widget.js';
import { initValueWidget } from './value-widget.js';
import { initSearchWidget } from './search-widget.js';
import { makeIndex, vectorOf, topTerms, idfTable, weightAll, rank, topK } from './tfkit.js';

/* The references were rounded to nine decimals on the way out, so two values
 * that agree exactly can still differ by 5e-10 once read back. Anything above
 * that is a real disagreement about the arithmetic. */
const TOL = 2e-9;

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

function check(data, ix, deep = false) {
  const problems = [];
  const refs = data.refs;
  const idf = idfTable(ix);

  /* ---- the counts decoded to the totals the generator saw */
  const nnz = ix.ptr[ix.docs];
  if (nnz !== data.search.nnz) {
    problems.push(`the index decoded to ${nnz} non-zeros against ${data.search.nnz} offline, so `
      + 'the packing is not being read the way it was written');
  }

  /* ---- the six weightings, term for term and magnitude for magnitude.
     The ranking alone would not catch a formula that is wrong by a constant
     factor, so the sum of squares goes with it. */
  ['binary', 'raw', 'sublinear'].forEach((tf) => {
    [0, 1].forEach((u) => {
      const key = `${tf}_${u}`;
      const ref = refs[key];
      if (!ref) { problems.push(`no reference shipped for ${key}`); return; }
      refs.probe_docs.forEach((d, k) => {
        const vec = vectorOf(ix, d, { tf, useIdf: !!u, normalise: true, idf });
        const mine = topTerms(ix, vec, 10);
        const theirs = ref.top[k];
        mine.forEach((t, i) => {
          if (!theirs[i]) return;
          if (t.word !== theirs[i][0]) {
            problems.push(`${key}, document ${d}, position ${i + 1}: this page says `
              + `"${t.word}" and the reference says "${theirs[i][0]}"`);
          } else if (Math.abs(t.value - theirs[i][1]) > TOL) {
            problems.push(`${key}, document ${d}, "${t.word}": ${t.value.toFixed(10)} here `
              + `against ${theirs[i][1]} offline`);
          }
        });
        let ss = 0;
        for (let i = 0; i < vec.val.length; i++) ss += vec.val[i] * vec.val[i];
        if (Math.abs(ss - ref.sumsq[k]) > 1e-9) {
          problems.push(`${key}, document ${d}: the vector's own length is ${ss.toFixed(10)} `
            + `here against ${ref.sumsq[k]} offline`);
        }
      });
    });
  });

  /* ---- the search, which is the part the reader drives.
     Every one of the shipped queries is re-ranked here against the whole split
     and compared document by document, not just on the top score: a ranking is
     an order, and two orders can share a first place and disagree everywhere
     after it. */
  const W = weightAll(ix, { tf: 'sublinear', useIdf: true, normalise: true });
  const queries = deep ? refs.cosine_top : refs.cosine_top.slice(0, 2);
  queries.forEach((q) => {
    const scores = rank(ix, W, q.q, 'cosine');
    const mine = topK(scores, q.top.length);
    q.top.forEach(([d, s], i) => {
      if (mine[i] !== d) {
        problems.push(`query ${q.q}, position ${i + 1}: document ${mine[i]} here against ${d} `
          + 'offline, so the two rankings differ');
      }
      if (Math.abs(scores[d] - s) > TOL) {
        problems.push(`query ${q.q}, document ${d}: cosine ${scores[d].toFixed(10)} here against `
          + `${s} offline`);
      }
    });
  });

  /* ---- the two claims the article makes with a direction in them, checked
     against the table rather than trusted */
  const F = data.factorial;
  const floor = data.noise.f1_floor;
  const cell = (tf, i, n) => F.cells.find((c) => c.tf === tf && c.idf === i && c.norm === n);
  const normGaps = ['binary', 'raw', 'sublinear'].flatMap(
    (tf) => [false, true].map((u) => cell(tf, u, true).svm_f1 - cell(tf, u, false).svm_f1));
  if (!normGaps.every((v) => v > floor)) {
    problems.push('the page says normalising the length clears the noise floor in all six rows, '
      + `and ${normGaps.filter((v) => v <= floor).length} of them do not`);
  }
  const idfGaps = ['binary', 'raw', 'sublinear'].map(
    (tf) => cell(tf, true, true).svm_f1 - cell(tf, false, true).svm_f1);
  if (idfGaps.some((v) => Math.abs(v) > floor)) {
    problems.push('the page says none of the three rarity-weight effects clears the floor, and '
      + `${idfGaps.filter((v) => Math.abs(v) > floor).length} of them do`);
  }

  /* ---- the entropy ceiling is arithmetic, so it can be checked as arithmetic */
  const V = data.term_value;
  let worstCap = 0;
  V.ceiling.forEach((c) => {
    const p = c.df / V.n_train;
    const cap = -(p * Math.log(p)) - (1 - p) * Math.log(1 - p);
    worstCap = Math.max(worstCap, Math.abs(cap - c.cap));
  });
  if (worstCap > 1e-12) {
    problems.push(`the information ceiling recomputes to within ${worstCap.toExponential(2)}, `
      + 'which is too far for a closed form');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every weight, every ranking and every direction on this page matches the '
      + `reference that owns it${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/tfidf.json').then((r) => r.json());
  const ix = makeIndex(data.search);

  safely('the numbered sentences', () => initProse(data));
  const scrolly = safely('the shape of the problem', () => initScrollyZipf(data));
  safely('three decisions', () => initWeightsWidget(data, ix));
  safely('all twelve', () => initFactorialWidget(data));
  safely('what idf rewards', () => initValueWidget(data));
  safely('the search', () => initSearchWidget(data, ix));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data, ix));
  } else {
    setTimeout(() => check(data, ix), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, ix, deep);
  window.__atlasProseIds = PROSE_IDS;
  return scrolly;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
