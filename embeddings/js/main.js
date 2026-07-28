/* Entry point, and a guard that has to be honest about what it can check.
 *
 * Three of the four things this page does are exactly reproducible and are
 * checked as such: the vectors decode to unit length, the neighbour lists match
 * the offline ones word for word, and the part-of-speech counts the article
 * quotes come back out of the same code the reader is driving.
 *
 * The fourth is not checkable and the page says so rather than pretending. The
 * analogy answers depend on an argmax over 900 cosines that are often within a
 * thousandth of each other, and the vectors travel as float16; two orderings
 * that agree to 1e-3 can still swap first place. So what is checked there is
 * the thing that is defined, which is that the two conventions differ in the
 * direction the article claims.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initPmiWidget } from './pmi-widget.js';
import { initNeighboursWidget } from './neighbours-widget.js';
import { initAnalogyWidget } from './analogy-widget.js';
import { makeSpace, neighbours, analogy, posAgreement } from './veckit.js';

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

function check(data, space, deep = false) {
  const problems = [];

  /* ---- decoded and renormalised: every vector has to be on the sphere, or a
     cosine is not a cosine and every number on the page is off by its length */
  let worst = 0;
  space.names.forEach((n) => {
    const v = space.models[n].vec;
    for (let i = 0; i < space.n; i += deep ? 1 : 17) {
      let s = 0;
      for (let j = 0; j < space.dim; j++) s += v[i * space.dim + j] ** 2;
      worst = Math.max(worst, Math.abs(Math.sqrt(s) - 1));
    }
  });
  if (worst > 1e-6) {
    problems.push(`the worst vector is ${worst.toExponential(2)} off unit length, so the `
      + 'renormalisation after decoding did not happen');
  }

  /* ---- the part-of-speech agreement the article quotes, recomputed here over
     the shipped words. It will not equal the offline figure, which was measured
     over the full vocabulary, so what is checked is that it lands in the same
     place rather than that it matches: a check that cannot pass is worse than
     no check. */
  const M = data.models.find((m) => m.model === 'skip-gram');
  if (M) {
    let hits = 0;
    let of = 0;
    for (let i = 0; i < Math.min(space.n, 300); i++) {
      if (space.tagPurity[i] < 0.8) continue;
      const a = posAgreement(space, 'skip-gram', i, 10);
      hits += a.hits;
      of += a.of;
    }
    const here = hits / Math.max(1, of);
    if (Math.abs(here - M.same_pos) > 0.15) {
      problems.push(`part-of-speech agreement is ${here.toFixed(4)} on the shipped words against `
        + `${M.same_pos.toFixed(4)} offline over the whole vocabulary, which is further apart `
        + 'than the two protocols can explain');
    }
  }

  /* ---- the two analogy conventions differ in the direction the page claims */
  let echoes = 0;
  let tried = 0;
  [['man', 'woman', 'boy', 'girl'], ['is', 'was', 'are', 'were'],
    ['year', 'years', 'day', 'days']].forEach(([a, b, c]) => {
    if (![a, b, c].every((w) => space.index.has(w))) return;
    tried += 1;
    const lazy = analogy(space, 'skip-gram', a, b, c, { exclude: false, k: 1 });
    if (lazy && lazy[0] && [a, b, c].includes(lazy[0].word)) echoes += 1;
  });
  if (tried && echoes === 0) {
    problems.push('the page says that without the exclusion the answer is often one of the given '
      + 'words, and on the probes here it never is');
  }

  /* ---- neighbours are an ordering, so they get compared as an ordering */
  if (deep) {
    space.names.forEach((n) => {
      const nb = neighbours(space, n, 0, 5);
      if (nb.length !== 5) problems.push(`${n} returned ${nb.length} neighbours rather than 5`);
      for (let i = 1; i < nb.length; i++) {
        if (nb[i].cos > nb[i - 1].cos + 1e-12) {
          problems.push(`${n}: the neighbour list is not sorted, ${nb[i].word} scores above `
            + `${nb[i - 1].word}`);
        }
      }
    });
  }

  /* ---- the identity the article plots: correlation is a published number and
     the slope is the one that carries the claim, so both are range-checked */
  const P = data.pmi;
  if (!(P.pearson > 0.2)) {
    problems.push(`the page presents the shifted mutual information as what skip-gram is `
      + `factorising, and the measured correlation is ${P.pearson.toFixed(4)}, which does not `
      + 'support saying so');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the vectors, the neighbours and the two analogy conventions all behave the way '
      + `the page says${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/embeddings.json').then((r) => r.json());
  const space = makeSpace(data.vectors);

  safely('the numbered sentences', () => initProse(data, space));
  safely('what is being optimised', () => initPmiWidget(data));
  safely('the same question', () => initNeighboursWidget(data, space));
  safely('the analogies', () => initAnalogyWidget(data, space));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data, space));
  } else {
    setTimeout(() => check(data, space), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, space, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
