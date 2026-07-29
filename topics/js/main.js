/* Entry point, and the strictest guard on this site.
 *
 * The page runs a Markov chain. Two implementations of a Markov chain that
 * agree "closely" have not agreed at all: one different draw and the two
 * histories separate for good. So this guard does not use a tolerance. The
 * random number generator is written identically in Python and JavaScript, the
 * counts are integers, and the normalising sum is a sequential loop on both
 * sides, which means the browser has to reproduce the generator's counts
 * exactly, token for token, or something is genuinely different.
 *
 * The checkpoints are at one, two, five and twenty sweeps rather than only at
 * the end, so a failure says WHEN the two chains parted rather than just that
 * they did.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initSamplerWidget } from './sampler-widget.js';
import { initKSweepWidget } from './ksweep-widget.js';
import { initPairWidget } from './pair-widget.js';
import { makeCorpus, newSampler, sweep, jointLoglik, alignScore, thetaOf } from './ldakit.js';

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

function check(data, deep = false) {
  const problems = [];
  const S = data.sampler;
  const corpus = makeCorpus(S);

  if (corpus.nTok !== S.tokens) {
    problems.push(`the corpus unfolded to ${corpus.nTok} tokens against ${S.tokens} offline`);
  }

  const refs = deep ? data.refs : data.refs.slice(0, 2);
  const maxSweeps = Math.max(...refs.map((r) => r.sweeps));
  const st = newSampler(corpus, { k: S.k, alpha: S.alpha, beta: S.beta, seed: S.seed });
  const byS = new Map(refs.map((r) => [r.sweeps, r]));
  let parted = false;
  for (let s = 1; s <= maxSweeps; s++) {
    sweep(st);
    const ref = byS.get(s);
    if (!ref || parted) continue;
    /* Per-topic totals, the sum of squares of the topic-word counts, and the
       sum of squares of the document-topic counts. Three summaries rather than
       the whole matrix: the totals alone would survive a permutation of the
       words within a topic, and the squares would not. */
    const rowsum = new Array(S.k).fill(0);
    const sq = new Array(S.k).fill(0);
    for (let w = 0; w < corpus.nWords; w++) {
      for (let t = 0; t < S.k; t++) {
        const v = st.nwk[w * S.k + t];
        rowsum[t] += v;
        sq[t] += v * v;
      }
    }
    let ndkSq = 0;
    for (let i = 0; i < st.ndk.length; i++) ndkSq += st.ndk[i] * st.ndk[i];

    const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
    if (!same(Array.from(st.nk), ref.nk)) {
      problems.push(`after ${s} sweep(s) the tokens per topic are [${Array.from(st.nk)}] here `
        + `against [${ref.nk}] offline: the two chains have parted`);
      parted = true;
    } else if (!same(rowsum, ref.nkw_rowsum) || !same(sq, ref.nkw_sq) || ndkSq !== ref.ndk_sq) {
      problems.push(`after ${s} sweep(s) the topic totals agree but their squares do not, so the `
        + 'same tokens are being spread differently');
      parted = true;
    }
    const z0 = Array.from(st.z.slice(0, ref.first_z.length));
    if (!parted && !same(z0, ref.first_z)) {
      problems.push(`after ${s} sweep(s) the first ${ref.first_z.length} token assignments differ`);
      parted = true;
    }
    if (!parted && Math.abs(jointLoglik(st) - ref.loglik) > 1e-4) {
      problems.push(`after ${s} sweep(s) the joint log likelihood is ${jointLoglik(st).toFixed(4)} `
        + `here against ${ref.loglik} offline`);
    }
  }

  /* The scoring code is used to make a claim in prose, so it gets checked too:
     a random assignment has to score about nothing against the answer key. */
  const fresh = newSampler(corpus, { k: S.k, alpha: S.alpha, beta: S.beta, seed: S.seed + 1 });
  const al = alignScore(thetaOf(fresh), corpus.labels, data.meta.categories);
  if (Math.abs(al.ari) > 0.02) {
    problems.push(`a random start scores ${al.ari.toFixed(4)} against the answer key, and the `
      + 'page calls that number the floor, so it should be about zero');
  }
  if (al.purity < 0.1 || al.purity > 0.95) {
    problems.push(`a random start has purity ${al.purity.toFixed(4)}, which is not what an `
      + 'unstructured assignment should look like');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the sampler in this page reproduces the generator\'s counts exactly'
      + `${deep ? '' : ' (first checkpoints; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/topics.json').then((r) => r.json());

  safely('the numbered sentences', () => initProse(data));
  safely('the sampler', () => initSamplerWidget(data));
  safely('how many topics', () => initKSweepWidget(data));
  safely('run it again', () => initPairWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
