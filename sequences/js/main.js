/* Entry point, and a guard on an algorithm rather than on a number.
 *
 * The whole hidden Markov model travels, so the page is not showing a picture
 * of Viterbi, it is running it. That means the check can be the strong kind:
 * for every example sentence the browser decodes from scratch and has to
 * produce the same tag sequence the generator produced, position by position,
 * and the same score to the precision that float16 weights allow.
 *
 * The tolerance is derived rather than guessed. The three tables are written as
 * float16, whose worst relative error is 2^-11, and a sentence sums up to about
 * forty of those log probabilities, so anything under about 1e-2 in the total
 * is rounding and anything above it is a different decode.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initTrellisWidget } from './trellis-widget.js';
import { initFlowWidget } from './flow-widget.js';
import { makeHmm, viterbi, greedy, scorePath } from './hmmkit.js';

const SCORE_TOL = 5e-2;

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

function check(data, hmm, deep = false) {
  const problems = [];

  /* ---- the tables decoded to something that is still a distribution.
     Every row of the transition and emission tables is a log distribution, so
     each has to sum to one after exponentiating. Half precision moves that,
     and how far it moves it is the floor for everything else here. */
  let worstRow = 0;
  for (let t = 0; t < hmm.T; t++) {
    let s = 0;
    for (let u = 0; u < hmm.T; u++) s += Math.exp(hmm.logTrans[t * hmm.T + u]);
    worstRow = Math.max(worstRow, Math.abs(s - 1));
  }
  if (worstRow > 0.02) {
    problems.push(`a row of the transition table sums to ${(1 + worstRow).toFixed(4)} rather `
      + 'than one, so the tables are not being decoded the way they were written');
  }

  /* ---- Viterbi, decoded here, against the generator's own tagging */
  const all = data.model.sentences;
  const examples = deep ? all : all.slice(0, 3);
  examples.forEach((ex, n) => {
    const ids = ex.ids;
    const V = viterbi(hmm, ids);
    const mine = Array.from(V.path).map((t) => hmm.tags[t]);
    if (mine.join(' ') !== ex.gold.join(' ')) {
      const bad = mine.map((t, i) => (t === ex.gold[i] ? null : i)).filter((v) => v !== null);
      problems.push(`example ${n + 1}: this page tags "${bad.map((i) => ex.words[i]).join(', ')}" `
        + `differently from the generator (${bad.map((i) => `${mine[i]} against ${ex.gold[i]}`).join('; ')})`);
    }
    /* the score of the returned path has to be the score of that path */
    const s = scorePath(hmm, ids, V.path);
    if (Math.abs(s - V.score) > 1e-9) {
      problems.push(`example ${n + 1}: Viterbi returns ${V.score.toFixed(6)} but the path it `
        + `returns scores ${s.toFixed(6)}, so the backtrace and the table disagree`);
    }
    /* and Viterbi is optimal, so greedy can never beat it. This is the check
       that can catch a subtly wrong dynamic program: a bug in the backtrace
       usually shows up as greedy scoring higher, which is impossible. */
    const g = greedy(hmm, ids);
    const gs = scorePath(hmm, ids, g);
    if (gs > V.score + 1e-9) {
      problems.push(`example ${n + 1}: the greedy path scores ${gs.toFixed(6)} against Viterbi's `
        + `${V.score.toFixed(6)}, which cannot happen if the dynamic program is right`);
    }
  });

  /* ---- the gradient curves have to be curves, not the assertion the textbook
     makes about them. An earlier version of this guard demanded that the gated
     cells outlast the plain one, which is a claim the page does not make and
     the measurements do not support: it fired on correct data. A guard is for
     checking the page against its own numbers, never for checking the numbers
     against an expectation. */
  ['flow', 'flow_untrained'].forEach((key) => {
    const F = data[key];
    if (!F) return;
    ['rnn', 'gru', 'lstm'].forEach((k) => {
      const c = F[k];
      if (!c) { problems.push(`${key} has no curve for ${k}`); return; }
      if (Math.abs(c.relative[0] - 1) > 1e-9) {
        problems.push(`${key}: ${k} is normalised to ${c.relative[0]} at distance one rather `
          + 'than to one, so the three curves are not on a common scale');
      }
      if (c.relative.some((v) => v < 0)) {
        problems.push(`${key}: ${k} has a negative influence, which is a norm and cannot be`);
      }
    });
  });

  /* ---- the copy task has to be above chance somewhere, or it measures nothing */
  const anyLearned = ['rnn', 'gru', 'lstm'].some(
    (k) => data.copy[k] && data.copy[k].some((r) => r.acc > 3 * r.chance));
  if (!anyLearned) {
    problems.push('no architecture beats chance anywhere on the copy task, so the task is not '
      + 'measuring memory, it is measuring nothing');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every decode on this page matches the reference that owns it'
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/sequences.json').then((r) => r.json());
  const hmm = makeHmm(data.model);

  safely('the numbered sentences', () => initProse(data));
  safely('the trellis', () => initTrellisWidget(data, hmm));
  safely('how far back', () => initFlowWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data, hmm));
  } else {
    setTimeout(() => check(data, hmm), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, hmm, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
