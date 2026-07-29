/* Entry point. The guard here checks a forward pass rather than a table.
 *
 * The page reimplements the encoder, so the thing that can be wrong is the
 * implementation: a transposed weight, a layer norm with the wrong epsilon, a
 * head sliced along the wrong axis. Any of those still produces plausible
 * pictures. What they do not survive is being asked to reproduce the tags the
 * generator's torch model produced on the same sentences, which is what this
 * checks, plus two structural properties that are true of the architecture and
 * not of a particular fit:
 *
 *   every attention row sums to one, because it is a softmax;
 *   without positions, permuting the input permutes the output and does
 *   nothing else, which the browser can verify on its own rather than trusting
 *   the number the generator measured.
 *
 * The second one is the good kind of check: it does not compare against a
 * stored value at all, so it cannot pass by agreeing with a stale reference.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initMatrixWidget } from './matrix-widget.js';
import { initCurvesWidget } from './curves-widget.js';
import { loadModel, forward } from './attnkit.js';

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

function check(data, model, deep = false) {
  const problems = [];
  const sents = deep ? model.sentences : model.sentences.slice(0, 3);

  sents.forEach((s, n) => {
    const res = forward(model, s.ids);

    /* every row of every head is a distribution */
    let worstRow = 0;
    res.maps.forEach((layer) => layer.forEach((head) => head.forEach((row) => {
      worstRow = Math.max(worstRow, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
    })));
    if (worstRow > 1e-9) {
      problems.push(`sentence ${n + 1}: an attention row sums to ${(1 + worstRow).toFixed(9)}, `
        + 'so the softmax is not being applied along the axis it should be');
    }

    /* the tags this page predicts against the ones torch predicted.
       float16 weights and a different order of operations mean the logits are
       not identical, so what is required is the same argmax, which is what the
       reader sees. */
    const mine = res.tags.map((t) => model.tags[t]);
    const theirs = data.examples[n] && data.examples[n].predicted;
    if (theirs) {
      const bad = mine.map((t, i) => (t === theirs[i] ? null : i)).filter((v) => v !== null);
      if (bad.length) {
        problems.push(`sentence ${n + 1}: this page tags ${bad.length} word(s) differently from `
          + `the generator (${bad.map((i) => `${s.words[i]}: ${mine[i]} against ${theirs[i]}`).join('; ')})`);
      }
    }

    /* permutation blindness, verified here rather than taken on trust */
    const perm = s.ids.map((_, i) => i).sort((a, b) => ((a * 7919) % 101) - ((b * 7919) % 101));
    const a = forward(model, s.ids, { positional: false });
    const b = forward(model, perm.map((i) => s.ids[i]), { positional: false });
    let worst = 0;
    perm.forEach((orig, pos) => {
      for (let i = 0; i < model.d; i++) {
        worst = Math.max(worst, Math.abs(a.hidden[orig][i] - b.hidden[pos][i]));
      }
    });
    if (worst > 1e-6) {
      problems.push(`sentence ${n + 1}: with the positions off, permuting the input changes the `
        + `output by ${worst.toExponential(2)}, and the page says the layer is a function of a set`);
    }
    /* and the control: with positions on, it had better change */
    const c = forward(model, s.ids, { positional: true });
    const dd = forward(model, perm.map((i) => s.ids[i]), { positional: true });
    let moved = 0;
    perm.forEach((orig, pos) => {
      for (let i = 0; i < model.d; i++) {
        moved = Math.max(moved, Math.abs(c.hidden[orig][i] - dd.hidden[pos][i]));
      }
    });
    if (moved < 1e-3) {
      problems.push(`sentence ${n + 1}: with the positions on, permuting the input barely changes `
        + `the output (${moved.toExponential(2)}), so the positional encoding is doing nothing `
        + 'and the contrast the page draws is not there');
    }
  });

  /* the direction the article claims about the scaling */
  const S = data.scaling.rows;
  const last = S[S.length - 1];
  if (!(last.entropy_scaled > last.entropy_raw)) {
    problems.push('the page says the undivided scores collapse as the head grows, and at the '
      + `largest head measured the undivided entropy is ${last.entropy_raw.toFixed(4)} against `
      + `${last.entropy_scaled.toFixed(4)}`);
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the encoder in this page reproduces the generator\'s answers and is provably '
      + `order-blind without positions${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/attention.json').then((r) => r.json());
  const model = loadModel(data.model);
  /* The previous article's numbers, for the joint tables. Optional on purpose:
     if it is not there the prose says so instead of inventing rows. */
  let seq = null;
  try {
    seq = await fetch('../sequences/data/sequences.json').then((r) => (r.ok ? r.json() : null));
  } catch (err) {
    seq = null;
  }

  safely('the numbered sentences', () => initProse(data, seq));
  safely('who looks at whom', () => initMatrixWidget(data, model));
  safely('two lines of the formula', () => initCurvesWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data, model));
  } else {
    setTimeout(() => check(data, model), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, model, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
