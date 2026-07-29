/* Entry point.
 *
 * This is the one article in the branch whose models do not travel. Three
 * transformers with a thirty-thousand-word vocabulary are megabytes of
 * embedding table each, and unlike the attention article there is no small set
 * of sentences the page needs them for: what this article measures is a
 * comparison between training runs, not a forward pass.
 *
 * So the guard checks what a guard can honestly check here, which is that the
 * comparison is a fair one and that the page is not stating a direction the
 * table contradicts. Those are the two ways this article could be wrong.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initArmsWidget } from './arms-widget.js';
import { initObjectiveWidget } from './objective-widget.js';
import { initRateWidget } from './rate-widget.js';

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

function check(data) {
  const problems = [];
  const A = data.arms;

  /* ---- the whole article rests on the arms being comparable, so that is the
     first thing checked rather than something taken on trust */
  const p = A.map((a) => a.params);
  const spread = (Math.max(...p) - Math.min(...p)) / Math.min(...p);
  if (spread > 0.05) {
    problems.push(`the three arms differ by ${(100 * spread).toFixed(1)}% in parameters, and the `
      + 'page presents them as sharing a budget');
  }
  const seen = A.map((a) => a.tokens_seen);
  if (Math.max(...seen) !== Math.min(...seen)) {
    problems.push(`the arms read different numbers of tokens (${seen.join(', ')}), so any `
      + 'difference between them is partly a difference of exposure');
  }

  /* ---- the claim that carries the closing paragraph */
  const gpt = A.find((a) => a.kind === 'gpt');
  const bert = A.find((a) => a.kind === 'bert');
  if (gpt && bert && !(gpt.signal_per_token > bert.signal_per_token)) {
    problems.push('the page says the causal objective extracts more supervision per token than '
      + `the masked one, and the measurements say ${gpt.signal_per_token.toFixed(3)} against `
      + `${bert.signal_per_token.toFixed(3)}`);
  }
  /* the masked arm's signal has to be about its own masking rate, or the
     accounting is wrong somewhere rather than the finding being surprising */
  if (bert && Math.abs(bert.signal_per_token - data.meta.mask_rate) > 0.05) {
    problems.push(`the masked arm reports ${bert.signal_per_token.toFixed(3)} predictions per `
      + `token at a masking rate of ${data.meta.mask_rate}, and those should be about equal`);
  }

  /* ---- the sweep has to actually vary something, or the table is decoration */
  const S = data.rate_sweep;
  const probes = S.map((r) => r.probe);
  if (Math.max(...probes) - Math.min(...probes) < 1e-6) {
    problems.push('every masking rate scores identically, which means the sweep is not sweeping');
  }
  const sup = S.map((r) => r.supervised);
  if (!sup.every((v, i) => i === 0 || v >= sup[i - 1])) {
    problems.push('masking more should never grade the model on fewer predictions, and it does '
      + 'somewhere in the sweep');
  }

  /* ---- the worked examples, checked against what the page says they show.
     These are cheap and they catch the failure that would be hardest to see:
     an example drawn from the wrong arm looks like a perfectly good picture of
     some objective, just not the one named above it. */
  const E = data.examples;
  if (E) {
    if (E.gpt.replaced.length) {
      problems.push('the page says the causal objective destroys nothing, and its example has '
        + `${E.gpt.replaced.length} altered tokens`);
    }
    if (E.gpt.graded.length !== E.gpt.input.length) {
      problems.push('the causal example should be graded on every position it reads, and it is '
        + `graded on ${E.gpt.graded.length} of ${E.gpt.input.length}`);
    }
    if (!(E.bert.graded.length > 0 && E.bert.graded.length < E.bert.input.length)) {
      problems.push('the masked example is graded on every position or on none, and the whole '
        + 'point of the comparison is that it is graded on a fraction');
    }
    if (!E.t5.target || E.t5.target.length < 2) {
      problems.push('the span example has no second sequence, and the page says its supervision '
        + 'lives there');
    }
  }

  /* ---- the claim the slider's caption makes: the five draws share a seed, so
     raising the rate adds placeholders rather than reshuffling them. That is a
     property of the export, and if a future run breaks it the caption becomes
     a lie about a picture the reader is looking at. */
  const X = data.rate_examples;
  if (X && X.length > 1) {
    for (let k = 1; k < X.length; k++) {
      const prev = new Set(X[k - 1].graded);
      const missing = [...prev].filter((p) => !X[k].graded.includes(p));
      if (missing.length) {
        problems.push(`the widget says raising the rate only adds hidden words, and ${missing.length} `
          + `of them come back at ${(100 * X[k].rate).toFixed(0)}%`);
      }
    }
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((pr) => console.warn(`  ${pr}`));
  } else {
    console.log('the three arms share a budget and an exposure, and every direction the page '
      + 'states matches its table');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/pretraining.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('one passage, three objectives', () => initObjectiveWidget(data));
  safely('three questions', () => initArmsWidget(data));
  safely('the rate, swept', () => initRateWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = () => check(data);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
