/* Entry point.
 *
 * The two identities this article rests on are checkable in the browser from
 * the export alone, and they are different in kind: efficiency is forced by
 * the definition and would hold for a wrong answer whose errors cancel, while
 * agreement with the additive model's own terms would not. The guard checks
 * both and says which is which, because a page that only checked the first
 * would be checking nothing.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initExactWidget } from './exact-widget.js';
import { initKernelWidget } from './kernel-widget.js';
import { initTwoWidget } from './two-widget.js';
import { initLimeWidget } from './lime-widget.js';

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
  const K = data.answer_key;

  /* ---- efficiency, recomputed per instance.
     Values travelled at six decimals and there are ten of them, so the floor
     under this sum is about 5e-6 and anything above it is a real gap. */
  K.rows.forEach((row) => {
    const total = row.phi.reduce((s, v) => s + v, 0);
    if (Math.abs(total - (row.pred - K.base)) > 1e-5) {
      problems.push(`bottle ${row.i}: the attributions sum to ${total.toFixed(6)} and the `
        + `prediction minus the average is ${(row.pred - K.base).toFixed(6)}`);
    }
    /* ---- and the part that is not forced: each value equals the additive
       model's own centred term. */
    row.phi.forEach((v, j) => {
      if (Math.abs(v - row.key[j]) > 1e-5) {
        problems.push(`bottle ${row.i}, column ${j}: enumeration gives ${v} and the closed `
          + `form gives ${row.key[j]}`);
      }
    });
  });
  if (K.worst > 1e-9) {
    problems.push(`the export reports a worst disagreement of ${K.worst} with the answer key`);
  }

  /* ---- the interaction split, which is a property of the axioms */
  data.interaction.rows.forEach((row) => {
    /* Tres millonésimas y no una: los tres números de esta comparación
     * viajaron redondeados a seis decimales, así que la suma de dos de ellos
     * puede alejarse del tercero por dos unidades del último dígito. Con 1e-6
     * el guardia disparaba sobre datos correctos, que es la forma más cara de
     * tener un guardia. */
    if (Math.abs(row.phi0 - (row.main0 + row.share)) > 3e-6) {
      problems.push(`interaction instance ${row.i}: ${row.phi0} against a closed form of `
        + `${row.main0 + row.share}`);
    }
    if (Math.abs(row.share - row.share1) > 3e-6) {
      problems.push(`the interaction is supposed to split in half and the two shares are `
        + `${row.share} and ${row.share1}`);
    }
  });

  /* ---- the directions the prose states */
  const KE = data.kernel.rows;
  for (let i = 1; i < KE.length; i++) {
    if (KE[i].err > KE[i - 1].err * 1.15) {
      problems.push(`sampling more coalitions should not make the error worse, and it does `
        + `between ${KE[i - 1].m} and ${KE[i].m}`);
    }
  }
  const T = data.two_shaps;
  if (!(T.max_gap > 0.02)) {
    problems.push('the two value functions agree here, so that section has no subject');
  }
  const L = data.lime.rows;
  const tops = new Set(L.map((r) => r.top));
  if (tops.size < 2) {
    problems.push('the kernel width never changes which column comes first, and the section '
      + 'says it does');
  }
  if (!(L[0].stability < L[L.length - 1].stability)) {
    problems.push('the page says a narrow neighbourhood is less stable, and the sweep '
      + 'disagrees');
  }
  const MO = data.models;
  if (!(MO.mse_gap < 2 * MO.mse_seed_spread)) {
    problems.push(`the page calls the two models equally accurate and their errors differ by `
      + `${MO.mse_gap} against a seed spread of ${MO.mse_seed_spread}`);
  }
  if (!(MO.same_top < 1)) {
    problems.push('the two models always agree on the top column, so the last section has '
      + 'nothing to report');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every attribution sums to its prediction, every value matches the additive '
      + 'model closed form, and the interaction splits exactly in half');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/attribution.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('one thousand and twenty four coalitions', () => initExactWidget(data));
  safely('what the sampling costs', () => initKernelWidget(data));
  safely('two answers, both called SHAP', () => initTwoWidget(data));
  safely('the other one is not the same thing', () => initLimeWidget(data));
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
