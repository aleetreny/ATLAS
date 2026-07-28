/* Entry point, and the one guard on this site that checks a theorem.
 *
 * The article's first claim is that a linear recurrence and a convolution with
 * the right kernel are the same function. That is not a claim about a fit or a
 * corpus, it is arithmetic, so the page recomputes both from the same
 * parameters and requires them to agree to the accumulated rounding of a few
 * hundred additions and no further.
 *
 * The tolerance is derived from what the file contains rather than chosen: the
 * parameters travel as float16, but both computations here read the SAME
 * decoded values, so the rounding of the export cancels out entirely and what
 * is left is only the difference between the two orders of summation. That is
 * why this one can be held to 1e-10 while the guards in the other articles
 * cannot.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initDualityWidget } from './duality-widget.js';
import { initKernelWidget } from './kernel-widget.js';
import { loadDemo, recurrent, kernel, convolve, dualityError } from './ssmkit.js';

const DUALITY_TOL = 1e-10;

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

function check(data, demo, deep = false) {
  const problems = [];

  /* ---- the identity, here, on the shipped system */
  const err = dualityError(demo.a, demo.b, demo.c, demo.x);
  if (!(err.worst < DUALITY_TOL)) {
    problems.push(`the recurrence and the convolution disagree by ${err.worst.toExponential(2)} `
      + 'relative, and they are supposed to be the same function');
  }

  /* ---- and against the generator's own values, which is a different check:
     the one above says the page is self-consistent, this one says the page and
     the generator are computing the same thing */
  const k = kernel(demo.a, demo.b, demo.c, demo.kernelRef.length);
  let worstK = 0;
  demo.kernelRef.forEach((v, t) => { worstK = Math.max(worstK, Math.abs(k[t] - v)); });
  if (worstK > 1e-9) {
    problems.push(`the convolution kernel differs from the generator's by ${worstK.toExponential(2)}`);
  }
  const y = recurrent(demo.a, demo.b, demo.c, demo.x).y;
  let worstY = 0;
  demo.yRef.forEach((v, t) => { worstY = Math.max(worstY, Math.abs(y[t] - v)); });
  if (worstY > 1e-9) {
    problems.push(`the recurrence output differs from the generator's by ${worstY.toExponential(2)}`);
  }

  /* ---- the identity has to survive the whole range the slider covers, not
     just the setting the widget opens on. A control that walks off the region
     where the claim holds is the failure this site has paid for before. */
  if (deep) {
    [0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 0.995, 0.999].forEach((decay) => {
      const a = new Float64Array(demo.n);
      for (let i = 0; i < demo.n; i++) a[i] = decay ** (1 + (i / Math.max(1, demo.n - 1)));
      const e = dualityError(a, demo.b, demo.c, demo.x);
      if (!(e.worst < 1e-8)) {
        problems.push(`at a decay of ${decay} the two forms disagree by ${e.worst.toExponential(2)}`);
      }
    });
  }

  /* ---- the claim the initialisation table makes, checked as arithmetic:
     a multiplier above one cannot produce a decaying kernel */
  data.reach.forEach((r) => {
    if (r.unstable && r.hundredth_at !== null) {
      problems.push(`the ${r.init} initialisation is marked unstable and also reports a decay `
        + 'point, and it cannot be both');
    }
    if (!r.unstable && r.max_abs_a > 1) {
      problems.push(`the ${r.init} initialisation has a multiplier of ${r.max_abs_a} and is not `
        + 'marked unstable');
    }
    /* the curve the widget draws and the number the table prints have to be
       the same measurement, which is exactly the kind of pair that drifts
       apart when a generator is re-run and only one of them is updated */
    if (Math.abs(r.kernel[0] - 1) > 1e-9) {
      problems.push(`the ${r.init} kernel is drawn relative to its own first value and starts at `
        + `${r.kernel[0]} instead of 1`);
    }
    const crossed = r.kernel.findIndex((v) => Math.abs(v) < 0.01);
    if (r.hundredth_at !== null && crossed >= 0 && crossed !== r.hundredth_at) {
      problems.push(`the ${r.init} table row says the kernel reaches a hundredth at step `
        + `${r.hundredth_at}, and the curve above it crosses at ${crossed}`);
    }
  });

  /* ---- the cost table is arithmetic too, so it can be recomputed rather than
     trusted: attention's pairwise term overtakes its own linear term at twice
     the model dimension, which the table should show and not merely assert */
  const C = data.cost;
  const crossing = C.rows.find((r) => r.pairwise > r.projections);
  if (crossing && crossing.length < C.predicted) {
    problems.push(`the page says the quadratic term takes over at ${C.predicted}, and the table `
      + `shows it already ahead at ${crossing.length}`);
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the recurrence and the convolution computed in this page agree to '
      + `${err.worst.toExponential(2)}, which is rounding`
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs the whole range)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/statespace.json').then((r) => r.json());
  const demo = loadDemo(data.demo);
  /* The three previous articles' copy-task rows, for the joint table. Optional:
     if a file is missing the prose says so rather than inventing a column. */
  const grab = async (url) => {
    try {
      const r = await fetch(url);
      return r.ok ? r.json() : null;
    } catch (err) {
      return null;
    }
  };
  const seq = await grab('../sequences/data/sequences.json');
  const attn = await grab('../attention/data/attention.json');

  safely('the numbered sentences', () => initProse(data, seq, attn));
  safely('two computations', () => initDualityWidget(demo));
  safely('three initialisations', () => initKernelWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data, demo));
  } else {
    setTimeout(() => check(data, demo), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, demo, deep);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
