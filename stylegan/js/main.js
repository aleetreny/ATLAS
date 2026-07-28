/* Entry point: load the measurements, rebuild the trained generator from its
 * float16 weights, boot every widget in isolation, and then check that the
 * generator running here is the one the article measured.
 *
 * That check is the whole licence for the three live widgets below. They let
 * the reader mix styles, reroll the noise and truncate the latent, and each of
 * those is a claim about a network; if the network in the tab were not the one
 * in the tables, every one of those claims would be about something else.
 */
import { initProse } from './prose.js';
import { initArchScrolly } from './arch-scrolly.js';
import { initHoleWidget, loadSprites } from './hole-widget.js';
import { initMetricsWidget } from './metrics-widget.js';
import { initMixWidget, initMixTable } from './mix-widget.js';
import { initNoiseWidget } from './noise-widget.js';
import { initTruncWidget } from './trunc-widget.js';
import { StyleGen, measure } from './stylekit.js';

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

/* Four latents, rendered offline with every noise input at zero, against the
 * same four rendered here. The tolerance is set by the export: weights are
 * float16, so the accumulation floor is a few times 1e-4 per pixel and
 * anything an order above that is an implementation difference rather than
 * rounding. The mapped latents are checked separately, so a failure says
 * which half of the network to look at. */
function check(data, gen) {
  const problems = [];
  const M = data.meta;
  const K = data.check;
  const nb = M.blocks.length;
  let worstW = 0;
  let worstPx = 0;

  K.z.forEach((z, i) => {
    const w = gen.map(Float32Array.from(z));
    K.w[i].forEach((v, k) => { worstW = Math.max(worstW, Math.abs(v - w[k])); });
    const img = gen.synth(new Array(nb).fill(w), null);
    K.idx.forEach((p, k) => {
      for (let c = 0; c < 3; c++) {
        worstPx = Math.max(worstPx, Math.abs(img[p * 3 + c] - K.rgb[i][k][c]));
      }
    });
  });
  if (worstW > 2e-3) {
    problems.push(`the mapping network here differs from the generator's by ${worstW.toExponential(2)}`);
  }
  if (worstPx > 4e-3) {
    problems.push(`the synthesis network here differs by ${worstPx.toExponential(2)} on a pixel`);
  }

  /* and the measurement functions, which every readout on this page uses */
  const probe = gen.fromZ(Float32Array.from(K.z[0]));
  const m = measure(probe, M.res);
  if (!Number.isFinite(m.size) || !Number.isFinite(m.cx)) {
    problems.push('the measurement functions returned something that is not a number');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log(`the generator in this tab matches the one the article measured: worst mapped `
      + `latent ${worstW.toExponential(1)}, worst pixel ${worstPx.toExponential(1)}`);
  }
  return { problems, worstW, worstPx };
}

async function boot() {
  const data = await fetch('./data/style.json').then((r) => r.json());
  const gen = new StyleGen(data);
  const sheets = await loadSprites(data.meta);

  safely('the numbered sentences', () => initProse(data));
  safely('the hole in the data', () => initHoleWidget(data, sheets));
  safely('where the latent enters', () => initArchScrolly(data));
  safely('path length and separability', () => initMetricsWidget(data));
  safely('style mixing, live', () => initMixWidget(data, gen));
  safely('the mixing table', () => initMixTable(data));
  safely('the noise inputs', () => initNoiseWidget(data, gen));
  safely('truncation', () => initTruncWidget(data, gen));
  renderMath();

  const run = () => check(data, gen);
  if (window.requestIdleCallback) window.requestIdleCallback(run);
  else setTimeout(run, 400);
  window.__atlasCheck = run;
  window.__atlas = { data, gen, sheets };
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
