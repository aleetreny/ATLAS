/* Entry point: load the measurements, rebuild both trained translators, boot
 * every widget in isolation, and check that the networks running here are the
 * ones the article measured.
 *
 * Two networks rather than one, because the article's first claim is a
 * comparison between them and the reader gets to make that comparison on their
 * own drawing.
 */
import { initProse } from './prose.js';
import { initCondScrolly } from './cond-scrolly.js';
import { initStrip, initCondWidget, initDrawWidget, initPatchWidget } from './pix-widget.js';
import { initCycleWidget } from './cycle-widget.js';
import { loadSheet } from './panels.js';
import { UNet } from './unetkit.js';

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

/* The generator translated two held-out plans with the quantised weights and
 * wrote down ninety-six pixels of each. Both networks have to reproduce them.
 * The tolerance is set by the export: float16 weights through six convolutions
 * put the floor at a few times 1e-4 per pixel. */
function check(data, nets, sheet) {
  const problems = [];
  const M = data.meta;
  const worst = {};
  Object.entries(data.check).forEach(([name, ref]) => {
    let w = 0;
    ref.rgb.forEach((pixels, i) => {
      const plan = sheet.tiles[i];
      const out = nets[name].run(plan, M.res);
      ref.idx.forEach((p, k) => {
        for (let c = 0; c < 3; c++) w = Math.max(w, Math.abs(out[p * 3 + c] - pixels[k][c]));
      });
    });
    worst[name] = w;
    if (w > 6e-3) {
      problems.push(`the ${name} network here differs from the generator's by `
        + `${w.toExponential(2)} on a pixel`);
    }
  });

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('both translators in this tab match the ones the article measured, worst pixel: '
      + Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(', '));
  }
  return { problems, worst };
}

async function boot() {
  const data = await fetch('./data/translation.json').then((r) => r.json());
  const sheet = await loadSheet('./img/strip.png', data.meta.tile, data.meta.sheet_cols);
  const nets = {
    l1: new UNet(data.nets.l1),
    l1gan: new UNet(data.nets.l1gan),
  };

  safely('the numbered sentences', () => initProse(data));
  safely('why blur is correct', () => initCondScrolly(data));
  safely('the detail bars', () => initCondWidget(data));
  safely('the strip', () => initStrip(data, sheet));
  safely('the patch table', () => initPatchWidget(data));
  safely('drawing a plan', () => initDrawWidget(data, nets));
  safely('the unpaired runs', () => initCycleWidget(data));
  renderMath();

  const run = () => check(data, nets, sheet);
  if (window.requestIdleCallback) window.requestIdleCallback(run);
  else setTimeout(run, 400);
  window.__atlasCheck = run;
  window.__atlas = { data, nets, sheet };
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
