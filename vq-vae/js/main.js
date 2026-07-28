/* Entry point.
 *
 * The guards check the three things this page runs itself: the encoder against
 * the codes the generator wrote for the same digits, the quantiser against the
 * indices it chose, and the prior over codes against its own reference logits.
 * All three are compared on the quantised weights, because those are the
 * numbers the browser has.
 */
import { initScrollyVq } from './scrolly-vq.js';
import { initRateWidget } from './rate-widget.js';
import { initCellsWidget } from './cells-widget.js';
import { initHealthWidget } from './health-widget.js';
import { initProse } from './prose.js';
import { makeVq, makePrior } from './vqkit.js';
import { mse } from '../../assets/js/halfweights.js';

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

function check(label, ok, detail) {
  if (!ok) console.warn(`[vq-vae] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, vq, prior) {
  const N = data.net;

  /* 1. the encoder, then the quantiser: the indices this page picks must be the
        indices the generator recorded, and quantising the weights is allowed to
        move a couple of them, which is a number the generator measured */
  let moved = 0;
  let total = 0;
  let worstZ = 0;
  N.reference.forEach((r) => {
    const z = vq.encode(r.pixels);
    for (let i = 0; i < z.length; i++) worstZ = Math.max(worstZ, Math.abs(z[i] - r.z[i]));
    vq.quantise(z).forEach((q, c) => {
      total += 1;
      if (q.code !== r.codes[c]) moved += 1;
    });
  });
  check('encoder', worstZ < 5e-3,
    `a reference code differs from the generator's by ${worstZ.toExponential(2)}, where `
    + `quantising the weights alone accounts for ${N.quant.code_shift}`);
  check('quantiser', moved <= 1,
    `${moved} of ${total} reference assignments differ, where the generator measured `
    + `${N.quant.index_changes} of ${N.quant.index_total} moving under quantisation`);

  /* 2. the decoder, against the pictures the generator decoded from the same
        codes on the same quantised weights */
  let worstPix = 0;
  N.reference.forEach((r) => {
    worstPix = Math.max(worstPix, mse(vq.decode(r.codes), r.decoded));
  });
  check('decoder', worstPix < 1e-6,
    `the decoded picture differs from the generator's by ${worstPix.toExponential(2)} in mean `
    + 'squared error, on the same weights and the same codes');

  /* 3. the prior over codes, against its own reference logits */
  let worstLogit = 0;
  N.prior.reference_grids.forEach((grid, gi) => {
    for (let c = 0; c < vq.cells; c++) {
      const lg = prior.logitsFor(grid, c);
      const ref = N.prior.reference_logits[gi * vq.cells + c];
      for (let j = 0; j < ref.length; j++) {
        worstLogit = Math.max(worstLogit, Math.abs(lg[j] - ref[j]));
      }
    }
  });
  check('prior', worstLogit < 5e-3,
    `the prior's logits differ from the generator's by ${worstLogit.toExponential(2)}`);

  /* 4. the claim the widget makes about the codebook being alive */
  const used = N.usage.filter((u) => u > 0).length;
  check('codebook usage', used === N.usage.length,
    `${N.usage.length - used} of the ${N.usage.length} entries are never used by the model the `
    + 'page ships, which the health widget says should not happen for this variant');

  /* 5. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[vq-vae] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/vqvae.json').then((r) => r.json());
  const vq = makeVq(data);
  const prior = makePrior(data);

  safely('prose', () => initProse(data));
  safely('the plane', () => initScrollyVq(data));
  safely('rate against distortion', () => initRateWidget(data));
  safely('the cells', () => initCellsWidget(data, vq, prior));
  safely('codebook health', () => initHealthWidget(data));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, vq, prior));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
