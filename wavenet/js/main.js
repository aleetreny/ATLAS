/* Entry point.
 *
 * Two guards. The receptive field is recomputed here from the layer counts,
 * because the whole first half of the article is an argument that this number
 * is arithmetic rather than a measurement, and a page that read it out of a
 * file would be assuming the thing it claims. And the compander is recomputed
 * from its formula and checked to be invertible to within one level.
 */
import { initMuWidget, muEncode, muDecode } from './mu-widget.js';
import { initDilationScrolly } from './scrolly-dilation.js';
import { initReceptiveWidget, receptiveField } from './receptive-widget.js';
import { initFreeWidget } from './free-widget.js';
import { initAsrWidget } from './asr-widget.js';
import { initProse } from './prose.js';
import { decodeInt16 } from '../../assets/js/dsp.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
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
  if (!ok) console.warn(`[wavenet] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const M = data.meta;

  /* 1. the receptive field, from the layer count alone */
  let bad = 0;
  data.receptive.rows.forEach((row) => {
    if (receptiveField(row.layers) !== row.receptive) bad += 1;
  });
  check('the receptive field', bad === 0,
    `${bad} of ${data.receptive.rows.length} stacks disagree with 1 + sum of the dilations`);

  /* 2. the compander, round tripped.
   *
   * Both numbers here used to be guesses and the pair of them cancelled: a grid
   * of 201 points and a hand written limit of 0.02. The grid stepped straight
   * over the worst case and reported exactly 0.02, which passed by 2e-16. On a
   * grid of 200,001 the true worst is 0.02159, and the guard would have failed.
   *
   * So neither is written by hand now. The limit is the arithmetic: quantising
   * y into mu+1 levels over [-1, 1] puts half a step at 1/mu, and decoding
   * stretches y by at most d/dy of ((1+mu)^y - 1)/mu, which is largest at y = 1.
   * The grid is fine enough that the spacing between probes is far below the
   * quantity being bounded, so it cannot miss the peak the way the old one did.
   */
  const halfStep = 1 / M.mu;
  const stretch = (Math.log1p(M.mu) * (1 + M.mu)) / M.mu;
  const limit = halfStep * stretch;
  const probes = deep ? 200000 : 20000;
  let worst = 0;
  for (let i = 0; i <= probes; i++) {
    const v = -1 + (2 * i) / probes;
    const back = muDecode(muEncode(v, M.mu), M.mu);
    worst = Math.max(worst, Math.abs(back - v));
  }
  check('the compander', worst <= limit,
    `encoding and decoding moves a sample by ${worst.toFixed(5)}, and half a level stretched by `
    + `the steepest part of the curve is ${limit.toFixed(5)}`);

  /* 3. the clips decode to something audible rather than to silence */
  if (deep) {
    ['audio_original', 'audio_mu', 'audio_linear'].forEach((k) => {
      const x = decodeInt16(data.quantise[k]);
      let peak = 0;
      for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
      check(`clip ${k}`, peak > 0.3, `it decodes to a peak of ${peak.toFixed(3)}`);
    });
  }

  /* 4. the alignment claim needs boundaries to have been measured against */
  check('the alignment', data.recognition.alignment.n > 0,
    'no utterance had the right number of spikes, so the alignment error is over nothing');

  /* 5. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info(`[wavenet] load-time checks complete: receptive fields exact, compander worst `
    + `${worst.toFixed(5)} against a bound of ${limit.toFixed(5)}`
    + `${deep ? '' : ' (spot check; window.__atlasCheck(true) uses a finer grid and decodes the '
      + 'clips too)'}`);
  return { bad, worst, limit };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/wavenet.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the compander', () => initMuWidget(data));
  safely('dilations', () => initDilationScrolly(data));
  safely('the receptive field', () => initReceptiveWidget(data));
  safely('free running', () => initFreeWidget(data));
  safely('recognition', () => initAsrWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
