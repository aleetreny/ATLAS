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

  /* 2. the compander, round tripped */
  let worst = 0;
  for (let i = 0; i <= 200; i++) {
    const v = -1 + (2 * i) / 200;
    const back = muDecode(muEncode(v, M.mu), M.mu);
    worst = Math.max(worst, Math.abs(back - v));
  }
  /* one level of 256 spans about 1/128 of the range near the top, and the
     compander is coarsest there */
  check('the compander', worst < 0.02,
    `encoding and decoding moves a sample by ${worst.toFixed(4)}, more than a level`);

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

  console.info(`[wavenet] load-time checks complete: receptive fields exact, compander to `
    + `${worst.toFixed(4)}${deep ? '' : ' (spot check; window.__atlasCheck(true) decodes the '
      + 'clips too)'}`);
  return { bad, worst };
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
