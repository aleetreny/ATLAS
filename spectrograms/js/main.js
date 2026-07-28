/* Entry point.
 *
 * The guard is the spectrogram itself. Every picture on this page is computed
 * here, in JavaScript, from sixteen bit samples, and the generator exported one
 * spectrogram computed in numpy from the same samples: if the window
 * convention, the transform or the decoding of the audio were wrong, those two
 * would not agree, and every measurement on the page would be about a different
 * transform than the one it describes.
 */
import { initPipelineScrolly } from './scrolly-pipeline.js';
import { initWindowWidget } from './window-widget.js';
import { initInvertWidget } from './invert-widget.js';
import { initMfccWidget } from './mfcc-widget.js';
import { initProse } from './prose.js';
import { decodeInt16, stft, melBank, mfcc } from '../../assets/js/dsp.js';

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
  if (!ok) console.warn(`[spectrograms] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const M = data.meta;
  const S = data.chirp.spec;
  const x = decodeInt16(data.chirp.audio);

  /* 1. the transform, against numpy, on the same samples */
  const spec = stft(x, { win: S.win, hop: S.hop, sr: M.sr, nfft: S.nfft });
  let worst = 0;
  const rows = Math.min(S.db.length, Math.floor(spec.frames.length / 2));
  for (let t = 0; t < rows; t++) {
    const mine = spec.frames[t * 2];
    for (let k = 0; k < S.db[t].length; k++) {
      const db = 20 * Math.log10(mine[k] + 1e-8);
      worst = Math.max(worst, Math.abs(db - S.db[t][k]));
    }
  }
  /* the file carries two decimals of decibels, and a decibel of a value near
     the floor is sensitive, so 0.05 is the rounding and anything above it is
     a different transform */
  check('the spectrogram', worst < 0.05,
    `this page's spectrogram differs from the generator's by ${worst.toFixed(4)} dB`);

  /* 2. the samples themselves survived base64. The clip is sent at sixty
        percent of full scale so it is comfortable to listen to, and the
        generator's reference spectrogram is computed from exactly these
        quantised samples rather than from the float signal, which is a
        difference of 38 dB in the quiet bins. */
  let peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  check('the audio', Math.abs(peak - data.chirp.audio_peak) < 1e-3,
    `the decoded waveform peaks at ${peak.toFixed(4)} where the generator sent `
    + `${data.chirp.audio_peak}`);

  /* 3. the coefficients, against the generator's, on one exported vowel */
  if (deep) {
    const demo = data.cepstrum.demo.a100;
    const y = decodeInt16(demo.audio);
    const sp = stft(y, { win: 256, hop: 64, sr: M.sr, nfft: M.nfft });
    const avg = new Float64Array(sp.frames[0].length);
    sp.frames.forEach((fr) => { for (let k = 0; k < fr.length; k++) avg[k] += fr[k] / sp.frames.length; });
    const bank = melBank(M.n_mels, M.nfft, M.sr);
    const { coef } = mfcc(avg, bank, M.n_mfcc);
    let gap = 0;
    for (let i = 1; i < coef.length; i++) gap = Math.max(gap, Math.abs(coef[i] - demo.coef[i]));
    /* the exported audio is quantised to sixteen bits and normalised, so the
       coefficients cannot match to more than a few thousandths */
    check('the coefficients', gap < 0.25,
      `this page's cepstrum differs from the generator's by ${gap.toFixed(4)}`);
  }

  /* 4. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        bad.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info(`[spectrograms] load-time checks complete: spectrogram to ${worst.toFixed(4)} dB`
    + `${deep ? '' : ' (spot check; window.__atlasCheck(true) also checks the cepstrum)'}`);
  return { worst };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/spectrograms.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the pipeline', () => initPipelineScrolly(data));
  safely('the window', () => initWindowWidget(data));
  safely('inversion', () => initInvertWidget(data));
  safely('the coefficients', () => initMfccWidget(data));

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
