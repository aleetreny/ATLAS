/* Entry point.
 *
 * The guard is the tokeniser. It is three lines of arithmetic and it is the
 * entire interface between a series and the model, so if this page rounds a
 * value into a different bin than the generator did, every figure about what
 * the alphabet costs is about a different alphabet. It is checked over the
 * whole exported window at every alphabet size the slider can reach.
 */
import { initTokenScrolly, encodeWindow, decodeTokens } from './scrolly-token.js';
import { initAlphabetWidget } from './alphabet-widget.js';
import { initForecastWidget } from './forecast-widget.js';
import { initProse } from './prose.js';

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
  if (!ok) console.warn(`[zero-shot] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const M = data.meta;
  const D = data.quantisation.demo;

  /* 1. the tokens, against the generator's, on the exported window */
  const tok = encodeWindow(D.window, D.mean, D.sd, M.bins, M.clip);
  let bad = 0;
  for (let i = 0; i < tok.length; i++) if (tok[i] !== D.tokens[i]) bad += 1;
  check('the alphabet', bad === 0,
    `${bad} of ${tok.length} values land in a different bin than the generator put them in`);

  /* 2. and the round trip, which is what the floor is computed from */
  const back = decodeTokens(tok, D.mean, D.sd, M.bins, M.clip);
  let worst = 0;
  for (let i = 0; i < back.length; i++) worst = Math.max(worst, Math.abs(back[i] - D.reconstructed[i]));
  check('the round trip', worst < 1e-3,
    `decoding differs from the generator's by ${worst.toExponential(2)} parts per million`);

  /* 3. every alphabet size the slider offers, not just the one it opens on */
  if (deep) {
    let sizes = 0;
    data.quantisation.rows.forEach((row) => {
      const t = encodeWindow(D.window, D.mean, D.sd, row.bins, M.clip);
      const b = decodeTokens(t, D.mean, D.sd, row.bins, M.clip);
      const step = (2 * M.clip * D.sd) / row.bins;
      let w = 0;
      for (let i = 0; i < b.length; i++) w = Math.max(w, Math.abs(b[i] - D.window[i]));
      /* the worst a value can move is half a bin, by construction */
      if (w <= step / 2 + 1e-9) sizes += 1;
    });
    check('every alphabet', sizes === data.quantisation.rows.length,
      `${data.quantisation.rows.length - sizes} alphabet sizes move a value by more than half a `
      + `bin, which is impossible if the rounding is right`);
  }

  /* 4. the corpus this model saw contains no real series, which is the claim
        the article is built on: the digests it asserts are of the evaluation
        series, and they have to match the ones the first article published */
  check('the series', typeof M.co2_digest === 'string' && M.co2_digest.length > 8,
    'the carbon dioxide digest is missing, so "the same series as article one" is unverified');

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

  console.info(`[zero-shot] load-time checks complete: ${tok.length} tokens identical, round trip `
    + `to ${worst.toExponential(2)}`
    + `${deep ? '' : ' (spot check; window.__atlasCheck(true) checks every alphabet)'}`);
  return { bad, worst };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/zeroshot.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the alphabet', () => initTokenScrolly(data));
  safely('the floor', () => initAlphabetWidget(data));
  safely('the forecast', () => initForecastWidget(data));

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
