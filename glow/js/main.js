/* Entry point.
 *
 * The guard that carries this page is the round trip. A flow is invertible by
 * construction, so if the page has read the weights or the channel layout
 * wrongly, encoding a held out digit and inverting it will not return it, no
 * matter how plausible the samples look. It is checked against the generator's
 * own round trip on the same quantised weights, and so is the log likelihood.
 */
import { initScrollyGlow } from './scrolly-glow.js';
import { initTrapWidget } from './trap-widget.js';
import { initArmsWidget } from './arms-widget.js';
import { initTempWidget } from './temp-widget.js';
import { initProse } from './prose.js';
import { makeGlow } from './glowkit.js';

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
  if (!ok) console.warn(`[glow] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, glow) {
  const P = data.net.probe;

  /* 1. the log likelihood this page computes, against the generator's, on the
        same images and the same quantised weights */
  let worstLp = 0;
  let worstLd = 0;
  P.pixels.forEach((px, i) => {
    const { logdet } = glow.forward(px);
    worstLd = Math.max(worstLd, Math.abs(logdet - P.logdet[i]));
    worstLp = Math.max(worstLp, Math.abs(glow.logProb(px) - P.logp[i]));
  });
  check('log likelihood', worstLp < 1.0,
    `a probe image's log density differs from the generator's by ${worstLp.toFixed(4)} nats over `
    + `${data.meta.dim} dimensions`);
  check('log determinant', worstLd < 1.0,
    `the log determinant differs from the generator's by ${worstLd.toFixed(4)}`);

  /* 2. invertibility, which is the property the widget shows off */
  let trip = 0;
  P.pixels.forEach((px) => {
    const back = glow.inverse(glow.forward(px).z);
    for (let i = 0; i < px.length; i++) trip = Math.max(trip, Math.abs(back[i] - px[i]));
  });
  check('round trip', trip < 1e-4,
    `encoding and inverting a probe image moves a pixel by ${trip.toExponential(2)}, where the `
    + `generator measured ${data.net.round_trip}`);

  /* 3. the arms table has a floor and the page uses it */
  check('the floor', data.arms.floor > 0,
    'the ablation table reports a seed floor of zero, so no verdict on it can be read');

  /* 4. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[glow] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/glow.json').then((r) => r.json());
  const glow = makeGlow(data);

  safely('prose', () => initProse(data));
  safely('what a picture costs', () => initScrollyGlow(data));
  safely('the trap', () => initTrapWidget(data));
  safely('the pieces', () => initArmsWidget(data));
  safely('temperature', () => initTempWidget(data, glow));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, glow));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
