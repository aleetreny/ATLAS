/* Entry point.
 *
 * The probe here covers the two things that can be misread independently: the
 * conditional score network (whose label embedding is a lookup into a table
 * stored the same way every matrix in this module is stored, and reading it the
 * other way round returns another class's vector while still producing
 * plausible digits) and the decoder.
 */
import { initDialWidget } from './dial-widget.js';
import { initTradeWidget } from './trade-widget.js';
import { initHeldWidget } from './held-widget.js';
import { initLiveWidget } from './live-widget.js';
import { initProse } from './prose.js';
import { makeControl } from './controlkit.js';

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
  if (!ok) console.warn(`[control] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, kit) {
  const P = data.net.probe;

  let worstEps = 0;
  P.z.forEach((z, i) => {
    const got = kit.epsOne(Float64Array.from(z), P.t, P.labels[i]);
    const want = P.eps[i];
    const n = Math.hypot(...want);
    if (n > 1e-6) {
      let d = 0;
      for (let k = 0; k < want.length; k++) d += (got[k] - want[k]) ** 2;
      worstEps = Math.max(worstEps, Math.sqrt(d) / n);
    }
  });
  check('the conditional score network', worstEps < 5e-2,
    `its output differs from the generator's by ${worstEps.toExponential(2)} relative, which `
    + 'means the weights, the time embedding or the label table were read wrongly');

  let worstPix = 0;
  P.pixels.forEach((want, i) => {
    const got = kit.decode(kit.unstandardise(Float64Array.from(P.z[i])));
    for (let p = 0; p < want.length; p++) {
      worstPix = Math.max(worstPix, Math.abs(got[p] - want[p]));
    }
  });
  check('the decoder', worstPix < 5e-3,
    `a pixel it produces differs from the generator's by ${worstPix.toExponential(2)}`);

  /* the null token has to be a different answer from every real label, or the
     guidance direction is zero and the dial does nothing while looking fine */
  const z0 = Float64Array.from(P.z[0]);
  const un = kit.epsOne(z0, P.t, kit.NULL);
  let minGap = Infinity;
  for (let lab = 0; lab < 10; lab++) {
    const co = kit.epsOne(z0, P.t, lab);
    let d = 0;
    for (let k = 0; k < co.length; k++) d += (co[k] - un[k]) ** 2;
    minGap = Math.min(minGap, Math.sqrt(d));
  }
  check('the null token', minGap > 1e-3,
    `the unconditioned answer is within ${minGap.toExponential(2)} of some label's answer, so `
    + 'the guidance direction is nearly zero and the dial would do nothing');

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[control] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/control.json').then((r) => r.json());
  const kit = makeControl(data);

  safely('prose', () => initProse(data));
  safely('the dial', () => initDialWidget(data));
  safely('the trade', () => initTradeWidget(data));
  safely('the held out labels', () => initHeldWidget(data));
  safely('sampling here', () => initLiveWidget(data, kit));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, kit));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
