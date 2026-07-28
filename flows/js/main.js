/* Entry point.
 *
 * The guards here check the identity the whole article rests on, in the page
 * rather than in the file: the log determinant this page computes against the
 * one the generator measured on the same quantised weights, the round trip
 * through the inverse, and the density's own integral over the drawn box.
 */
import { initScrollyFlow } from './scrolly-flow.js';
import { initDepthWidget } from './depth-widget.js';
import { initHoleWidget } from './hole-widget.js';
import { initProse } from './prose.js';
import { makeFlow, unpack } from './flowkit.js';

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
  if (!ok) console.warn(`[flows] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, flow) {
  const P = data.net.probe;

  /* 1. the forward map and its log determinant, against the generator's */
  let worstZ = 0;
  let worstLd = 0;
  P.x.forEach((p, i) => {
    const { z, logdet } = flow.forwardTo(p, flow.depth);
    worstZ = Math.max(worstZ, Math.abs(z[0] - P.z[i][0]), Math.abs(z[1] - P.z[i][1]));
    worstLd = Math.max(worstLd, Math.abs(logdet - P.logdet[i]));
  });
  check('forward map', worstZ < 5e-3,
    `a probe point lands ${worstZ.toExponential(2)} from where the generator put it, where `
    + `quantising the weights moves a log density by ${data.net.quant.logp_shift}`);
  check('log determinant', worstLd < 5e-3,
    `the log determinant differs from the generator's by ${worstLd.toExponential(2)}`);

  /* 2. the identity itself, in this page: analytic against central differences */
  let worstJac = 0;
  P.x.slice(0, 8).forEach((p) => {
    const d = flow.detAt(p);
    worstJac = Math.max(worstJac, Math.abs(d.analytic - d.numeric) / Math.max(d.analytic, 1e-12));
  });
  check('the change of variables', worstJac < 2e-3,
    `the analytic stretching factor and a central difference of the same map differ by `
    + `${(worstJac * 100).toFixed(3)}% at worst, where the generator measured `
    + `${data.guards.jacobian_gap} in double precision`);

  /* 3. invertibility, which is the property the topology section is about */
  let worstTrip = 0;
  P.x.forEach((p) => {
    const back = flow.inverse(flow.forward(p).z);
    worstTrip = Math.max(worstTrip, Math.abs(back[0] - p[0]), Math.abs(back[1] - p[1]));
  });
  check('invertibility', worstTrip < 1e-4,
    `pushing a probe point forward and back moves it by ${worstTrip.toExponential(2)}`);

  /* 4. the drawn density integrates to about one over the drawn box */
  const pic = data.picture;
  const model = unpack(pic.model_b64, pic.n * pic.n);
  let s = 0;
  for (let i = 0; i < model.length; i++) s += model[i];
  const cell = ((pic.hi - pic.lo) / (pic.n - 1)) ** 2;
  check('the picture', Math.abs(s * cell - 1) < 0.08,
    `the drawn density integrates to ${(s * cell).toFixed(4)} over the box that is drawn, where `
    + `the generator measured ${data.guards.mass} over the whole quadrature square`);

  /* 5. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[flows] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/flows.json').then((r) => r.json());
  const flow = makeFlow(data);

  safely('prose', () => initProse(data));
  safely('the warping', () => initScrollyFlow(data, flow));
  safely('depth', () => initDepthWidget(data));
  safely('the hole', () => initHoleWidget(data, flow));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, flow));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
