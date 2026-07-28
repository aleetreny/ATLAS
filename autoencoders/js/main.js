/* Entry point.
 *
 * The guards compare this page's forward pass, on the quantised weights it
 * downloaded, against the numbers the generator measured on the same quantised
 * weights. That is the only comparison worth making: checking against the full
 * precision network would be checking the quantisation rather than the code.
 */
import { initScrollyAe } from './scrolly-ae.js';
import { initLatentWidget } from './latent-widget.js';
import { initCurveWidget } from './curve-widget.js';
import { initRollWidget } from './roll-widget.js';
import { initProse } from './prose.js';
import { makeNet, mse } from './netkit.js';
import { toRows } from '../../assets/js/embed.js';

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
  if (!ok) console.warn(`[autoencoders] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, net) {
  const N = data.net;
  const Z = toRows(N.latent);

  /* 1. the encoder, against the codes the generator wrote for the same digits */
  let worst = 0;
  N.showcase.forEach((s) => {
    const z = net.encode(s.pixels);
    worst = Math.max(worst, Math.abs(z[0] - Z[s.index][0]), Math.abs(z[1] - Z[s.index][1]));
  });
  check('encoder', worst < 5e-3,
    `a showcase digit's code differs from the generator's by ${worst.toExponential(2)}, `
    + `where quantising the weights alone accounts for ${N.quant.code_shift}`);

  /* 2. the decoder: the round trip error on the showcase digits should sit in
        the neighbourhood of the held out error the generator measured */
  let sum = 0;
  N.showcase.forEach((s) => {
    sum += mse(net.decode(net.encode(s.pixels)), s.pixels);
  });
  const avg = sum / N.showcase.length;
  check('decoder', Math.abs(avg - data.generalisation.test_mse) < 0.02,
    `the round trip on ten digits averages ${avg.toFixed(6)} against a held out `
    + `${data.generalisation.test_mse}`);

  /* 3. the decoder must stay inside the range the data lives in */
  let lo = Infinity;
  let hi = -Infinity;
  for (let a = -3; a <= 3; a += 1) {
    for (let b = -3; b <= 3; b += 1) {
      const px = net.decode([a, b]);
      for (let i = 0; i < px.length; i++) {
        lo = Math.min(lo, px[i]);
        hi = Math.max(hi, px[i]);
      }
    }
  }
  check('decoder range', lo >= -1e-9 && hi <= 1 + 1e-9,
    `the decoder produced values from ${lo.toExponential(2)} to ${hi.toFixed(6)}, outside [0, 1]`);

  /* 4. the theorem the article opens with, restated from the file rather than
        recomputed: it is a claim about a training run, not about this page */
  check('linear subspace', data.linear.subspace_gap < 1e-3,
    `the file reports the linear network's plane ${data.linear.subspace_gap} from PCA's`);
  check('linear error', Math.abs(data.linear.mse_ratio - 1) < 0.01,
    `the file reports the linear network at ${data.linear.mse_ratio} times PCA's error`);

  /* 5. nothing on the page may read "undefined" or "NaN" */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .ae-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[autoencoders] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/autoencoders.json').then((r) => r.json());
  const net = makeNet(data);

  safely('prose', () => initProse(data));
  safely('the mechanism', () => initScrollyAe(data, net));
  safely('two numbers and a digit', () => initLatentWidget(data, net));
  safely('what the bending buys', () => initCurveWidget(data));
  safely('the roll', () => initRollWidget(data));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, net));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
