/* Entry point.
 *
 * Two trained networks run on this page and only one of them is ours, so the
 * guards below check both: this article's encoder against the codes its own
 * generator measured on the same quantised weights, and the autoencoders
 * article's encoder against the codes that article published. The second one is
 * the load bearing check of the whole comparison, because if the previous
 * article's file were being read wrong then the plane on the left would be a
 * picture of a bug rather than of a model.
 */
import { initScrollyVae } from './scrolly-vae.js';
import { initPlaneWidget } from './plane-widget.js';
import { initTrickWidget } from './trick-widget.js';
import { initBetaWidget } from './beta-widget.js';
import { initBoundWidget } from './bound-widget.js';
import { initProse } from './prose.js';
import { makeVae, loadAutoencoder, unpack } from './vaekit.js';
import { mse, lcg } from '../../assets/js/halfweights.js';
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
  if (!ok) console.warn(`[vae] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, vae, ae) {
  const N = data.net;
  const MU = toRows(N.mu);

  /* 1. our own encoder, against the codes the generator wrote for the same
        digits on the same quantised weights */
  let worst = 0;
  N.showcase.forEach((s) => {
    const { mu } = vae.encode(s.pixels);
    for (let j = 0; j < mu.length; j++) {
      worst = Math.max(worst, Math.abs(mu[j] - MU[s.index][j]));
    }
  });
  check('encoder', worst < 5e-3,
    `a showcase digit's code differs from the generator's by ${worst.toExponential(2)}, `
    + `where quantising the weights alone accounts for ${N.quant.code_shift}`);

  /* 2. the decoder round trip, against the held out error the generator measured */
  let sum = 0;
  N.showcase.forEach((s) => {
    sum += mse(vae.decode(vae.encode(s.pixels).mu), s.pixels);
  });
  check('decoder', Math.abs(sum / N.showcase.length - data.compare.vae_test_mse) < 0.02,
    `the round trip on ten digits averages ${(sum / N.showcase.length).toFixed(6)} against a `
    + `held out ${data.compare.vae_test_mse}`);

  /* 3. the other article's network, read from the other article's file */
  if (ae) {
    const AEZ = toRows(N.ae_latent);
    let worstAe = 0;
    N.showcase.forEach((s) => {
      const z = ae.encode(s.pixels);
      for (let j = 0; j < z.length; j++) {
        worstAe = Math.max(worstAe, Math.abs(z[j] - AEZ[s.index][j]));
      }
    });
    check('the previous article', worstAe < 5e-3,
      `the autoencoders article's own encoder, run here, misses its own published codes by `
      + `${worstAe.toExponential(2)}`);
    check('the previous article, size', ae.count === data.meta.reuse.ae_count,
      `that file now holds ${ae.count} weights and the generator measured ${data.meta.reuse.ae_count}`);
  }

  /* 4. the live demo of the two estimators, against the closed form the
        generator exported. This is the one guard on the page that checks a
        calculation the reader watches happen rather than a stored number. */
  const D = data.estimators.demo;
  const rng = lcg(4242);
  let n = 0;
  let s1 = 0;
  let s2 = 0;
  let q1 = 0;
  let q2 = 0;
  for (let i = 0; i < 200000; i++) {
    const z = D.mu + D.sigma * rng.normal();
    const a = -2 * (z - D.c);
    const b = -((z - D.c) ** 2) * (z - D.mu) / (D.sigma * D.sigma);
    n += 1;
    s1 += a; q1 += a * a;
    s2 += b; q2 += b * b;
  }
  const v1 = q1 / n - (s1 / n) ** 2;
  const v2 = q2 / n - (s2 / n) ** 2;
  check('estimator variances', Math.abs(v1 / D.var_pathwise - 1) < 0.05
    && Math.abs(v2 / D.var_score - 1) < 0.10,
  `this page measures ${v1.toFixed(3)} and ${v2.toFixed(1)} where the closed form says `
    + `${D.var_pathwise} and ${D.var_score}`);

  /* 5. the two density pictures, which are float16 payloads and could be
        truncated without anything looking wrong */
  const P = data.toy.picture;
  const truth = unpack(P.truth_b64, P.n * P.n);
  const model = unpack(P.model_b64, P.n * P.n);
  let tSum = 0;
  let mSum = 0;
  for (let i = 0; i < P.n * P.n; i++) { tSum += truth[i]; mSum += model[i]; }
  const cell = ((P.hi - P.lo) / (P.n - 1)) ** 2;
  check('the pictures', Math.abs(tSum * cell - 1) < 0.05 && Math.abs(mSum * cell - 1) < 0.05,
    `the drawn truth integrates to ${(tSum * cell).toFixed(4)} and the drawn fit to `
    + `${(mSum * cell).toFixed(4)} over the box that is drawn`);

  /* 6. nothing on the page may read "undefined" or "NaN" */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[vae] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/vae.json').then((r) => r.json());
  const vae = makeVae(data);
  let ae = null;
  try {
    ae = await loadAutoencoder();
  } catch (err) {
    console.error('could not load the autoencoders article\'s network:', err);
  }

  safely('prose', () => initProse(data));
  safely('the mechanism', () => initScrollyVae(data));
  if (ae) safely('two planes', () => initPlaneWidget(data, vae, ae));
  safely('the two estimators', () => initTrickWidget(data));
  safely('the beta sweep', () => initBetaWidget(data));
  safely('the bound', () => initBoundWidget(data));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, vae, ae));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
