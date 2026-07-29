/* Entry point.
 *
 * Three networks travel in one payload and the page runs all three, so there
 * are three ways to read them wrongly and the probe checks all three: the score
 * network's output at a known noise level, the decoder's pictures from known
 * latents, and the encoder by way of the round trip the ceiling section draws.
 * Any of them can be wrong in a way that still produces a plausible looking
 * digit, which is exactly why none of them is taken on trust.
 */
import { initCeilingWidget } from './ceiling-widget.js';
import { initLadderWidget } from './ladder-widget.js';
import { initTradeWidget } from './trade-widget.js';
import { initLiveWidget } from './live-widget.js';
import { initProse } from './prose.js';
import { makeLatent } from './latentkit.js';

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
  if (!ok) console.warn(`[latent] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, kit) {
  const P = data.net.probe;

  /* 1. the score network, at the noise level the generator measured */
  let worstEps = 0;
  P.z.forEach((z, i) => {
    const got = kit.epsOne(Float64Array.from(z), P.t);
    const want = P.eps[i];
    const n = Math.hypot(...want);
    if (n > 1e-6) {
      let d = 0;
      for (let k = 0; k < want.length; k++) d += (got[k] - want[k]) ** 2;
      worstEps = Math.max(worstEps, Math.sqrt(d) / n);
    }
  });
  check('the score network', worstEps < 5e-2,
    `its output here differs from the generator's by ${worstEps.toExponential(2)} relative, `
    + 'which means the weights or the time embedding were read wrongly');

  /* 2. the decoder, on the same latents */
  let worstPix = 0;
  P.pixels.forEach((want, i) => {
    const got = kit.decode(kit.unstandardise(Float64Array.from(P.z[i])));
    for (let p = 0; p < want.length; p++) {
      worstPix = Math.max(worstPix, Math.abs(got[p] - want[p]));
    }
  });
  check('the decoder', worstPix < 5e-3,
    `a pixel it produces differs from the generator's by ${worstPix.toExponential(2)}`);

  /* 3. the encoder, by the property the ceiling section is about: a round trip
        of a real digit has to land near that digit, and the article's own
        measured round trip error says how near */
  const dim = data.meta.pixels;
  const bin = atob(data.strip.real_b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const u16 = new Uint16Array(buf);
  const half = (hh) => {
    const s = (hh & 0x8000) ? -1 : 1;
    const e = (hh & 0x7c00) >> 10;
    const f = hh & 0x03ff;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 0x1f) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  };
  const measured = data.ceiling.find((r) => r.k === data.meta.k_show).mse;
  let se = 0;
  const nCheck = 8;
  for (let i = 0; i < nCheck; i++) {
    const x = new Float64Array(dim);
    for (let p = 0; p < dim; p++) x[p] = half(u16[i * dim + p]);
    const back = kit.roundTrip(x);
    for (let p = 0; p < dim; p++) se += (back[p] - x[p]) ** 2;
  }
  const mse = se / (nCheck * dim);
  check('the round trip', mse < 4 * measured,
    `the page's own round trip error is ${mse.toFixed(5)} against the ${measured} this article `
    + 'measured, so the encoder and the decoder are not the pair that was measured');

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[latent] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/latent.json').then((r) => r.json());
  const kit = makeLatent(data);

  safely('prose', () => initProse(data));
  safely('the ceiling', () => initCeilingWidget(data, kit));
  safely('the ladder', () => initLadderWidget(data));
  safely('the trade', () => initTradeWidget(data));
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
