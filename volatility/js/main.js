/* Entry point.
 *
 * The guard here is the variance path. The page runs its own filter over the
 * real returns and compares it with the one `arch` produced from the same three
 * numbers, which catches the whole family of errors that make a volatility
 * widget look plausible and be wrong: a recursion started in the wrong place, a
 * mean that was not subtracted, a variance confused with a volatility.
 */
import { initReturnsScrolly } from './scrolly-returns.js';
import { initGarchWidget } from './garch-widget.js';
import { initVixWidget } from './vix-widget.js';
import { initRiskWidget } from './risk-widget.js';
import { initIrfWidget } from './irf-widget.js';
import { initProse } from './prose.js';
import { garchFilter, garchNLL, acf } from '../../assets/js/series.js';

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
  if (!ok) console.warn(`[volatility] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const G = data.garch;
  const y = data.returns.y.map((v) => v - G.mean);

  /* 1. the variance path, against the library's, from the same parameters */
  const start = G.omega + (G.alpha + G.beta) * G.backcast;
  const v = garchFilter(y, G.omega, G.alpha, G.beta, start);
  let worst = 0;
  for (let i = 0; i < y.length; i++) {
    worst = Math.max(worst, Math.abs(Math.sqrt(v[i]) - G.vol[i]));
  }
  /* the file carries five decimals of volatility, so 1e-4 is the rounding */
  check('variance path', worst < 1e-4,
    `this page's filter differs from the fitted one by ${worst.toExponential(2)} in volatility`);

  /* 2. the likelihood, which is the only thing that scores a path nobody sees */
  const nll = garchNLL(y, G.omega, G.alpha, G.beta, start);
  check('likelihood', Math.abs(nll - G.nll) < 0.05,
    `this page computes ${nll.toFixed(3)} where the generator wrote ${G.nll}`);

  /* 3. the starting point matters and the page uses the library's, not the
        obvious one: this checks the difference is still what was measured */
  if (deep) {
    const naive = garchFilter(y, G.omega, G.alpha, G.beta, data.garch.sample_var);
    let gap = 0;
    for (let i = 0; i < y.length; i++) gap = Math.max(gap, Math.abs(naive[i] - v[i]));
    check('the starting point', Math.abs(gap - G.start_gap) < 0.02,
      `starting at the sample variance moves the path by ${gap.toFixed(4)} where the generator `
      + `measured ${G.start_gap}`);
  }

  /* 4. the two correlograms the scrolly draws */
  const a = acf(data.returns.y, 40);
  let worstAcf = 0;
  for (let k = 0; k <= 40; k++) worstAcf = Math.max(worstAcf, Math.abs(a[k] - data.two_r2.acf_return[k]));
  check('correlogram', worstAcf < 2e-5,
    `the return correlogram differs from the generator's by ${worstAcf.toExponential(2)}`);

  /* 5. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        bad.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info(`[volatility] load-time checks complete: variance path to ${worst.toExponential(2)}, `
    + `likelihood ${nll.toFixed(2)} against ${G.nll}`
    + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  return { worst, nll, worstAcf };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/volatility.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('returns', () => initReturnsScrolly(data));
  safely('garch', () => initGarchWidget(data));
  safely('vix', () => initVixWidget(data));
  safely('value at risk', () => initRiskWidget(data));
  safely('impulse response', () => initIrfWidget(data));

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
