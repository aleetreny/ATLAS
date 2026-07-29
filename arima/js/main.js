/* Entry point.
 *
 * The guards that carry this page are the two recursions. The correlograms and
 * the smoother are recomputed here from the same rounded series the generator
 * measured on, so if this page has read the file wrongly, or if the two
 * definitions have drifted apart, the numbers stop matching and say so. The
 * identity between the smoother and the moving average model is checked at
 * every rate the slider can reach rather than at the one it opens on.
 */
import { initStationarityScrolly } from './scrolly-stationarity.js';
import { initIdentifyWidget } from './identify-widget.js';
import { initIdentityWidget } from './identity-widget.js';
import { initForecastWidget } from './forecast-widget.js';
import { initCoverageWidget } from './coverage-widget.js';
import { initBoardWidget } from './board-widget.js';
import { initProse } from './prose.js';
import { acf, pacf, sesLevels, ma1Forecasts } from '../../assets/js/series.js';

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
  if (!ok) console.warn(`[arima] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  /* 1. the correlograms, against the generator's, on all four differences */
  let worstAcf = 0;
  let worstPacf = 0;
  Object.entries(data.stationarity.curves).forEach(([, cur]) => {
    const a = acf(cur.y, 36);
    const p = pacf(cur.y, 36);
    for (let k = 0; k <= 36; k++) {
      worstAcf = Math.max(worstAcf, Math.abs(a[k] - cur.acf[k]));
      worstPacf = Math.max(worstPacf, Math.abs(p[k] - cur.pacf[k]));
    }
  });
  /* the file carries six decimals, so anything under about 1e-6 is the
     rounding and not the arithmetic */
  check('autocorrelation', worstAcf < 2e-6,
    `this page's correlogram differs from the generator's by ${worstAcf.toExponential(2)}`);
  check('partial autocorrelation', worstPacf < 2e-5,
    `this page's partial correlogram differs from the generator's by `
    + `${worstPacf.toExponential(2)}`);

  /* 2. the identity, at every rate the control can take, not just the one on
        screen: a slider that indexes a table is exactly where a page can look
        fine in both end states and be broken in the middle */
  const y = data.series.co2.y;
  const rates = deep ? d3.range(1, 51).map((i) => i * 0.02) : [0.02, 0.3, 1.0];
  let worstId = 0;
  rates.forEach((a) => {
    const l = sesLevels(y, a);
    const m = ma1Forecasts(y, a - 1);
    for (let i = 0; i < y.length; i++) worstId = Math.max(worstId, Math.abs(l[i] - m[i]));
  });
  check('the identity', worstId < 1e-8,
    `the smoother and the moving average model disagree by ${worstId.toExponential(2)} at some `
    + `rate, where the generator measured ${data.equivalence.identity_worst}`);

  /* 3. the file's own numbers have to be the ones the page believes */
  check('leaderboard', data.rolling.rows.length > 0 && data.rolling.n_origins > 20,
    `the rolling evaluation has ${data.rolling.n_origins} origins`);

  /* 4. nothing may read undefined or NaN, in any state */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        bad.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info(`[arima] load-time checks complete: correlogram to ${worstAcf.toExponential(2)}, `
    + `identity to ${worstId.toExponential(2)}`
    + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs every rate)'}`);
  return { worstAcf, worstPacf, worstId };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/arima.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('stationarity', () => initStationarityScrolly(data));
  safely('identification', () => initIdentifyWidget(data));
  safely('the identity', () => initIdentityWidget(data));
  safely('one origin', () => initForecastWidget(data));
  safely('coverage', () => initCoverageWidget(data));
  safely('leaderboard', () => initBoardWidget(data));

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
