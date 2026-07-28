/* Entry point.
 *
 * The guard here is the stack's own arithmetic: the trend blocks and the
 * seasonal blocks are exported separately, and the forecast is exported too,
 * so the page can check that the parts it draws are the parts the forecast is
 * made of. If they were not, the scrolly would be a decorative diagram of a
 * network that does something else, which is exactly the failure this
 * repository has paid for twice in other articles.
 */
import { initStackScrolly } from './scrolly-stack.js';
import { initScalingWidget } from './scaling-widget.js';
import { initAttentionWidget } from './attention-widget.js';
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
  if (!ok) console.warn(`[deep-forecast] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  /* 1. the parts add up to the forecast, on every exported example */
  let worst = 0;
  data.bases.examples.forEach((ex) => {
    for (let i = 0; i < ex.forecast.length; i++) {
      worst = Math.max(worst, Math.abs((ex.trend_hat[i] + ex.season_hat[i]) - ex.forecast[i]));
    }
  });
  /* four decimals in the file, and two of them are being added */
  check('the stack adds up', worst < 5e-4,
    `the trend and seasonal parts differ from the forecast by ${worst.toExponential(2)}`);

  /* 2. the two rankings the attention widget draws must come from the same
        model: the base loss has to sit below every permuted loss but one */
  const A = data.attention;
  const notWorse = A.rows.filter((row) => row.loss_when_permuted < A.base_loss - 1e-4);
  check('permutation', notWorse.length <= 1,
    `${notWorse.length} inputs make the model better when they are destroyed, which means the `
    + `base loss and the permuted losses are not from the same model`);

  /* 3. the seed spreads have to exist, or no verdict on the two bases is
        readable */
  check('a floor under the bases', data.bases.generic.spread > 0
    && data.bases.interpretable.spread > 0,
  'the two bases were trained once each, so their difference has nothing to be read against');

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

  console.info(`[deep-forecast] load-time checks complete: the stack adds up to `
    + `${worst.toExponential(2)}`);
  return { worst };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/deepforecast.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the stack', () => initStackScrolly(data));
  safely('the crossover', () => initScalingWidget(data));
  safely('attention', () => initAttentionWidget(data));

  renderMath();

  window.__atlasCheck = () => safely('guards', () => runGuards(data));
  const guards = () => window.__atlasCheck();
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
