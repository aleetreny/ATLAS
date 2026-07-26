/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyAngle } from './scrolly-angle.js';
import { initDialWidget } from './dial-widget.js';
import { initGammaWidget } from './gamma-widget.js';
import { kdaScores, accuracy, project, bestCut } from './kdakit.js';

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

/* One widget throwing must not take the rest of the page down with it. */
function safely(name, fn) {
  try {
    fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
  }
}

async function boot() {
  renderMath();
  const data = await fetch('./data/discriminants.json').then((r) => r.json());
  safely('angle scrolly', () => initScrollyAngle(data));
  safely('the dial', () => initDialWidget(data));
  safely('the gamma dial', () => initGammaWidget(data));

  /* Runtime checks, deferred: recompute every KDA test accuracy from the
   * shipped dual coefficients, and confirm the linear ceiling from the dial's
   * own machinery. */
  const check = () => {
    for (const row of data.kda) {
      const z = kdaScores(data.x_test, data.x_train, row.alpha, row.gamma);
      const pred = Array.from(z, (v) => (v > row.b ? 1 : 0));
      const acc = accuracy(pred, data.y_test);
      if (Math.abs(acc - row.acc_test) > 0.012) {
        console.warn(`KDA gamma ${row.gamma}: browser accuracy ${acc.toFixed(4)} vs generator ${row.acc_test}`);
      }
    }
    const zc = project(data.x_train, data.best_line.theta);
    const cut = bestCut(zc, data.y_train);
    const zt = project(data.x_test, data.best_line.theta);
    const pred = zt.map((v) => {
      const side = v > cut.cut ? 1 : 0;
      return cut.flip ? 1 - side : side;
    });
    const acc = accuracy(pred, data.y_test);
    if (Math.abs(acc - data.best_line.acc_test) > 0.012) {
      console.warn(`linear ceiling: browser ${acc.toFixed(4)} vs generator ${data.best_line.acc_test}`);
    }
  };
  if ('requestIdleCallback' in window) requestIdleCallback(check, { timeout: 4000 });
  else setTimeout(check, 800);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
