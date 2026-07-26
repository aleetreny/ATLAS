/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyTube } from './scrolly-tube.js';
import { initTubeWidget } from './tube-widget.js';
import { initKrrWidget } from './krr-widget.js';
import { initTwinWidget } from './twin-widget.js';
import { krrFit, rmse } from './svrkit.js';

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
  const data = await fetch('./data/machines.json').then((r) => r.json());
  safely('tube scrolly', () => initScrollyTube(data));
  safely('two dials', () => initTubeWidget(data));
  safely('live kernel ridge', () => initKrrWidget(data));
  safely('the twins', () => initTwinWidget(data));

  /* Runtime check, deferred: the browser's kernel ridge must reproduce every
   * scikit-learn reference fit, curve and test error alike. */
  const check = () => {
    let worstCurve = 0;
    let worstErr = 0;
    for (const ref of data.kr_refs) {
      const predict = krrFit(data.x_train, data.y_train, ref.gamma, ref.lambda);
      const curve = predict(data.zgrid);
      for (let i = 0; i < curve.length; i++) {
        worstCurve = Math.max(worstCurve, Math.abs(curve[i] - ref.curve[i]));
      }
      worstErr = Math.max(worstErr, Math.abs(rmse(data.y_test, predict(data.x_test)) - ref.rmse_test));
    }
    if (worstCurve > 0.01 || worstErr > 0.005) {
      console.warn(`browser kernel ridge diverges from sklearn: curve ${worstCurve.toFixed(4)} mpg, rmse ${worstErr.toFixed(4)}`);
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
