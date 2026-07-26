/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyEm } from './scrolly-em.js';
import { initDragWidget } from './drag-widget.js';
import { initTempWidget } from './temp-widget.js';
import { initShapeWidget } from './shape-widget.js';
import { initBicWidget } from './bic-widget.js';
import { initRedeemWidget } from './redeem-widget.js';
import { initWineWidget } from './wine-widget.js';
import { em, initFrom, seedMeans, rng } from './gmmkit.js';

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
  const data = await fetch('./data/mixtures.json').then((r) => r.json());
  safely('em scrolly', () => initScrollyEm(data));
  safely('drag and run', () => initDragWidget(data));
  safely('thermostat', () => initTempWidget(data));
  safely('covariance wardrobe', () => initShapeWidget(data));
  safely('bic', () => initBicWidget(data));
  safely('gallery revisited', () => initRedeemWidget(data));
  safely('wine softly', () => initWineWidget(data));

  /* Runtime check: this page's own EM, restarted a few times, must reach
   * scikit-learn's mean log-likelihood on the blobs. If the two disagree the
   * whole article is decoration. */
  const pts = data.blobs.points;
  const r = rng(4242);
  let best = -Infinity;
  for (let attempt = 0; attempt < 5; attempt++) {
    const path = em(pts, initFrom(pts, seedMeans(pts, 3, r)));
    best = Math.max(best, path[path.length - 1].loglik);
  }
  if (Math.abs(best - data.blobs.ref_loglik) > Math.abs(data.blobs.ref_loglik) * 0.005) {
    console.warn(`browser EM best ${best.toFixed(4)} never matched sklearn's ${data.blobs.ref_loglik}`);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
