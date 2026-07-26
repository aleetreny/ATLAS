/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyLloyd } from './scrolly-lloyd.js';
import { initStepWidget } from './step-widget.js';
import { initInitWidget } from './init-widget.js';
import { initKWidget } from './k-widget.js';
import { initFailWidget } from './fail-widget.js';
import { initWineWidget } from './wine-widget.js';
import { assign, lloyd, initPP, rng } from './kmeanskit.js';

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
  const data = await fetch('./data/clusters.json').then((r) => r.json());
  safely('lloyd scrolly', () => initScrollyLloyd(data));
  safely('step by step', () => initStepWidget(data));
  safely('initialisation', () => initInitWidget(data));
  safely('choosing k', () => initKWidget(data));
  safely('failure gallery', () => initFailWidget(data));
  safely('wine', () => initWineWidget(data));

  /* Runtime check: the browser's own k-means, started with k-means++ and run
   * to convergence, must land at scikit-learn's reference inertia. If the two
   * implementations disagree the whole page is decoration. */
  const pts = data.blobs.points;
  let ok = false;
  const r = rng(4242);
  for (let attempt = 0; attempt < 5 && !ok; attempt++) {
    const path = lloyd(pts, initPP(pts, 3, r));
    const J = path[path.length - 1].inertia;
    if (Math.abs(J - data.blobs.ref_inertia) / data.blobs.ref_inertia < 0.01) ok = true;
  }
  if (!ok) console.warn(`browser k-means never matched sklearn's inertia ${data.blobs.ref_inertia}`);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
