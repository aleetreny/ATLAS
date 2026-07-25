/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyMargin } from './scrolly-margin.js';
import { initCWidget } from './c-widget.js';
import { initLiftWidget } from './lift-widget.js';
import { initGammaWidget } from './gamma-widget.js';
import { initCurveWidget } from './curve-widget.js';
import { initAxesWidget } from './axes-widget.js';
import { initBenchWidget } from './bench-widget.js';

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
  const data = await fetch('./data/geometry.json').then((r) => r.json());
  safely('margin scrolly', () => initScrollyMargin(data));
  safely('the C knob', () => initCWidget(data));
  safely('the lift', () => initLiftWidget(data));
  safely('gamma', () => initGammaWidget(data));
  safely('discriminant curve', () => initCurveWidget(data));
  safely('three axes', () => initAxesWidget(data));
  safely('benchmark', () => initBenchWidget(data));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
