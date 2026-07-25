/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyOdds } from './scrolly-odds.js';
import { initLogitWidget } from './logit-widget.js';
import { initLossWidget } from './loss-widget.js';
import { initNbWidget } from './nb-widget.js';
import { initBoundaryWidget } from './boundary-widget.js';
import { initBenchWidget } from './bench-widget.js';
import { initCalibWidget } from './calib-widget.js';
import { initThreshWidget } from './thresh-widget.js';

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
  const data = await fetch('./data/cancer.json').then((r) => r.json());
  safely('one-dimensional scrolly', () => initScrollyOdds(data));
  safely('log odds', () => initLogitWidget(data));
  safely('loss surface', () => initLossWidget(data));
  safely('naive bayes shapes', () => initNbWidget(data));
  safely('decision boundaries', () => initBoundaryWidget(data));
  safely('benchmark', () => initBenchWidget(data));
  safely('calibration', () => initCalibWidget(data));
  safely('threshold', () => initThreshWidget(data));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
