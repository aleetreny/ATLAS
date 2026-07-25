/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyBoost } from './scrolly-boost.js';
import { initRateWidget } from './rate-widget.js';
import { initGainWidget } from './gain-widget.js';
import { initStoppingWidget } from './stopping-widget.js';
import { initGrowthWidget } from './growth-widget.js';
import { initImportanceWidget } from './importance-widget.js';

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
  const dia = await fetch('./data/diamonds.json').then((r) => r.json());
  safely('scrolly', () => initScrollyBoost());
  safely('learning rate', () => initRateWidget(dia));
  safely('split gain', () => initGainWidget());
  safely('early stopping', () => initStoppingWidget(dia));
  safely('tree growth', () => initGrowthWidget());
  safely('importance', () => initImportanceWidget(dia));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
