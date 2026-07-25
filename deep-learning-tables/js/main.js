/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyRotation } from './scrolly-rotation.js';
import { initRotationWidget } from './rotation-widget.js';
import { initBenchWidget } from './bench-widget.js';
import { initNamWidget } from './nam-widget.js';
import { initJunkWidget } from './junk-widget.js';
import { initAttentionWidget } from './attention-widget.js';

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
  const data = await fetch('./data/tabular.json').then((r) => r.json());
  safely('rotation scrolly', () => initScrollyRotation());
  safely('rotation on diamonds', () => initRotationWidget(data));
  safely('benchmark', () => initBenchWidget(data));
  safely('additive model', () => initNamWidget(data));
  safely('junk columns', () => initJunkWidget(data));
  safely('sparse attention', () => initAttentionWidget(data));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
