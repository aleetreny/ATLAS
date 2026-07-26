/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyScan } from './scrolly-scan.js';
import { initIntegralWidget } from './integral-widget.js';
import { initFeatureWidget } from './feature-widget.js';
import { initCascadeWidget } from './cascade-widget.js';

const IMG_BASE = './img/';

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

async function safely(name, fn) {
  try {
    return await fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

async function boot() {
  renderMath();
  const data = await fetch('./data/cascade.json').then((r) => r.json());
  await safely('scan scrolly', () => initScrollyScan(data, IMG_BASE));
  await safely('integral image', () => initIntegralWidget(data));
  await safely('learned features', () => initFeatureWidget(data, IMG_BASE));
  await safely('the cascade', () => initCascadeWidget(data, IMG_BASE));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
