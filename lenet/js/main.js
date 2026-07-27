/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyConv } from './scrolly-conv.js';
import { initKernelWidget } from './kernel-widget.js';
import { initFiltersWidget } from './filters-widget.js';
import { initShiftWidget } from './shift-widget.js';
import { initCurvesWidget } from './curves-widget.js';
import { initProse } from './prose.js';

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
  const data = await fetch('./data/conv.json').then((r) => r.json());
  await safely('prose', () => initProse(data));
  await safely('convolution scrolly', () => initScrollyConv(data, IMG_BASE));
  await safely('kernel editor', () => initKernelWidget(data, IMG_BASE));
  await safely('learned filters', () => initFiltersWidget(data, IMG_BASE));
  await safely('translation', () => initShiftWidget(data, IMG_BASE));
  await safely('training curves', () => initCurvesWidget(data));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
