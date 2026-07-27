/* Entry point: render math, load data, boot every widget in isolation. */
import { initProse } from './prose.js';
import { initGarments } from './garments-widget.js';
import { initScrollyDepth } from './scrolly-depth.js';
import { initFieldWidget } from './field-widget.js';
import { initLadderWidget } from './ladder-widget.js';
import { initGradientWidget } from './gradient-widget.js';
import { initResidualWidget } from './residual-widget.js';

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
  const data = await fetch('./data/depth.json').then((r) => r.json());
  await safely('the numbered sentences', () => initProse(data));
  await safely('the garments', () => initGarments(data, './img/'));
  await safely('depth scrolly', () => initScrollyDepth(data));
  await safely('receptive field', () => initFieldWidget(data));
  await safely('the ladder', () => initLadderWidget(data));
  await safely('gradients', () => initGradientWidget(data));
  await safely('residual sizes', () => initResidualWidget(data));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
