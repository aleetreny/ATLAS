/* Entry point: render math, load data, boot every widget in isolation. */
import { initProse } from './prose.js';
import { initScalingWidget } from './scaling-widget.js';
import { initResolutionWidget } from './resolution-widget.js';
import { initLadderWidget } from './ladder-widget.js';
import { initSeparableWidget } from './separable-widget.js';

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
  const data = await fetch('./data/convnext.json').then((r) => r.json());
  await safely('the numbered sentences', () => initProse(data));
  await safely('compound scaling', () => initScalingWidget(data));
  await safely('resolution', () => initResolutionWidget(data));
  await safely('the ladder', () => initLadderWidget(data));
  await safely('separable arithmetic', () => initSeparableWidget(data));
  /* after prose.js has written the sentences that contain formulas */
  renderMath();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
