/* Entry point: render math, load data, boot every widget in isolation. */
import { initProse } from './prose.js';
import { initMetricWidget } from './metric-widget.js';
import { initUnetWidget } from './unet-widget.js';
import { initInstanceWidget } from './instance-widget.js';
import { initPromptWidget } from './prompt-widget.js';

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
  const [data, net] = await Promise.all([
    fetch('./data/segment.json').then((r) => r.json()),
    fetch('./data/promptnet.json').then((r) => r.json()),
  ]);
  await safely('the numbered sentences', () => initProse(data, net));
  await safely('the metric', () => initMetricWidget(data, './img/'));
  await safely('the U-Net', () => initUnetWidget(data, './img/'));
  await safely('semantic against instance', () => initInstanceWidget(data, './img/'));
  await safely('the promptable network', () => initPromptWidget(data, net, './img/'));
  /* after prose.js has written the sentences that contain formulas */
  renderMath();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
