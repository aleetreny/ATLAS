/* Entry point: render math, load data, boot every widget. */
import { initScrollyOLS } from './scrolly-ols.js';
import { initMSEPlayground } from './mse-playground.js';
import { initPolyWidget } from './poly-widget.js';
import { initGLMWidget } from './glm-widget.js';
import { initPLSWidget } from './pls-widget.js';

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

async function boot() {
  renderMath();

  const [mpg, tips, wine] = await Promise.all([
    fetch('./data/mpg.json').then((r) => r.json()),
    fetch('./data/tips.json').then((r) => r.json()),
    fetch('./data/wine.json').then((r) => r.json()),
  ]);

  initScrollyOLS(mpg);
  initMSEPlayground(mpg);
  initPolyWidget(mpg);
  initGLMWidget(tips);
  initPLSWidget(wine);
}

// Module scripts run after deferred classic scripts (d3, katex), but guard anyway.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
