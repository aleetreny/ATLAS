/* Entry point: render math, load data, boot every widget. */
import { initScrollyStars } from './scrolly-stars.js';
import { initLossWidget } from './loss-widget.js';
import { initLeverageLab } from './leverage-lab.js';
import { initTheilSen } from './theilsen-widget.js';
import { initRansac } from './ransac-widget.js';

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
  const stars = await fetch('./data/stars.json').then((r) => r.json());
  initScrollyStars(stars);
  initLossWidget();
  initLeverageLab();
  initTheilSen(stars);
  initRansac(stars);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
