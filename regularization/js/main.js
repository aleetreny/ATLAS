/* Entry point: render math, load data, boot every widget. */
import { initScrollyMonster } from './scrolly-monster.js';
import { initLambdaKnob } from './lambda-knob.js';
import { initGeometry } from './geometry.js';
import { initPathWidget } from './path-widget.js';
import { initCollinear } from './collinear.js';

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

  const [sine, dia, col] = await Promise.all([
    fetch('./data/sine.json').then((r) => r.json()),
    fetch('./data/diabetes.json').then((r) => r.json()),
    fetch('./data/collinear.json').then((r) => r.json()),
  ]);

  initScrollyMonster(sine);
  initLambdaKnob(sine);
  initGeometry(dia);
  initPathWidget(dia);
  initCollinear(col);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
