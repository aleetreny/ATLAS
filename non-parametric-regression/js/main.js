/* Entry point: render math, load data, boot every widget. */
import { initScrollyNeighbours } from './scrolly-neighbours.js';
import { initKnnKnob } from './knn-knob.js';
import { initBoundaryWidget } from './boundary-widget.js';
import { initGpWidget } from './gp-widget.js';
import { initCurseWidget } from './curse-widget.js';
import { initSpatialWidget } from './spatial-widget.js';

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

/* Boot each widget in isolation. One throwing used to abort the whole boot and
 * silently take every later widget down with it, which is how a single bad call
 * in the curse chart left the spatial map missing entirely. */
function safely(name, fn) {
  try {
    fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
  }
}

async function boot() {
  renderMath();
  const [mpg, curse, spatial] = await Promise.all([
    fetch('./data/mpg.json').then((r) => r.json()),
    fetch('./data/curse.json').then((r) => r.json()),
    fetch('./data/spatial.json').then((r) => r.json()),
  ]);
  safely('scrolly', () => initScrollyNeighbours(mpg));
  safely('knn', () => initKnnKnob(mpg));
  safely('boundary', () => initBoundaryWidget(mpg));
  safely('gaussian process', () => initGpWidget());
  safely('curse of dimensionality', () => initCurseWidget(curse));
  safely('spatial', () => initSpatialWidget(spatial));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
