/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyTree } from './scrolly-tree.js';
import { initSplitWidget } from './split-widget.js';
import { initWobbleWidget } from './wobble-widget.js';
import { initVarWidget } from './var-widget.js';
import { initGrowWidget } from './grow-widget.js';
import { initImportanceWidget } from './importance-widget.js';
import { initExtrapWidget } from './extrap-widget.js';

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
  const data = await fetch('./data/trees.json').then((r) => r.json());
  safely('tree scrolly', () => initScrollyTree(data));
  safely('the split', () => initSplitWidget(data));
  safely('the wobble', () => initWobbleWidget(data));
  safely('the variance equation', () => initVarWidget(data));
  safely('growing the forest', () => initGrowWidget(data));
  safely('importance', () => initImportanceWidget(data));
  safely('extrapolation', () => initExtrapWidget(data));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
