/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyTrap } from './scrolly-trap.js';
import { initThresholdWidget } from './threshold-widget.js';
import { initCurvesWidget } from './curves-widget.js';
import { initSmoteWidget } from './smote-widget.js';
import { initBenchWidget } from './bench-widget.js';
import { initRelWidget } from './rel-widget.js';

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
  const data = await fetch('./data/forest.json').then((r) => r.json());
  safely('accuracy trap scrolly', () => initScrollyTrap(data));
  safely('threshold', () => initThresholdWidget(data));
  safely('roc vs pr', () => initCurvesWidget(data));
  safely('smote geometry', () => initSmoteWidget(data));
  safely('benchmark', () => initBenchWidget(data));
  safely('reliability', () => initRelWidget(data));

  /* Runtime consistency checks: the sweep's confusion matrices must agree with
   * the counts the prose quotes, and each matrix must sum to the test set. */
  const t = data.trap;
  for (const s of data.sweep) {
    if (s.tp + s.fp + s.fn + s.tn !== t.n_test) {
      console.warn(`sweep row t=${s.t} does not sum to the test set`);
      break;
    }
  }
  const b = data.baseline;
  if (b.tp + b.fn !== t.n_pos) console.warn('baseline row disagrees with the positive count');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
