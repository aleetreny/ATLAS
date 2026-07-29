/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyRotation } from './scrolly-rotation.js';
import { initRotationWidget } from './rotation-widget.js';
import { initBenchWidget } from './bench-widget.js';
import { initNamWidget } from './nam-widget.js';
import { initJunkWidget } from './junk-widget.js';
import { initAttentionWidget } from './attention-widget.js';

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
  const data = await fetch('./data/tabular.json').then((r) => r.json());
  safely('rotation scrolly', () => initScrollyRotation());
  safely('rotation on diamonds', () => initRotationWidget(data));
  safely('benchmark', () => initBenchWidget(data));
  safely('additive model', () => initNamWidget(data));
  safely('junk columns', () => initJunkWidget(data));
  safely('sparse attention', () => initAttentionWidget(data));

  /* The same handles every other article exposes, so a sweep can re-run the
     check in any state. This page predates the prose-from-JSON convention
     (its numbers were written into the HTML from the measured file), so what
     it can promise is that no readout carries an undefined or a NaN. */
  window.__atlasCheck = (loud = false) => {
    const problems = [];
    document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td').forEach((el) => {
      const t = el.textContent;
      if (/undefined|\bNaN\b|\$\{/.test(t)) {
        problems.push(`bad text in ${el.id || el.className}: ${t.slice(0, 80)}`);
      }
    });
    if (loud) console.info(`[tables] text check: ${problems.length === 0 ? 'clean' : problems.join(' | ')}`);
    return problems;
  };
  window.__atlasProseIds = [];
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
