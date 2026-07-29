/* Entry point: render math, load data, boot every widget in isolation. */
import { initProse, PROSE_IDS } from './prose.js';
import { initIouWidget } from './iou-widget.js';
import { initWindowsWidget } from './windows-widget.js';
import { initArmsWidget } from './arms-widget.js';
import { initDetectorWidget } from './detector-widget.js';
import { initNmsWidget } from './nms-widget.js';

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
  const [data, det] = await Promise.all([
    fetch('./data/detect.json').then((r) => r.json()),
    fetch('./data/detector.json').then((r) => r.json()),
  ]);
  await safely('the numbered sentences', () => initProse(data, det));
  await safely('intersection over union', () => initIouWidget(data));
  await safely('the window bill', () => initWindowsWidget(data));
  await safely('the anchor factorial', () => initArmsWidget(data));
  await safely('the live detector', () => initDetectorWidget(det, './img/'));
  await safely('suppression', () => initNmsWidget(data));
  /* after prose.js has written the sentences that contain formulas */
  renderMath();

  /* The same handles every other article exposes, so a sweep can re-run the
     check in any state. This page predates the numeric guards, so what it can
     promise is that no composed sentence carries an undefined or a NaN. */
  window.__atlasCheck = (loud = false) => {
    const problems = [];
    document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td').forEach((el) => {
      const t = el.textContent;
      if (/undefined|\bNaN\b|\$\{/.test(t)) {
        problems.push(`bad text in ${el.id || el.className}: ${t.slice(0, 80)}`);
      }
    });
    if (loud) console.info(`[detection] text check: ${problems.length === 0 ? 'clean' : problems.join(' | ')}`);
    return problems;
  };
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
