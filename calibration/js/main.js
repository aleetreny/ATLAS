/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyPromise } from './scrolly-promise.js';
import { initModelsWidget } from './models-widget.js';
import { initPavWidget } from './pav-widget.js';
import { initRepairWidget } from './repair-widget.js';
import { initSizeWidget } from './size-widget.js';
import { initDecompWidget } from './decomp-widget.js';

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
  const data = await fetch('./data/repair.json').then((r) => r.json());
  safely('promise scrolly', () => initScrollyPromise(data));
  safely('four models', () => initModelsWidget(data));
  safely('pav', () => initPavWidget(data));
  safely('repair', () => initRepairWidget(data));
  safely('size curve', () => initSizeWidget(data));
  safely('decomposition', () => initDecompWidget(data));

  /* Runtime check: the Brier decomposition must actually reassemble the
   * Brier score for every model and state (binned decomposition is only
   * approximate, so the tolerance is loose but bounded). */
  for (const [name, m] of Object.entries(data.models)) {
    for (const st of ['raw', 'platt', 'isotonic', 'logit_shift']) {
      if (!m[st]) continue;
      const s = m[st];
      if (Math.abs(s.rel - s.res + s.unc - s.brier) > 0.002) {
        console.warn(`decomposition of ${name}/${st} does not reassemble its Brier score`);
      }
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
