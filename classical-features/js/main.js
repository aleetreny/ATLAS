/* Entry point: render math, load data, boot every widget in isolation.
 *
 * The vision articles are async all the way down: a canvas cannot compute
 * anything until its PNG has decoded, so each init returns a promise and the
 * page tolerates one of them failing without losing the rest.
 */
import { initScrollyHog } from './scrolly-hog.js';
import { initHogWidget } from './hog-widget.js';
import { initMatchWidget } from './match-widget.js';

const IMG_BASE = './img/';

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
  renderMath();
  const data = await fetch('./data/features.json').then((r) => r.json());
  await safely('hog scrolly', () => initScrollyHog(data, IMG_BASE));
  const worst = await safely('hog widget', () => initHogWidget(data, IMG_BASE));
  await safely('matching', () => initMatchWidget(data, IMG_BASE));

  /* The load-time check lives in the widget (it has the pixels); surface it
   * here so a mismatch is one console line rather than a silent drift. */
  if (worst !== null && worst > 1e-4) {
    console.warn(`HOG reference check failed by ${worst.toExponential(2)}`);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
