/* Entry point.
 *
 * The guard that carries this page is the probe. The page recomputes the exact
 * forward density and the exact score itself, in closed form, and separately
 * runs the trained network out of float16 weights. Either of those can be
 * wrong in a way that still produces a plausible picture: a misread weight
 * layout, a time embedding written with the wrong frequencies, a radial
 * profile indexed off by one. So the generator ships forty points at each noise
 * level with the answers it measured, and the page checks all three quantities
 * against them before it claims anything.
 */
import { initScrollyDDPM } from './scrolly-ddpm.js';
import { initScoreWidget } from './score-widget.js';
import { initReverseWidget } from './reverse-widget.js';
import { initPriorWidget } from './prior-widget.js';
import { initPixelWidget } from './pixel-widget.js';
import { initProse, PROSE_IDS } from './prose.js';
import { makeDDPM } from './ddpmkit.js';

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

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(label, ok, detail) {
  if (!ok) console.warn(`[ddpm] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, kit) {
  const P = data.net.probe;
  let worstQ = 0;
  let worstScore = 0;
  let worstEps = 0;

  P.t.forEach((t, li) => {
    const v = Math.max(1 - data.net.abar[li], 1e-12);
    /* the points are drawn from q(x_t), so there is one set per level */
    P.points[li].forEach((pt, i) => {
      const T = kit.truthAt(li, pt[0], pt[1]);
      const q = P.q[li][i];
      if (q > 1e-6) worstQ = Math.max(worstQ, Math.abs(T.q - q) / q);
      const s = P.score[li][i];
      const ns = Math.hypot(s[0], s[1]);
      if (ns > 1e-6) {
        worstScore = Math.max(worstScore, Math.hypot(T.sx - s[0], T.sy - s[1]) / ns);
      }
      /* the network's own output, which is what the sampler actually runs on */
      const L = kit.learnedAt(li, pt[0], pt[1]);
      const e = P.eps[li][i];
      const ne = Math.hypot(e[0], e[1]);
      if (ne > 1e-6) {
        worstEps = Math.max(worstEps,
          Math.hypot(-L.sx * Math.sqrt(v) - e[0], -L.sy * Math.sqrt(v) - e[1]) / ne);
      }
    });
  });

  /* The tolerances come from the rounding in the file, not from taste: the
     density travels in scientific notation with six figures, the vectors to
     four decimals, and the ring's radial profile travels as float16, which
     carries about three. The probe points are drawn from q(x_t) itself, so
     none of them sits in a tail where float16 has nothing left to carry. */
  check('the exact density', worstQ < 5e-3,
    `the page's own q(x_t) differs from the generator's by ${worstQ.toExponential(2)} relative`);
  check('the exact score', worstScore < 5e-2,
    `the page's own score differs from the generator's by ${worstScore.toExponential(2)} relative`);
  check('the network', worstEps < 5e-2,
    `the network's output here differs from the generator's by ${worstEps.toExponential(2)} `
    + 'relative, which means the weights or the time embedding were read wrongly');

  /* the reverse chain has to end somewhere sensible: with the full budget its
     draws should sit inside the square the truth lives in */
  const { frames } = kit.reverse(200, 50, 12345, 'deterministic');
  const last = frames[frames.length - 1];
  let far = 0;
  for (let i = 0; i < last.length / 2; i++) {
    far = Math.max(far, Math.hypot(last[i * 2], last[i * 2 + 1]));
  }
  check('the reverse chain', far < 8,
    `a draw landed ${far.toFixed(2)} from the origin, well outside anything the truth reaches`);

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[ddpm] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/ddpm.json').then((r) => r.json());
  const kit = makeDDPM(data);

  safely('prose', () => initProse(data));
  safely('the data disappearing', () => initScrollyDDPM(data, kit));
  safely('the score field', () => initScoreWidget(data, kit));
  safely('the reverse chain', () => initReverseWidget(data, kit));
  safely('the prior term', () => initPriorWidget(data));
  safely('the pixels', () => initPixelWidget(data));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, kit));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);

  /* The same handles every other article exposes, so a sweep can re-run the
     guards in any state instead of only reading what they logged on load. */
  window.__atlasCheck = () => { runGuards(data, kit); return []; };
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
