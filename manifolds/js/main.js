/* Entry point: render maths, load the data, boot every widget in isolation,
 * then check this page's arithmetic against the generator's.
 *
 * The guards are written so they CAN fire. Everything expensive is computed
 * once here and shared, because the graph at k = 5 is wanted by the scrolly,
 * the k dial, the comparison and two of the guards, and it costs about a third
 * of a second to build.
 */
import { initScrollyRoll } from './scrolly-roll.js';
import { initKWidget } from './k-widget.js';
import { initCompareWidget, computeMethods } from './compare-widget.js';
import { initHoleWidget, holeState } from './hole-widget.js';
import { initProse } from './prose.js';
import { buildState, isomapAt, pca2 } from './roll.js';
import { procrustes, toRows } from '../../assets/js/embed.js';

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
  if (!ok) console.warn(`[manifolds] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, state, computed, caches, methods, holeCache) {
  const G = data.graph;

  /* 1. the graph itself, at every position the slider can take */
  G.rows.forEach((row) => {
    if (!caches.k[row.k]) caches.k[row.k] = isomapAt(state, row.k);
    const R = caches.k[row.k];
    check(`edges k=${row.k}`, R.edges.length === row.edges,
      `${R.edges.length} edges here, ${row.edges} in the file`);
    check(`components k=${row.k}`, R.components === row.components,
      `${R.components} components here, ${row.components} in the file`);
    check(`short circuits k=${row.k}`, R.shorts.length === row.short_circuits,
      `${R.shorts.length} short circuits here, ${row.short_circuits} in the file`);
    if (row.disparity === undefined) return;
    check(`geodesic correlation k=${row.k}`, Math.abs(R.geoCorr - row.geodesic_corr) < 5e-5,
      `${R.geoCorr.toFixed(6)} against ${row.geodesic_corr}`);
    check(`disparity k=${row.k}`, Math.abs(R.disparity - row.disparity) < 5e-4,
      `${R.disparity.toFixed(6)} against ${row.disparity}`);
  });

  /* 2. the four methods, each against its own reference */
  ['pca', 'isomap', 'lle'].forEach((key) => {
    const ref = data.methods[key];
    const got = methods[key];
    check(`${key} disparity`, Math.abs(got.disparity - ref.disparity) < 5e-4,
      `${got.disparity.toFixed(6)} against ${ref.disparity}`);
    check(`${key} trustworthiness`, Math.abs(got.trust - ref.trustworthiness) < 5e-4,
      `${got.trust.toFixed(6)} against ${ref.trustworthiness}`);
    /* and the configuration itself, not just its score */
    const d = procrustes(toRows(ref.coords), got.coords).disparity;
    check(`${key} configuration`, d < 1e-4,
      `the browser's map is ${d.toExponential(2)} from the generator's`);
  });

  /* 3. the self-organising map, which the browser trained from the same
        prototypes and the same sample order rather than downloading */
  const S = data.som;
  const sc = methods.som.scores;
  check('som quantisation', Math.abs(sc.quantisation - S.quantisation) < 5e-4,
    `${sc.quantisation.toFixed(6)} against ${S.quantisation}`);
  check('som topographic', Math.abs(sc.topographic - S.topographic) < 5e-4,
    `${sc.topographic.toFixed(6)} against ${S.topographic}`);
  check('som disparity', Math.abs(methods.som.disparity - S.disparity) < 5e-4,
    `${methods.som.disparity.toFixed(6)} against ${S.disparity}`);
  let wp = 0;
  const refP = toRows(S.prototypes);
  for (let i = 0; i < refP.length; i++) {
    for (let j = 0; j < refP[0].length; j++) {
      wp = Math.max(wp, Math.abs(methods.som.som.P[i][j] - refP[i][j]));
    }
  }
  check('som prototypes', wp < 5e-4,
    `a prototype differs from the generator's by ${wp.toExponential(2)}`);

  /* 4. the holed roll */
  const H = data.hole;
  check('hole disparity', Math.abs(holeCache.holed.disparity - H.disparity) < 5e-4,
    `${holeCache.holed.disparity.toFixed(6)} against ${H.disparity}`);
  check('hole connected', holeCache.holed.components === H.components,
    `${holeCache.holed.components} components against ${H.components}`);

  /* 5. nothing on the page may read "undefined" or "NaN" */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .man-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[manifolds] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/manifolds.json').then((r) => r.json());

  const state = buildState(data);
  const caches = { k: {} };
  const kShow = data.meta.k_show;
  caches.k[kShow] = isomapAt(state, kShow);
  const computed = { iso: caches.k[kShow], pca: pca2(state.points) };

  const methods = computeMethods(data, state, caches.k);

  const holed = holeState(data);
  const holeCache = {
    intact: caches.k[kShow],
    holed: isomapAt(holed, kShow),
  };

  safely('prose', () => initProse(data));
  safely('the roll', () => initScrollyRoll(data, state, computed));
  safely('the k dial', () => initKWidget(data, state, caches.k));
  safely('four methods', () => initCompareWidget(data, state, methods));
  safely('the hole', () => initHoleWidget(data, state, holed, holeCache));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, state, computed, caches, methods, holeCache));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 8000 });
  else setTimeout(guards, 1500);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
