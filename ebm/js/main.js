/* Entry point.
 *
 * The guard that matters here is the one on the energy: the chains the reader
 * watches are walking on a function this page evaluates itself, so if the page
 * read the weights wrongly the walkers would still look plausible. They are
 * checked against the probe energies the generator measured on the same
 * quantised weights, and against its own log of the normalising constant.
 */
import { initScrollyEbm } from './scrolly-ebm.js';
import { initChainsWidget } from './chains-widget.js';
import { initBiasWidget } from './bias-widget.js';
import { initJemWidget } from './jem-widget.js';
import { initProse } from './prose.js';
import { makeEnergy, unpack } from './ebmkit.js';

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
  if (!ok) console.warn(`[ebm] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, E) {
  const M = data.model;
  const A = M.arms.exact;

  /* 1. the energy this page computes, against the energies the generator
        measured on the same quantised weights at the same points */
  let worst = 0;
  M.probes.forEach((p, i) => {
    worst = Math.max(worst, Math.abs(E.energy(p) - A.probe_energy[i]));
  });
  check('energy', worst < 5e-3,
    `a probe point's energy differs from the generator's by ${worst.toExponential(2)}, where `
    + `quantising the weights alone moves the drawn grid by ${A.quant_energy_shift}`);

  /* 2. the exported energy grid, against the same function evaluated here */
  const G = M.grid;
  const grid = unpack(A.energy_b64, G.m * G.m);
  let gworst = 0;
  for (let i = 0; i < G.m; i += 17) {
    for (let j = 0; j < G.m; j += 17) {
      const x = -G.half + ((i + 0.5) * 2 * G.half) / G.m;
      const y = -G.half + ((j + 0.5) * 2 * G.half) / G.m;
      gworst = Math.max(gworst, Math.abs(E.energy([x, y]) - grid[i * G.m + j]));
    }
  }
  check('energy grid', gworst < 0.05,
    `the exported landscape and the network that produced it differ by ${gworst.toFixed(4)} at `
    + `worst, against a float16 grid tolerance of ${A.grid_quant_max}`);

  /* 3. the gradient, against a central difference of the page's own energy */
  let dworst = 0;
  const h = 1e-4;
  M.probes.slice(0, 6).forEach((p) => {
    const g = E.grad(p);
    for (let j = 0; j < 2; j++) {
      const up = [p[0], p[1]];
      const dn = [p[0], p[1]];
      up[j] += h;
      dn[j] -= h;
      dworst = Math.max(dworst, Math.abs(g[j] - (E.energy(up) - E.energy(dn)) / (2 * h)));
    }
  });
  check('gradient', dworst < 1e-3,
    `the hand written gradient and a central difference of the same function differ by `
    + `${dworst.toExponential(2)}`);

  /* 4. the constant the whole article is about: stable across grid sizes */
  const zs = Object.values(A.logz);
  check('log Z', Math.max(...zs) - Math.min(...zs) < 1e-4,
    `the file reports the normalising constant moving by ${(Math.max(...zs) - Math.min(...zs))
      .toExponential(2)} between grid sizes`);

  /* 5. nothing may read undefined or NaN */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[ebm] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/ebm.json').then((r) => r.json());
  const E = makeEnergy(data, 'exact');

  safely('prose', () => initProse(data));
  safely('the landscape', () => initScrollyEbm(data));
  safely('the chains', () => initChainsWidget(data, E));
  safely('the bias', () => initBiasWidget(data));
  safely('the classifier', () => initJemWidget(data));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, E));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
