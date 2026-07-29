/* Entry point.
 *
 * The guards here are heavier than the previous articles' for one reason: this
 * page does not just evaluate a network, it WALKS one, and a walk has its own
 * ways of being wrong that a single forward pass cannot expose. The step size,
 * the time the field is asked at, and whether the loop runs from noise to data
 * or the other way round are all decisions made outside the layers, so the
 * probe carries eight step endpoints and the page has to reproduce them.
 */
import { initPathWidget } from './path-widget.js';
import { initStepsWidget } from './steps-widget.js';
import { initSpreadWidget } from './spread-widget.js';
import { initStripWidget } from './strip-widget.js';
import { initLiveWidget } from './live-widget.js';
import { initProse } from './prose.js';
import { makeFlow } from './flowkit.js';

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
  if (!ok) console.warn(`[consistency] ${label}: ${detail}`);
  return ok;
}

/* worst relative error between what the page computed and what the generator
   measured, over a list of vectors */
function worstRel(got, want) {
  let worst = 0;
  want.forEach((wv, i) => {
    const n = Math.hypot(...wv);
    if (n < 1e-6) return;
    let d = 0;
    for (let k = 0; k < wv.length; k++) d += (got[i][k] - wv[k]) ** 2;
    worst = Math.max(worst, Math.sqrt(d) / n);
  });
  return worst;
}

function runGuards(data, kit) {
  const P = data.net.probe;
  const Z = P.z.map((z) => Float64Array.from(z));

  [['flow', P.flow, P.t], ['reflow', P.reflow, P.t], ['student', P.student, 0]]
    .forEach(([tag, want, t]) => {
      const got = Z.map((z) => kit.field(tag, z, t));
      check(`the ${tag} network`, worstRel(got, want) < 5e-2,
        `its output differs from the generator's by ${worstRel(got, want).toExponential(2)} `
        + 'relative, which means the weights or the time embedding were read wrongly');
    });

  /* the walk itself, which is where the step size and the time schedule live */
  [['flow', P.walk_flow], ['reflow', P.walk_reflow]].forEach(([tag, want]) => {
    const got = Z.map((z) => Array.from(kit.walk(tag, z, P.walk_steps).z));
    const e = worstRel(got, want);
    check(`walking the ${tag} arm`, e < 5e-2,
      `an ${P.walk_steps} step walk lands ${e.toExponential(2)} relative away from where the `
      + 'generator landed, so the step size or the times the field is asked at are wrong');
  });

  let worstPix = 0;
  P.dec_pixels.forEach((want, i) => {
    const got = kit.decode(Z[i]);
    for (let p = 0; p < want.length; p++) worstPix = Math.max(worstPix, Math.abs(got[p] - want[p]));
  });
  check('the decoder', worstPix < 5e-3,
    `a pixel it produces differs from the generator's by ${worstPix.toExponential(2)}`);

  /* the projection, because a live path drawn in a different frame from the
     stored ones would look like a result and be an artefact */
  let worstProj = 0;
  P.proj.forEach((want, i) => {
    const got = kit.project(Z[i]);
    worstProj = Math.max(worstProj, Math.abs(got[0] - want[0]), Math.abs(got[1] - want[1]));
  });
  check('the projection', worstProj < 5e-3,
    `it puts a probe point ${worstProj.toExponential(2)} away from where the generator put it, `
    + 'so stored paths and live ones are not in the same frame');

  /* the claim of the article, re-derived here rather than trusted: fewer steps
     must move the plain flow's answer more than they move the reflowed one */
  const z0 = Z.slice(0, 8);
  const gap = (tag) => {
    const ref = z0.map((z) => kit.walk(tag, z, 32).z);
    const one = z0.map((z) => kit.walk(tag, z, 1).z);
    let s = 0;
    for (let i = 0; i < z0.length; i++) {
      let d = 0;
      for (let k = 0; k < ref[i].length; k++) d += (one[i][k] - ref[i][k]) ** 2;
      s += Math.sqrt(d);
    }
    return s / z0.length;
  };
  const gFlow = gap('flow');
  const gRe = gap('reflow');
  check('the article\'s claim, re-derived here', data.straight.straighter === (gRe < gFlow),
    `the page finds one step off by ${gFlow.toFixed(3)} for the flow and ${gRe.toFixed(3)} after `
    + `reflow, which does not agree with the shipped straightness verdict `
    + `(${data.straight.straighter ? 'reflow straightened them' : 'it did not'})`);

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[consistency] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/consistency.json').then((r) => r.json());
  const kit = makeFlow(data);

  safely('prose', () => initProse(data));
  safely('the strips', () => initStripWidget(data));
  safely('the trajectories', () => initPathWidget(data));
  safely('steps against quality', () => initStepsWidget(data));
  safely('what the statistic can resolve', () => initSpreadWidget(data));
  safely('sampling here', () => initLiveWidget(data, kit));

  renderMath();

  const guards = () => safely('guards', () => runGuards(data, kit));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
