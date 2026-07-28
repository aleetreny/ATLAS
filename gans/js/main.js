/* Entry point: load the three measurement files, boot every widget in
 * isolation, and then check that the arithmetic this page performs agrees with
 * the arithmetic the generator performed.
 *
 * Two independent things run here rather than being displayed here: the
 * divergences on the line, which are integrated on load, and the two player
 * game in the ring widget, which trains in the tab. Both have references in
 * the JSON and both are checked, because a page that computes its own figures
 * is only better than a page that quotes them if somebody verifies it.
 */
import { initProse } from './prose.js';
import { initGameScrolly } from './game-scrolly.js';
import { initLineWidget } from './line-widget.js';
import { initGradWidget, initDisjointWidget } from './grad-widget.js';
import { initRingGrid, initRingLive, initFalsify } from './ring-widget.js';
import {
  initMetricWidget, initSampleWidget, initCurveWidget, initTrackWidget, initInterpWidget,
} from './image-widget.js';
import { loadSheet } from './panels.js';
import { makeGrid, mixturePdf, normalPdf, divergences, valueAtOptimum } from './divkit.js';
import { LCG, MiniMLP, LiveGAN, modeReport } from './gankit.js';

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
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

/* Everything this page computes for itself, against what the generator
 * measured. The divergences have a tight tolerance because both sides run the
 * same quadrature on the same grid; the live GAN has a looser one on its
 * trajectory and a tight one on its frozen forward pass, because two hundred
 * Adam steps of tanh in two languages diverge for reasons that have nothing to
 * do with either being wrong, and saying which half is at fault needs the
 * frozen pass. */
function check(game, ring) {
  const problems = [];
  const worst = {};

  const grid = makeGrid(game.meta.grid);
  const p = mixturePdf(grid.x, game.meta.mixture);
  let dworst = 0;
  game.identity.rows.forEach((row) => {
    const q = normalPdf(grid.x, row.mu, row.sd);
    const dv = divergences(p, q, grid.dx);
    ['fwd', 'rev', 'js', 'w1'].forEach((k) => {
      dworst = Math.max(dworst, Math.abs(dv[k] - row[k]));
    });
    dworst = Math.max(dworst, Math.abs(valueAtOptimum(p, q, grid.dx) - row.value));
  });
  worst.divergences = dworst;
  if (dworst > 1e-7) {
    problems.push(`the divergences here differ from the generator's by ${dworst.toExponential(2)}`);
  }

  const rng = new LCG(ring.live.frozen.seed);
  const G = new MiniMLP(rng, [2, ring.live.h, ring.live.h, 2], 'tanh');
  const D = new MiniMLP(rng, [2, ring.live.h, ring.live.h, 1], 'leaky');
  const probe = ring.live.frozen.probe;
  const flat = new Float64Array(probe.length * 2);
  probe.forEach((pt, i) => { flat[i * 2] = pt[0]; flat[i * 2 + 1] = pt[1]; });
  const go = G.forward(flat, probe.length).a;
  const dopt = D.forward(flat, probe.length).a;
  let fworst = 0;
  ring.live.frozen.g_out.forEach((row, i) => row.forEach((v, k) => {
    fworst = Math.max(fworst, Math.abs(v - go[go.length - 1][i * 2 + k]));
  }));
  ring.live.frozen.d_out.forEach((row, i) => {
    fworst = Math.max(fworst, Math.abs(row[0] - dopt[dopt.length - 1][i]));
  });
  worst.frozen_pass = fworst;
  if (fworst > 1e-8) {
    problems.push(`the frozen forward pass differs by ${fworst.toExponential(2)}: the network in `
      + `this page is not the network the generator initialised`);
  }

  const live = new LiveGAN({
    seed: 7, h: ring.live.h, batch: ring.live.batch, arm: 'ns',
    k: ring.meta.k, r: ring.meta.r, sd: ring.meta.sd,
  });
  live.run(ring.live.steps);
  const ref = ring.live.runs.ns.losses;
  let lworst = 0;
  ref.forEach((row, i) => {
    lworst = Math.max(lworst, Math.abs(row[0] - live.losses[i][0]),
      Math.abs(row[1] - live.losses[i][1]));
  });
  worst.trajectory = lworst;
  if (lworst > 1e-3) {
    problems.push(`the live trajectory drifts from the generator's by ${lworst.toExponential(2)} `
      + `over ${ring.live.steps} steps`);
  }

  /* and the mode counter, on a cloud whose answer the generator already
     published, so the two counters are known to be the same counter */
  const rep = modeReport(ring.live.runs.ns.points, ring.meta.centres, ring.meta.sd);
  if (rep.covered !== ring.live.runs.ns.report_shown.covered) {
    problems.push(`the mode counter here says ${rep.covered} where the generator says `
      + `${ring.live.runs.ns.report_shown.covered} on the same points`);
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((pb) => console.warn(`  ${pb}`));
  } else {
    console.log('everything this page computes matches the generator, worst differences: '
      + Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(', '));
  }
  return { problems, worst };
}

async function boot() {
  const [game, ring, images] = await Promise.all([
    fetch('./data/game.json').then((r) => r.json()),
    fetch('./data/ring.json').then((r) => r.json()),
    fetch('./data/images.json').then((r) => r.json()),
  ]);

  const sheets = {};
  await Promise.all([
    ['samples', './img/samples.png', images.meta.tile, images.meta.sheet_cols],
    ['interp', './img/interp.png', images.meta.tile, 8],
  ].map(async ([k, src, tile, cols]) => {
    sheets[k] = await loadSheet(src, tile, cols);
  }));

  safely('the numbered sentences', () => initProse(game, ring, images));
  safely('the game scrolly', () => initGameScrolly(game));
  safely('which wrong answer', () => initLineWidget(game));
  safely('the vanishing gradient', () => initGradWidget(game));
  safely('disjoint supports', () => initDisjointWidget(game));
  safely('the ring of eight', () => initRingGrid(ring));
  safely('breaking the ring result', () => initFalsify(ring));
  safely('the live game', () => initRingLive(ring));
  safely('the yardstick', () => initMetricWidget(images));
  safely('the samples', () => initSampleWidget(images, sheets));
  safely('the training curves', () => initCurveWidget(images));
  safely('what you can watch', () => initTrackWidget(images));
  safely('the latent walk', () => initInterpWidget(images, sheets));
  renderMath();

  const run = () => check(game, ring);
  if (window.requestIdleCallback) window.requestIdleCallback(run);
  else setTimeout(run, 400);
  window.__atlasCheck = run;
  window.__atlas = { game, ring, images, sheets };
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
