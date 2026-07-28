/* Entry point: load the scene, the field and the blobs, boot every widget in
 * isolation, and then check that all three renderers on this page agree with
 * the ones the generator measured.
 *
 * The checks matter more here than on any previous page, because this article
 * has three independent implementations of the same arithmetic living in the
 * browser: the closed-form scene, the MLP, and the rasteriser. Any one of them
 * quietly disagreeing with its reference would turn every PSNR on the page
 * into a number about the wrong thing.
 */
import { initProse } from './prose.js';
import { initGallery, initViewdep } from './figures.js';
import { initRayScrolly, initQuadWidget } from './ray-widget.js';
import { initEncodingWidget } from './encoding-widget.js';
import { initFieldWidget } from './field-widget.js';
import { initViewsWidget } from './views-widget.js';
import { initBlobWidget, initAnisoWidget } from './blob-widget.js';
import { initCostWidget } from './cost-widget.js';
import { Scene, renderTruth, rayDir, camera } from './truthkit.js';
import { Field, renderField } from './nerfkit.js';
import { Cloud, renderSplats } from './splatkit.js';
import { loadSheet } from './panels.js';

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

function compare(name, got, ref, tol, problems) {
  let worst = 0;
  let where = -1;
  ref.idx.forEach((p, k) => {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(got[p * 3 + c] - ref.rgb[k][c]);
      if (d > worst) { worst = d; where = p; }
    }
  });
  if (worst > tol) {
    problems.push(`${name}: worst pixel differs by ${worst.toExponential(2)} (pixel ${where}), `
      + `tolerance ${tol.toExponential(1)}`);
  }
  return worst;
}

function check(data, scene, field, cloud) {
  const problems = [];
  const K = data.check;
  const cam = K.cam;
  const worst = {};

  /* the camera itself, before anything is rendered through it */
  const dp = Math.max(...cam.pos.map((v, i) => Math.abs(v - cam.stored[i])));
  if (dp > 1e-6) problems.push(`camera position rebuilt from its angles is ${dp.toExponential(2)} away`);

  /* the closed form. Stopping a ray early is switched off here, so this is the
     same arithmetic on both sides and the tolerance can be tight. */
  worst.truth = compare('the scene in closed form',
    renderTruth(scene, cam, K.res, K.truth_steps, { stopAt: 0 }), K.truth, 1e-5, problems);

  /* the field, with every sample evaluated and then with the grid skipping.
     Both sides are float32 in the weights and float64 in the accumulation, so
     the floor is a few times 1e-6 and the tolerance is two orders above it. */
  const dirFn = (i, j) => rayDir(cam, K.res, data.meta.fov_deg, i, j);
  worst.field = compare('the field, every sample',
    renderField(field, cam, K.res, K.field_steps, dirFn, { skip: false, bg: data.meta.bg }).rgb,
    K.field, 2e-4, problems);
  worst.skipped = compare('the field, empty space skipped',
    renderField(field, cam, K.res, K.field_steps, dirFn, { skip: true, bg: data.meta.bg }).rgb,
    K.field_skipped, 2e-4, problems);

  /* the rasteriser */
  worst.splat = compare('the blob rasteriser',
    renderSplats(cloud, cam, K.res, data.meta.fov_deg, { bg: data.meta.bg, stopAt: 0 }).rgb,
    K.splat, 1e-4, problems);

  /* and the two things that are supposed to be true by construction */
  if (field.grid) {
    const occ = Array.from(field.grid).reduce((s, v) => s + v, 0) / field.grid.length;
    if (Math.abs(occ - data.nerf.grid_occupied) > 5e-4) {
      problems.push(`occupancy grid: ${occ.toFixed(4)} here against ${data.nerf.grid_occupied} offline`);
    }
  }
  if (cloud.n !== data.splat.n) problems.push(`blob count ${cloud.n} against ${data.splat.n}`);

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('every renderer on this page matches the generator, worst pixel differences: '
      + Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(', '));
  }
  return { problems, worst };
}

async function boot() {
  const [data, fieldModel, blobModel] = await Promise.all([
    fetch('./data/scene.json').then((r) => r.json()),
    fetch('./data/field.json').then((r) => r.json()),
    fetch('./data/blobs.json').then((r) => r.json()),
  ]);
  const scene = new Scene(data.meta, data.shapes);
  const field = new Field(fieldModel);
  const cloud = new Cloud(blobModel);

  /* Rebuild every camera from its two angles rather than reading the position
     the JSON carries. Six decimals of position is a ten thousandth of a pixel
     of ray, which is nothing to look at and everything to a guard: it put the
     worst check disagreement at 2.6e-5 when the renderers themselves agree to
     3e-6. The angles are exact, the formula is the generator's, so the guard
     now measures the implementation instead of the rounding. */
  const rebuild = (c) => ({ ...camera(c.az, c.el, data.meta), stored: c.pos });
  ['train', 'inside', 'outside'].forEach((k) => {
    data.cameras[k] = data.cameras[k].map(rebuild);
  });
  data.check.cam = rebuild(data.check.cam);

  const S = data.sheets;
  const sheets = {};
  const want = [
    ['train', './img/truth-train.png', S.train.tile, S.train.cols],
    ['encoding', './img/encoding.png', S.encoding.tile, S.encoding.cols],
    ['views', './img/views.png', S.views.tile, S.views.cols],
    ['viewdep', './img/viewdep.png', S.viewdep.tile, S.viewdep.cols],
    ['blobs', './img/blobs.png', S.blobs.tile, S.blobs.cols],
    ['compare', './img/compare.png', S.compare.tile, S.compare.cols],
  ];
  await Promise.all(want.map(async ([k, src, tile, cols]) => {
    sheets[k] = await loadSheet(src, tile, cols);
  }));

  safely('the numbered sentences', () => initProse(data));
  safely('the training set', () => initGallery(data, sheets));
  safely('the ray scrolly', () => initRayScrolly(data, scene));
  safely('the quadrature widget', () => initQuadWidget(data, scene, sheets));
  safely('the encoding widget', () => initEncodingWidget(data, sheets));
  safely('the live field', () => initFieldWidget(data, scene, field));
  safely('the camera rig', () => initViewsWidget(data, sheets));
  safely('view dependence', () => initViewdep(data, sheets));
  safely('the blobs', () => initBlobWidget(data, scene, cloud));
  safely('storage against quality', () => initAnisoWidget(data, sheets));
  safely('what a frame costs', () => initCostWidget(data, scene, field, cloud, sheets));
  renderMath();

  /* after the page is interactive: the checks take a moment and nothing on
     screen depends on them */
  const run = () => check(data, scene, field, cloud);
  if (window.requestIdleCallback) window.requestIdleCallback(run);
  else setTimeout(run, 400);
  window.__atlasCheck = run;
  window.__atlas = { data, scene, field, cloud, sheets };
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
