/* The fitted field, rendered here, next to the scene it was fitted to.
 *
 * Both frames are computed in this tab at the camera the reader drags to: the
 * left one from the scene's closed form, the middle one from eighteen thousand
 * half precision weights. Nothing is precomputed, which is the point, and it
 * is also why the resolution and the sample count are exposed as controls
 * rather than fixed: the cost of a frame is the product of those two numbers
 * and the reader should get to feel it.
 */
import { renderTruthFull, camera, rayDir, psnr, depthImage, depthCompare } from './truthkit.js';
import { renderField } from './nerfkit.js';
import { makeFrame, attachOrbit, buttonRow, errorImage, fmt } from './panels.js';

const RES_CHOICES = [32, 40, 48, 64];
const STEP_CHOICES = [8, 16, 24, 32, 48, 64];
const TRUTH_STEPS = 96;
/* A frame of the field is pixels times samples times one network evaluation,
   and in scalar javascript that is a fraction of a gigaflop per second. The
   defaults are chosen so a release renders in well under a second on a modest
   machine, and the drag is cheaper still; both numbers are printed, so the
   reader can push them up and watch the cost follow. */
const DRAG_RES = 32;
const DRAG_STEPS = 12;

export function initFieldWidget(data, scene, field) {
  const meta = data.meta;
  const host = document.querySelector('#field-frames');
  host.innerHTML = '';
  const truthF = makeFrame(host, 'the scene', { max: 210 });
  const fieldF = makeFrame(host, 'the field', { max: 210 });
  const errF = makeFrame(host, 'difference, ×6', { max: 210 });

  const resSlider = document.querySelector('#field-res');
  const stepSlider = document.querySelector('#field-steps');
  const resVal = document.querySelector('#field-res-value');
  const stepVal = document.querySelector('#field-steps-value');
  const readout = document.querySelector('#field-readout');
  resSlider.max = String(RES_CHOICES.length - 1);
  stepSlider.max = String(STEP_CHOICES.length - 1);

  /* Open a quarter turn away from the held-out camera: opening exactly on it
     left the "go to a held-out camera" shortcut with nothing to do. */
  const state = {
    az: data.cameras.inside[0].az + 40, el: data.cameras.inside[0].el,
    resIdx: 1, stepIdx: 2, skip: true, importance: false, view: 'colour',
  };
  /* the depth ramp is fixed to the scene box, not to the frame, or the two
     panels would be using different scales and the difference between them
     would be unreadable */
  const dLo = meta.cam_radius - 1.4;
  const dHi = meta.cam_radius + 1.4;

  const bandLo = Math.min(...meta.rings);
  const bandHi = Math.max(...meta.rings);

  const render = (fast) => {
    const res = fast ? DRAG_RES : RES_CHOICES[state.resIdx];
    const steps = fast ? DRAG_STEPS : STEP_CHOICES[state.stepIdx];
    resVal.textContent = String(RES_CHOICES[state.resIdx]);
    stepVal.textContent = String(STEP_CHOICES[state.stepIdx]);
    const cam = camera(state.az, state.el, meta);
    const t0 = performance.now();
    const T = renderTruthFull(scene, cam, res, fast ? 48 : TRUTH_STEPS);
    const truth = T.rgb;
    const truthMs = performance.now() - t0;
    const dirFn = (i, j) => rayDir(cam, res, meta.fov_deg, i, j);
    const got = renderField(field, cam, res, steps, dirFn,
      { skip: state.skip, importance: state.importance, bg: meta.bg });
    /* A control that changes the picture has to say what the change was worth,
       or it is a button with an opinion. When the second pass is on, the same
       budget is also spent evenly, so the readout can compare them instead of
       implying that the fancier rule is the better one. */
    let evenly = null;
    if (state.importance && !fast) {
      evenly = renderField(field, cam, res, steps, dirFn,
        { skip: state.skip, importance: false, bg: meta.bg });
    }
    const geo = depthCompare(T.depth, T.alpha, got.depth, got.alpha);
    if (state.view === 'depth') {
      truthF.draw(depthImage(T.depth, T.alpha, meta.bg, dLo, dHi), res);
      fieldF.draw(depthImage(got.depth, got.alpha, meta.bg, dLo, dHi), res);
      const dErr = new Float32Array(res * res * 3);
      for (let p = 0; p < res * res; p++) {
        const both = T.alpha[p] > 0.5 && got.alpha[p] > 0.5;
        const e = both ? Math.min(1, Math.abs(T.depth[p] - got.depth[p]) * 8) : 0;
        dErr[p * 3] = 1;
        dErr[p * 3 + 1] = 1 - e;
        dErr[p * 3 + 2] = 1 - e * 0.92;
      }
      errF.draw(dErr, res);
      truthF.cap('the scene, depth');
      fieldF.cap('the field, depth');
      errF.cap('depth difference, ×8');
    } else {
      truthF.draw(truth, res);
      fieldF.draw(got.rgb, res);
      errF.draw(errorImage(got.rgb, truth), res);
      truthF.cap('the scene');
      fieldF.cap('the field');
      errF.cap('difference, ×6');
    }
    const inBand = state.el >= bandLo && state.el <= bandHi;
    readout.innerHTML = `Camera at azimuth <span class="bold">${state.az.toFixed(0)}°</span>, elevation `
      + `<span class="bold">${state.el.toFixed(0)}°</span>, `
      + (inBand
        ? `inside the band the photographs cover (${bandLo}° to ${bandHi}°). `
        : `<span class="bold">outside</span> the band the photographs cover (${bandLo}° to ${bandHi}°). `)
      + `The field scores <span class="bold">${fmt.db(psnr(got.rgb, truth))}</span> against the closed `
      + `form here, and puts the surface <span class="bold">${geo.mean.toFixed(4)}</span> scene units `
      + `away from where it is, averaged over the ${fmt.pc(geo.both / (res * res))} of pixels where both `
      + `agree there is something at all; they disagree about that on `
      + `<span class="bold">${fmt.pc(geo.frac)}</span> of the frame. `
      + `It took <span class="bold">${fmt.ms(got.ms)}</span> and `
      + `<span class="bold">${fmt.n0(got.evals)}</span> network evaluations for `
      + `${fmt.n0(res * res)} pixels at ${steps} samples a ray`
      + (state.skip ? `, after the occupancy grid threw away the empty ones` : `, with nothing skipped`)
      + `. Marching the closed form at the same resolution took ${fmt.ms(truthMs)}.`
      + (evenly
        ? ` Spending the same ${steps} samples evenly instead scores `
          + `<span class="bold">${fmt.db(psnr(evenly.rgb, truth))}</span>, so redrawing half of them by `
          + `weight ${psnr(got.rgb, truth) > psnr(evenly.rgb, truth) ? 'is worth' : 'costs'} `
          + `${Math.abs(psnr(got.rgb, truth) - psnr(evenly.rgb, truth)).toFixed(2)} dB here.`
        : '');
  };

  attachOrbit(host, state, (dragging) => render(dragging));
  resSlider.addEventListener('input', () => { state.resIdx = +resSlider.value; render(false); });
  stepSlider.addEventListener('input', () => { state.stepIdx = +stepSlider.value; render(false); });

  const views = document.querySelector('#field-views');
  const paintView = buttonRow(views, [
    { key: 'colour', label: 'colour' },
    { key: 'depth', label: 'depth' },
  ], state.view, (k) => { state.view = k; paintView(k); render(false); });

  const toggles = document.querySelector('#field-toggles');
  toggles.innerHTML = '';
  const mk = (label, get, set) => {
    const b = document.createElement('button');
    b.type = 'button';
    const paint = () => {
      b.textContent = label(get());
      b.className = get() ? 'atlas-btn' : 'atlas-btn ghost';
    };
    b.addEventListener('click', () => { set(!get()); paint(); render(false); });
    paint();
    toggles.appendChild(b);
    return paint;
  };
  mk((v) => (v ? 'skipping empty space' : 'every sample evaluated'),
    () => state.skip, (v) => { state.skip = v; });
  mk((v) => (v ? 'half the samples redrawn by weight' : 'evenly spaced samples'),
    () => state.importance, (v) => { state.importance = v; });

  /* Two shortcuts to the cameras the numbers in the prose refer to. These are
     actions rather than a view state, so neither of them is ever the filled
     button: nothing on this row is "currently selected". */
  [['a held-out camera', data.cameras.inside[0]], ['below the band', data.cameras.outside[0]]]
    .forEach(([label, c]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'atlas-btn ghost';
      b.textContent = `go to ${label}`;
      b.addEventListener('click', () => {
        state.az = c.az;
        state.el = c.el;
        render(false);
      });
      toggles.appendChild(b);
    });

  /* The first frame of the field is the most expensive thing on the page, so
     the text paints before it rather than after. Whichever of the two timers
     fires first does the render and the other becomes a no-op: an idle
     callback can be deferred a long way in a background tab, and a timeout
     runs whether or not the browser is composing frames at all. */
  truthF.blank(RES_CHOICES[state.resIdx]);
  fieldF.blank(RES_CHOICES[state.resIdx]);
  errF.blank(RES_CHOICES[state.resIdx]);
  readout.innerHTML = 'rendering the first frame...';
  let started = false;
  const first = () => {
    if (started) return;
    started = true;
    render(false);
  };
  if (window.requestIdleCallback) window.requestIdleCallback(first, { timeout: 500 });
  setTimeout(first, 250);
  return { render, state, first };
}
