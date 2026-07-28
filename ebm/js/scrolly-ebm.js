/* Two landscapes and the truth they are trying to be, on the same square.
 *
 * The pictures are the grids the generator exported: the truth's log density,
 * the energy the exactly trained arm ended with, and the energy the one step
 * shortcut ended with. Nothing here is redrawn from a description: the last
 * step is the difference between the two learned landscapes, which is what the
 * shortcut costs, drawn rather than summarised.
 */
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { unpack } from './ebmkit.js';

export function initScrollyEbm(data) {
  const G = data.model.grid;
  const n = G.m;
  const truth = unpack(G.truth_logp_b64, n * n);
  const exact = unpack(data.model.arms.exact.energy_b64, n * n);
  const cd1 = unpack(data.model.arms.cd1.energy_b64, n * n);
  const holder = document.querySelector('#ebm-chart');
  const ZOOM = Math.max(2, Math.round(430 / n));
  const cv = makeCanvas(holder, n * ZOOM, n * ZOOM);
  const caption = document.createElement('div');
  caption.className = 'canvas-caption';
  caption.style.maxWidth = `${n * ZOOM}px`;
  caption.style.margin = '0.6rem auto 0';
  holder.appendChild(caption);

  const ink = (t) => [255 - 160 * t, 255 - 219 * t, 255 - 237 * t];
  const anchor = (t) => [255 - 255 * t, 255 - 206 * t, 255 - 126 * t];

  function normalise(src, invert) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < src.length; i++) {
      if (!Number.isFinite(src[i])) continue;
      lo = Math.min(lo, src[i]);
      hi = Math.max(hi, src[i]);
    }
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = (src[i * n + j] - lo) / Math.max(hi - lo, 1e-9);
        if (invert) v = 1 - v;
        out[(n - 1 - j) * n + i] = Math.sqrt(Math.max(0, Math.min(1, v)));
      }
    }
    return out;
  }

  const views = [
    { src: truth, invert: false, colour: ink, dots: true,
      text: `the density this module is fitted against, and ${data.truth.sample.length} draws of it` },
    { src: truth, invert: false, colour: ink, dots: false,
      text: 'the same picture with the data taken away: this is what the energy has to become' },
    { src: exact, invert: true, colour: ink, dots: false,
      text: 'the energy the exactly trained arm ended with, dark where it says likely' },
    { src: cd1, invert: true, colour: ink, dots: false,
      text: 'and the energy one sampling step per update ends with instead' },
    { src: null, invert: false, colour: null, dots: false,
      text: 'the difference between the two, in this article’s accent where the shortcut is more confident' },
  ];

  function render(step) {
    const v = views[Math.min(step, views.length - 1)];
    if (v.src) {
      paintGray(cv.ctx, normalise(v.src, v.invert), n, n, { zoom: ZOOM, colour: v.colour });
    } else {
      const diff = new Float64Array(n * n);
      let m = 0;
      for (let i = 0; i < n * n; i++) m = Math.max(m, Math.abs(cd1[i] - exact[i]));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          diff[(n - 1 - j) * n + i] = (exact[i * n + j] - cd1[i * n + j]) / (m || 1);
        }
      }
      paintGray(cv.ctx, diff, n, n, { zoom: ZOOM,
        colour: (t) => (t >= 0 ? ink(Math.min(1, t)) : anchor(Math.min(1, -t))) });
    }
    if (v.dots) {
      const span = 2 * G.half;
      cv.ctx.save();
      cv.ctx.fillStyle = 'rgba(35,47,62,0.8)';
      data.truth.sample.forEach((p) => {
        const cx = ((p[0] + G.half) / span) * n * ZOOM;
        const cy = (1 - (p[1] + G.half) / span) * n * ZOOM;
        cv.ctx.beginPath();
        cv.ctx.arc(cx, cy, 1.2, 0, 2 * Math.PI);
        cv.ctx.fill();
      });
      cv.ctx.restore();
    }
    canvasLabel(cv.ctx, `${-G.half}`, 4, n * ZOOM - 5, { align: 'start', size: 10 });
    canvasLabel(cv.ctx, `${G.half}`, n * ZOOM - 4, n * ZOOM - 5, { align: 'end', size: 10 });
    caption.textContent = v.text;
  }

  render(0);
  const handlers = {};
  for (let i = 0; i < 5; i++) handlers[i] = () => render(i);
  const ctl = scrolly('#ebm-scrolly .step', handlers);
  return { render, goTo: ctl.goTo };
}
