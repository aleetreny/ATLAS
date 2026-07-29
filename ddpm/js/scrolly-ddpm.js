/* The forward process, drawn as the reader scrolls.
 *
 * Every panel is q(x_t) computed here in closed form, not a stored picture, so
 * what is on screen is the function at whatever size the canvas happens to be.
 * The circle is the ring's hole, which is the same circle the flows article
 * drew and measured, and watching it fill in is the point of the sequence: the
 * forward process destroys structure on purpose and at a rate this page can
 * put a number on.
 */
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initScrollyDDPM(data, kit) {
  const host = document.querySelector('#ddpm-chart');
  const NPIX = 150;
  const HALF = 4.6;
  const ZOOM = 3;
  const cv = makeCanvas(host, NPIX * ZOOM, NPIX * ZOOM);
  /* which of the shipped levels each scrolly step shows */
  const STEP_LEVEL = [0, 1, 3, 5, 6];
  const ink = (t) => [255 - 216 * t, 255 - 177 * t, 255 - 158 * t];

  function render(step) {
    const level = STEP_LEVEL[Math.min(step, STEP_LEVEL.length - 1)];
    const t = data.net.page_ts[level];
    const vals = new Float64Array(NPIX * NPIX);
    let peak = 0;
    for (let i = 0; i < NPIX; i++) {
      for (let j = 0; j < NPIX; j++) {
        const x = -HALF + ((i + 0.5) / NPIX) * 2 * HALF;
        const y = -HALF + ((j + 0.5) / NPIX) * 2 * HALF;
        const q = kit.truthAt(level, x, y).q;
        vals[(NPIX - 1 - j) * NPIX + i] = q;
        if (q > peak) peak = q;
      }
    }
    const root = Math.sqrt(peak);
    for (let k = 0; k < vals.length; k++) vals[k] = Math.min(1, Math.sqrt(vals[k]) / root);
    paintGray(cv.ctx, vals, NPIX, NPIX, { zoom: ZOOM, colour: ink });

    const S = NPIX * ZOOM;
    const px = (x) => ((x + HALF) / (2 * HALF)) * S;
    cv.ctx.save();
    cv.ctx.strokeStyle = '#8a4b12';
    cv.ctx.lineWidth = 1.6;
    cv.ctx.setLineDash([5, 4]);
    cv.ctx.beginPath();
    cv.ctx.arc(px(0), px(0), (1.2 / (2 * HALF)) * S, 0, Math.PI * 2);
    cv.ctx.stroke();
    cv.ctx.restore();

    const row = data.destroy.rows.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
    canvasLabel(cv.ctx, `t = ${t}`, 8, 18, { align: 'start', size: 13 });
    canvasLabel(cv.ctx, `${row.kl_normal} nats from a standard normal`, 8, S - 8,
      { align: 'start', size: 11 });
    canvasLabel(cv.ctx, `hole ${row.hole}`, S - 8, S - 8, { align: 'end', size: 11 });
  }

  render(0);
  const handlers = {};
  for (let i = 0; i < 5; i++) handlers[i] = () => render(i);
  return scrolly('#ddpm-scrolly .step', handlers);
}
