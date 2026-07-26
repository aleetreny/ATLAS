/* The idea worth the article: any rectangle's sum in four lookups.
 *
 * The reader drags a rectangle over a real 25x25 face crop. The left panel
 * shows the pixels with the selection lit; the right shows the summed-area
 * table with exactly four cells lit, the four the answer needs. The readout
 * does the arithmetic both ways and counts the operations, which is where the
 * method stops being clever and starts being obviously right.
 */
import { makeCanvas, paintGray, pixelGrid, canvasLabel } from '../../assets/js/imagery.js';
import { integralImage } from './vjkit.js';

const ZOOM = 11;

export function initIntegralWidget(data) {
  const px = data.demo.pixels;                    // 25x25, integers 0..255
  const N = px.length;
  const gray = new Float32Array(N * N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) gray[r * N + c] = px[r][c] / 255;
  const ii = integralImage(gray, N, N);           // (N+1) x (N+1)

  const panel = N * ZOOM;
  const gap = 30;
  const W = panel * 2 + gap + 20;
  const H = panel + 54;
  const { canvas, ctx } = makeCanvas('#integral-chart', W, H);
  const x0 = 10;
  const x1 = x0 + panel + gap;
  const y0 = 30;

  const readout = document.querySelector('#integral-readout');
  /* the selection, in inclusive pixel coordinates */
  const sel = { r0: 6, c0: 5, r1: 14, c1: 13 };
  let drag = null;

  const toCell = (evt) => {
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width / W;
    const x = (evt.clientX - rect.left) / scale;
    const y = (evt.clientY - rect.top) / scale;
    return {
      c: Math.max(0, Math.min(N - 1, Math.floor((x - x0) / ZOOM))),
      r: Math.max(0, Math.min(N - 1, Math.floor((y - y0) / ZOOM))),
      inside: x >= x0 && x < x0 + panel && y >= y0 && y < y0 + panel,
    };
  };

  canvas.style.cursor = 'crosshair';
  canvas.addEventListener('pointerdown', (e) => {
    const p = toCell(e);
    if (!p.inside) return;
    drag = { r: p.r, c: p.c };
    sel.r0 = sel.r1 = p.r;
    sel.c0 = sel.c1 = p.c;
    canvas.setPointerCapture(e.pointerId);
    render();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const p = toCell(e);
    sel.r0 = Math.min(drag.r, p.r);
    sel.r1 = Math.max(drag.r, p.r);
    sel.c0 = Math.min(drag.c, p.c);
    sel.c1 = Math.max(drag.c, p.c);
    render();
  });
  canvas.addEventListener('pointerup', () => { drag = null; });

  function render() {
    ctx.clearRect(0, 0, W, H);

    /* left: the pixels, selection lit */
    paintGray(ctx, gray, N, N, { zoom: ZOOM, x0, y0 });
    ctx.save();
    ctx.fillStyle = 'rgba(30,64,175,0.30)';
    ctx.fillRect(x0 + sel.c0 * ZOOM, y0 + sel.r0 * ZOOM,
      (sel.c1 - sel.c0 + 1) * ZOOM, (sel.r1 - sel.r0 + 1) * ZOOM);
    ctx.strokeStyle = '#1e40af';
    ctx.lineWidth = 2;
    ctx.strokeRect(x0 + sel.c0 * ZOOM, y0 + sel.r0 * ZOOM,
      (sel.c1 - sel.c0 + 1) * ZOOM, (sel.r1 - sel.r0 + 1) * ZOOM);
    ctx.restore();
    pixelGrid(ctx, N, N, ZOOM, { x0, y0, colour: 'rgba(35,47,62,0.18)' });

    /* right: the summed-area table, four cells lit */
    const maxII = ii[ii.length - 1] || 1;
    const table = new Float32Array(N * N);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) table[r * N + c] = ii[(r + 1) * (N + 1) + (c + 1)] / maxII;
    }
    paintGray(ctx, table, N, N, { zoom: ZOOM, x0: x1, y0 });
    pixelGrid(ctx, N, N, ZOOM, { x0: x1, y0, colour: 'rgba(35,47,62,0.18)' });

    /* The four corners live on the ZERO-padded table, so a corner at row r0
     * means table row r0-1: draw them at the pixel grid's corners instead,
     * which is where a reader looks for them. */
    const corners = [
      { r: sel.r0 - 1, c: sel.c0 - 1, sign: '+' },
      { r: sel.r0 - 1, c: sel.c1, sign: '-' },
      { r: sel.r1, c: sel.c0 - 1, sign: '-' },
      { r: sel.r1, c: sel.c1, sign: '+' },
    ];
    ctx.save();
    for (const k of corners) {
      if (k.r < 0 || k.c < 0) continue;
      ctx.fillStyle = k.sign === '+' ? 'rgba(14,138,85,0.65)' : 'rgba(185,28,28,0.65)';
      ctx.fillRect(x1 + k.c * ZOOM, y0 + k.r * ZOOM, ZOOM, ZOOM);
      ctx.strokeStyle = '#232f3e';
      ctx.lineWidth = 1.4;
      ctx.strokeRect(x1 + k.c * ZOOM + 0.5, y0 + k.r * ZOOM + 0.5, ZOOM - 1, ZOOM - 1);
    }
    ctx.restore();

    canvasLabel(ctx, 'the pixels: drag a rectangle', x0 + panel / 2, 18, { size: 11 });
    canvasLabel(ctx, 'the summed-area table: four lookups', x1 + panel / 2, 18, { size: 11 });

    /* the arithmetic, both ways */
    let direct = 0;
    for (let r = sel.r0; r <= sel.r1; r++) for (let c = sel.c0; c <= sel.c1; c++) direct += px[r][c];
    const S = N + 1;
    const A = ii[sel.r0 * S + sel.c0];
    const B = ii[sel.r0 * S + (sel.c1 + 1)];
    const C = ii[(sel.r1 + 1) * S + sel.c0];
    const D = ii[(sel.r1 + 1) * S + (sel.c1 + 1)];
    const viaTable = (D - B - C + A) * 255;
    const nPix = (sel.r1 - sel.r0 + 1) * (sel.c1 - sel.c0 + 1);

    readout.innerHTML =
      `<table class="vj-table">
         <tr><td>rectangle</td><td>${nPix} pixels (${sel.r1 - sel.r0 + 1} by ${sel.c1 - sel.c0 + 1})</td>
             <td>adding them up</td><td><span class="value">${nPix}</span> operations</td></tr>
         <tr><td>sum</td><td><span class="value">${Math.round(direct).toLocaleString('en-US')}</span></td>
             <td>through the table</td><td><span class="value">4</span> lookups: D - B - C + A</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `The table says <span class="value">${Math.round(viaTable).toLocaleString('en-US')}</span>, ` +
      `the pixels say <span class="value">${Math.round(direct).toLocaleString('en-US')}</span>, and they ` +
      `agree because they must. Now drag a bigger rectangle: the left number climbs with the area and the ` +
      `right one stays at four. That is the whole trick, and it is why a detector can afford features the ` +
      `size of a whole face.</div>`;
  }

  render();
}
