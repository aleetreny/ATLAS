/* What the networks on this page are actually looking at.
 *
 * Not interactive, and it does not need to be: the point is that the reader
 * should be able to see the task before reading twelve training runs about
 * it, and that a pullover, a coat and a shirt are genuinely hard to tell
 * apart at 28 pixels, which is why this article uses garments rather than
 * the handwritten digits of the previous one.
 */
import { loadImage, imageToGray, makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

const ZOOM = 3;

export async function initGarments(data, base) {
  const { image, tile, count, labels } = data.garments;
  const names = data.data.classes;
  const img = await loadImage(`${base}${image}`);
  const strip = imageToGray(img, tile * count, tile);

  const cell = tile * ZOOM;
  const gap = 8;
  const pad = 10;
  const W = pad * 2 + count * cell + (count - 1) * gap;
  const H = cell + 40;
  const { ctx } = makeCanvas('#garments-chart', W, H);
  const y0 = 8;

  for (let i = 0; i < count; i++) {
    const g = new Float32Array(tile * tile);
    for (let r = 0; r < tile; r++) {
      for (let c = 0; c < tile; c++) g[r * tile + c] = strip.gray[r * strip.w + i * tile + c];
    }
    const x = pad + i * (cell + gap);
    paintGray(ctx, g, tile, tile, { zoom: ZOOM, x0: x, y0 });
    ctx.save();
    ctx.strokeStyle = 'rgba(35,47,62,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y0 + 0.5, cell - 1, cell - 1);
    ctx.restore();
    /* The names are long and the cells are narrow, so the label goes on two
       lines where it has to rather than colliding with its neighbour. */
    const name = names[labels[i]];
    const parts = name.split(' ');
    canvasLabel(ctx, parts[0], x + cell / 2, y0 + cell + 14, { size: 9 });
    if (parts[1]) canvasLabel(ctx, parts[1], x + cell / 2, y0 + cell + 24, { size: 9 });
  }
}
