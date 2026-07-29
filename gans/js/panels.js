/* Shared furniture for this page: reading a contact sheet of samples back out
 * of a PNG, and painting a row of them.
 *
 * The samples are greyscale, so a tile is one Float32Array of tile*tile values
 * in [0, 1] and not three. Everything the article says about these images was
 * measured on the quantised bytes that the PNG carries, which is what the
 * browser reads, rather than on the floats the generator had in memory.
 */
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';

export async function loadSheet(src, tile, cols) {
  const img = await new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(`could not load ${src}`));
    im.src = src;
  });
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const rows = Math.round(img.naturalHeight / tile);
  const tiles = [];
  for (let k = 0; k < rows * cols; k++) {
    const r = Math.floor(k / cols);
    const cc = k % cols;
    const px = ctx.getImageData(cc * tile, r * tile, tile, tile).data;
    const f = new Float32Array(tile * tile);
    for (let i = 0; i < tile * tile; i++) f[i] = px[i * 4] / 255;
    tiles.push(f);
  }
  return { tiles, rows, cols, tile };
}

/* A block of tiles at an integer zoom, with an optional caption above it.
 * Integer zoom only: half a pixel of interpolation on a 28 pixel digit is the
 * difference between seeing what the generator produced and seeing what the
 * browser's resampler thinks about it. */
export function drawTiles(container, tiles, tile, cols, { zoom = 2, gap = 2, label = '' } = {}) {
  container.innerHTML = '';
  if (label) {
    const cap = document.createElement('div');
    cap.className = 'sheet-label';
    cap.innerHTML = label;
    container.appendChild(cap);
  }
  const rows = Math.ceil(tiles.length / cols);
  const w = cols * tile * zoom + (cols - 1) * gap;
  const h = rows * tile * zoom + (rows - 1) * gap;
  const { ctx } = makeCanvas(container, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);
  tiles.forEach((t, k) => {
    const r = Math.floor(k / cols);
    const c = k % cols;
    paintGray(ctx, t, tile, tile, {
      zoom,
      x0: c * (tile * zoom + gap),
      y0: r * (tile * zoom + gap),
    });
  });
  return ctx;
}
