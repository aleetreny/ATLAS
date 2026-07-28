/* Reading the contact strip out of its PNG, and painting rows of tiles.
 *
 * The strip carries every figure on this page that is not computed live: the
 * plans, the true renders, what each arm produced from them, and the middle of
 * the conditional distribution. One file, one read, and the same painter for
 * everything so that a tile made offline and a tile made in this tab are drawn
 * identically.
 */
import { makeCanvas, paintRGB } from '../../assets/js/imagery.js';

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
    const f = new Float32Array(tile * tile * 3);
    for (let i = 0; i < tile * tile; i++) {
      f[i * 3] = px[i * 4] / 255;
      f[i * 3 + 1] = px[i * 4 + 1] / 255;
      f[i * 3 + 2] = px[i * 4 + 2] / 255;
    }
    tiles.push(f);
  }
  return { tiles, rows, cols, tile };
}

export function drawTiles(container, tiles, tile, cols, { zoom = 3, gap = 4, label = '' } = {}) {
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
    paintRGB(ctx, t, tile, tile, { zoom, x0: c * (tile * zoom + gap), y0: r * (tile * zoom + gap) });
  });
  return ctx;
}
