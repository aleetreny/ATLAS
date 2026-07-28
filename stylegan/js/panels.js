/* Reading a contact sheet of RGB sprites back out of a PNG, and painting
 * tiles that this page produced itself.
 *
 * Two sources of pixels live on this page: the PNG sheets the generator wrote,
 * and the images the generator in stylekit.js produces here. Both go through
 * the same painter so that a sprite made in this tab and a sprite made offline
 * are drawn identically and can be compared by eye without an asterisk.
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

/* A row or grid of RGB tiles at an integer zoom, with a caption under each.
 * Integer zoom only: at 32 pixels a sprite is small enough that half a pixel
 * of resampling is the difference between a shape and a smudge. */
export function drawTiles(container, tiles, tile, cols, {
  zoom = 3, gap = 4, caps = null, label = '',
} = {}) {
  container.innerHTML = '';
  if (label) {
    const cap = document.createElement('div');
    cap.className = 'sheet-label';
    cap.innerHTML = label;
    container.appendChild(cap);
  }
  const rows = Math.ceil(tiles.length / cols);
  const capH = caps ? 16 : 0;
  const w = cols * tile * zoom + (cols - 1) * gap;
  const h = rows * (tile * zoom + capH) + (rows - 1) * gap;
  const { ctx } = makeCanvas(container, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);
  tiles.forEach((t, k) => {
    const r = Math.floor(k / cols);
    const c = k % cols;
    const x0 = c * (tile * zoom + gap);
    const y0 = r * (tile * zoom + capH + gap);
    paintRGB(ctx, t, tile, tile, { zoom, x0, y0 });
    if (caps && caps[k]) {
      ctx.fillStyle = '#232f3e';
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(caps[k], x0 + (tile * zoom) / 2, y0 + tile * zoom + 12);
    }
  });
  return ctx;
}
