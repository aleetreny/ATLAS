/* The two figures that are pictures rather than instruments: the training set,
 * and what happens to the specular lobe when the network is not told which way
 * the camera is pointing.
 */
import { paintRGB } from '../../assets/js/imagery.js';
import { drawTiles, fmt } from './panels.js';

/* All the photographs at once, on one canvas, with a hairline between frames
   so twenty-four small pictures do not read as one large one. */
export function initGallery(data, sheets) {
  const tile = data.sheets.train.tile;
  const cols = data.sheets.train.cols;
  const tiles = sheets.train.tiles;
  const rows = Math.ceil(tiles.length / cols);
  const gap = 3;
  const zoom = 2;
  const host = document.querySelector('#gallery-sheet');
  host.innerHTML = '';
  const canvas = document.createElement('canvas');
  const W = cols * tile * zoom + (cols - 1) * gap;
  const H = rows * tile * zoom + (rows - 1) * gap;
  canvas.width = W;
  canvas.height = H;
  canvas.style.width = '100%';
  canvas.style.maxWidth = `${W}px`;
  canvas.style.height = 'auto';
  canvas.style.display = 'block';
  canvas.style.margin = '0 auto';
  canvas.style.imageRendering = 'pixelated';
  host.appendChild(canvas);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#dfe3e3';
  ctx.fillRect(0, 0, W, H);
  tiles.forEach((t, k) => {
    const r = Math.floor(k / cols);
    const c = k % cols;
    paintRGB(ctx, t, tile, tile, { zoom, x0: c * (tile * zoom + gap), y0: r * (tile * zoom + gap) });
  });
  const cams = data.cameras.train;
  const els = [...new Set(cams.map((c) => c.el))].sort((a, b) => a - b);
  const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve'];
  const word = (v) => (v >= 0 && v <= 12 ? WORDS[v] : fmt.n0(v));
  document.querySelector('#gallery-readout').innerHTML =
    `${fmt.n0(cams.length)} photographs, ${data.meta.res} by ${data.meta.res} pixels each, on `
    + `${word(els.length)} rings at ${els.map((e) => `${e}°`).join(', ')} of elevation, `
    + `${word(cams.length / els.length)} to a ring. Every one is the same integral evaluated at `
    + `${fmt.n0(data.meta.gt_steps)} steps and then rounded to bytes, because a photograph is bytes: `
    + `nothing on this page ever sees a value the reader's browser could not read back out of the PNG.`;
}

/* Where the specular lobe lives, and what taking the direction away does to
   it. The frame average cannot see this, which is the point of the split. */
export function initViewdep(data, sheets) {
  const cfg = data.sheets.viewdep;
  drawTiles(document.querySelector('#viewdep-sheet'), sheets.viewdep.tiles, cfg.labels, cfg.tile,
    { max: 150 });
  const V = data.viewdep;
  document.querySelector('#viewdep-readout').innerHTML =
    `The view-dependent part of the radiance covers <span class="bold">${fmt.pc(V.frac, 2)}</span> of the `
    + `frame. Over the whole frame, taking the direction away moves the held-out PSNR from `
    + `${fmt.db(V.on_all)} to ${fmt.db(V.off_all)}. Inside those pixels it moves from `
    + `<span class="bold">${fmt.db(V.on_hot)}</span> to <span class="bold">${fmt.db(V.off_hot)}</span>, `
    + `and outside them from ${fmt.db(V.on_cold)} to ${fmt.db(V.off_cold)}.`;
}
