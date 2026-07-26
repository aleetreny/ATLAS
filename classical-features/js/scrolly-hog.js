/* From pixels to a descriptor, in five steps, all computed live from the
 * shipped PNG: the raw image, its gradients, the cell grid, one cell's
 * histogram opened up, and the whole rose.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { loadImage, imageToGray, makeCanvas, paintGray, pixelGrid, divergingColour, canvasLabel } from '../../assets/js/imagery.js';
import { gradients, cellHistograms, drawRose, maxAbs } from './hogkit.js';

const ZOOM = 3;                 // 128 * 3 = 384 px on screen
const CELL = 8;                 // Dalal and Triggs' own choice, and the rose stays legible
const ORI = 9;
/* The cell opened up in step 3, chosen by measurement rather than by eye: of
 * the cells in the top decile of gradient energy it is the most decisive one,
 * 88% of its weight in a single orientation bin. */
const FOCUS = { r: 11, c: 14 };

export async function initScrollyHog(data, base) {
  const img = await loadImage(`${base}${data.hog.image}`);
  const N = data.hog.size;
  const { gray } = imageToGray(img, N, N);
  const { mag, ori } = gradients(gray, N, N);
  const cells = cellHistograms(mag, ori, N, N, ORI, CELL);
  const magMax = maxAbs(mag);

  /* Tall enough for the image AND the opened-up histogram of step 3 sitting
   * under it: 26 + 384 for the picture, then a 60px gap, 44 of bars and three
   * lines of labels. */
  const W = 420;
  const H = 532;
  const { ctx } = makeCanvas('#hog-scrolly-chart', W, H);
  const x0 = (W - N * ZOOM) / 2;
  const y0 = 26;

  const clear = () => {
    ctx.clearRect(0, 0, W, H);
  };

  const caption = (text) => {
    canvasLabel(ctx, text, W / 2, 16, { size: 11 });
  };

  function paintImage() {
    paintGray(ctx, gray, N, N, { zoom: ZOOM, x0, y0 });
  }

  function paintGradients() {
    const norm = new Float32Array(mag.length);
    for (let i = 0; i < mag.length; i++) norm[i] = mag[i] / (magMax || 1);
    paintGray(ctx, norm, N, N, { zoom: ZOOM, x0, y0 });
  }

  function cellGrid(highlight) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= cells.nC; i++) {
      ctx.beginPath();
      ctx.moveTo(x0 + i * CELL * ZOOM + 0.5, y0);
      ctx.lineTo(x0 + i * CELL * ZOOM + 0.5, y0 + N * ZOOM);
      ctx.stroke();
    }
    for (let j = 0; j <= cells.nR; j++) {
      ctx.beginPath();
      ctx.moveTo(x0, y0 + j * CELL * ZOOM + 0.5);
      ctx.lineTo(x0 + N * ZOOM, y0 + j * CELL * ZOOM + 0.5);
      ctx.stroke();
    }
    if (highlight) {
      ctx.strokeStyle = '#df2a5d';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x0 + FOCUS.c * CELL * ZOOM, y0 + FOCUS.r * CELL * ZOOM,
        CELL * ZOOM, CELL * ZOOM);
    }
    ctx.restore();
  }

  /* The focused cell's histogram, drawn as a little bar chart under the image. */
  function focusHistogram() {
    const base_ = (FOCUS.r * cells.nC + FOCUS.c) * ORI;
    const vals = Array.from({ length: ORI }, (_, k) => cells.hist[base_ + k]);
    const peak = Math.max(...vals) || 1;
    const bw = 26;
    const totalW = ORI * bw;
    const bx = (W - totalW) / 2;
    const by = y0 + N * ZOOM + 60;
    ctx.save();
    ctx.fillStyle = '#78350f';
    vals.forEach((v, k) => {
      const hgt = (v / peak) * 44;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx + k * bw + 3, by - hgt, bw - 6, hgt);
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#232f3e';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx, by + 0.5);
    ctx.lineTo(bx + totalW, by + 0.5);
    ctx.stroke();
    ctx.restore();
    canvasLabel(ctx, '0', bx + 6, by + 14, { size: 9, align: 'center' });
    canvasLabel(ctx, '180', bx + totalW - 6, by + 14, { size: 9, align: 'center' });
    canvasLabel(ctx, 'gradient orientation, degrees', W / 2, by + 28, { size: 10 });

    /* State the reading rather than asserting it in prose: the dominant bin
     * and its share are measured here, so the caption cannot drift from the
     * picture if the focus cell or the settings ever change. */
    const total = vals.reduce((a, v) => a + v, 0) || 1;
    const k = vals.indexOf(peak);
    const edge = Math.round((k + 0.5) * (180 / ORI) + 90) % 180;
    canvasLabel(ctx, `dominant: an edge near ${edge} degrees, ` +
      `${Math.round((peak / total) * 100)}% of this cell's weight`, W / 2, by + 44, { size: 10 });
  }

  const states = {
    0: () => {
      clear();
      paintImage();
      caption(`${N} by ${N} pixels: ${data.hog.n_pixels.toLocaleString('en-US')} numbers`);
    },
    1: () => {
      clear();
      paintGradients();
      caption('gradient magnitude: where the brightness changes, and how fast');
    },
    2: () => {
      clear();
      paintGradients();
      cellGrid(false);
      caption(`chopped into ${cells.nR} by ${cells.nC} cells of ${CELL} by ${CELL} pixels`);
    },
    3: () => {
      clear();
      paintGradients();
      cellGrid(true);
      focusHistogram();
      caption(`one cell, ${CELL * CELL} pixels, becomes ${ORI} numbers`);
    },
    4: () => {
      clear();
      ctx.save();
      ctx.fillStyle = '#232f3e';
      ctx.fillRect(x0, y0, N * ZOOM, N * ZOOM);
      ctx.restore();
      drawRose(ctx, cells, CELL, ZOOM, { x0, y0, gain: 1.5 });
      caption('every cell as a rose of oriented lines: this is the descriptor');
    },
  };

  scrolly('#hog-scrolly .step', states);
  return { gray, N };
}
