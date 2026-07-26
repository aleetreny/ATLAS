/* What a convolution actually is, one step at a time.
 *
 * The whole operation is a small grid of numbers dragged over an image, and
 * every step here draws the same two panels: the pixels on the left with the
 * current window marked, the output being filled in on the right.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { loadImage, imageToGray, makeCanvas, paintGray, pixelGrid, canvasLabel } from '../../assets/js/imagery.js';
import { convolve, signedToUnit, shiftImage } from './convkit.js';

const S = 56;      // a crop small enough that individual pixels stay visible
const ZOOM = 6;    // six, not five, so the pixel grid on step 0 is drawable
const KERNEL = [0, -1, 0, -1, 4, -1, 0, -1, 0];   // the plain edge detector
const SHIFT = 9;

export async function initScrollyConv(data, base) {
  const img = await loadImage(`${base}${data.kernel_image.image}`);
  /* Take the top-left quarter of the photograph and resample it to S x S:
     a crop rather than the whole frame, so that a pixel is a visible square
     rather than a speck. */
  const full = imageToGray(img, data.kernel_image.size, data.kernel_image.size);
  const gray = new Float32Array(S * S);
  const step = Math.floor(full.w / 2 / S) || 1;
  for (let r = 0; r < S; r++) {
    for (let c = 0; c < S; c++) gray[r * S + c] = full.gray[(r * step + 24) * full.w + (c * step + 30)];
  }

  const pad = 14;
  const panel = S * ZOOM;
  const W = pad * 3 + panel * 2;
  const H = panel + 158;
  const { ctx } = makeCanvas('#conv-scrolly-chart', W, H);
  const xL = pad;
  const xR = pad * 2 + panel;
  const y0 = 26;

  const out = convolve(gray, S, S, KERNEL, 3);
  const outUnit = signedToUnit(out);

  /* Demonstrate the arithmetic where it has something to say: the interior
     position with the strongest response, rather than a patch of flat coat
     where all nine products are the same number. */
  let best = { r: 1, c: 1, v: -1 };
  for (let r = 2; r < S - 2; r++) {
    for (let c = 2; c < S - 2; c++) {
      const v = Math.abs(out[r * S + c]);
      if (v > best.v) best = { r, c, v };
    }
  }

  const frame = (x) => {
    ctx.save();
    ctx.strokeStyle = 'rgba(35,47,62,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y0 + 0.5, panel - 1, panel - 1);
    ctx.restore();
  };

  const left = (g = gray) => {
    paintGray(ctx, g, S, S, { zoom: ZOOM, x0: xL, y0 });
    frame(xL);
    canvasLabel(ctx, 'the pixels', xL + panel / 2, y0 - 9);
  };

  /* The output, revealed only as far as the window has travelled. */
  const right = (upto = S * S, arr = outUnit, label = 'what comes out') => {
    const shown = new Float32Array(S * S).fill(0.5);
    for (let i = 0; i < Math.min(upto, S * S); i++) shown[i] = arr[i];
    paintGray(ctx, shown, S, S, { zoom: ZOOM, x0: xR, y0 });
    frame(xR);
    canvasLabel(ctx, label, xR + panel / 2, y0 - 9);
  };

  const INK = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#15803d';

  const window3 = (r, c, x = xL) => {
    ctx.save();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x + (c - 1) * ZOOM - 1.5, y0 + (r - 1) * ZOOM - 1.5, 3 * ZOOM + 3, 3 * ZOOM + 3);
    ctx.restore();
  };

  const dot = (r, c, x) => {
    ctx.save();
    ctx.fillStyle = INK;
    ctx.fillRect(x + c * ZOOM, y0 + r * ZOOM, ZOOM, ZOOM);
    ctx.restore();
  };

  /* The operation itself, drawn rather than described: the nine pixels the
     window is sitting on, the nine weights, and the single number they make. */
  const CELL = 26;
  const arithmetic = (r, c) => {
    const gw = CELL * 3;
    const yT = y0 + panel + 22;
    const total = gw + 30 + gw + 96;
    const xA = (W - total) / 2;
    const xB = xA + gw + 30;

    const cellText = (x, y, i, j, text) => {
      canvasLabel(ctx, text, x + (j + 0.5) * CELL, y + (i + 0.6) * CELL + 3, { size: 10 });
    };
    const box = (x, y) => {
      ctx.save();
      ctx.strokeStyle = 'rgba(35,47,62,0.45)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 3; i++) {
        ctx.beginPath();
        ctx.moveTo(x + i * CELL + 0.5, y);
        ctx.lineTo(x + i * CELL + 0.5, y + gw);
        ctx.moveTo(x, y + i * CELL + 0.5);
        ctx.lineTo(x + gw, y + i * CELL + 0.5);
        ctx.stroke();
      }
      ctx.restore();
    };

    let acc = 0;
    const patch = new Float32Array(9);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const v = gray[(r + i - 1) * S + (c + j - 1)];
        patch[i * 3 + j] = v;
        acc += KERNEL[i * 3 + j] * v;
      }
    }

    paintGray(ctx, patch, 3, 3, { zoom: CELL, x0: xA, y0: yT });
    box(xA, yT);
    box(xB, yT);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        /* dark cells need light type and the other way round */
        const v = patch[i * 3 + j];
        ctx.save();
        ctx.fillStyle = v > 0.5 ? '#232f3e' : '#f1f3f3';
        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(v.toFixed(2), xA + (j + 0.5) * CELL, yT + (i + 0.6) * CELL + 3);
        ctx.restore();
        cellText(xB, yT, i, j, String(KERNEL[i * 3 + j]));
      }
    }
    canvasLabel(ctx, '×', xA + gw + 15, yT + gw / 2 + 4, { size: 15 });
    canvasLabel(ctx, `= ${acc.toFixed(3)}`, xB + gw + 48, yT + gw / 2 + 4, { size: 14 });
    canvasLabel(ctx, 'the nine pixels', xA + gw / 2, yT - 10, { size: 10 });
    canvasLabel(ctx, 'the nine weights', xB + gw / 2, yT - 10, { size: 10 });
    canvasLabel(ctx, 'one number, written at one place in the output', W / 2, yT + gw + 22, { size: 11 });
  };

  const clear = () => ctx.clearRect(0, 0, W, H);

  const steps = {
    0() {
      clear();
      left();
      right(0);
      canvasLabel(ctx, 'a grid of brightness values, nothing more', W / 2, y0 + panel + 30, { size: 12 });
    },
    1() {
      clear();
      const { r, c } = best;
      left();
      window3(r, c);
      right(0);
      dot(r, c, xR);
      window3(r, c, xR);
      arithmetic(r, c);
    },
    2() {
      clear();
      const idx = Math.round(S * S * 0.42);
      const r = Math.floor(idx / S);
      const c = idx % S;
      left();
      window3(Math.max(1, r), Math.max(1, c));
      right(idx);
      canvasLabel(ctx, 'the same nine weights, every position, no exceptions', W / 2, y0 + panel + 30, { size: 12 });
    },
    3() {
      clear();
      left();
      right();
      canvasLabel(ctx, 'nine numbers turned an image into an edge map', W / 2, y0 + panel + 30, { size: 12 });
    },
    4() {
      clear();
      const moved = shiftImage(gray, S, S, SHIFT, SHIFT);
      const movedOut = signedToUnit(convolve(moved, S, S, KERNEL, 3));
      paintGray(ctx, moved, S, S, { zoom: ZOOM, x0: xL, y0 });
      frame(xL);
      canvasLabel(ctx, `the pixels, moved ${SHIFT} px`, xL + panel / 2, y0 - 9);
      right(S * S, movedOut, 'the output, moved the same 9 px');
      canvasLabel(ctx, 'move the input, the output moves with it: equivariance', W / 2, y0 + panel + 30, { size: 12 });
      canvasLabel(ctx, 'nothing was retrained, and nothing had to be', W / 2, y0 + panel + 52, { size: 12 });
    },
  };

  /* A pixel grid is only legible on the very first step, where the point is
     that an image is discrete; later steps are about the sweep. */
  const withGrid = (fn) => () => {
    fn();
    pixelGrid(ctx, S, S, ZOOM, { x0: xL, y0, colour: 'rgba(35,47,62,0.18)' });
  };
  steps[0] = withGrid(steps[0]);

  scrolly('#conv-scrolly .step', steps);
  steps[0]();
}
