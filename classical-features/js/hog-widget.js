/* HOG under the reader's hands: cell size and orientation count, recomputed
 * from the actual pixels on every input, with the descriptor length reported
 * next to the pixel count, because the surprise of this widget is that HOG is
 * not always a compression.
 */
import { loadImage, imageToGray, makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { gradients, cellHistograms, drawRose, descriptorLength } from './hogkit.js';

const ZOOM = 3;
const CELLS = [4, 8, 16, 32];
const ORIS = [4, 6, 9, 12, 18];

export async function initHogWidget(data, base) {
  const N = data.hog.size;
  const img = await loadImage(`${base}${data.hog.image}`);
  const { gray } = imageToGray(img, N, N);
  const { mag, ori } = gradients(gray, N, N);

  /* Two panels of 384 plus the gap between them: the canvas has to be at
   * least 784 wide or the left image is silently clipped. */
  const gap = 18;
  const panel = N * ZOOM;
  const W = panel * 2 + gap + 16;
  const H = N * ZOOM + 44;
  const { ctx } = makeCanvas('#hog-chart', W, H);
  const xLeft = (W - (panel * 2 + gap)) / 2;
  const xRight = xLeft + panel + gap;
  const y0 = 30;

  const cellSlider = document.querySelector('#hog-cell');
  const oriSlider = document.querySelector('#hog-ori');
  const readout = document.querySelector('#hog-readout');
  const overlayBtn = document.querySelector('#hog-overlay');
  const state = { overlay: false };

  function render() {
    const cell = CELLS[+cellSlider.value];
    const orientations = ORIS[+oriSlider.value];
    document.querySelector('#hog-cell-label').textContent = `${cell}x${cell}`;
    document.querySelector('#hog-ori-label').textContent = orientations;

    const cells = cellHistograms(mag, ori, N, N, orientations, cell);
    ctx.clearRect(0, 0, W, H);

    /* left: the photograph, so the eye can hold the original */
    paintGray(ctx, gray, N, N, { zoom: ZOOM, x0: xLeft, y0 });

    /* right: the descriptor, over the image or over ink */
    if (state.overlay) {
      paintGray(ctx, gray, N, N, { zoom: ZOOM, x0: xRight, y0 });
      ctx.save();
      ctx.fillStyle = 'rgba(35,47,62,0.55)';
      ctx.fillRect(xRight, y0, panel, panel);
      ctx.restore();
    } else {
      ctx.save();
      ctx.fillStyle = '#232f3e';
      ctx.fillRect(xRight, y0, panel, panel);
      ctx.restore();
    }
    drawRose(ctx, cells, cell, ZOOM, { x0: xRight, y0, colour: '#f7d9a0', gain: 1.15 });

    ctx.save();
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    ctx.textAlign = 'center';
    ctx.fillText('the pixels', xLeft + panel / 2, 18);
    ctx.fillText(`${cells.nR} x ${cells.nC} cells, ${orientations} orientations each`,
      xRight + panel / 2, 18);
    ctx.restore();

    const len = descriptorLength(N, N, orientations, cell);
    const ratio = len / data.hog.n_pixels;
    const shipped = data.hog.lengths.find((r) => r.orientations === orientations && r.cell === cell);
    readout.innerHTML =
      `<table class="feat-table">
         <tr><td>pixels in</td><td><span class="value">${data.hog.n_pixels.toLocaleString('en-US')}</span></td>
             <td>descriptor out</td><td><span class="value">${len.toLocaleString('en-US')}</span> numbers</td>
             <td>ratio</td><td><span class="value">${ratio.toFixed(2)}x</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        ratio > 1
          ? `Look at the ratio: at ${cell} by ${cell} cells this descriptor holds <span class="bold">more ` +
            `numbers than the image had pixels</span>. HOG's 2x2 blocks overlap, so every cell is counted ` +
            `four times over, once per block it belongs to. Compression was never the point; ` +
            `<span class="bold">invariance</span> was, and the next section measures whether it was bought.`
          : cell >= 32
            ? `Cells this large blur whole regions into one histogram: cheap (${len.toLocaleString('en-US')} ` +
              `numbers, ${(1 / ratio).toFixed(0)}x smaller than the pixels) and now visibly indifferent to ` +
              `where inside the cell anything sat. That indifference is the trade: position for robustness.`
            : `${len.toLocaleString('en-US')} numbers instead of ${data.hog.n_pixels.toLocaleString('en-US')} ` +
              `pixels, and notice what survives the shrinking: the shoulders, the helmet rim, the arm. ` +
              `Orientation histograms keep edges and throw away exact brightness and exact position.`
      }${shipped ? '' : ''}</div>`;
  }

  cellSlider.addEventListener('input', render);
  oriSlider.addEventListener('input', render);
  overlayBtn.addEventListener('click', () => {
    state.overlay = !state.overlay;
    overlayBtn.classList.toggle('ghost', !state.overlay);
    overlayBtn.textContent = state.overlay ? 'over the photograph' : 'on its own';
    render();
  });
  render();

  /* Runtime check: at the shipped setting, this page's cell histograms must
   * equal the ones scikit-image produced from the same PNG. */
  const ref = data.hog.ref;
  const mine = cellHistograms(mag, ori, N, N, ref.orientations, ref.cell).hist;
  let worst = 0;
  for (let i = 0; i < ref.hist.length; i++) worst = Math.max(worst, Math.abs(mine[i] - ref.hist[i]));
  if (worst > 1e-4) {
    console.warn(`browser HOG differs from the shipped reference by up to ${worst.toExponential(2)}`);
  }
  return worst;
}
