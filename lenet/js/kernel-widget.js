/* Nine numbers you can change, applied to a real photograph on every keystroke.
 *
 * The point of letting the reader type the weights is that the famous kernels
 * stop being incantations: a blur is nine ninths, an edge detector is a centre
 * that disagrees with its neighbours, and the sum of the nine weights predicts
 * what happens to the overall brightness before you press anything.
 */
import { loadImage, imageToGray, makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';
import { convolve, signedToUnit, PRESETS } from './convkit.js';

const ZOOM = 2;

export async function initKernelWidget(data, base) {
  const S = data.kernel_image.size;
  const img = await loadImage(`${base}${data.kernel_image.image}`);
  const { gray } = imageToGray(img, S, S);

  const pad = 16;
  const panel = S * ZOOM;
  const W = pad * 3 + panel * 2;
  const H = panel + 44;
  const { ctx } = makeCanvas('#kernel-chart', W, H);
  const xL = pad;
  const xR = pad * 2 + panel;
  const y0 = 24;

  let kernel = PRESETS.edges.k.slice();

  /* ---------------------------------------------- the nine editable cells */
  const grid = document.querySelector('#kernel-grid');
  const inputs = [];
  for (let i = 0; i < 9; i++) {
    const inp = document.createElement('input');
    inp.type = 'number';
    /* A step of one, so the arrow keys do what the section above asks for:
       walk the centre weight from 5 down to 4 in a single press. Typing any
       decimal still works, since the step only constrains the arrows.
       ("any" would be the other candidate, but it disables the arrows
       entirely.) */
    inp.step = '1';
    inp.className = 'kernel-cell';
    inp.setAttribute('aria-label', `weight at row ${Math.floor(i / 3) + 1}, column ${(i % 3) + 1}`);
    inp.addEventListener('input', () => {
      const v = parseFloat(inp.value);
      kernel[i] = Number.isFinite(v) ? v : 0;
      draw();
    });
    grid.appendChild(inp);
    inputs.push(inp);
  }

  const buttons = document.querySelector('#kernel-presets');
  Object.entries(PRESETS).forEach(([key, p], i) => {
    const b = document.createElement('button');
    b.className = `atlas-button${key === 'edges' ? ' is-active' : ''}`;
    b.textContent = p.label;
    b.addEventListener('click', () => {
      kernel = p.k.slice();
      buttons.querySelectorAll('.atlas-button').forEach((n) => n.classList.remove('is-active'));
      b.classList.add('is-active');
      sync();
      draw();
    });
    buttons.appendChild(b);
  });

  /* Typing a weight by hand means the reader has left the preset behind. */
  inputs.forEach((inp) => inp.addEventListener('input', () => {
    buttons.querySelectorAll('.atlas-button').forEach((n) => n.classList.remove('is-active'));
  }));

  const sync = () => inputs.forEach((inp, i) => {
    const v = kernel[i];
    inp.value = Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, '');
  });

  /* ------------------------------------------------------------ rendering */
  const readout = document.querySelector('#kernel-readout');

  function verdict(sum) {
    if (Math.abs(sum) < 1e-9) {
      return 'The weights sum to zero, so flat regions cancel out exactly and come back mid grey. Only places where the pixels disagree survive: this is an edge detector, whatever else it is.';
    }
    if (Math.abs(sum - 1) < 1e-9) {
      return 'The weights sum to one, so a flat region keeps its brightness. This kernel rearranges detail without making the picture lighter or darker.';
    }
    if (sum > 1) {
      return `The weights sum to ${sum.toFixed(2)}, above one, so every flat region is amplified and the result runs bright before the rescaling brings it back.`;
    }
    if (sum > 0) {
      return `The weights sum to ${sum.toFixed(2)}, below one, so the picture darkens overall as well as changing.`;
    }
    return `The weights sum to ${sum.toFixed(2)}, below zero, so light and dark trade places: the output is a negative of what a positive kernel would give.`;
  }

  function draw() {
    const out = convolve(gray, S, S, kernel, 3);
    const unit = signedToUnit(out);
    ctx.clearRect(0, 0, W, H);

    paintGray(ctx, gray, S, S, { zoom: ZOOM, x0: xL, y0 });
    paintGray(ctx, unit, S, S, { zoom: ZOOM, x0: xR, y0 });
    ctx.save();
    ctx.strokeStyle = 'rgba(35,47,62,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(xL + 0.5, y0 + 0.5, panel - 1, panel - 1);
    ctx.strokeRect(xR + 0.5, y0 + 0.5, panel - 1, panel - 1);
    ctx.restore();
    canvasLabel(ctx, 'the photograph', xL + panel / 2, y0 - 8);
    canvasLabel(ctx, 'after your nine weights', xR + panel / 2, y0 - 8);

    const sum = kernel.reduce((a, b) => a + b, 0);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < out.length; i++) {
      if (out[i] < lo) lo = out[i];
      if (out[i] > hi) hi = out[i];
    }
    readout.innerHTML =
      `<span class="bold">Weights sum to ${sum.toFixed(3)}</span>. Responses run from ${lo.toFixed(2)} to ${hi.toFixed(2)}, `
      + `painted with zero as mid grey. ${verdict(sum)} `
      + `Every one of the ${(S * S).toLocaleString('en-US')} output pixels was computed here, in this page, when you last touched a cell.`;
  }

  sync();
  draw();
}
