/* Stack small kernels and watch the window grow.
 *
 * VGG's whole design argument is a piece of arithmetic that is much easier to
 * believe once you have watched it: a stack of 3x3 layers sees as far as one
 * big kernel, costs less, and puts a nonlinearity between every step. Both
 * sliders redraw the picture and the counts at once.
 */
import { makeCanvas, canvasLabel } from '../../assets/js/imagery.js';
import { receptiveField, layerParams } from './depthkit.js';

const SIDE = 372;     // the drawing area, fixed; the grid inside it is not

export function initFieldWidget(data) {
  const C = data.vgg.channels;
  const pad = 16;
  const W = SIDE + pad * 2;
  const H = SIDE + 54;
  const { ctx } = makeCanvas('#field-chart', W, H);
  const x0 = pad;
  const y0 = 32;

  const style = getComputedStyle(document.documentElement);
  const INK = style.getPropertyValue('--primary').trim() || '#4c1d95';
  const SECOND = style.getPropertyValue('--secondary').trim() || '#3b0764';

  const kEl = document.querySelector('#field-k');
  const lEl = document.querySelector('#field-layers');
  const kLabel = document.querySelector('#field-k-value');
  const lLabel = document.querySelector('#field-layers-value');
  const readout = document.querySelector('#field-readout');
  let doubling = false;

  function draw() {
    const k = +kEl.value;
    const layers = +lEl.value;
    const rf = receptiveField(k, layers);

    /* The grid grows with the window rather than the other way round: at
       eleven-pixel kernels six deep the reach is 61 pixels, and a fixed grid
       would have the squares drawn off the edge of the canvas. */
    const GRID = rf + 4;
    const CELL = SIDE / GRID;
    const mid = Math.floor(GRID / 2);

    ctx.clearRect(0, 0, W, H);

    /* the input grid, drawn only while a cell is big enough to see */
    if (CELL >= 5) {
      ctx.save();
      ctx.strokeStyle = 'rgba(35,47,62,0.22)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= GRID; i++) {
        ctx.beginPath();
        ctx.moveTo(Math.round(x0 + i * CELL) + 0.5, y0);
        ctx.lineTo(Math.round(x0 + i * CELL) + 0.5, y0 + SIDE);
        ctx.moveTo(x0, Math.round(y0 + i * CELL) + 0.5);
        ctx.lineTo(x0 + SIDE, Math.round(y0 + i * CELL) + 0.5);
        ctx.stroke();
      }
      ctx.restore();
    }

    /* one square per layer, each reaching (k-1)/2 further out than the last */
    for (let l = layers; l >= 1; l--) {
      const r = receptiveField(k, l);
      const half = (r - 1) / 2;
      const px = x0 + (mid - half) * CELL;
      const py = y0 + (mid - half) * CELL;
      const size = r * CELL;
      ctx.save();
      ctx.fillStyle = INK;
      ctx.globalAlpha = 0.1;
      ctx.fillRect(px, py, size, size);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = l === layers ? INK : 'rgba(76,29,149,0.45)';
      ctx.lineWidth = l === layers ? 2.5 : 1.2;
      ctx.strokeRect(px + 0.5, py + 0.5, size - 1, size - 1);
      ctx.restore();
    }

    /* the one output pixel all of this feeds */
    ctx.save();
    ctx.fillStyle = SECOND;
    ctx.fillRect(x0 + mid * CELL, y0 + mid * CELL, CELL, CELL);
    ctx.restore();

    canvasLabel(ctx, `${layers} × ${k}×${k} sees a ${rf}×${rf} window`, W / 2, 20, { size: 13 });
    canvasLabel(ctx, 'the solid square is the one pixel that comes out',
      W / 2, y0 + SIDE + 20, { size: 11 });

    /* What it costs, against the single kernel that sees the same window.
       The usual telling of this argument quietly assumes the layer keeps the
       same number of channels. Real networks double the channels wherever
       they halve the resolution, and there the arithmetic can flip, so the
       reader gets to switch between the two. */
    let stacked;
    let single;
    if (doubling) {
      // channels go C in, 2C out, and only the first layer of the stack does it
      stacked = layerParams(k, C) * 2 + (layers - 1) * layerParams(k, C) * 4;
      single = layerParams(rf, C) * 2;
    } else {
      stacked = layers * layerParams(k, C);
      single = layerParams(rf, C);
    }
    const cheaper = single / stacked;
    const where = doubling
      ? `at ${C} channels in and ${C * 2} out`
      : `at ${C} channels in and out`;

    let verdict;
    if (layers === 1) {
      verdict = `One layer, so there is nothing to compare yet: this <span class="bold">is</span> the single ${rf}×${rf} kernel. Add a layer.`;
    } else if (cheaper > 1) {
      verdict = `A single ${rf}×${rf} kernel would see exactly the same window for `
        + `<span class="bold">${single.toLocaleString('en-US')}</span> weights, which is `
        + `<span class="bold">${cheaper.toFixed(2)}×</span> more. The stack also gets ${layers - 1} `
        + `extra nonlinearit${layers - 1 === 1 ? 'y' : 'ies'} the single kernel does not have, `
        + `for free.`;
    } else {
      verdict = `Here the stack is the <span class="bold">expensive</span> way round: a single `
        + `${rf}×${rf} kernel does the same job for ${single.toLocaleString('en-US')} weights `
        + `against your ${stacked.toLocaleString('en-US')}`
        + (doubling
          ? `. Doubling the channels is what flips it: every layer after the first works at the
             wider count, so the stack pays ${k}²·4C² per extra layer while the single kernel
             pays ${rf}²·2C² once.`
          : `. Small kernels only pay off while ${k}² × layers stays under ${rf}².`);
    }

    kLabel.textContent = `${k}×${k}`;
    lLabel.textContent = String(layers);
    readout.innerHTML =
      `<span class="bold">${layers} × ${k}×${k} = ${stacked.toLocaleString('en-US')} weights</span> `
      + `${where}, seeing a ${rf}×${rf} window. ${verdict}`;
  }

  const row = document.querySelector('#field-buttons');
  [['same channels in and out', false], ['channels double here', true]].forEach(([label, val], i) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${i === 0 ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      doubling = val;
      row.querySelectorAll('.atlas-btn').forEach((n) => n.classList.add('ghost'));
      b.classList.remove('ghost');
      draw();
    });
    row.appendChild(b);
  });

  kEl.addEventListener('input', draw);
  lEl.addEventListener('input', draw);
  draw();
}
