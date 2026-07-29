/* The same process on 196 pixels, and what it costs there.
 *
 * This widget ships PICTURES rather than weights, and that is the measurement
 * rather than a shortcut. Reaching a sample statistic worth quoting on 14 by 14
 * digits took three quarters of a million weights, which is one and a half
 * megabytes in float16 and roughly seventy times the network that runs the
 * two dimensional process elsewhere on this page. So the page shows what that
 * model produced, beside real held out digits at the same size, and puts the
 * whole capacity ladder underneath.
 *
 * It is also, exactly, the argument for doing diffusion somewhere smaller than
 * pixel space, which is the next article.
 */
import { decodeWeights } from '../../assets/js/halfweights.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initPixelWidget(data) {
  const P = data.pixels;
  const side = data.meta.digit_side;
  const dim = side * side;
  const n = data.meta.strip;
  const caption = document.querySelector('#pixel-caption');
  const readout = document.querySelector('#pixel-readout');

  const samples = decodeWeights(P.samples_b64, n * dim);
  const real = decodeWeights(P.real_b64, n * dim);

  const COLS = 12;
  /* one blank row of pixels between the two blocks and above the first, so the
     captions sit on white instead of on the top row of digits */
  const PAD = 1;
  const ROWS = Math.ceil(n / COLS) * 2 + PAD * 2;
  const ZOOM = 6;
  const cv = makeCanvas(document.querySelector('#pixel-canvas'),
    COLS * side * ZOOM, ROWS * side * ZOOM);

  /* The values are stored inverted (1 - pixel), so the blank state of this
     array is ONE, not zero: left at zero the padding rows painted solid black
     bands between the two blocks. */
  const big = new Float64Array(COLS * side * ROWS * side).fill(1);
  const place = (src, index, rowBlock) => {
    const r = rowBlock + Math.floor(index / COLS);
    const c = index % COLS;
    for (let a = 0; a < side; a++) {
      for (let b = 0; b < side; b++) {
        const v = src[index * dim + a * side + b];
        big[(r * side + a) * (COLS * side) + c * side + b] = 1 - v;
      }
    }
  };
  for (let i = 0; i < n; i++) place(samples, i, PAD);
  for (let i = 0; i < n; i++) place(real, i, Math.ceil(n / COLS) + PAD * 2);
  paintGray(cv.ctx, big, COLS * side, ROWS * side, { zoom: ZOOM });
  const W = COLS * side * ZOOM;
  const bandH = (Math.ceil(n / COLS) + PAD) * side * ZOOM;
  canvasLabel(cv.ctx, 'made by the model', 6, PAD * side * ZOOM - 4,
    { align: 'start', size: 11 });
  canvasLabel(cv.ctx, 'real held out digits', 6, bandH + PAD * side * ZOOM - 4,
    { align: 'start', size: 11 });

  const best = P.ladder[P.ladder.length - 1];
  caption.textContent = `${n} draws from the ${best.params.toLocaleString('en-US')} weight model, `
    + `and ${n} real held out digits underneath, at the same size`;

  const ratio = P.biggest / P.toy_params;
  readout.innerHTML = `
    <table class="gen-table">
      <tr>
        <th>weights</th><th>float16</th><th>epochs</th>
        <th>distance from real</th><th>separable at</th>
      </tr>
      ${P.ladder.map((r) => `<tr>
        <td>${r.params.toLocaleString('en-US')}</td>
        <td>${r.kb_half} kB</td>
        <td>${r.epochs}</td>
        <td><span class="value">${r.mmd2}</span></td>
        <td>${r.c2st}</td>
      </tr>`).join('')}
    </table>
    <div class="gen-note">
      The floor here is ${P.floor.mmd2_max}, from real digits split in two, so every row of this
      table is still outside it: none of these models is passing for real data. What the ladder
      measures is the price. The two dimensional process elsewhere on this page runs on
      <span class="value">${P.toy_params.toLocaleString('en-US')}</span> weights; getting to
      ${best.mmd2} on ${dim} pixels took
      <span class="value">${P.biggest.toLocaleString('en-US')}</span>, which is
      ${ratio.toFixed(0)} times as many and ${best.kb_half} kB. That is why this widget ships
      ${n * 2} pictures instead of a network: at ${dim} numbers each they cost a fiftieth of the
      weights that made them.
      ${P.ink.confounded
    ? `The shared judge cannot be read here: its confidence correlates ${P.ink.corr} with the ink
       of the samples, which is the failure the energy based article measured on this same judge.`
    : `The ink guard passes, for once: the judge's confidence correlates ${P.ink.corr} with the
       ink across these pools, which is below the bar at which its ranking stops meaning
       anything. Worth saying out loud because on the previous article it did not pass.`}
      The samples average ${best.ink} ink against ${P.real_ink} for a real digit.
    </div>`;

  return { render: () => {} };
}
