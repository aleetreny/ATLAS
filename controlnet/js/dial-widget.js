/* Ten digits asked for, at four settings of the dial, side by side.
 *
 * The pictures are the generator's, because the sweep they belong to was scored
 * on five hundred draws rather than the twenty on screen, and a reader
 * comparing settings should be comparing the thing that was measured. The live
 * widget further down runs the model instead.
 */
import { decodeWeights } from '../../assets/js/halfweights.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initDialWidget(data) {
  const M = data.meta;
  const side = M.side;
  const dim = M.pixels;
  const per = data.strips.per_label;
  const n = 10 * per;
  const slider = document.querySelector('#dial-slider');
  const valueTag = document.querySelector('#dial-value');
  const caption = document.querySelector('#dial-caption');
  const readout = document.querySelector('#dial-readout');
  const scales = M.show_scales;

  slider.min = 0;
  slider.max = scales.length - 1;
  slider.value = 1;

  /* Python wrote these keys with str(0.0) = "0.0"; JavaScript's String(0.0) is
     "0". Looking one up with the wrong spelling returns undefined and atob
     throws on it, which takes the whole widget down. Try both spellings. */
  const stripFor = (w) => data.strips.by_scale[String(w)]
    ?? data.strips.by_scale[Number(w).toFixed(1)]
    ?? data.strips.by_scale[String(Number(w))];
  const strips = {};
  scales.forEach((w) => { strips[w] = decodeWeights(stripFor(w), n * dim); });
  const real = decodeWeights(data.strips.real_b64, n * dim);

  const COLS = 10;
  const PAD = 1;
  const ROWS = per * 2 + PAD * 2;
  const ZOOM = 7;
  const cv = makeCanvas(document.querySelector('#dial-canvas'),
    COLS * side * ZOOM, ROWS * side * ZOOM);

  function render() {
    const w = scales[+slider.value];
    valueTag.textContent = `${w}`;
    const big = new Float64Array(COLS * side * ROWS * side).fill(1);
    /* the strips are laid out label major (two of each digit in a row), so a
       column of this grid is one requested label all the way down */
    const place = (src, index, rowBlock) => {
      const lab = Math.floor(index / per);
      const rep = index % per;
      const r = rowBlock + rep;
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[(r * side + a) * (COLS * side) + lab * side + b] =
            1 - src[index * dim + a * side + b];
        }
      }
    };
    for (let i = 0; i < n; i++) place(strips[w], i, PAD);
    for (let i = 0; i < n; i++) place(real, i, per + PAD * 2);
    paintGray(cv.ctx, big, COLS * side, ROWS * side, { zoom: ZOOM });
    const bandH = (per + PAD) * side * ZOOM;
    canvasLabel(cv.ctx, `asked for 0 to 9, guidance ${w}`, 6, PAD * side * ZOOM - 4,
      { align: 'start', size: 11 });
    canvasLabel(cv.ctx, 'real held out digits, same labels', 6, bandH + PAD * side * ZOOM - 4,
      { align: 'start', size: 11 });

    const row = data.sweep.reduce((a, r) => (Math.abs(r.scale - w) < Math.abs(a.scale - w) ? r : a));
    caption.textContent = `each column is one requested label, ${per} draws of it`;
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>guidance</th><th>makes what was asked</th><th>distance from real</th>
          <th>variety</th><th>real digits</th>
        </tr>
        <tr>
          <td><span class="value">${row.scale}</span></td>
          <td><span class="value">${(row.obeys * 100).toFixed(1)}%</span></td>
          <td><span class="value">${row.mmd2}</span></td>
          <td>${row.variety}</td>
          <td>${(data.real.obeys * 100).toFixed(1)}% and ${data.real.variety}</td>
        </tr>
      </table>
      <div class="gen-note">
        Obedience is the shared judge's <span class="bold">label</span> against the label that was
        requested, over ${M.asked} draws, and the column beside it is what that judge scores on
        real held out digits: ${(data.real.obeys * 100).toFixed(1)}%. That is deliberately not its
        confidence. The <a href="../ebm/">energy article</a> measured that this judge's confidence
        tracks ink rather than digits, and the guard says the same thing here
        (${data.ink.corr}), but which label it picks is a different question and a much steadier
        one. Variety is the mean distance between two draws of the same requested digit, so a
        model that answers every request with the same perfect picture scores one hundred percent
        on obedience and nearly zero here.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
