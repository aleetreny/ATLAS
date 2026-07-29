/* The four samplers, side by side, before any explanation of why.
 *
 * Same twelve pieces of noise all the way down, so a row differs from the row
 * above it only in what was done with them. Real digits on top, for scale,
 * because "these look fine" is a sentence that needs something to be fine
 * against.
 */
import { decodeWeights } from '../../assets/js/halfweights.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initStripWidget(data) {
  const M = data.meta;
  const S = data.strips;
  const side = M.side;
  const dim = M.pixels;
  const per = S.per_pool;
  const caption = document.querySelector('#strip-caption');
  const readout = document.querySelector('#strip-readout');

  const rows = [{ key: null, label: 'real digits', pix: decodeWeights(S.real_b64, per * dim) }];
  S.show.forEach(({ arm, steps }) => {
    rows.push({
      key: `${arm}|${steps}`,
      label: `${arm}, ${steps} evaluation${steps === 1 ? '' : 's'}`,
      pix: decodeWeights(S.pools[`${arm}|${steps}`], per * dim),
    });
  });

  const ZOOM = 4;
  const GUT = 15;                       /* room for the label above each row */
  const W = per * side * ZOOM;
  /* the last row ends exactly at the canvas edge without the trailing pad, and
     a digit whose bottom pixel is the last pixel of the canvas reads as clipped */
  const H = rows.length * (side * ZOOM + GUT) + 6;
  const cv = makeCanvas(document.querySelector('#strip-canvas'), W, H);

  rows.forEach((row, r) => {
    const y0 = r * (side * ZOOM + GUT) + GUT;
    const big = new Float64Array(per * side * side).fill(1);
    for (let i = 0; i < per; i++) {
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[a * (per * side) + i * side + b] = 1 - row.pix[i * dim + a * side + b];
        }
      }
    }
    paintGray(cv.ctx, big, per * side, side, { zoom: ZOOM, x0: 0, y0 });
    canvasLabel(cv.ctx, row.label, 2, y0 - 4, { align: 'start', size: 11 });
  });

  const ink = (p) => {
    let s = 0;
    for (let i = 0; i < p.length; i++) s += p[i];
    return s / p.length;
  };
  const realInk = ink(rows[0].pix);

  caption.textContent = `the same ${per} pieces of noise down every row, so the only thing that `
    + 'changes is what the sampler did with them';

  const A = data.arms;
  readout.innerHTML = `
    <table class="gen-table">
      <tr><th>row</th><th>evaluations</th><th>distance from real</th><th>ink</th></tr>
      ${rows.map((row, i) => {
    if (i === 0) {
      return `<tr><td>real digits</td><td>&middot;</td><td>&middot;</td>`
        + `<td>${realInk.toFixed(4)}</td></tr>`;
    }
    const { arm, steps } = S.show[i - 1];
    const r = A.find((q) => q.arm === arm && q.steps === steps);
    return `<tr>
      <td>${arm}</td><td>${steps}</td>
      <td><span class="value">${r.mmd2}</span></td>
      <td>${ink(row.pix).toFixed(4)}</td>
    </tr>`;
  }).join('')}
    </table>
    <div class="gen-note">
      The second row is the one to look at, and the reason the rest of this page exists: the
      same model that produces the third row given four evaluations produces that given one.
      Nothing about the model changed, only how coarsely its trajectory was walked. The ink
      column is here because the shared judge is known to score ink rather than realism, so
      it is reported and never ranked.
    </div>`;

  return { rows };
}
