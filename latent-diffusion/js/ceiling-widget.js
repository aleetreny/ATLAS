/* The limit, drawn before any diffusion is involved.
 *
 * Real held out digits go in, the autoencoder's own round trip comes out, and
 * that is the best any model sampling in that space could ever produce. The
 * round trip for the size the page ships is computed HERE, from the encoder and
 * decoder in the data file, so the top row is not a stored picture: it is what
 * this decoder does to this digit, now.
 */
import { decodeWeights } from '../../assets/js/halfweights.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initCeilingWidget(data, kit) {
  const M = data.meta;
  const side = M.side;
  const dim = M.pixels;
  const n = data.strip.n;
  const real = decodeWeights(data.strip.real_b64, n * dim);
  const buttons = document.querySelector('#ceiling-buttons');
  const caption = document.querySelector('#ceiling-caption');
  const readout = document.querySelector('#ceiling-readout');
  const st = { view: 'roundtrip' };

  const COLS = 12;
  const PAD = 1;
  const ROWS = Math.ceil(n / COLS) * 2 + PAD * 2;
  const ZOOM = 6;
  const cv = makeCanvas(document.querySelector('#ceiling-canvas'),
    COLS * side * ZOOM, ROWS * side * ZOOM);

  const btns = {};
  for (const [key, text] of [['roundtrip', 'what the decoder can manage'],
    ['diff', 'what it lost, amplified']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  /* the round trips, computed once from the shipped encoder and decoder */
  const trips = [];
  for (let i = 0; i < n; i++) {
    trips.push(kit.roundTrip(real.subarray(i * dim, (i + 1) * dim)));
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    const big = new Float64Array(COLS * side * ROWS * side).fill(1);
    const place = (get, index, rowBlock) => {
      const r = rowBlock + Math.floor(index / COLS);
      const c = index % COLS;
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[(r * side + a) * (COLS * side) + c * side + b] = 1 - get(index, a * side + b);
        }
      }
    };
    for (let i = 0; i < n; i++) place((j, p) => real[j * dim + p], i, PAD);
    if (st.view === 'roundtrip') {
      for (let i = 0; i < n; i++) place((j, p) => trips[j][p], i, Math.ceil(n / COLS) + PAD * 2);
    } else {
      /* the difference, scaled so it is visible at all: the point is where the
         loss lives, not how big it is */
      let peak = 1e-9;
      for (let i = 0; i < n; i++) {
        for (let p = 0; p < dim; p++) {
          peak = Math.max(peak, Math.abs(trips[i][p] - real[i * dim + p]));
        }
      }
      for (let i = 0; i < n; i++) {
        place((j, p) => Math.abs(trips[j][p] - real[j * dim + p]) / peak,
          i, Math.ceil(n / COLS) + PAD * 2);
      }
    }
    paintGray(cv.ctx, big, COLS * side, ROWS * side, { zoom: ZOOM });
    const bandH = (Math.ceil(n / COLS) + PAD) * side * ZOOM;
    canvasLabel(cv.ctx, 'real held out digits', 6, PAD * side * ZOOM - 4,
      { align: 'start', size: 11 });
    canvasLabel(cv.ctx, st.view === 'roundtrip'
      ? `the same digits, in and out of ${/^[aeiou8]/.test(String(M.k_show)) ? 'an' : 'a'} ${M.k_show} number bottleneck`
      : 'what the bottleneck threw away, scaled to be visible',
    6, bandH + PAD * side * ZOOM - 4, { align: 'start', size: 11 });

    const row = data.ceiling.find((r) => r.k === M.k_show);
    caption.textContent = st.view === 'roundtrip'
      ? `computed in this page, from the encoder and decoder in its data file`
      : `the absolute difference between the two rows above, divided by its own largest value`;

    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>bottleneck</th><th>round trip error</th><th>distance from real</th>
          <th>separable at</th><th>inside the floor</th>
        </tr>
        ${data.ceiling.map((r) => `<tr>
          <td>${r.k === M.k_show ? '<span class="bold">' + r.k + '</span>' : r.k}</td>
          <td>${r.mse}</td>
          <td><span class="value">${r.mmd2}</span></td>
          <td>${r.c2st}</td>
          <td>${r.resolves ? 'no' : '<span class="bold">yes</span>'}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        No diffusion has happened yet. Every number in this table is a real digit encoded and
        decoded again, which is what a latent model's output passes through on its way to the
        reader: whatever the sampler does in there, the picture you see came out of this decoder.
        So each row is a <span class="bold">ceiling</span>, and the floor beside it is the same
        one the pixel article used, ${data.floor.mmd2_max} from real digits split in two.
        ${data.ceiling.some((r) => !r.resolves)
    ? `At ${data.ceiling.filter((r) => !r.resolves).map((r) => r.k).join(' and ')} numbers the
       round trip is already inside that floor, which means a perfect sampler in that space
       would be indistinguishable from real data at this sample size.`
    : 'None of these round trips is inside that floor, so every one of them is a real limit.'}
        The smallest one is worth staring at: at ${data.ceiling[0].k} numbers the ceiling is
        ${data.ceiling[0].mmd2}, and the pixel article's best model, with
        ${data.pixel_ladder[data.pixel_ladder.length - 1].params.toLocaleString('en-US')}
        weights, scored ${data.pixel_ladder[data.pixel_ladder.length - 1].mmd2}. A latent model
        in that space loses to it before it starts.
      </div>`;
  }

  render();
  return { render };
}
