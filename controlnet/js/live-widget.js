/* The conditional sampler, run here, with the dial live.
 *
 * Two forward passes per step per picture, which is what guidance costs and is
 * worth feeling: the same weights answer "denoise this" and "denoise this, and
 * it is a four", and the sampler steps along the difference.
 */
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initLiveWidget(data, kit) {
  const M = data.meta;
  const side = M.side;
  const dim = M.pixels;
  const slider = document.querySelector('#live-slider');
  const valueTag = document.querySelector('#live-value');
  const caption = document.querySelector('#live-caption');
  const readout = document.querySelector('#live-readout');
  const buttons = document.querySelector('#live-buttons');
  const st = { label: 4, scaleIdx: 4 };

  /* half steps up to four, then whole ones: the interesting part is the bottom */
  const SCALES = [];
  for (let i = 0; i <= 8; i++) SCALES.push(i * 0.5);
  for (let i = 5; i <= 12; i++) SCALES.push(i);
  slider.min = 0;
  slider.max = SCALES.length - 1;
  slider.value = st.scaleIdx;

  const btns = {};
  for (let lab = 0; lab < 10; lab++) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = `${lab}`;
    b.addEventListener('click', () => { st.label = lab; render(); });
    buttons.appendChild(b);
    btns[lab] = b;
  }

  const COLS = 8;
  const ROWS = 2;
  const N = COLS * ROWS;
  const ZOOM = 7;
  const cv = makeCanvas(document.querySelector('#live-canvas'),
    COLS * side * ZOOM, ROWS * side * ZOOM);

  const cache = new Map();
  function draws(lab, scale) {
    const key = `${lab}:${scale}`;
    if (!cache.has(key)) {
      cache.set(key, kit.sample(new Array(N).fill(lab), scale, 40, 20250729));
    }
    return cache.get(key);
  }

  function render() {
    st.scaleIdx = +slider.value;
    const scale = SCALES[st.scaleIdx];
    valueTag.textContent = scale.toFixed(1);
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', +k !== st.label));

    const { pictures } = draws(st.label, scale);
    const big = new Float64Array(COLS * side * ROWS * side).fill(1);
    pictures.forEach((pic, index) => {
      const r = Math.floor(index / COLS);
      const c = index % COLS;
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[(r * side + a) * (COLS * side) + c * side + b] = 1 - pic[a * side + b];
        }
      }
    });
    paintGray(cv.ctx, big, COLS * side, ROWS * side, { zoom: ZOOM });
    canvasLabel(cv.ctx, `asked for ${st.label}, guidance ${scale.toFixed(1)}`, 6, 14,
      { align: 'start', size: 11 });

    /* how alike these eight are, which is the variety measure the sweep uses,
       computed here on what is actually on screen */
    let sum = 0;
    let pairs = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        let d = 0;
        for (let p = 0; p < dim; p++) d += (pictures[i][p] - pictures[j][p]) ** 2;
        sum += Math.sqrt(d);
        pairs++;
      }
    }
    const variety = sum / pairs;
    const ink = pictures.reduce((a, p) => a + p.reduce((x, v) => x + v, 0) / dim, 0) / N;

    caption.textContent = `${N} pictures made here: ${kit.steps > 0 ? 40 : 0} steps in a `
      + `${M.k} number space at guidance ${scale.toFixed(1)}, then one pass through the decoder`;

    const near = data.sweep.reduce(
      (a, r) => (Math.abs(r.scale - scale) < Math.abs(a.scale - scale) ? r : a));
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>asked for</th><th>guidance</th><th>variety here</th>
          <th>real digits</th><th>ink here</th>
        </tr>
        <tr>
          <td><span class="value">${st.label}</span></td>
          <td><span class="value">${scale.toFixed(1)}</span></td>
          <td><span class="value">${variety.toFixed(4)}</span></td>
          <td>${data.real.variety}</td>
          <td>${ink.toFixed(4)} against ${M.real_ink}</td>
        </tr>
      </table>
      <div class="gen-note">
        Everything above is computed in this page from
        <span class="value">${data.net.count.toLocaleString('en-US')}</span> float16 numbers, and
        every step costs two passes through the score network rather than one: guidance needs the
        answer with the label and the answer without it, and steps along the difference. That is
        the real price of the dial, and it is paid at sampling time rather than at training time.
        Drag it past ${data.best_obey} and watch the eight pictures become the same picture.
        The sweep above, on ${M.asked} draws rather than ${N}, scored ${near.variety} for variety
        at guidance ${near.scale}, against ${data.real.variety} for real digits.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
