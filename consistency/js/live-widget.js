/* All three samplers, run here, with the step budget as a dial.
 *
 * The interesting thing to do with this widget is not to look at the pictures
 * but at the counter under them: at one evaluation two of these three arms are
 * doing the same amount of arithmetic as the third does at a hundred, and the
 * question the whole article asks is what that difference is worth. Drag the
 * budget down on the plain rectified flow and it collapses; drag it down on the
 * other two and, on this model, very little happens.
 */
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

const ARMS = [['flow', 'rectified flow'], ['reflow', 'after reflow'],
  ['student', 'distilled student']];
const LONG = { flow: 'rectified flow', reflow: 'after reflow', student: 'distilled student' };

export function initLiveWidget(data, kit) {
  const M = data.meta;
  const B = M.budgets;
  const side = M.side;
  const dim = M.pixels;
  const slider = document.querySelector('#live-slider');
  const valueTag = document.querySelector('#live-value');
  const buttons = document.querySelector('#live-buttons');
  const caption = document.querySelector('#live-caption');
  const readout = document.querySelector('#live-readout');
  const st = { arm: 'flow', budget: 0 };

  slider.min = 0;
  slider.max = B.length - 1;
  slider.value = 0;

  const btns = {};
  ARMS.forEach(([key, text]) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.arm = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  });

  const COLS = 8;
  const ROWS = 2;
  const N = COLS * ROWS;
  const ZOOM = 7;
  const cv = makeCanvas(document.querySelector('#live-canvas'),
    COLS * side * ZOOM, ROWS * side * ZOOM);

  /* one set of starting points for every arm and every budget, so a change on
     screen is a change in the sampler and never a change in the noise */
  const Z0 = kit.noise(N, 20250729);

  const cache = new Map();
  function draws(arm, steps) {
    const key = `${arm}:${steps}`;
    if (!cache.has(key)) {
      cache.set(key, Z0.map((z) => kit.decode(kit.run(arm, z, steps).z)));
    }
    return cache.get(key);
  }

  function render() {
    st.budget = +slider.value;
    const steps = st.arm === 'student' ? 1 : B[st.budget];
    /* The student has no budget, so the control is switched off rather than
       left responding to a reader who would then wonder why nothing moved. A
       disabled slider with an unchanged number beside it is the same dead
       control from the reader's side, so the number says why. */
    slider.disabled = st.arm === 'student';
    valueTag.textContent = st.arm === 'student'
      ? 'one, by construction'
      : `${steps} evaluation${steps === 1 ? '' : 's'}`;
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.arm));

    const pictures = draws(st.arm, steps);
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
    canvasLabel(cv.ctx, `${LONG[st.arm]}, ${steps} evaluation${steps === 1 ? '' : 's'}`,
      6, 14, { align: 'start', size: 11 });

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

    caption.textContent = `${N} pictures made here: ${steps} pass`
      + `${steps === 1 ? '' : 'es'} through a ${M.hidden} wide network on a ${M.k} number `
      + 'latent, then one pass through the decoder';

    const row = data.arms.find((r) => r.arm === LONG[st.arm] && r.steps === steps);
    const nEval = N * steps;
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>sampler</th><th>evaluations each</th><th>network passes here</th>
          <th>variety</th><th>ink</th><th>measured distance</th>
        </tr>
        <tr>
          <td>${LONG[st.arm]}</td>
          <td><span class="value">${steps}</span></td>
          <td>${nEval.toLocaleString('en-US')}</td>
          <td><span class="value">${variety.toFixed(4)}</span></td>
          <td>${ink.toFixed(4)}</td>
          <td>${row ? row.mmd2 : '&middot;'}</td>
        </tr>
      </table>
      <div class="gen-note">
        Everything above runs in this page from
        <span class="value">${data.net.count.toLocaleString('en-US')}</span> float16 numbers
        holding all three samplers and the decoder. The distance in the last column is the
        measurement from the chart above, on ${'250'} samples rather than these ${N}, and it is
        the number to trust: sixteen pictures is a look, not a measurement.
        ${st.arm === 'flow'
    ? `Take this arm down to one evaluation and watch it fall apart. That is a curved
       trajectory being crossed with a single straight step, and it is the failure the rest of
       the page is about.`
    : (st.arm === 'reflow'
      ? `This arm was retrained on ${M.pairs.toLocaleString('en-US')} pairs the first one
         produced, for the same ${M.base_steps.toLocaleString('en-US')} gradient steps the
         first one got, so anything it gains here it gains from the method and not from more
         compute.`
      : `The student has no budget to drag. It was trained to name the endpoint of a teacher
         trajectory from anywhere along it, so it is asked once and the answer is the sample.`)}
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
