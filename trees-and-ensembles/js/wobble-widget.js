/* Eight bootstrap samples, eight different models.
 *
 * The number that matters is the spread of held-out accuracy across the eight,
 * so it is printed as a range and each sample's own score is marked on it. The
 * reader should be able to see that this is one dataset and eight answers.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { drawBoxes, drawPoints } from './boxkit.js';

export function initWobbleWidget(data) {
  const D = data.two_d;
  const runs = D.wobble;

  const { g, w, h } = makeChart('#wobble-chart', {
    width: 660,
    height: 470,
    margin: { top: 34, right: 26, bottom: 58, left: 68 },
  });

  const x = d3.scaleLinear().domain(D.bounds[0]).range([0, w]);
  const y = d3.scaleLinear().domain(D.bounds[1]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, {
    x: D.features[0].replace(/_/g, ' '),
    y: D.features[1].replace(/_/g, ' '),
  });

  const clip = 'wob-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', 0).attr('y', 0)
    .attr('width', w).attr('height', h);
  const boxG = g.append('g').attr('clip-path', `url(#${clip})`);
  drawPoints(g.append('g'), D.train, x, y, { r: 2.8 });

  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const readout = document.querySelector('#wobble-readout');
  const playBtn = document.querySelector('#wobble-play');
  const state = { i: 0 };
  let timer = null;

  const lo = d3.min(runs, (r) => r.test);
  const hi = d3.max(runs, (r) => r.test);

  function render() {
    const r = runs[state.i];
    drawBoxes(boxG, r.boxes, x, y, { duration: 380 });
    title.text(`bootstrap sample ${state.i + 1} of ${runs.length}: ${r.boxes.length} boxes`);

    /* A strip of every sample's held-out score, with the current one marked, so
     * the spread is a picture and not a sentence. */
    const dots = runs
      .map((s, k) => {
        const pos = ((s.test - lo) / Math.max(hi - lo, 1e-9)) * 100;
        return `<span style="position:absolute;left:${pos}%;top:0;transform:translateX(-50%);` +
          `width:9px;height:9px;border-radius:50%;background:${k === state.i ? 'var(--cosmos)' : 'var(--stone)'};` +
          `border:1px solid var(--squidink)"></span>`;
      })
      .join('');
    readout.innerHTML =
      `<div class="slider-label">held-out accuracy of this tree: ` +
      `<span class="value">${r.test.toFixed(4)}</span></div>` +
      `<div style="position:relative;height:11px;margin:.7rem 6px .3rem 6px">${dots}</div>` +
      `<div class="slider-label" style="display:flex;justify-content:space-between;font-size:12px;opacity:.75">` +
      `<span>${lo.toFixed(4)}</span><span>${hi.toFixed(4)}</span></div>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">` +
      `Eight resamples of the same 398 patients, eight different trees, and a spread of ` +
      `<span class="value">${((hi - lo) * 100).toFixed(1)}</span> accuracy points between the best and the ` +
      `worst. Nothing about the underlying problem changed between these eight pictures.</div>`;
  }

  const step = (d) => {
    state.i = (state.i + d + runs.length) % runs.length;
    render();
  };
  const stop = () => {
    if (timer) timer.stop();
    timer = null;
    playBtn.textContent = 'Cycle through all eight';
  };

  document.querySelector('#wobble-next').addEventListener('click', () => {
    stop();
    step(1);
  });
  document.querySelector('#wobble-prev').addEventListener('click', () => {
    stop();
    step(-1);
  });
  playBtn.addEventListener('click', () => {
    if (timer) {
      stop();
      return;
    }
    playBtn.textContent = 'Stop';
    let last = 0;
    timer = d3.timer((elapsed) => {
      if (elapsed - last < 900) return;
      last = elapsed;
      step(1);
    });
  });

  render();
}
