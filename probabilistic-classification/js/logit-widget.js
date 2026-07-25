/* One coefficient, two views of the same model.
 *
 * The point of the toggle is that the curved view and the straight view are the
 * same object: a line in log-odds space IS the sigmoid in probability space.
 * Seeing the same slider bend one and tilt the other is the whole argument of
 * the section it sits under.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { sigmoid } from './mathkit.js';

export function initLogitWidget() {
  const { g, w, h } = makeChart('#logit-chart', {
    width: 620,
    height: 420,
    margin: { top: 30, right: 26, bottom: 56, left: 66 },
  });

  const x = d3.scaleLinear().domain([-3.2, 3.2]).range([0, w]);
  const yP = d3.scaleLinear().domain([-0.05, 1.05]).range([h, 0]);
  /* The log-odds axis has to hold whatever the two sliders can reach, which is
   * b1 = 4 at x = 3, so 12 and not 8. At 8 the reading marker climbed straight
   * out of the chart at the top of both sliders. */
  const yL = d3.scaleLinear().domain([-13.5, 13.5]).range([h, 0]);

  const clip = 'logit-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -2).attr('y', -2)
    .attr('width', w + 4).attr('height', h + 4);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  const grid = d3.range(-3.2, 3.21, 0.02);
  const halo = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7);
  const curve = plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);
  const zero = plot.append('line').attr('x1', 0).attr('x2', w)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.3).attr('stroke-dasharray', '6 5').attr('opacity', 0.55);

  const marker = plot.append('circle').attr('r', 7)
    .attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 2);
  const drop = plot.append('line')
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 4').attr('opacity', 0.8);

  const state = { b1: 1, xq: 0, view: 'p' };
  const btnP = document.querySelector('#logit-view-p');
  const btnL = document.querySelector('#logit-view-l');
  const b1El = document.querySelector('#logit-b1');
  const xEl = document.querySelector('#logit-x');
  const statsEl = document.querySelector('#logit-stats');

  /* Guard both ends. Odds run from about 1e-6 to 1e6 across these two sliders,
   * and a plain toFixed(3) prints "0.000" at the small end, which is the exact
   * failure this article spends a section complaining about. */
  const fmt = (v) =>
    (v !== 0 && (Math.abs(v) >= 1000 || Math.abs(v) < 0.001) ? d3.format('.2e')(v) : v.toFixed(3));

  function render() {
    const { b1, xq, view } = state;
    document.querySelector('#logit-b1-label').textContent = b1.toFixed(2);
    document.querySelector('#logit-x-label').textContent = xq.toFixed(1);

    const y = view === 'p' ? yP : yL;
    /* Both axes are redrawn on every toggle, and the grid with them: a grid
     * left over from the other scale would rule the canvas at heights the
     * printed labels disagree with. */
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: view === 'p' ? d3.format('.1f') : d3.format('.0f') });
    axisLabels(g, w, h, {
      x: 'feature, in standard deviations',
      y: view === 'p' ? 'probability of malignancy' : 'log odds',
    });

    const f = (v) => (view === 'p' ? sigmoid(b1 * v) : b1 * v);
    const line = d3.line().x((d) => x(d)).y((d) => y(f(d)));
    const d = line(grid);
    halo.attr('d', d);
    curve.attr('d', d);
    zero.attr('y1', y(view === 'p' ? 0.5 : 0)).attr('y2', y(view === 'p' ? 0.5 : 0));

    marker.attr('cx', x(xq)).attr('cy', y(f(xq)));
    drop.attr('x1', x(xq)).attr('x2', x(xq)).attr('y1', y(f(xq))).attr('y2', h);

    btnP.classList.toggle('ghost', view !== 'p');
    btnL.classList.toggle('ghost', view === 'p');

    const p = sigmoid(b1 * xq);
    const odds = p / (1 - p);
    const pNext = sigmoid(b1 * (xq + 1));
    const oddsNext = pNext / (1 - pNext);
    statsEl.innerHTML =
      `<table class="pc-table">
         <tr><th></th><th>at x = ${xq.toFixed(1)}</th><th>at x = ${(xq + 1).toFixed(1)}</th></tr>
         <tr><td>probability</td><td>${fmt(p)}</td><td>${fmt(pNext)}</td></tr>
         <tr><td>odds</td><td>${fmt(odds)}</td><td>${fmt(oddsNext)}</td></tr>
         <tr><td>log odds</td><td>${(b1 * xq).toFixed(3)}</td><td>${(b1 * (xq + 1)).toFixed(3)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.7rem;line-height:1.45">` +
      `One more standard deviation multiplies the odds by ` +
      `<span class="value">${Math.exp(b1).toFixed(2)}</span>, and it does that at every ` +
      `point on the curve. The probability gain from the same step is ` +
      `<span class="value">${((pNext - p) * 100).toFixed(1)}</span> points here, and ` +
      `would be something else somewhere else. That is why the coefficient is quoted ` +
      `in odds.</div>`;
  }

  b1El.addEventListener('input', (e) => {
    state.b1 = +e.target.value;
    render();
  });
  xEl.addEventListener('input', (e) => {
    state.xq = +e.target.value;
    render();
  });
  btnP.addEventListener('click', () => {
    state.view = 'p';
    render();
  });
  btnL.addEventListener('click', () => {
    state.view = 'l';
    render();
  });
  render();
}
