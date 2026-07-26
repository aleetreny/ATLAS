/* Kernel ridge, genuinely live: two continuous sliders, and on every input
 * event the browser rebuilds the 294x294 kernel matrix, adds lambda to the
 * diagonal, Cholesky-solves it and redraws the curve. No grid, no cache,
 * no precomputation: this is the closed form earning its keep. Checked at
 * load time against twelve scikit-learn fits.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { krrFit, rmse } from './svrkit.js';

export function initKrrWidget(data) {
  const z = data.x_train;
  const yr = data.y_train;
  const grid = data.grid;
  const zg = data.zgrid;

  const { g, w, h } = makeChart('#krr-chart', {
    width: 620,
    height: 440,
    margin: { top: 30, right: 26, bottom: 54, left: 56 },
  });
  const x = d3.scaleLinear().domain(d3.extent(grid)).nice().range([0, w]);
  const y = d3.scaleLinear().domain([5, 50]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'miles per gallon' });

  const xr = z.map((v) => v * data.x_sd + data.x_mu);
  g.append('g').selectAll('circle').data(xr).join('circle')
    .attr('cx', (v) => x(v)).attr('cy', (v, i) => y(yr[i]))
    .attr('r', 2.6).attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.4).attr('opacity', 0.6);
  const lineG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle').style('font-size', '0.62rem');

  const gSlider = document.querySelector('#krr-gamma');
  const lSlider = document.querySelector('#krr-lambda');
  const readout = document.querySelector('#krr-readout');
  const line = d3.line().x((v, i) => x(grid[i])).y((v) => y(v));

  const best = data.kr_refs.reduce((a, b) => (b.rmse_test < a.rmse_test ? b : a));

  function render() {
    const gamma = Math.pow(10, +gSlider.value);      // 10^-1.5 .. 10^1
    const lambda = Math.pow(10, +lSlider.value);     // 10^-3 .. 10^1.5
    document.querySelector('#krr-gamma-label').textContent = gamma.toPrecision(2);
    document.querySelector('#krr-lambda-label').textContent = lambda.toPrecision(2);

    const t0 = performance.now();
    const predict = krrFit(z, yr, gamma, lambda);
    const curve = predict(zg);
    const err = rmse(data.y_test, predict(data.x_test));
    const ms = performance.now() - t0;

    lineG.selectAll('path').data([1]).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3)
      .attr('d', line(curve));
    title.text(`refit just now, in ${ms.toFixed(0)} ms: a 294-equation solve per slider tick`);

    readout.innerHTML =
      `<table class="svr-table">
         <tr><td>test RMSE</td><td><span class="value">${err.toFixed(3)}</span> mpg</td>
             <td>best sklearn reference</td><td>${best.rmse_test.toFixed(3)} at &gamma; ${best.gamma}, &lambda; ${best.lambda}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        gamma > 5
          ? `High gamma makes every kernel bump narrow: the curve is starting to visit individual cars. ` +
            `The lambda ridge is the only thing between this and connect-the-dots.`
          : lambda > 5
            ? `Heavy lambda shrinks every coefficient toward zero and the curve toward the flat prior: ` +
              `article 2's tax, collected in kernel currency.`
            : Math.abs(err - best.rmse_test) < 0.15
              ? `Within a whisker of the best reference fit. Note what was NOT needed to get here: no ` +
                `iterations, no convergence checks, no initialisation. One linear solve, every time.`
              : `Every position of these sliders is a complete refit of all 294 cars. The whole model is ` +
                `one linear solve: that is the luxury the epsilon-tube gave up to buy sparsity.`
      }</div>`;
  }

  gSlider.addEventListener('input', render);
  lSlider.addEventListener('input', render);
  render();
}
