/* The epsilon-tube, told in five steps on the banana the reader has already
 * met twice (articles 1 and 4). The straight line is fitted live; every SVR
 * state comes from the measured grid, so what is on screen is exactly what
 * scikit-learn produced.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { ols1d } from './svrkit.js';

export function initScrollyTube(data) {
  const xr = data.x_train.map((z) => z * data.x_sd + data.x_mu);   // raw horsepower
  const yr = data.y_train;
  const grid = data.grid;

  const { g, w, h } = makeChart('#tube-chart', {
    width: 620,
    height: 560,
    margin: { top: 34, right: 26, bottom: 56, left: 56 },
  });
  const x = d3.scaleLinear().domain(d3.extent(grid)).nice().range([0, w]);
  const y = d3.scaleLinear().domain([5, 50]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'miles per gallon' });

  const tubeG = g.append('g');
  const lineG = g.append('g');
  const dotG = g.append('g');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const olsCurve = ols1d(xr, yr)(grid);
  const at = (eps, C) => data.svr_grid.find((r) => r.eps === eps && r.C === C);
  const line = d3.line().x((v, i) => x(grid[i])).y((v) => y(v));
  const area = (curve, eps) => d3.area()
    .x((v, i) => x(grid[i])).y0((v) => y(v - eps)).y1((v) => y(v + eps))(curve);

  function draw(curve, eps, svSet) {
    tubeG.selectAll('path').data(eps ? [1] : []).join('path')
      .attr('fill', 'var(--primary)').attr('opacity', 0.13)
      .attr('d', area(curve, eps));
    lineG.selectAll('path').data([1]).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3)
      .attr('d', line(curve));
    dotG.selectAll('circle').data(xr).join('circle')
      .attr('cx', (v) => x(v)).attr('cy', (v, i) => y(yr[i]))
      .attr('r', (v, i) => (svSet && svSet.has(i) ? 4.2 : 3))
      .attr('fill', (v, i) => (svSet ? (svSet.has(i) ? 'var(--cosmos)' : 'var(--stone)') : 'var(--stone)'))
      .attr('stroke', (v, i) => (svSet && svSet.has(i) ? 'var(--squidink)' : 'var(--squidink)'))
      .attr('stroke-width', (v, i) => (svSet && svSet.has(i) ? 1.2 : 0.5))
      .attr('opacity', (v, i) => (svSet && !svSet.has(i) ? 0.45 : 0.85));
  }

  const states = {
    0: () => {
      draw(olsCurve, null, null);
      caption.text('the banana, third appearance: a straight line still is not it');
    },
    1: () => {
      const r = at(2.0, 10.0);
      draw(r.curve, 2.0, null);
      caption.text('the tube: within 2 mpg of the curve, an error costs exactly nothing');
    },
    2: () => {
      const r = at(2.0, 10.0);
      draw(r.curve, 2.0, new Set(r.sv));
      caption.text(`only ${r.n_sv} cars touch or leave the tube: they alone hold the curve up`);
    },
    3: () => {
      const r = at(0.5, 10.0);
      draw(r.curve, 0.5, new Set(r.sv));
      caption.text(`squeeze to 0.5 mpg and ${r.n_sv} of 294 are back on the payroll`);
    },
    4: () => {
      const r = at(3.0, 10.0);
      draw(r.curve, 3.0, new Set(r.sv));
      caption.text(`eps 3, C 10: ${r.n_sv} support vectors and the best test RMSE of the sweep, ${r.rmse_test}`);
    },
  };

  scrolly('#tube-scrolly .step', states);
}
