/* The tube under two dials, next to the payroll it implies.
 *
 * Left panel: the measured SVR fit at the chosen (epsilon, C), tube shaded,
 * support vectors marked. Right panel: the whole sparsity story at once,
 * support-vector count and test RMSE against epsilon, with the current
 * position marked. Every state is a real scikit-learn fit from the shipped
 * grid; the page only replays.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

export function initTubeWidget(data) {
  const xr = data.x_train.map((z) => z * data.x_sd + data.x_mu);
  const yr = data.y_train;
  const grid = data.grid;
  const EPS = data.eps_values;
  const CS = data.c_values;

  const wrap = d3.select('#dial-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  const L = makeChart(left, { width: 360, height: 360, margin: { top: 26, right: 14, bottom: 46, left: 46 } });
  const lx = d3.scaleLinear().domain(d3.extent(grid)).nice().range([0, L.w]);
  const ly = d3.scaleLinear().domain([5, 50]).range([L.h, 0]);
  drawGrid(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 5 });
  drawAxes(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 5 });
  axisLabels(L.g, L.w, L.h, { x: 'horsepower' });
  const tubeG = L.g.append('g');
  const dotG = L.g.append('g');
  const lineG = L.g.append('g');

  const R = makeChart(right, { width: 360, height: 360, margin: { top: 26, right: 52, bottom: 46, left: 50 } });
  const rx = d3.scalePoint().domain(EPS).range([0, R.w]).padding(0.3);
  const ry = d3.scaleLinear().domain([0, 300]).range([R.h, 0]);
  const ry2 = d3.scaleLinear().domain([4.6, 5.4]).range([R.h, 0]);
  drawAxes(R.g, rx, ry, R.w, R.h, { yTicks: 5 });
  axisLabels(R.g, R.w, R.h, { x: 'epsilon (mpg)' });

  const readout = document.querySelector('#dial-readout');
  const epsSlider = document.querySelector('#dial-eps');
  const cSlider = document.querySelector('#dial-c');
  const line = d3.line().x((v, i) => lx(grid[i])).y((v) => ly(v));
  const area = (curve, eps) => d3.area()
    .x((v, i) => lx(grid[i])).y0((v) => ly(v - eps)).y1((v) => ly(v + eps))(curve);

  function seriesFor(C) {
    return EPS.map((e) => data.svr_grid.find((r) => r.eps === e && r.C === C));
  }

  let curveSel = null;
  let marker = null;

  function render() {
    const eps = EPS[+epsSlider.value];
    const C = CS[+cSlider.value];
    document.querySelector('#dial-eps-label').textContent = eps;
    document.querySelector('#dial-c-label').textContent = C;
    const r = data.svr_grid.find((q) => q.eps === eps && q.C === C);
    const svSet = new Set(r.sv);

    tubeG.selectAll('path').data([1]).join('path')
      .attr('fill', 'var(--primary)').attr('opacity', 0.13).attr('d', area(r.curve, eps));
    lineG.selectAll('path').data([1]).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('d', line(r.curve));
    dotG.selectAll('circle').data(xr).join('circle')
      .attr('cx', (v) => lx(v)).attr('cy', (v, i) => ly(yr[i]))
      .attr('r', (v, i) => (svSet.has(i) ? 3.6 : 2.4))
      .attr('fill', (v, i) => (svSet.has(i) ? 'var(--cosmos)' : 'var(--stone)'))
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', (v, i) => (svSet.has(i) ? 1 : 0.4))
      .attr('opacity', (v, i) => (svSet.has(i) ? 0.9 : 0.45));

    /* Right: the two curves for the current C. */
    const series = seriesFor(C);
    if (!curveSel) {
      curveSel = {
        sv: R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 3),
        rm: R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2)
          .attr('stroke-dasharray', '5 4').attr('opacity', 0.7),
        labSv: R.g.append('text').attr('class', 'chart-annotation').style('font-size', '0.7rem')
          .style('text-transform', 'none').attr('fill', 'var(--cosmos)').attr('text-anchor', 'end').text('support vectors'),
        labRm: R.g.append('text').attr('class', 'chart-annotation').style('font-size', '0.7rem')
          .style('text-transform', 'none').attr('fill', 'var(--squidink)').attr('text-anchor', 'end').text('test RMSE'),
      };
      marker = R.g.append('circle').attr('r', 5).attr('fill', 'var(--cosmos)')
        .attr('stroke', 'white').attr('stroke-width', 1.6);
    }
    curveSel.sv.attr('d', d3.line().x((s) => rx(s.eps)).y((s) => ry(s.n_sv))(series));
    curveSel.rm.attr('d', d3.line().x((s) => rx(s.eps)).y((s) => ry2(Math.min(s.rmse_test, 5.4)))(series));
    curveSel.labSv.attr('x', rx(EPS[EPS.length - 1])).attr('y', ry(series[series.length - 1].n_sv) - 8);
    curveSel.labRm.attr('x', rx(EPS[EPS.length - 1])).attr('y', ry2(Math.min(series[series.length - 1].rmse_test, 5.4)) - 8);
    marker.attr('cx', rx(eps)).attr('cy', ry(r.n_sv));

    const best = data.svr_grid.reduce((a, b) => (b.rmse_test < a.rmse_test ? b : a));
    readout.innerHTML =
      `<table class="svr-table">
         <tr><td>support vectors</td><td><span class="value">${r.n_sv}</span> of 294</td>
             <td>test RMSE</td><td><span class="value">${r.rmse_test.toFixed(3)}</span> mpg</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        r.eps === best.eps && r.C === best.C
          ? `The sweep's best: ${best.rmse_test} mpg of test error from ${best.n_sv} support vectors, ` +
            `less than half the payroll, and better generalisation than the tightest tube. Ignoring ` +
            `small errors is not laziness, it is regularisation.`
          : C === 0.1
            ? `C is the price of leaving the tube, and at 0.1 leaving is nearly free: the curve barely ` +
              `bothers to bend. Slack everywhere, structure nowhere.`
            : r.n_sv > 220
              ? `A tube this tight puts ${r.n_sv} of 294 cars on the payroll: nearly every point is an ` +
                `active constraint, the sparsity is gone, and the test error is no better for it.`
              : r.n_sv < 50
                ? `${r.n_sv} support vectors is a beautifully small committee, and the test error is ` +
                  `paying for the parsimony: the tube is now wider than real structure in the data.`
                : `${r.n_sv} support vectors hold the curve; the other ${294 - r.n_sv} cars sit inside ` +
                  `the tube and could be deleted without moving the fit at all. That deletability is ` +
                  `SVR's product.`
      }</div>`;
  }

  epsSlider.addEventListener('input', render);
  cSlider.addEventListener('input', render);
  render();
}
