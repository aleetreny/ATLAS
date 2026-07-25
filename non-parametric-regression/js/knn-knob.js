/* The K knob: the bias-variance dial made literal, plus the uniform against
 * distance-weighted comparison, scored on data the model never saw. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { knnPredict, rmse } from './mathkit.js';

export function initKnnKnob(mpg) {
  const pts = mpg.points;
  const tr = mpg.train_idx;
  const te = mpg.test_idx;
  const xtr = tr.map((i) => pts[i].x);
  const ytr = tr.map((i) => pts[i].y);
  const xte = te.map((i) => pts[i].x);
  const yte = te.map((i) => pts[i].y);
  const grid = mpg.grid;

  const { g, w, h } = makeChart('#knn-chart', {
    width: 620,
    height: 460,
    margin: { top: 30, right: 24, bottom: 58, left: 62 },
  });
  const x = d3.scaleLinear().domain([35, 235]).range([0, w]);
  const y = d3.scaleLinear().domain([5, 50]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });

  g.append('g').selectAll('circle.tr').data(tr.map((i) => pts[i])).join('circle')
    .attr('class', 'tr').attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y))
    .attr('r', 3.6).attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.7);
  g.append('g').selectAll('circle.te').data(te.map((i) => pts[i])).join('circle')
    .attr('class', 'te').attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y))
    .attr('r', 4).attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 1.8);

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveLinear);
  const haloU = g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const pathU = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);
  const haloD = g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const pathD = g.append('path').attr('fill', 'none').attr('stroke', 'var(--smile)').attr('stroke-width', 3).attr('stroke-dasharray', '7 5');

  const slider = document.querySelector('#knn-k');
  const label = document.querySelector('#knn-k-label');
  const stats = document.querySelector('#knn-stats');

  function render() {
    const k = +slider.value;
    label.textContent = k;
    const cu = knnPredict(xtr, ytr, grid, k, 'uniform');
    const cd = knnPredict(xtr, ytr, grid, k, 'distance');
    haloU.attr('d', line(grid.map((v, i) => [v, cu[i]])));
    pathU.attr('d', haloU.attr('d'));
    haloD.attr('d', line(grid.map((v, i) => [v, cd[i]])));
    pathD.attr('d', haloD.attr('d'));

    const eu = rmse(yte, knnPredict(xtr, ytr, xte, k, 'uniform'));
    const ed = rmse(yte, knnPredict(xtr, ytr, xte, k, 'distance'));
    const trU = rmse(ytr, knnPredict(xtr, ytr, xtr, k, 'uniform'));

    // Runtime check against the scikit-learn reference. Tolerance is loose on
    // purpose: horsepower is recorded as whole numbers so many cars sit at
    // exactly the same distance from a query, and which of those tied cars ends
    // up inside the k is an implementation detail rather than a fact about the
    // method. Differences of a percent or so are ties resolving differently.
    const ref = mpg.ref.knn[`${k}_uniform`];
    if (ref && Math.abs(eu - ref.rmse_test) / ref.rmse_test > 0.03) {
      console.warn(`knn k=${k}: JS test rmse ${eu.toFixed(3)} vs sklearn ${ref.rmse_test}`);
    }

    const verdict =
      k === 1
        ? 'Memorised: it reproduces every training car exactly and generalises worst.'
        : k <= 4
          ? 'Still twitchy. Each prediction rests on very few opinions.'
          : k <= 30
            ? 'The useful range: enough neighbours to average away noise, few enough to stay local.'
            : k <= 90
              ? 'Over-smoothed. The curve is flattening towards the global mean.'
              : 'This is barely a model any more. Almost every car gets the same answer.';

    stats.innerHTML =
      `<table class="np-table">
        <tr><th></th><th style="color:var(--primary)">uniform</th><th style="color:var(--smile)">1/distance</th></tr>
        <tr><td>held-out RMSE</td><td>${eu.toFixed(3)}</td><td>${ed.toFixed(3)}</td></tr>
        <tr><td>training RMSE</td><td>${trU.toFixed(3)}</td><td>${k === 1 ? '0.000' : rmse(ytr, knnPredict(xtr, ytr, xtr, k, 'distance')).toFixed(3)}</td></tr>
       </table>
       <div class="slider-label" style="margin-top:.7rem;line-height:1.45">${verdict}</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
