/* Early stopping, on the real diamonds data: the training curve falling forever
 * while the held-out curve turns around, and the tree count where you should
 * have stopped. */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

export function initStoppingWidget(dia) {
  const curve = dia.curve;
  const n = curve.valid.length;
  const best = dia.best_iter;

  const { g, w, h } = makeChart('#stopping-chart', {
    width: 660, height: 360, margin: { top: 30, right: 116, bottom: 58, left: 68 },
  });
  const x = d3.scaleLinear().domain([1, n]).range([0, w]);
  const y = d3.scaleLog()
    .domain([Math.min(...curve.train) * 0.85, Math.max(...curve.valid, ...curve.train) * 1.05])
    .range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h, { yTicks: 4, yFmt: d3.format('~s') });
  axisLabels(g, w, h, { x: 'trees in the ensemble', y: 'RMSE ($)' });

  const bestBand = g.append('rect')
    .attr('x', x(Math.max(1, best - 2))).attr('width', Math.max(3, x(best + 2) - x(best - 2)))
    .attr('y', 0).attr('height', h).attr('fill', 'var(--lab)').attr('opacity', 0.25);

  const line = d3.line().x((d, i) => x(i + 1)).y((d) => y(Math.max(d, y.domain()[0])));
  for (const [key, color, dash] of [['train', 'var(--squidink)', '6 5'], ['valid', 'var(--primary)', null]]) {
    g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6).attr('d', line(curve[key]));
    g.append('path').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 3.2)
      .attr('stroke-dasharray', dash).attr('d', line(curve[key]));
  }

  const labels = spread([
    { t: 'training', y: y(curve.train[n - 1]), c: 'var(--squidink)' },
    { t: 'held out', y: y(curve.valid[n - 1]), c: 'var(--primary)' },
  ], 15);
  g.selectAll('text.series').data(labels).join('text')
    .attr('class', 'series chart-annotation')
    .attr('x', w + 10).attr('y', (d) => d.y + 4)
    .style('font-size', '0.64rem').style('letter-spacing', '0.4px').style('text-transform', 'none')
    .attr('fill', (d) => d.c).text((d) => d.t);

  /* Chrome, not data: --cosmos is this article's --primary, which is already the
   * validation curve, so a marker wearing it read as a third series. */
  const marker = g.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.8);
  const dotT = g.append('circle').attr('r', 5).attr('fill', 'var(--squidink)').attr('stroke', 'white').attr('stroke-width', 1.5);
  const dotV = g.append('circle').attr('r', 5).attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.5);

  g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(best)).attr('y', -10).attr('text-anchor', 'middle')
    .style('font-size', '0.62rem').text(`best at ${best}`);

  const slider = document.querySelector('#stop-n');
  const label = document.querySelector('#stop-n-label');
  const stats = document.querySelector('#stopping-stats');
  slider.max = n;

  function render() {
    const m = +slider.value;
    label.textContent = m;
    marker.attr('x1', x(m)).attr('x2', x(m));
    dotT.attr('cx', x(m)).attr('cy', y(curve.train[m - 1]));
    dotV.attr('cx', x(m)).attr('cy', y(curve.valid[m - 1]));

    const gap = curve.valid[m - 1] - curve.train[m - 1];
    const lost = curve.valid[m - 1] - curve.valid[best - 1];
    stats.innerHTML =
      `<table class="gb-table">
        <tr><th>at ${m} trees</th><th>RMSE ($)</th></tr>
        <tr><td>training</td><td>${curve.train[m - 1].toFixed(0)}</td></tr>
        <tr><td>held out</td><td>${curve.valid[m - 1].toFixed(0)}</td></tr>
        <tr><td>gap</td><td>${gap.toFixed(0)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">${
        m < best
          ? `Still improving. The held-out error bottoms out at tree ${best}.`
          : lost < 5
            ? 'This is the stopping point. Every tree after this one costs money and buys nothing.'
            : `Past the optimum: ${lost.toFixed(0)} dollars of held-out error given back, while the training error keeps falling. The model is memorising stones.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  slider.value = best;
  render();
}
