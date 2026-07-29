/* The multiplications, counted, with no clock anywhere near it.
 *
 * A wall clock measures the machine and the contention on it. What decides
 * this architecture is arithmetic that can be counted exactly: scoring n items
 * with a frozen item tower is n dot products, and scoring them with a network
 * that sees the pair is n forward passes, so the ratio is fixed by the shapes
 * and is the same on every machine anybody will ever run it on.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initCostWidget(data) {
  const C = data.cost;
  const readout = document.querySelector('#cost-readout');

  const chart = makeChart('#cost-chart', {
    width: 640, height: 360, margin: { top: 44, right: 118, bottom: 58, left: 82 }, clip: true,
  });
  const { g, plot, w, h } = chart;

  const rows = C.rows;
  const x = d3.scaleLog().domain([rows[0].n * 0.7, rows[rows.length - 1].n * 1.4]).range([0, w]);
  const vals = rows.flatMap((d) => [d.tower, d.cross]);
  const y = d3.scaleLog().domain([d3.min(vals) * 0.6, d3.max(vals) * 1.6]).range([h, 0]);
  const xTicks = rows.map((d) => d.n);
  const yTicks = [1e3, 1e5, 1e7, 1e9, 1e11].filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
  drawGrid(g, x, y, w, h, { xValues: xTicks, yValues: yTicks });

  [['tower', 'two towers', 'var(--primary)'], ['cross', 'sees the pair', 'var(--anchor)']]
    .forEach(([key, label, colour]) => {
      plot.append('path').attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.3)
        .attr('d', d3.line().x((d) => x(d.n)).y((d) => y(d[key]))(rows));
      plot.selectAll(null).data(rows).enter().append('circle')
        .attr('cx', (d) => x(d.n)).attr('cy', (d) => y(d[key])).attr('r', 3).attr('fill', colour);
      g.append('text').attr('class', 'chart-note')
        .attr('x', w + 6).attr('y', y(rows[rows.length - 1][key]) + 3.5)
        .style('font-size', '11px').style('fill', colour).text(label);
    });

  drawAxes(g, x, y, w, h, {
    xValues: xTicks, yValues: yTicks, xFmt: d3.format('~s'), yFmt: d3.format('~s'),
  });
  axisLabels(g, w, h, { x: 'items to score', y: 'multiplications per query' });
  g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px')
    .text(`${C.k} numbers a side against a ${C.hidden} wide network over ${C.feats} features`);

  const here = rows.find((d) => d.n >= C.catalogue) || rows[rows.length - 1];
  const big = rows[rows.length - 1];
  readout.innerHTML =
    `At this catalogue's size, <span class="value">${C.catalogue.toLocaleString('en-US')}</span> `
    + `books, the two tower query costs <span class="value">${here.tower.toLocaleString('en-US')}</span> `
    + `multiplications and running a network on every pair costs `
    + `<span class="value">${here.cross.toLocaleString('en-US')}</span>, a factor of `
    + `<span class="value">${here.ratio.toFixed(0)}</span>. At a million items the factor is `
    + `<span class="value">${big.ratio.toFixed(0)}</span>. And the count understates it, because `
    + `the two tower numbers are the ones you would pay if you scored every item: the item vectors `
    + `do not depend on the query, so they are computed once and put in an index that answers a `
    + `nearest neighbour question without touching most of them. The other model has nothing to `
    + `precompute, because its input is the pair.`;

  return { rows };
}
