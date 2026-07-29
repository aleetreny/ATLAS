/* How far down to let the gradient reach.
 *
 * Three arms at one sample size, and the horizontal axis is the count of
 * parameters allowed to move, on a log scale, because that is the quantity the
 * decision is really about. It is also the axis the next article spends its
 * whole length on.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initFreezeWidget(data) {
  const F = data.freeze;
  const readout = document.querySelector('#freeze-readout');

  const chart = makeChart('#freeze-chart', {
    width: 640, height: 320, margin: { top: 40, right: 40, bottom: 62, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const rows = F.rows;
  const x = d3.scaleLog()
    .domain([rows[0].trainable * 0.6, rows[rows.length - 1].trainable * 1.6]).range([0, w]);
  const vals = rows.flatMap((r) => [r.mean - r.sd, r.mean + r.sd]);
  const y = d3.scaleLinear().domain([d3.min(vals) - 0.01, d3.max(vals) + 0.01]).range([h, 0]);
  const ticks = rows.map((r) => r.trainable);
  drawGrid(g, x, y, w, h, { xValues: ticks, yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xValues: ticks, yTicks: 5, yFmt: d3.format('.3f'),
    xFmt: (v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v)),
  });
  axisLabels(g, w, h, { x: 'parameters allowed to move', y: 'accuracy' });

  layer.append('path')
    .attr('d', d3.line().x((r) => x(r.trainable)).y((r) => y(r.mean))(rows))
    .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
  rows.forEach((r, i) => {
    layer.append('line').attr('x1', x(r.trainable)).attr('x2', x(r.trainable))
      .attr('y1', y(r.mean - r.sd)).attr('y2', y(r.mean + r.sd))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4);
    layer.append('circle').attr('cx', x(r.trainable)).attr('cy', y(r.mean)).attr('r', 4.6)
      .attr('fill', 'var(--primary)');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(r.trainable)).attr('y', y(r.mean) - 14 + (i === rows.length - 1 ? 0 : 0))
      .attr('text-anchor', i === rows.length - 1 ? 'end' : 'middle')
      .style('font-size', '11px').text(r.arm);
  });

  const best = rows.reduce((a, b) => (b.mean > a.mean ? b : a));
  const head = rows[0];
  const all = rows[rows.length - 1];
  const tie = Math.abs(all.mean - head.mean) <= all.sd + head.sd;
  readout.innerHTML = `<div class="gen-note">At ${F.n} labelled rows, moving only the head `
    + `(${head.trainable.toLocaleString('en-US')} parameters) reaches `
    + `${head.mean.toFixed(4)} and moving everything `
    + `(${all.trainable.toLocaleString('en-US')}) reaches ${all.mean.toFixed(4)}, `
    + (tie
      ? `which the seed spreads cannot separate: ${(head.sd + all.sd).toFixed(4)} between `
        + `them and ${Math.abs(all.mean - head.mean).toFixed(4)} of difference. Thawing `
        + `${(all.trainable / head.trainable).toFixed(0)} times more parameters bought `
        + `nothing measurable here.`
      : `a difference of ${Math.abs(all.mean - head.mean).toFixed(4)} against a combined `
        + `seed spread of ${(head.sd + all.sd).toFixed(4)}, in favour of `
        + `${all.mean > head.mean ? 'thawing everything' : 'keeping it frozen'}.`)
    + ` The best of the three is <span class="bold">${best.arm}</span> at `
    + `${best.mean.toFixed(4)}. The whole network is `
    + `${F.total.toLocaleString('en-US')} parameters, which is the number the next article `
    + `spends its length on.</div>`;

  return { render: () => {} };
}
