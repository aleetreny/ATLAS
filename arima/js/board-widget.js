/* Error against horizon, which is the shape a single average hides.
 *
 * The naive forecast is the reason this is a chart and not a column: its error
 * is not monotone in the horizon at all. Six months out it is at its worst,
 * because the cycle is half a period away; twelve months out it is nearly as
 * good as one month out, because the cycle has come back round. An average
 * over the twelve horizons turns that into one number and loses the argument.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, spread } from '../../assets/js/chart.js';

export function initBoardWidget(data) {
  const rows = data.rolling.rows;
  const chart = makeChart('#board-chart', {
    width: 640, height: 380, margin: { top: 44, right: 148, bottom: 48, left: 62 },
  });
  const { g, w, h } = chart;
  const x = d3.scaleLinear().domain([1, 12]).range([0, w]);
  const hi = d3.max(rows, (r) => d3.max(r.mase));
  const y = d3.scaleLinear().domain([0, hi * 1.05]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: [1, 3, 6, 9, 12], yTicks: 5 });
  drawAxes(g, x, y, w, h, { xValues: [1, 3, 6, 9, 12], yTicks: 5 });
  axisLabels(g, w, h, { x: 'horizon, in months', y: 'error, seasonal naive units' });
  g.select('text.axis-label.y').attr('y', -64);

  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(1)).attr('y2', y(1))
    .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '5 4').attr('opacity', 0.6);
  g.append('text').attr('class', 'chart-note').attr('x', 3).attr('y', y(1) - 6)
    .style('font-size', '11px').text('one: what last year already gave you');

  const colours = {
    SARIMA: 'var(--primary)', ETS: 'var(--anchor)',
    'seasonal naive': 'var(--aqua)', drift: 'var(--galaxy)', naive: 'var(--cosmos)',
  };
  const line = d3.line().x((d, i) => x(i + 1)).y((d) => y(d));
  rows.forEach((row) => {
    g.append('path').datum(row.mase).attr('fill', 'none')
      .attr('stroke', colours[row.model] || 'var(--squidink)')
      .attr('stroke-width', row.model === 'SARIMA' ? 2.8 : 1.8)
      .attr('d', line);
  });
  const labels = spread(rows.map((row) => ({
    text: row.model, y: y(row.mase[11]), colour: colours[row.model] || 'var(--squidink)',
  })), 15);
  g.selectAll('text.series-label').data(labels).join('text')
    .attr('class', 'chart-note series-label')
    .attr('x', w + 6).attr('y', (d) => d.y + 4)
    .style('font-size', '11.5px').attr('fill', (d) => d.colour).text((d) => d.text);

  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');
  wrapLabel(title, `${data.rolling.n_origins} origins, each one fitted from scratch on `
    + `everything above it`, w - 8);
  return { render: () => {} };
}
