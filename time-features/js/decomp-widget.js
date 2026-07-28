/* A decomposition, drawn against the answer it was supposed to find.
 *
 * On real data this picture cannot be drawn: nobody ever observed the trend of
 * the carbon dioxide record separately from its season, so any decomposition of
 * it is a definition rather than a measurement. On a series whose parts were
 * written down before it was generated, the error is a number.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initDecompWidget(data) {
  const D = data.decomposition;
  const E = D.example;
  const chart = makeChart('#decomp-chart', {
    width: 640, height: 440, margin: { top: 42, right: 118, bottom: 46, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const n = E.y.length;
  const topH = Math.round(h * 0.5);
  const botY = topH + 58;
  const botH = h - botY;
  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const x = d3.scaleLinear().domain([0, n - 1]).range([0, w]);

  const yTop = d3.scaleLinear()
    .domain([d3.min(E.y) - 1, d3.max(E.y) + 1]).range([topH, 0]);
  drawGrid(gTop, x, yTop, w, topH, { xValues: [0, 60, 120, 179], yTicks: 4 });
  drawAxes(gTop, x, yTop, w, topH, { xValues: [0, 60, 120, 179], yTicks: 4 });
  axisLabels(gTop, w, topH, { y: 'the series', leftBudget: margin.left });
  gTop.append('path').datum(E.y).attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.9).attr('opacity', 0.5)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yTop(d)));
  gTop.append('path').datum(E.trend).attr('fill', 'none').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 2.4)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yTop(d)));
  gTop.append('path').datum(E.trend_hat).attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2).attr('stroke-dasharray', '6 4')
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yTop(d)));
  gTop.append('text').attr('class', 'chart-note').attr('x', w + 6)
    .attr('y', yTop(E.trend[n - 1]) + 4).style('font-size', '11px')
    .attr('fill', 'var(--anchor)').text('the real one');
  gTop.append('text').attr('class', 'chart-note').attr('x', w + 6)
    .attr('y', yTop(E.trend_hat[n - 1]) + 18).style('font-size', '11px')
    .attr('fill', 'var(--primary)').text('recovered');

  const seasonAll = E.season.concat(E.season_hat);
  const yBot = d3.scaleLinear()
    .domain([d3.min(seasonAll) * 1.15, d3.max(seasonAll) * 1.15]).range([botH, 0]);
  drawGrid(gBot, x, yBot, w, botH, { xValues: [0, 60, 120, 179], yTicks: 3 });
  drawAxes(gBot, x, yBot, w, botH, { xValues: [0, 60, 120, 179], yTicks: 3 });
  axisLabels(gBot, w, botH, {
    x: 'month', y: 'the season', leftBudget: margin.left, xOffset: 36,
  });
  gBot.append('path').datum(E.season).attr('fill', 'none').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 1.8)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yBot(d)));
  gBot.append('path').datum(E.season_hat).attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4')
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yBot(d)));

  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px');
  wrapLabel(title, `one series of the ${D.n_series} written down ones, family `
    + `"${E.family.replace('_', ' ')}"`, w - 8);

  const readout = document.querySelector('#decomp-readout');
  readout.innerHTML = `
    <div class="gen-note">
      Both panels are the same fit. The trend it recovered sits on top of the
      real one because this family is exactly what the model is: a slope that
      changes at a few places, plus a Fourier season. This is the easy case, and
      it is the only kind of picture anybody ever puts in a slide. The table
      below runs the same comparison over ${D.n_series} series from four
      families, and the spread across it is the whole story.
    </div>`;

  return { render: () => {} };
}
