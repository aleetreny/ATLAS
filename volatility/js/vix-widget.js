/* The model's forecast, the options market's forecast, and what happened.
 *
 * Almost every volatility tutorial stops at the likelihood, which scores the
 * model against itself. There is a second, completely independent forecast of
 * the same quantity published every day, and the honest comparison is a three
 * way one: what the model said, what the market said, and what the following
 * month actually delivered.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';

export function initVixWidget(data) {
  const V = data.vix;
  let view = 'path';
  const readout = document.querySelector('#vix-readout');
  const controls = d3.select('#vix-controls');
  const chart = makeChart('#vix-chart', {
    width: 640, height: 400, margin: { top: 46, right: 110, bottom: 50, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');

  function drawPath() {
    layer.selectAll('*').remove();
    const rows = V.path;
    const x = d3.scaleLinear().domain([0, rows.length - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([0, d3.max(rows, (d) => Math.max(d.g, d.v, d.r)) * 1.06])
      .range([h, 0]);
    const years = ['2014', '2015', '2016', '2017', '2018']
      .map((p) => rows.findIndex((d) => d.t.startsWith(p))).filter((i) => i >= 0);
    drawGrid(layer, x, y, w, h, { xValues: years, yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xValues: years, yTicks: 5, xFmt: (i) => rows[i].t.slice(0, 4) });
    axisLabels(layer, w, h, { y: 'annualised volatility, percent', leftBudget: margin.left });
    const series = [
      ['what happened', 'r', 'var(--squidink)', 1.4],
      ['the VIX', 'v', 'var(--anchor)', 1.8],
      ['GARCH', 'g', 'var(--primary)', 1.8],
    ];
    series.forEach(([label, key, colour, width]) => {
      layer.append('path').datum(rows).attr('fill', 'none').attr('stroke', colour)
        .attr('stroke-width', width).attr('opacity', key === 'r' ? 0.55 : 1)
        .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d[key])));
    });
    const last = rows[rows.length - 1];
    const labels = series.map(([label, key, colour]) => ({ label, y: y(last[key]), colour }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
    }
    layer.selectAll('text.lab').data(labels).join('text').attr('class', 'chart-note lab')
      .attr('x', w + 6).attr('y', (d) => d.y + 4).style('font-size', '11.5px')
      .attr('fill', (d) => d.colour).text((d) => d.label);
    wrapLabel(title, `${V.n.toLocaleString('en-US')} days, ${V.start} to ${V.end}: `
      + `two forecasts of the next ${V.horizon} trading days, and the answer`, w - 8);
  }

  function drawScatter() {
    layer.selectAll('*').remove();
    const pts = V.scatter;
    const x = d3.scaleLinear().domain([0, d3.max(pts, (d) => Math.max(d.g, d.v)) * 1.05])
      .range([0, w]);
    const y = d3.scaleLinear().domain([0, d3.max(pts, (d) => d.r) * 1.05]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(layer, w, h, {
      x: 'what was forecast', y: 'what happened', leftBudget: margin.left,
    });
    const dom = Math.min(x.domain()[1], y.domain()[1]);
    layer.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(dom)).attr('y2', y(dom))
      .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '5 4').attr('opacity', 0.6);
    /* the cloud sits below the diagonal, because both forecasts run above what
       happens, so the empty side is the upper left and the label goes there
       rather than at the end of the line where the title lives */
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(dom * 0.5) - 6).attr('y', y(dom * 0.5) - 9)
      .attr('text-anchor', 'end').style('font-size', '11px').text('forecast = outcome');
    layer.selectAll('circle.g').data(pts).join('circle').attr('class', 'g')
      .attr('cx', (d) => x(d.g)).attr('cy', (d) => y(d.r)).attr('r', 2.4)
      .attr('fill', 'var(--primary)').attr('opacity', 0.6);
    layer.selectAll('circle.v').data(pts).join('circle').attr('class', 'v')
      .attr('cx', (d) => x(d.v)).attr('cy', (d) => y(d.r)).attr('r', 2.4)
      .attr('fill', 'var(--anchor)').attr('opacity', 0.6);
    layer.append('text').attr('class', 'chart-note').attr('x', w + 6).attr('y', 16)
      .style('font-size', '11.5px').attr('fill', 'var(--primary)').text('GARCH');
    layer.append('text').attr('class', 'chart-note').attr('x', w + 6).attr('y', 34)
      .style('font-size', '11.5px').attr('fill', 'var(--anchor)').text('the VIX');
    wrapLabel(title, `every point is one day: what was promised, against what the next `
      + `${V.horizon} days delivered`, w - 8);
  }

  function render() {
    if (view === 'path') drawPath(); else drawScatter();
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>GARCH</th><th>the VIX</th><th>what happened</th></tr>
        <tr><td>average, annualised percent</td>
            <td>${V.mean_garch}</td><td>${V.mean_vix}</td>
            <td><span class="value">${V.mean_real}</span></td></tr>
        <tr><td>correlation with what happened</td>
            <td><span class="value">${V.corr_garch_real}</span></td>
            <td><span class="value">${V.corr_vix_real}</span></td>
            <td>1</td></tr>
        <tr><td>correlation with each other</td>
            <td colspan="2"><span class="value">${V.corr_garch_vix}</span></td>
            <td></td></tr>
      </table>
      <div class="gen-note">
        The two forecasts agree with each other (${V.corr_garch_vix}) more than
        either agrees with the outcome, which is what you would expect of two
        instruments measuring the same slowly moving thing with different noise.
        ${V.corr_vix_real > V.corr_garch_real
    ? `Of the two, the market's number is the better forecast here:
           ${V.corr_vix_real} against ${V.corr_garch_real}. It also sits
           ${V.premium} points <span class="bold">above</span> what actually
           happened on average (${V.mean_vix} against ${V.mean_real}), which is
           not an error: an option is insurance, and insurance is sold above its
           expected cost. That gap has a name, the variance risk premium, and it
           is the reason a model fitted to returns and a number quoted by a
           market are not competing to be the same thing.`
    : `Of the two, the fitted model is the better forecast here:
           ${V.corr_garch_real} against ${V.corr_vix_real}, while the market's
           number sits ${V.premium} points from the outcome on average.`}
      </div>`;
  }

  controls.selectAll('button').data([['over time', 'path'], ['forecast against outcome', 'scatter']])
    .join('button')
    .attr('class', (d) => `atlas-btn${d[1] === view ? '' : ' ghost'}`)
    .text((d) => d[0])
    .on('click', (evt, d) => {
      view = d[1];
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd[1] === view ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
