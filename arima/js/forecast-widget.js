/* One origin out of the rolling evaluation, drawn.
 *
 * The bands belong to the seasonal ARIMA, because it is the only model here
 * that carries a distribution rather than a number: a seasonal naive forecast
 * has no interval unless you build one for it, and pretending otherwise is how
 * a chart ends up implying a confidence nobody computed.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

const MODELS = [
  { key: 'sarima', label: 'SARIMA', bands: true },
  { key: 'ets', label: 'exponential smoothing', bands: false },
  { key: 'snaive', label: 'seasonal naive', bands: false },
  { key: 'naive', label: 'naive', bands: false },
];

export function initForecastWidget(data) {
  const D = data.rolling.demo;
  const labels = data.series.co2.labels;
  let current = 0;
  let level = '95';

  const readout = document.querySelector('#forecast-readout');
  const controls = d3.select('#forecast-controls');
  const chart = makeChart('#forecast-chart', {
    width: 640, height: 380, margin: { top: 44, right: 26, bottom: 46, left: 64 },
  });
  const { g, w, h } = chart;

  const nh = D.history.length;
  const nf = D.truth.length;
  const x = d3.scaleLinear().domain([0, nh + nf - 1]).range([0, w]);
  const all = D.history.concat(D.truth, D.lo95, D.hi95, D.naive);
  const ys = d3.scaleLinear()
    .domain([d3.min(all) - 1, d3.max(all) + 1]).range([h, 0]);
  const xt = [0, 12, 24, 36, 48, nh, nh + nf - 1];
  drawGrid(g, x, ys, w, h, { xValues: xt, yTicks: 5 });
  drawAxes(g, x, ys, w, h, {
    xValues: xt, yTicks: 5,
    xFmt: (i) => labels[D.origin - nh + i].slice(0, 4),
  });
  axisLabels(g, w, h, { y: 'parts per million' });

  const band = g.append('path').attr('opacity', 0.16).attr('fill', 'var(--primary)');
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.5)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => ys(d))(D.history));
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.5).attr('opacity', 0.45).attr('stroke-dasharray', '3 3')
    .attr('d', d3.line().x((d, i) => x(nh - 1 + i))
      .y((d) => ys(d))([D.history[nh - 1]].concat(D.truth)));
  g.selectAll('circle.truth').data(D.truth).join('circle').attr('class', 'truth')
    .attr('cx', (d, i) => x(nh + i)).attr('cy', (d) => ys(d)).attr('r', 3)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
  const line = g.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
  const dots = g.append('g');
  g.append('line').attr('x1', x(nh - 0.5)).attr('x2', x(nh - 0.5))
    .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--squidink)')
    .attr('stroke-dasharray', '4 4').attr('opacity', 0.6);
  g.append('text').attr('class', 'chart-note').attr('x', x(nh - 0.5) - 6).attr('y', 14)
    .attr('text-anchor', 'end').style('font-size', '11px').text('everything it may use');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');

  function render() {
    const m = MODELS[current];
    const f = D[m.key];
    const showBand = m.bands && level !== 'none';
    const lo = level === '80' ? D.lo80 : D.lo95;
    const hi = level === '80' ? D.hi80 : D.hi95;
    band.attr('d', showBand
      ? d3.area().x((d, i) => x(nh + i)).y0((d, i) => ys(lo[i])).y1((d, i) => ys(hi[i]))(f)
      : null);
    line.datum([D.history[nh - 1]].concat(f))
      .transition('fc').duration(400)
      .attr('d', d3.line().x((d, i) => x(nh - 1 + i)).y((d) => ys(d)));
    dots.selectAll('circle').data(f).join('circle')
      .attr('cx', (d, i) => x(nh + i)).attr('r', 3.2).attr('fill', 'var(--primary)')
      .transition('fcdot').duration(400)
      .attr('cy', (d) => ys(d));

    const err = f.map((v, i) => Math.abs(v - D.truth[i]));
    const mae = err.reduce((a, b) => a + b, 0) / err.length;
    const outside = showBand
      ? D.truth.map((v, i) => (v < lo[i] || v > hi[i] ? i + 1 : 0)).filter(Boolean)
      : [];
    wrapLabel(title, `forecasting the twelve months after ${D.label}, `
      + `from ${m.label}`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>horizon</th>${D.truth.map((_, i) => `<th>${i + 1}</th>`).join('')}
            <th>mean</th></tr>
        <tr><td>absolute error, ppm</td>
            ${err.map((e) => `<td>${e.toFixed(2)}</td>`).join('')}
            <td><span class="value">${mae.toFixed(3)}</span></td></tr>
      </table>
      <div class="gen-note">
        ${m.bands
    ? (level === 'none'
      ? `The bands are off. ${m.label} is the only model here that produces
             them, and the next section scores them.`
      : `The shaded band is the ${level}% interval this model reports.
             ${outside.length === 0
        ? `All twelve months fall inside it at this origin.`
        : `${outside.length} of the twelve fall outside it here, at horizon
                ${outside.join(', ')}.`}
             One origin proves nothing either way, which is why the next
             section runs ${data.rolling.n_origins} of them.`)
    : `${m.label} has no interval to draw: it is a rule for producing a
           number, not a distribution over what comes next. That is not a
           detail. Half of what a forecast is asked for is how sure it is, and
           three of the four models on this chart cannot answer.`}
      </div>`;
  }

  controls.selectAll('button.model').data(MODELS).join('button')
    .attr('class', (d, i) => `atlas-btn model${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = MODELS.indexOf(d);
      controls.selectAll('button.model')
        .attr('class', (dd) => `atlas-btn model${dd === d ? '' : ' ghost'}`);
      render();
    });
  controls.selectAll('button.level').data([['95', '95% band'], ['80', '80% band'], ['none', 'no band']])
    .join('button')
    .attr('class', (d) => `atlas-btn level${d[0] === level ? '' : ' ghost'}`)
    .text((d) => d[1])
    .on('click', (evt, d) => {
      level = d[0];
      controls.selectAll('button.level')
        .attr('class', (dd) => `atlas-btn level${dd[0] === level ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
