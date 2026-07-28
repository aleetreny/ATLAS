/* The crossover: error against how many series the one network was shown.
 *
 * The horizontal lines are the alternatives that need no sharing at all, and
 * they are what the curve has to get under before any of this is worth doing.
 * The second view is the same numbers per horizon, because an average over
 * twelve months can hide a model that is good at one step and hopeless at
 * twelve.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initScalingWidget(data) {
  const S = data.scaling;
  let view = 'curve';
  const readout = document.querySelector('#scaling-readout');
  const controls = d3.select('#scaling-controls');
  const chart = makeChart('#scaling-chart', {
    width: 640, height: 380, margin: { top: 46, right: 132, bottom: 50, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');

  /* the labels live in the right margin, so they are as short as they can be
     and still be read: the first draft used "exponential smoothing" and it ran
     25 units past the canvas */
  const refs = [
    ['one per series', S.local_nbeats, 'var(--cosmos)'],
    ['smoothing', S.ets, 'var(--galaxy)'],
    ['seasonal naive', S.seasonal_naive, 'var(--squidink)'],
  ];

  function drawCurve() {
    layer.selectAll('*').remove();
    const xs = S.rows.map((r) => r.series);
    const x = d3.scaleLog().domain([xs[0] * 0.8, xs[xs.length - 1] * 1.25]).range([0, w]);
    const vals = S.rows.map((r) => r.mase).concat(refs.map((r) => r[1]));
    const y = d3.scaleLinear().domain([d3.min(vals) * 0.9, d3.max(vals) * 1.08]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xValues: xs, yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xValues: xs, xFmt: d3.format('d'), yTicks: 5 });
    axisLabels(layer, w, h, {
      x: 'series the one network was trained on', y: 'error on held out series',
      leftBudget: margin.left,
    });
    const placed = [];
    refs.forEach(([label, v, colour]) => {
      layer.append('line').attr('x1', 0).attr('x2', w + 6).attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', colour).attr('stroke-width', 1.3).attr('stroke-dasharray', '5 4')
        .attr('opacity', 0.85);
      let yy = y(v) + 4;
      placed.forEach((p) => { if (Math.abs(p - yy) < 13) yy = p + 13; });
      placed.push(yy);
      layer.append('text').attr('class', 'chart-note').attr('x', w + 10).attr('y', yy)
        .style('font-size', '11px').attr('fill', colour).text(label);
    });
    layer.append('path').datum(S.rows).attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d) => x(d.series)).y((d) => y(d.mase)));
    layer.selectAll('circle').data(S.rows).join('circle')
      .attr('cx', (d) => x(d.series)).attr('cy', (d) => y(d.mase)).attr('r', 4.5)
      .attr('fill', 'var(--primary)');
    wrapLabel(title, `one network, four training set sizes, scored on the same `
      + `${S.n_test} held out series`, w - 8);
  }

  function drawHorizons() {
    layer.selectAll('*').remove();
    const H = S.rows[0].per_h.length;
    const x = d3.scaleLinear().domain([1, H]).range([0, w]);
    const series = S.rows.map((r) => [`${r.series} series`, r.per_h, 'var(--primary)'])
      .concat([['seasonal naive', S.seasonal_naive_h, 'var(--squidink)'],
        ['smoothing', S.ets_h, 'var(--galaxy)']]);
    const vals = series.flatMap((s) => s[1]);
    const y = d3.scaleLinear().domain([d3.min(vals) * 0.9, d3.max(vals) * 1.06]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xValues: [1, 3, 6, 9, 12], yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xValues: [1, 3, 6, 9, 12], yTicks: 5 });
    axisLabels(layer, w, h, {
      x: 'horizon, in months', y: 'error on held out series', leftBudget: margin.left,
    });
    const line = d3.line().x((d, i) => x(i + 1)).y((d) => y(d));
    const placed = [];
    series.forEach(([label, vals_, colour], i) => {
      const shade = colour === 'var(--primary)'
        ? d3.interpolateRgb('#c3cadd', '#3c4e85')(i / Math.max(1, S.rows.length - 1))
        : colour;
      layer.append('path').datum(vals_).attr('fill', 'none').attr('stroke', shade)
        .attr('stroke-width', 2).attr('d', line);
      let yy = y(vals_[vals_.length - 1]) + 4;
      placed.forEach((p) => { if (Math.abs(p - yy) < 13) yy = p + 13; });
      placed.push(yy);
      layer.append('text').attr('class', 'chart-note').attr('x', w + 6).attr('y', yy)
        .style('font-size', '11px').attr('fill', shade).text(label);
    });
    wrapLabel(title, `the same models, horizon by horizon`, w - 8);
  }

  function render() {
    if (view === 'curve') drawCurve(); else drawHorizons();
    const beatsLocal = S.rows.filter((r) => r.mase < S.local_nbeats);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>trained on</th>${S.rows.map((r) => `<th>${r.series}</th>`).join('')}
            <th>one per series</th><th>smoothing</th><th>seasonal naive</th></tr>
        <tr><td>error</td>
            ${S.rows.map((r) => `<td><span class="value">${r.mase}</span></td>`).join('')}
            <td>${S.local_nbeats}</td><td>${S.ets}</td><td>${S.seasonal_naive}</td></tr>
        <tr><td>windows it saw</td>
            ${S.rows.map((r) => `<td>${r.windows.toLocaleString('en-US')}</td>`).join('')}
            <td colspan="3"></td></tr>
      </table>
      <div class="gen-note">
        ${view === 'curve'
    ? `Three dashed lines and one solid curve. Two of the dashed lines are
           models that need no sharing at all, and where the curve ends up
           relative to each of them is the result;
           ${beatsLocal.length
      ? `it is already under one of them at the left hand end of the axis.`
      : `it is under neither of them anywhere on this axis.`}`
    : `The same four models, split by how far ahead they are asked to
           forecast. A curve that is flat across the horizons and one that
           climbs are two different kinds of model, whatever their averages
           say.`}
        The paragraph below reads the whole picture, including the two things
        this axis is not.
      </div>`;
  }

  controls.selectAll('button').data([['the crossover', 'curve'], ['by horizon', 'horizon']])
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
