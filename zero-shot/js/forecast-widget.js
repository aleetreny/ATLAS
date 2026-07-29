/* A hundred continuations of a series the model was never fitted to.
 *
 * Sampling is not a decoration here. A point forecast has to be extracted from
 * these paths (the median is used), and the spread between them is an interval
 * that costs nothing extra and can be scored the same way the analytic ones in
 * the first article of this branch were scored. The second view does exactly
 * that, beside those.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initForecastWidget(data) {
  const P = data.pretrain;
  const D = P.demo;
  let view = 'paths';
  const controls = d3.select('#forecast-controls');
  const readout = document.querySelector('#forecast-readout');
  const chart = makeChart('#forecast-chart', {
    width: 640, height: 370, margin: { top: 46, right: 100, bottom: 50, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');
  const H = D.truth.length;
  const nh = D.history.length;

  function drawPaths() {
    layer.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, nh + H - 1]).range([0, w]);
    const all = D.history.concat(D.truth, D.lo95, D.hi95);
    const y = d3.scaleLinear().domain([d3.min(all) - 1, d3.max(all) + 1]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(layer, w, h, {
      x: 'months, the last twelve unseen', y: 'parts per million', leftBudget: margin.left,
    });
    layer.append('path').datum(D.history).attr('fill', 'none').attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 1.5)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    D.paths.forEach((p) => {
      layer.append('path').datum([D.history[nh - 1]].concat(p)).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 0.7).attr('opacity', 0.22)
        .attr('d', d3.line().x((d, i) => x(nh - 1 + i)).y((d) => y(d)));
    });
    layer.append('path').datum([D.history[nh - 1]].concat(D.median)).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d, i) => x(nh - 1 + i)).y((d) => y(d)));
    layer.selectAll('circle.truth').data(D.truth).join('circle').attr('class', 'truth')
      .attr('cx', (d, i) => x(nh + i)).attr('cy', (d) => y(d)).attr('r', 3)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
    layer.append('line').attr('x1', x(nh - 0.5)).attr('x2', x(nh - 0.5))
      .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--squidink)')
      .attr('stroke-dasharray', '4 4').attr('opacity', 0.6);
    layer.append('text').attr('class', 'chart-note').attr('x', w + 6)
      .attr('y', y(D.median[H - 1]) + 4).style('font-size', '11px')
      .attr('fill', 'var(--primary)').text('the median');
    layer.append('text').attr('class', 'chart-note').attr('x', w + 6)
      .attr('y', y(D.truth[H - 1]) + 18).style('font-size', '11px').text('what happened');
    wrapLabel(title, `${D.paths.length} of the sampled continuations after ${D.label}, from a `
      + `model that has never seen this series`, w - 8);
  }

  function drawCoverage() {
    layer.selectAll('*').remove();
    const rows = P.co2.coverage;
    const prev = data.board ? data.board.coverage : null;
    const x = d3.scaleLinear().domain([0.4, 12.6]).range([0, w]);
    const y = d3.scaleLinear().domain([0.55, 1.0]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xValues: [1, 3, 6, 9, 12], yValues: [0.6, 0.7, 0.8, 0.9, 1] });
    drawAxes(layer, x, y, w, h, {
      xValues: [1, 3, 6, 9, 12], yValues: [0.6, 0.7, 0.8, 0.9, 1], yFmt: d3.format('.0%'),
    });
    axisLabels(layer, w, h, {
      x: 'horizon, in months', y: 'share of the truth inside', leftBudget: margin.left,
    });
    [[0.95, '95% promised'], [0.8, '80% promised']].forEach(([v, label]) => {
      layer.append('line').attr('x1', 0).attr('x2', w + 6).attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '5 4').attr('opacity', 0.7);
      layer.append('text').attr('class', 'chart-note').attr('x', w + 10).attr('y', y(v) + 4)
        .style('font-size', '11px').text(label);
    });
    const bw = (w / 12) * 0.3;
    layer.selectAll('rect.a').data(rows).join('rect').attr('class', 'a')
      .attr('x', (d) => x(d.h) - bw - 1).attr('width', bw)
      .attr('y', (d) => y(d.c95)).attr('height', (d) => h - y(d.c95))
      .attr('fill', 'var(--primary)');
    if (prev) {
      layer.selectAll('rect.b').data(prev).join('rect').attr('class', 'b')
        .attr('x', (d) => x(d.h) + 1).attr('width', bw)
        .attr('y', (d) => y(d.c95)).attr('height', (d) => h - y(d.c95))
        .attr('fill', 'var(--anchor)').attr('opacity', 0.85);
    }
    layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
      .style('font-size', '11px')
      .text('accent: sampled continuations. blue: the seasonal ARIMA of the first article');
    wrapLabel(title, `the 95% interval, counted over ${P.co2.n_origins} origins`, w - 8);
  }

  function render() {
    if (view === 'paths') drawPaths(); else drawCoverage();
    const c = P.co2.coverage;
    const m95 = c.reduce((a, r) => a + r.c95, 0) / c.length;
    const prev = data.board ? data.board.coverage : null;
    const p95 = prev ? prev.reduce((a, r) => a + r.c95, 0) / prev.length : null;
    const best = data.board
      ? data.board.rows.reduce((a, b) => (b.mase_mean < a.mase_mean ? b : a), data.board.rows[0])
      : null;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>error, one month</th><th>error, averaged over twelve</th>
            <th>inside the 95% interval</th></tr>
        <tr><td>this model, never fitted to the series</td>
            <td>${P.co2.per_h[0]}</td>
            <td><span class="value">${P.co2.mase}</span></td>
            <td><span class="value">${(m95 * 100).toFixed(1)}%</span></td></tr>
        ${best ? `<tr><td>${best.model}, from the first article, fitted to it</td>
            <td>${best.mase[0]}</td>
            <td>${best.mase_mean}</td>
            <td>${(p95 * 100).toFixed(1)}%</td></tr>` : ''}
      </table>
      <div class="gen-note">
        ${view === 'paths'
    ? `Each faint line is one sampled continuation, drawn letter by letter. The
           point forecast is their median and the band is their percentiles, so
           both come out of the same hundred draws and neither needed a formula.`
    : `Each bar counts, at one horizon, how often the truth landed inside the
           band that promised to contain it. The dashed lines are the two
           promises.`}
        Nothing in the last column of the table was assumed; it was counted.
        The paragraph below puts both rows next to what the rest of this branch
        managed.
      </div>`;
  }

  controls.selectAll('button').data([['the continuations', 'paths'], ['what they covered', 'cov']])
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
