/* Prediction intervals, scored.
 *
 * A point forecast gets checked constantly and an interval almost never does,
 * although it is the half of the answer that says how much to trust the other
 * half. Both series are here because they disagree: on the one the model was
 * designed for the intervals nearly keep their promise, and on the one it was
 * not they do not, which says that interval coverage is a measurement of
 * misspecification and not of sample size.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initCoverageWidget(data) {
  const SETS = [
    { key: 'co2', label: 'CO2, seasonal ARIMA', rows: data.rolling.coverage,
      n: data.rolling.n_origins, unit: 'months' },
    { key: 'sunspots', label: 'sunspots, AR(9)', rows: data.sunspots.coverage,
      n: data.sunspots.n_origins, unit: 'years' },
  ];
  let current = 0;

  const controls = d3.select('#coverage-controls');
  const readout = document.querySelector('#coverage-readout');
  /* the right margin is wide because the two reference lines have to be
     labelled and there is nowhere inside a full width bar chart to put a
     label: drawn inside, both captions landed on top of the bars and were
     unreadable in the first version of this widget */
  const chart = makeChart('#coverage-chart', {
    width: 640, height: 360, margin: { top: 44, right: 104, bottom: 48, left: 62 },
  });
  const { g, w, h } = chart;
  const x = d3.scaleLinear().domain([0.4, 12.6]).range([0, w]);
  const y = d3.scaleLinear().domain([0.55, 1.0]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: [1, 3, 6, 9, 12], yValues: [0.6, 0.7, 0.8, 0.9, 1.0] });
  drawAxes(g, x, y, w, h, {
    xValues: [1, 3, 6, 9, 12], yValues: [0.6, 0.7, 0.8, 0.9, 1.0],
    yFmt: d3.format('.0%'),
  });
  axisLabels(g, w, h, { y: 'share of the truth inside' });
  const xlab = g.append('text').attr('class', 'axis-label x')
    .attr('x', w / 2).attr('y', h + 42).attr('text-anchor', 'middle');

  [95, 80].forEach((lvl) => {
    const v = lvl / 100;
    g.append('line').attr('x1', 0).attr('x2', w + 6).attr('y1', y(v)).attr('y2', y(v))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2)
      .attr('stroke-dasharray', '5 4').attr('opacity', 0.7);
    g.append('text').attr('class', 'chart-note').attr('x', w + 10).attr('y', y(v) + 4)
      .style('font-size', '11.5px').text(`${lvl}% promised`);
  });

  const bars95 = g.append('g');
  const bars80 = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');

  function render() {
    const S = SETS[current];
    const bw = (w / 12) * 0.34;
    bars95.selectAll('rect').data(S.rows).join('rect')
      .attr('x', (d) => x(d.h) - bw - 1).attr('width', bw)
      .attr('fill', 'var(--anchor)')
      .transition('c95').duration(400)
      .attr('y', (d) => y(d.c95)).attr('height', (d) => h - y(d.c95));
    bars80.selectAll('rect').data(S.rows).join('rect')
      .attr('x', (d) => x(d.h) + 1).attr('width', bw)
      .attr('fill', 'var(--primary)')
      .transition('c80').duration(400)
      .attr('y', (d) => y(d.c80)).attr('height', (d) => h - y(d.c80));
    xlab.text(`horizon, in ${S.unit}`);
    wrapLabel(title, `${S.label}: ${S.n} origins, dark is the 95% interval and `
      + `the accent is the 80%`, w - 8);

    const m95 = S.rows.reduce((a, r) => a + r.c95, 0) / S.rows.length;
    const m80 = S.rows.reduce((a, r) => a + r.c80, 0) / S.rows.length;
    const worst95 = S.rows.reduce((a, r) => (r.c95 < a.c95 ? r : a), S.rows[0]);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>horizon</th>${S.rows.map((r) => `<th>${r.h}</th>`).join('')}<th>mean</th></tr>
        <tr><td>inside the 95% interval</td>
            ${S.rows.map((r) => `<td>${(r.c95 * 100).toFixed(0)}%</td>`).join('')}
            <td><span class="value">${(m95 * 100).toFixed(1)}%</span></td></tr>
        <tr><td>inside the 80% interval</td>
            ${S.rows.map((r) => `<td>${(r.c80 * 100).toFixed(0)}%</td>`).join('')}
            <td><span class="value">${(m80 * 100).toFixed(1)}%</span></td></tr>
      </table>
      <div class="gen-note">
        ${Math.abs(m95 - 0.95) < 0.02
    ? `Averaged over the twelve horizons the 95% interval holds
           ${(m95 * 100).toFixed(1)}% of the truth, which is what it promised.`
    : `Averaged over the twelve horizons the 95% interval holds
           ${(m95 * 100).toFixed(1)}% of the truth rather than 95%, and the
           worst horizon is ${worst95.h} at ${(worst95.c95 * 100).toFixed(0)}%.`}
        ${Math.abs(m80 - 0.8) < 0.02
    ? `The 80% interval holds ${(m80 * 100).toFixed(1)}%.`
    : `The 80% interval holds ${(m80 * 100).toFixed(1)}%, which is
           ${m80 < 0.8 ? 'narrower' : 'wider'} than it claims.`}
        Both are counted over ${S.n} origins. Press the other button and watch
        both rows move together: the paragraph below is about what that
        movement is actually measuring.
      </div>`;
  }

  controls.selectAll('button').data(SETS).join('button')
    .attr('class', (d, i) => `atlas-btn${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = SETS.indexOf(d);
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd === d ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
