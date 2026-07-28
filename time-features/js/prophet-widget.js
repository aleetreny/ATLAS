/* The trend as a sum of hinges, and the price of a bend.
 *
 * Twenty five possible bends are offered at fixed places and each one is
 * charged for in absolute value, so most of them come out at exactly zero and
 * the ones that survive are the ones the data insisted on. The slider is that
 * price. It is the parameter people tune when a Prophet forecast looks wrong,
 * and it is worth seeing what it actually moves: not the fit, which barely
 * changes, but the extrapolation, which changes by a factor of three.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initProphetWidget(data) {
  const P = data.prophet;
  const y = data.series.y;
  const labels = data.series.labels;
  const slider = document.querySelector('#tau-slider');
  const valueOut = document.querySelector('#tau-value');
  const readout = document.querySelector('#prophet-readout');
  slider.min = 0;
  slider.max = P.sweep.length - 1;
  slider.value = String(P.sweep.findIndex((s) => s.tau === P.tau));

  const chart = makeChart('#prophet-chart', {
    width: 640, height: 380, margin: { top: 46, right: 26, bottom: 50, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const n = y.length;
  const x = d3.scaleLinear().domain([0, n - 1]).range([0, w]);
  const yy = d3.scaleLinear().domain([d3.min(y) - 2, d3.max(y) + 2]).range([h, 0]);
  const years = [1960, 1970, 1980, 1990, 2000].map((yr) => labels.indexOf(`${yr}-01`))
    .filter((i) => i >= 0);
  drawGrid(g, x, yy, w, h, { xValues: years, yTicks: 5 });
  drawAxes(g, x, yy, w, h, { xValues: years, yTicks: 5, xFmt: (i) => labels[i].slice(0, 4) });
  axisLabels(g, w, h, { y: 'parts per million', leftBudget: margin.left });
  g.append('path').datum(y).attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.9).attr('opacity', 0.55)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yy(d)));
  const trendPath = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2.6);
  const marks = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  g.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
    .style('font-size', '11px')
    .text('grey: the series. accent: the trend, with its surviving bends marked');

  const best = P.sweep.reduce((a, b) => (b.mase < a.mase ? b : a), P.sweep[0]);

  function render() {
    const row = P.sweep[+slider.value];
    trendPath.datum(row.trend).transition('tr').duration(350)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => yy(d)));
    /* the changepoints are at fixed places; which ones survive is the fit */
    const cps = [];
    for (let j = 0; j < P.n_changepoints; j++) {
      const b = row.beta[2 + j];
      if (Math.abs(b) > 1e-8) {
        const t = ((j + 1) / (P.n_changepoints + 1)) * 0.8;
        cps.push({ i: Math.round(t * (n - 1)), b });
      }
    }
    marks.selectAll('line').data(cps).join('line')
      .attr('x1', (d) => x(d.i)).attr('x2', (d) => x(d.i))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1).attr('opacity', 0.35);
    valueOut.textContent = `${row.tau}`;
    wrapLabel(title, `a bend costs ${row.tau}: ${row.active} of ${P.n_changepoints} survive, `
      + `and the forecast scores ${row.mase}`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>price of a bend</th>${P.sweep.map((s) => `<th>${s.tau}</th>`).join('')}</tr>
        <tr><td>bends that survive</td>
            ${P.sweep.map((s) => `<td>${s.active}</td>`).join('')}</tr>
        <tr><td>error, ${P.n_origins} rolling origins</td>
            ${P.sweep.map((s) => `<td${s.tau === row.tau ? ' style="font-weight:600"' : ''}>`
    + `${s.mase}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        ${row.active === 0
    ? `At this price nothing bends: the trend is a straight line through
           forty three years of a curve that is not straight, and the forecast
           scores ${row.mase}, the worst in the row.`
    : `${row.active} bends survive here.`}
        The library's own default is ${P.tau}, which on this series leaves
        ${P.sweep.find((s) => s.tau === P.tau).active} bends and scores
        ${P.sweep.find((s) => s.tau === P.tau).mase}; the best price on this
        sweep is <span class="value">${best.tau}</span> at
        <span class="value">${best.mase}</span>, with ${best.active} bends. That
        is a factor of ${(P.sweep[P.sweep.length - 1].mase / best.mase).toFixed(2)}
        between the ends of one slider, on a model whose fit to the history
        looks reasonable at every setting: the price of a bend is not a
        smoothing preference, it is the extrapolation.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
