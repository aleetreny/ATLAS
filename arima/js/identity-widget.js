/* The identity, checked in the page at every rate the slider can reach.
 *
 * Simple exponential smoothing and ARIMA(0,1,1) are the same recursion when
 * theta = alpha - 1, so the two lines drawn here are one line, and the number
 * that matters is the largest gap between them over all 526 months. Then the
 * panel underneath shows why fitting the two gives different answers anyway:
 * the sum of squares keeps falling all the way to the edge of what a smoother
 * is allowed to be, and the likelihood the other estimator maximises is not
 * confined to that interval.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { sesLevels, ma1Forecasts } from '../../assets/js/series.js';

export function initIdentityWidget(data) {
  const E = data.equivalence;
  const y = data.series.co2.y;
  const labels = data.series.co2.labels;
  const SHOW = 96;                       // months drawn, of the 526 measured on
  const grid = d3.range(1, 51).map((i) => i * 0.02);

  const slider = document.querySelector('#alpha-slider');
  const valueOut = document.querySelector('#alpha-value');
  const readout = document.querySelector('#identity-readout');
  const chart = makeChart('#identity-chart', {
    width: 640, height: 470, margin: { top: 46, right: 22, bottom: 52, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const topH = Math.round(h * 0.56);
  const botY = topH + 82;
  const botH = h - botY;
  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px');

  /* the sum of squares of the one step error, at every rate the control can
     take: a reference swept at the resolution of the control, so the best it
     names is both unbeatable and reachable */
  const sse = grid.map((a) => {
    const lv = sesLevels(y, a);
    let s = 0;
    for (let i = 1; i < y.length; i++) s += (y[i] - lv[i - 1]) ** 2;
    return s;
  });
  const bestI = sse.indexOf(Math.min(...sse));

  const tail = y.slice(y.length - SHOW);
  const x = d3.scaleLinear().domain([0, SHOW - 1]).range([0, w]);
  const ys = d3.scaleLinear()
    .domain([d3.min(tail) - 1.5, d3.max(tail) + 1.5]).range([topH, 0]);
  const xt = [0, 24, 48, 72, SHOW - 1];
  drawGrid(gTop, x, ys, w, topH, { xValues: xt, yTicks: 5 });
  drawAxes(gTop, x, ys, w, topH, {
    xValues: xt, yTicks: 5,
    xFmt: (i) => labels[y.length - SHOW + i].slice(0, 4),
  });
  axisLabels(gTop, w, topH, { y: 'parts per million', leftBudget: margin.left });
  gTop.selectAll('circle.obs').data(tail).join('circle').attr('class', 'obs')
    .attr('cx', (d, i) => x(i)).attr('cy', (d) => ys(d)).attr('r', 1.9)
    .attr('fill', 'var(--squidink)').attr('opacity', 0.5);

  const pSes = gTop.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 3.4).attr('opacity', 0.85);
  const pMa = gTop.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--aqua)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  gTop.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
    .style('font-size', '11px').attr('fill', 'var(--primary)')
    .text('thick: the smoother. dashed: the moving average model on top of it');

  const xb = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const yb = d3.scaleLinear().domain([Math.min(...sse) * 0.92, Math.max(...sse) * 1.02])
    .range([botH, 0]);
  drawAxes(gBot, xb, yb, w, botH, {
    xValues: [0, 0.25, 0.5, 0.75, 1], yTicks: 4, xFmt: d3.format('.2f'),
  });
  axisLabels(gBot, w, botH, {
    x: 'smoothing rate', y: 'sum of squares', leftBudget: margin.left, xOffset: 38,
  });
  gBot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 2)
    .attr('d', d3.line().x((d, i) => xb(grid[i])).y((d) => yb(d))(sse));
  const marker = gBot.append('line').attr('y1', 0).attr('y2', botH)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2);
  gBot.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -12)
    .style('font-size', '11.5px')
    .text(`one step sum of squares over all ${y.length} months`);

  function render() {
    const idx = +slider.value;
    const alpha = grid[idx - 1];
    const theta = alpha - 1;
    const lv = sesLevels(y, alpha);
    const ma = ma1Forecasts(y, theta);
    let worst = 0;
    for (let i = 0; i < y.length; i++) worst = Math.max(worst, Math.abs(lv[i] - ma[i]));
    const lvTail = lv.slice(lv.length - SHOW);
    const maTail = ma.slice(ma.length - SHOW);
    pSes.datum(lvTail).attr('d', d3.line().x((d, i) => x(i)).y((d) => ys(d)));
    pMa.datum(maTail).attr('d', d3.line().x((d, i) => x(i)).y((d) => ys(d)));
    marker.attr('x1', xb(alpha)).attr('x2', xb(alpha));
    valueOut.textContent = `${alpha.toFixed(2)}, so theta = ${theta.toFixed(2)}`;
    wrapLabel(title, `alpha = ${alpha.toFixed(2)} and theta = ${theta.toFixed(2)} `
      + `are the same model, drawn twice`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>value</th><th></th></tr>
        <tr><td>largest gap between the two lines, over all ${y.length} months</td>
            <td><span class="value">${worst.toExponential(2)}</span></td>
            <td>parts per million, which is float arithmetic and not a difference</td></tr>
        <tr><td>sum of squared one step errors here</td>
            <td><span class="value">${sse[idx - 1].toFixed(1)}</span></td>
            <td>${idx - 1 === bestI ? 'the best rate the slider can reach'
    : `${(sse[idx - 1] / sse[bestI]).toFixed(3)} times the best the slider can reach, `
      + `which is ${grid[bestI].toFixed(2)}`}</td></tr>
        <tr><td>fitted by sum of squares, with alpha kept inside [0, 1]</td>
            <td><span class="value">${E.co2.alpha_hat.toFixed(4)}</span></td>
            <td>next month: ${E.co2.f_ses} ppm</td></tr>
        <tr><td>fitted by exact likelihood, with theta free</td>
            <td><span class="value">${E.co2.theta_hat}</span></td>
            <td>implies alpha = ${E.co2.implied_alpha}; next month: ${E.co2.f_arima} ppm</td></tr>
      </table>
      <div class="gen-note">
        The largest gap the generator found over the same
        ${E.identity_grid[2]} rates was
        <span class="value">${E.identity_worst.toExponential(2)}</span>, so the
        identity is exact to the last bit of a double at every rate. The two
        <i>fits</i> are ${E.co2.h1_gap} parts per million apart at one month,
        and the reason is in the last two rows: the likelihood lands at theta =
        ${E.co2.theta_hat}, which implies a smoothing rate of
        ${E.co2.implied_alpha}, and a smoothing rate above one is not a
        smoothing rate at all. The estimator that is allowed to go there goes
        there.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render, sse, grid, bestI };
}
