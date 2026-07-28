/* Four simulated series with their orders hidden, and the textbook rule for
 * reading an order off a correlogram.
 *
 * The rule is: the partial autocorrelation of an AR(p) cuts off after lag p, so
 * take the last lag that pokes outside the band. That is ten hypothesis tests
 * at the five percent level, and the readout says what that costs before the
 * reader has looked at any of them.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { acf, pacf } from '../../assets/js/series.js';

export function initIdentifyWidget(data) {
  const I = data.identify;
  const ex = I.examples;
  const names = ex.map((_, i) => `series ${String.fromCharCode(65 + i)}`);
  let current = 0;
  let revealed = false;

  const controls = d3.select('#identify-controls');
  const readout = document.querySelector('#identify-readout');
  const chart = makeChart('#identify-chart', {
    width: 640, height: 420, margin: { top: 40, right: 16, bottom: 46, left: 54 },
  });
  const { g, w, h } = chart;

  const stripH = 84;
  const panelY = stripH + 62;
  const panelH = h - panelY;
  const panelW = (w - 34) / 2;
  const gStrip = g.append('g');
  const gAcf = g.append('g').attr('transform', `translate(0,${panelY})`);
  const gPacf = g.append('g').attr('transform', `translate(${panelW + 34},${panelY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px');

  function bars(sel, vals, band, label, colourLast) {
    const x = d3.scaleLinear().domain([0.4, vals.length - 0.4]).range([0, panelW]);
    const lo = Math.min(-0.6, d3.min(vals) - 0.08);
    const y = d3.scaleLinear().domain([lo, 1]).range([panelH, 0]);
    drawGrid(sel, x, y, panelW, panelH, { xValues: [2, 4, 6, 8, 10], yValues: [-0.5, 0, 0.5, 1] });
    drawAxes(sel, x, y, panelW, panelH, {
      xValues: [2, 4, 6, 8, 10], yValues: [-0.5, 0, 0.5, 1],
    });
    axisLabels(sel, panelW, panelH, { x: 'lag' });
    sel.selectAll('line.bar').data(vals.slice(1)).join('line')
      .attr('class', 'bar').attr('stroke-width', 6)
      .attr('x1', (d, i) => x(i + 1)).attr('x2', (d, i) => x(i + 1))
      .attr('y1', y(0))
      .attr('stroke', (d, i) => (colourLast && Math.abs(d) > band
        ? 'var(--cosmos)' : 'var(--anchor)'))
      .transition('bar').duration(400)
      .attr('y2', (d) => y(d));
    sel.selectAll('line.band').data([band, -band]).join('line')
      .attr('class', 'band').attr('x1', 0).attr('x2', panelW)
      .attr('y1', (d) => y(d)).attr('y2', (d) => y(d))
      .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '4 4').attr('opacity', 0.55);
    sel.selectAll('text.panel-title').data([label]).join('text')
      .attr('class', 'chart-note panel-title')
      .attr('x', 0).attr('y', -12).style('font-size', '11.5px').text(label);
  }

  function render() {
    const e = ex[current];
    const tail = e.y.slice(-80);
    const x = d3.scaleLinear().domain([0, tail.length - 1]).range([0, w]);
    const ys = d3.scaleLinear().domain(d3.extent(tail)).range([stripH, 0]);
    let p = gStrip.select('path.series');
    if (p.empty()) {
      p = gStrip.append('path').attr('class', 'series').attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.3);
    }
    p.datum(tail).transition('strip').duration(400)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => ys(d)));
    gStrip.selectAll('text.strip-label').data([0]).join('text')
      .attr('class', 'chart-note strip-label').attr('x', 0).attr('y', stripH + 16)
      .style('font-size', '11px').text('the last 80 observations');

    /* recomputed here over the whole series, so what is drawn is this page's
       own arithmetic and the readout can say how far it lands from the
       generator's */
    const a = acf(e.y, I.max_lag);
    const pa = pacf(e.y, I.max_lag);
    bars(gAcf, a, e.band, 'autocorrelation', false);
    bars(gPacf, pa, e.band, 'partial autocorrelation', true);
    wrapLabel(title, revealed
      ? `${names[current]} is ${e.process}`
      : `${names[current]}: what order would you fit?`, w - 8);

    const last = pa.map((v, i) => [i, Math.abs(v)])
      .filter(([i, v]) => i > 0 && v > e.band).map(([i]) => i);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>what it says</th></tr>
        <tr><td>lags outside the band, in the partial plot</td>
            <td><span class="value">${last.length ? last.join(', ') : 'none'}</span></td></tr>
        <tr><td>the textbook rule, the last of those</td>
            <td><span class="value">AR(${e.rule_p})</span></td></tr>
        <tr><td>the smallest AIC over AR(0) to AR(${I.max_lag})</td>
            <td><span class="value">AR(${e.aic_p})</span></td></tr>
        <tr><td>the truth</td>
            <td>${revealed
    ? `<span class="value">${e.process}</span>`
    : 'hidden. Pick an order, then press reveal'}</td></tr>
      </table>
      <div class="gen-note">
        ${revealed
    ? `${e.rule_p === e.true_p && e.aic_p === e.true_p
      ? 'Both got it here.'
      : `${e.rule_p === e.true_p ? 'The rule got it here and AIC did not.'
        : e.aic_p === e.true_p ? 'AIC got it here and the rule did not.'
          : 'Neither got it here.'}`}
           One series is an anecdote, which is why the table below runs
           ${I.n_sim.toLocaleString('en-US')} of them per cell.`
    : `Both plots are drawn from this page's own arithmetic over all ${e.n}
           simulated points, and the largest disagreement with the generator's
           over these ${I.max_lag} lags is
           ${Math.max(...a.map((v, i) => Math.abs(v - e.acf[i])),
      ...pa.map((v, i) => Math.abs(v - e.pacf[i]))).toExponential(1)}.`}
      </div>`;
  }

  controls.selectAll('button').data(names).join('button')
    .attr('class', (d, i) => `atlas-btn${i === current ? '' : ' ghost'}`)
    .text((d) => d)
    .on('click', (evt, d) => {
      current = names.indexOf(d);
      revealed = false;
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd === d ? '' : ' ghost'}`);
      d3.select('#reveal-btn').text('reveal');
      render();
    });
  controls.append('button').attr('id', 'reveal-btn').attr('class', 'atlas-btn ghost')
    .text('reveal').on('click', function onReveal() {
      revealed = !revealed;
      d3.select(this).text(revealed ? 'hide again' : 'reveal');
      render();
    });

  render();
  return { render };
}
