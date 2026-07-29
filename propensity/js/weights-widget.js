/* The score, drawn, with the five estimators computed on the spot.
 *
 * The point of running them in the browser is not speed. It is that the
 * numbers under the picture are the numbers those points produce, and the
 * guard in main.js checks them against the ones the generator got on the same
 * rows. An estimate printed next to a cloud it was not computed from is a
 * caption, not a measurement.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { naive, regression, ipw, strata, matching, aipw } from './statkit.js';

const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);

export function initWeightsWidget(data) {
  const host = document.querySelector('#weights-widget');
  if (!host) return;
  const C = data.cloud;
  const M = data.meta;
  const chart = makeChart('#weights-chart', {
    width: 640, height: 410, margin: { top: 36, right: 26, bottom: 58, left: 66 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#weights-readout');
  const views = d3.select('#weights-views');
  let view = 'scores';

  const buttons = [
    ['scores', 'who was likely to be treated'],
    ['weights', 'what weighting does to them'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const est = {
    naive: naive(C.X, C.Y),
    regression: regression(C.V, C.X, C.Y),
    ipw: ipw(C.X, C.Y, C.e_hat),
    strata: strata(C.X, C.Y, C.e_hat),
    matching: matching(C.X, C.Y, C.e_hat),
    aipw: aipw(C.V, C.X, C.Y, C.e_hat),
  };
  const weights = C.X.map((x, i) => (x === 1 ? 1 / C.e_hat[i] : 1 / (1 - C.e_hat[i])));
  const maxShare = Math.max(...weights) / weights.reduce((s, v) => s + v, 0);
  const ess = weights.reduce((s, v) => s + v, 0) ** 2
    / weights.reduce((s, v) => s + v * v, 0);

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, 1]).range([0, w]);

    if (view === 'scores') {
      const bins = d3.bin().domain([0, 1]).thresholds(24);
      const t = bins(C.e_hat.filter((_, i) => C.X[i] === 1));
      const c = bins(C.e_hat.filter((_, i) => C.X[i] === 0));
      const y = d3.scaleLinear()
        .domain([-d3.max(c, (b) => b.length), d3.max(t, (b) => b.length)])
        .nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h, { yFmt: (v) => Math.abs(v) });
      axisLabels(g, w, h, { x: 'estimated probability of being treated', y: 'rows' });
      const draw = (bin, sign, colour) => {
        g.append('g').selectAll('rect')
          .data(bin)
          .join('rect')
          .attr('x', (d) => x(d.x0) + 1)
          .attr('width', (d) => Math.max(1, x(d.x1) - x(d.x0) - 2))
          .attr('y', (d) => (sign > 0 ? y(d.length) : y(0)))
          .attr('height', (d) => Math.abs(y(d.length) - y(0)))
          .attr('fill', colour)
          .attr('fill-opacity', 0.85);
      };
      draw(t, 1, 'var(--primary)');
      draw(c, -1, 'var(--smile)');
      annotate(g, 4, -14, 'treated, above the line', { anchor: 'start' })
        .style('font-size', '12px').attr('fill', 'var(--primary)');
      annotate(g, w - 4, -14, 'untreated, below', { anchor: 'end' })
        .style('font-size', '12px').attr('fill', 'var(--smile)');
      readout.innerHTML =
        `<p>Every row of this sample, placed by the probability the fitted model gives it of `
        + `being treated. The two histograms overlap across the whole range, which is the `
        + `condition everything in this article needs: for each score there are examples of `
        + `both treatments, so there is somebody to compare with. The treated pile leans `
        + `right and the untreated pile leans left, and that lean <span class="bold">is</span> `
        + `the confounding. The difference in means on these rows is `
        + `<span class="bold">${f3(est.naive)}</span> against a true effect of `
        + `${f3(M.tau0)}.</p>`;
    } else {
      const y = d3.scaleLinear().domain([0, d3.max(weights) * 1.05]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'estimated probability of being treated', y: 'weight' });
      g.append('g').selectAll('circle')
        .data(C.e_hat.map((e, i) => [e, weights[i], C.X[i]]))
        .join('circle')
        .attr('cx', (d) => x(d[0]))
        .attr('cy', (d) => y(d[1]))
        .attr('r', 3)
        .attr('fill', (d) => (d[2] === 1 ? 'var(--primary)' : 'var(--smile)'))
        .attr('fill-opacity', 0.7);
      annotate(g, 4, -14, 'treated', { anchor: 'start' })
        .style('font-size', '12px').attr('fill', 'var(--primary)');
      annotate(g, 92, -14, 'untreated', { anchor: 'start' })
        .style('font-size', '12px').attr('fill', 'var(--smile)');
      readout.innerHTML =
        `<p>Weighting turns each row into as many rows as it was unlikely: a treated unit `
        + `with probability 0.05 stands for twenty. The curve is a hyperbola on each side, so `
        + `the largest weight here is <span class="bold">${f3(Math.max(...weights))}</span> `
        + `and carries ${(100 * maxShare).toFixed(2)}% of the total. The effective sample `
        + `size, which is what the weights leave of ${C.X.length} rows, is `
        + `<span class="bold">${ess.toFixed(0)}</span>. On this sample the weighted estimate `
        + `is ${f3(est.ipw)}, matching gives ${f3(est.matching)}, five strata give `
        + `${f3(est.strata)}, the doubly robust version gives ${f3(est.aipw)} and the `
        + `regression gives ${f3(est.regression)}, against a truth of ${f3(M.tau0)}. All six `
        + `numbers were computed in this page from the points above.</p>`;
    }
  }

  render();
}
