/* Folds with no positive in them, counted and predicted.
 *
 * The two curves are the same quantity by two routes, which is the only kind of
 * agreement worth drawing: one is twenty thousand simulated splits and the
 * other is an inclusion and exclusion sum over ratios of binomial coefficients,
 * correcting for the fact that the folds are not independent. They have to
 * agree, and if they did not, one of them would be wrong in a way no amount of
 * staring at either would reveal. That is not hypothetical here. The first
 * version of the formula stopped after two terms, on the usual argument that
 * higher ones are negligible, and returned MINUS 20.69 where the simulation
 * said 1. The argument fails exactly where the question matters: with two
 * positives at most two folds can be occupied, so eight of the ten are empty
 * with certainty, and no truncation of the series can reach a certainty.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initEmptyWidget(data) {
  const E = data.empty;
  const readout = document.querySelector('#empty-readout');

  const chart = makeChart('#empty-chart', {
    width: 640, height: 330, margin: { top: 50, right: 34, bottom: 58, left: 80 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const x = d3.scaleLog().domain([E[0].positives * 0.8, E[E.length - 1].positives * 1.25])
    .range([0, w]);
  const y = d3.scaleLinear().domain([0, 1.03]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: E.map((r) => r.positives), yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xValues: E.map((r) => r.positives), yTicks: 5,
    xFmt: (v) => String(v), yFmt: d3.format('.0%'),
  });
  axisLabels(g, w, h, { x: 'positives among the 500 rows', y: 'chance a fold has none' });

  [['exact', 'var(--primary)', 'the formula', null],
    ['measured', 'var(--anchor)', `${E[0].trials.toLocaleString('en-US')} simulated splits`, '5 3']]
    .forEach(([key, colour, name, dash], i) => {
      layer.append('path').attr('d', d3.line().x((r) => x(r.positives)).y((r) => y(r[key]))(E))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
        .attr('stroke-dasharray', dash);
      E.forEach((r) => layer.append('circle').attr('cx', x(r.positives)).attr('cy', y(r[key]))
        .attr('r', 3.8).attr('fill', colour));
      layer.append('line').attr('x1', 0).attr('x2', 22).attr('y1', -34 + i * 15 - 4)
        .attr('y2', -34 + i * 15 - 4).attr('stroke', colour).attr('stroke-width', 2.4)
        .attr('stroke-dasharray', dash);
      layer.append('text').attr('class', 'chart-note').attr('x', 28).attr('y', -34 + i * 15)
        .style('font-size', '11px').attr('fill', colour).text(name);
    });

  const worst = E.reduce((a, b) => (Math.abs(b.exact - b.measured) > Math.abs(a.exact - a.measured) ? b : a));
  const rows = E.map((r) => `<tr><td>${r.positives}</td>`
    + `<td>${(r.prevalence * 100).toFixed(1)}%</td><td>${(r.exact * 100).toFixed(2)}%</td>`
    + `<td>${(r.measured * 100).toFixed(2)}%</td>`
    + `<td>${Math.abs(r.exact - r.measured).toFixed(4)}</td></tr>`).join('');
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>positives</th>`
    + `<th>prevalence</th><th>formula</th><th>simulated</th><th>difference</th></tr></thead>`
    + `<tbody>${rows}</tbody></table>`
    + `<div class="gen-note">With ${E[0].n} rows in ${E[0].k} folds, the two routes agree to `
    + `${Math.abs(worst.exact - worst.measured).toFixed(4)} at worst, which is about what `
    + `${E[0].trials.toLocaleString('en-US')} draws can resolve. This is the failure mode `
    + `stratification exists for, and it is a hard one rather than a soft one: a fold with `
    + `no positives in it cannot produce a recall, an average precision or an area under a `
    + `curve, so the average over folds is being taken over a different number of folds than `
    + `it says.</div>`;

  return { render: () => {} };
}
