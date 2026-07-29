/* What sampling costs, on two axes that disagree about when to stop.
 *
 * The error curve and the ranking curve are drawn together because they answer
 * different questions and practitioners only ever ask the second one: an
 * attribution plot is read for its order long before anybody reads the numbers
 * printed next to the bars.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const n0 = (v) => v.toLocaleString('en-US');
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initKernelWidget(data) {
  const host = document.querySelector('#kernel-widget');
  if (!host) return;
  const K = data.kernel;
  const chart = makeChart('#kernel-chart', {
    width: 640, height: 400, margin: { top: 44, right: 62, bottom: 58, left: 68 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#kernel-readout');

  const x = d3.scaleLog().domain([K.rows[0].m, K.rows[K.rows.length - 1].m]).range([0, w]);
  const y = d3.scaleLog()
    .domain([d3.min(K.rows, (r) => r.err) * 0.7, d3.max(K.rows, (r) => r.err) * 1.3])
    .range([h, 0]);
  const y2 = d3.scaleLinear().domain([0.7, 1.02]).range([h, 0]);
  const xt = K.rows.map((r) => r.m);
  drawGrid(g, x, y, w, h, { xValues: xt });
  drawAxes(g, x, y, w, h, { xValues: xt, xFmt: (v) => n0(v) });
  axisLabels(g, w, h, { x: 'coalitions sampled', y: 'worst error in an attribution' });

  const line = d3.line().x((d) => d[0]).y((d) => d[1]);
  g.append('path')
    .attr('d', line(K.rows.map((r) => [x(r.m), y(r.err)])))
    .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
  K.rows.forEach((r) => {
    g.append('circle').attr('cx', x(r.m)).attr('cy', y(r.err)).attr('r', 4)
      .attr('fill', 'var(--primary)');
  });
  g.append('path')
    .attr('d', line(K.rows.map((r) => [x(r.m), y2(r.top1)])))
    .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4)
    .attr('stroke-dasharray', '5 4');
  K.rows.forEach((r) => {
    g.append('circle').attr('cx', x(r.m)).attr('cy', y2(r.top1)).attr('r', 3.4)
      .attr('fill', 'var(--cosmos)');
  });
  /* The second axis gets its own ticks on the right: a series on a scale with
   * nothing to read it against is decoration, which this atlas has paid for
   * once already. */
  const ax2 = g.append('g').attr('class', 'axis y2').attr('transform', `translate(${w},0)`);
  ax2.call(d3.axisRight(y2).ticks(4).tickFormat(d3.format('.0%')).tickSizeOuter(0));
  annotate(g, 0, -26, 'worst error, left axis', { anchor: 'start' })
    .style('font-size', '11.5px').attr('fill', 'var(--primary)');
  annotate(g, 0, -10, 'how often the largest column is identified, right axis',
    { anchor: 'start' }).style('font-size', '11.5px').attr('fill', 'var(--cosmos)');

  const cheap = K.rows[0];
  const dear = K.rows[K.rows.length - 1];
  const enough = K.rows.find((r) => r.top1 >= 1) ?? dear;
  readout.innerHTML =
    `<p>Both curves are the same runs, measured against the exact values on `
    + `${K.instances} bottles of a ${K.model}. With ${n0(cheap.m)} sampled coalitions the `
    + `worst attribution is off by <span class="bold">${f4(cheap.err)}</span>, which is `
    + `${pc(cheap.relative)} of the largest attribution in the problem; at ${n0(dear.m)} it is `
    + `${f4(dear.err)}. The dashed line is the question people actually ask of these plots, `
    + `which is which bar is longest, and it gets there first: `
    + `<span class="bold">${pc(enough.top1, 0)}</span> agreement by ${n0(enough.m)} `
    + `coalitions, while the values are still moving in the third decimal. Reading the order `
    + `off a sampled attribution is cheap. Reading the numbers is not.</p>`;
}
