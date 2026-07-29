/* LIME, with its neighbourhood as a control the reader can turn.
 *
 * The true local gradient is drawn alongside because it is the thing a local
 * linear explanation is implicitly claiming to be, and the distance between
 * them is the measurement that a stability plot alone would hide.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));
const pc = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
const pretty = (s) => s.replace(/_/g, ' ').replace('od280/od315 of diluted wines', 'od280/od315');

export function initLimeWidget(data) {
  const host = document.querySelector('#lime-widget');
  if (!host) return;
  const M = data.meta;
  const L = data.lime;
  const chart = makeChart('#lime-chart', {
    width: 640, height: 430, margin: { top: 46, right: 30, bottom: 58, left: 168 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#lime-readout');
  const label = document.querySelector('#lime-width-label');
  const slider = document.querySelector('#lime-width');
  let i = 2;
  slider.max = String(L.rows.length - 1);

  /* The gradient is on the model's own scale and the surrogate's coefficients
   * are on the standardised one, so they are compared after scaling each to
   * its own largest value: what is being compared is the shape and the order,
   * which is what a bar chart of an explanation is read for. */
  const norm = (a) => {
    const m = Math.max(...a.map(Math.abs)) || 1;
    return a.map((v) => v / m);
  };
  const grad = norm(L.grad);

  function render() {
    const row = L.rows[i];
    label.textContent = f2(row.width);
    g.selectAll('*').remove();
    const coef = norm(row.coef);
    const order = coef.map((_, j) => j).sort((a, b) => Math.abs(coef[b]) - Math.abs(coef[a]));
    const names = order.map((j) => pretty(M.columns[j]));
    const y = d3.scaleBand().domain(names).range([0, h]).padding(0.24);
    const x = d3.scaleLinear().domain([-1.1, 1.1]).range([0, w]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'weight, scaled to the largest in each series' });
    g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
    order.forEach((j, k) => {
      const yy = y(names[k]);
      g.append('rect')
        .attr('x', Math.min(x(0), x(coef[j]))).attr('y', yy + 2)
        .attr('width', Math.max(1.5, Math.abs(x(coef[j]) - x(0))))
        .attr('height', y.bandwidth() - 4)
        .attr('fill', j === row.top ? 'var(--primary)' : 'var(--secondary)')
        .attr('fill-opacity', j === row.top ? 0.95 : 0.55);
      g.append('line')
        .attr('x1', x(grad[j])).attr('x2', x(grad[j]))
        .attr('y1', yy - 1).attr('y2', yy + y.bandwidth() + 1)
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
    });
    annotate(g, 0, -28, 'bars: what the local surrogate says', { anchor: 'start' })
      .style('font-size', '11.5px').attr('fill', 'var(--primary)');
    annotate(g, 0, -12, 'ticks: the model true local gradient', { anchor: 'start' })
      .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');

    readout.innerHTML =
      `<p>At a neighbourhood width of <span class="bold">${f2(row.width)}</span> the surrogate `
      + `names <span class="bold">${pretty(M.columns[row.top])}</span> as the most important `
      + `column, and it does so in ${pc(row.top_share)} of twelve runs that differ only in `
      + `their random seed: across those runs the rank correlation is `
      + `<span class="bold">${f3(row.stability)}</span>, where one would be perfect `
      + `agreement. ${row.distinct_tops > 1
        ? `At this width the twelve runs put <span class="bold">${row.distinct_tops} different `
          + `columns</span> in first place between them.`
        : 'At this width every run agrees on which column comes first.'} Against the model's `
      + `true local gradient, measured by finite differences on the model itself, the rank `
      + `correlation is <span class="bold">${sgn(row.grad_corr, 3)}</span>. Move the slider: `
      + `the neighbourhood is a parameter of the explanation, not of the model, and no plot `
      + `carries it.</p>`;
  }

  slider.addEventListener('input', (e) => { i = +e.target.value; render(); });
  render();
}
