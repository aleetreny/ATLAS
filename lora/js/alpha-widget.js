/* Alpha, swept twice.
 *
 * The scaling factor in front of the adapter is the parameter people tune as if
 * it were capacity. It is not: BA is linear in B, so multiplying the whole
 * product by alpha/r multiplies the gradient reaching B by the same factor,
 * which is a learning rate in a different hat. The test is one line long. Sweep
 * alpha with the learning rate fixed, then sweep it again with the learning
 * rate divided by the same factor. If the explanation is right, the second
 * curve is flat.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const ARMS = [['fixed lr', 'var(--primary)', null],
  ['compensated lr', 'var(--anchor)', '5 3']];

export function initAlphaWidget(data) {
  const A = data.alpha;
  const readout = document.querySelector('#alpha-readout');

  const chart = makeChart('#alpha-chart', {
    width: 640, height: 350, margin: { top: 54, right: 30, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const alphas = Array.from(new Set(A.rows.map((r) => r.alpha))).sort((a, b) => a - b);
  const x = d3.scaleLog().domain([alphas[0] * 0.85, alphas[alphas.length - 1] * 1.2]).range([0, w]);
  const vals = A.rows.flatMap((r) => [r.mean - r.sd, r.mean + r.sd]);
  const y = d3.scaleLinear().domain([d3.min(vals) - 0.006, d3.max(vals) + 0.006]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: alphas, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xValues: alphas, yTicks: 5, xFmt: (v) => String(v), yFmt: d3.format('.3f') });
  axisLabels(g, w, h, { x: 'alpha, at rank ' + A.rank, y: 'accuracy on the new task' });

  ARMS.forEach(([arm, colour, dash], i) => {
    const rows = A.rows.filter((r) => r.arm === arm).sort((a, b) => a.alpha - b.alpha);
    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.alpha)).y((r) => y(r.mean))(rows))
      .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
      .attr('stroke-dasharray', dash);
    rows.forEach((r) => {
      layer.append('line').attr('x1', x(r.alpha)).attr('x2', x(r.alpha))
        .attr('y1', y(r.mean - r.sd)).attr('y2', y(r.mean + r.sd))
        .attr('stroke', colour).attr('stroke-width', 1.2);
      layer.append('circle').attr('cx', x(r.alpha)).attr('cy', y(r.mean)).attr('r', 4)
        .attr('fill', colour);
    });
    layer.append('line').attr('x1', 0).attr('x2', 22).attr('y1', -44 + i * 15 - 4)
      .attr('y2', -44 + i * 15 - 4).attr('stroke', colour).attr('stroke-width', 2.4)
      .attr('stroke-dasharray', dash);
    layer.append('text').attr('class', 'chart-note').attr('x', 28).attr('y', -44 + i * 15)
      .style('font-size', '11px').attr('fill', colour)
      .text(arm === 'fixed lr' ? 'learning rate held fixed'
        : 'learning rate divided by the same factor');
  });

  const spanOf = (arm) => {
    const rows = A.rows.filter((r) => r.arm === arm);
    const hi = rows.reduce((a, b) => (b.mean > a.mean ? b : a));
    const lo = rows.reduce((a, b) => (b.mean < a.mean ? b : a));
    return { span: hi.mean - lo.mean, hi, lo, floor: hi.sd + lo.sd };
  };
  const fixed = spanOf('fixed lr');
  const comp = spanOf('compensated lr');
  const flat = comp.span <= comp.floor;
  readout.innerHTML = `<div class="gen-note">With the learning rate held fixed, alpha moves `
    + `the result by <span class="bold">${fixed.span.toFixed(4)}</span> across `
    + `${alphas[0]} to ${alphas[alphas.length - 1]}, best at alpha = ${fixed.hi.alpha}, `
    + `against ${fixed.floor.toFixed(4)} of seed spread at its two ends. With the learning `
    + `rate divided by the same factor it moves it by `
    + `<span class="bold">${comp.span.toFixed(4)}</span> against `
    + `${comp.floor.toFixed(4)} of seed spread, `
    + (flat
      ? `which is flat to the precision this measurement has. Alpha is a learning rate. `
        + `Tuning it and tuning the learning rate are the same experiment run twice.`
      : `which is still outside its own noise, so on this setup alpha is doing something `
        + `beyond rescaling the step. That is worth knowing and it is not what the algebra `
        + `predicts.`)
    + `</div>`;

  return { render: () => {} };
}
