/* How much each block actually changes what it was handed.
 *
 * The argument for the shortcut is that a deeper network should never be
 * worse than a shallower one, because it could always set the extra blocks to
 * do nothing. A plain block cannot do nothing without learning to copy its
 * input exactly; a residual block does nothing by outputting zero, which is
 * the easiest thing a layer with weight decay can do. If that argument is
 * right, the trained residuals should be small, and they should get smaller
 * with depth. This is that measurement.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initResidualWidget(data) {
  const withRes = data.ladder.filter((r) => r.residual.block_residuals);
  const W = 700;
  const H = 380;
  const margin = { top: 30, right: 128, bottom: 54, left: 70 };
  const { g, w, h } = makeChart('#residual-chart', { width: W, height: H, margin });

  let shown = withRes[withRes.length - 1];

  const bars = g.append('g');
  const axes = g.append('g');
  const grid = g.append('g');
  axisLabels(g, w, h, { x: 'block, counted from the input', y: '‖change‖ / ‖input‖' });

  function draw() {
    const vals = shown.residual.block_residuals;
    const x = d3.scaleBand().domain(d3.range(vals.length)).range([0, w]).padding(0.18);
    const y = d3.scaleLinear().domain([0, Math.max(...vals) * 1.12]).range([h, 0]);

    grid.selectAll('*').remove();
    axes.selectAll('*').remove();
    bars.selectAll('*').remove();
    drawGrid(grid, x, y, w, h, { yTicks: 5 });
    drawAxes(axes, x, y, w, h, { xTicks: Math.min(vals.length, 8), yTicks: 5, yFmt: d3.format('.2f') });

    bars.selectAll('rect')
      .data(vals)
      .join('rect')
      .attr('x', (d, i) => x(i))
      .attr('width', x.bandwidth())
      .attr('y', (d) => y(d))
      .attr('height', (d) => h - y(d))
      .attr('fill', 'var(--primary)')
      .attr('opacity', 0.8);

    /* Two blocks in the whole network carry a 1x1 convolution on their
       shortcut: the first of the second stage and the first of the third,
       where the channel count doubles and the resolution halves. The first
       block of all is NOT one of them: it takes sixteen channels from the
       stem and returns sixteen, so its shortcut is a plain copy. */
    const perStage = vals.length / 3;
    const projections = [perStage, perStage * 2];
    projections.forEach((i) => {
      bars.append('line')
        .attr('x1', x(i) - x.step() * 0.09).attr('x2', x(i) - x.step() * 0.09)
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', '#8a94a6')
        .attr('stroke-dasharray', '3 3');
    });
    bars.selectAll('.shape-change')
      .data(projections)
      .join('rect')
      .attr('class', 'shape-change')
      .attr('x', (i) => x(i))
      .attr('width', x.bandwidth())
      .attr('y', (i) => y(vals[i]))
      .attr('height', (i) => h - y(vals[i]))
      /* Ink, not the theme's second violet: outlining a violet bar in a
         slightly different violet marks nothing at all. */
      .attr('fill', 'none')
      .attr('stroke', '#232f3e')
      .attr('stroke-width', 2);

    const first = vals[0];
    const last = vals[vals.length - 1];
    const small = vals.filter((v) => v < 0.5).length;
    document.querySelector('#residual-readout').innerHTML =
      `<span class="bold">${shown.depth} layers, ${vals.length} blocks.</span> `
      + `The first block changes its input by ${first.toFixed(2)} of the input's own size; the last by `
      + `${last.toFixed(2)}. ${small} of ${vals.length} blocks move their input by less than half its size, `
      + `which is the shortcut doing its job: a block that has nothing to add can stay near zero and be ignored, `
      + `and most of them mostly do. The two outlined bars are the only blocks whose shortcut is a 1×1 `
      + `convolution rather than a plain copy, because there the channel count doubles and the resolution `
      + `halves, so those two cannot sit still even if they wanted to.`;
  }

  const row = document.querySelector('#residual-buttons');
  withRes.forEach((r, i) => {
    const b = document.createElement('button');
    b.className = `atlas-button${i === withRes.length - 1 ? ' is-active' : ''}`;
    b.textContent = `${r.depth} layers`;
    b.addEventListener('click', () => {
      shown = r;
      row.querySelectorAll('.atlas-button').forEach((n) => n.classList.remove('is-active'));
      b.classList.add('is-active');
      draw();
    });
    row.appendChild(b);
  });

  draw();
}
