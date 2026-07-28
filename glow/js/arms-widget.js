/* Three pieces, ablated one at a time, with a floor under every verdict.
 *
 * Each bar is the same architecture with exactly one thing changed, trained on
 * the same data for the same number of epochs, and the range on it is the two
 * seeds. The article does not call a difference a difference unless it is
 * bigger than the widest of those ranges, and it says so on the bars that are
 * not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initArmsWidget(data) {
  const A = data.arms;
  const chart = makeChart('#arms-chart', {
    width: 640, height: 380, margin: { top: 52, right: 40, bottom: 60, left: 214 },
  });
  const NAME_W = 194;
  const { g, w, h } = chart;
  const readout = document.querySelector('#arms-readout');

  const rows = A.rows;
  const y = d3.scaleBand().domain(rows.map((r) => r.name)).range([0, h]).padding(0.28);
  const lo = Math.min(...rows.map((r) => r.min));
  const hi = Math.max(...rows.map((r) => r.max));
  const pad = (hi - lo) * 0.25 + 1e-6;
  const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, w]);

  drawGrid(g, x, y, w, h, { xTicks: 5, yValues: [] });
  drawAxes(g, x, y, w, h, { xTicks: 5, yValues: [] });
  axisLabels(g, w, h, { x: 'bits per pixel, lower is better', y: '' });

  const layer = g.append('g');
  const full = rows.find((r) => r.name === 'full');
  layer.append('line').attr('x1', x(full.mean)).attr('x2', x(full.mean))
    .attr('y1', -6).attr('y2', h)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');

  rows.forEach((r) => {
    const cy = y(r.name) + y.bandwidth() / 2;
    layer.append('line').attr('x1', x(r.min)).attr('x2', x(r.max))
      .attr('y1', cy).attr('y2', cy)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5).attr('opacity', 0.65);
    layer.append('circle').attr('cx', x(r.mean)).attr('cy', cy).attr('r', 6)
      .attr('fill', r.name === 'full' ? 'var(--primary)' : 'var(--squidink)');
    /* the names are sentences, not words, so they wrap: measured first, then
       lifted by half of what they turned out to need so the block stays
       centred on its own row */
    const name = layer.append('text').attr('class', 'chart-note')
      .attr('x', -10).attr('y', cy + 4).attr('text-anchor', 'end')
      .style('font-size', '11.5px');
    const lines = wrapLabel(name, r.label, NAME_W, { lineHeight: 13 });
    name.attr('y', cy + 4 - (lines - 1) * 13 / 2);
    name.selectAll('tspan').attr('x', -10);
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(r.max) + 8).attr('y', cy + 4)
      .style('font-size', '11px')
      .text(r.name === 'full' ? '' : (r.resolves ? `${r.gap > 0 ? '+' : ''}${r.gap}` : 'inside the floor'));
  });

  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');
  wrapLabel(title, `every bar is ${A.rows[0].seeds} seeds, and the line is the full model`, w - 8);

  const resolved = rows.filter((r) => r.name !== 'full' && r.resolves);
  const inside = rows.filter((r) => r.name !== 'full' && !r.resolves);
  readout.innerHTML = `
    <table class="gen-table">
      <tr>
        <th>what changed</th><th>bits</th><th>seeds</th>
        <th>gap</th><th>its floor</th>
      </tr>
      ${rows.map((r) => `<tr>
        <td>${r.name === 'full' ? '<span class="bold">' + r.short + '</span>' : r.short}</td>
        <td><span class="value">${r.mean}</span></td>
        <td>${r.min} to ${r.max}</td>
        <td>${r.name === 'full' ? 'reference'
    : (r.resolves
      ? `<span class="bold">${r.gap > 0 ? '+' : ''}${r.gap}</span>`
      : `${r.gap > 0 ? '+' : ''}${r.gap}`)}</td>
        <td>${r.name === 'full' ? '' : r.floor}</td>
      </tr>`).join('')}
    </table>
    <div class="gen-note">
      A gap counts only when it is bigger than the floor beside it, and that floor is the seed
      range of <i>this row and the full model</i>, not the widest range in the table: that widest
      range is ${A.floor} bits and belongs to a single noisy arm, so using it everywhere would
      score the one real difference here against a bar almost ten times its own.
      ${resolved.length
    ? `${resolved.length === 1 ? 'One change clears' : `${resolved.length} changes clear`} their
       own: ${resolved.map((r) => `${r.label} (${r.gap > 0 ? '+' : ''}${r.gap}, ${r.times_floor}
       times its floor of ${r.floor})`).join(', ')}.`
    : 'Nothing in this table clears the floor of its own comparison.'}
      ${inside.length
    ? ` ${inside.length === 1 ? 'The other one is' : 'The others are'} inside it:
        ${inside.map((r) => `${r.label} (${r.gap > 0 ? '+' : ''}${r.gap})`).join(', ')}, and this
        page will not claim those. At this size, with four channels after the squeeze, a learned
        mixing has very little to learn that a permutation does not already do, which is exactly
        what a small model should be expected to show.`
    : ''}
    </div>`;

  return { render: () => {} };
}
