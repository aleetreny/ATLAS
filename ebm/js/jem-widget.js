/* The classifier from the other three articles, read as an energy model.
 *
 * Nobody trained it to say how likely a picture is. But a classifier's logits
 * define an energy anyway, and the question of whether that energy separates
 * real digits from things that are not digits has an answer with a number on
 * it. It is not the answer the idea suggests, and the control that explains it
 * is in the chart: the same uniform noise, rescaled to carry as much ink as a
 * digit, moves from one end of the scale to the other.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

const NAMES = {
  real: 'real held out digits',
  uniform: 'uniform noise',
  uniform_matched: 'the same noise, at a digit’s ink',
  gaussian: 'gaussian noise, digit statistics',
  gaussian_clipped: 'the same, clipped',
  shuffled: 'a digit, pixels shuffled',
  fashion: 'Fashion-MNIST',
};

export function initJemWidget(data) {
  const C = data.classifier;
  const chart = makeChart('#jem-chart', {
    width: 720, height: 380, margin: { top: 52, right: 30, bottom: 60, left: 252 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#jem-readout');

  const rows = C.rows;
  const y = d3.scaleBand().domain(rows.map((r) => r.pool)).range([0, h]).padding(0.26);
  const lo = Math.min(...rows.map((r) => r.energy - r.energy_sd));
  const hi = Math.max(...rows.map((r) => r.energy + r.energy_sd));
  const x = d3.scaleLinear().domain([lo - 1, hi + 1]).range([0, w]);

  drawGrid(g, x, y, w, h, { xTicks: 6, yValues: [] });
  drawAxes(g, x, y, w, h, { xTicks: 6, yValues: [] });
  axisLabels(g, w, h, { x: 'energy, lower means more likely', y: '' });

  const layer = g.append('g');
  const real = rows.find((r) => r.pool === 'real');
  layer.append('line').attr('x1', x(real.energy)).attr('x2', x(real.energy))
    .attr('y1', -6).attr('y2', h)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');

  rows.forEach((r) => {
    const cy = y(r.pool) + y.bandwidth() / 2;
    layer.append('line')
      .attr('x1', x(r.energy - r.energy_sd)).attr('x2', x(r.energy + r.energy_sd))
      .attr('y1', cy).attr('y2', cy)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('opacity', 0.6);
    layer.append('circle').attr('cx', x(r.energy)).attr('cy', cy).attr('r', 6)
      .attr('fill', r.pool === 'real' ? 'var(--primary)' : 'var(--squidink)');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', -10).attr('y', cy + 4).attr('text-anchor', 'end')
      .style('font-size', '11.5px').text(NAMES[r.pool] || r.pool);
  });
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');
  wrapLabel(title, `the energy a digit classifier assigns, on ${C.pool_n} pictures per pool`,
    w - 8);

  const noise = rows.find((r) => r.pool === 'uniform');
  const matched = rows.find((r) => r.pool === 'uniform_matched');
  readout.innerHTML = `
    <table class="gen-table">
      <tr>
        <th>pool</th><th>mean energy</th><th>ink</th>
        <th>ranked against real digits by energy</th><th>by plain confidence</th>
      </tr>
      ${rows.map((r) => `<tr>
        <td>${NAMES[r.pool] || r.pool}</td>
        <td><span class="value">${r.energy}</span> &plusmn; ${r.energy_sd}</td>
        <td>${r.ink}</td>
        <td>${r.auc_energy === null || r.auc_energy === undefined ? 'the reference' : `<span class="value">${r.auc_energy}</span>`}</td>
        <td>${r.auc_confidence === null || r.auc_confidence === undefined ? '' : r.auc_confidence}</td>
      </tr>`).join('')}
    </table>
    <div class="gen-note">
      Read the third column first. An area under the curve of one would mean the energy always
      ranks a real digit above that pool, a half means it cannot tell them apart, and
      <span class="bold">below a half means it prefers the pool to the digits</span>. Uniform
      noise scores ${noise.auc_energy}, so this energy likes noise more than it likes handwriting.
      The reason is measurable and it is not subtle: across all pools the energy correlates
      ${C.ink_correlation} with how much ink a picture has, and uniform noise at full brightness
      has ${noise.ink} against a digit's ${real.ink}. The control settles it. Rescale that same
      noise so it carries a digit's ink and nothing else changes, and the score goes from
      ${noise.auc_energy} to ${matched.auc_energy}. So "a classifier is already an energy model"
      is true, and what its energy has mostly learned here is how bright the picture is.
    </div>`;

  return { render: () => {} };
}
