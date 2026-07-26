/* The wall at the edge of the data.
 *
 * The shaded half is the region no model ever saw. The tree and the forest go
 * flat there; the straight line does not. The readout quotes the flat value
 * itself, because "it predicts 8.04 for every x from 7 to infinity" is a more
 * useful sentence than "it extrapolates poorly".
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

const SERIES = {
  tree: { label: 'one tree', colour: 'var(--primary)', btn: '#ext-tree' },
  forest: { label: 'forest of 200', colour: 'var(--cosmos)', btn: '#ext-forest' },
  linear: { label: 'a straight line', colour: 'var(--aqua)', btn: '#ext-all' },
};

export function initExtrapWidget(data) {
  const E = data.extrapolation;
  const { g, w, h } = makeChart('#ext-chart', {
    width: 680,
    height: 400,
    /* "a straight line" is the longest end label and it has to fit. */
    margin: { top: 34, right: 132, bottom: 58, left: 64 },
  });

  const x = d3.scaleLinear().domain(d3.extent(E.grid)).range([0, w]);
  const y = d3
    .scaleLinear()
    .domain([d3.min(E.points, (p) => p.y) - 1, d3.max(E.points, (p) => p.y) + 1])
    .range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'y' });

  g.insert('rect', ':first-child')
    .attr('x', x(E.boundary)).attr('y', 0)
    .attr('width', w - x(E.boundary)).attr('height', h)
    .attr('fill', 'var(--squidink)').attr('opacity', 0.06);
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', (x(E.boundary) + w) / 2).attr('y', 16).attr('text-anchor', 'middle')
    .style('font-size', '0.6rem').style('text-transform', 'none').attr('opacity', 0.7)
    .text('no model ever saw this half');

  const clip = 'ext-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -1).attr('y', -1)
    .attr('width', w + 2).attr('height', h + 2);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  g.append('g').selectAll('circle').data(E.points).join('circle')
    .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 3.2)
    .attr('fill', (p) => (p.train ? 'var(--squidink)' : 'none'))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.1)
    .attr('opacity', 0.55);

  const lineG = plot.append('g');
  const labG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const readout = document.querySelector('#ext-readout');
  const state = { keys: ['tree'] };
  const line = d3.line().x((d, i) => x(E.grid[i])).y((d) => y(d));

  function render() {
    const keys = state.keys;
    lineG.selectAll('path.halo').data(keys, (k) => k).join('path')
      .attr('class', 'halo').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5)
      .attr('d', (k) => line(E[k]));
    lineG.selectAll('path.body').data(keys, (k) => k).join('path')
      .attr('class', 'body').attr('fill', 'none')
      .attr('stroke', (k) => SERIES[k].colour).attr('stroke-width', 3.2)
      .attr('d', (k) => line(E[k]));

    /* The tree and the forest both end up flat within a tenth of each other,
     * which is the finding, and it means their end labels land on top of one
     * another. Push them apart with the shared spreader. */
    const placed = spread(
      keys.map((k) => ({ k, y: y(E[k][E[k].length - 1]) + 4, yMax: h })),
      13
    );
    labG.selectAll('text').data(placed, (d) => d.k).join('text')
      .attr('class', 'chart-annotation')
      .attr('x', w + 8)
      .attr('y', (d) => d.y)
      .style('font-size', '0.62rem').style('text-transform', 'none')
      .attr('fill', (d) => SERIES[d.k].colour)
      .text((d) => SERIES[d.k].label);

    title.text('trained on the left half only, then asked about the right');
    document.querySelector('#ext-tree').classList.toggle('ghost', !(keys.length === 1 && keys[0] === 'tree'));
    document.querySelector('#ext-forest').classList.toggle('ghost', !(keys.length === 1 && keys[0] === 'forest'));
    document.querySelector('#ext-all').classList.toggle('ghost', keys.length !== 3);

    const rowsFor = keys.length === 3 ? ['tree', 'forest', 'linear'] : keys;
    readout.innerHTML =
      `<table class="tr-table">
         <tr><th></th><th>rmse beyond the wall</th><th>range it predicts there</th></tr>
         ${rowsFor.map((k) => `<tr><td style="color:${SERIES[k].colour}">${SERIES[k].label}</td>` +
           `<td>${E[k + '_rmse'].toFixed(3)}</td>` +
           `<td>${E[k + '_range'][0].toFixed(2)} to ${E[k + '_range'][1].toFixed(2)}</td></tr>`).join('')}
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `The truth out there runs up to <span class="value">${d3.max(E.points.filter((p) => !p.train), (p) => p.y).toFixed(2)}</span>. ` +
      `The tree answers <span class="value">${E.tree_range[0].toFixed(2)}</span> for every x past the wall, and would ` +
      `answer it for x = 1000. The forest averages two hundred such flat lines and gets ` +
      `<span class="value">${E.forest_range[0].toFixed(2)}</span>, flat. ` +
      `The straight line, which committed to a shape and could therefore have been wrong in an interesting ` +
      `way, gets an rmse of <span class="value">${E.linear_rmse.toFixed(3)}</span> against the tree's ` +
      `<span class="value">${E.tree_rmse.toFixed(3)}</span>.</div>`;
  }

  document.querySelector('#ext-tree').addEventListener('click', () => {
    state.keys = ['tree'];
    render();
  });
  document.querySelector('#ext-forest').addEventListener('click', () => {
    state.keys = ['forest'];
    render();
  });
  document.querySelector('#ext-all').addEventListener('click', () => {
    state.keys = ['tree', 'forest', 'linear'];
    render();
  });
  render();
}
