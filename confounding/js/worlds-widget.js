/* Two graphs, one dataset.
 *
 * The widget deliberately draws the graph and the data in the same frame,
 * because the argument is about which of the two you are allowed to change.
 * The reader can switch the graph. The scatter underneath does not move.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f3 = (v) => v.toFixed(3);

export function initWorldsWidget(data) {
  const host = document.querySelector('#worlds-widget');
  if (!host) return;
  const W = data.worlds;
  const chart = makeChart('#worlds-chart', {
    width: 640, height: 430, margin: { top: 118, right: 26, bottom: 54, left: 62 },
  });
  const { g, svg, w, h, margin } = chart;
  const readout = document.querySelector('#worlds-readout');
  const views = d3.select('#worlds-views');
  let view = 'a';

  const buttons = [
    ['a', 'Z causes both'],
    ['b', 'the treatment causes Z'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  /* The graph lives above the plot area in the top margin, which is the one
   * band of the canvas no data can occupy. */
  function drawGraph() {
    svg.selectAll('g.dag').remove();
    const dag = svg.append('g').attr('class', 'dag')
      .attr('transform', `translate(${margin.left + w / 2 - 130}, 22)`);
    const nodes = view === 'a'
      ? { Z: [130, 0], X: [0, 66], Y: [260, 66] }
      : { X: [0, 66], Z: [130, 0], Y: [260, 66] };
    const edges = view === 'a'
      ? [['Z', 'X'], ['Z', 'Y'], ['X', 'Y']]
      : [['X', 'Z'], ['Z', 'Y'], ['X', 'Y']];
    const defs = dag.append('defs');
    defs.append('marker')
      .attr('id', 'arrow-worlds').attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', 'var(--squidink)');
    edges.forEach(([a, b]) => {
      const p = nodes[a];
      const q = nodes[b];
      const dx = q[0] - p[0];
      const dy = q[1] - p[1];
      const len = Math.hypot(dx, dy);
      const r = 22;
      dag.append('line')
        .attr('x1', p[0] + (dx / len) * r).attr('y1', p[1] + (dy / len) * r)
        .attr('x2', q[0] - (dx / len) * (r + 6)).attr('y2', q[1] - (dy / len) * (r + 6))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.8)
        .attr('marker-end', 'url(#arrow-worlds)');
    });
    Object.entries(nodes).forEach(([name, p]) => {
      dag.append('circle')
        .attr('cx', p[0]).attr('cy', p[1]).attr('r', 20)
        .attr('fill', name === 'Z' ? 'var(--primary)' : 'var(--paper)')
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
      dag.append('text')
        .attr('x', p[0]).attr('y', p[1] + 5)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '700')
        .attr('fill', name === 'Z' ? 'var(--paper)' : 'var(--squidink)')
        .text(name);
    });
    dag.append('text')
      .attr('class', 'chart-note')
      .attr('x', 130).attr('y', 104)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text(view === 'a'
        ? 'Z is a confounder, so it has to be adjusted for'
        : 'Z is a mediator, so adjusting for it removes part of the answer');
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    drawGraph();
    const P = W.points;
    const xs = view === 'a' ? P.xa : P.xb;
    const ys = view === 'a' ? P.ya : P.yb;
    const zs = view === 'a' ? P.za : P.zb;
    const world = view === 'a' ? W.world_a : W.world_b;

    const x = d3.scaleLinear().domain(d3.extent(xs)).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(ys)).nice().range([h, 0]);
    /* An interpolator needs a real colour: `var(--primary)` is an instruction
     * for the style engine and means nothing to d3, which fails silently and
     * paints every cell the same pale tone. Resolve it once, from the document
     * that already knows the accent, so a re-themed article re-themes here. */
    const accent = getComputedStyle(document.documentElement)
      .getPropertyValue('--primary').trim() || '#202c47';
    const colour = d3.scaleSequential(d3.interpolateLab('#f1f3f3', accent))
      .domain(d3.extent(zs));
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'treatment X', y: 'outcome Y' });
    g.append('g').selectAll('circle')
      .data(xs.map((v, i) => [v, ys[i], zs[i]]))
      .join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.2)
      .attr('fill', (d) => colour(d[2]))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.3);
    annotate(g, w - 4, -12, 'shade: the value of Z', { anchor: 'end' })
      .style('font-size', '11.5px');

    readout.innerHTML =
      `<p>In this world the effect of X on Y is <span class="bold">${f3(world.truth)}</span>. `
      + `An analyst who adjusts for Z reports <span class="bold">${f3(world.adjusted)}</span> `
      + `and one who does not reports <span class="bold">${f3(world.unadjusted)}</span>, so `
      + `${Math.abs(world.adjusted - world.truth) < Math.abs(world.unadjusted - world.truth)
        ? 'adjusting is the correct move here'
        : 'adjusting is the wrong move here and doing nothing is correct'}. Now press the `
      + `other button. The scatter changes only because it is a different sample: the two `
      + `worlds have the same means, the same variances and the same covariances, agreeing `
      + `to ${W.cov_max_diff.toExponential(1)}, so every number either analyst can compute `
      + `is the same in both. What changed is the arrow between X and Z, and no test on `
      + `these columns can see it.</p>`;
  }

  render();
}
