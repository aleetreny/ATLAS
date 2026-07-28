/* Drawing shared by this article's panels, keeping the module's palette and its
 * grammar for what a centre looks like: a ring around a point that is really
 * there. Affinity propagation's exemplars are medoids by another name, so they
 * are drawn the way the previous article drew medoids, on purpose.
 *
 * The one thing new here is edges. Both algorithms in this article live on the
 * n by n table, and an edge is the only way to put an entry of that table on
 * screen next to the points it belongs to.
 */

export const COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)',
  'var(--smile)', 'var(--galaxy)', '#0e8a55', '#7c5aed', '#a8410f',
  '#0f766e', '#b45309', '#155e75', '#86198f', '#4d7c0f', '#9f1239',
  '#0891b2', '#c026d3'];

export const colourOf = (l) => (l < 0 ? 'none' : COLOURS[l % COLOURS.length]);

/* Points tinted by cluster; -1 comes out hollow and crossed, as in the density
 * article. */
export function drawPoints(g, points, labels, x, y, { r = 3.4, faded = null, stroke = 0.7 } = {}) {
  const sel = g.selectAll('g.pt').data(points);
  const enter = sel.enter().append('g').attr('class', 'pt');
  enter.append('circle');
  enter.append('line').attr('class', 'x1');
  enter.append('line').attr('class', 'x2');
  sel.exit().remove();
  const merged = enter.merge(sel);
  const isNoise = (i) => labels && labels[i] === -1;
  merged.attr('transform', (p) => `translate(${x(p[0])},${y(p[1])})`);
  merged.select('circle')
    .attr('r', (p, i) => (isNoise(i) ? r * 0.85 : r))
    .attr('fill', (p, i) => (labels ? colourOf(labels[i]) : 'var(--stone)'))
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', stroke)
    .attr('opacity', (p, i) => (faded && faded(i) ? 0.22 : 0.85));
  merged.select('line.x1')
    .attr('x1', -r * 0.6).attr('y1', -r * 0.6).attr('x2', r * 0.6).attr('y2', r * 0.6)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
    .attr('opacity', (p, i) => (isNoise(i) ? 0.75 : 0));
  merged.select('line.x2')
    .attr('x1', -r * 0.6).attr('y1', r * 0.6).attr('x2', r * 0.6).attr('y2', -r * 0.6)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
    .attr('opacity', (p, i) => (isNoise(i) ? 0.75 : 0));
  return merged;
}

/* An exemplar: the module's ring, because that is what a centre which is also
 * one of the points has looked like since the previous article. */
export function drawExemplars(g, centers, x, y, { r = 11, colour = null } = {}) {
  const sel = g.selectAll('circle.exemplar').data(centers, (d, i) => i);
  sel.exit().remove();
  const merged = sel.enter().append('circle').attr('class', 'exemplar').merge(sel);
  merged
    .attr('cx', (c) => x(c[0]))
    .attr('cy', (c) => y(c[1]))
    .attr('r', r)
    .attr('fill', 'none')
    .attr('stroke', (c, i) => colour || colourOf(i))
    .attr('stroke-width', 3)
    .attr('opacity', 0.95);
  return merged;
}

/* Edges of the graph. `items` are {a, b, w} with w already scaled to [0,1];
 * width and opacity both carry the weight, because on a fully connected
 * gaussian graph almost every edge is nearly nothing and only the ink can say
 * so. */
export function drawEdges(g, items, points, x, y, {
  colour = 'var(--squidink)', maxWidth = 1.6, floor = 0.12,
} = {}) {
  const sel = g.selectAll('line.edge').data(items);
  sel.exit().remove();
  const merged = sel.enter().append('line').attr('class', 'edge').merge(sel);
  merged
    .attr('x1', (d) => x(points[d.a][0]))
    .attr('y1', (d) => y(points[d.a][1]))
    .attr('x2', (d) => x(points[d.b][0]))
    .attr('y2', (d) => y(points[d.b][1]))
    .attr('stroke', colour)
    .attr('stroke-width', (d) => 0.35 + maxWidth * d.w)
    .attr('opacity', (d) => floor + 0.5 * d.w);
  return merged;
}

/* Legend chips in the top margin, drawing the actual glyph rather than a dot
 * standing in for it. */
export function legend(g, items, { x0 = 0, y0 = -12, gap = 118 } = {}) {
  const sel = g.selectAll('g.legend-chip').data(items);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', 'legend-chip');
  enter.append('path').attr('class', 'mark');
  enter.append('text').attr('x', 11).attr('dy', '0.32em')
    .attr('class', 'chart-annotation').style('font-size', '0.62rem')
    .style('letter-spacing', '1px');
  const merged = enter.merge(sel);
  merged.attr('transform', (d, i) => `translate(${x0 + i * gap},${y0})`);
  merged.select('path.mark')
    .attr('d', (d) => (d.shape === 'diamond'
      ? 'M0,-5 L5,0 L0,5 L-5,0 Z'
      : d.shape === 'ring'
        ? 'M-5,0 A5,5 0 1,0 5,0 A5,5 0 1,0 -5,0'
        : d.shape === 'line'
          ? 'M-6,0 L6,0'
          : 'M-4,0 A4,4 0 1,0 4,0 A4,4 0 1,0 -4,0'))
    .attr('fill', (d) => (d.shape === 'ring' || d.shape === 'line' ? 'none' : d.colour))
    .attr('stroke', (d) => (d.shape === 'ring' || d.shape === 'line' ? d.colour : 'none'))
    .attr('stroke-width', 2.2);
  merged.select('text').text((d) => d.label);
  return merged;
}
