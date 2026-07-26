/* Shared drawing for every clustering panel on the page.
 *
 * The visual language is fixed once: points tinted by cluster, centres as
 * large ringed diamonds, and territory borders as the perpendicular bisectors
 * that nearest-centre assignment actually creates, clipped to the canvas.
 */

export const CLUSTER_COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)',
  'var(--smile)', 'var(--galaxy)', 'var(--seablue)', 'var(--violet)', 'var(--magenta)'];

export function drawClusteredPoints(g, points, labels, x, y, { r = 3.6, unlabelled = false } = {}) {
  return g.selectAll('circle').data(points).join('circle')
    .attr('cx', (p) => x(p[0]))
    .attr('cy', (p) => y(p[1]))
    .attr('r', r)
    .attr('fill', (p, i) => (unlabelled || labels === null ? 'var(--stone)' : CLUSTER_COLOURS[labels[i] % CLUSTER_COLOURS.length]))
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.7)
    .attr('opacity', 0.85);
}

export function drawCenters(g, centers, x, y) {
  const sel = g.selectAll('g.centre').data(centers, (d, i) => i);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', 'centre');
  enter.append('rect')
    .attr('width', 15).attr('height', 15)
    .attr('transform', 'rotate(45 7.5 7.5)')
    .attr('x', -7.5).attr('y', -7.5)
    .attr('stroke', 'white').attr('stroke-width', 2.2);
  const merged = enter.merge(sel);
  merged.select('rect').attr('fill', (c, i) => CLUSTER_COLOURS[i % CLUSTER_COLOURS.length]);
  merged.attr('transform', (c) => `translate(${x(c[0])},${y(c[1])})`);
  return merged;
}

/* The Voronoi borders of the current centres, using d3-delaunay (bundled in
 * d3 v7). Drawn as one path so it can fade as a unit. */
export function drawBorders(g, centers, x, y, w, h) {
  let path = null;
  if (centers.length > 1) {
    const dl = d3.Delaunay.from(centers.map((c) => [x(c[0]), y(c[1])]));
    const vor = dl.voronoi([0, 0, w, h]);
    path = vor.render();
  }
  return g.selectAll('path.borders').data(path ? [path] : []).join('path')
    .attr('class', 'borders')
    .attr('d', (d) => d)
    .attr('fill', 'none')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.2)
    .attr('stroke-dasharray', '6 5')
    .attr('opacity', 0.5);
}
