/* Shared drawing for a tree's leaves.
 *
 * The generator walks the fitted tree and exports every leaf as a rectangle in
 * data coordinates, so the browser never has to sample a grid: the boxes ARE
 * the model, and drawing them as boxes rather than as a shaded field is the
 * whole reason this article opens in two dimensions.
 */

/* Paint a set of leaf rectangles into a group. Tinted by which class wins,
 * with opacity carrying how sure that leaf is: a 51/49 leaf should not look
 * like a pure one, because the difference between them is the article. */
export function drawBoxes(g, boxes, x, y, { duration = 0 } = {}) {
  const sel = g.selectAll('rect').data(boxes, (d, i) => `${d.x0},${d.y0},${d.x1},${d.y1},${i}`);
  sel.exit().remove();
  const merged = sel
    .enter()
    .append('rect')
    .attr('stroke', 'var(--white)')
    .attr('stroke-width', 1)
    .merge(sel);

  const target = duration ? merged.transition('box').duration(duration) : merged;
  target
    .attr('x', (d) => x(d.x0))
    .attr('y', (d) => y(d.y1))
    .attr('width', (d) => Math.max(0, x(d.x1) - x(d.x0)))
    .attr('height', (d) => Math.max(0, y(d.y0) - y(d.y1)))
    .attr('fill', (d) => (d.c ? 'var(--cosmos)' : 'var(--aqua)'))
    /* Confidence of the leaf, mapped away from zero so an even leaf is still
     * visible as a box rather than disappearing into the background. */
    .attr('fill-opacity', (d) => 0.1 + 0.45 * Math.abs(d.p - 0.5) * 2);
  return merged;
}

/* The points, drawn on top of the boxes so the reader can see which ones each
 * leaf was built around. */
export function drawPoints(g, pts, x, y, { r = 3.4, hollow = false } = {}) {
  return g
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('cx', (p) => x(p.x))
    .attr('cy', (p) => y(p.y))
    .attr('r', r)
    .attr('fill', (p) => (hollow ? 'none' : p.c ? 'var(--cosmos)' : 'var(--aqua)'))
    .attr('stroke', (p) => (hollow ? (p.c ? 'var(--cosmos)' : 'var(--aqua)') : 'var(--squidink)'))
    .attr('stroke-width', hollow ? 1.6 : 0.6)
    .attr('opacity', 0.85);
}
