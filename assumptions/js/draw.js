/* Drawing shared by this article's panels. The module's palette is kept, and
 * so is its grammar for what a centre looks like, with one addition: a medoid
 * is drawn as a ring around a point that is really there, never as a floating
 * marker, because that is the whole difference between it and a mean.
 */

export const COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)',
  'var(--smile)', 'var(--galaxy)', '#0e8a55', '#7c5aed', '#a8410f'];

export const colourOf = (l) => (l < 0 ? 'none' : COLOURS[l % COLOURS.length]);

/* Points tinted by cluster; -1 (noise, or "belongs to nothing") comes out
 * hollow and crossed, as in the density article. */
export function drawPoints(g, points, labels, x, y, { r = 3.4, faded = null } = {}) {
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
    .attr('stroke-width', 0.7)
    .attr('opacity', (p, i) => (faded && faded(i) ? 0.25 : 0.85));
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

/* A mean: the module's ringed diamond, floating wherever the arithmetic put
 * it, which is very often nowhere in particular. */
export function drawMeans(g, centers, x, y, { size = 15 } = {}) {
  const sel = g.selectAll('g.mean').data(centers, (d, i) => i);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', 'mean');
  enter.append('rect')
    .attr('width', size).attr('height', size)
    .attr('transform', `rotate(45 ${size / 2} ${size / 2})`)
    .attr('x', -size / 2).attr('y', -size / 2)
    .attr('stroke', 'white').attr('stroke-width', 2.2);
  const merged = enter.merge(sel);
  merged.select('rect').attr('fill', (c, i) => colourOf(i));
  merged.attr('transform', (c) => `translate(${x(c[0])},${y(c[1])})`);
  return merged;
}

/* A medoid: a ring drawn around one of the points already on screen. Wide
 * enough to stay visible when a mean lands in the same place, which on clean
 * data is most of the time, and which is exactly the moment the reader is
 * being asked to compare the two. */
export function drawMedoids(g, centers, x, y, { r = 11 } = {}) {
  const sel = g.selectAll('circle.medoid').data(centers, (d, i) => i);
  sel.exit().remove();
  const merged = sel.enter().append('circle').attr('class', 'medoid').merge(sel);
  merged
    .attr('cx', (c) => x(c[0]))
    .attr('cy', (c) => y(c[1]))
    .attr('r', r)
    .attr('fill', 'none')
    .attr('stroke', (c, i) => colourOf(i))
    .attr('stroke-width', 3)
    .attr('opacity', 0.95);
  return merged;
}

/* Legend chips in the top margin: cheaper than a key off to one side, and it
 * survives a phone.
 *
 * The mark is the actual glyph, not a dot standing in for it. A round dot
 * beside the words "medoids, rings" is a key that has to be read twice, and it
 * was worse than that here: the marks on the chart are tinted per cluster, so
 * a single-coloured dot was inviting the reader to look for a magenta ring
 * that does not exist. */
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
        : 'M-4,0 A4,4 0 1,0 4,0 A4,4 0 1,0 -4,0'))
    .attr('fill', (d) => (d.shape === 'ring' ? 'none' : d.colour))
    .attr('stroke', (d) => (d.shape === 'ring' ? d.colour : 'none'))
    .attr('stroke-width', 2.2);
  merged.select('text').text((d) => d.label);
  return merged;
}
