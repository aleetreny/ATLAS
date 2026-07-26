/* Drawing for the density article: clusters keep the module's shared palette,
 * and noise gets its own unmistakable mark, a small hollow grey cross-hatched
 * circle, because "belongs to nothing" is this article's new output value and
 * it must never be readable as just another cluster colour.
 */

export const COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)',
  'var(--smile)', 'var(--galaxy)', '#0e8a55', '#7c5aed', '#a8410f'];

export function drawLabelled(g, points, labels, role, x, y, opts = {}) {
  const r = opts.r || 3.4;
  const sel = g.selectAll('g.pt').data(points);
  const enter = sel.enter().append('g').attr('class', 'pt');
  enter.append('circle');
  enter.append('line').attr('class', 'x1');
  enter.append('line').attr('class', 'x2');
  const merged = enter.merge(sel);
  merged.attr('transform', (p) => `translate(${x(p[0])},${y(p[1])})`);
  merged.select('circle')
    .attr('r', (p, i) => (labels && labels[i] === -1 ? r * 0.85 : role && role[i] === 'border' ? r * 0.9 : r))
    .attr('fill', (p, i) => {
      if (!labels) return 'var(--stone)';
      if (labels[i] === -1) return 'none';
      return COLOURS[labels[i] % COLOURS.length];
    })
    .attr('stroke', (p, i) => {
      if (!labels) return 'var(--squidink)';
      if (labels[i] === -1) return 'var(--squidink)';
      return role && role[i] === 'border' ? 'var(--squidink)' : 'white';
    })
    .attr('stroke-width', (p, i) => (role && role[i] === 'border' && labels && labels[i] !== -1 ? 1.6 : 0.6))
    .attr('opacity', (p, i) => (labels && labels[i] === -1 ? 0.75 : 0.85));
  /* the little cross through noise points */
  merged.select('line.x1')
    .attr('x1', -r * 0.6).attr('y1', -r * 0.6).attr('x2', r * 0.6).attr('y2', r * 0.6)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
    .attr('opacity', (p, i) => (labels && labels[i] === -1 ? 0.75 : 0));
  merged.select('line.x2')
    .attr('x1', -r * 0.6).attr('y1', r * 0.6).attr('x2', r * 0.6).attr('y2', -r * 0.6)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
    .attr('opacity', (p, i) => (labels && labels[i] === -1 ? 0.75 : 0));
}
