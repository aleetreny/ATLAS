/* Drawing shared by this article's panels.
 *
 * One dimension needs a grammar the rest of the site has not needed yet: a
 * number line with the sample on it and the intervals each rule declares drawn
 * as bands behind it. The rule is that a band is a claim ("everything in here
 * is ordinary") and is therefore drawn as an area, while a score is a number
 * and is drawn as a curve. Nothing is coloured by which rule likes it: the
 * marks are the data, the bands are the opinions.
 */

/* Series names go through this rather than `.chart-annotation`, whose CSS
 * uppercases (which eats greek letters) and adds 2px of letter spacing (which
 * makes four words wider than a panel). Same trick as the module before. */
export function seriesLabel(g, x, y, text, { anchor = 'start', size = 11.5, opacity = 0.8 } = {}) {
  return g.append('text')
    .attr('x', x).attr('y', y)
    .attr('text-anchor', anchor)
    .style('font-family', 'var(--font-mono)')
    .style('font-size', `${size}px`)
    .style('letter-spacing', '0.4px')
    .style('fill', 'var(--squidink)')
    .style('opacity', opacity)
    .text(text);
}

/* Legend chips in the top margin, drawing the glyph that is actually used. */
export function legend(g, items, { x0 = 0, y0 = -14, gap = 120, size = 11 } = {}) {
  const sel = g.selectAll('g.legend-chip').data(items);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', 'legend-chip');
  enter.append('path').attr('class', 'mark');
  enter.append('text').attr('x', 12).attr('dy', '0.32em')
    .style('font-family', 'var(--font-mono)')
    .style('letter-spacing', '0.4px')
    .style('fill', 'var(--squidink)');
  const merged = enter.merge(sel);
  merged.attr('transform', (d, i) => `translate(${x0 + i * gap},${y0})`);
  merged.select('path.mark')
    .attr('d', (d) => (d.shape === 'band'
      ? 'M-6,-5 L6,-5 L6,5 L-6,5 Z'
      : d.shape === 'ring'
        ? 'M-5,0 A5,5 0 1,0 5,0 A5,5 0 1,0 -5,0'
        : d.shape === 'line'
          ? 'M-7,0 L7,0'
          : d.shape === 'dash'
            ? 'M-7,0 L-2,0 M2,0 L7,0'
            : d.shape === 'square'
              ? 'M-4,-4 L4,-4 L4,4 L-4,4 Z'
              : 'M-4,0 A4,4 0 1,0 4,0 A4,4 0 1,0 -4,0'))
    .attr('fill', (d) => (['ring', 'line', 'dash'].includes(d.shape) ? 'none' : d.colour))
    .attr('fill-opacity', (d) => (d.shape === 'band' ? 0.18 : 1))
    .attr('stroke', (d) => d.colour)
    .attr('stroke-width', (d) => (d.shape === 'band' ? 1 : 2.2))
    .attr('stroke-opacity', (d) => (d.shape === 'band' ? 0.55 : 1));
  merged.select('text').style('font-size', `${size}px`).text((d) => d.label);
  return merged;
}

/* The sample on a line, stacked when values collide so that a repeated reading
 * is visible as height rather than hidden under itself. */
export function drawSample(g, values, x, baseline, {
  r = 5, colour = 'var(--anchor)', tol = 9, highlight = () => false, klass = 'sample',
  anim = (s) => s,
} = {}) {
  const placed = [];
  const rows = values.map((v, i) => {
    const px = x(v);
    let level = 0;
    while (placed.some((q) => q.level === level && Math.abs(q.px - px) < tol)) level += 1;
    placed.push({ px, level });
    return { v, i, px, level };
  });
  const sel = g.selectAll(`circle.${klass}`).data(rows);
  sel.exit().remove();
  const merged = sel.enter().append('circle').attr('class', klass)
    .attr('cx', (d) => d.px)
    .attr('cy', (d) => baseline - d.level * (2 * r + 2))
    .merge(sel);
  merged
    .attr('r', r)
    .attr('fill', (d) => (highlight(d.i) ? 'var(--primary)' : colour))
    .attr('stroke', 'var(--paper)')
    .attr('stroke-width', 1.2)
    .attr('opacity', 0.92);
  anim(merged)
    .attr('cx', (d) => d.px)
    .attr('cy', (d) => baseline - d.level * (2 * r + 2));
  return merged;
}

/* A rule's interval, drawn as the area it calls ordinary. */
export function drawBand(g, x, lo, hi, y0, y1, {
  colour = 'var(--anchor)', klass = 'band', opacity = 0.1, anim = (s) => s,
} = {}) {
  const [x0, x1] = [x.range()[0], x.range()[1]];
  const a = Math.max(Math.min(x0, x1), Math.min(x(lo), x(hi)));
  const b = Math.min(Math.max(x0, x1), Math.max(x(lo), x(hi)));
  let sel = g.select(`rect.${klass}`);
  if (sel.empty()) {
    sel = g.append('rect').attr('class', klass)
      .attr('x', a).attr('width', Math.max(0, b - a));
  }
  sel.attr('y', Math.min(y0, y1)).attr('height', Math.abs(y1 - y0))
    .attr('fill', colour).attr('opacity', opacity)
    .attr('stroke', colour).attr('stroke-opacity', 0.4).attr('stroke-width', 1);
  anim(sel).attr('x', a).attr('width', Math.max(0, b - a));
  return sel;
}

/* A vertical marker with a label that knows which side of the panel it is on,
 * because a label anchored right at 92% of the width leaves the canvas. */
export function marker(g, px, y0, y1, label, {
  colour = 'var(--squidink)', dash = '4 3', width = 1.4, size = 11, w = null, dy = -4,
} = {}) {
  const grp = g.append('g').attr('class', 'marker');
  grp.append('line')
    .attr('x1', px).attr('x2', px).attr('y1', y0).attr('y2', y1)
    .attr('stroke', colour).attr('stroke-width', width).attr('stroke-dasharray', dash)
    .attr('opacity', 0.85);
  if (label) {
    const right = w !== null && px > 0.6 * w;
    seriesLabel(grp, px + (right ? -6 : 6), y0 + dy, label,
      { anchor: right ? 'end' : 'start', size, opacity: 0.9 });
  }
  return grp;
}

/* A one dimensional swarm of scores: used for the 178 squared distances, where
 * the interesting structure is a long right tail and a wall of ordinary
 * bottles at the left. */
export function swarm(g, values, x, baseline, {
  r = 3, colour = () => 'var(--anchor)', maxLevel = 9, tol = 6, klass = 'swarm', opacity = 0.85,
} = {}) {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const placed = [];
  order.forEach((d) => {
    const px = x(d.v);
    let level = 0;
    while (level < maxLevel && placed.some((q) => q.level === level && Math.abs(q.px - px) < tol)) {
      level += 1;
    }
    placed.push({ px, level });
    d.px = px;
    d.level = level;
  });
  const sel = g.selectAll(`circle.${klass}`).data(order, (d) => d.i);
  sel.exit().remove();
  const merged = sel.enter().append('circle').attr('class', klass).merge(sel);
  merged
    .attr('cx', (d) => d.px)
    .attr('cy', (d) => baseline - d.level * (2 * r + 1.5))
    .attr('r', r)
    .attr('fill', (d) => colour(d.i))
    .attr('stroke', 'var(--paper)')
    .attr('stroke-width', 0.8)
    .attr('opacity', opacity);
  return merged;
}
