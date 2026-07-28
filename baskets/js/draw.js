/* Drawing shared by this article's panels.
 *
 * Two shapes this site has not needed before: a lattice, which is every subset
 * of a small set laid out by size, and a prefix tree. Both are drawn from the
 * same data the algorithms run on, so what is on screen is the search rather
 * than a picture of one.
 */

export function seriesLabel(g, x, y, text, { anchor = 'start', size = 11.5, opacity = 0.85 } = {}) {
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

export function legend(g, items, { x0 = 0, y0 = -14, gap = 150, size = 10.5, cols = 0 } = {}) {
  const sel = g.selectAll('g.legend-chip').data(items);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', 'legend-chip');
  enter.append('path').attr('class', 'mark');
  enter.append('text').attr('x', 12).attr('dy', '0.32em')
    .style('font-family', 'var(--font-mono)')
    .style('letter-spacing', '0.4px')
    .style('fill', 'var(--squidink)');
  const merged = enter.merge(sel);
  const perRow = cols || items.length;
  merged.attr('transform', (d, i) => `translate(${x0 + (i % perRow) * gap},${y0 + Math.floor(i / perRow) * 17})`);
  merged.select('path.mark')
    .attr('d', (d) => (d.shape === 'ring'
      ? 'M-5,0 A5,5 0 1,0 5,0 A5,5 0 1,0 -5,0'
      : d.shape === 'line'
        ? 'M-7,0 L7,0'
        : d.shape === 'dash'
          ? 'M-7,0 L-2,0 M2,0 L7,0'
          : d.shape === 'square'
            ? 'M-4,-4 L4,-4 L4,4 L-4,4 Z'
            : 'M-4,0 A4,4 0 1,0 4,0 A4,4 0 1,0 -4,0'))
    .attr('fill', (d) => (['ring', 'line', 'dash'].includes(d.shape) ? 'none' : d.colour))
    .attr('stroke', (d) => d.colour)
    .attr('stroke-width', 2.2);
  merged.select('text').style('font-size', `${size}px`).text((d) => d.label);
  return merged;
}

/* Lay every subset out by how many items it holds: one row per size, evenly
 * spaced across the width. Rows are wildly uneven (there are 252 subsets of
 * size five out of ten and only one of size ten), which is itself the point:
 * the lattice is fat in the middle and that is where the counting would go. */
export function latticeLayout(keys, k, w, h, { top = 6, bottom = 6 } = {}) {
  const bySize = Array.from({ length: k + 1 }, () => []);
  keys.forEach((key) => bySize[key.split(',').length].push(key));
  const rows = bySize.map((r) => r.length);
  const usable = h - top - bottom;
  const pos = new Map();
  bySize.forEach((row, size) => {
    if (!row.length) return;
    const y = top + (usable * (size - 1)) / Math.max(1, k - 1);
    row.forEach((key, i) => {
      const x = row.length === 1 ? w / 2 : (w * (i + 0.5)) / row.length;
      pos.set(key, { x, y, size });
    });
  });
  return { pos, rows };
}

/* A prefix tree, laid out by depth with each subtree given width in proportion
 * to how many leaves it has. Only the first `maxDepth` levels are drawn: the
 * whole tree of a real database is tens of thousands of nodes and the point of
 * the picture is the shape near the root. */
export function treeLayout(root, w, h, { maxDepth = 4, minCount = 1 } = {}) {
  const nodes = [];
  const links = [];
  let drawn = 0;
  const walk = (node, x0, x1, depth, parent) => {
    const x = (x0 + x1) / 2;
    const y = (h * depth) / Math.max(1, maxDepth);
    const rec = { node, x, y, depth, count: node.count, item: node.item };
    nodes.push(rec);
    drawn += 1;
    if (parent) links.push({ from: parent, to: rec });
    if (depth >= maxDepth) return;
    const kids = [...node.children.values()].filter((c) => c.count >= minCount)
      .sort((a, b) => b.count - a.count);
    const total = kids.reduce((s, c) => s + c.count, 0);
    let cursor = x0;
    kids.forEach((c) => {
      const width = total ? ((x1 - x0) * c.count) / total : 0;
      walk(c, cursor, cursor + width, depth + 1, rec);
      cursor += width;
    });
  };
  walk(root, 0, w, 0, null);
  return { nodes, links, drawn };
}
