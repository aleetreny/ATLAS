/* Drawing shared by this article's panels.
 *
 * The one new thing here is a score field: every detector on this page is a
 * function of position, not just of the points, and the honest way to show what
 * a detector believes is to evaluate it everywhere and paint that. It is
 * rendered to an offscreen canvas and placed in the SVG as an image, because a
 * sixty by sixty grid of rects is three and a half thousand nodes that the
 * browser then has to re-layout on every slider move.
 */

/* CSS custom properties do not exist inside a canvas, so the accent is read
 * from the document once and kept as a literal. Copying the value into the file
 * instead is how an earlier article ended up with a ramp that no longer matched
 * its own page. */
let ACCENT = null;
export function accent() {
  if (ACCENT === null) {
    ACCENT = getComputedStyle(document.documentElement)
      .getPropertyValue('--primary').trim() || '#065f46';
  }
  return ACCENT;
}

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
          : d.shape === 'cross'
            ? 'M-4,-4 L4,4 M-4,4 L4,-4'
            : d.shape === 'square'
              ? 'M-4,-4 L4,-4 L4,4 L-4,4 Z'
              : 'M-4,0 A4,4 0 1,0 4,0 A4,4 0 1,0 -4,0'))
    .attr('fill', (d) => (['ring', 'line', 'dash', 'cross'].includes(d.shape) ? 'none' : d.colour))
    .attr('stroke', (d) => d.colour)
    .attr('stroke-width', 2.2);
  merged.select('text').style('font-size', `${size}px`).text((d) => d.label);
  return merged;
}

/* A colour ramp from the page background to the article's accent. Anomalous is
 * dark, ordinary is nearly invisible, which keeps the points readable on top. */
export function ramp(t) {
  const a = d3.color('#f1f3f3');
  const b = d3.color(accent());
  const u = Math.max(0, Math.min(1, t));
  return d3.interpolateRgb(a, b)(u ** 1.25);
}

/* Paint a field of values, given as a row-major grid of `n` by `n` samples over
 * the same domain the scales cover, as a smoothed image behind the plot. */
export function scoreField(g, values, n, x, y, w, h, { lo = null, hi = null, opacity = 0.9 } = {}) {
  const min = lo === null ? Math.min(...values) : lo;
  const max = hi === null ? Math.max(...values) : hi;
  const cv = document.createElement('canvas');
  cv.width = n;
  cv.height = n;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(n, n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      /* the grid runs bottom to top in data space and top to bottom in image
         space, so the row index is flipped once, here, and nowhere else */
      const v = values[(n - 1 - r) * n + c];
      const col = d3.color(ramp(max > min ? (v - min) / (max - min) : 0));
      const k = (r * n + c) * 4;
      img.data[k] = col.r;
      img.data[k + 1] = col.g;
      img.data[k + 2] = col.b;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  let node = g.select('image.field');
  if (node.empty()) node = g.insert('image', ':first-child').attr('class', 'field');
  node
    .attr('x', 0).attr('y', 0).attr('width', w).attr('height', h)
    .attr('preserveAspectRatio', 'none')
    .attr('opacity', opacity)
    .attr('href', cv.toDataURL());
  return { min, max };
}

/* A colour bar for the field above, in the top margin where nothing is drawn. */
export function fieldLegend(g, x0, y0, width, min, max, label) {
  const grp = g.append('g').attr('class', 'field-legend');
  const steps = 40;
  for (let i = 0; i < steps; i++) {
    grp.append('rect')
      .attr('x', x0 + (i * width) / steps).attr('y', y0)
      .attr('width', width / steps + 0.6).attr('height', 9)
      .attr('fill', ramp(i / (steps - 1)));
  }
  grp.append('rect').attr('x', x0).attr('y', y0).attr('width', width).attr('height', 9)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-opacity', 0.3);
  seriesLabel(grp, x0 - 6, y0 + 8, min.toFixed(2), { anchor: 'end', size: 10 });
  seriesLabel(grp, x0 + width + 6, y0 + 8, max.toFixed(2), { size: 10 });
  if (label) seriesLabel(grp, x0 + width / 2, y0 - 6, label, { anchor: 'middle', size: 10 });
  return grp;
}

/* The points. Anomalies are rings so that the truth is readable through
 * whatever the detector has painted underneath. */
export function drawPoints(g, points, labels, x, y, {
  r = 3.2, klass = 'pt', fill = () => 'var(--squidink)', highlight = null,
} = {}) {
  const sel = g.selectAll(`g.${klass}`).data(points);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', klass);
  enter.append('circle');
  const merged = enter.merge(sel);
  merged.attr('transform', (p) => `translate(${x(p[0])},${y(p[1])})`);
  merged.select('circle')
    .attr('r', (p, i) => (labels && labels[i] ? r * 1.7 : r))
    .attr('fill', (p, i) => (labels && labels[i] ? 'none' : fill(i)))
    .attr('stroke', (p, i) => (labels && labels[i] ? 'var(--cosmos)' : 'var(--paper)'))
    .attr('stroke-width', (p, i) => (labels && labels[i] ? 2 : 0.8))
    .attr('opacity', (p, i) => (highlight && !highlight(i) ? 0.35 : 0.95));
  return merged;
}
