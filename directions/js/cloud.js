/* Shared drawing for the two-dimensional cloud: the scrolly and the dial are
 * the same picture, one told as a story and one handed to the reader.
 *
 * The one rule this file exists to enforce is EQUAL ASPECT. The whole section
 * asks the reader to judge an angle by eye and then reports a number for it,
 * so a degree across the screen has to be a degree in the data. Two
 * independently fitted d3 scales would stretch the cloud to fill the panel and
 * quietly make every angle on screen a lie.
 */

/* Exactly centred coordinates, memoised per array.
 *
 * The shipped points are rounded to four decimals, which moves their mean off
 * zero by about 1e-5. Every claim in this section assumes the cloud is
 * centred: the projection's variance is measured about its own mean while the
 * residual is measured from a line through the origin, so a mean of m leaves
 * captured + thrown away short of the total by exactly m squared. That is
 * 1e-9 here, invisible on screen, and enough to stop the identity guard from
 * being an identity guard. Subtract the mean once and it is exact again.
 */
const centreCache = new WeakMap();
export function centred(points) {
  if (centreCache.has(points)) return centreCache.get(points);
  let mx = 0;
  let my = 0;
  for (const [a, b] of points) {
    mx += a;
    my += b;
  }
  mx /= points.length;
  my /= points.length;
  const out = points.map(([a, b]) => [a - mx, b - my]);
  centreCache.set(points, out);
  return out;
}

/* Scales with identical units per pixel, centred on the data. */
export function equalScales(points, w, h, { pad = 1.08 } = {}) {
  const xs = points.map((d) => d[0]);
  const ys = points.map((d) => d[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const spanX = (Math.max(...xs) - Math.min(...xs)) * pad;
  const spanY = (Math.max(...ys) - Math.min(...ys)) * pad;
  const perPixel = Math.max(spanX / w, spanY / h);
  const halfX = (perPixel * w) / 2;
  const halfY = (perPixel * h) / 2;
  return {
    x: d3.scaleLinear().domain([cx - halfX, cx + halfX]).range([0, w]),
    y: d3.scaleLinear().domain([cy - halfY, cy + halfY]).range([h, 0]),
    perPixel,
  };
}

/* Unit vector for an angle measured anticlockwise from the x axis. */
export function unit(deg) {
  const r = (deg * Math.PI) / 180;
  return [Math.cos(r), Math.sin(r)];
}

/* Everything the readouts need about one direction through one cloud.
 * captured + lost is total at every angle, and that is the point, so it is
 * computed rather than assumed: `total` here is the sum of both, and the
 * caller compares it against the cloud's own total variance. */
export function project(points, deg) {
  const [ux, uy] = unit(deg);
  const t = points.map(([x, y]) => x * ux + y * uy);
  const m = t.reduce((s, v) => s + v, 0) / t.length;
  let captured = 0;
  let lost = 0;
  for (let i = 0; i < points.length; i++) {
    captured += (t[i] - m) ** 2;
    const rx = points[i][0] - t[i] * ux;
    const ry = points[i][1] - t[i] * uy;
    lost += rx * rx + ry * ry;
  }
  return {
    t,
    shadows: t.map((v) => [v * ux, v * uy]),
    captured: captured / points.length,
    lost: lost / points.length,
  };
}

/* The variance budget as one stacked bar, drawn in the chart's TOP margin.
 * Below the plot it would collide with the x axis, and on a phone the step
 * card would sit on top of it. */
export function budgetBar(g, w, { y = -40, height = 15 } = {}) {
  const box = g.append('g').attr('class', 'budget').attr('transform', `translate(0,${y})`);
  box.append('rect').attr('class', 'kept')
    .attr('height', height).attr('fill', 'var(--primary)');
  box.append('rect').attr('class', 'gone')
    .attr('height', height).attr('fill', 'var(--stone)');
  box.append('rect').attr('width', w).attr('height', height)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
  box.append('text').attr('class', 'kept-label chart-annotation')
    .attr('y', -6).style('font-size', '11px').style('letter-spacing', '1px');
  box.append('text').attr('class', 'gone-label chart-annotation')
    .attr('y', -6).attr('text-anchor', 'end').attr('x', w)
    .style('font-size', '11px').style('letter-spacing', '1px');
  return {
    update(captured, total, { animate = false } = {}) {
      const frac = total > 0 ? captured / total : 0;
      const kept = box.select('rect.kept');
      const gone = box.select('rect.gone');
      /* A named transition, because two unnamed ones on the same nodes cancel
         each other and the second silently does nothing. */
      (animate ? kept.transition('budget').duration(700) : kept)
        .attr('width', Math.max(0, frac * w));
      (animate ? gone.transition('budget').duration(700) : gone)
        .attr('x', frac * w).attr('width', Math.max(0, (1 - frac) * w));
      box.select('text.kept-label').text(`kept ${(frac * 100).toFixed(2)}%`);
      box.select('text.gone-label').text(`thrown away ${((1 - frac) * 100).toFixed(2)}%`);
    },
    node: box,
  };
}
