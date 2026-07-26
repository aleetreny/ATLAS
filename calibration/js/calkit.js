/* Shared pieces for the calibration article: reliability drawing and the
 * pool-adjacent-violators algorithm itself, implemented so the page can run
 * isotonic regression live and check itself against scikit-learn.
 */

/* The diagonal of honesty, with its label. */
export function drawDiagonal(g, x, y, w, h) {
  g.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(1))
    .attr('stroke', 'var(--stone)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(0.62)).attr('y', y(0.62) - 10)
    .attr('transform', `rotate(${-Math.atan2(h, w) * 180 / Math.PI} ${x(0.62)} ${y(0.62) - 10})`)
    .style('font-size', '0.54rem').style('text-transform', 'none')
    .text('honesty: claimed = true');
}

/* One reliability curve: line through the bins plus population-sized dots. */
export function drawReliability(layer, bins, x, y, colour, opts = {}) {
  const rDot = (n) => 2.5 + 2.4 * Math.log10(Math.max(n, 10) / 10);
  const line = d3.line().x((b) => x(b.predicted)).y((b) => y(b.observed));
  layer.selectAll('path').data([bins]).join('path')
    .attr('fill', 'none').attr('stroke', colour)
    .attr('stroke-width', opts.width || 2.6).attr('opacity', opts.opacity || 1)
    .attr('d', line(bins));
  layer.selectAll('circle').data(bins).join('circle')
    .attr('cx', (b) => x(b.predicted)).attr('cy', (b) => y(b.observed))
    .attr('r', (b) => (opts.dots === false ? 0 : rDot(b.n)))
    .attr('fill', colour).attr('opacity', opts.opacity || 1)
    .attr('stroke', 'white').attr('stroke-width', 1.2);
}

/* Pool-adjacent-violators, block by block, exposed step-wise so the widget
 * can animate every merge. Ties in x are pooled first, exactly as
 * scikit-learn does, so the two implementations answer identically.
 */
export function pavBlocks(scores, y) {
  const blocks = [];
  for (let i = 0; i < scores.length; i++) {
    const last = blocks[blocks.length - 1];
    if (last && last.x1 === scores[i]) {
      /* Same score: pool immediately, it is one x-position. */
      last.sum += y[i];
      last.n += 1;
      last.hi = i;
      last.v = last.sum / last.n;
    } else {
      blocks.push({ x0: scores[i], x1: scores[i], lo: i, hi: i, sum: y[i], n: 1, v: y[i] });
    }
  }
  return blocks;
}

/* One PAV step: merge the first adjacent violating pair. Returns false when
 * the sequence is already monotone. */
export function pavStep(blocks) {
  for (let i = 0; i + 1 < blocks.length; i++) {
    if (blocks[i].v > blocks[i + 1].v + 1e-12) {
      const a = blocks[i];
      const b = blocks[i + 1];
      blocks.splice(i, 2, {
        x0: a.x0, x1: b.x1, lo: a.lo, hi: b.hi,
        sum: a.sum + b.sum, n: a.n + b.n, v: (a.sum + b.sum) / (a.n + b.n),
      });
      return true;
    }
  }
  return false;
}

/* Run to completion; returns the fitted value for every input point. */
export function pavFit(scores, y) {
  const blocks = pavBlocks(scores, y);
  while (pavStep(blocks)) { /* merge until monotone */ }
  const fit = new Array(scores.length);
  for (const b of blocks) {
    for (let i = b.lo; i <= b.hi; i++) fit[i] = b.v;
  }
  return { fit, blocks };
}
