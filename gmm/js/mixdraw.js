/* Drawing helpers for the mixtures article: soft-tinted points, mean markers
 * and covariance ellipses. The three cluster colours are the same ones the
 * k-means article used, on purpose: the reader has already learned them.
 */
import { ellipse } from './gmmkit.js';

/* Literal values of --aqua, --cosmos, --anchor: literal because responsibility
 * blending needs RGB arithmetic, same order as the k-means article's panels. */
export const COLOURS = ['#007faa', '#df2a5d', '#003181', '#ff9900', '#330066'];
const RGB = COLOURS.map((c) => {
  const m = /^#(..)(..)(..)$/.exec(c);
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
});

/* Blend the cluster colours by responsibility: the soft assignment IS the
 * colour. A point at 50/50 renders as the visual midpoint of two clusters. */
export function softColour(resp) {
  let r = 0;
  let g = 0;
  let b = 0;
  for (let c = 0; c < resp.length; c++) {
    r += resp[c] * RGB[c % RGB.length][0];
    g += resp[c] * RGB[c % RGB.length][1];
    b += resp[c] * RGB[c % RGB.length][2];
  }
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

export function drawSoftPoints(layer, points, resp, x, y, opts = {}) {
  layer.selectAll('circle').data(points).join('circle')
    .attr('cx', (p) => x(p[0]))
    .attr('cy', (p) => y(p[1]))
    .attr('r', opts.r || 3.2)
    .attr('fill', (p, i) => (resp ? softColour(resp[i]) : 'var(--stone)'))
    .attr('stroke', resp ? 'white' : 'var(--squidink)')
    .attr('stroke-width', 0.5)
    .attr('opacity', opts.opacity || 0.85);
}

export function drawMeans(layer, means, x, y) {
  layer.selectAll('rect').data(means).join('rect')
    .attr('width', 13).attr('height', 13)
    .attr('transform', (m) => `translate(${x(m[0])},${y(m[1])}) rotate(45)`)
    .attr('x', -6.5).attr('y', -6.5)
    .attr('fill', (m, i) => COLOURS[i % COLOURS.length])
    .attr('stroke', 'white').attr('stroke-width', 2);
}

/* One and two sigma outlines per component. */
export function drawEllipses(layer, model, x, y, opts = {}) {
  const line = d3.line().x((p) => x(p[0])).y((p) => y(p[1]));
  const data = [];
  model.means.forEach((m, c) => {
    for (const nsig of opts.sigmas || [1, 2]) {
      data.push({ c, nsig, pts: ellipse(m, model.covs[c], nsig) });
    }
  });
  layer.selectAll('path').data(data).join('path')
    .attr('fill', 'none')
    .attr('stroke', (d) => COLOURS[d.c % COLOURS.length])
    .attr('stroke-width', (d) => (d.nsig === 1 ? 2.2 : 1.3))
    .attr('stroke-dasharray', (d) => (d.nsig === 1 ? null : '5 4'))
    .attr('opacity', 0.8)
    .attr('d', (d) => line(d.pts));
}
