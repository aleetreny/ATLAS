/* ATLAS chart helpers — shared d3 conventions (axis styling, labels,
 * annotations) so every article's charts look like one system.
 * Assumes d3 v7 is loaded globally (vendored, see assets/js/vendor/).
 */

/* Create a responsive SVG inside `container` (selector or node) with a fixed
 * internal coordinate system via viewBox. Returns {svg, g, w, h} where g is
 * the inner group translated by margins and w/h the inner drawing size. */
export function makeChart(container, { width = 640, height = 460, margin = { top: 32, right: 24, bottom: 52, left: 60 } } = {}) {
  const svg = d3
    .select(typeof container === 'string' ? document.querySelector(container) : container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  return { svg, g, w, h, margin, width, height };
}

/* Bottom + left axes with the ATLAS look. Call again on scale change. */
export function drawAxes(g, x, y, w, h, { xTicks = 6, yTicks = 6, xFmt = null, yFmt = null } = {}) {
  let gx = g.select('g.axis.x');
  if (gx.empty()) gx = g.append('g').attr('class', 'axis x');
  gx.attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(xTicks).tickFormat(xFmt).tickSizeOuter(0));

  let gy = g.select('g.axis.y');
  if (gy.empty()) gy = g.append('g').attr('class', 'axis y');
  gy.call(d3.axisLeft(y).ticks(yTicks).tickFormat(yFmt).tickSizeOuter(0));
}

/* Faint background grid (MLU: stroke-opacity .075). */
export function drawGrid(g, x, y, w, h, { xTicks = 6, yTicks = 6 } = {}) {
  let gg = g.select('g.grid');
  if (gg.empty()) gg = g.insert('g', ':first-child').attr('class', 'grid');
  gg.selectAll('*').remove();
  gg.append('g')
    .selectAll('line')
    .data(x.ticks(xTicks))
    .join('line')
    .attr('x1', (d) => x(d))
    .attr('x2', (d) => x(d))
    .attr('y1', 0)
    .attr('y2', h);
  gg.append('g')
    .selectAll('line')
    .data(y.ticks(yTicks))
    .join('line')
    .attr('x1', 0)
    .attr('x2', w)
    .attr('y1', (d) => y(d))
    .attr('y2', (d) => y(d));
}

/* Uppercase axis labels, MLU style. */
export function axisLabels(g, w, h, { x = '', y = '' } = {}) {
  if (x) {
    let lx = g.select('text.axis-label.x');
    if (lx.empty()) lx = g.append('text').attr('class', 'axis-label x');
    lx.attr('x', w / 2).attr('y', h + 42).attr('text-anchor', 'middle').text(x);
  }
  if (y) {
    let ly = g.select('text.axis-label.y');
    if (ly.empty()) ly = g.append('text').attr('class', 'axis-label y');
    ly.attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle').text(y);
  }
}

/* Chart annotation text with white halo (paint-order trick). */
export function annotate(g, x, y, text, { anchor = 'start' } = {}) {
  return g
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', x)
    .attr('y', y)
    .attr('text-anchor', anchor)
    .text(text);
}

/* Singleton tooltip div. */
let tooltipDiv = null;
export function tooltip() {
  if (!tooltipDiv) {
    tooltipDiv = d3.select('body').append('div').attr('class', 'atlas-tooltip');
  }
  return {
    show(html, event) {
      tooltipDiv
        .html(html)
        .style('left', `${event.pageX + 12}px`)
        .style('top', `${event.pageY - 10}px`)
        .style('opacity', 1);
    },
    hide() {
      tooltipDiv.style('opacity', 0);
    },
  };
}

/* Ordinary least squares on arrays (used by several widgets).
 * Returns {intercept, slope}. */
export function ols1d(xs, ys) {
  const n = xs.length;
  const mx = d3.mean(xs);
  const my = d3.mean(ys);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = num / den;
  return { intercept: my - slope * mx, slope };
}

/* R^2 of predictions vs truth. */
export function r2(yTrue, yPred) {
  const my = d3.mean(yTrue);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < yTrue.length; i++) {
    ssRes += (yTrue[i] - yPred[i]) ** 2;
    ssTot += (yTrue[i] - my) ** 2;
  }
  return 1 - ssRes / ssTot;
}
