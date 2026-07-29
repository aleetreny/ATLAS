/* The simplex walk over the same polytope, one pivot per step.
 *
 * The path is not read out of the file: it comes from the simplex in lpkit.js,
 * which is the same algorithm the generator wrote in Python, and the guard
 * checks the two agree corner by corner.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { enumerateVertices, ring, simplex } from './lpkit.js';

export function initSimplexScrolly(data) {
  const P = data.polytope;
  const A = P.rows.map((r) => r.a);
  const b = P.rows.map((r) => r.b);
  const c = P.profit;
  const A2 = A.concat([[-1, 0], [0, -1]]);
  const b2 = b.concat([0, 0]);
  const R = ring(enumerateVertices(A2, b2).vertices);
  const run = simplex(c, A, b);

  const chart = makeChart('#simplex-chart', {
    width: 640, height: 430, margin: { top: 52, right: 30, bottom: 54, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().domain([-0.4, 4.7]).range([0, w]);
  const y = d3.scaleLinear().domain([-0.3, 3.0]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'lathe product, x', y: 'press product, y', leftBudget: margin.left });

  g.append('path')
    .attr('d', `M${R.map((p) => `${x(p.x)},${y(p.y)}`).join('L')}Z`)
    .attr('fill', 'var(--primary)').attr('fill-opacity', 0.09)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4);

  const corners = g.selectAll('circle.c').data(R).join('circle').attr('class', 'c')
    .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 5.5)
    .attr('fill', '#ffffff').attr('stroke', 'var(--primary)').attr('stroke-width', 1.6);
  const cornerText = g.selectAll('text.cv').data(R).join('text').attr('class', 'chart-note cv')
    .attr('x', (p) => x(p.x) + (p.x > 3.2 ? -9 : 9))
    .attr('y', (p) => y(p.y) + (p.y < 0.2 ? 16 : -9))
    .attr('text-anchor', (p) => (p.x > 3.2 ? 'end' : 'start'))
    .style('font-size', '11px').style('opacity', 0)
    .text((p) => (5 * p.x + 4 * p.y).toFixed(1));

  const walk = g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)')
    .attr('stroke-width', 3).attr('stroke-linejoin', 'round');
  const here = g.append('circle').attr('r', 7.5).attr('fill', 'var(--cosmos)').style('opacity', 0);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', 0).attr('y', -30).style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note')
    .attr('x', 0).attr('y', -12).style('font-size', '11.5px');

  function draw(k, { showValues = false, dim = false } = {}) {
    const pts = run.path.slice(0, k + 1);
    walk.transition('walk').duration(600)
      .attr('d', pts.length > 1
        ? `M${pts.map((p) => `${x(p[0])},${y(p[1])}`).join('L')}`
        : `M${x(pts[0][0])},${y(pts[0][1])}L${x(pts[0][0])},${y(pts[0][1])}`);
    const last = pts[pts.length - 1];
    here.style('opacity', 1).transition('here').duration(600)
      .attr('cx', x(last[0])).attr('cy', y(last[1]));
    cornerText.transition('cv').duration(400).style('opacity', showValues ? 1 : 0);
    corners.transition('cc').duration(400)
      .attr('fill', (p) => {
        const seen = pts.some((q) => Math.abs(q[0] - p.x) < 1e-9 && Math.abs(q[1] - p.y) < 1e-9);
        if (dim) return seen ? 'var(--cosmos)' : '#ffffff';
        return seen ? 'var(--cosmos)' : '#ffffff';
      })
      .attr('stroke', (p) => {
        const seen = pts.some((q) => Math.abs(q[0] - p.x) < 1e-9 && Math.abs(q[1] - p.y) < 1e-9);
        return dim && !seen ? 'var(--stone)' : (seen ? 'var(--cosmos)' : 'var(--primary)');
      });
    const value = 5 * last[0] + 4 * last[1];
    title.text(k === 0
      ? 'pivot 0: build nothing, profit 0'
      : `pivot ${k}: profit ${value.toFixed(1)}`);
    sub.text(dim
      ? `${pts.length} of the ${R.length} corners were ever visited`
      : `at (${last[0].toFixed(1)}, ${last[1].toFixed(1)})`);
  }

  const handlers = {
    0: () => draw(0),
    1: () => draw(1),
    2: () => draw(2, { showValues: true }),
    3: () => draw(run.path.length - 1, { showValues: true, dim: true }),
  };
  draw(0);
  const ctrl = scrolly('#simplex-scrolly .step', handlers);
  return { ...ctrl, run };
}
