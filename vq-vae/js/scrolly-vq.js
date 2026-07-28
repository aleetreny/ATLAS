/* Quantisation, on the plane, at three rates.
 *
 * The points are the module's two dimensional truth; the crosses are the
 * codebook the generator fitted at that size; the colours are the Voronoi cell
 * each point falls into. The last step draws the separable quantiser at the
 * same total budget, which is the only honest baseline: the same bits, spent
 * one coordinate at a time.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initScrollyVq(data) {
  const S = data.toy.showcase;
  const pts = S.points;
  const chart = makeChart('#vq-chart', {
    width: 540, height: 470, margin: { top: 52, right: 24, bottom: 54, left: 56 },
  });
  const { g, w, h } = chart;
  const scales = equalScales(pts, w, h);
  const x = scales.x;
  const y = scales.y;

  const dots = g.append('g');
  const marks = g.append('g');
  const lines = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -30).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const KS = data.toy.showcase.assign_ks;
  const books = {};
  KS.forEach((k) => {
    const row = data.toy.rows.find((r) => r.k === k);
    books[k] = row ? row.codebook : [];
  });
  const shade = d3.scaleSequential(d3.interpolateWarm);

  function render(step) {
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: 'first coordinate', y: 'second coordinate' });

    const k = step === 0 ? null : KS[Math.min(step, KS.length) - 1];
    const assign = k ? S.assign[String(k)] : null;
    dots.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 2.1)
      .attr('fill', (p, i) => (assign ? shade((assign[i] % 11) / 11) : 'var(--squidink)'))
      .attr('opacity', assign ? 0.72 : 0.5);

    const book = k && step <= KS.length ? books[k] || [] : [];
    marks.selectAll('path').data(book).join('path')
      .attr('d', d3.symbol().type(d3.symbolCross).size(64))
      .attr('transform', (p) => `translate(${x(p[0])},${y(p[1])}) rotate(45)`)
      .attr('fill', 'var(--primary)').attr('stroke', 'var(--white)').attr('stroke-width', 1);

    /* the separable quantiser: a grid of lines, which is exactly what it is */
    const wantGrid = step === 4;
    lines.selectAll('line')
      .data(wantGrid ? S.levels_x.map((v) => ['x', v]).concat(S.levels_y.map((v) => ['y', v])) : [])
      .join('line')
      .attr('x1', (d) => (d[0] === 'x' ? x(d[1]) : 0))
      .attr('x2', (d) => (d[0] === 'x' ? x(d[1]) : w))
      .attr('y1', (d) => (d[0] === 'x' ? 0 : y(d[1])))
      .attr('y2', (d) => (d[0] === 'x' ? h : y(d[1])))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1).attr('opacity', 0.8);

    const labels = [
      `${pts.length} points from the density this module is fitted against`,
      `${KS[0]} entries: ${Math.round(Math.log2(KS[0]))} bits per point`,
      `${KS[1]} entries: ${Math.round(Math.log2(KS[1]))} bits`,
      `${KS[2]} entries: ${Math.round(Math.log2(KS[2]))} bits`,
      `the same ${S.bits} bits spent ${S.split[0]} and ${S.split[1]} on the two axes`,
    ];
    wrapLabel(title, labels[step], w - 8);
  }

  render(0);
  const handlers = {};
  for (let i = 0; i < 5; i++) handlers[i] = () => render(i);
  const ctl = scrolly('#vq-scrolly .step', handlers);
  return { render, goTo: ctl.goTo };
}
