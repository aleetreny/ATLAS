/* The last unlit chip's story, in five steps: the labelled moons, what a
 * linear projection can and cannot do to them (measured over every angle),
 * and the kernel discriminant separating them cleanly. Scatter above,
 * projection histogram below, in one chart.
 */
import { drawGrid, drawAxes, makeChart, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { kdaScores, project, bestCut, histo } from './kdakit.js';

const C0 = 'var(--aqua)';
const C1 = 'var(--cosmos)';

export function initScrollyAngle(data) {
  const X = data.x_train;
  const y = data.y_train;
  const { g, w, h } = makeChart('#angle-chart', {
    width: 620,
    height: 600,
    margin: { top: 34, right: 26, bottom: 24, left: 50 },
  });
  const sh = h * 0.62;                       // scatter height
  const hy0 = sh + 64;                       // histogram top
  const hh = h - hy0 - 6;

  const x = d3.scaleLinear().domain(d3.extent(X, (p) => p[0])).nice().range([0, w]);
  const ys = d3.scaleLinear().domain(d3.extent(X, (p) => p[1])).nice().range([sh, 0]);
  drawGrid(g, x, ys, w, sh, { xTicks: 5, yTicks: 4 });
  drawAxes(g, x, ys, w, sh, { xTicks: 5, yTicks: 4 });

  const fieldG = g.append('g');
  const dotG = g.append('g');
  const arrowG = g.append('g');
  const histG = g.append('g').attr('transform', `translate(0,${hy0})`);
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.64rem');
  /* three of the step captions are longer than the canvas; they wrap by
     measure and climb a lane so the second line stays inside the margin */
  const setCaption = (t) => {
    const lines = wrapLabel(caption, t, w - 8);
    caption.attr('y', lines > 1 ? -25 : -14);
  };
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', hy0 - 10).attr('text-anchor', 'middle')
    .style('font-size', '0.54rem').text('the projection, seen in 1-D');

  function dots(faded) {
    dotG.selectAll('circle').data(X).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => ys(p[1]))
      .attr('r', 3.4)
      .attr('fill', (p, i) => (y[i] ? C1 : C0))
      .attr('stroke', 'white').attr('stroke-width', 0.6)
      .attr('opacity', faded ? 0.35 : 0.85);
  }

  function drawHist(z, show) {
    if (!show) { histG.selectAll('*').remove(); return; }
    const lo = Math.min(...z);
    const hi = Math.max(...z);
    const NB = 34;
    const { b0, b1 } = histo(z, y, lo, hi + 1e-9, NB);
    const hx = d3.scaleLinear().domain([0, NB]).range([0, w]);
    const maxC = Math.max(...b0, ...b1);
    const hyy = d3.scaleLinear().domain([0, maxC]).range([hh, 0]);
    const bw = w / NB;
    histG.selectAll('rect.c0').data(b0).join('rect').attr('class', 'c0')
      .attr('x', (v, i) => hx(i) + 1).attr('width', bw - 2)
      .attr('y', (v) => hyy(v)).attr('height', (v) => hh - hyy(v))
      .attr('fill', C0).attr('opacity', 0.62);
    histG.selectAll('rect.c1').data(b1).join('rect').attr('class', 'c1')
      .attr('x', (v, i) => hx(i) + 1).attr('width', bw - 2)
      .attr('y', (v) => hyy(v)).attr('height', (v) => hh - hyy(v))
      .attr('fill', C1).attr('opacity', 0.62);
    histG.selectAll('line.base').data([1]).join('line').attr('class', 'base')
      .attr('x1', 0).attr('x2', w).attr('y1', hh).attr('y2', hh)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
  }

  function arrow(theta, show) {
    const cx = w / 2;
    const cy = sh / 2;
    const L = 90;
    arrowG.selectAll('line').data(show ? [1] : []).join('line')
      .attr('x1', cx - L * Math.cos(theta)).attr('y1', cy + L * Math.sin(theta))
      .attr('x2', cx + L * Math.cos(theta)).attr('y2', cy - L * Math.sin(theta))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 2.6)
      .attr('marker-end', null).attr('opacity', 0.85);
  }

  function field(gamma, show) {
    if (!show) { fieldG.selectAll('*').remove(); return; }
    const krow = data.kda.find((r) => r.gamma === gamma);
    const NG = 55;
    const xs = d3.range(NG).map((i) => x.domain()[0] + (i + 0.5) / NG * (x.domain()[1] - x.domain()[0]));
    const yv = d3.range(NG).map((j) => ys.domain()[0] + (j + 0.5) / NG * (ys.domain()[1] - ys.domain()[0]));
    const pts = [];
    for (const yy of yv) for (const xx of xs) pts.push([xx, yy]);
    const z = kdaScores(pts, X, krow.alpha, gamma);
    const cw = w / NG;
    const ch = sh / NG;
    fieldG.selectAll('rect').data(pts).join('rect')
      .attr('x', (p) => x(p[0]) - cw / 2).attr('y', (p) => ys(p[1]) - ch / 2)
      .attr('width', cw + 0.5).attr('height', ch + 0.5)
      .attr('fill', (p, i) => (z[i] > krow.b ? C1 : C0))
      .attr('opacity', 0.13);
  }

  const ldaTheta = Math.atan2(data.lda.w[1], data.lda.w[0]);
  const states = {
    0: () => {
      dots(false); drawHist(null, false); arrow(0, false); field(0, false);
      setCaption('the moons again, labels restored: 210 training points, two classes');
    },
    1: () => {
      dots(false); arrow(ldaTheta, true); field(0, false);
      const z = project(X, ldaTheta);
      drawHist(z, true);
      setCaption(`LDA's direction: project everything onto the line, and the classes pile into each other`);
    },
    2: () => {
      dots(false); arrow(data.best_line.theta, true); field(0, false);
      drawHist(project(X, data.best_line.theta), true);
      setCaption(`the BEST of 721 angles ever measured here: ${data.best_line.acc_test} test accuracy, the linear ceiling`);
    },
    3: () => {
      dots(false); arrow(0, false); field(0, false);
      const kr = data.kda.find((r) => r.gamma === 0.3);
      drawHist(Array.from(kdaScores(X, X, kr.alpha, 0.3)), true);
      setCaption('the kernel direction: the same projection idea, taken in feature space, separates them');
    },
    4: () => {
      dots(false); arrow(0, false); field(0.3, true);
      const kr = data.kda.find((r) => r.gamma === 0.3);
      drawHist(Array.from(kdaScores(X, X, kr.alpha, 0.3)), true);
      setCaption(`seen back in 2-D, that direction is a curved boundary: test accuracy ${kr.acc_test}`);
    },
  };

  scrolly('#angle-scrolly .step', states);
}
