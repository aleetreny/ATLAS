/* Scrollytelling: a degree-15 polynomial on 20 points, and what a penalty does
 * to it. The chart carries two panels, the fit and the coefficient spectrum,
 * because overfitting is visible in both geometry and arithmetic. */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { makeModel, polyDesign, nonZero } from './mathkit.js';

const DUR = 700;
const truth = (x) => Math.sin(x) + x / 2;

/* symlog: keeps a coefficient of 100000 and one of 0.3 on the same axis */
const symlog = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v));

export function initScrollyMonster(sine) {
  const xs = sine.points.map((d) => d.x);
  const ys = sine.points.map((d) => d.y);
  const X = polyDesign(xs, sine.degree, sine.poly_mu, sine.poly_sd);
  const model = makeModel(X, ys);
  const yMean = d3.mean(ys);

  const grid = d3.range(0, 5.001, 0.02);
  const fitCurve = (b) => {
    const Xg = polyDesign(grid, sine.degree, sine.poly_mu, sine.poly_sd);
    return grid.map((x, i) => [x, yMean + Xg[i].reduce((s, v, j) => s + v * b[j], 0)]);
  };

  // The coefficient spectrum lives in the TOP margin, above the plot.
  // On narrow screens the step cards overlay the lower third of the sticky
  // chart, and one of the steps asks the reader to look at these bars, so they
  // cannot live down there. Reading order also improves: here are the weights,
  // and here is the curve they produce.
  const { g, w, h } = makeChart('#monster-chart', {
    width: 620,
    height: 660,
    margin: { top: 165, right: 24, bottom: 62, left: 58 },
  });

  const x = d3.scaleLinear().domain([-0.15, 5.15]).range([0, w]);
  const y = d3.scaleLinear().domain([-1.6, 4.6]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'y' });

  // clip: a monster curve must not paint over the whole page
  g.append('clipPath').attr('id', 'monster-clip').append('rect').attr('width', w).attr('height', h);
  const plot = g.append('g').attr('clip-path', 'url(#monster-clip)');

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);

  // ground truth, always visible: the thing we are trying to recover
  plot
    .append('path')
    .attr('d', line(grid.map((v) => [v, truth(v)])))
    .attr('fill', 'none')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', '7 6')
    .attr('opacity', 0.45);

  const haloPath = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7).attr('opacity', 0);
  const fitPath = plot.append('path').attr('fill', 'none').attr('stroke-width', 3.5).attr('opacity', 0);

  g.append('g')
    .selectAll('circle')
    .data(sine.points)
    .join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 5)
    .attr('fill', 'var(--squidink)')
    .attr('stroke', 'white')
    .attr('stroke-width', 1.4);

  const readout = g
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w / 2)
    .attr('y', -145)
    .attr('text-anchor', 'middle');

  // ---------- coefficient spectrum panel (in the top margin) ----------
  const specTop = -73;
  const specH = 40;
  const spec = g.append('g').attr('transform', `translate(0,${specTop})`);
  const bx = d3.scaleBand().domain(d3.range(sine.degree)).range([0, w]).padding(0.25);
  const by = d3.scaleLinear().domain([-5.2, 5.2]).range([specH, -specH]);

  spec
    .append('line')
    .attr('x1', 0).attr('x2', w)
    .attr('y1', by(0)).attr('y2', by(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5);

  spec
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', 0).attr('y', specH + 20)
    .style('font-size', '0.68rem')
    .text('coefficient size (symlog)');

  const bars = spec
    .selectAll('rect')
    .data(new Array(sine.degree).fill(0))
    .join('rect')
    .attr('x', (d, i) => bx(i))
    .attr('width', bx.bandwidth())
    .attr('y', by(0))
    .attr('height', 0)
    .attr('fill', 'var(--primary)');

  const specNote = spec
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w)
    .attr('y', specH + 20)
    .attr('text-anchor', 'end');

  const t = () => d3.transition().duration(DUR).ease(d3.easeCubicOut);

  // the "unpenalized" reference, and a check that this browser resolves it the
  // same way scikit-learn did (see the generator script for why 1e-9)
  const BARE = sine.ref.ols_alpha;
  const bareMax = d3.max(model.fit(BARE, 0), (v) => Math.abs(v));
  if (Math.abs(bareMax - sine.ref.ols_max_abs) / sine.ref.ols_max_abs > 0.02) {
    console.warn(`bare fit: JS ${bareMax.toFixed(1)} vs sklearn ${sine.ref.ols_max_abs}`);
  }

  function show({ alpha, l1Ratio = 0, color = 'var(--anchor)', label, hideFit = false }) {
    if (hideFit) {
      haloPath.transition(t()).attr('opacity', 0);
      fitPath.transition(t()).attr('opacity', 0);
      bars.transition(t()).attr('y', by(0)).attr('height', 0);
      specNote.text('');
      readout.text(label);
      return;
    }
    const b = model.fit(alpha, l1Ratio);
    const d = line(fitCurve(b));
    haloPath.transition(t()).attr('d', d).attr('opacity', 1);
    fitPath.transition(t()).attr('d', d).attr('stroke', color).attr('opacity', 1);

    bars
      .data(b)
      .transition(t())
      .attr('fill', color)
      .attr('y', (v) => (v >= 0 ? by(symlog(v)) : by(0)))
      .attr('height', (v) => Math.abs(by(symlog(v)) - by(0)));

    const maxAbs = d3.max(b, (v) => Math.abs(v));
    const shrunk = bareMax / maxAbs;
    specNote.text(
      `max |coef| ${maxAbs >= 100 ? d3.format(',.0f')(maxAbs) : maxAbs.toFixed(2)}` +
        (alpha > BARE * 10 ? ` · ${d3.format(',.0f')(shrunk)}× smaller` : '')
    );
    readout.text(label);
  }

  const handlers = {
    0: () => show({ hideFit: true, label: 'the truth we are trying to recover: y = sin(x) + x/2' }),
    1: () => show({ alpha: BARE, color: 'var(--cosmos)', label: 'degree 15, no penalty' }),
    2: () => show({ alpha: BARE, color: 'var(--cosmos)', label: 'the smoking gun is in the coefficients' }),
    // alphas chosen from the actual error-against-truth curve, so the story is
    // monotonic: 0.82 -> 0.25 -> 0.17 (the optimum) -> 0.46 (overdose)
    3: () => show({ alpha: 1e-5, color: 'var(--anchor)', label: 'ridge, a small toll on size' }),
    4: () => show({ alpha: 3.16e-4, color: 'var(--anchor)', label: 'ridge, the best this data allows' }),
    5: () => show({ alpha: 3, color: 'var(--anchor)', label: 'overdose: nothing left to say' }),
  };

  handlers[0]();
  scrolly('#monster-scrolly .step', handlers);
}
