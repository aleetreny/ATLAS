/* Scrollytelling: the boosting loop, assembled one stump at a time.
 * The lower panel is the residual, which is the only thing each new tree ever
 * sees, and watching it shrink is watching the algorithm work. */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { makeData, boost, ensemblePredict, stumpPredict, TRUTH } from './mathkit.js';

const DUR = 650;

export function initScrollyBoost() {
  const { xs, ys } = makeData();
  const model = boost(xs, ys, { rounds: 120, lr: 0.25 });
  const grid = d3.range(0, 9.601, 0.05);

  const { g, w, h } = makeChart('#boost-chart', {
    width: 620,
    height: 600,
    margin: { top: 42, right: 26, bottom: 190, left: 58 },
  });

  const x = d3.scaleLinear().domain([-0.2, 9.8]).range([0, w]);
  const y = d3.scaleLinear().domain([-4.5, 9]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'y' });

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
  const stepLine = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveStepAfter);

  g.append('path').attr('d', line(grid.map((v) => [v, TRUTH(v)])))
    .attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2).attr('stroke-dasharray', '7 6').attr('opacity', 0.35);

  const dots = g.append('g').selectAll('circle').data(xs.map((v, i) => [v, ys[i]])).join('circle')
    .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.6)
    .attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.7);

  const halo = g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5).attr('opacity', 0);
  const fit = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4).attr('opacity', 0);

  const readout = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -18).attr('text-anchor', 'middle');

  /* ---- residual panel ---- */
  const resTop = h + 74;
  const resH = 78;
  const res = g.append('g').attr('transform', `translate(0,${resTop})`);
  const ry = d3.scaleLinear().domain([-5, 5]).range([resH, -resH]);
  res.append('line').attr('x1', 0).attr('x2', w).attr('y1', ry(0)).attr('y2', ry(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
  res.append('text').attr('class', 'chart-annotation').attr('x', 0).attr('y', -resH - 10)
    .style('font-size', '0.64rem').style('letter-spacing', '0.5px').style('text-transform', 'none')
    .text('what is still wrong: the residual each new tree is fitted to');
  const resDots = res.append('g').selectAll('circle').data(xs).join('circle')
    .attr('cx', (v) => x(v)).attr('r', 3).attr('fill', 'var(--cosmos)').attr('opacity', 0.75);
  const resFit = res.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.6).attr('opacity', 0);

  const t = () => d3.transition().duration(DUR).ease(d3.easeCubicOut);

  function show({ m, showStump = false, label }) {
    const pred = ensemblePredict(model, grid, m);
    const predAt = ensemblePredict(model, xs, m);
    if (m === 0) {
      const flat = grid.map((v) => [v, model.base]);
      halo.transition(t()).attr('d', line(flat)).attr('opacity', 1);
      fit.transition(t()).attr('d', line(flat)).attr('opacity', 1);
    } else {
      const d = stepLine(grid.map((v, i) => [v, pred[i]]));
      halo.transition(t()).attr('d', d).attr('opacity', 1);
      fit.transition(t()).attr('d', d).attr('opacity', 1);
    }

    const residuals = ys.map((v, i) => v - predAt[i]);
    resDots.transition(t()).attr('cy', (v, i) => ry(residuals[i]));

    if (showStump && model.history[m]) {
      const s = model.history[m].stump;
      resFit.transition(t())
        .attr('d', stepLine(grid.map((v) => [v, stumpPredict(s, v)])).replace(/NaN/g, '0'))
        .attr('opacity', 1);
    } else {
      resFit.transition(t()).attr('opacity', 0);
    }

    const rmseNow = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / residuals.length);
    readout.text(`${label} · rmse ${rmseNow.toFixed(2)}`);
  }

  const handlers = {
    0: () => show({ m: 0, label: 'round 0 · predict the mean and nothing else' }),
    1: () => show({ m: 0, showStump: true, label: 'fit one stump to the residual' }),
    2: () => show({ m: 1, label: 'round 1 · add a shrunken copy of it' }),
    3: () => show({ m: 5, label: 'round 5' }),
    4: () => show({ m: 25, label: 'round 25' }),
    5: () => show({ m: 120, label: 'round 120 · a staircase that learned a curve' }),
  };

  handlers[0]();
  scrolly('#boost-scrolly .step', handlers);
}
