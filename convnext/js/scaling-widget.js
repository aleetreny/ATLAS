/* Where to spend a tripled budget: on depth, on width, on resolution, or on
 * the mixture EfficientNet's own exponents produce.
 *
 * Two things this chart refuses to hide. The horizontal axis is ACHIEVED
 * multiply-accumulates, not the requested budget, because integer block
 * counts and integer channels mean the arms do not land on the same number.
 * And the accuracy axis carries the measured noise band, because several of
 * the gaps between allocations are smaller than it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const NICE = {
  depth: 'all of it on depth',
  width: 'all of it on width',
  res: 'all of it on resolution',
  'depth-width': 'split: depth and width',
  'depth-res': 'split: depth and resolution',
  'width-res': 'split: width and resolution',
  efficientnet: "EfficientNet's own exponents",
};

export function initScalingWidget(data) {
  const arms = data.scaling.arms;
  const base = data.runs[data.scaling.base];
  const sigma = data.ladder.noise_floor;
  const W = 700;
  const H = 400;
  const margin = { top: 32, right: 30, bottom: 58, left: 72 };
  const { g, w, h } = makeChart('#scaling-chart', { width: W, height: H, margin });

  const pts = arms.map((a) => ({ ...a, label: NICE[a.axis] || a.axis }));
  const all = [base, ...pts];
  const x = d3.scaleLinear()
    .domain([Math.min(...all.map((p) => p.macs)) * 0.9, Math.max(...all.map((p) => p.macs)) * 1.06])
    .range([0, w]);
  const accs = all.map((p) => p.test_acc_mean);
  const pad = (Math.max(...accs) - Math.min(...accs)) * 0.22 + 0.004;
  const y = d3.scaleLinear().domain([Math.min(...accs) - pad, Math.max(...accs) + pad]).range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xTicks: 6, yTicks: 5,
    xFmt: (v) => `${(v / 1e6).toFixed(1)}`,
    yFmt: d3.format('.1%'),
  });
  axisLabels(g, w, h, { x: 'multiply-accumulates actually spent, in millions', y: 'test accuracy' });

  /* The base network, and a band of one noise floor around it: anything that
     lands inside is a tripled budget that bought nothing measurable. The
     generator decides whether a size model beats a flat pool and says which
     it used, so read that rather than assuming the size model won. */
  const sd = sigma.used === 'size model'
    ? sigma.a + sigma.b / Math.sqrt(base.macs / 1e6)
    : sigma.flat_pool_sd;
  const band = sigma.t_critical * sd;
  g.append('rect')
    .attr('x', 0).attr('width', w)
    .attr('y', y(base.test_acc_mean + band))
    .attr('height', Math.max(1, y(base.test_acc_mean - band) - y(base.test_acc_mean + band)))
    .attr('fill', '#8a94a6').attr('opacity', 0.13);
  g.append('line')
    .attr('x1', 0).attr('x2', w)
    .attr('y1', y(base.test_acc_mean)).attr('y2', y(base.test_acc_mean))
    .attr('stroke', '#232f3e').attr('stroke-dasharray', '5 4').attr('stroke-width', 1.4);
  g.append('text')
    .attr('x', 4).attr('y', y(base.test_acc_mean) - 8)
    .attr('class', 'annotation').style('font-size', '11px').style('fill', '#5b6572')
    .text(`the base network, and one noise floor either side of it`);

  const dots = g.append('g');
  const labels = g.append('g');

  function draw(sel) {
    dots.selectAll('circle')
      .data(pts)
      .join('circle')
      .attr('cx', (p) => x(p.macs))
      .attr('cy', (p) => y(p.test_acc_mean))
      .attr('r', (p, i) => (i === sel ? 8 : 5.5))
      .attr('fill', (p, i) => (i === sel ? 'var(--primary)' : 'var(--secondary)'))
      .attr('opacity', (p, i) => (i === sel ? 1 : 0.72))
      .style('cursor', 'pointer')
      .on('click', (ev, p) => draw(pts.indexOf(p)));

    dots.selectAll('.base')
      .data([base])
      .join('circle')
      .attr('class', 'base')
      .attr('cx', (p) => x(p.macs))
      .attr('cy', (p) => y(p.test_acc_mean))
      .attr('r', 6)
      .attr('fill', 'none').attr('stroke', '#232f3e').attr('stroke-width', 2);

    /* Allocations land close together in cost, so the labels alternate above
       and below their dot: side by side at the same height they overlap, and
       the two that overlap are the two the reader most wants to compare. */
    const order = [...pts].sort((a, b) => a.macs - b.macs).map((p) => p.axis);
    labels.selectAll('text')
      .data(pts)
      .join('text')
      .attr('x', (p) => x(p.macs))
      .attr('y', (p) => y(p.test_acc_mean) + (order.indexOf(p.axis) % 2 === 0 ? -13 : 20))
      .attr('text-anchor', 'middle')
      .attr('class', 'annotation')
      .style('font-size', '10px')
      .style('fill', (p, i) => (i === sel ? 'var(--primary)' : '#5b6572'))
      .text((p) => p.axis);

    const p = pts[sel];
    const delta = (p.test_acc_mean - base.test_acc_mean) * 100;
    const inside = Math.abs(p.test_acc_mean - base.test_acc_mean) < band;
    const best = pts.reduce((a, b) => (b.test_acc_mean > a.test_acc_mean ? b : a));
    const worst = pts.reduce((a, b) => (b.test_acc_mean < a.test_acc_mean ? b : a));

    document.querySelector('#scaling-readout').innerHTML =
      `<span class="bold">${p.label}.</span> ${p.blocks.join('+')} blocks, ${p.width} channels, `
      + `${p.res}×${p.res} input: ${(p.macs / 1e6).toFixed(2)} MMAC, which is `
      + `<span class="bold">${p.achieved_budget.toFixed(2)}×</span> the base network rather than the `
      + `${data.scaling.budget_target.toFixed(1)}× that was asked for, because blocks and channels `
      + `come in integers. It scores ${(p.test_acc_mean * 100).toFixed(2)}%, `
      + `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} points against the base, `
      + `<span class="bold">${inside ? 'inside the noise band' : 'outside it'}</span>. `
      + `<br /><br />Across all ${pts.length} allocations the best is ${NICE[best.axis] || best.axis} at `
      + `${(best.test_acc_mean * 100).toFixed(2)}% and the worst is ${NICE[worst.axis] || worst.axis} at `
      + `${(worst.test_acc_mean * 100).toFixed(2)}%, a spread of `
      + `${((best.test_acc_mean - worst.test_acc_mean) * 100).toFixed(2)} points. The achieved budgets `
      + `themselves span a factor of ${data.scaling.budget_spread.toFixed(2)}, which is worth keeping in `
      + `view: it is larger than several of the accuracy gaps on this chart.`;
  }

  draw(pts.findIndex((p) => p.axis === 'efficientnet') >= 0
    ? pts.findIndex((p) => p.axis === 'efficientnet')
    : 0);
}
