/* Why the answer is not unique, in four states.
 *
 * The chart is the distribution of one window pixel across every render of one
 * plan. It is bimodal, because a window is either lit or it is not, and the
 * single number an averaging loss will produce is the middle of that, which is
 * a brightness the data never contains.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initCondScrolly(data) {
  const C = data.conditional[0];
  const hist = C.window_hist || [];
  const c = makeChart('#cond-scrolly-chart', {
    width: 660, height: 400, margin: { top: 40, right: 30, bottom: 58, left: 70 },
  });
  const x = d3.scaleLinear().domain([0, 1]).range([0, c.w]);
  const y = d3.scaleLinear().domain([0, d3.max(hist, (h) => h.n) * 1.15 || 1]).range([c.h, 0]);
  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 5 });
  const bars = c.g.append('g').attr('opacity', 0);
  bars.selectAll('rect').data(hist).enter().append('rect')
    .attr('x', (h) => x(h.lo)).attr('y', (h) => y(h.n))
    .attr('width', (h) => Math.max(x(h.hi) - x(h.lo) - 1, 1))
    .attr('height', (h) => c.h - y(h.n))
    .attr('fill', 'var(--squidink)').attr('fill-opacity', 0.75);
  drawAxes(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 5 });
  axisLabels(c.g, c.w, c.h, {
    x: 'brightness of one window pixel',
    y: `renders of the same plan, out of ${C.m}`,
  });

  const one = c.g.append('g').attr('opacity', 0);
  const lit = hist.length ? hist[hist.length - 1] : null;
  const dark = hist.length ? hist[0] : null;
  [[dark, 'unlit'], [lit, 'lit']].forEach(([h, label]) => {
    if (!h) return;
    one.append('line').attr('x1', x((h.lo + h.hi) / 2)).attr('x2', x((h.lo + h.hi) / 2))
      .attr('y1', 0).attr('y2', c.h).attr('stroke', 'var(--anchor)').attr('stroke-width', 1.4);
    one.append('text').attr('class', 'chart-annotation')
      .attr('x', x((h.lo + h.hi) / 2) + (label === 'lit' ? -6 : 6)).attr('y', 16)
      .attr('text-anchor', label === 'lit' ? 'end' : 'start')
      .style('font-size', '11.5px').style('letter-spacing', '0.4px')
      .style('fill', 'var(--anchor)').text(label);
  });

  /* Two marks, not one, because the two losses do different things here and
     the difference is the point: a squared error is minimised by the mean,
     which lands in the empty middle, and an absolute error by the median,
     which on a two valued distribution snaps to whichever value is commoner. */
  const med = c.g.append('g').attr('opacity', 0);
  [{ v: C.window_mean, col: 'var(--cosmos)', label: 'the mean, which squared error picks', dy: -14, dx: 8, anchor: 'start' },
    { v: C.window_median, col: 'var(--anchor)', label: 'the median, which absolute error picks', dy: 14, dx: -8, anchor: 'end' }]
    .forEach((m) => {
      med.append('line').attr('x1', x(m.v)).attr('x2', x(m.v)).attr('y1', 0).attr('y2', c.h)
        .attr('stroke', m.col).attr('stroke-width', 2.4).attr('stroke-dasharray', '5 3');
      med.append('text').attr('class', 'chart-annotation')
        .attr('x', x(m.v) + m.dx).attr('y', c.h / 2 + m.dy).attr('text-anchor', m.anchor)
        .style('font-size', '11px').style('letter-spacing', '0.4px').style('fill', m.col)
        .text(m.label);
    });

  const show = (sel, on) => sel.transition('fade').duration(400).attr('opacity', on ? 1 : 0);
  const states = {
    0: () => { show(bars, false); show(one, false); show(med, false); },
    1: () => { show(bars, true); show(one, false); show(med, false); },
    2: () => { show(bars, true); show(one, true); show(med, false); },
    3: () => { show(bars, true); show(one, true); show(med, true); },
  };
  const sc = scrolly('#cond-scrolly .step', states);
  states[0]();
  return sc;
}
