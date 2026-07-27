/* The scrolly: one cloud, one line, and the trade that makes PCA two
 * algorithms at once.
 *
 * Handlers are idempotent state declarations, so scrolling backwards works for
 * free: each one fully specifies the picture rather than nudging the previous.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { equalScales, unit, project, budgetBar, centred } from './cloud.js';

export function initScrollyLine(data) {
  const C = data.cloud;
  const P = centred(C.raw.points);
  const W = 700;
  const H = 560;
  /* The bottom margin carries the x axis label at h+42 AND the caption below
     it. At 46 the two sat on top of each other, which the overlap check found
     and no amount of looking at the picture would have. */
  const margin = { top: 66, right: 26, bottom: 78, left: 58 };
  const { g, w, h } = makeChart('#line-chart', { width: W, height: H, margin, clip: false });
  const { x, y } = equalScales(P, w, h);

  drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
  drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
  axisLabels(g, w, h, { x: `${C.features[0]} (centred)`, y: 'colour absorbance' });

  const stubG = g.append('g').attr('class', 'stubs');
  const lineG = g.append('g').attr('class', 'axis-line');
  const dotG = g.append('g').attr('class', 'dots');
  const shadowG = g.append('g').attr('class', 'shadows');
  const budget = budgetBar(g, w, { y: -44 });
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', h + 68).attr('text-anchor', 'middle')
    .style('font-size', '0.66rem').style('text-transform', 'none')
    .style('letter-spacing', '0.5px');

  dotG.selectAll('circle').data(P).join('circle')
    .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
    .attr('r', 3.4).attr('fill', 'var(--squidink)')
    .attr('opacity', 0.55).attr('stroke', 'white').attr('stroke-width', 0.6);

  /* The line is drawn to the edge of the plotting box rather than to the
     extreme data point, so it reads as a direction and not as a segment. */
  const reach = Math.hypot(x.domain()[1] - x.domain()[0], y.domain()[1] - y.domain()[0]);
  const mainLine = lineG.append('line')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6).attr('opacity', 0);
  const perpLine = lineG.append('line')
    .attr('stroke', 'var(--anchor)').attr('stroke-width', 2)
    .attr('stroke-dasharray', '7 5').attr('opacity', 0);

  function place(sel, deg, len) {
    const [ux, uy] = unit(deg);
    return sel
      .attr('x1', x(-ux * len)).attr('y1', y(-uy * len))
      .attr('x2', x(ux * len)).attr('y2', y(uy * len));
  }

  /* One place that renders a whole state, so every step is a declaration. */
  function render(deg, { showLine, showShadows, showStubs, showPerp, animate, text }) {
    const pr = project(P, deg);
    place(animate ? mainLine.transition('line').duration(900) : mainLine, deg, reach / 2)
      .attr('opacity', showLine ? 1 : 0);
    place(animate ? perpLine.transition('perp').duration(900) : perpLine, deg + 90, reach / 2)
      .attr('opacity', showPerp ? 0.9 : 0);

    const sh = shadowG.selectAll('circle').data(showShadows ? pr.shadows : []);
    sh.exit().remove();
    const shEnter = sh.enter().append('circle')
      .attr('r', 3).attr('fill', 'var(--primary)').attr('stroke', 'white')
      .attr('stroke-width', 0.7);
    const shAll = shEnter.merge(sh);
    (animate ? shAll.transition('shadow').duration(900) : shAll)
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]));

    const st = stubG.selectAll('line').data(showStubs ? P.map((p, i) => [p, pr.shadows[i]]) : []);
    st.exit().remove();
    const stEnter = st.enter().append('line')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1).attr('opacity', 0.4);
    const stAll = stEnter.merge(st);
    (animate ? stAll.transition('stub').duration(900) : stAll)
      .attr('x1', (d) => x(d[0][0])).attr('y1', (d) => y(d[0][1]))
      .attr('x2', (d) => x(d[1][0])).attr('y2', (d) => y(d[1][1]));

    budget.update(pr.captured, C.raw.total, { animate });
    budget.node.attr('opacity', showLine ? 1 : 0);
    caption.text(text || '');
  }

  const START = 0;
  const MID = 78;
  const BEST = C.raw.dial_best_deg;

  const steps = {
    0: () => render(START, {
      showLine: false, showShadows: false, showStubs: false, showPerp: false,
      animate: false, text: `${P.length} bottles, two measurements each`,
    }),
    1: () => render(START, {
      showLine: true, showShadows: true, showStubs: false, showPerp: false,
      animate: true, text: `a direction at ${START}°, and every bottle's shadow on it`,
    }),
    2: () => render(START, {
      showLine: true, showShadows: true, showStubs: true, showPerp: false,
      animate: true, text: 'the stubs are what one number cannot carry',
    }),
    3: () => render(MID, {
      showLine: true, showShadows: true, showStubs: true, showPerp: false,
      animate: true, text: `turned to ${MID}°: the bar above moves, its total does not`,
    }),
    4: () => render(BEST, {
      showLine: true, showShadows: true, showStubs: true, showPerp: true,
      animate: true,
      text: `${BEST}°: the first principal component, and its perpendicular`,
    }),
  };

  steps[0]();
  return scrolly('#line-scrolly .step', steps);
}
