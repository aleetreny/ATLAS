/* Six steps in one dimension: why a straight line cannot be a probability.
 *
 * Handlers are idempotent state declarations, so scrolling back up works for
 * free. The two curves live in a clipped group: the least squares line leaves
 * the [0,1] band by design, which is the whole point, but it must leave the
 * plotting area cleanly rather than painting over the axis.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { auc } from './mathkit.js';

export function initScrollyOdds(data) {
  const D = data.one_d;
  const pts = D.points;

  const { g, w, h } = makeChart('#odds-chart', {
    width: 660,
    height: 470,
    margin: { top: 34, right: 26, bottom: 58, left: 66 },
  });

  const x = d3.scaleLinear().domain(d3.extent(D.grid)).nice().range([0, w]);
  const y = d3.scaleLinear().domain([-0.35, 1.35]).range([h, 0]);

  drawGrid(g, x, y, w, h, { yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.1f') });
  axisLabels(g, w, h, { x: D.feature.replace(/_/g, ' '), y: 'malignant (1) or benign (0)' });

  const clip = 'odds-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -2).attr('y', -2)
    .attr('width', w + 4).attr('height', h + 4);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  /* The legal band for a probability, drawn once so the reader can see the
   * line leave it rather than being told that it does. */
  const band = g.append('rect')
    .attr('x', 0).attr('y', y(1)).attr('width', w).attr('height', y(0) - y(1))
    .attr('fill', 'var(--primary)').attr('opacity', 0);
  const bandLabel = g.append('text').attr('class', 'chart-annotation')
    .attr('x', 6).attr('y', y(1) - 7).style('font-size', '0.62rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)').attr('opacity', 0).text('everything a probability is allowed to be');

  const dots = g.append('g').selectAll('circle').data(pts).join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 3.6)
    .attr('fill', (d) => (d.y ? 'var(--cosmos)' : 'var(--aqua)'))
    .attr('stroke', 'white')
    .attr('stroke-width', 0.7)
    .attr('opacity', 0.62);

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
  const olsData = D.grid.map((v, i) => [v, D.ols[i]]);
  const logData = D.grid.map((v, i) => [v, D.logistic[i]]);

  const mk = (colour, dash) => {
    const halo = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7).attr('opacity', 0);
    const path = plot.append('path').attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 3.4)
      .attr('opacity', 0);
    if (dash) path.attr('stroke-dasharray', dash);
    return { halo, path };
  };
  const ols = mk('var(--squidink)');
  const log = mk('var(--primary)');
  ols.halo.attr('d', line(olsData));
  ols.path.attr('d', line(olsData));
  log.halo.attr('d', line(logData));
  log.path.attr('d', line(logData));

  /* Points where the straight line promises something impossible. */
  const bad = pts.filter((d) => {
    const p = D.ols_coef.intercept + D.ols_coef.slope * d.x;
    return p < 0 || p > 1;
  });
  const badDots = g.append('g').selectAll('circle').data(bad).join('circle')
    .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y))
    .attr('r', 5.6).attr('fill', 'none')
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('opacity', 0);

  const half = g.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', y(0.5)).attr('y2', y(0.5))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
    .attr('stroke-dasharray', '6 5').attr('opacity', 0);
  const cut = g.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2).attr('opacity', 0);
  /* Inside the plot, not above it: at y = -12 this label sat on the caption. */
  const cutLabel = g.append('text').attr('class', 'chart-annotation')
    .attr('y', y(1.22)).attr('text-anchor', 'middle').style('font-size', '0.64rem')
    .attr('fill', 'var(--primary)').attr('opacity', 0);

  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* The threshold in feature units: where the log odds cross zero. */
  const xCut = -D.log_coef.intercept / D.log_coef.slope;

  const show = ({ olsOn = 0, logOn = 0, bandOn = 0, badOn = 0, cutOn = 0, label = '' }) => {
    const t = (sel) => sel.transition('fade').duration(420);
    t(ols.path).attr('opacity', olsOn);
    t(ols.halo).attr('opacity', olsOn ? 1 : 0);
    t(log.path).attr('opacity', logOn);
    t(log.halo).attr('opacity', logOn ? 1 : 0);
    t(band).attr('opacity', bandOn ? 0.07 : 0);
    t(bandLabel).attr('opacity', bandOn ? 1 : 0);
    t(badDots).attr('opacity', badOn);
    t(half).attr('opacity', cutOn);
    t(cut).attr('opacity', cutOn).attr('x1', x(xCut)).attr('x2', x(xCut));
    t(cutLabel).attr('opacity', cutOn).attr('x', x(xCut)).text(cutOn ? `p = 0.5 at ${xCut.toFixed(1)}` : '');
    t(dots).attr('opacity', badOn ? 0.28 : 0.62);
    caption.text(label);
  };

  scrolly('#odds-scrolly .step', {
    0: () => show({ label: '398 training biopsies, and the answer is only ever 0 or 1' }),
    1: () => show({ olsOn: 1, label: 'least squares, handed a target it was never designed for' }),
    2: () => show({
      olsOn: 1,
      bandOn: 1,
      badOn: 1,
      label: `${D.out_of_range} of ${pts.length} fitted values fall outside [0, 1]`,
    }),
    3: () => show({ bandOn: 1, logOn: 1, label: 'the sigmoid: free in the middle, flat at both ends' }),
    4: () => show({ olsOn: 0.45, logOn: 1, label: 'same ordering, different promise' }),
    5: () => show({ logOn: 1, cutOn: 1, label: 'the decision is one crossing' }),
  });

  /* Runtime check on the claim step 4 makes, computed on the points actually
   * drawn: the two curves rank these 398 training patients identically. Both
   * are monotone functions of one feature so this is arithmetic rather than
   * luck, but the article prints the number, so the number gets measured and
   * cross-checked against what the generator exported. */
  const ys = pts.map((d) => d.y);
  const aOls = auc(ys, pts.map((d) => D.ols_coef.intercept + D.ols_coef.slope * d.x));
  const aLog = auc(ys, pts.map((d) => D.log_coef.intercept + D.log_coef.slope * d.x));
  if (Math.abs(aOls - aLog) > 1e-9) {
    console.warn(`one-d auc mismatch: ols ${aOls.toFixed(6)} vs logistic ${aLog.toFixed(6)}`);
  }
  if (D.auc_train_log !== undefined && Math.abs(aLog - D.auc_train_log) > 0.001) {
    console.warn(`one-d train auc ${aLog.toFixed(4)} vs generator ${D.auc_train_log}`);
  }
}
