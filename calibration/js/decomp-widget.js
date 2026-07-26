/* The Brier decomposition: calibration is only one of the two things a
 * probability can be good at.
 *
 * Brier = reliability - resolution + uncertainty. Reliability (drawn leftward,
 * a penalty) is what this article has been repairing; resolution (rightward,
 * a prize) is what the dummy model proves you cannot repair into existence.
 * One paired bar per model-state.
 */
import { makeChart } from '../../assets/js/chart.js';

const ROWS = [
  { key: 'constant', st: 'raw', label: 'always say 1%' },
  { key: 'naive_bayes', st: 'raw', label: 'naive Bayes, raw' },
  { key: 'naive_bayes', st: 'isotonic', label: 'naive Bayes, isotonic' },
  { key: 'random_forest', st: 'isotonic', label: 'random forest, isotonic' },
  { key: 'smote_logistic', st: 'isotonic', label: 'SMOTE, isotonic' },
  { key: 'logistic', st: 'raw', label: 'logistic, raw' },
];

export function initDecompWidget(data) {
  const rows = ROWS.map((r) => ({ ...r, m: data.models[r.key][r.st] }));
  const { g, w, h } = makeChart('#decomp-chart', {
    width: 700,
    height: 340,
    margin: { top: 46, right: 30, bottom: 34, left: 190 },
  });

  const maxV = d3.max(rows, (r) => Math.max(r.m.rel, r.m.res)) * 1.15;
  const mid = w * 0.5;
  const xL = d3.scaleLinear().domain([0, maxV]).range([mid, mid - w * 0.48]);
  const xR = d3.scaleLinear().domain([0, maxV]).range([mid, mid + w * 0.48]);
  const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, h]).padding(0.32);

  g.append('line').attr('x1', mid).attr('x2', mid).attr('y1', -6).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
  g.append('text').attr('class', 'chart-annotation').attr('x', mid - 10).attr('y', -26)
    .attr('text-anchor', 'end').style('font-size', '0.6rem').attr('fill', 'var(--cosmos)')
    .text('reliability penalty (lies)');
  g.append('text').attr('class', 'chart-annotation').attr('x', mid + 10).attr('y', -26)
    .attr('text-anchor', 'start').style('font-size', '0.6rem').attr('fill', 'var(--primary)')
    .text('resolution prize (knowledge)');
  g.append('text').attr('class', 'chart-annotation').attr('x', mid).attr('y', -12)
    .attr('text-anchor', 'middle').style('font-size', '0.5rem').style('text-transform', 'none')
    .text('Brier = uncertainty + penalty - prize');

  for (const r of rows) {
    g.append('text').attr('x', -12).attr('y', y(r.label) + y.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end')
      .style('font-family', 'var(--font-mono)').style('font-size', '12px')
      .attr('fill', 'var(--squidink)').text(r.label);
    g.append('rect').attr('x', xL(r.m.rel)).attr('y', y(r.label))
      .attr('width', mid - xL(r.m.rel)).attr('height', y.bandwidth())
      .attr('fill', 'var(--cosmos)').attr('opacity', 0.75)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8);
    g.append('rect').attr('x', mid).attr('y', y(r.label))
      .attr('width', xR(r.m.res) - mid).attr('height', y.bandwidth())
      .attr('fill', 'var(--primary)').attr('opacity', 0.85)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8);
    g.append('text').attr('x', xR(r.m.res) + 6).attr('y', y(r.label) + y.bandwidth() / 2 + 4)
      .style('font-family', 'var(--font-mono)').style('font-size', '10px')
      .attr('fill', 'var(--squidink)').text(r.m.res.toFixed(4));
  }

  const readout = document.querySelector('#decomp-readout');
  const cst = data.models.constant.raw;
  const best = rows.reduce((a, b) => (b.m.res > a.m.res ? b : a));
  readout.innerHTML =
    `<div class="slider-label" style="line-height:1.5">The dummy's row is the whole argument: reliability ` +
    `penalty ${cst.rel.toFixed(5)}, essentially honest, and a resolution prize of exactly ` +
    `${cst.res.toFixed(4)}, because claiming 1% everywhere separates nothing from nothing. Its Brier is ` +
    `the uncertainty floor of ${cst.unc.toFixed(4)}, the score of knowing only the base rate. Meanwhile ` +
    `${best.label} earns a prize of ${best.m.res.toFixed(4)} against a penalty of ${best.m.rel.toFixed(5)}: ` +
    `it knows a great deal and lies almost never. Calibration fixed the left side of every row; the right ` +
    `side, the knowledge, was earned in training or not at all. This pairing, honest and sharp, is the ` +
    `standard this atlas has been building toward since the probabilistic classification article.</div>`;
}
