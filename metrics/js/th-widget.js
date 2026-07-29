/* Four metrics over the same scores, and the four different places they peak.
 *
 * A ranking metric hands you an ordering. Turning that into a decision needs a
 * threshold, and the threshold is not part of the model: it is part of the
 * metric you decided to maximise. The vertical lines are where each of four
 * common choices puts it, which on an imbalanced problem can be very far apart.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const SERIES = [
  ['accuracy', 'var(--squidink)'],
  ['f1', 'var(--cosmos)'],
  ['mcc', 'var(--primary)'],
  ['youden', 'var(--anchor)'],
];
const NAME = {
  accuracy: 'accuracy', f1: 'F1', mcc: 'correlation coefficient', youden: 'Youden index',
};

export function initThresholdWidget(data) {
  const T = data.threshold;
  const readout = document.querySelector('#th-readout');

  const chart = makeChart('#th-chart', {
    width: 640, height: 380, margin: { top: 68, right: 34, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([-0.05, 1.02]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('.1f') });
  axisLabels(g, w, h, { x: 'the threshold', y: 'what each metric says' });

  SERIES.forEach(([key, colour], i) => {
    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.t)).y((r) => y(r[key]))(T.curve))
      .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
    const best = T.rows.find((r) => r.metric === key);
    layer.append('line').attr('x1', x(best.threshold)).attr('x2', x(best.threshold))
      .attr('y1', y(best.value)).attr('y2', h)
      .attr('stroke', colour).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3');
    layer.append('circle').attr('cx', x(best.threshold)).attr('cy', y(best.value)).attr('r', 4)
      .attr('fill', colour);
    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -52 + i * 15)
      .style('font-size', '11px').attr('fill', colour)
      .text(`${NAME[key]}: best at ${best.threshold.toFixed(3)}`);
  });

  const spread = Math.max(...T.rows.map((r) => r.threshold))
    - Math.min(...T.rows.map((r) => r.threshold));
  const acc = T.rows.find((r) => r.metric === 'accuracy');
  const mcc = T.rows.find((r) => r.metric === 'mcc');
  const rows = T.rows.map((r) => `<tr><td>${NAME[r.metric]}</td>`
    + `<td>${r.threshold.toFixed(3)}</td><td>${r.value.toFixed(4)}</td>`
    + `<td>${r.predicted_positive}</td><td>${r.tp}</td><td>${r.fp}</td><td>${r.fn}</td></tr>`)
    .join('');
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>maximising</th>`
    + `<th>puts the threshold at</th><th>and scores</th><th>flags</th><th>true positives</th>`
    + `<th>false positives</th><th>missed</th></tr></thead><tbody>${rows}</tbody></table>`
    + `<div class="gen-note">One set of scores, one model, four answers to "which rows do I `
    + `act on". The thresholds span ${spread.toFixed(3)} of the range, and the number of rows `
    + `flagged runs from ${Math.min(...T.rows.map((r) => r.predicted_positive))} to `
    + `${Math.max(...T.rows.map((r) => r.predicted_positive))}. The prevalence here is `
    + `${(T.prevalence * 100).toFixed(2)}%, which is why the accuracy row is the one to read `
    + `first: flagging nothing at all scores ${T.always_negative.accuracy.toFixed(4)} on `
    + `accuracy, ${T.always_negative.f1.toFixed(4)} on F1 and `
    + `${T.always_negative.mcc.toFixed(4)} on the correlation coefficient. Maximising `
    + `accuracy here reaches ${acc.value.toFixed(4)} while the correlation coefficient at `
    + `its own best is ${mcc.value.toFixed(4)}; the first of those two numbers is almost `
    + `entirely a statement about how rare the positives are.</div>`;

  return { render: () => {} };
}
