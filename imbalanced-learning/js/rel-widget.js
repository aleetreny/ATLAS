/* The reliability diagram: what was claimed against what came true.
 *
 * One method at a time, because the five broken ones lie on top of each other
 * and the story is the comparison against the baseline, which stays as a
 * faint reference on every view. Dot area tracks bin population on a log
 * scale, so the huge low-probability bin does not visually drown the story
 * told by the small confident ones.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

const LABELS = {
  baseline: 'baseline',
  class_weights: 'class weights',
  undersampling: 'undersampling',
  oversampling: 'oversampling',
  SMOTE: 'SMOTE',
  ADASYN: 'ADASYN',
};

export function initRelWidget(data) {
  const calib = data.calibration;
  const bench = data.benchmark;
  const { g, w, h } = makeChart('#rel-chart', {
    width: 620,
    height: 480,
    margin: { top: 30, right: 26, bottom: 56, left: 62 },
  });

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'probability the model claimed', y: 'share that really was the tree' });

  g.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(1))
    .attr('stroke', 'var(--stone)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(0.62)).attr('y', y(0.62) - 10)
    .attr('transform', `rotate(${-Math.atan2(h, w) * 180 / Math.PI} ${x(0.62)} ${y(0.62) - 10})`)
    .style('font-size', '0.54rem').style('text-transform', 'none')
    .text('honesty: claimed = true');

  const refG = g.append('g');
  const mainG = g.append('g');

  const rDot = (n) => 2.5 + 2.4 * Math.log10(Math.max(n, 10) / 10);
  const lineOf = (bins) => d3.line().x((b) => x(b.predicted)).y((b) => y(b.observed))(bins);

  /* The baseline as a permanent faint reference. */
  refG.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2).attr('opacity', 0.3).attr('d', lineOf(calib.baseline));

  const btnRow = document.querySelector('#rel-buttons');
  const readout = document.querySelector('#rel-readout');
  const keys = Object.keys(LABELS);
  const state = { key: 'baseline' };

  const btns = {};
  for (const k of keys) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (k === 'baseline' ? '' : ' ghost');
    b.textContent = LABELS[k];
    b.addEventListener('click', () => { state.key = k; render(); });
    btnRow.appendChild(b);
    btns[k] = b;
  }

  function render() {
    const k = state.key;
    const bins = calib[k];
    const row = bench.find((b2) => b2.name === LABELS[k]);
    for (const [k2, b2] of Object.entries(btns)) b2.classList.toggle('ghost', k2 !== k);

    mainG.selectAll('path').data([bins]).join('path')
      .attr('fill', 'none').attr('stroke', k === 'baseline' ? 'var(--primary)' : 'var(--cosmos)')
      .attr('stroke-width', 2.6).attr('d', lineOf(bins));
    mainG.selectAll('circle').data(bins).join('circle')
      .attr('cx', (b2) => x(b2.predicted)).attr('cy', (b2) => y(b2.observed))
      .attr('r', (b2) => rDot(b2.n))
      .attr('fill', k === 'baseline' ? 'var(--primary)' : 'var(--cosmos)')
      .attr('stroke', 'white').attr('stroke-width', 1.2);

    const top = bins[bins.length - 1];
    readout.innerHTML =
      `<table class="imb-table">
         <tr><td>implied base rate</td><td><span class="value">${(row.mean_p * 100).toFixed(2)}%</span> (truth: 1.00%)</td></tr>
         <tr><td>Brier score</td><td><span class="value">${row.brier.toFixed(5)}</span></td></tr>
         <tr><td>most confident bin</td><td>claimed ${(top.predicted * 100).toFixed(1)}%, true ${(top.observed * 100).toFixed(1)}%, across ${top.n.toLocaleString('en-US')} cells</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        k === 'baseline'
          ? `The promise kept: every dot hugs the diagonal, and the faint teal line under the other five ` +
            `views is this one. Dot size is bin population on a log scale; the giant dot at the origin is ` +
            `${calib.baseline[0].n.toLocaleString('en-US')} cells the model correctly waves through with ` +
            `almost no probability.`
          : `The bow away from the diagonal is the resampling talking: trained at 50% prevalence, the model ` +
            `slides every claim upward. The cure is not to abandon the model, because the AUROC button in ` +
            `the last widget showed its ranking is intact. The cure is to repair the probabilities, which ` +
            `is the next article.`
      }</div>`;
  }

  render();
}
