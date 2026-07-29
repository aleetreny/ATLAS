/* ROC and precision-recall, side by side, from the same predictions.
 *
 * The curves come thinned from the generator; the moving operating point is
 * computed exactly from the sweep's confusion matrices, so the same slider
 * position is the same threshold on both panels. The point of the widget is
 * the disagreement: the marker crawls along the ROC's left wall while it
 * plunges down the PR cliff.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

export function initCurvesWidget(data) {
  const sweep = data.sweep;
  const c = data.curves;

  const wrap = d3.select('#cur-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  function panel(node, xl, yl, titleText) {
    const P = makeChart(node, { width: 360, height: 380, margin: { top: 34, right: 16, bottom: 48, left: 50 } });
    const x = d3.scaleLinear().domain([0, 1]).range([0, P.w]);
    const y = d3.scaleLinear().domain([0, 1]).range([P.h, 0]);
    drawGrid(P.g, x, y, P.w, P.h, { xTicks: 4, yTicks: 4 });
    drawAxes(P.g, x, y, P.w, P.h, { xTicks: 4, yTicks: 4 });
    axisLabels(P.g, P.w, P.h, { x: xl, y: yl });
    P.g.append('text').attr('class', 'chart-annotation')
      .attr('x', P.w / 2).attr('y', -16).attr('text-anchor', 'middle').style('font-size', '0.7rem')
      .text(titleText);
    return { ...P, x, y };
  }

  const R = panel(left, 'false positive rate', 'recall (true positive rate)', `ROC, area ${c.auroc.toFixed(3)}`);
  const P = panel(right, 'recall', 'precision', `precision-recall, area ${c.auprc.toFixed(3)}`);

  /* Chance references: the diagonal for ROC, the base-rate floor for PR. */
  R.g.append('line').attr('x1', R.x(0)).attr('y1', R.y(0)).attr('x2', R.x(1)).attr('y2', R.y(1))
    .attr('stroke', 'var(--stone)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  P.g.append('line').attr('x1', P.x(0)).attr('y1', P.y(c.base_rate)).attr('x2', P.x(1)).attr('y2', P.y(c.base_rate))
    .attr('stroke', 'var(--stone)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  P.g.append('text').attr('class', 'chart-annotation').attr('x', P.x(0.98)).attr('y', P.y(c.base_rate) - 6)
    .attr('text-anchor', 'end').style('font-size', '0.66rem').style('text-transform', 'none')
    .text(`chance precision, ${c.base_rate.toFixed(2)}`);

  const rocLine = d3.line().x((d) => R.x(d[0])).y((d) => R.y(d[1]));
  const prLine = d3.line().x((d) => P.x(d[0])).y((d) => P.y(d[1]));
  R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('d', rocLine(c.roc));
  P.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('d', prLine(c.pr));

  const mR = R.g.append('circle').attr('r', 5.5).attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 1.6);
  const mP = P.g.append('circle').attr('r', 5.5).attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 1.6);

  const slider = document.querySelector('#cur-slider');
  const readout = document.querySelector('#cur-readout');

  function render() {
    const s = sweep[+slider.value];
    document.querySelector('#cur-label').textContent = s.t;
    const fpr = s.fp / (s.fp + s.tn);
    mR.attr('cx', R.x(fpr)).attr('cy', R.y(s.recall));
    mP.attr('cx', P.x(s.recall)).attr('cy', P.y(s.precision));
    const alarms = s.tp + s.fp;
    readout.innerHTML =
      `<table class="imb-table">
         <tr><th>at t = ${s.t}</th><th>the ROC sees</th><th>the PR curve sees</th></tr>
         <tr><td>${s.fp.toLocaleString('en-US')} false alarms</td>
           <td>rate ${fpr.toFixed(3)} of ${(s.fp + s.tn).toLocaleString('en-US')} negatives</td>
           <td>${(100 - s.precision * 100).toFixed(1)}% of the ${alarms.toLocaleString('en-US')} alarms raised</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        s.recall > 0.95 && s.precision < 0.3
          ? `This is the trap corner. On the ROC panel this point hugs the top-left wall, the textbook picture ` +
            `of a great classifier. On the PR panel the same threshold has fallen off a cliff: the alarm queue ` +
            `is ${(100 - s.precision * 100).toFixed(0)}% noise. Both panels are telling the truth. Only one of ` +
            `them is telling it about your workload.`
          : `Drag toward the left end and watch the two markers part company: the ROC point barely moves off ` +
            `its wall while the PR point dives. The divergence is the imbalance itself: false alarms are ` +
            `divided by ${(108781).toLocaleString('en-US')} negatives on the left, and by your own alarm ` +
            `count on the right.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  slider.value = sweep.reduce((bi, s, i) => (Math.abs(s.t - 0.5) < Math.abs(sweep[bi].t - 0.5) ? i : bi), 0);
  render();
}
