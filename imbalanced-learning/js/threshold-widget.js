/* The dial: sixty-three real operating points of one fitted model.
 *
 * Nothing is recomputed in the browser; every row of the sweep was measured
 * offline on the full test set, so dragging the slider replays real confusion
 * matrices. The chart is the precision and recall curves over threshold with
 * the current point marked; the matrix and the verdict text do the narrating.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { makeMatrix } from './matrix.js';

export function initThresholdWidget(data) {
  const sweep = data.sweep;
  const { g, w, h } = makeChart('#thr-chart', {
    width: 560,
    height: 540,
    margin: { top: 30, right: 24, bottom: 200, left: 58 },
  });

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'decision threshold t', y: 'score' });

  const lp = d3.line().x((s) => x(s.t)).y((s) => y(s.precision));
  const lr = d3.line().x((s) => x(s.t)).y((s) => y(s.recall));
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('d', lp(sweep));
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 3).attr('d', lr(sweep));
  g.append('text').attr('class', 'chart-annotation').attr('x', x(0.8)).attr('y', y(sweep[sweep.length - 5].precision) - 10)
    .style('font-size', '0.56rem').style('text-transform', 'none').attr('fill', 'var(--primary)').text('precision');
  g.append('text').attr('class', 'chart-annotation').attr('x', x(0.8)).attr('y', y(sweep[sweep.length - 5].recall) - 10)
    .style('font-size', '0.56rem').style('text-transform', 'none').attr('fill', 'var(--cosmos)').text('recall');

  const marker = g.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  const dotP = g.append('circle').attr('r', 4.5).attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.4);
  const dotR = g.append('circle').attr('r', 4.5).attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 1.4);

  const matrix = makeMatrix(g, w / 2 - 150, h + 62, 300, 104);

  const slider = document.querySelector('#thr-slider');
  const stats = document.querySelector('#thr-stats');
  const bestI = sweep.reduce((bi, s, i) => (s.f1 > sweep[bi].f1 ? i : bi), 0);
  const defaultI = sweep.reduce((bi, s, i) => (Math.abs(s.t - 0.5) < Math.abs(sweep[bi].t - 0.5) ? i : bi), 0);

  function verdict(s, i) {
    if (i === bestI) {
      return `The F1 peak. Precision and recall meet at their best harmonic compromise, ` +
        `and note where it is: t = ${s.t}, nowhere near 0.5.`;
    }
    if (s.recall >= 0.99) {
      return `Screening mode: ${s.tp.toLocaleString('en-US')} of ${(s.tp + s.fn).toLocaleString('en-US')} trees caught, ` +
        `and ${s.fp.toLocaleString('en-US')} false alarms to wade through, ` +
        `${(s.fp / Math.max(s.tp, 1)).toFixed(1)} wasted checks per real find. Right if missing a tree is expensive and a field check is cheap.`;
    }
    if (s.fp === 0) {
      return `Zero false alarms: every one of the ${s.tp} calls is a real tree, and ${s.fn.toLocaleString('en-US')} ` +
        `stands go unfound. Right if a false accusation is the expensive mistake.`;
    }
    if (Math.abs(s.t - 0.5) < 0.02) {
      return `The default. Not chosen, inherited: it is simply where "more likely than not" falls, ` +
        `which treats both mistakes as equally priced. Here that misses ${s.fn} of ${s.tp + s.fn} trees.`;
    }
    return `Every step of the dial rebalances the same 1,099 trees and ${(108781).toLocaleString('en-US')} ` +
      `non-trees. Accuracy barely notices: it reads ${(s.acc * 100).toFixed(2)}% here.`;
  }

  function render() {
    const i = +slider.value;
    const s = sweep[i];
    document.querySelector('#thr-label').textContent = s.t;
    marker.attr('x1', x(s.t)).attr('x2', x(s.t));
    dotP.attr('cx', x(s.t)).attr('cy', y(s.precision));
    dotR.attr('cx', x(s.t)).attr('cy', y(s.recall));
    matrix.update(s);
    stats.innerHTML =
      `<table class="imb-table">
         <tr><td>precision</td><td><span class="value">${s.precision.toFixed(3)}</span></td></tr>
         <tr><td>recall</td><td><span class="value">${s.recall.toFixed(3)}</span></td></tr>
         <tr><td>F1</td><td><span class="value">${s.f1.toFixed(3)}</span></td></tr>
         <tr><td>accuracy</td><td>${s.acc.toFixed(4)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${verdict(s, i)}</div>`;
  }

  slider.addEventListener('input', render);
  const jump = (i) => { slider.value = i; render(); };
  document.querySelector('#thr-screen').addEventListener('click', () => jump(0));
  document.querySelector('#thr-f1').addEventListener('click', () => jump(bestI));
  document.querySelector('#thr-default').addEventListener('click', () => jump(defaultI));
  document.querySelector('#thr-sniper').addEventListener('click', () => jump(sweep.length - 1));
  jump(defaultI);
}
