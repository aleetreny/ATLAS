/* Every confusion matrix of a hundred cases, plotted against itself.
 *
 * There are 176,851 of them and they are all in the generator; a thinned sample
 * is drawn here because a scatter of that many marks is a rectangle. The
 * disagreement counts in the readout are not from the thinned sample: they come
 * from four hundred thousand random pairs of the full enumeration, and the
 * number of pairs is printed next to them, because a rate without its
 * denominator is a claim without a measurement.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initEnumWidget(data) {
  const E = data.enumerate;
  const readout = document.querySelector('#enum-readout');

  const chart = makeChart('#enum-chart', {
    width: 640, height: 400, margin: { top: 46, right: 34, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([-1, 1]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('.1f') });
  axisLabels(g, w, h, { x: 'F1', y: 'correlation coefficient' });

  layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.5);

  E.cloud.forEach(([tp, , f1, mcc]) => {
    layer.append('circle').attr('cx', x(f1)).attr('cy', y(mcc)).attr('r', 1.6)
      .attr('fill', 'var(--primary)').attr('opacity', tp === 0 ? 0.85 : 0.22);
  });
  layer.append('text').attr('class', 'chart-note').attr('x', 4).attr('y', -22)
    .style('font-size', '11px').attr('fill', 'var(--primary)')
    .text(`one mark per confusion matrix, thinned from ${E.matrices.toLocaleString('en-US')}`);
  layer.append('text').attr('class', 'chart-note').attr('x', 4).attr('y', y(0) - 8)
    .style('font-size', '11px')
    .text('below this line the model is worse than guessing');

  const [a, b, f1a, f1b, mcca, mccb] = E.worst_flip;
  const tnCase = E.worst_tn_case;
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>pair of metrics</th>`
    + `<th>disagree on which matrix is better</th></tr></thead><tbody>`
    + `<tr><td>F1 against the correlation coefficient</td>`
    + `<td>${(E.flip_f1_mcc * 100).toFixed(2)}% of pairs</td></tr>`
    + `<tr><td>accuracy against the correlation coefficient</td>`
    + `<td>${(E.flip_acc_mcc * 100).toFixed(2)}%</td></tr>`
    + `<tr><td>accuracy against F1</td><td>${(E.flip_acc_f1 * 100).toFixed(2)}%</td></tr>`
    + `</tbody></table>`
    + `<div class="gen-note">Over ${E.pairs.toLocaleString('en-US')} random pairs drawn from `
    + `all ${E.matrices.toLocaleString('en-US')} matrices with ${E.n} cases. The widest `
    + `single disagreement between F1 and the correlation coefficient in that sample is `
    + `[${a.join(', ')}] against [${b.join(', ')}] as true positives, false positives, false `
    + `negatives and true negatives: F1 says ${f1a.toFixed(3)} against ${f1b.toFixed(3)} and `
    + `the correlation coefficient says ${mcca.toFixed(3)} against ${mccb.toFixed(3)}, `
    + `pointing opposite ways. `
    + (tnCase
      ? `And the structural reason, which needs a table larger than a hundred cases to show, `
        + `because with the total fixed the true negatives are determined by the other three: `
        + `hold the positives side at ${tnCase[0].slice(0, 3).join(', ')} and let the true `
        + `negatives run from ${tnCase[0][3]} to ${tnCase[1][3]}. F1 stays at exactly `
        + `${tnCase[2].toFixed(3)} the whole way, and the correlation coefficient goes from `
        + `${tnCase[3].toFixed(3)} to ${tnCase[4].toFixed(3)}, a span of `
        + `${E.worst_tn.toFixed(3)}. `
      : '')
    + `Swapping which class is called positive changes F1 by `
    + `${E.f1_asym_max.toFixed(3)} at worst and by ${E.f1_asym.toFixed(3)} on average, and `
    + `changes the correlation coefficient by exactly `
    + `${E.mcc_asym.toFixed(3)}: it is symmetric by construction, which is the property that `
    + `makes it hard to game by relabelling. Marks with no true positives at all are drawn `
    + `darker; the best F1 any of them reaches is `
    + `${E.f1_all_negative.toFixed(3)}.</div>`;

  return { render: () => {} };
}
