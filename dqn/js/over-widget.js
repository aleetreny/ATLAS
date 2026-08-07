/* What the network says a state is worth, against what its own policy collects
 * from that state. Those are the same quantity, so the gap between them is a
 * bias and not a disagreement.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initOverWidget(data) {
  const O = data.overestimate;
  const readout = document.querySelector('#over-readout');
  const rows = O.rows.map((d) => ({ ...d, label: d.double ? 'two networks' : 'one network' }));

  const chart = makeChart('#over-chart', {
    width: 640, height: 340, margin: { top: 50, right: 128, bottom: 60, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scalePoint().domain(rows.map((d) => d.label)).range([0, w]).padding(0.7);
  const vals = rows.flatMap((d) => [d.believes, d.gets]).concat([O.truth_at_start]);
  const y = d3.scaleLinear().domain([Math.min(...vals) * 1.1, Math.max(...vals, 0) + 3])
    .range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5 });
  axisLabels(g, w, h, { y: 'value of the starting square', leftBudget: margin.left });
  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(O.truth_at_start))
    .attr('y2', y(O.truth_at_start)).attr('stroke', '#8e9aa6').attr('stroke-width', 1.6)
    .attr('stroke-dasharray', '5 4');
  g.append('text').attr('class', 'chart-note').attr('x', 4).attr('y', y(O.truth_at_start) - 6)
    .style('font-size', '10.5px').text(`the truth: ${O.truth_at_start}`);

  rows.forEach((d) => {
    [['believes', 'var(--primary)', -16], ['gets', 'var(--anchor)', 16]].forEach(([f, col, dx]) => {
      g.append('rect').attr('x', x(d.label) + dx - 14).attr('width', 28)
        .attr('y', y(Math.max(d[f], 0))).attr('height', Math.abs(y(d[f]) - y(0)))
        .attr('fill', col);
      /* the two bars of a pair can be within a whisker of each other, so their
         labels are staggered rather than both sitting on the bar top */
      g.append('text').attr('class', 'chart-note').attr('x', x(d.label) + dx)
        .attr('y', y(Math.max(d[f], 0)) - (f === 'believes' ? 20 : 6))
        .attr('text-anchor', 'middle').style('font-size', '10.5px').text(d[f]);
    });
  });
  const key = g.append('g').attr('transform', `translate(${w + 8}, 8)`);
  [['var(--primary)', ['what it', 'believes']], ['var(--anchor)', ['what its', 'policy gets']]]
    .forEach(([col, lines], i) => {
      key.append('rect').attr('y', i * 40).attr('width', 12).attr('height', 12).attr('fill', col);
      lines.forEach((t, j) => {
        key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 40 + 26 + j * 16)
          .style('font-size', '10.5px').text(t);
      });
    });
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px')
    .text('the same quantity, computed two ways, on the starting square');
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px')
    .text(`averaged over ${rows[0].of} seeds`);

  const single = rows.find((d) => !d.double);
  const double = rows.find((d) => d.double);
  readout.innerHTML = `
    <table class="gen-table">
      <tr><th></th><th>what it believes</th><th>what its policy collects</th>
          <th>the gap</th><th>farthest from the answer</th></tr>
      ${rows.map((d) => `
        <tr><td>${d.label}</td><td>${d.believes}</td>
            <td>${d.gets === null ? 'never arrives' : d.gets}</td>
            <td>${d === double ? '<span class="value">' : ''}${d.bias}${d === double ? '</span>' : ''}</td>
            <td>${d.gap}</td></tr>`).join('')}
    </table>
    <div class="gen-note">
      Both numbers in the first two columns are answers to the same question:
      what is the starting square worth. One is read off the network, the other
      is collected by walking. With one network the answer it gives itself is
      <span class="value">${single.believes}</span> and what it collects is
      ${single.gets}, a gap of ${single.bias}; with two, ${double.believes}
      against ${double.gets}, a gap of ${double.bias}. The cause is the maximum:
      picking the largest of four noisy estimates picks the luckiest error along
      with the best action, and then that number is used as the target for
      everything upstream of it. Two networks, one choosing and the other
      scoring, break the coupling, and the same idea appeared two articles ago
      with two tables instead of two networks.
    </div>`;
  return { render: () => {} };
}
