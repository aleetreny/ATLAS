/* The two crutches, in a two by two, with the floor under the numbers.
 *
 * Four cells, five seeds each. A cell is only different from another cell if it
 * is further apart than the same cell is from itself across seeds, so the
 * spread is drawn on every bar rather than mentioned underneath.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initAblationWidget(data) {
  const A = data.ablation;
  const readout = document.querySelector('#ablation-readout');
  const rows = A.rows.map((d) => ({ ...d,
    label: `${d.replay ? 'replay' : 'no replay'}, ${d.target ? 'target' : 'no target'}` }));

  const chart = makeChart('#ablation-chart', {
    width: 640, height: 360, margin: { top: 50, right: 122, bottom: 82, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scalePoint().domain(rows.map((d) => d.label)).range([0, w]).padding(0.6);
  const top = Math.max(...rows.map((d) => d.gap + d.gap_spread)) * 1.15;
  const y = d3.scaleLinear().domain([0, top]).range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5, xValues: [] });
  axisLabels(g, w, h, { y: 'distance to the answer',
    leftBudget: margin.left });
  rows.forEach((d) => {
    d.label.split(', ').forEach((line, j) => {
      g.append('text').attr('class', 'chart-note').attr('x', x(d.label))
        .attr('y', h + 18 + j * 16).attr('text-anchor', 'middle')
        .style('font-size', '10.5px').text(line);
    });
  });
  g.selectAll('rect').data(rows).join('rect')
    .attr('x', (d) => x(d.label) - 26).attr('width', 52)
    .attr('y', (d) => y(d.gap)).attr('height', (d) => h - y(d.gap))
    .attr('fill', (d) => (d.replay && d.target ? 'var(--primary)' : '#c9ced0'));
  g.selectAll('line.err').data(rows).join('line').attr('class', 'err')
    .attr('x1', (d) => x(d.label)).attr('x2', (d) => x(d.label))
    .attr('y1', (d) => y(Math.max(d.gap - d.gap_spread, 0)))
    .attr('y2', (d) => y(d.gap + d.gap_spread))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
  g.selectAll('text.v').data(rows).join('text').attr('class', 'chart-note v')
    .attr('x', (d) => x(d.label)).attr('y', (d) => y(d.gap + d.gap_spread) - 7)
    .attr('text-anchor', 'middle').style('font-size', '10.5px')
    .text((d) => d.gap);
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px')
    .text(`${A.seeds} seeds per cell, ${A.episodes} episodes each`);
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px')
    .text('the whisker is how far apart two seeds of the same cell land');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 8)`);
  key.append('rect').attr('width', 12).attr('height', 12).attr('fill', 'var(--primary)');
  ['both crutches'].forEach((t, i) => {
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 28 + i * 11)
      .style('font-size', '10.5px').text(t);
  });
  key.append('rect').attr('y', 46).attr('width', 12).attr('height', 12).attr('fill', '#c9ced0');
  ['one or neither'].forEach((t, i) => {
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 74 + i * 11)
      .style('font-size', '10.5px').text(t);
  });

  const both = rows.find((d) => d.replay && d.target);
  const neither = rows.find((d) => !d.replay && !d.target);
  const worst = rows.reduce((a, b) => (b.gap > a.gap ? b : a), rows[0]);
  readout.innerHTML = `
    <table class="gen-table">
      <tr><th>replay</th><th>target</th><th>policy</th><th>reaches goal</th>
          <th>distance</th><th>largest value</th></tr>
      ${rows.map((d) => `
        <tr><td>${d.replay ? 'yes' : 'no'}</td><td>${d.target ? 'yes' : 'no'}</td>
            <td>${d.greedy === null ? 'never arrives' : d.greedy}</td>
            <td>${d.reaches_goal} of ${d.of}</td>
            <td>${d === both ? '<span class="value">' : ''}${d.gap}${d === both ? '</span>' : ''}
                &plusmn; ${d.gap_spread}</td>
            <td>${(d.qmax * data.meta.scale).toFixed(1)}</td></tr>`).join('')}
    </table>
    <div class="gen-note">
      With both crutches the values end <span class="value">${both.gap}</span>
      from the exact answer and the policy is worth
      ${both.greedy === null ? 'nothing' : both.greedy}; with neither, ${neither.gap}
      and ${neither.greedy === null ? 'a policy that never arrives' : neither.greedy}.
      Read those against the whisker, ${both.gap_spread}, which is how far apart
      two seeds of the same cell land: a difference smaller than that is not a
      difference. The last column is the one worth watching in your own runs.
      A network that is drifting does not announce it by getting worse at the
      task first; it announces it by holding values that no route in this world
      could ever pay, and the worst cell here holds ${(worst.qmax * data.meta.scale).toFixed(0)} where nothing
      in the world is worth more than ${Math.abs(data.meta.truth_return)}.
    </div>`;
  return { render: () => {} };
}
