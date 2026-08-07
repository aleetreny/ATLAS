/* How hard a bandit is, as a function of how close its two machines are.
 *
 * Two policies, because the answer is different for each and the difference is
 * the point: a fixed rate pays its tax whatever the gap, so its regret only
 * grows; a policy that stops exploring has an easy case at each end and its
 * worst case in the middle.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initGapWidget(data) {
  const G = data.gaps;
  const rows = G.rows;
  const readout = document.querySelector('#gap-readout');
  const chart = makeChart('#gap-chart', {
    width: 640, height: 360, margin: { top: 50, right: 126, bottom: 56, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scalePoint().domain(rows.map((d) => String(d.delta))).range([0, w]).padding(0.5);
  const y = d3.scaleLinear().domain([0, Math.max(...rows.flatMap((d) => [d.eps, d.etc])) * 1.15])
    .range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5 });
  axisLabels(g, w, h, { x: 'how much better the good machine is', y: 'regret',
    leftBudget: margin.left });

  [['eps', 'var(--cosmos)'], ['etc', 'var(--primary)']].forEach(([f, col]) => {
    g.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d) => x(String(d.delta))).y((d) => y(d[f]))(rows));
    g.selectAll(`circle.${f}`).data(rows).join('circle').attr('class', f)
      .attr('cx', (d) => x(String(d.delta))).attr('cy', (d) => y(d[f]))
      .attr('r', 4).attr('fill', col);
  });
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  [['var(--cosmos)', ['explore 10%', 'of the time']],
    ['var(--primary)', ['explore then', 'commit']]].forEach(([col, lines], i) => {
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 38 + 6).attr('y2', i * 38 + 6)
      .attr('stroke', col).attr('stroke-width', 2.6);
    lines.forEach((line, j) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 38 + 20 + j * 16)
        .style('font-size', '10.5px').text(line);
    });
  });
  const worst = rows.reduce((a, b) => (b.etc > a.etc ? b : a), rows[0]);
  g.append('circle').attr('cx', x(String(worst.delta))).attr('cy', y(worst.etc)).attr('r', 8)
    .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px')
    .text(`two machines, ${G.horizon.toLocaleString('en-US')} pulls, the gap swept`);
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px')
    .text('the hardest problem is not the one with the smallest gap');

  readout.innerHTML = `
    <table class="gen-table">
      <tr><th>gap</th><th>explore 10%</th><th>explore then commit</th>
          <th>it ended on the wrong one</th></tr>
      ${rows.map((d) => `
        <tr><td>${d === worst ? '<span class="value">' : ''}${d.delta}${d === worst ? '</span>' : ''}</td>
            <td>${d.eps}</td><td>${d.etc}</td>
            <td>${d.etc_wrong} of ${d.of}</td></tr>`).join('')}
    </table>
    <div class="gen-note">
      Read the second row first. A gap of zero costs
      <span class="value">${rows[0].etc}</span>, because when both machines are
      the same there is nothing to lose by picking either. A gap of
      ${rows[rows.length - 1].delta} costs ${rows[rows.length - 1].etc}, because a
      difference that large is obvious in a handful of pulls. The worst case is in
      between, at <span class="value">${worst.delta}</span>, where the gap is big
      enough to matter and small enough to hide: it ends on the wrong machine
      ${worst.etc_wrong} times out of ${worst.of}. The scale that theory predicts
      for where that peak sits is one over the square root of the horizon,
      ${G.predicted_delta} here, the same order and not the same number.
      The other row never peaks at all: a fixed rate pays the same tax at every
      gap, so its regret only grows, and the hardest problem for it is the one
      the other policy finds easiest.
    </div>`;
  return { render: () => {} };
}
