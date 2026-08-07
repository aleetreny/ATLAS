/* What each learner earns while it is learning, against what its policy is
 * worth once it stops. Those are two different numbers and the whole comparison
 * lives in the gap between them.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const LABEL = { sarsa: 'SARSA', qlearning: 'Q-learning' };

export function initCurvesWidget(data) {
  const L = data.learners;
  const readout = document.querySelector('#curves-readout');
  const controls = d3.select('#curves-controls');
  let pick = 'sarsa';

  const chart = makeChart('#curves-chart', {
    width: 640, height: 360, margin: { top: 48, right: 122, bottom: 54, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const episodes = data.meta.episodes;
  const x = d3.scaleLinear().domain([0, episodes]).range([0, w]);
  const y = d3.scaleLinear().domain([-100, 0]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'episodes', y: 'what it earned that episode',
    leftBudget: margin.left });

  const smooth = (arr, k = 10) => arr.map((_, i) => {
    const lo = Math.max(0, i - k);
    const seg = arr.slice(lo, i + 1);
    return seg.reduce((a, b) => a + b, 0) / seg.length;
  });
  Object.keys(L).forEach((key) => {
    g.append('path').attr('class', `curve ${key}`).attr('fill', 'none')
      .attr('stroke', key === 'sarsa' ? 'var(--anchor)' : 'var(--primary)')
      .attr('stroke-width', 1.6).attr('opacity', 0.45)
      .attr('d', d3.line().x((_, i) => x(i)).y((d) => y(Math.max(d, -100)))(smooth(L[key].online)));
  });
  const front = g.append('path').attr('fill', 'none').attr('stroke-width', 3);
  const refs = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  [['var(--anchor)', ['SARSA while', 'learning']],
    ['var(--primary)', ['Q-learning', 'while learning']]].forEach(([col, lines], i) => {
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 38 + 6).attr('y2', i * 38 + 6)
      .attr('stroke', col).attr('stroke-width', 2);
    lines.forEach((line, j) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 38 + 20 + j * 11)
        .style('font-size', '10px').text(line);
    });
  });
  key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 90).attr('y2', 90)
    .attr('stroke', '#8e9aa6').attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
  ['what its', 'policy is', 'worth after'].forEach((t, i) => {
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 104 + i * 16)
      .style('font-size', '10px').text(t);
  });

  function render() {
    const colour = pick === 'sarsa' ? 'var(--anchor)' : 'var(--primary)';
    front.attr('stroke', colour)
      .attr('d', d3.line().x((_, i) => x(i)).y((d) => y(Math.max(d, -100)))(smooth(L[pick].online)));
    g.selectAll('path.curve').attr('opacity', function op() {
      return d3.select(this).classed(pick) ? 0 : 0.4;
    });
    refs.selectAll('*').remove();
    refs.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(L[pick].greedy_return)).attr('y2', y(L[pick].greedy_return))
      .attr('stroke', '#8e9aa6').attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    refs.append('text').attr('class', 'chart-note').attr('x', 4)
      .attr('y', y(L[pick].greedy_return) - 6).style('font-size', '10.5px')
      .text(`its policy: ${L[pick].greedy_return}`);
    title.text(`${LABEL[pick]}: ${L[pick].online_last100} an episode over the last hundred`);
    sub.text(`averaged over ${L[pick].seeds} seeds, exploring `
      + `${(data.meta.eps * 100).toFixed(0)}% of the time`);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>while learning</th><th>what its policy is worth</th>
            <th>farthest its table is from the answer</th></tr>
        ${Object.keys(L).map((k) => `
          <tr><td>${k === pick ? '<span class="value">' : ''}${LABEL[k]}${k === pick ? '</span>' : ''}</td>
              <td>${L[k].online_last100}</td><td>${L[k].greedy_return}</td>
              <td>${L[k].q_gap}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The two columns disagree about who won, and both are right.
        <span class="value">SARSA earns ${L.sarsa.online_last100}</span> an episode
        while it learns against Q-learning's ${L.qlearning.online_last100}, because
        Q-learning walks the edge of the drop and falls in every time exploration
        pushes it sideways. But the policy Q-learning ends up holding is worth
        <span class="value">${L.qlearning.greedy_return}</span> against SARSA's
        ${L.sarsa.greedy_return}. One is learning what to do if you stop
        exploring; the other is learning what to do while you are still
        exploring. If the cliff is a simulator, take the first. If it is a real
        machine, the falls are real too.
      </div>`;
  }

  controls.selectAll('button').data(Object.keys(L)).join('button')
    .attr('class', (d) => `atlas-btn${d === pick ? '' : ' ghost'}`)
    .text((d) => LABEL[d])
    .on('click', (_, d) => {
      pick = d;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q === pick ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
