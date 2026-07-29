/* Three sweeps: how much exploring, how big a step, and what a maximum over
 * noisy estimates does to a value.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const VIEWS = [
  { key: 'eps', label: 'how much exploring' },
  { key: 'alpha', label: 'how big a step' },
  { key: 'bias', label: 'the cost of a maximum' },
];

export function initDialsWidget(data) {
  const readout = document.querySelector('#dials-readout');
  const controls = d3.select('#dials-controls');
  let view = 'eps';

  const chart = makeChart('#dials-chart', {
    width: 640, height: 350, margin: { top: 50, right: 124, bottom: 56, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function pair(rows, field, xlabel, xvals, fmt) {
    const x = d3.scalePoint().domain(xvals.map(String)).range([0, w]).padding(0.5);
    const all = rows.flatMap((d) => [d.sarsa[field], d.qlearning[field]]).filter((v) => v !== null);
    const y = d3.scaleLinear().domain([Math.min(...all) * 1.1, Math.max(...all) * 0.9])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5 });
    axisLabels(g, w, h, { x: xlabel, y: field === 'online' ? 'earned while learning'
      : 'what its policy is worth', leftBudget: margin.left });
    plot.selectAll('*').remove();
    [['sarsa', 'var(--anchor)'], ['qlearning', 'var(--primary)']].forEach(([kk, col]) => {
      const pts = rows.filter((d) => d[kk][field] !== null);
      plot.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.6)
        .attr('d', d3.line().x((d) => x(String(fmt(d)))).y((d) => y(d[kk][field]))(pts));
      plot.selectAll(`circle.${kk}`).data(pts).join('circle').attr('class', kk)
        .attr('cx', (d) => x(String(fmt(d)))).attr('cy', (d) => y(d[kk][field]))
        .attr('r', 4).attr('fill', col);
    });
    key.selectAll('*').remove();
    [['var(--anchor)', 'SARSA'], ['var(--primary)', 'Q-learning']].forEach(([col, t], i) => {
      key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 30 + 6).attr('y2', i * 30 + 6)
        .attr('stroke', col).attr('stroke-width', 2.6);
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 30 + 20)
        .style('font-size', '10.5px').text(t);
    });
  }

  function renderEps() {
    const rows = data.epsilon.rows;
    pair(rows, 'greedy', 'share of moves taken at random', rows.map((d) => d.eps), (d) => d.eps);
    title.text('what each one leaves behind, as the exploring is turned down');
    sub.text(`the best possible route is worth ${data.epsilon.optimal_return}`);
    const zero = rows[rows.length - 1];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>exploring</th><th>SARSA</th><th>best routes</th>
            <th>Q-learning</th><th>best routes</th></tr>
        ${rows.map((d) => `
          <tr><td>${d.eps}</td><td>${d.sarsa.greedy}</td>
              <td>${d.sarsa.optimal_paths}</td>
              <td>${d.qlearning.greedy}</td><td>${d.qlearning.optimal_paths}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        Q-learning returns the best route at every setting, including the ones
        where exploring is so heavy that it spends the whole run falling off the
        cliff: what it learns does not depend on what it does.
        SARSA returns the safe route at every setting except the last one, and at
        <span class="value">zero exploring the two become the same algorithm</span>
        and both find ${zero.sarsa.greedy} in ${zero.sarsa.optimal_paths} of 20
        seeds. That is the whole difference in one row: SARSA is learning about
        the policy it follows, so when the policy stops taking random steps, the
        thing it learns about is the greedy policy.
      </div>`;
  }

  function renderAlpha() {
    const rows = data.alpha.rows;
    pair(rows, 'online', 'how much of the new estimate to keep',
      rows.map((d) => d.alpha), (d) => d.alpha);
    title.text('the size of the step, against what it earns while learning');
    sub.text('averaged over the last hundred of 500 episodes, 20 seeds');
    const bestS = rows.reduce((a, b) => (b.sarsa.online > a.sarsa.online ? b : a), rows[0]);
    const bestQ = rows.reduce((a, b) => (b.qlearning.online > a.qlearning.online ? b : a), rows[0]);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>step</th><th>SARSA, learning</th><th>its policy</th>
            <th>Q, learning</th><th>its policy</th></tr>
        ${rows.map((d) => `
          <tr><td>${d.alpha}</td><td>${d.sarsa.online}</td><td>${d.sarsa.greedy}</td>
              <td>${d.qlearning.online}</td><td>${d.qlearning.greedy}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The step size is the least interesting dial on this page and it is worth
        seeing that: SARSA earns most at ${bestS.alpha} and Q-learning at
        ${bestQ.alpha}, and the policies both come back with the same two routes
        at every setting. In a world with no noise in it, a big step is not
        dangerous, it just means the last thing that happened is most of what you
        believe. The panel next door is where noise arrives.
      </div>`;
  }

  function renderBias() {
    const B = data.bias;
    const x = d3.scaleLinear().domain([0, B.episodes]).range([0, w]);
    const y = d3.scaleLinear().domain([0, Math.max(...B.single) * 1.15]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, yFmt: d3.format('.0%') });
    axisLabels(g, w, h, { x: 'episodes', y: 'times it went the wrong way',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    [[B.single, 'var(--primary)'], [B.double, 'var(--anchor)']].forEach(([arr, col]) => {
      plot.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.6)
        .attr('d', d3.line().x((_, i) => x(i * B.stride)).y((d) => y(d))(arr));
    });
    plot.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(B.eps / 2)).attr('y2', y(B.eps / 2))
      .attr('stroke', '#8e9aa6').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
    plot.append('text').attr('class', 'chart-note').attr('x', w - 2).attr('y', y(B.eps / 2) - 6)
      .attr('text-anchor', 'end').style('font-size', '10.5px')
      .text(`what exploring alone forces: ${(B.eps / 2 * 100).toFixed(0)}%`);
    key.selectAll('*').remove();
    [['var(--primary)', ['one table']], ['var(--anchor)', ['two tables']]]
      .forEach(([col, lines], i) => {
        key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 30 + 6)
          .attr('y2', i * 30 + 6).attr('stroke', col).attr('stroke-width', 2.6);
        key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 30 + 20)
          .style('font-size', '10.5px').text(lines[0]);
      });
    title.text(`two states, ${B.n_actions} noisy actions, and a wrong turn worth `
      + `${B.truth_left}`);
    sub.text(`${B.seeds.toLocaleString('en-US')} runs; the wrong turn pays `
      + `${B.mu} on average with a spread of ${B.sigma}`);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>episodes</th>${[0, 10, 20, 40, 59].map((i) => `<th>${i * B.stride}</th>`).join('')}</tr>
        <tr><td>one table goes wrong</td>
            ${[0, 10, 20, 40, 59].map((i) => `<td>${(B.single[i] * 100).toFixed(1)}%</td>`).join('')}</tr>
        <tr><td>two tables go wrong</td>
            ${[0, 10, 20, 40, 59].map((i) => `<td>${(B.double[i] * 100).toFixed(1)}%</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        The wrong turn leads to ${B.n_actions} actions that each pay ${B.mu} on
        average, so its true value is ${B.truth_left} and the right turn's is
        ${B.truth_right}. But a table that updates towards the largest of
        ${B.n_actions} noisy estimates is updating towards the luckiest one, and
        the luckiest of ${B.n_actions} draws is above the average even when every
        one of them is centred below zero. One table takes the wrong turn
        <span class="value">${(B.final_single * 100).toFixed(1)}%</span> of the
        time at the end of the run against
        <span class="value">${(B.final_double * 100).toFixed(1)}%</span> for two,
        and the dashed line is the floor: exploring at ${(B.eps * 100).toFixed(0)}%
        forces ${(B.eps / 2 * 100).toFixed(0)}% of wrong turns whatever the table
        believes. Read against that floor, one table is
        ${((B.final_single - B.eps / 2) / Math.max(B.final_double - B.eps / 2, 1e-9)).toFixed(1)}
        times as wrong as two.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    plot.selectAll('*').remove();
    if (view === 'eps') renderEps();
    else if (view === 'alpha') renderAlpha();
    else renderBias();
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
