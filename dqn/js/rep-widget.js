/* Three ways to hold the same values, on the same world.
 *
 * The point of the middle one is to separate two things that get said as one:
 * that you have to generalise, and that you have to generalise with a network.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const LABEL = { table: 'a number per square', linear: 'a straight line in two inputs',
  network: 'a small network' };
/* the table has five columns at 596 units, so the first one gets short names */
const SHORT = { table: 'a table', linear: 'a straight line', network: 'a network' };

export function initRepWidget(data) {
  const R = data.representations;
  const keys = Object.keys(R);
  const readout = document.querySelector('#rep-readout');
  const controls = d3.select('#rep-controls');
  let pick = 'network';

  const chart = makeChart('#rep-chart', {
    width: 640, height: 350, margin: { top: 48, right: 124, bottom: 54, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const stride = R[keys[0]].stride;
  const x = d3.scaleLinear().domain([0, data.meta.episodes]).range([0, w]);
  const y = d3.scaleLinear().domain([-140, 0]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'episodes', y: 'what it earned that episode',
    leftBudget: margin.left });
  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(data.meta.truth_return))
    .attr('y2', y(data.meta.truth_return)).attr('stroke', '#8e9aa6')
    .attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
  g.append('text').attr('class', 'chart-note').attr('x', 4)
    .attr('y', y(data.meta.truth_return) - 6).style('font-size', '10.5px')
    .text(`the best possible route: ${data.meta.truth_return}`);

  keys.forEach((k) => {
    g.append('path').attr('class', `curve ${k}`).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4).attr('opacity', 0.3)
      .attr('d', d3.line().x((_, i) => x(i * stride)).y((d) => y(Math.max(d, -140)))(R[k].curve));
  });
  const front = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 3);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  keys.forEach((k, i) => {
    key.append('line').attr('x1', 0).attr('x2', 12).attr('y1', i * 42 + 4).attr('y2', i * 42 + 4)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6).attr('opacity', 0.6);
    const words = LABEL[k].split(' ');
    [words.slice(0, 3).join(' '), words.slice(3).join(' ')].filter(Boolean).forEach((t, j) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 42 + 16 + j * 16)
        .style('font-size', '10px').text(t);
    });
  });

  function render() {
    front.attr('d', d3.line().x((_, i) => x(i * stride))
      .y((d) => y(Math.max(d, -140)))(R[pick].curve));
    g.selectAll('path.curve').attr('opacity', function op() {
      return d3.select(this).classed(pick) ? 0 : 0.28;
    });
    title.text(`${LABEL[pick]}: ${R[pick].last50} an episode over the last fifty`);
    sub.text(R[pick].greedy === null
      ? `its policy never reaches the goal in any of ${R[pick].seeds} seeds`
      : `the policy it leaves is worth ${R[pick].greedy}, `
        + `reaching the goal in ${R[pick].reaches_goal} of ${R[pick].seeds} seeds`);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>learning</th><th>policy</th><th>reaches goal</th>
            <th>distance</th></tr>
        ${keys.map((k) => `
          <tr><td>${k === pick ? '<span class="value">' : ''}${SHORT[k]}${k === pick ? '</span>' : ''}</td>
              <td>${R[k].last50}</td>
              <td>${R[k].greedy === null ? 'never arrives' : R[k].greedy}</td>
              <td>${R[k].reaches_goal} of ${R[k].seeds}</td>
              <td>${R[k].gap}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The straight line is the control that matters. It sees exactly what the
        network sees, two numbers, and it
        ${R.linear.greedy === null ? 'never reaches the goal at all'
    : `leaves a policy worth ${R.linear.greedy}`}: a value function that is a
        plane in those two numbers cannot bend around a cliff, so generalising is
        not enough on its own, it has to be generalising with something that can
        represent the answer. The last column is the price of leaving the table
        behind: the network's worst square is ${R.network.gap} away from the
        exact value where the table's is ${R.table.gap}, and it still comes back
        with a route worth ${R.network.greedy === null ? 'nothing' : R.network.greedy}.
        Being wrong about a value and being wrong about which action is best are
        not the same thing, and only the second one costs anything.
      </div>`;
  }

  controls.selectAll('button').data(keys).join('button')
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
