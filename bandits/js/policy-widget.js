/* Five policies on the same five machines, and what each of them regrets.
 *
 * Every curve is an average over four thousand runs, so a line that looks
 * smooth is smooth because of the averaging: the spread column in the table is
 * the number that says how little one run tells you.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { laiRobbins } from './banditkit.js';

const LABEL = {
  greedy: 'always the best estimate',
  eps_10: 'explore 10% of the time',
  eps_01: 'explore 1% of the time',
  eps_decay: 'a rate that decays',
  etc: 'explore then commit',
};
/* the legend lines are written out rather than cut off the label: splitting on
   spaces left "always the" and "a rate" standing on their own */
const SHORT = {
  greedy: ['always the', 'best estimate'],
  eps_10: ['explore', '10% of pulls'],
  eps_01: ['explore', '1% of pulls'],
  eps_decay: ['a rate that', 'decays'],
  etc: ['explore then', 'commit'],
};

export function initPolicyWidget(data) {
  const P = data.policies;
  const keys = Object.keys(P);
  const readout = document.querySelector('#policy-readout');
  const controls = d3.select('#policy-controls');
  let pick = 'eps_10';

  const chart = makeChart('#policy-chart', {
    width: 640, height: 360, margin: { top: 48, right: 120, bottom: 54, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const T = data.meta.horizon;
  const stride = P[keys[0]].stride;
  const x = d3.scaleLinear().domain([0, T]).range([0, w]);
  const y = d3.scaleLinear()
    .domain([0, Math.max(...keys.map((k) => P[k].final)) * 1.15]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'pulls', y: 'regret so far', leftBudget: margin.left });

  const line = (arr) => d3.line().x((_, i) => x(i * stride)).y((d) => y(d))(arr);
  keys.forEach((k) => {
    g.append('path').attr('class', `curve ${k}`).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4)
      .attr('opacity', 0.35).attr('d', line(P[k].regret));
  });
  const front = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 3);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 4)`);

  function render() {
    front.attr('d', line(P[pick].regret));
    g.selectAll('path.curve').attr('opacity', function opacity() {
      return d3.select(this).classed(pick) ? 0 : 0.32;
    });
    title.text(`${LABEL[pick]}: regret ${P[pick].final} after ${T.toLocaleString('en-US')} pulls`);
    /* short: the plot is 458 units wide and the long version lives in the
       table underneath, where there is room for it */
    sub.text(`spread ${P[pick].std}, best machine found in `
      + `${data.meta.runs - P[pick].wrong_arm} of ${data.meta.runs}`);
    key.selectAll('*').remove();
    keys.forEach((k, i) => {
      key.append('line').attr('x1', 0).attr('x2', 12).attr('y1', i * 42 + 4).attr('y2', i * 42 + 4)
        .attr('stroke', 'var(--primary)').attr('stroke-width', k === pick ? 3 : 1.4)
        .attr('opacity', k === pick ? 1 : 0.4);
      SHORT[k].forEach((line, j) => {
        key.append('text').attr('class', 'chart-note').attr('x', 0)
          .attr('y', i * 42 + 16 + j * 16).style('font-size', '10px').text(line);
      });
    });

    const sorted = keys.slice().sort((a, b) => P[a].final - P[b].final);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>policy</th><th>regret</th><th>spread</th><th>ended on the best</th>
            <th>pulls on the best</th></tr>
        ${sorted.map((k) => `
          <tr><td>${k === pick ? '<span class="value">' : ''}${LABEL[k]}${k === pick ? '</span>' : ''}</td>
              <td>${P[k].final}</td><td>${P[k].std}</td>
              <td>${data.meta.runs - P[k].wrong_arm} of ${data.meta.runs}</td>
              <td>${(P[k].share_best * 100).toFixed(1)}%</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The worst of them is <span class="value">${LABEL[sorted[sorted.length - 1]]}</span>
        at ${P[sorted[sorted.length - 1]].final}, and the reason is in the fourth
        column, not the first: it finishes believing the wrong machine is best in
        ${P.greedy.wrong_arm} of ${data.meta.runs} runs, and once it does there is
        nothing in it that can find out. Notice the spread column as well. The
        best policy here regrets ${P[sorted[0]].final} give or take
        ${P[sorted[0]].std}, which is most of the value: one run of a bandit
        algorithm tells you almost nothing, and every number on this page is an
        average over ${data.meta.runs.toLocaleString('en-US')} of them.
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
