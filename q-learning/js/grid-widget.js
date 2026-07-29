/* Three tables over the same world: the exact answer and the two learners.
 *
 * The arrows are read off each table rather than described, and the path is the
 * one that table produces. The exact answer is recomputed in the page.
 */
import { makeChart } from '../../assets/js/chart.js';
import { makeWorld, valueIteration, greedyPath } from './gridkit.js';
import { drawGrid, paintSpecials, paintValues, paintArrows, paintPath } from './gridview.js';

const WHO = [
  { key: 'truth', label: 'the exact answer' },
  { key: 'qlearning', label: 'Q-learning' },
  { key: 'sarsa', label: 'SARSA' },
];

export function initGridWidget(data) {
  const T = data.truth;
  const world = makeWorld({ rows: T.rows, cols: T.cols, start: T.start, goal: T.goal });
  world.cliffCells = T.cliff;
  const solved = valueIteration(world);
  const readout = document.querySelector('#grid-readout');
  const controls = d3.select('#grid-controls');
  let who = 'truth';

  const chart = makeChart('#grid-chart', {
    width: 640, height: 300, margin: { top: 50, right: 20, bottom: 30, left: 20 },
  });
  const { g, w, h } = chart;
  const geo = drawGrid(g, world, { w, h });
  const specials = paintSpecials(g, world, geo);
  const overlay = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const V = solved.Q.map((row) => Math.max(...row));
  const shade = d3.scaleLinear().domain([d3.min(V), d3.max(V)]).range(['#e8e2f2', '#ffffff']);

  function render() {
    overlay.selectAll('*').remove();
    paintValues(geo.layer, world, geo, V, shade);
    specials.raise();
    const path = who === 'truth' ? greedyPath(world, solved.Q).path : data.learners[who].path;
    if (who === 'truth') paintArrows(overlay, world, geo, solved.Q, null);
    paintPath(overlay, geo, path, who === 'sarsa' ? 'var(--anchor)' : 'var(--primary)');
    const total = who === 'truth' ? T.return : data.learners[who].path_return;
    title.text(`${WHO.find((d) => d.key === who).label}: `
      + `${path.length - 1} steps, worth ${total}`);
    sub.text(who === 'truth'
      ? `${solved.sweeps} sweeps; where two moves are equally good, both are drawn`
      : `the most common route over ${data.learners[who].seeds} seeds, `
        + `learnt ${(data.learners[who].path_share * 100).toFixed(0)}% of the time`);

    const S = data.learners.sarsa;
    const Q = data.learners.qlearning;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>steps</th><th>worth</th><th>learnt this often</th>
            <th>reaches the goal</th></tr>
        <tr><td>${who === 'truth' ? '<span class="value">' : ''}the exact answer${who === 'truth' ? '</span>' : ''}</td>
            <td>${T.path.length - 1}</td><td>${T.return}</td><td>by construction</td>
            <td>always</td></tr>
        <tr><td>${who === 'qlearning' ? '<span class="value">' : ''}Q-learning${who === 'qlearning' ? '</span>' : ''}</td>
            <td>${Q.path.length - 1}</td><td>${Q.path_return}</td>
            <td>${(Q.path_share * 100).toFixed(0)}%</td>
            <td>${Q.reaches_goal} of ${Q.seeds}</td></tr>
        <tr><td>${who === 'sarsa' ? '<span class="value">' : ''}SARSA${who === 'sarsa' ? '</span>' : ''}</td>
            <td>${S.path.length - 1}</td><td>${S.path_return}</td>
            <td>${(S.path_share * 100).toFixed(0)}%</td>
            <td>${S.reaches_goal} of ${S.seeds}</td></tr>
      </table>
      <div class="gen-note">
        Q-learning comes back with the route value iteration found,
        <span class="value">${Q.path_return}</span>, in
        ${(Q.path_share * 100).toFixed(0)}% of seeds. SARSA comes back with a
        route one row further from the drop, ${S.path_return}, four steps longer.
        Neither is a mistake, and the last column is why: SARSA is learning the
        value of the policy it is actually following, which explores
        ${(data.meta.eps * 100).toFixed(0)}% of the time, and a policy that
        sometimes steps at random should not walk along a cliff. Its table is only
        readable where it has been: in ${S.seeds - S.reaches_goal} of ${S.seeds}
        seeds a greedy reading of it does not reach the goal at all, because the
        entries it never touched still hold their initial zero, which beats
        everything it learnt.
      </div>`;
  }

  controls.selectAll('button').data(WHO).join('button')
    .attr('class', (d) => `atlas-btn${d.key === who ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      who = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === who ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render, solved };
}
