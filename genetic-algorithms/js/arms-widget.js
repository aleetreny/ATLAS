/* Five searchers, four landscapes, one budget.
 *
 * The bars are success out of thirty seeds. The pair to compare is the last
 * two landscapes: the same trap function with the bit positions shuffled, which
 * changes nothing about the problem and everything about what crossover can do
 * with it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const LANDS = [
  { key: 'onemax', label: 'count the ones' },
  { key: 'royal', label: 'blocks, all or nothing' },
  { key: 'trap_tight', label: 'traps, side by side' },
  { key: 'trap_loose', label: 'traps, shuffled' },
];

export function initArmsWidget(data) {
  const readout = document.querySelector('#arms-readout');
  const controls = d3.select('#arms-controls');
  let land = 'trap_tight';

  const chart = makeChart('#arms-chart', {
    width: 640, height: 360, margin: { top: 62, right: 26, bottom: 92, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const arms = data.arms.onemax.rows.map((r) => r.arm);
  const x = d3.scalePoint().domain(arms).range([0, w]).padding(0.6);
  const y = d3.scaleLinear().domain([0, data.meta.seeds]).range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 4 });
  drawAxes(g, x, y, w, h, { yTicks: 4, xValues: [] });
  axisLabels(g, w, h, { y: `runs that found it, of ${data.meta.seeds}`,
    leftBudget: margin.left });

  /* the arm names go under the bars on two lines each: as tick labels they
     collide at 375px, and as one line they collide at 1440 too */
  const names = g.append('g');
  arms.forEach((arm) => {
    const label = data.arms.onemax.rows.find((r) => r.arm === arm).label;
    const words = label.split(' ');
    const mid = Math.ceil(words.length / 2);
    [words.slice(0, mid).join(' '), words.slice(mid).join(' ')].forEach((line, k) => {
      names.append('text').attr('class', 'chart-note')
        .attr('x', x(arm)).attr('y', h + 18 + k * 16).attr('text-anchor', 'middle')
        .style('font-size', '10.5px').text(line);
    });
  });

  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -42)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -24)
    .style('font-size', '11.5px');

  function render() {
    const D = data.arms[land];
    const rows = D.rows;
    g.selectAll('rect.bar').data(rows).join('rect').attr('class', 'bar')
      .attr('x', (d) => x(d.arm) - 24).attr('width', 48)
      .attr('y', (d) => y(d.success)).attr('height', (d) => h - y(d.success))
      .attr('fill', (d) => (d.success === 0 ? '#c9ced0' : 'var(--primary)'));
    g.selectAll('text.n').data(rows).join('text').attr('class', 'chart-note n')
      .attr('x', (d) => x(d.arm)).attr('y', (d) => y(d.success) - 6)
      .attr('text-anchor', 'middle').style('font-size', '11px')
      .text((d) => `${d.success}`);
    title.text(`${LANDS.find((l) => l.key === land).label}: `
      + `how many of ${data.meta.seeds} runs found the optimum`);
    sub.text(`budget ${data.meta.budget.toLocaleString('en-US')} evaluations, `
      + `population ${data.meta.pop}, optimum ${D.optimum}`);

    const winner = rows.reduce((a, b) => (b.success > a.success ? b : a), rows[0]);
    const tight = data.arms.trap_tight.rows.find((r) => r.arm === 'one_point');
    const loose = data.arms.trap_loose.rows.find((r) => r.arm === 'one_point');
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>searcher</th><th>found it</th><th>evals, median</th>
            <th>best reached, mean</th></tr>
        ${rows.map((rr) => `
          <tr><td>${rr.arm === winner.arm ? '<span class="value">' : ''}${rr.label}${rr.arm === winner.arm ? '</span>' : ''}</td>
              <td>${rr.success} of ${rr.of}</td>
              <td>${rr.median_evals === null ? 'never' : rr.median_evals.toLocaleString('en-US')}</td>
              <td>${rr.mean_best} of ${rr.best_possible}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        ${land === 'onemax'
    ? `Every one of them solves this, which is why it is the landscape a genetic
           algorithm is usually demonstrated on. Read the second column instead:
           the hill climber needs
           <span class="value">${rows.find((r) => r.arm === 'hill_climber').median_evals}</span>
           evaluations and the crossover arm needs
           ${rows.find((r) => r.arm === 'one_point').median_evals.toLocaleString('en-US')},
           a factor of
           ${Math.round(rows.find((r) => r.arm === 'one_point').median_evals
             / rows.find((r) => r.arm === 'hill_climber').median_evals)}.`
    : land === 'royal'
      ? `Now the hill climber gets nothing: a block pays only when all
             ${data.meta.block} of its bits are set, so flipping one bit never
             improves anything and the climb has nowhere to go. It reaches
             ${rows.find((r) => r.arm === 'hill_climber').mean_best} on average
             against an optimum of ${D.optimum}, all of it from the random
             restarts.`
      : land === 'trap_tight'
        ? `This is the landscape that is supposed to need crossover, and it does:
               one point crossover finds it
               <span class="value">${tight.success} times out of ${tight.of}</span>
               while uniform crossover finds it
               ${rows.find((r) => r.arm === 'uniform').success} and mutation alone
               ${rows.find((r) => r.arm === 'mutation_only').success}.`
        : `Same function, same optimum, same budget: only the bit positions were
               shuffled, and one point crossover fell from
               <span class="value">${tight.success} of ${tight.of}</span> to
               <span class="value">${loose.success}</span>. Mutation alone is
               unmoved (${data.arms.trap_tight.rows.find((r) => r.arm === 'mutation_only').success}
               and ${rows.find((r) => r.arm === 'mutation_only').success}), which
               is what tells you the problem itself did not get harder.`}
      </div>`;
  }

  controls.selectAll('button').data(LANDS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === land ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      land = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === land ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
