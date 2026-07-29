/* Forty cities, and the same number of evaluations for both searchers.
 *
 * The tours drawn here are the best each method found, and their lengths are
 * recomputed in the page from the coordinates rather than read from the file,
 * so the two numbers under the picture are the picture.
 */
import { makeChart, equalScales } from '../../assets/js/chart.js';
import { tourLength } from './gakit.js';

export function initTspWidget(data) {
  const T = data.tsp;
  const readout = document.querySelector('#tsp-readout');
  const controls = d3.select('#tsp-controls');
  let who = 'two_opt';

  const chart = makeChart('#tsp-chart', {
    width: 640, height: 400, margin: { top: 50, right: 30, bottom: 30, left: 30 },
  });
  const { g, w, h } = chart;
  const { x, y } = equalScales(T.points.map(([a, b]) => [a, b]), w, h);
  const path = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2);
  g.selectAll('circle').data(T.points).join('circle')
    .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 3.6)
    .attr('fill', '#ffffff').attr('stroke', 'var(--secondary)').attr('stroke-width', 1.4);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function render() {
    const tour = T[who].tour;
    const here = tourLength(tour, T.points);
    path.attr('d', `M${tour.map((i) => `${x(T.points[i][0])},${y(T.points[i][1])}`).join('L')}Z`);
    const label = who === 'ga' ? 'the genetic algorithm' : 'two opt with restarts';
    title.text(`${label}: its best tour of ${T.seeds} runs, length ${here.toFixed(4)}`);
    sub.text(`${T.n} cities, ${T.budget.toLocaleString('en-US')} tour lengths `
      + 'evaluated by each run');

    const ga = T.ga;
    const opt = T.two_opt;
    const better = ga.mean < opt.mean ? 'the genetic algorithm' : 'two opt';
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>searcher</th><th>mean over ${T.seeds} runs</th><th>best run</th>
            <th>spread between runs</th></tr>
        <tr><td>${who === 'ga' ? '<span class="value">' : ''}order crossover, population 60${who === 'ga' ? '</span>' : ''}</td>
            <td>${ga.mean}</td><td>${ga.best}</td><td>${ga.spread}</td></tr>
        <tr><td>${who === 'two_opt' ? '<span class="value">' : ''}two opt with restarts${who === 'two_opt' ? '</span>' : ''}</td>
            <td>${opt.mean}</td><td>${opt.best}</td><td>${opt.spread}</td></tr>
        <tr><td>the best of 2,000 random tours</td>
            <td>${T.random.mean}</td><td>${T.random.best}</td><td></td></tr>
      </table>
      <div class="gen-note">
        At the same budget, ${better} is ahead on average, by
        <span class="value">${(Math.abs(ga.mean / opt.mean - 1) * 100).toFixed(2)}%</span>.
        The column that matters more is the last one: two opt varies by
        ${opt.spread} between runs and the genetic algorithm by
        ${ga.spread}, ${(ga.spread / opt.spread).toFixed(1)} times as much, so
        the genetic algorithm's best run (${ga.best}) is close to two opt's
        (${opt.best}) while its typical run is not. Both are far below the
        ${T.random.mean} that the best of two thousand random tours reaches,
        which is the only thing on this page that random guessing settles.
      </div>`;
  }

  controls.selectAll('button')
    .data([{ k: 'two_opt', t: 'two opt with restarts' }, { k: 'ga', t: 'the genetic algorithm' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.k === who ? '' : ' ghost'}`)
    .text((d) => d.t)
    .on('click', (_, d) => {
      who = d.k;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.k === who ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
