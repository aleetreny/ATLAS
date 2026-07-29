/* The surrogate against the dartboard, on the quantity that can separate them.
 *
 * Two views because the obvious one is nearly blind. "Best score after thirty
 * evaluations" is an average over runs that have all nearly finished, so it
 * compresses every difference into the fourth decimal. "How many evaluations
 * until you reach the target" spreads the same runs out, and it is also the
 * question anyone paying for evaluations is actually asking.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const SERIES = [
  ['random', 'var(--squidink)', '7 4'],
  ['noisy', 'var(--primary)', null],
  ['noiseless', 'var(--cosmos)', '2 3'],
];
const NAMES = {
  random: 'random draws',
  noisy: 'gaussian process, told the score is noisy',
  noiseless: 'gaussian process, told it is exact',
};

export function initBayesWidget(data) {
  const G = data.gp;
  const buttons = document.querySelector('#bayes-buttons');
  const readout = document.querySelector('#bayes-readout');
  const st = { view: 'trace' };

  const chart = makeChart('#bayes-chart', {
    width: 640, height: 380, margin: { top: 56, right: 30, bottom: 56, left: 74 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['trace', 'best score so far'],
    ['hits', 'evaluations to reach the target']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'trace') {
      const n = G.arms.random.trace.length;
      const x = d3.scaleLinear().domain([1, n]).range([0, w]);
      const all = SERIES.flatMap(([k]) => G.arms[k].trace);
      const y = d3.scaleLinear()
        .domain([d3.min(all) - 0.004, Math.max(G.best_cv, d3.max(all)) + 0.002])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'configurations evaluated', y: 'best cross validated score so far' });

      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(G.best_cv)).attr('y2', y(G.best_cv))
        .attr('stroke', 'var(--smile)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
      /* Below the line and anchored right: this line is the maximum of the
         chart, so anything placed above it lands in the legend. */
      layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
        .attr('y', y(G.best_cv) + 14).attr('text-anchor', 'end')
        .style('font-size', '11px').attr('fill', 'var(--smile)')
        .text(`the best of all ${(data.meta.grid ** 2).toLocaleString('en-US')} cells, `
          + `${G.best_cv.toFixed(4)}`);

      SERIES.forEach(([key, colour, dash]) => {
        const tr = G.arms[key].trace;
        layer.append('path')
          .attr('d', d3.line().x((_, i) => x(i + 1)).y((v) => y(v))(tr))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
          .attr('stroke-dasharray', dash);
      });
    } else {
      const x = d3.scaleBand().domain(SERIES.map(([k]) => k)).range([0, w]).padding(0.35);
      const vals = SERIES.map(([k]) => G.arms[k].hits.median || 0);
      const y = d3.scaleLinear().domain([0, d3.max(vals) * 1.25]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5 });
      axisLabels(g, w, h, { x: '', y: 'median evaluations to the target' });
      SERIES.forEach(([key, colour]) => {
        const v = G.arms[key].hits;
        layer.append('rect').attr('x', x(key)).attr('width', x.bandwidth())
          .attr('y', y(v.median)).attr('height', h - y(v.median)).attr('fill', colour);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(key) + x.bandwidth() / 2).attr('y', y(v.median) - 8)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(`${v.median.toFixed(1)}`);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(key) + x.bandwidth() / 2).attr('y', h + 18)
          .attr('text-anchor', 'middle').style('font-size', '11px')
          .text(`${(v.reached * 100).toFixed(0)}% of runs got there`);
      });
    }

    SERIES.forEach(([key, colour, dash], i) => {
      const ly = -40 + i * 15;
      layer.append('text').attr('class', 'chart-note').attr('x', 30).attr('y', ly)
        .style('font-size', '11px').attr('fill', colour).text(NAMES[key]);
      layer.append('line').attr('x1', 0).attr('x2', 24).attr('y1', ly - 4).attr('y2', ly - 4)
        .attr('stroke', colour).attr('stroke-width', 2.4).attr('stroke-dasharray', dash);
    });

    const gpBest = Math.max(G.arms.noisy.test, G.arms.noiseless.test);
    const diff = gpBest - G.arms.random.test;
    const sd = G.arms.random.test_sd;
    const cells = data.meta.grid * data.meta.grid;
    readout.innerHTML = `<div class="gen-note">After ${G.budget} evaluations of `
      + `${cells.toLocaleString('en-US')} cells, averaged over `
      + `${G.seeds} runs: the surrogate ends on a held out score of `
      + `<span class="bold">${gpBest.toFixed(4)}</span> and random draws end on `
      + `<span class="bold">${G.arms.random.test.toFixed(4)}</span>. `
      + (Math.abs(diff) <= sd
        ? `The ${Math.abs(diff).toFixed(4)} between them is well inside the `
          + `${sd.toFixed(4)} spread across runs, so on this surface the model of the `
          + `surface bought nothing measurable. `
        : `The ${Math.abs(diff).toFixed(4)} between them is larger than the `
          + `${sd.toFixed(4)} spread across runs. `)
      + `The reason is on the first chart of the page: `
      + `<span class="bold">${(G.near_optimal * 100).toFixed(1)}%</span> of the cells are `
      + `already within one fold to fold wobble of the best, so a blind draw expects to `
      + `hit one in ${G.expected_draws.toFixed(1)} tries. There is very little for a `
      + `smarter search to win.</div>`;
  }

  render();
  return { render, state: st };
}
