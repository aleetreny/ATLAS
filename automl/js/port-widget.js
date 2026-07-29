/* A short list learned on three tables, spent on a fourth.
 *
 * This is the piece that makes an automated system different from a search: it
 * arrives knowing something. The list is built greedily on the three warm
 * tables, maximising the best of the list rather than the average of it, which
 * is why the second entry is chosen to cover the tables where the first fails.
 * Then it is spent, cold, on a table it has never seen, and compared against
 * the same number of blind draws.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initPortWidget(data) {
  const P = data.portfolio;
  const readout = document.querySelector('#port-readout');
  const cold = data.tables[P.cold];

  const chart = makeChart('#port-chart', {
    width: 640, height: 340, margin: { top: 52, right: 30, bottom: 58, left: 74 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const rows = P.rows;
  const pf = rows.find((r) => r.arm === 'portfolio');
  const rd = rows.find((r) => r.arm === 'random');
  const full = rows.find((r) => r.arm === 'full');

  /* the distribution the two are being read against: every pipeline in the
     catalogue, scored on the cold table's held out rows */
  const vals = cold.test;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const bins = 40;
  const hist = new Array(bins).fill(0);
  vals.forEach((v) => {
    const k = Math.min(bins - 1, Math.floor((v - lo) / (hi - lo || 1) * bins));
    hist[k] += 1;
  });

  const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
  const y = d3.scaleLinear().domain([0, Math.max(...hist) * 1.15]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 4 });
  drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 4, xFmt: d3.format('.2f') });
  axisLabels(g, w, h, { x: 'held out accuracy on the cold table', y: 'pipelines' });

  const bw = w / bins;
  hist.forEach((c, k) => {
    layer.append('rect').attr('x', x(lo + (hi - lo) * k / bins)).attr('width', bw + 0.5)
      .attr('y', y(c)).attr('height', h - y(c))
      .attr('fill', 'var(--squidink)').attr('opacity', 0.22);
  });

  [[full.test, 'the whole catalogue searched', 'var(--anchor)', -40],
    [pf.test, `the ${P.size} from the warm tables`, 'var(--primary)', -24],
    [rd.test, `${P.size} blind draws, on average`, 'var(--cosmos)', -8]]
    .forEach(([v, label, colour, ly]) => {
      layer.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', 0).attr('y2', h)
        .attr('stroke', colour).attr('stroke-width', 2);
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', ly)
        .style('font-size', '11px').attr('fill', colour)
        .text(`${label}: ${v.toFixed(4)}`);
    });

  const win = pf.test > rd.test;
  readout.innerHTML = `<div class="gen-note">The list, chosen without ever seeing `
    + `<span class="bold">${P.cold}</span>: `
    + `${P.picks.map((p) => `<span class="mono">${p}</span>`).join(', ')}. `
    + `Spent cold it reaches <span class="bold">${pf.test.toFixed(4)}</span> in `
    + `${P.size} evaluations. ${P.size} blind draws average `
    + `${rd.test.toFixed(4)} with a spread of ${rd.test_sd.toFixed(4)}, and `
    + `${(rd.beats * 100).toFixed(1)}% of them do at least as well, so the warm start `
    + (win
      ? `is worth ${(pf.test - rd.test).toFixed(4)} on average `
      : `is ${(rd.test - pf.test).toFixed(4)} behind them on average `)
    + `at the same price. Searching the whole catalogue, ${full.evals} evaluations, `
    + `reaches ${full.test.toFixed(4)}: the short list gets `
    + `${((pf.test - rd.test) / Math.max(1e-9, full.test - rd.test) * 100).toFixed(0)}% of `
    + `the way from a blind draw to the full search for `
    + `${(P.size / full.evals * 100).toFixed(1)}% of the cost.</div>`;

  return { render: () => {} };
}
