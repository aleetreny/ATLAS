/* Two value functions, one model, two answers.
 *
 * Drawn as paired bars per column rather than as two charts, because the whole
 * claim is about which columns move between the two and by how much, and that
 * is a comparison the reader should not have to hold in their head.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));
const pretty = (s) => s.replace(/_/g, ' ').replace('od280/od315 of diluted wines', 'od280/od315');

export function initTwoWidget(data) {
  const host = document.querySelector('#two-widget');
  if (!host) return;
  const M = data.meta;
  const T = data.two_shaps;
  const chart = makeChart('#two-chart', {
    width: 640, height: 430, margin: { top: 46, right: 30, bottom: 58, left: 168 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#two-readout');
  const views = d3.select('#two-views');
  let which = 0;

  const buttons = T.rows.slice(0, 4).map((r, k) => [String(k), `bottle ${k + 1}`]);
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (+d[0] === which ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { which = +d[0]; render(); });

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (+d[0] === which ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const row = T.rows[which];
    const order = row.interventional.map((_, j) => j)
      .sort((a, b) => Math.abs(row.conditional[b]) - Math.abs(row.conditional[a]));
    const names = order.map((j) => pretty(M.columns[j]));
    const y0 = d3.scaleBand().domain(names).range([0, h]).padding(0.24);
    const y1 = d3.scaleBand().domain(['i', 'c']).range([0, y0.bandwidth()]).padding(0.1);
    const vals = order.flatMap((j) => [row.interventional[j], row.conditional[j]]);
    const x = d3.scaleLinear()
      .domain([Math.min(0, d3.min(vals)) * 1.12, Math.max(0, d3.max(vals)) * 1.12])
      .nice().range([0, w]);
    drawGrid(g, x, y0, w, h);
    drawAxes(g, x, y0, w, h);
    axisLabels(g, w, h, { x: 'attribution' });
    g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
    order.forEach((j, k) => {
      [['i', row.interventional[j], 'var(--primary)'],
        ['c', row.conditional[j], 'var(--cosmos)']].forEach(([key, v, colour]) => {
        g.append('rect')
          .attr('x', Math.min(x(0), x(v))).attr('y', y0(names[k]) + y1(key))
          .attr('width', Math.max(1.5, Math.abs(x(v) - x(0))))
          .attr('height', y1.bandwidth())
          .attr('fill', colour).attr('fill-opacity', 0.85);
      });
    });
    annotate(g, 0, -28, 'breaking the correlation (interventional)', { anchor: 'start' })
      .style('font-size', '11.5px').attr('fill', 'var(--primary)');
    annotate(g, 0, -12, 'averaging over it (conditional)', { anchor: 'start' })
      .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');

    const gaps = order.map((j) => Math.abs(row.interventional[j] - row.conditional[j]));
    const worst = order[gaps.indexOf(Math.max(...gaps))];
    readout.innerHTML =
      `<p>The same model, the same bottle, the same axioms, and two bars per column. The `
      + `biggest gap here is <span class="bold">${pretty(M.columns[worst])}</span>: `
      + `${sgn(row.interventional[worst])} when the correlation is broken and `
      + `${sgn(row.conditional[worst])} when it is averaged over. Across the instances `
      + `measured the two versions differ by ${f4(T.mean_gap)} on average and up to `
      + `${f4(T.max_gap)}. The column the model barely uses, `
      + `<span class="bold">${pretty(T.unused_name)}</span>, gets `
      + `${f4(T.unused_interventional)} from the first and `
      + `${f4(T.unused_conditional)} from the second, because it correlates `
      + `${sgn(T.unused_corr, 3)} with ${pretty(T.partner_name)}, which the model does use. `
      + `Both numbers are exact. They are answers to different questions, and no plot ever `
      + `says which one was asked.</p>`;
  }

  render();
}
