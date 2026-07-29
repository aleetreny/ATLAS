/* Marginal against conditional, three ways.
 *
 * The grouped bars carry the argument: one interval for everybody produces two
 * bars on opposite sides of the line and a correct average, and the reader
 * should see the average being right at the same moment as they see nobody
 * getting it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initCondWidget(data) {
  const host = document.querySelector('#cond-widget');
  if (!host) return;
  const M = data.meta;
  const C = data.conditional;
  const chart = makeChart('#cond-chart', {
    width: 640, height: 410, margin: { top: 46, right: 30, bottom: 84, left: 72 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#cond-readout');
  const views = d3.select('#cond-views');
  let view = 'coverage';

  const buttons = [
    ['coverage', 'coverage by group'],
    ['width', 'width by group'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const groups = [['g0', 'the easy group'], ['g1', 'the hard group']];
  const wkeys = { g0: 'w0', g1: 'w1' };

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const x0 = d3.scaleBand().domain(C.rows.map((r) => r.key)).range([0, w]).padding(0.28);
    const x1 = d3.scaleBand().domain(groups.map((gr) => gr[0]))
      .range([0, x0.bandwidth()]).padding(0.16);
    const target = 1 - M.alpha;
    const y = view === 'coverage'
      ? d3.scaleLinear().domain([0.72, 1.02]).range([h, 0])
      : d3.scaleLinear().domain([0, d3.max(C.rows, (r) => Math.max(r.w0, r.w1)) * 1.14])
        .range([h, 0]);
    drawGrid(g, x0, y, w, h);
    drawAxes(g, x0, y, w, h, {
      yFmt: view === 'coverage' ? d3.format('.0%') : null, xValues: [],
    });
    axisLabels(g, w, h, {
      y: view === 'coverage' ? 'coverage inside the group' : 'interval width',
      xOffset: 66,
    });
    if (view === 'coverage') {
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(target)).attr('y2', y(target))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4');
      /* Same lane rule as the models chart, but this margin already holds the
         two group chips on the left: the line's name takes the right end. */
      annotate(g, w - 4, -16, `dashed line: asked for ${pc(target, 0)}`, { anchor: 'end' })
        .style('font-size', '11.5px');
    }
    C.rows.forEach((row) => {
      groups.forEach(([gk], k) => {
        const v = view === 'coverage' ? row[gk] : row[wkeys[gk]];
        const cx = x0(row.key) + x1(gk);
        g.append('rect')
          .attr('x', cx).attr('y', y(v))
          .attr('width', x1.bandwidth()).attr('height', h - y(v))
          .attr('fill', k === 0 ? 'var(--primary)' : 'var(--cosmos)')
          .attr('fill-opacity', 0.85);
        g.append('text').attr('class', 'chart-note')
          .attr('x', cx + x1.bandwidth() / 2).attr('y', y(v) - 7)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(view === 'coverage' ? f4(v) : f2(v));
      });
      const words = row.label.split(' ');
      const lines = [];
      let cur = '';
      words.forEach((word) => {
        if ((`${cur} ${word}`).trim().length > 18) { lines.push(cur.trim()); cur = word; } else cur += ` ${word}`;
      });
      lines.push(cur.trim());
      lines.forEach((ln, k) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x0(row.key) + x0.bandwidth() / 2).attr('y', h + 20 + k * 15)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(ln);
      });
    });
    let cursor = 0;
    groups.forEach(([, name], k) => {
      const label = annotate(g, cursor, -16, name, { anchor: 'start' })
        .style('font-size', '12px')
        .attr('fill', k === 0 ? 'var(--primary)' : 'var(--cosmos)');
      cursor += label.node().getComputedTextLength() + 22;
    });

    const by = Object.fromEntries(C.rows.map((r) => [r.key, r]));
    readout.innerHTML = view === 'coverage'
      ? `<p>One interval for everybody covers <span class="bold">${f4(by.plain.all)}</span> `
        + `overall, which is the promise, and it does it by covering `
        + `<span class="bold">${f4(by.plain.g0)}</span> of the easy group and `
        + `<span class="bold">${f4(by.plain.g1)}</span> of the hard one. Calibrating within `
        + `each group brings them to ${f4(by.mondrian.g0)} and ${f4(by.mondrian.g1)}, and `
        + `conformalised quantile regression gets ${f4(by.cqr.g0)} and ${f4(by.cqr.g1)} `
        + `without being told the groups exist. The bar to remember is the first pair: the `
        + `average was right the whole time.</p>`
      : `<p>What the repairs cost. One interval for everybody is `
        + `${f2(by.plain.w0)} wide for both groups, by construction. Calibrating per group `
        + `gives <span class="bold">${f2(by.mondrian.w0)}</span> to the easy one and `
        + `<span class="bold">${f2(by.mondrian.w1)}</span> to the hard one, which is the `
        + `honest picture of a problem where half the rows are harder than the other half. `
        + `Quantile regression pays more in width (${f2(by.cqr.w0)} and ${f2(by.cqr.w1)}) and `
        + `buys something the other cannot: it adapts without anybody naming the groups, `
        + `which matters when the hard cases are not a column you have.</p>`;
  }

  render();
}
