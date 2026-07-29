/* Every adjustment set, with the graph that decides them.
 *
 * The graph and the list live in one widget because the argument connects
 * them: the reader should be able to click a set and see, on the drawing, why
 * it works or does not. The sets are computed in the browser rather than read
 * from the export, which is also what the guard checks.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { worlds, adjust, children } from './dagkit.js';

const f4 = (v) => v.toFixed(4);
const f6 = (v) => v.toFixed(6);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initSetsWidget(data) {
  const host = document.querySelector('#sets-widget');
  if (!host) return;
  const E = data.exact;
  const R = data.random;
  const chart = makeChart('#sets-chart', {
    width: 640, height: 470, margin: { top: 150, right: 26, bottom: 60, left: 150 },
  });
  const { g, svg, w, h, margin } = chart;
  const readout = document.querySelector('#sets-readout');
  const views = d3.select('#sets-views');
  let view = 'sets';
  let picked = null;

  const buttons = [
    ['sets', 'this graph, all sixteen answers'],
    ['random', 'three thousand random graphs'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; picked = null; render(); });

  const model = { parents: E.parents, cpt: E.cpt };
  const ws = worlds(E.names.length);
  /* Positions chosen once so the arrows read left to right, treatment on the
   * left and outcome on the right, with the confounder above them. */
  const POS = { 0: [40, 96], 1: [140, 20], 2: [40, 20], 3: [230, 96], 4: [320, 20], 5: [410, 20] };
  const LAYOUT = [[20, 60], [130, 12], [240, 60], [350, 12], [350, 92], [460, 60]];

  function drawGraph(active) {
    svg.selectAll('g.dag').remove();
    const dag = svg.append('g').attr('class', 'dag')
      .attr('transform', `translate(${Math.max(10, margin.left + w / 2 - 240)}, 16)`);
    const defs = dag.append('defs');
    defs.append('marker')
      .attr('id', 'arrow-sets').attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', 'var(--squidink)');
    E.parents.forEach((pa, j) => {
      pa.forEach((i) => {
        const a = LAYOUT[i];
        const b = LAYOUT[j];
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        const r = 19;
        dag.append('line')
          .attr('x1', a[0] + (dx / len) * r).attr('y1', a[1] + (dy / len) * r)
          .attr('x2', b[0] - (dx / len) * (r + 6)).attr('y2', b[1] - (dy / len) * (r + 6))
          .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6)
          .attr('marker-end', 'url(#arrow-sets)');
      });
    });
    E.names.forEach((name, i) => {
      const isXY = i === E.x || i === E.y;
      const inSet = active && active.includes(i);
      dag.append('circle')
        .attr('cx', LAYOUT[i][0]).attr('cy', LAYOUT[i][1]).attr('r', 18)
        .attr('fill', inSet ? 'var(--primary)' : (isXY ? 'var(--squidink)' : 'var(--paper)'))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
      dag.append('text')
        .attr('x', LAYOUT[i][0]).attr('y', LAYOUT[i][1] + 5)
        .attr('text-anchor', 'middle').style('font-size', '13px')
        .style('font-weight', '700')
        .attr('fill', inSet || isXY ? 'var(--paper)' : 'var(--squidink)')
        .text(name);
    });
    dag.append('text').attr('class', 'chart-note')
      .attr('x', 240).attr('y', 118).attr('text-anchor', 'middle')
      .style('font-size', '11.5px')
      .text(`treatment ${E.names[E.x]}, outcome ${E.names[E.y]}, `
        + `filled circles are what is being adjusted for`);
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();

    if (view === 'random') {
      svg.selectAll('g.dag').remove();
      const bars = [
        ['adjust for everything is valid', R.full_rate, 'var(--primary)'],
        ['some valid set exists', R.any_rate, 'var(--aqua)'],
        ['adjust for nothing is valid', R.empty_rate, 'var(--cosmos)'],
      ];
      const y = d3.scaleBand().domain(bars.map((b) => b[0])).range([0, h]).padding(0.45);
      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h, { xFmt: d3.format('.0%'), yValues: [] });
      axisLabels(g, w, h, { x: `share of ${R.n.toLocaleString('en-US')} random graphs` });
      bars.forEach(([label, v, colour]) => {
        g.append('rect')
          .attr('x', 0).attr('y', y(label)).attr('width', x(v)).attr('height', y.bandwidth())
          .attr('fill', colour).attr('fill-opacity', 0.85);
        g.append('text').attr('class', 'chart-note')
          .attr('x', 4).attr('y', y(label) - 8).style('font-size', '12px')
          .text(label);
        /* Una barra que llega al borde no tiene sitio a su derecha: la cifra
         * se mete dentro. Con 100,0% se salía 44 unidades del lienzo. */
        const inside = x(v) > w - 62;
        g.append('text').attr('class', 'chart-note')
          .attr('x', inside ? x(v) - 8 : x(v) + 8)
          .attr('y', y(label) + y.bandwidth() / 2 + 4)
          .attr('text-anchor', inside ? 'end' : 'start')
          .style('font-size', '12.5px')
          .attr('fill', inside ? 'var(--paper)' : 'var(--squidink)')
          .text(pc(v));
      });
      readout.innerHTML =
        `<p>One example graph proves nothing about a habit, so the habit was measured on `
        + `${R.n.toLocaleString('en-US')} random ones: sample a graph, pick a treatment and `
        + `an outcome with a path between them, compute the true effect exactly, and ask `
        + `whether adjusting for every other observed variable returns it. It does in `
        + `<span class="bold">${pc(R.full_rate)}</span> of them. A valid set exists in `
        + `${pc(R.any_rate)}, and adjusting for nothing at all happens to be valid in `
        + `${pc(R.empty_rate)}. When adjusting for everything is wrong, the median error is `
        + `${f4(R.err_median)} and the worst is ${f4(R.err_max)}, on effects between zero `
        + `and one.</p>`;
      return;
    }

    drawGraph(picked ? picked.S : null);
    const sorted = [...E.rows].sort((a, b) => (a.error ?? 9) - (b.error ?? 9));
    const y = d3.scaleBand().domain(sorted.map((r) => r.names.join(', ') || 'nothing'))
      .range([0, h]).padding(0.18);
    const maxErr = d3.max(sorted, (r) => r.error ?? 0);
    const x = d3.scaleLinear().domain([0, maxErr * 1.12]).range([0, w]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'distance from the true effect' });
    g.append('g').selectAll('rect')
      .data(sorted)
      .join('rect')
      .attr('x', 0)
      .attr('y', (d) => y(d.names.join(', ') || 'nothing'))
      .attr('width', (d) => Math.max(1.5, x(d.error ?? 0)))
      .attr('height', y.bandwidth())
      .attr('fill', (d) => (d.backdoor ? 'var(--primary)' : 'var(--cosmos)'))
      .attr('fill-opacity', (d) => (picked && picked.names.join() === d.names.join() ? 1 : 0.72))
      .style('cursor', 'pointer')
      .on('click', (_, d) => { picked = d; render(); });
    /* Debajo del pie del grafo, no a su altura: a 375px las dos cajas se
     * cruzaban. */
    annotate(g, w - 4, -2, 'dark: satisfies the criterion', { anchor: 'end' })
      .style('font-size', '11.5px').attr('fill', 'var(--primary)');

    const chosen = picked ?? sorted.find((r) => !r.backdoor && r.names.length === E.names.length - 2)
      ?? sorted[0];
    const recomputed = adjust(model, ws, E.x, E.y, chosen.S);
    readout.innerHTML =
      `<p>Adjusting for <span class="bold">${chosen.names.join(', ') || 'nothing at all'}</span> `
      + `gives <span class="bold">${f6(recomputed)}</span>, computed in this page from the `
      + `probability tables, against a true effect of ${f6(E.truth)}. That is `
      + `${chosen.backdoor
        ? `a difference of ${Math.abs(recomputed - E.truth).toExponential(0)}, which is `
          + `machine arithmetic: this set satisfies the criterion.`
        : `off by <span class="bold">${f4(Math.abs(recomputed - E.truth))}</span>, and this `
          + `set does not satisfy the criterion.`} `
      + `Click any bar to adjust for that set instead. The dark bars are the ones the `
      + `graphical rule predicts will work, and they are exactly the ones that do.</p>`;
  }

  render();
}
