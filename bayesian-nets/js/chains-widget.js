/* Four Metropolis chains against the grid.
 *
 * Drawn as curves and not as a table of one number, because the whole point is
 * that the disagreement lives in one place and is invisible everywhere else: a
 * worst case of 0.12 reported on its own reads as a broken quadrature, and the
 * picture says immediately that it is a gap.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f4 = (v) => v.toFixed(4);

function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

export function initChainsWidget(data) {
  const host = document.querySelector('#chains-widget');
  if (!host) return;
  const D = data.data;
  const E = data.exact;
  const C = data.chain_worst;
  const chains = data.chains;
  const curves = data.chain_curves;
  const chart = makeChart('#chains-chart', {
    width: 640, height: 420, margin: { top: 58, right: 34, bottom: 62, left: 78 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;
  const readout = document.querySelector('#chains-readout');
  const views = d3.select('#chains-views');
  let view = 'curves';

  const buttons = [
    ['curves', 'four chains and the grid'],
    ['diff', 'how far apart they are'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const x = d3.scaleLinear().domain([D.xs[0], D.xs[D.xs.length - 1]]).range([0, w]);
  const diff = D.xs.map((_, i) => d3.max(curves, (c) => Math.abs(c[i] - E.pred_mean[i])));
  const band = D.xs.map((_, i) => d3.max(curves, (c) => c[i]) - d3.min(curves, (c) => c[i]));

  /* La etiqueta del hueco va DENTRO del área de dibujo, no encima: arriba
   * chocaba con la leyenda de dos líneas, que empieza en el mismo x = 0 y llega
   * más allá del centro del panel. */
  function gapShade() {
    plot.append('rect').attr('x', x(E.gap_lo)).attr('y', 0)
      .attr('width', Math.max(0, x(E.gap_hi) - x(E.gap_lo))).attr('height', h)
      .attr('fill', 'var(--cosmos)').attr('fill-opacity', 0.12);
    note(g, x((E.gap_hi + E.gap_lo) / 2), h - 10, 'the gap', { anchor: 'middle' })
      .attr('fill', 'var(--cosmos)');
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    clear();

    if (view === 'diff') {
      const y = d3.scaleLinear()
        .domain([0, d3.max(diff.concat(band)) * 1.15]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'x', y: 'disagreement' });
      gapShade();
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(band))
        .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 2.4)
        .attr('stroke-dasharray', '5 4');
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(diff))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
      note(g, 0, -34, 'solid: the grid against the furthest chain')
        .attr('fill', 'var(--primary)');
      note(g, 0, -19, 'dashed: the four chains against each other')
        .attr('fill', 'var(--stone)');
      readout.innerHTML =
        `<p>The same comparison as one number per test point, and the dashed line is the one `
        + `that settles it. Outside the gap the grid and the furthest chain differ by at most `
        + `<span class="bold">${f4(C.tight)}</span>. At x = ${f2(C.worst_at)}, in the `
        + `middle of the gap, they differ by ${f4(C.mean)}, and the four chains differ `
        + `<span class="bold">from each other</span> by ${f4(C.spread)} at the same place. The `
        + `grid is not off to one side of the chains: it is inside their envelope at `
        + `${C.n_points - C.outside_band} of ${C.n_points} points, and where it is outside it `
        + `is outside by ${C.excursion.toExponential(0)}. A guard on this page originally `
        + `demanded agreement to 0.01 everywhere and fired here; the guard was wrong and the `
        + `measurement was the result.</p>`;
      return;
    }

    const y = d3.scaleLinear()
      .domain([d3.min(D.y) - 0.4, d3.max(D.y) + 0.4]).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'x', y: 'predictive mean' });
    gapShade();
    curves.forEach((c) => {
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(c))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.6)
        .attr('stroke-opacity', 0.75);
    });
    plot.append('path')
      .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(E.pred_mean))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.8);
    D.x.forEach((xi, i) => {
      plot.append('circle').attr('cx', x(xi)).attr('cy', y(D.y[i])).attr('r', 3)
        .attr('fill', 'var(--squidink)').attr('fill-opacity', 0.7);
    });
    note(g, 0, -34, `thick: the grid, all ${(data.meta.grid ** 3).toLocaleString('en-US')} `
      + 'settings summed').attr('fill', 'var(--primary)');
    note(g, 0, -19, `thin: ${chains.length} Metropolis chains, started apart`)
      .attr('fill', 'var(--cosmos)');
    readout.innerHTML =
      `<p>${chains.length} chains of ${(chains[0].n + 1).toLocaleString('en-US')} steps, `
      + `acceptance around ${f2(d3.mean(chains, (c) => c.rate))}, drawn over the sum the last `
      + `section computed. Away from the middle the five curves are one curve. In the gap they `
      + `come apart, and the reason is in where each chain lived: `
      + `${chains.map((c) => f2(c.theta_mean[0])).join(', ')} for the average input weight. `
      + `Two chains sat in one mode and two in the other, and since the two modes describe `
      + `<span class="bold">the same function</span>, that costs nothing wherever the data `
      + `decide the answer. In the gap the answer is decided by the shape of the posterior `
      + `instead, and there the chains disagree with each other as much as any of them `
      + `disagrees with the grid.</p>`;
  }

  render();
}
