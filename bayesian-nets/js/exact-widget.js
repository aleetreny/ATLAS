/* The exact posterior, drawn three ways.
 *
 * The band comes first because it is what a reader came for, but the two views
 * behind it are what makes the band trustworthy: the posterior itself is a
 * two-humped thing over a cube, and the spread is not shaped the way the usual
 * sentence about it says.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);

/* Las leyendas van por .chart-note: annotate() va en versalitas y con dos
 * unidades de espaciado, que es media línea más por cada veinte caracteres. */
function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

export function initExactWidget(data) {
  const host = document.querySelector('#exact-widget');
  if (!host) return;
  const D = data.data;
  const E = data.exact;
  const chart = makeChart('#exact-chart', {
    width: 640, height: 430, margin: { top: 46, right: 34, bottom: 62, left: 76 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;
  const readout = document.querySelector('#exact-readout');
  const views = d3.select('#exact-views');
  let view = 'band';

  const buttons = [
    ['band', 'what the model believes'],
    ['spread', 'where the doubt actually is'],
    ['posterior', 'the posterior itself'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--primary').trim() || '#381a66';

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    clear();

    if (view === 'posterior') {
      /* El marginal conjunto de w y b, con v sumado fuera: S[i][j] es
       * (w = ax[i], b = ax[j]), así que la primera dimensión va en el eje x y
       * la segunda en el y. La rejilla exportada está adelgazada 1 de cada 4,
       * y el eje se reconstruye con el mismo paso. */
      const S = E.slice_wb;
      const n = S.length;
      const ax = E.ax.filter((_, i) => i % 4 === 0).slice(0, n);
      const x = d3.scaleLinear().domain([ax[0], ax[n - 1]]).range([0, w]);
      const y = d3.scaleLinear().domain([ax[0], ax[n - 1]]).range([h, 0]);
      const hi = d3.max(S, (row) => d3.max(row));
      const scale = d3.scaleSequential(d3.interpolateLab('#f4f4f5', accent)).domain([0, hi]);
      const cw = w / n;
      const ch = h / n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          plot.append('rect')
            .attr('x', i * cw).attr('y', h - (j + 1) * ch)
            .attr('width', cw + 0.4).attr('height', ch + 0.4)
            .attr('fill', scale(S[i][j]));
        }
      }
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'w, the input weight', y: 'b, the bias' });
      note(g, 0, -16, 'two humps, and they are the same function');
      readout.innerHTML =
        `<p>The joint posterior of the input weight and the bias, with the output weight summed `
        + `out. There are `
        + `<span class="bold">two</span> humps, and they are not two answers: flipping the sign `
        + `of all three weights leaves the function <code>v tanh(wx + b)</code> exactly unchanged, `
        + `so the two modes describe the same function. That is why the posterior mean of w is `
        + `${E.mean_w === 0 ? 'zero to every digit exported' : E.mean_w.toExponential(1)} and `
        + `why the mean of the weights is a useless summary of this posterior while the mean of `
        + `the predictions is not. It is also the thing a sampler has to cross, and the next `
        + `section measures whether one does.</p>`;
      return;
    }

    const x = d3.scaleLinear().domain([D.xs[0], D.xs[D.xs.length - 1]]).range([0, w]);

    if (view === 'spread') {
      const y = d3.scaleLinear().domain([0, d3.max(E.pred_sd) * 1.12]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'x', y: 'predictive spread' });
      /* Las tres regiones se pintan antes que la curva y con el mismo criterio
       * que las mide el generador, para que el lector vea de dónde sale cada
       * uno de los tres números del párrafo. */
      [[D.xs[0], E.gap_lo, 'var(--stone)', 0.16],
        [E.gap_lo, E.gap_hi, 'var(--cosmos)', 0.13],
        [E.gap_hi, D.xs[D.xs.length - 1], 'var(--stone)', 0.16]].forEach(([a, b, c, o]) => {
        plot.append('rect').attr('x', x(a)).attr('y', 0)
          .attr('width', Math.max(0, x(b) - x(a))).attr('height', h)
          .attr('fill', c).attr('fill-opacity', o);
      });
      [[D.x_lo, 'var(--anchor)'], [D.x_hi, 'var(--anchor)']].forEach(([v, c]) => {
        plot.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', 0).attr('y2', h)
          .attr('stroke', c).attr('stroke-width', 1.2).attr('stroke-dasharray', '4 4');
      });
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(E.pred_sd))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
      note(g, x((E.gap_lo + E.gap_hi) / 2), -16, 'the gap', { anchor: 'middle' })
        .attr('fill', 'var(--cosmos)');
      note(g, w, -16, 'beyond the data, both sides', { anchor: 'end' })
        .attr('fill', 'var(--stone)');
      readout.innerHTML =
        `<p>The same posterior, read as one number per test point. Where the points are it is `
        + `<span class="bold">${f4(E.regions.on_data)}</span>. Beyond the last point on either `
        + `side, which is where the sentence "uncertainty grows away from the data" says it `
        + `should flare, it is ${f4(E.regions.far)}: a factor of ${f2(E.far_ratio)}, which is `
        + `to say no change. A hyperbolic tangent saturates, and out there every function still in the `
        + `posterior has flattened at the same height. The growth is all in the gap between the `
        + `two clumps, ${f2(E.gap_lo)} to ${f2(E.gap_hi)}: `
        + `<span class="bold">${f4(E.regions.in_gap)}</span>, a factor of `
        + `${f2(E.gap_growth)}. Everything the rest of this page measures is measured in those `
        + `two places separately, because an approximation can be right in one and useless in `
        + `the other.</p>`;
      return;
    }

    const lo = E.pred_mean.map((m, i) => m - 2 * E.pred_sd[i]);
    const hi = E.pred_mean.map((m, i) => m + 2 * E.pred_sd[i]);
    const y = d3.scaleLinear()
      .domain([d3.min(lo.concat(D.y)) - 0.3, d3.max(hi.concat(D.y)) + 0.3])
      .nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'x', y: 'y' });
    plot.append('path')
      .attr('d', d3.area().x((_, i) => x(D.xs[i])).y0((_, i) => y(lo[i])).y1((_, i) => y(hi[i]))(
        E.pred_mean))
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.18);
    plot.append('path')
      .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(E.pred_mean))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
    D.x.forEach((xi, i) => {
      plot.append('circle').attr('cx', x(xi)).attr('cy', y(D.y[i])).attr('r', 3.6)
        .attr('fill', 'var(--squidink)').attr('fill-opacity', 0.85);
    });
    note(g, 0, -16, 'the band is two standard deviations of the exact predictive');
    readout.innerHTML =
      `<p>The posterior predictive of a network with three weights, computed by adding up `
      + `${(data.meta.grid ** 3).toLocaleString('en-US')} settings of those weights rather than `
      + `approximating the sum. The ${data.meta.n} points are the data; the band is two `
      + `standard deviations. Nothing here was fitted: every setting of the weights contributes `
      + `in proportion to how well it explains the points, and the band is what is left when `
      + `they are all averaged. The sum of those contributions is the evidence, `
      + `${E.mass.toExponential(3)}, log ${f4(E.log_evidence)}, and the whole point of the `
      + `article is that a number like that exists at all. Look at what the band does between `
      + `the two clumps, then press the middle button.</p>`;
  }

  render();
}
