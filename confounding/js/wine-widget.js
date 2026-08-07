/* The reversal, on bottles.
 *
 * Two views of one scatter, because the whole content of the paradox is that
 * the same points support two opposite statements depending on whether the
 * grouping is drawn or not. Showing them as two pictures side by side would
 * lose that: it has to be the same picture twice.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { slope } from './statkit.js';

const f3 = (v) => v.toFixed(3);
const COLOURS = ['var(--primary)', 'var(--cosmos)', 'var(--aqua)'];
const LEGEND_CHAR_W = 7.6;

export function initWineWidget(data) {
  const host = document.querySelector('#wine-widget');
  if (!host) return;
  const B = data.wine.best;
  const names = data.wine.class_names;
  const chart = makeChart('#wine-chart', {
    width: 640, height: 420, margin: { top: 34, right: 24, bottom: 56, left: 62 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#wine-readout');
  const views = d3.select('#wine-views');
  let view = 'pooled';

  const buttons = [
    ['pooled', 'all the bottles at once'],
    ['groups', 'one line per cultivar'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const pretty = (s) => s.replace(/_/g, ' ');

  function fitLine(xs, ys) {
    const b = slope(xs, ys);
    const mx = d3.mean(xs);
    const my = d3.mean(ys);
    const lo = d3.min(xs);
    const hi = d3.max(xs);
    return [[lo, my + b * (lo - mx)], [hi, my + b * (hi - mx)]];
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const x = d3.scaleLinear().domain(d3.extent(B.x)).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(B.y)).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, {
      x: `${pretty(B.xname)}, standardised`,
      y: `${pretty(B.yname)}, standardised`,
    });

    g.append('g')
      .selectAll('circle')
      .data(B.x.map((v, i) => [v, B.y[i], B.g[i]]))
      .join('circle')
      .attr('cx', (d) => x(d[0]))
      .attr('cy', (d) => y(d[1]))
      .attr('r', 3.4)
      .attr('fill', (d) => (view === 'groups' ? COLOURS[d[2]] : 'var(--anchor)'))
      .attr('fill-opacity', 0.75);

    const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
    if (view === 'pooled') {
      g.append('path')
        .attr('d', line(fitLine(B.x, B.y)))
        .attr('fill', 'none')
        .attr('stroke', 'var(--cosmos)')
        .attr('stroke-width', 2.6);
      annotate(g, w - 4, -14, `one line through all of them: slope ${f3(B.pooled)}`,
        { anchor: 'end' }).style('font-size', '12px');
    } else {
      [0, 1, 2].forEach((k) => {
        const xs = B.x.filter((_, i) => B.g[i] === k);
        const ys = B.y.filter((_, i) => B.g[i] === k);
        g.append('path')
          .attr('d', line(fitLine(xs, ys)))
          .attr('fill', 'none')
          .attr('stroke', COLOURS[k])
          .attr('stroke-width', 2.4);
      });
      /* Colocada midiendo, no estimando: "class_0: -0.297" no mide lo mismo
       * que "class_2: -0.158" y un paso fijo de 108 unidades los solapaba. Se
       * pregunta el ancho real y se avanza con él, que es la regla que el
       * artículo de las cestas ya pagó con la etiqueta de su árbol. */
      const legend = g.append('g');
      let cursor = 0;
      [0, 1, 2].forEach((k) => {
        legend.append('circle')
          .attr('cx', cursor + 4).attr('cy', -18).attr('r', 4)
          .attr('fill', COLOURS[k]);
        const label = legend.append('text')
          .attr('class', 'chart-note')
          .attr('x', cursor + 14).attr('y', -14)
          .style('font-size', '12px')
          .text(`${names[k]}: ${f3(B.within[k])}`);
        cursor += 14 + label.text().length * LEGEND_CHAR_W + 20;
      });
    }
  }

  const agree = B.within.filter((v) => Math.sign(v) === Math.sign(B.pooled)).length;
  readout.innerHTML =
    `<p>Pooled, the slope is <span class="bold">${f3(B.pooled)}</span>. Split by cultivar, `
    + `the three slopes are <span class="bold">${B.within.map(f3).join(', ')}</span>, and `
    + `${agree === 0 ? 'not one of them' : `${agree} of them`} has the pooled sign. Nothing `
    + `was fabricated to make that happen: these are ${data.wine.n} real bottles, and the `
    + `pair was picked by testing every ordered pair of the ${data.wine.p} chemical columns `
    + `and keeping the reversal with the largest effect. The cultivar is doing what a `
    + `confounder does. It sets the level of both chemicals at once, so comparing bottles `
    + `across cultivars compares grape varieties while looking like it compares chemistry.</p>`;
  render();
}
