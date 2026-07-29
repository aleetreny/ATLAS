/* The assumption, broken on a real series, with its control.
 *
 * The series is drawn with the misses marked rather than summarised, because
 * the failure has a shape: the intervals are not randomly too small, they are
 * too small at the end, which is exactly what a distribution that drifts does
 * to a calibration set.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initExchWidget(data) {
  const host = document.querySelector('#exch-widget');
  if (!host) return;
  const M = data.meta;
  const E = data.exchange;
  const chart = makeChart('#exch-chart', {
    width: 640, height: 400, margin: { top: 44, right: 26, bottom: 58, left: 68 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#exch-readout');
  const views = d3.select('#exch-views');
  let view = 'series';

  const buttons = [
    ['series', 'the intervals, month by month'],
    ['compare', 'in time order against shuffled'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const S = E.series;

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();

    if (view === 'compare') {
      const rows = [
        ['split in time order', E.chrono.coverage, 'var(--cosmos)'],
        ['the same rows, shuffled', E.shuffled.coverage, 'var(--primary)'],
      ];
      const y = d3.scaleBand().domain(rows.map((r) => r[0])).range([0, h]).padding(0.5);
      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h, { xFmt: d3.format('.0%'), yValues: [] });
      axisLabels(g, w, h, { x: 'coverage' });
      rows.forEach(([label, v, colour]) => {
        g.append('rect').attr('x', 0).attr('y', y(label))
          .attr('width', x(v)).attr('height', y.bandwidth())
          .attr('fill', colour).attr('fill-opacity', 0.85);
        g.append('text').attr('class', 'chart-note')
          .attr('x', 4).attr('y', y(label) - 8).style('font-size', '12px').text(label);
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(v) - 8).attr('y', y(label) + y.bandwidth() / 2 + 5)
          .attr('text-anchor', 'end').style('font-size', '13px')
          .attr('fill', 'var(--paper)')
          .text(pc(v));
      });
      const target = 1 - M.alpha;
      g.append('line').attr('x1', x(target)).attr('x2', x(target))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4');
      /* A la izquierda de la linea: al 90% del ancho no queda sitio a la
       * derecha y la etiqueta se salia 24 unidades. */
      annotate(g, x(target) - 8, 16, `asked for ${pc(target, 0)}`, { anchor: 'end' })
        .style('font-size', '12px');
      readout.innerHTML =
        `<p>Same series, same features, same model, same calibration size. The only `
        + `difference is which rows went into which block. Split in time order the intervals `
        + `cover <span class="bold">${pc(E.chrono.coverage)}</span>; shuffled, they cover `
        + `<span class="bold">${pc(E.shuffled.coverage)}</span> with a spread of `
        + `${f3(E.shuffled.coverage_sd)} over ${E.reps} shuffles. The reason is in the `
        + `residuals: they average ${f3(E.chrono.mean_abs_cal)} on the calibration block and `
        + `${f3(E.chrono.mean_abs_test)} on the test block, because the series is still `
        + `rising and the last third is not a random sample of the whole.</p>`;
      return;
    }

    const n = S.y.length;
    const idx = d3.range(n);
    const x = d3.scaleLinear().domain([0, n - 1]).range([0, w]);
    const lo = d3.min(S.y.concat(S.pred)) - S.q * 1.4;
    const hi = d3.max(S.y.concat(S.pred)) + S.q * 1.4;
    const y = d3.scaleLinear().domain([lo, hi]).nice().range([h, 0]);
    const ticks = d3.range(0, n, Math.max(1, Math.floor(n / 5)));
    drawGrid(g, x, y, w, h, { xValues: ticks });
    drawAxes(g, x, y, w, h, { xValues: ticks, xFmt: (v) => (S.labels[v] || '').slice(0, 4) });
    axisLabels(g, w, h, { x: 'month of the test block', y: 'parts per million' });

    const area = d3.area().x((d) => x(d)).y0((d) => y(S.pred[d] - S.q))
      .y1((d) => y(S.pred[d] + S.q));
    g.append('path').attr('d', area(idx))
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.18);
    const line = d3.line().x((d) => x(d)).y((d) => y(S.pred[d]));
    g.append('path').attr('d', line(idx))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 1.8);
    const missed = idx.filter((i) => Math.abs(S.y[i] - S.pred[i]) > S.q);
    g.append('g').selectAll('circle')
      .data(idx)
      .join('circle')
      .attr('cx', (d) => x(d)).attr('cy', (d) => y(S.y[d]))
      .attr('r', (d) => (missed.includes(d) ? 3.2 : 2))
      .attr('fill', (d) => (missed.includes(d) ? 'var(--cosmos)' : 'var(--squidink)'));
    annotate(g, 4, -26, 'band: the conformal interval', { anchor: 'start' })
      .style('font-size', '11.5px').attr('fill', 'var(--primary)');
    annotate(g, 4, -10, 'red dots: the months it missed', { anchor: 'start' })
      .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');

    readout.innerHTML =
      `<p>The band is a ${pc(1 - M.alpha, 0)} conformal interval, half width `
      + `<span class="bold">${f2(S.q)}</span> parts per million, calibrated on the middle `
      + `third of the series (${S.cal_labels[0]} to ${S.cal_labels[1]}) and applied to the `
      + `last third. It misses <span class="bold">${missed.length}</span> of the `
      + `${n} months, a coverage of ${pc(E.chrono.coverage)}. Look at where the misses are: `
      + `they are not scattered, they pile up as the series climbs away from the block the `
      + `interval was calibrated on. Exchangeability is the assumption that any ordering of `
      + `the rows is as likely as any other, and a rising series is the plainest possible `
      + `violation of it.</p>`;
  }

  render();
}
