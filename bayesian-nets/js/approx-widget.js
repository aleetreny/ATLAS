/* Six approximations, marked.
 *
 * The bars come second and the curves first, because the number that matters
 * here is a shape: two of the six produce a band of literally zero width in the
 * gap, and "0.000" in a table reads as a rounding artefact while a flat line
 * through the middle of the exact band does not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);

function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

export function initApproxWidget(data) {
  const host = document.querySelector('#approx-widget');
  if (!host) return;
  const D = data.data;
  const E = data.exact;
  const R = data.approx.rows;
  const K = data.approx.curves;
  const chart = makeChart('#approx-chart', {
    width: 640, height: 440, margin: { top: 52, right: 36, bottom: 62, left: 152 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;
  const readout = document.querySelector('#approx-readout');
  const views = d3.select('#approx-views');
  const by = Object.fromEntries(R.map((r) => [r.key, r]));
  let view = 'curve';
  let pick = 'map';

  const buttons = [
    ['curve', 'one method against the truth'],
    ['tv', 'distance to the exact answer'],
    ['sd', 'how much of the doubt survives'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  /* La segunda fila de botones solo tiene sentido en la vista de curvas, así
   * que se esconde entera en las otras dos en vez de quedarse muerta. */
  const picks = d3.select('#approx-widget').insert('div', '#approx-chart')
    .attr('class', 'controls-row').style('justify-content', 'center');
  picks.selectAll('button')
    .data(R)
    .join('button')
    .attr('class', (d) => (d.key === pick ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d.label)
    .on('click', (_, d) => { pick = d.key; render(); });

  const x = d3.scaleLinear().domain([D.xs[0], D.xs[D.xs.length - 1]]).range([0, w]);

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    picks.selectAll('button')
      .attr('class', (d) => (d.key === pick ? 'atlas-btn' : 'atlas-btn ghost'));
    picks.style('display', view === 'curve' ? null : 'none');
    clear();

    if (view === 'curve') {
      const c = K[pick];
      const row = by[pick];
      const eLo = E.pred_mean.map((m, i) => m - 2 * E.pred_sd[i]);
      const eHi = E.pred_mean.map((m, i) => m + 2 * E.pred_sd[i]);
      const aLo = c.mean.map((m, i) => m - 2 * c.sd[i]);
      const aHi = c.mean.map((m, i) => m + 2 * c.sd[i]);
      const y = d3.scaleLinear()
        .domain([d3.min(eLo.concat(aLo)) - 0.2, d3.max(eHi.concat(aHi)) + 0.2])
        .nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'x', y: 'y' });
      plot.append('path')
        .attr('d', d3.area().x((_, i) => x(D.xs[i])).y0((_, i) => y(eLo[i]))
          .y1((_, i) => y(eHi[i]))(E.pred_mean))
        .attr('fill', 'var(--primary)').attr('fill-opacity', 0.16);
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(E.pred_mean))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((_, i) => y(aHi[i]))(c.mean))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4');
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((_, i) => y(aLo[i]))(c.mean))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4');
      plot.append('path')
        .attr('d', d3.line().x((_, i) => x(D.xs[i])).y((v) => y(v))(c.mean))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      D.x.forEach((xi, i) => {
        plot.append('circle').attr('cx', x(xi)).attr('cy', y(D.y[i])).attr('r', 3)
          .attr('fill', 'var(--squidink)').attr('fill-opacity', 0.7);
      });
      note(g, 0, -34, 'filled: the exact posterior predictive').attr('fill', 'var(--primary)');
      note(g, 0, -19, `dashed: ${row.label}`).attr('fill', 'var(--cosmos)');
      readout.innerHTML =
        `<p><span class="bold">${row.label}</span>, drawn over the answer. Its distance to the `
        + `exact predictive is <span class="bold">${f4(row.tv_all)}</span> in total variation, `
        + `${f4(row.tv_in)} inside the gap and ${f4(row.tv_out)} outside it, and its band is `
        + `<span class="bold">${f3(row.sd_ratio_in)}</span> as wide as the true one where it `
        + `matters most. `
        + `${row.sd_ratio_in < 0.01
          ? 'That is not a small band. That is no band: this method reports a single function '
            + 'and calling it uncertainty quantification is a category error.'
          : row.sd_ratio_in > 1
            ? 'It is the only row that errs on the wide side, which is the safer direction to '
              + 'be wrong in and still a failure to reproduce.'
            : 'Narrow, which is the dangerous direction: the gap is exactly where a deployed '
              + 'model meets an input it has no business being confident about.'} `
        + `The centre is easier: its mean is off by ${f4(row.mean_abs)} on average, and every `
        + `method on this page gets that roughly right.</p>`;
      return;
    }

    const key = view === 'tv' ? 'tv_all' : 'sd_ratio_in';
    const rows = [...R].sort((a, b) => (view === 'tv' ? a[key] - b[key] : b[key] - a[key]));
    const y = d3.scaleBand().domain(rows.map((r) => r.short)).range([0, h]).padding(0.26);
    const xb = d3.scaleLinear()
      .domain([0, Math.max(view === 'tv' ? 0 : 1, d3.max(rows, (r) => r[key])) * 1.1])
      .nice().range([0, w]);
    drawGrid(g, xb, y, w, h);
    drawAxes(g, xb, y, w, h);
    axisLabels(g, w, h, {
      x: view === 'tv' ? 'total variation distance to the exact predictive'
        : 'fraction of the true spread, in the gap',
    });
    if (view === 'sd') {
      plot.append('line').attr('x1', xb(1)).attr('x2', xb(1)).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6)
        .attr('stroke-dasharray', '5 4');
      note(g, xb(1), -16, 'the right amount', { anchor: 'middle' }).attr('fill', 'var(--anchor)');
    }
    rows.forEach((r) => {
      const yy = y(r.short);
      plot.append('rect').attr('x', 0).attr('y', yy)
        .attr('width', Math.max(1, xb(r[key]))).attr('height', y.bandwidth())
        .attr('fill', r.key === 'map' || r.key === 'ensemble' ? 'var(--cosmos)'
          : 'var(--primary)')
        .attr('fill-opacity', 0.88);
      /* Dentro si el texto medido cabe dentro, y sin halo blanco cuando va en
       * blanco: con un umbral fijo sobre el largo de la barra, la etiqueta de
       * una barra corta se sale por la izquierda encima del eje. */
      const t = note(g, xb(r[key]) - 8, yy + y.bandwidth() / 2 + 4,
        view === 'tv' ? f4(r[key]) : f3(r[key]), { anchor: 'end', size: 12 });
      if (t.node().getComputedTextLength() + 12 > xb(r[key])) {
        t.attr('x', xb(r[key]) + 8).attr('text-anchor', 'start');
      } else {
        t.attr('fill', 'var(--paper)').attr('stroke', 'none');
      }
    });
    const best = [...R].sort((a, b) => a.tv_all - b.tv_all)[0];
    readout.innerHTML = view === 'tv'
      ? `<p>Total variation distance between each approximate predictive distribution and the `
        + `exact one, averaged over the test points. It is a distance between whole `
        + `distributions, so a method cannot buy a good score by getting the mean right and the `
        + `spread wrong. <span class="bold">${best.label}</span> wins at ${f4(best.tv_all)}. `
        + `The two red bars are the single best fit and the ensemble trained on the same data, `
        + `and they are the same bar to six decimals, which is the subject of the paragraph `
        + `below.</p>`
      : `<p>The same six, scored on the only thing a Bayesian treatment is bought for: the size `
        + `of the doubt where the data run out. One is the right answer. `
        + `${f3(by.map.sd_ratio_in)} means a flat line. `
        + `<span class="bold">${by.laplace.label}</span> is the only one past the mark, at `
        + `${f3(by.laplace.sd_ratio_in)}, which is a Gaussian fitted to the curvature at one `
        + `mode being too generous about a posterior that has two. Outside the gap the same `
        + `numbers are ${R.map((r) => f2(r.sd_ratio_out)).join(', ')}, in the order of the `
        + `table below, and the spread there is ${f2(E.gap_growth)} times smaller anyway.</p>`;
  }

  render();
}
