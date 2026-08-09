/* The prior, as functions and as consequences.
 *
 * A prior over weights is unreadable and a prior over functions is not, so the
 * first view draws twelve draws from each width. The second view is the one
 * that makes the section: the same assumption is invisible where the data are
 * and decides the whole answer where they are not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);

function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

export function initPriorWidget(data) {
  const host = document.querySelector('#prior-widget');
  if (!host) return;
  const P = data.prior;
  const S = data.prior_sweep;
  const chart = makeChart('#prior-chart', {
    width: 640, height: 420, margin: { top: 50, right: 40, bottom: 62, left: 82 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;
  const readout = document.querySelector('#prior-readout');
  const views = d3.select('#prior-views');
  let view = 'draws';

  const buttons = [
    ['draws', 'what each width believes'],
    ['effect', 'what it costs to be wrong about it'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    clear();

    if (view === 'draws') {
      const x = d3.scaleLinear().domain([P.xs[0], P.xs[P.xs.length - 1]]).range([0, w]);
      const lim = d3.max(P.draws, (d) => d3.max(d, Math.abs));
      const y = d3.scaleLinear().domain([-lim * 1.08, lim * 1.08]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'x', y: 'f(x)' });
      P.draws.forEach((d) => {
        plot.append('path')
          .attr('d', d3.line().x((_, i) => x(P.xs[i])).y((v) => y(v))(d))
          .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 1.6)
          .attr('stroke-opacity', 0.6);
      });
      note(g, 0, -16, `${P.draws.length} functions drawn from the prior at width ${P.scale}`);
      const row = P.rows.find((r) => r.scale === P.scale) || P.rows[P.rows.length - 1];
      readout.innerHTML =
        `<p>The prior is written down as a Gaussian on each weight, which tells a reader `
        + `nothing, so here it is as functions: ${P.draws.length} networks whose weights were `
        + `drawn from it, before any data. At width ${P.scale} a typical function swings `
        + `<span class="bold">${f2(row.amplitude)}</span> in amplitude with a standard `
        + `deviation of ${f2(row.sd)} and a steepest slope around ${f1(row.steep)}. That is the `
        + `belief the posterior falls back to wherever the data are silent, and the second `
        + `view is what happens when that assumption is changed: `
        + `${P.rows.map((r) => `width ${r.scale} swings ${f2(r.amplitude)}`).join(', ')}.</p>`;
      return;
    }

    /* Dos escalas en un panel, no dos paneles: el argumento es que la MISMA
     * fila cambia mucho en una columna y nada en la otra, y eso se lee de un
     * vistazo solo si comparten el eje x. */
    const x = d3.scalePoint().domain(S.map((r) => `width ${r.scale}`)).range([0, w]).padding(0.5);
    const y = d3.scaleLinear()
      .domain([0, d3.max(S, (r) => r.sd_gap) * 1.15]).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'width of the prior on each weight', y: 'predictive spread' });
    [['sd_gap', 'var(--primary)', 'in the gap'], ['sd_in', 'var(--cosmos)', 'where the data are']]
      .forEach(([key, colour, name], k) => {
        plot.append('path')
          .attr('d', d3.line().x((r) => x(`width ${r.scale}`)).y((r) => y(r[key]))(S))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.6);
        S.forEach((r) => {
          plot.append('circle').attr('cx', x(`width ${r.scale}`)).attr('cy', y(r[key]))
            .attr('r', 4.2).attr('fill', colour);
        });
        note(g, 0, -34 + k * 15, name).attr('fill', colour);
      });
    const bestP = S.reduce((p, c) => (c.log_evidence > p.log_evidence ? c : p), S[0]);
    plot.append('line')
      .attr('x1', x(`width ${bestP.scale}`)).attr('x2', x(`width ${bestP.scale}`))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '5 4');
    note(g, w, -19, `the evidence picks ${bestP.scale}`, { anchor: 'end' })
      .attr('fill', 'var(--anchor)');
    readout.innerHTML =
      `<p>Four widths, each one a full recomputation of the exact posterior over the same cube. `
      + `The lower line is the predictive spread where the data are: it moves from `
      + `${f4(S[0].sd_in)} to ${f4(S[S.length - 1].sd_in)} across a factor of `
      + `${S[S.length - 1].scale / S[0].scale} in the assumption, which is to say the data have `
      + `taken over. The upper line is the same quantity in the gap: `
      + `<span class="bold">${f3(S[0].sd_gap)} to ${f3(S[S.length - 1].sd_gap)}</span>, a `
      + `factor of ${f1(S[S.length - 1].sd_gap / S[0].sd_gap)}. The assumption nobody reports `
      + `decides the answer exactly where the answer is being asked for. The dashed line is the `
      + `width the evidence prefers, ${bestP.scale} at ${f2(bestP.log_evidence)}, against `
      + `${S.map((r) => f2(r.log_evidence)).join(', ')} across the row: a preference of `
      + `${f1(bestP.log_evidence - Math.min(...S.map((r) => r.log_evidence)))} in log units, `
      + `which is worth having and is not the same thing as being told the answer.</p>`;
  }

  render();
}
