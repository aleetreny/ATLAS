/* Overlap, swept, with the diagnostics next to the damage.
 *
 * The two views exist because of what the sweep turned out to say. The
 * diagnostics do not fail monotonically: the effective sample size gets worse
 * and then better again while the bias keeps climbing, so a page that showed
 * only the diagnostics would be showing a reader something that recovers while
 * the answer does not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));

export function initOverlapWidget(data) {
  const host = document.querySelector('#overlap-widget');
  if (!host) return;
  const O = data.overlap;
  const T = data.trimming;
  const M = data.meta;
  const chart = makeChart('#overlap-chart', {
    /* El margen superior es el carril de las tres notas de serie de la vista
     * de recorte: con 36 la de arriba se salía del viewBox por 6 unidades. */
    width: 640, height: 410, margin: { top: 52, right: 34, bottom: 56, left: 68 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#overlap-readout');
  const label = document.querySelector('#overlap-label');
  const slider = document.querySelector('#overlap-slider');
  const views = d3.select('#overlap-views');
  let view = 'diagnostics';
  let i = 1;

  const buttons = [
    ['diagnostics', 'what the weights look like'],
    ['damage', 'what the estimate does'],
    ['trim', 'and what trimming costs'],
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
    g.selectAll('*').remove();
    const row = O[i];
    label.textContent = f2(row.scale);
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    if (view === 'trim') {
      const both = T.filter((r) => r.side !== 'upper');
      const upper = T.filter((r) => r.side === 'upper');
      const x = d3.scaleBand().domain(both.map((r) => pc(r.trim, 0)))
        .range([0, w]).padding(0.3);
      const y = d3.scaleLinear()
        .domain([Math.min(M.tau0 - 0.1, d3.min(T, (r) => Math.min(r.estimate, r.truth_kept)) - 0.05),
          d3.max(T, (r) => Math.max(r.estimate, r.truth_kept)) + 0.12])
        .range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      /* El nombre del eje baja por debajo del carril de los "% kept", que
       * viven en h + 34: con el desplazamiento por defecto se cruzaban. */
      axisLabels(g, w, h, { x: 'scores trimmed from each tail', y: 'effect', xOffset: 52 });
      g.append('line')
        .attr('x1', 0).attr('x2', w).attr('y1', y(M.tau0)).attr('y2', y(M.tau0))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 4');
      annotate(g, 4, y(M.tau0) - 8, `the effect for everybody, ${f2(M.tau0)}`,
        { anchor: 'start' }).style('font-size', '11.5px');
      both.forEach((r) => {
        const cx = x(pc(r.trim, 0)) + x.bandwidth() / 2;
        g.append('line')
          .attr('x1', cx).attr('x2', cx)
          .attr('y1', y(r.estimate)).attr('y2', y(r.truth_kept))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 2);
        g.append('circle').attr('cx', cx).attr('cy', y(r.estimate)).attr('r', 5)
          .attr('fill', 'var(--cosmos)');
        g.append('circle').attr('cx', cx).attr('cy', y(r.truth_kept)).attr('r', 5)
          .attr('fill', 'var(--primary)');
        g.append('text').attr('class', 'chart-note')
          .attr('x', cx).attr('y', h + 34).attr('text-anchor', 'middle')
          .style('font-size', '11.5px')
          .text(`${pc(r.kept, 0)} kept`);
      });
      /* El brazo de una sola cola va como línea, no como pareja de puntos: es
       * el control, y mezclarlo con los puntos haría creer que es otra
       * columna de lo mismo. */
      if (upper.length) {
        g.append('path')
          .attr('d', line(upper.map((r, k) => [
            x(pc(both[Math.min(k, both.length - 1)].trim, 0)) + x.bandwidth() / 2,
            y(r.truth_kept)])))
          .attr('fill', 'none').attr('stroke', 'var(--aqua)')
          .attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
        annotate(g, w - 4, -34, 'dashed: the truth when only the upper tail is cut',
          { anchor: 'end' }).style('font-size', '11.5px').attr('fill', 'var(--aqua)');
      }
      annotate(g, w - 4, -20, 'dot: the estimate', { anchor: 'end' })
        .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      annotate(g, w - 4, -6, 'dot: the truth for who is left', { anchor: 'end' })
        .style('font-size', '11.5px').attr('fill', 'var(--primary)');
      const some = both.find((r) => r.trim === 0.05) ?? both[2];
      const up = upper.length ? upper[upper.length - 1] : null;
      readout.innerHTML =
        `<p>Trimming works on the error and moves the target at the same time. At `
        + `${pc(some.trim, 0)} the estimate is ${f4(some.estimate)} and the truth for the `
        + `${pc(some.kept)} of rows that survive is `
        + `<span class="bold">${f4(some.truth_kept)}</span>. Those two dots sitting on top `
        + `of each other are the finding: cutting both tails did not move the target, `
        + `because the people dropped from the two ends have effects above and below the `
        + `average in equal measure. `
        + `${up ? `Cut only the upper tail and it does move, to `
          + `<span class="bold">${f4(up.truth_kept)}</span> at ${pc(up.trim, 0)}, which is `
          + `the dashed line.` : ''}</p>`;
      return;
    }

    const x = d3.scaleLinear().domain([O[0].scale, O[O.length - 1].scale]).range([0, w]);
    if (view === 'diagnostics') {
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'how separable the two groups are', y: 'fraction' });
      g.append('path').attr('d', line(O.map((r) => [x(r.scale), y(r.ess_frac)])))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      g.append('path').attr('d', line(O.map((r) => [x(r.scale), y(r.max_share * 10)])))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      O.forEach((r) => {
        g.append('circle').attr('cx', x(r.scale)).attr('cy', y(r.ess_frac)).attr('r', 3.6)
          .attr('fill', 'var(--primary)');
        g.append('circle').attr('cx', x(r.scale)).attr('cy', y(r.max_share * 10)).attr('r', 3.6)
          .attr('fill', 'var(--cosmos)');
      });
      annotate(g, 4, -24, 'effective sample size, as a fraction of the rows',
        { anchor: 'start' }).style('font-size', '11.5px').attr('fill', 'var(--primary)');
      annotate(g, 4, -4, 'share of the total weight in one row, times ten',
        { anchor: 'start' }).style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      readout.innerHTML =
        `<p>At separability <span class="bold">${f2(row.scale)}</span> the smallest distance `
        + `to a certain treatment is ${row.min_tail.toExponential(1)}, the single largest `
        + `weight carries ${pc(row.max_share, 2)} of the total, and the effective sample size `
        + `is <span class="bold">${pc(row.ess_frac)}</span> of the rows. Read the two curves `
        + `to the end before trusting either: past the middle of the sweep the effective `
        + `sample size climbs back, because once the groups barely overlap the fitted score `
        + `is near zero or one for almost everybody and the weights become uniform again. `
        + `The diagnostics recover. The next view shows what the estimate does over the same `
        + `range.</p>`;
    } else {
      const y = d3.scaleLinear()
        .domain([Math.min(0, d3.min(O, (r) => r.bias - r.sd)),
          d3.max(O, (r) => r.bias + r.sd) * 1.05]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'how separable the two groups are', y: 'error in the estimate' });
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 4');
      O.forEach((r) => {
        g.append('line')
          .attr('x1', x(r.scale)).attr('x2', x(r.scale))
          .attr('y1', y(r.bias - r.sd)).attr('y2', y(r.bias + r.sd))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 6)
          .attr('stroke-linecap', 'round');
      });
      g.append('path').attr('d', line(O.map((r) => [x(r.scale), y(r.bias)])))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      O.forEach((r) => {
        g.append('circle').attr('cx', x(r.scale)).attr('cy', y(r.bias)).attr('r', 4)
          .attr('fill', 'var(--cosmos)');
      });
      g.append('line').attr('x1', x(row.scale)).attr('x2', x(row.scale))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--stone)').attr('stroke-width', 1.4);
      annotate(g, 4, -14, 'bar: spread between samples', { anchor: 'start' })
        .style('font-size', '11.5px');
      readout.innerHTML =
        `<p>Same sweep, and this is what it costs. At separability `
        + `<span class="bold">${f2(row.scale)}</span> the estimate is off by `
        + `<span class="bold">${sgn(row.bias)}</span> with a spread between samples of `
        + `${f3(row.sd)}, on an effect of ${f2(M.tau0)}. The bias grows monotonically across `
        + `the whole sweep while the diagnostics in the other view turn around halfway, so a `
        + `weight histogram that looks healthier than it did is not evidence of anything. `
        + `What the far end of this slider means is that for a large part of the population `
        + `there is nobody of the other treatment to compare with, and no estimator can `
        + `invent one.</p>`;
    }
  }

  slider.addEventListener('input', (e) => { i = +e.target.value; render(); });
  render();
}
