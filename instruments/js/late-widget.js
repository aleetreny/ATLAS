/* Whose effect the ratio is about, drawn as people.
 *
 * A hundred squares rather than a bar chart, because the point is about
 * membership rather than magnitude: the estimator answers about one block of
 * this grid and is silent about the other two, and no column in the data says
 * which square is which.
 */
import { makeChart, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initLateWidget(data) {
  const host = document.querySelector('#late-widget');
  if (!host) return;
  const L = data.late;
  const chart = makeChart('#late-chart', {
    width: 640, height: 440, margin: { top: 44, right: 26, bottom: 84, left: 40 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#late-readout');
  const views = d3.select('#late-views');
  let view = 'types';

  const buttons = [
    ['types', 'who is who'],
    ['reach', 'who the lever can move'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const COLOURS = ['var(--primary)', 'var(--cosmos)', 'var(--aqua)'];
  /* One hundred squares filled in the measured proportions, laid out in
   * reading order so that the block sizes are the shares. */
  const counts = L.shares.map((s) => Math.round(100 * s));
  while (counts.reduce((a, b) => a + b, 0) < 100) counts[0] += 1;
  while (counts.reduce((a, b) => a + b, 0) > 100) counts[0] -= 1;
  const cells = [];
  counts.forEach((n, k) => { for (let i = 0; i < n; i++) cells.push(k); });

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const side = Math.min(w, h) / 10;
    const x0 = (w - side * 10) / 2;

    cells.forEach((k, i) => {
      const r = Math.floor(i / 10);
      const c = i % 10;
      const reachable = k === 0;
      g.append('rect')
        .attr('x', x0 + c * side + 1.5).attr('y', r * side + 1.5)
        .attr('width', side - 3).attr('height', side - 3)
        .attr('fill', view === 'types' ? COLOURS[k] : (reachable ? COLOURS[0] : 'var(--stone)'))
        .attr('fill-opacity', view === 'types' ? 0.85 : (reachable ? 0.9 : 0.55));
    });

    /* La leyenda va apilada y no en fila: las tres entradas juntas miden 913
     * unidades en un lienzo de 640 y la tercera se salía 412 px por la
     * derecha. Tres renglones caben siempre, cualquiera que sea el texto. */
    L.names.forEach((name, k) => {
      const dim = view === 'reach' && k > 0;
      const yy = h + 16 + k * 18;
      g.append('rect')
        .attr('x', x0).attr('y', yy - 10).attr('width', 12).attr('height', 12)
        .attr('fill', dim ? 'var(--stone)' : COLOURS[k]);
      g.append('text').attr('class', 'chart-note')
        .attr('x', x0 + 18).attr('y', yy)
        .style('font-size', '12px')
        .text(`${name}, ${pc(L.shares[k], 0)}, effect ${f2(L.by_type[k])}`);
    });
    annotate(g, x0, -14, view === 'types'
      ? 'one hundred people, in the measured proportions'
      : 'the lever only moves the first block', { anchor: 'start' })
      .style('font-size', '12px');

    readout.innerHTML = view === 'types'
      ? `<p>One hundred people in the proportions this world was built with: `
        + `${pc(L.shares[0])} compliers, ${pc(L.shares[1])} who take the treatment whatever `
        + `happens and ${pc(L.shares[2])} who never take it. The three groups have three `
        + `different effects, ${L.by_type.map(f2).join(', ')}, and the average across `
        + `everybody is <span class="bold">${f4(L.ate)}</span>. Nothing in the data labels `
        + `anybody: the type is a property of what a person would do under a push that never `
        + `happened to them.</p>`
      : `<p>Pushing the lever changes the treatment for the first block and for nobody else, `
        + `which is what the bottom of the fraction measures: ${f3(L.first_stage)} of the `
        + `population switches. The top measures what the same push did to the outcome, `
        + `${f3(L.reduced_form)}. Divide, and the answer is `
        + `<span class="bold">${f4(L.wald)}</span>, which is the compliers' effect `
        + `(${f4(L.late)}) and not the population's (${f4(L.ate)}). The estimator is not `
        + `approximating the average across everybody. It is answering exactly, about the `
        + `grey blocks it never touched, nothing at all.</p>`;
  }

  render();
}
