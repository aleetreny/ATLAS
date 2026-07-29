/* The randomisation test, as a ladder.
 *
 * Drawn as lines against depth rather than as a table of five numbers because
 * the shape is the argument: two lines that stay flat at the top of the chart
 * while a third falls to the floor at the first step and stays there.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const f3 = (v) => v.toFixed(3);
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));

/* Ver la nota de score-widget.js: annotate() va en versalitas y con dos
 * unidades de espaciado, así que una leyenda de cuatro nombres se salía del
 * lienzo por sesenta unidades. */
function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

export function initSanityWidget(data) {
  const host = document.querySelector('#sanity-widget');
  if (!host) return;
  const N = data.sanity;
  const S = Object.fromEntries(data.scores.rows.map((r) => [r.key, r]));
  const chart = makeChart('#sanity-chart', {
    width: 640, height: 400, margin: { top: 52, right: 34, bottom: 60, left: 74 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#sanity-readout');

  const series = [
    ['gradient', 'plain gradient', 'var(--cosmos)'],
    ['grad_input', 'gradient times input', 'var(--smile)'],
    ['gradcam', 'Grad-CAM', 'var(--primary)'],
    ['input', 'the input image itself', 'var(--stone)'],
  ];
  const order = [...N].sort((a, b) => b.randomised_from - a.randomised_from);
  const x = d3.scalePoint()
    .domain(order.map((r) => `from ${r.randomised_from}`)).range([0, w]).padding(0.5);
  const y = d3.scaleLinear().domain([-0.2, 1.05]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, {
    x: 'layers randomised, from the top down',
    y: 'agreement with the trained map',
  });
  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2)
    .attr('stroke-dasharray', '4 4');
  note(g, w - 4, y(0) - 8, 'no relationship at all', { anchor: 'end' });

  const line = d3.line().x((d) => d[0]).y((d) => d[1]);
  /* La leyenda se mide y se parte en filas cuando no cabe, en vez de confiar en
   * que cuatro nombres quepan en una: medidos suman más que el ancho útil. */
  let cursor = 0;
  let row = 0;
  series.forEach(([key, name, colour]) => {
    g.append('path')
      .attr('d', line(order.map((r) => [x(`from ${r.randomised_from}`), y(r[key])])))
      .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.6);
    order.forEach((r) => {
      g.append('circle')
        .attr('cx', x(`from ${r.randomised_from}`)).attr('cy', y(r[key])).attr('r', 4)
        .attr('fill', colour);
    });
    const label = note(g, cursor, -32 + row * 15, name).attr('fill', colour);
    const width = label.node().getComputedTextLength();
    if (cursor > 0 && cursor + width > w) {
      row += 1;
      cursor = 0;
      label.attr('x', 0).attr('y', -32 + row * 15);
    }
    cursor += width + 18;
  });

  const full = order[order.length - 1];
  readout.innerHTML =
    `<p>Left to right, more of the network is replaced by random numbers of the same scale, `
    + `starting at the head and working down. By the right hand edge nothing of the trained `
    + `model is left. The plain gradient's map still agrees with the trained one at `
    + `<span class="bold">${sgn(full.gradient)}</span> and gradient times input at `
    + `<span class="bold">${sgn(full.grad_input)}</span>: those pictures were never about the `
    + `weights. Grad-CAM is at <span class="bold">${sgn(full.gradcam)}</span> from the first `
    + `step onwards. The flat grey line is the input image, which by definition cannot notice `
    + `and is drawn to show what "did not notice" looks like. Set this against the previous `
    + `section, where gradient times input scored ${f3(S.grad_input.pointing)} and Grad-CAM `
    + `${f3(S.gradcam.pointing)}: the higher score belongs to the method that is not looking `
    + `at the model.</p>`;
}
