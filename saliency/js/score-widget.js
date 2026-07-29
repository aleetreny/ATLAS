/* The scoreboard, with the controls drawn as controls.
 *
 * The two control rows are a different colour and sorted in with everything
 * else rather than parked at the bottom, because the point is that one of them
 * lands in the middle of the methods.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const f3 = (v) => v.toFixed(3);
const n0 = (v) => v.toLocaleString('en-US');
const CONTROLS = new Set(['input', 'random']);

/* Legends go through .chart-note and not through annotate(), which uppercases
 * and adds two units of letter spacing: at 11.5px that is about nine units per
 * character instead of seven, and a legend of fifty characters left the canvas
 * by fifty units before anyone measured it. */
function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

/* La etiqueta de una barra va dentro si cabe dentro y fuera si no, decidido
 * midiendo el texto. Con un umbral fijo sobre el largo de la barra,
 * "0.220 with 12 channels" entraba en una barra de 90 unidades y se salía por
 * la izquierda encima de la etiqueta del eje. */
function place(g, end, y, text, size) {
  const t = note(g, end - 8, y, text, { anchor: 'end', size });
  /* Dentro de la barra el halo blanco de .chart-note sobra: el texto también es
   * blanco, así que el halo lo engorda hasta parecer un contorno. */
  if (t.node().getComputedTextLength() + 12 <= end) {
    return t.attr('fill', 'var(--paper)').attr('stroke', 'none');
  }
  return t.attr('x', end + 8).attr('text-anchor', 'start').attr('fill', 'var(--squidink)');
}

export function initScoreWidget(data) {
  const host = document.querySelector('#score-widget');
  if (!host) return;
  const S = data.scores;
  const D = data.depth;
  const chart = makeChart('#score-chart', {
    width: 640, height: 420, margin: { top: 44, right: 40, bottom: 60, left: 190 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#score-readout');
  const views = d3.select('#score-views');
  let view = 'pointing';

  const buttons = [
    ['pointing', 'the brightest pixel is inside'],
    ['overlap', 'overlap with the mask'],
    ['depth', 'and where in the network'],
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

    if (view === 'depth') {
      /* La forma del mapa va en la etiqueta del eje y no detrás de la barra:
       * "(32 by 32, 12 channels)" detrás de una barra que llega al 0,82 se sale
       * del lienzo, y el margen izquierdo ya estaba pagado. */
      const name = (r) => `block ${r.layer + 1}, ${r.side} by ${r.side}`;
      const y = d3.scaleBand().domain(D.map(name)).range([0, h]).padding(0.34);
      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'the brightest pixel is inside the mask' });
      D.forEach((r) => {
        const yy = y(name(r));
        g.append('rect').attr('x', 0).attr('y', yy)
          .attr('width', x(r.pointing)).attr('height', y.bandwidth())
          .attr('fill', 'var(--primary)').attr('fill-opacity', 0.9);
        place(g, x(r.pointing), yy + y.bandwidth() / 2 + 4,
          `${f3(r.pointing)} with ${r.channels} channels`, 11.5);
      });
      readout.innerHTML =
        `<p>The same map computed at four depths. The first block sees a ${D[0].side} by `
        + `${D[0].side} feature map with ${D[0].channels} channels and scores `
        + `<span class="bold">${f3(D[0].pointing)}</span>; the last sees ${D[3].side} by `
        + `${D[3].side} with ${D[3].channels} and scores `
        + `<span class="bold">${f3(D[3].pointing)}</span>. Resolution went down by a factor `
        + `of ${D[0].side / D[3].side} and the score went up by a factor of `
        + `${(D[3].pointing / D[0].pointing).toFixed(1)}. That is the whole compromise behind `
        + `Grad-CAM in one picture: the layers that know what the thing is have already `
        + `thrown away most of where it is.</p>`;
      return;
    }

    const key = view;
    const rows = [...S.rows].sort((a, b) => b[key] - a[key]);
    const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, h]).padding(0.26);
    const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, {
      x: key === 'pointing' ? 'the brightest pixel is inside the mask'
        : 'overlap of the top pixels with the mask',
    });
    rows.forEach((r) => {
      const yy = y(r.label);
      g.append('rect').attr('x', 0).attr('y', yy)
        .attr('width', Math.max(1, x(r[key]))).attr('height', y.bandwidth())
        .attr('fill', CONTROLS.has(r.key) ? 'var(--cosmos)' : 'var(--primary)')
        .attr('fill-opacity', CONTROLS.has(r.key) ? 0.6 : 0.92);
      place(g, x(r[key]), yy + y.bandwidth() / 2 + 4, f3(r[key]), 12);
    });
    note(g, 0, -14, 'red: the two controls, which explain nothing')
      .attr('fill', 'var(--cosmos)');

    const by = Object.fromEntries(S.rows.map((r) => [r.key, r]));
    readout.innerHTML =
      `<p>${n0(S.n)} scenes, each with the exact mask of the object being explained. `
      + `${key === 'pointing'
        ? `Occlusion, the expensive one, lands inside the mask ${f3(by.occlusion.pointing)} `
          + `of the time and gradient times input ${f3(by.grad_input.pointing)}; Grad-CAM `
          + `${f3(by.gradcam.pointing)}; the plain gradient ${f3(by.gradient.pointing)}.`
        : `Gradient times input overlaps the mask ${f3(by.grad_input.overlap)}, occlusion `
          + `${f3(by.occlusion.overlap)}, Grad-CAM ${f3(by.gradcam.overlap)}.`} `
      + `Now read the red bars. Smooth noise is at ${f3(by.random[key])}, which is the floor, `
      + `and <span class="bold">the input image with no model at all</span> is at `
      + `${f3(by.input[key])}`
      + `${by.input[key] > by.gradient[key]
        ? `, which is above the plain gradient. `
        : `. `}`
      + `On a dataset where the object is the bright thing, any map that tracks brightness `
      + `scores well without explaining anything, and that is what the next section is `
      + `about.</p>`;
  }

  render();
}
