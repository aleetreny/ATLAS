/* Truncation, live and measured.
 *
 * Pull every style vector towards the average of the prior and the samples get
 * more typical and less varied, which is a trade rather than an improvement.
 * The row of sprites is generated here at the current setting; the curve
 * underneath is the generator's sweep, so the reader can see both what the
 * setting does to a picture and what it does to the two numbers.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { drawTiles } from './panels.js';
import { makeNoise } from './stylekit.js';

export function initTruncWidget(data, gen) {
  const M = data.meta;
  const rows = data.truncation.rows;
  const state = { i: rows.findIndex((r) => r.psi === 1.0) };
  const nb = M.blocks.length;

  /* one fixed set of latents, so that moving the slider changes the setting
     and not the sample */
  const zs = [];
  let s = 20250728;
  const rng = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let k = 0; k < 8; k++) {
    const z = new Float32Array(M.z);
    for (let i = 0; i < M.z; i++) {
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    zs.push(z);
  }

  /* the noise is fixed per column too, so moving the slider changes the
     truncation and nothing else */
  const noises = zs.map((_, k) => makeNoise(M.blocks, 4242 + k * 7919));

  const c = makeChart('#trunc-chart', {
    width: 620, height: 340, margin: { top: 36, right: 74, bottom: 56, left: 68 },
  });
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.psi)]).range([0, c.w]);
  const yF = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.fd) * 1.15]).range([c.h, 0]);
  const yD = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.size_sd) * 1.2]).range([c.h, 0]);
  drawGrid(c.g, x, yF, c.w, c.h, { xTicks: 6, yTicks: 5 });
  drawAxes(c.g, x, yF, c.w, c.h, { xTicks: 6, yTicks: 5 });
  c.g.append('g').attr('class', 'axis y right').attr('transform', `translate(${c.w},0)`)
    .call(d3.axisRight(yD).ticks(5))
    .call((sel) => sel.selectAll('text').style('fill', 'var(--cosmos)'));
  axisLabels(c.g, c.w, c.h, { x: 'how much of the spread is kept', y: 'distance from the data' });
  c.g.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)')
    .attr('x', -c.h / 2).attr('y', c.w + 56).attr('text-anchor', 'middle')
    .style('fill', 'var(--cosmos)').text('spread of sizes produced');
  c.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4)
    .attr('d', d3.line().x((r) => x(r.psi)).y((r) => yF(r.fd))(rows));
  c.g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2)
    .attr('stroke-dasharray', '5 3')
    .attr('d', d3.line().x((r) => x(r.psi)).y((r) => yD(r.size_sd))(rows));
  const marker = c.g.append('circle').attr('r', 5).attr('fill', 'var(--squidink)');

  function render() {
    const r = rows[state.i];
    const imgs = zs.map((z, k) => gen.fromZ(z, { psi: r.psi, noises: noises[k] }));
    drawTiles(document.querySelector('#trunc-tiles'), imgs, M.res, 8, { zoom: 3, gap: 5 });
    marker.attr('cx', x(r.psi)).attr('cy', yF(r.fd));
    document.querySelector('#trunc-psi-value').textContent = r.psi.toFixed(2);
    const full = rows[rows.findIndex((q) => q.psi === 1.0)];
    document.querySelector('#trunc-readout').innerHTML =
      `<p>At <span class="bold">${r.psi.toFixed(2)}</span> of the spread, the distance from the `
      + `data is <span class="bold">${r.fd.toFixed(2)}</span> against ${full.fd.toFixed(2)} at `
      + `full spread, the sizes produced have a standard deviation of `
      + `${r.size_sd.toFixed(2)} against ${full.size_sd.toFixed(2)}, and `
      + `${(r.hole * 100).toFixed(2)}% of samples land in the forbidden rectangle against `
      + `${(full.hole * 100).toFixed(2)}%. ${r.shapes} of the ${M.shapes.length} shapes still `
      + `appear.</p>`
      + `<p>The eight sprites above are the same eight latents at every setting, so what changes `
      + `along the slider is the setting rather than the sample.</p>`;
  }

  const slider = document.querySelector('#trunc-psi');
  slider.min = 0;
  slider.max = rows.length - 1;
  slider.step = 1;
  slider.value = state.i;
  slider.addEventListener('input', () => { state.i = +slider.value; render(); });
  render();
  return { state, render };
}
