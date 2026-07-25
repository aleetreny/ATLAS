/* Three RBF boundaries on the same rings: too smooth, right, and shattered.
 *
 * The readout leads with the gap between training and held-out accuracy rather
 * than with either number, because the entire lesson of the right-hand panel is
 * that the training number looks perfect while the model is broken.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initGammaWidget(data) {
  const R = data.rings;
  const surfaces = R.gamma_surfaces;
  const pts = R.points;

  const { g, w, h } = makeChart('#gamma-chart', {
    width: 660,
    height: 470,
    margin: { top: 34, right: 26, bottom: 56, left: 58 },
  });

  const x = d3.scaleLinear().domain(d3.extent(R.gx)).range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(R.gy)).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x₁', y: 'x₂' });

  const nx = R.gx.length;
  const ny = R.gy.length;
  const cellW = w / (nx - 1);
  const cellH = h / (ny - 1);

  const heat = g.append('g');
  const bndG = g.append('g');
  const dotsG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* Contours arrive in grid units and with y counting upward, so the path has
   * to be rescaled to the plot and flipped. */
  const path = d3.geoPath(
    d3.geoTransform({
      point(a, b) {
        this.stream.point((a / (nx - 1)) * w, h - (b / (ny - 1)) * h);
      },
    })
  );

  /* Held-out points are drawn hollow. The panel's whole argument is the gap
   * between what the model was shown and what it was tested on, and it cannot
   * make that argument while every point looks the same. */
  const isTrain = new Set(R.train_idx || []);
  dotsG.selectAll('circle').data(pts.map((p, i) => ({ ...p, tr: isTrain.has(i) }))).join('circle')
    .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y))
    .attr('r', (p) => (p.tr ? 3.4 : 4.2))
    .attr('fill', (p) => (p.tr ? (p.c ? 'var(--cosmos)' : 'var(--aqua)') : 'none'))
    .attr('stroke', (p) => (p.tr ? 'white' : p.c ? 'var(--cosmos)' : 'var(--aqua)'))
    .attr('stroke-width', (p) => (p.tr ? 0.7 : 1.7))
    .attr('opacity', 0.9);

  const legend = g.append('g').attr('transform', `translate(${w - 118},6)`);
  [['solid: trained on', true], ['hollow: held out', false]].forEach(([label, tr], i) => {
    const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
    row.append('circle').attr('r', tr ? 3.4 : 4.2)
      .attr('fill', tr ? 'var(--squidink)' : 'none')
      .attr('stroke', tr ? 'white' : 'var(--squidink)').attr('stroke-width', tr ? 0.7 : 1.7);
    row.append('text').attr('x', 11).attr('y', 3.5).attr('class', 'chart-annotation')
      .style('font-size', '0.56rem').style('text-transform', 'none').text(label);
  });

  const btnRow = document.querySelector('#gamma-buttons');
  const readout = document.querySelector('#gamma-readout');
  const buttons = surfaces.map((s, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 1 ? '' : ' ghost');
    b.textContent = `γ = ${s.gamma}`;
    b.addEventListener('click', () => show(i));
    btnRow.appendChild(b);
    return b;
  });

  function show(i) {
    const s = surfaces[i];
    buttons.forEach((b, k) => b.classList.toggle('ghost', k !== i));

    const cells = [];
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nx; k++) cells.push({ i: k, j, v: s.field[j * nx + k] });
    }
    const lim = d3.max(s.field, Math.abs);
    const colour = d3.scaleDiverging(d3.interpolateRdBu).domain([lim, 0, -lim]);
    heat.selectAll('rect').data(cells).join('rect')
      .attr('x', (c) => x(R.gx[c.i]) - cellW / 2)
      .attr('y', (c) => y(R.gy[c.j]) - cellH / 2)
      .attr('width', cellW + 1)
      .attr('height', cellH + 1)
      .attr('fill', (c) => colour(c.v))
      .attr('opacity', 0.55);

    bndG.selectAll('path').data([0]).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.6)
      .attr('d', path(d3.contours().size([nx, ny]).contour(s.field, 0)));

    const gap = s.train - s.test;
    title.text(`γ = ${s.gamma}: ${s.sv} of the ${isTrain.size || pts.length} training points are support vectors`);
    readout.innerHTML =
      `<table class="gc-table">
         <tr><th></th><th>training</th><th>held out</th><th>gap</th></tr>
         <tr><td>accuracy at γ = ${s.gamma}</td><td>${s.train.toFixed(3)}</td>` +
      `<td>${s.test.toFixed(3)}</td><td><span class="value">${gap >= 0 ? '+' : ''}${gap.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">${
        gap > 0.05
          ? `The training set says this model is <span class="bold">perfect</span> and one held-out point in ` +
            `${Math.round(1 / Math.max(1 - s.test, 1e-9))} disagrees. Every island in the picture is a single ` +
            `training point defending its own doorstep, and there is nothing between the islands but guesswork.`
          : s.test < 0.97
            ? `Too smooth. At this γ every point feels every other one, so the boundary is a single wide ` +
              `curve that cannot tighten around the inner ring: it already gets ` +
              `<span class="value">${s.test.toFixed(3)}</span> and everything it misses is on that inner edge. ` +
              `Not linear, just blunt.`
            : `The boundary is a closed curve around the inner ring, which is the shape the data actually has. ` +
              `Training and held-out accuracy agree to <span class="value">${Math.abs(gap).toFixed(3)}</span>, ` +
              `which is what it looks like when γ is right.`
      }</div>`;
  }

  show(1);
}
