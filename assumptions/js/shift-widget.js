/* Mean shift: nobody says k, and every point walks uphill.
 *
 * The shaded background is the density mean shift actually climbs, which is
 * not a smooth kernel estimate but the flat one the algorithm uses: how many
 * points fall inside a window of the current bandwidth. The thin trails are
 * real trajectories, recomputed here every time the slider moves, and the
 * staircase underneath is the whole sweep, so the reader can see how much of
 * the bandwidth axis gives the right number of clusters and how much does not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { meanShift, ari } from './relaxkit.js';
import { drawPoints, colourOf } from './draw.js';

const CASES = [
  { key: 'blobs', label: 'three blobs' },
  { key: 'rings', label: 'rings' },
  { key: 'unequal', label: 'unequal' },
  { key: 'trap', label: 'the trap' },
];
const NX = 52;
const NY = 40;

export function initShiftWidget(data) {
  const state = { i: 0 };
  const main = makeChart('#shift-chart', {
    width: 640, height: 400, margin: { top: 32, right: 26, bottom: 44, left: 52 },
  });
  const stair = makeChart('#shift-stair', {
    width: 640, height: 230, margin: { top: 26, right: 30, bottom: 52, left: 56 },
  });
  const densG = main.g.append('g').attr('class', 'density');
  const trailG = main.g.append('g');
  const dotG = main.g.append('g');
  const modeG = main.g.append('g');
  const title = main.g.append('text').attr('class', 'chart-annotation')
    .attr('x', main.w / 2).attr('y', -12).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const btnRow = document.querySelector('#shift-cases');
  const slider = document.querySelector('#shift-bw');
  const readout = document.querySelector('#shift-readout');
  const btns = CASES.map((c, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = c.label;
    b.addEventListener('click', () => {
      state.i = i;
      render();
    });
    btnRow.appendChild(b);
    return b;
  });

  const bwGrid = data.shift.blobs.sweep.map((r) => r.bw);
  slider.min = 0;
  slider.max = bwGrid.length - 1;
  slider.step = 1;
  slider.value = bwGrid.indexOf(2.0) >= 0 ? bwGrid.indexOf(2.0) : Math.round(bwGrid.length / 2);

  function render() {
    const c = CASES[state.i];
    const S = data.shift[c.key];
    const ds = data.datasets[c.key];
    const pts = ds.points;
    const bw = bwGrid[+slider.value];
    const res = meanShift(pts, bw, { trails: 26 });
    const a = ari(ds.true_labels, res.labels);

    const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, main.w]);
    const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([main.h, 0]);
    drawGrid(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
    drawAxes(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });

    /* the flat-kernel density, on a coarse grid: exactly the quantity the
       algorithm hill-climbs, so the trails should run uphill on it */
    const [x0, x1] = x.domain();
    const [y0, y1] = y.domain();
    const values = new Float64Array(NX * NY);
    const bw2 = bw * bw;
    for (let gy = 0; gy < NY; gy++) {
      const py = y0 + ((gy + 0.5) / NY) * (y1 - y0);
      for (let gx = 0; gx < NX; gx++) {
        const px = x0 + ((gx + 0.5) / NX) * (x1 - x0);
        let n = 0;
        for (let i = 0; i < pts.length; i++) {
          const dx = pts[i][0] - px;
          const dy = pts[i][1] - py;
          if (dx * dx + dy * dy <= bw2) n++;
        }
        values[gy * NX + gx] = n;
      }
    }
    const top = d3.max(values) || 1;
    const thresholds = [0.12, 0.3, 0.55, 0.8].map((f) => f * top);
    const contours = d3.contours().size([NX, NY]).thresholds(thresholds)(values);
    const sx = (x(x1) - x(x0)) / NX;
    const sy = (y(y1) - y(y0)) / NY;
    densG.attr('transform', `translate(${x(x0)},${y(y0)}) scale(${sx},${sy})`);
    densG.selectAll('path').data(contours).join('path')
      .attr('d', d3.geoPath())
      .attr('fill', 'var(--primary)')
      .attr('opacity', (d, i) => 0.06 + i * 0.05)
      .attr('stroke', 'none');

    trailG.selectAll('path').data(res.paths).join('path')
      .attr('d', (p) => d3.line().x((q) => x(q[0])).y((q) => y(q[1]))(p))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 1).attr('opacity', 0.5);

    drawPoints(dotG, pts, res.labels, x, y, { r: 3.2 });
    const sel = modeG.selectAll('g.mode').data(res.centers, (d, i) => i);
    sel.exit().remove();
    const enter = sel.enter().append('g').attr('class', 'mode');
    enter.append('path').attr('d', 'M0,-9 L8,5 L-8,5 Z')
      .attr('stroke', 'white').attr('stroke-width', 2);
    const merged = enter.merge(sel);
    merged.attr('transform', (p) => `translate(${x(p[0])},${y(p[1])})`);
    merged.select('path').attr('fill', (p, i) => colourOf(i));

    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    document.querySelector('#shift-bw-label').textContent = bw.toFixed(2);
    title.text(`${res.centers.length} mode${res.centers.length === 1 ? '' : 's'} found, `
      + `nobody having said how many`);

    /* the staircase */
    const sxs = d3.scaleLinear().domain(d3.extent(bwGrid)).range([0, stair.w]);
    const maxModes = d3.max(S.sweep, (r) => r.modes);
    const sys = d3.scaleLog().domain([1, Math.max(2, maxModes)]).range([stair.h, 0]);
    drawGrid(stair.g, sxs, sys, stair.w, stair.h, { xTicks: 6, yTicks: 4 });
    drawAxes(stair.g, sxs, sys, stair.w, stair.h, {
      xTicks: 6, yTicks: 4, yFmt: d3.format('d'),
    });
    axisLabels(stair.g, stair.w, stair.h, { x: 'bandwidth', y: 'modes found' });
    stair.g.selectAll('path.stair').data([S.sweep]).join('path').attr('class', 'stair')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2)
      .attr('d', d3.line().curve(d3.curveStepAfter).x((r) => sxs(r.bw)).y((r) => sys(Math.max(1, r.modes))));
    stair.g.selectAll('line.truth').data([S.true_k]).join('line').attr('class', 'truth')
      .attr('x1', 0).attr('x2', stair.w).attr('y1', sys(S.true_k)).attr('y2', sys(S.true_k))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('stroke-dasharray', '4 4')
      .attr('opacity', 0.6);
    stair.g.selectAll('text.truth').data([S.true_k]).join('text').attr('class', 'truth chart-annotation')
      .attr('x', 4).attr('y', sys(S.true_k) - 5).style('font-size', '0.62rem')
      .text(`${S.true_k} real groups`);
    stair.g.selectAll('line.auto').data([S.auto_bw]).join('line').attr('class', 'auto')
      .attr('x1', sxs(S.auto_bw)).attr('x2', sxs(S.auto_bw)).attr('y1', 0).attr('y2', stair.h)
      .attr('stroke', 'var(--aqua)').attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
    stair.g.selectAll('text.auto').data([S.auto_bw]).join('text').attr('class', 'auto chart-annotation')
      .attr('x', sxs(S.auto_bw) + 4).attr('y', 12).style('font-size', '0.62rem')
      .style('text-transform', 'none').text(`the usual rule of thumb: ${S.auto_bw}`);
    stair.g.selectAll('line.here').data([bw]).join('line').attr('class', 'here')
      .attr('x1', sxs(bw)).attr('x2', sxs(bw)).attr('y1', 0).attr('y2', stair.h)
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);

    const rightK = S.sweep.filter((r) => r.modes === S.true_k);
    readout.innerHTML =
      `<table class="rel-table">
        <tr><th>bandwidth</th><th>modes</th><th>ARI</th><th>k-means, given the answer</th></tr>
        <tr><td>${bw.toFixed(2)}</td><td><span class="value">${res.centers.length}</span></td>
            <td><span class="value">${a.toFixed(3)}</span></td>
            <td>${board(data, c.key)}</td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      (rightK.length
        ? `${rightK.length} of the ${S.sweep.length} bandwidths in the sweep return exactly `
          + `${S.true_k} modes, between ${S.right_k_range[0]} and ${S.right_k_range[1]}. `
        : `No bandwidth anywhere in the sweep returns ${S.true_k} modes on this data: `
          + `one window size cannot describe it. `)
      + `The rule of thumb everyone uses, the mean distance to the nearest 30% of the data, `
      + `picks ${S.auto_bw} here, which finds ${S.auto_modes} and scores ${S.auto_ari.toFixed(3)}. `
      + `Best anywhere in the sweep: ${S.best.ari.toFixed(3)} at ${S.best.bw}.`
      + `</div>`;
  }

  const board = (d, key) => {
    const row = d.board.find((r) => r.name === key);
    return row ? row.kmeans.toFixed(3) : '';
  };

  /* Synchronous, like the outlier widget and for the same reason: a frame
     callback is never delivered to a page that is not being composited. */
  slider.addEventListener('input', render);
  render();
}
