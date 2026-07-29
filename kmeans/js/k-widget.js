/* Choosing k: the elbow that always falls against the silhouette that peaks.
 *
 * The slider refits k-means live at the chosen k (in the browser) and colours
 * the data, while the two curves from the generator sit alongside with the
 * current k marked. Seeing your five-cluster butchery of three clusters WHILE
 * the silhouette drops is the argument.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { lloyd, initPP, rng, silhouette } from './kmeanskit.js';
import { drawClusteredPoints, drawCenters, drawBorders } from './clusterdraw.js';

export function initKWidget(data) {
  const pts = data.blobs.points;
  const rows = data.choose_k;

  /* Two charts side by side inside one widget: the clustering, and the curves. */
  const wrap = d3.select('#k-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').attr('id', 'k-left').node();
  const right = wrap.append('div').attr('id', 'k-right').node();

  const L = makeChart(left, { width: 360, height: 360, margin: { top: 26, right: 14, bottom: 44, left: 44 } });
  const xs = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, L.w]);
  const ys = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([L.h, 0]);
  drawGrid(L.g, xs, ys, L.w, L.h, { xTicks: 4, yTicks: 4 });
  drawAxes(L.g, xs, ys, L.w, L.h, { xTicks: 4, yTicks: 4 });
  const borderG = L.g.append('g');
  const dotG = L.g.append('g');
  const centreG = L.g.append('g');

  const R = makeChart(right, { width: 360, height: 360, margin: { top: 26, right: 46, bottom: 44, left: 52 } });
  const kx = d3.scalePoint().domain(rows.map((r) => r.k)).range([0, R.w]).padding(0.4);
  const iy = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.inertia) * 1.08]).range([R.h, 0]);
  const sy = d3.scaleLinear().domain([0, 0.78]).range([R.h, 0]);
  drawAxes(R.g, kx, iy, R.w, R.h, { yTicks: 4, yFmt: d3.format('~s') });
  axisLabels(R.g, R.w, R.h, { x: 'k' });

  const li = d3.line().x((r) => kx(r.k)).y((r) => iy(r.inertia));
  const ls = d3.line().x((r) => kx(r.k)).y((r) => sy(r.silhouette));
  R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 3).attr('d', li(rows));
  R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('d', ls(rows));
  const lastRow = rows[rows.length - 1];
  R.g.append('text').attr('class', 'chart-annotation').attr('x', kx(lastRow.k))
    .attr('y', iy(lastRow.inertia) - 9).attr('text-anchor', 'end')
    .style('font-size', '0.7rem').style('text-transform', 'none')
    .attr('fill', 'var(--squidink)').attr('opacity', 0.7).text('inertia');
  R.g.append('text').attr('class', 'chart-annotation').attr('x', kx(lastRow.k))
    .attr('y', sy(lastRow.silhouette) - 9).attr('text-anchor', 'end')
    .style('font-size', '0.7rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)').text('silhouette');
  const marker = R.g.append('line').attr('y1', 0).attr('y2', R.h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
  const dotI = R.g.append('circle').attr('r', 4.5).attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
  const dotS = R.g.append('circle').attr('r', 4.5).attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.4);

  const slider = document.querySelector('#k-slider');
  const readout = document.querySelector('#k-readout');
  const cache = {};

  function render() {
    const k = +slider.value;
    document.querySelector('#k-label').textContent = k;

    if (!cache[k]) {
      /* Best of five k-means++ starts, so the picture at each k is the honest
       * best that k can do rather than one unlucky run. */
      let best = null;
      const r = rng(100 + k);
      for (let t = 0; t < 5; t++) {
        const path = lloyd(pts, initPP(pts, k, r));
        const end = path[path.length - 1];
        if (!best || end.inertia < best.inertia) best = end;
      }
      best.sil = silhouette(pts, best.labels, k);
      cache[k] = best;
    }
    const run = cache[k];

    drawClusteredPoints(dotG, pts, run.labels, xs, ys, { r: 3 });
    drawCenters(centreG, run.centers, xs, ys);
    drawBorders(borderG, run.centers, xs, ys, L.w, L.h);

    const row = rows.find((r2) => r2.k === k);
    marker.attr('x1', kx(k)).attr('x2', kx(k));
    dotI.attr('cx', kx(k)).attr('cy', iy(row.inertia));
    dotS.attr('cx', kx(k)).attr('cy', sy(row.silhouette));

    const peak = rows.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
    readout.innerHTML =
      `<table class="km-table">
         <tr><th>at k = ${k}</th><th>inertia</th><th>silhouette</th></tr>
         <tr><td>reference (sklearn)</td><td>${row.inertia.toLocaleString('en-US')}</td><td>${row.silhouette.toFixed(3)}</td></tr>
         <tr><td>this page's run</td><td>${run.inertia.toFixed(1)}</td><td>${run.sil.toFixed(3)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        k === peak.k
          ? `The silhouette peaks here. Tight clusters, far apart, and the picture on the left agrees.`
          : k < peak.k
            ? `Under-split: distinct groups are sharing clusters, and the silhouette says so before your eyes do.`
            : `Over-split: the inertia kept falling, as it always does, but the silhouette dropped from ` +
              `<span class="value">${peak.silhouette.toFixed(3)}</span> at k = ${peak.k} to ` +
              `<span class="value">${row.silhouette.toFixed(3)}</span> here. The extra clusters are cutting real ` +
              `groups in half, and only one of the two curves noticed.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
