/* One outlier, two break-even points.
 *
 * The 300 blobs of the k-means article, plus a 301st point the reader pushes
 * away along a fixed ray. Both algorithms are re-run in the browser at every
 * position, and the curve underneath is the same experiment swept offline, so
 * the reader can see where they are on it. The two cliffs in that curve are
 * not decoration: each one is predicted by an arithmetic the readout shows.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { distanceMatrix, pam, kmeans, ari } from './relaxkit.js';
import { drawPoints, drawMeans, drawMedoids, legend, colourOf } from './draw.js';

export function initOutlierWidget(data) {
  const base = data.datasets.blobs.points;
  const truth = data.datasets.blobs.true_labels;
  const O = data.outlier;
  const grid = O.grid;

  /* ---- the picture ---- */
  const main = makeChart('#outlier-chart', {
    width: 640, height: 400, margin: { top: 34, right: 26, bottom: 44, left: 52 },
  });
  /* The frame zooms out in whole steps to keep the outlier in view for as long
     as that leaves the blobs recognisable, and then gives up and points at it.
     Watching 300 points shrink into a corner because of one of them is the
     section's argument, made without a word. */
  const BASE = { x: [-7.2, 6.2], y: [-6.2, 7.2] };
  const ZOOMS = [1, 2, 4, 8];
  const x = d3.scaleLinear().range([0, main.w]);
  const y = d3.scaleLinear().range([main.h, 0]);
  const dotG = main.g.append('g');
  const medG = main.g.append('g');
  const meanG = main.g.append('g');
  const outG = main.g.append('g');
  legend(main.g, [
    { colour: 'var(--squidink)', label: 'k-means, diamonds' },
    { colour: 'var(--primary)', label: 'medoids, rings' },
  ], { x0: 2, y0: -16, gap: 168 });

  /* ---- the swept curve underneath ---- */
  const cur = makeChart('#outlier-curve', {
    width: 640, height: 250, margin: { top: 26, right: 96, bottom: 52, left: 56 },
  });
  const cx = d3.scaleLog().domain([grid[0], grid[grid.length - 1]]).range([0, cur.w]);
  const cy = d3.scaleLinear().domain([-0.05, 1.02]).range([cur.h, 0]);
  drawGrid(cur.g, cx, cy, cur.w, cur.h, { xTicks: 5, yTicks: 5 });
  drawAxes(cur.g, cx, cy, cur.w, cur.h, { xTicks: 5, yTicks: 5, xFmt: d3.format('~s') });
  axisLabels(cur.g, cur.w, cur.h, { x: 'distance of the outlier', y: 'ari, the other 300' });

  const line = d3.line().x((d, i) => cx(grid[i])).y((d) => cy(d));
  const SERIES = [
    { key: 'km', colour: 'var(--squidink)', label: 'k-means', dash: null, dy: 0 },
    { key: 'pam', colour: 'var(--primary)', label: 'PAM here', dash: null, dy: 13 },
    { key: 'fast', colour: 'var(--aqua)', label: 'FasterPAM', dash: '6 4', dy: 26 },
  ];
  SERIES.forEach((s) => {
    cur.g.append('path').datum(O.sweep[s.key]).attr('fill', 'none')
      .attr('stroke', s.colour).attr('stroke-width', 2)
      .attr('stroke-dasharray', s.dash).attr('d', line);
    const tail = O.sweep[s.key][grid.length - 1];
    cur.g.append('text').attr('class', 'chart-annotation')
      .attr('x', cur.w + 6).attr('y', cy(tail) + s.dy)
      .attr('dy', '0.32em').style('font-size', '0.6rem').style('letter-spacing', '0.5px')
      .style('fill', s.colour).style('stroke-width', '3px')
      .text(s.label);
  });
  [['break_km', 'k-means gives up'], ['break_pam', 'PAM here'], ['break_fast', 'the objective']]
    .filter(([k]) => O[k])
    .forEach(([k, label], i) => {
      cur.g.append('line')
        .attr('x1', cx(O[k])).attr('x2', cx(O[k])).attr('y1', 0).attr('y2', cur.h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3').attr('opacity', 0.55);
      /* 0.62rem is the floor: below it these labels render under 5px on a
         375px screen, which is smaller than anything else on the site */
      cur.g.append('text').attr('class', 'chart-annotation')
        .attr('x', cx(O[k]) + 4).attr('y', 12 + i * 14).style('font-size', '0.62rem')
        .style('letter-spacing', '0.5px').text(`${O[k]}`);
    });
  const marker = cur.g.append('line')
    .attr('y1', 0).attr('y2', cur.h)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2).attr('opacity', 0.9);
  const dotKm = cur.g.append('circle').attr('r', 4).attr('fill', 'var(--squidink)');
  const dotPam = cur.g.append('circle').attr('r', 4).attr('fill', 'var(--primary)');

  /* ---- state ---- */
  const slider = document.querySelector('#outlier-d');
  const readout = document.querySelector('#outlier-readout');
  slider.min = 0;
  slider.max = grid.length - 1;
  slider.step = 1;
  slider.value = grid.indexOf(12) >= 0 ? grid.indexOf(12) : Math.round(grid.length * 0.05);

  const cleanMedoids = data.medoids.medoids;
  const cleanCenters = data.medoids.km_centers;

  function render() {
    const d = grid[+slider.value];
    const out = [O.origin[0] + O.dir[0] * d, O.origin[1] + O.dir[1] * d];
    const pts = base.concat([out]);

    const km = kmeans(pts, 3, { restarts: 20 });
    const D = distanceMatrix(pts);
    const pm = pam(D, 3);

    const aKm = ari(truth, km.labels.slice(0, 300));
    const aPam = ari(truth, pm.labels.slice(0, 300));

    const cx0 = (BASE.x[0] + BASE.x[1]) / 2;
    const cy0 = (BASE.y[0] + BASE.y[1]) / 2;
    const hx = (BASE.x[1] - BASE.x[0]) / 2;
    const hy = (BASE.y[1] - BASE.y[0]) / 2;
    const need = Math.max(Math.abs(out[0] - cx0) / hx, Math.abs(out[1] - cy0) / hy) * 1.06;
    const zoom = ZOOMS.find((z) => z >= need) || ZOOMS[ZOOMS.length - 1];
    x.domain([cx0 - hx * zoom, cx0 + hx * zoom]);
    y.domain([cy0 - hy * zoom, cy0 + hy * zoom]);
    drawGrid(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
    drawAxes(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });

    /* the 300 keep their own colours; the outlier is drawn separately so it is
       never mistaken for one of them */
    drawPoints(dotG, base, km.labels.slice(0, 300), x, y, { r: zoom > 2 ? 2.4 : 3.4 });
    drawMeans(meanG, km.centers, x, y);
    drawMedoids(medG, pm.medoids.map((i) => pts[i]), x, y, { r: 8 });

    const inside = out[0] <= x.domain()[1] && out[1] <= y.domain()[1];
    const pad = (x.domain()[1] - x.domain()[0]) * 0.03;
    const edge = inside ? out : [Math.min(out[0], x.domain()[1] - pad), Math.min(out[1], y.domain()[1] - pad)];
    const gsel = outG.selectAll('g.outlier').data([edge]);
    const genter = gsel.enter().append('g').attr('class', 'outlier');
    genter.append('circle').attr('r', 6).attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5);
    genter.append('text').attr('class', 'chart-annotation').attr('text-anchor', 'end')
      .attr('x', -12).attr('y', 4).style('font-size', '0.62rem');
    const gm = genter.merge(gsel);
    gm.attr('transform', (p) => `translate(${x(p[0])},${y(p[1])})`);
    gm.select('circle').attr('fill', colourOf(km.labels[300]));
    gm.select('text').text(inside
      ? `d = ${d}${zoom > 1 ? `, frame zoomed out ${zoom}x` : ''}`
      : `d = ${d}, off the canvas`);

    marker.attr('x1', cx(d)).attr('x2', cx(d));
    dotKm.attr('cx', cx(d)).attr('cy', cy(aKm));
    dotPam.attr('cx', cx(d)).attr('cy', cy(aPam));

    document.querySelector('#outlier-d-label').textContent = d;

    /* what the two objectives are being asked to pay, right now */
    const keepKm = d * d;
    const keepPam = d;
    const kept = pm.medoids.filter((i) => cleanMedoids.includes(i)).length;
    const moved = matchDistance(km.centers, cleanCenters);
    const alone = (labels) => labels.filter((l) => l === labels[300]).length === 1;

    readout.innerHTML =
      `<table class="rel-table">
        <tr><th></th><th>cost of keeping it</th><th>cost of the centre it would take</th>
            <th>ARI, the other 300</th></tr>
        <tr><td>k-means</td><td>d&sup2; = <span class="value">${fmt(keepKm)}</span></td>
            <td>${(O.J2 - O.J3).toFixed(0)} of squared distance</td>
            <td><span class="value">${aKm.toFixed(3)}</span></td></tr>
        <tr><td>k-medoids</td><td>d = <span class="value">${fmt(keepPam)}</span></td>
            <td>${(O.L2 - O.L3).toFixed(0)} of distance</td>
            <td><span class="value">${aPam.toFixed(3)}</span></td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `${alone(km.labels)
        ? `K-means has given a whole centre to that one point, so the 300 are sharing two: `
          + `two of the blobs are now one cluster.`
        : `K-means is still holding on, and its three centres have drifted `
          + `${moved.toFixed(2)} from where they sat with clean data.`} ` +
      `${alone(pm.labels)
        ? `PAM has done the same: its search cannot take the medoid back.`
        : `The medoids have not moved at all: ${kept} of the original three are still medoids, `
          + `and a medoid does not drift, because it is a point.`}` +
      `</div>`;
  }

  function fmt(v) {
    return v >= 1000 ? d3.format(',.0f')(v) : v.toFixed(v < 10 ? 2 : 1);
  }

  /* mean displacement of the three centres, matched to the clean ones by the
     best of the six pairings rather than by index, which is arbitrary */
  function matchDistance(a, b) {
    const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    return Math.min(...perms.map((p) => p.reduce((s, j, i) =>
      s + Math.hypot(a[i][0] - b[j][0], a[i][1] - b[j][1]), 0) / 3));
  }

  /* Rendered straight from the event rather than through requestAnimationFrame:
     a full re-run of both algorithms is about 50 ms here, browsers coalesce
     input events during a drag anyway, and a frame callback never fires at all
     when the page is not being composited, which is precisely the situation
     the audit harness runs in. */
  slider.addEventListener('input', render);
  render();
}
