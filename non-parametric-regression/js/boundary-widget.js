/* The boundary-bias demonstration.
 *
 * The companion notebook asserts that Nadaraya-Watson flattens at the edges of
 * the data and that local linear regression fixes it, but never shows it. This
 * widget shows it three ways at once: the two curves, the moving window that
 * produced a chosen prediction, and the error measured separately on the edges
 * of the range against the middle. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { nadarayaWatson, localLinear, kernelWeights, rmse } from './mathkit.js';

export function initBoundaryWidget(mpg) {
  const pts = mpg.points;
  const tr = mpg.train_idx;
  const te = mpg.test_idx;
  const xtr = tr.map((i) => pts[i].x);
  const ytr = tr.map((i) => pts[i].y);
  const xte = te.map((i) => pts[i].x);
  const yte = te.map((i) => pts[i].y);
  const grid = mpg.grid;

  const allX = pts.map((d) => d.x);
  const p10 = d3.quantile([...allX].sort(d3.ascending), 0.1);
  const p90 = d3.quantile([...allX].sort(d3.ascending), 0.9);
  const edgeMask = xte.map((v) => v < p10 || v > p90);

  // The bottom margin has to hold the weighting profile AND its caption; at 90
  // the caption fell 32px past the edge of the canvas and the profile 9px.
  const { g, w, h } = makeChart('#boundary-chart', {
    width: 660,
    height: 500,
    margin: { top: 30, right: 24, bottom: 132, left: 62 },
  });
  const x = d3.scaleLinear().domain([35, 235]).range([0, w]);
  const y = d3.scaleLinear().domain([5, 50]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });

  // shade the two edge regions the error table reports on
  const edgeG = g.append('g');
  for (const [a, b] of [[x.domain()[0], p10], [p90, x.domain()[1]]]) {
    edgeG.append('rect')
      .attr('x', x(a)).attr('width', x(b) - x(a)).attr('y', 0).attr('height', h)
      .attr('fill', 'var(--squidink)').attr('fill-opacity', 0.05);
  }
  edgeG.append('text').attr('class', 'chart-annotation')
    .attr('x', x((x.domain()[0] + p10) / 2)).attr('y', 16).attr('text-anchor', 'middle')
    .style('font-size', '0.62rem').text('edge');
  edgeG.append('text').attr('class', 'chart-annotation')
    .attr('x', x((p90 + x.domain()[1]) / 2)).attr('y', 16).attr('text-anchor', 'middle')
    .style('font-size', '0.62rem').text('edge');

  const windowG = g.append('g');
  const dotsG = g.append('g');
  const linesG = g.append('g');

  const dots = dotsG.selectAll('circle').data(tr.map((i) => pts[i])).join('circle')
    .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y))
    .attr('r', 3.6).attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.7);

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);
  const haloNW = linesG.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const pathNW = linesG.append('path').attr('fill', 'none').attr('stroke', 'var(--smile)').attr('stroke-width', 3.4);
  const haloLL = linesG.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const pathLL = linesG.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);

  // the local weighting profile, drawn under the plot
  const profH = 46;
  const prof = g.append('g').attr('transform', `translate(0,${h + 52})`);
  prof.append('line').attr('x1', 0).attr('x2', w).attr('y1', profH).attr('y2', profH)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
  prof.append('text').attr('class', 'chart-annotation').attr('x', 0).attr('y', profH + 18)
    .style('font-size', '0.62rem').style('letter-spacing', '0.5px').style('text-transform', 'none')
    .text('weight each car receives for the query above');
  const profPath = prof.append('path').attr('fill', 'var(--primary)').attr('fill-opacity', 0.25)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6);

  const queryLine = g.append('line').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.5).attr('stroke-dasharray', '5 4');
  const nwDot = g.append('circle').attr('r', 6.5).attr('fill', 'var(--smile)').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
  const llDot = g.append('circle').attr('r', 6.5).attr('fill', 'var(--primary)').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);

  const hSlider = document.querySelector('#bnd-h');
  const hLabel = document.querySelector('#bnd-h-label');
  const qSlider = document.querySelector('#bnd-q');
  const qLabel = document.querySelector('#bnd-q-label');
  const stats = document.querySelector('#boundary-stats');

  function render() {
    const band = +hSlider.value;
    /* Scale the window to the horsepower the cars actually span, not to the
     * padded axis domain. Using the axis made the printed h differ from the one
     * the article quotes, so the widget could never be dragged to the setting
     * its own paragraph describes. */
    const hh = (d3.max(allX) - d3.min(allX)) * band;
    hLabel.textContent = hh.toFixed(0);
    const q = +qSlider.value;
    qLabel.textContent = q.toFixed(0);

    const nw = nadarayaWatson(xtr, ytr, grid, hh);
    const ll = localLinear(xtr, ytr, grid, hh);
    haloNW.attr('d', line(grid.map((v, i) => [v, nw[i]])));
    pathNW.attr('d', haloNW.attr('d'));
    haloLL.attr('d', line(grid.map((v, i) => [v, ll[i]])));
    pathLL.attr('d', haloLL.attr('d'));

    // the window that produced the prediction at the chosen query
    const ws = kernelWeights(xtr, q, hh);
    const maxW = Math.max(...ws, 1e-9);
    const sorted = xtr.map((v, i) => [v, ws[i]]).sort((a, b) => a[0] - b[0]);
    profPath.attr('d',
      `M${x(sorted[0][0])},${profH}` +
      sorted.map((p) => `L${x(p[0])},${profH - (p[1] / maxW) * profH}`).join('') +
      `L${x(sorted[sorted.length - 1][0])},${profH}Z`);
    dots.attr('fill', (d, i) => (ws[i] / maxW > 0.05 ? 'var(--primary)' : 'var(--stone)'))
      .attr('fill-opacity', (d, i) => 0.25 + 0.75 * (ws[i] / maxW));

    queryLine.attr('x1', x(q)).attr('x2', x(q)).attr('y1', 0).attr('y2', h);
    const nwq = nadarayaWatson(xtr, ytr, [q], hh)[0];
    const llq = localLinear(xtr, ytr, [q], hh)[0];
    nwDot.attr('cx', x(q)).attr('cy', y(nwq));
    llDot.attr('cx', x(q)).attr('cy', y(llq));

    const nwTe = nadarayaWatson(xtr, ytr, xte, hh);
    const llTe = localLinear(xtr, ytr, xte, hh);
    const sub = (arr, mask) => arr.filter((_, i) => mask[i]);
    const eNW = rmse(yte, nwTe);
    const eLL = rmse(yte, llTe);
    const eNWe = rmse(sub(yte, edgeMask), sub(nwTe, edgeMask));
    const eLLe = rmse(sub(yte, edgeMask), sub(llTe, edgeMask));
    const mid = edgeMask.map((v) => !v);
    const eNWm = rmse(sub(yte, mid), sub(nwTe, mid));
    const eLLm = rmse(sub(yte, mid), sub(llTe, mid));

    /* At the narrowest windows local linear is very slightly behind, and
     * printing "the kernel average is -2% worse" was a sentence that argued
     * against itself. Say what is actually true in that regime. */
    const gapPct = (eNWe / eLLe - 1) * 100;
    const verdict =
      gapPct <= 0.5
        ? 'At the edges the two are indistinguishable: the window is still so narrow that the neighbourhood is barely one-sided. Widen it and watch them separate.'
        : `At the edges the kernel average is <span class="bold">${gapPct.toFixed(0)}%</span> worse. ` +
          'Widen the window and watch that number grow: the more one-sided the ' +
          'neighbourhood, the more an average of it flattens the trend.';
    stats.innerHTML =
      `<table class="np-table">
        <tr><th>held-out RMSE</th><th style="color:var(--smile)">nadaraya-watson</th><th style="color:var(--primary)">local linear</th></tr>
        <tr><td>middle 80%</td><td>${eNWm.toFixed(2)}</td><td>${eLLm.toFixed(2)}</td></tr>
        <tr><td>outer 20% (edges)</td><td style="color:var(--cosmos)">${eNWe.toFixed(2)}</td><td>${eLLe.toFixed(2)}</td></tr>
        <tr><td>everywhere</td><td>${eNW.toFixed(2)}</td><td>${eLL.toFixed(2)}</td></tr>
       </table>
       <div class="slider-label" style="margin-top:.7rem;line-height:1.45">${verdict}</div>`;
  }

  hSlider.addEventListener('input', render);
  qSlider.addEventListener('input', render);
  render();
}
