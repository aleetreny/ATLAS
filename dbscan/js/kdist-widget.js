/* Reading eps off the data: the sorted 4th-neighbour-distance curve, with a
 * draggable horizontal line that IS the epsilon. Drag it and the scatter on
 * the right re-clusters live, so the classic "find the knee" advice stops
 * being folklore and becomes a dial with immediate consequences.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { dbscan, kdist, ari } from './dbkit.js';
import { drawLabelled } from './densdraw.js';

const K = 4;

export function initKdistWidget(data) {
  const ds = data.datasets.moons;
  const pts = ds.points;

  const wrap = d3.select('#kd-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  /* Left: the sorted k-distance curve. */
  const L = makeChart(left, { width: 360, height: 360, margin: { top: 26, right: 16, bottom: 46, left: 50 } });
  const dists = kdist(pts, K);
  const lx = d3.scaleLinear().domain([0, dists.length - 1]).range([0, L.w]);
  const ly = d3.scaleLinear().domain([0, Math.min(d3.max(dists), 1.4) * 1.05]).range([L.h, 0]);
  drawGrid(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 5 });
  drawAxes(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 5 });
  axisLabels(L.g, L.w, L.h, { x: 'points, sorted', y: `distance to neighbour ${K}` });
  L.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3)
    .attr('d', d3.line().x((d, i) => lx(i)).y((d) => ly(Math.min(d, ly.domain()[1])))(dists));

  const lineG = L.g.append('g').style('cursor', 'ns-resize');
  lineG.append('line').attr('x1', 0).attr('x2', L.w)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4).attr('stroke-dasharray', '7 5');
  lineG.append('rect').attr('x', 0).attr('width', L.w).attr('y', -10).attr('height', 20)
    .attr('fill', 'transparent');
  const lineLabel = L.g.append('text').attr('class', 'chart-annotation')
    .attr('x', 6).attr('text-anchor', 'start')
    .style('font-size', '0.7rem').style('text-transform', 'none').attr('fill', 'var(--cosmos)');

  /* Right: the moons under the dragged epsilon. */
  const R = makeChart(right, { width: 360, height: 360, margin: { top: 26, right: 16, bottom: 46, left: 44 } });
  const rx = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, R.w]);
  const ry = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([R.h, 0]);
  drawGrid(R.g, rx, ry, R.w, R.h, { xTicks: 4, yTicks: 4 });
  drawAxes(R.g, rx, ry, R.w, R.h, { xTicks: 4, yTicks: 4 });
  const dotG = R.g.append('g');
  const rTitle = R.g.append('text').attr('class', 'chart-annotation')
    .attr('x', R.w / 2).attr('y', -10).attr('text-anchor', 'middle').style('font-size', '0.7rem');

  const readout = document.querySelector('#kd-readout');
  let eps = 0.45;

  function render() {
    lineG.attr('transform', `translate(0,${ly(eps)})`);
    lineLabel.attr('y', ly(eps) - 7).text(`eps = ${eps.toFixed(2)}  (drag me)`);
    const r = dbscan(pts, eps, K + 1);
    const a = ari(r.labels, ds.true_labels);
    drawLabelled(dotG, pts, r.labels, r.role, rx, ry, { r: 2.8 });
    rTitle.text(`${r.nClusters} cluster${r.nClusters === 1 ? '' : 's'}, ${r.nNoise} noise, ARI ${a.toFixed(3)}`);

    const below = dists.filter((d) => d <= eps).length;
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">The curve sorts every point by how far away its ` +
      `${K}th neighbour is. Flat shelf on the left: points inside dense structure. Sharp climb on the ` +
      `right: points whose neighbourhoods are already emptying out. The classic advice is to set eps at ` +
      `the knee, and the drag line makes the advice testable: at ${eps.toFixed(2)}, ` +
      `<span class="value">${below}</span> of ${dists.length} points sit below the line (would-be cores), ` +
      `and the scatter shows what that buys. ${
        eps > 0.3 && eps < 0.62
          ? 'Anywhere on this shelf works: the knee is not a point, it is a plateau of good answers, which is why the folklore survives.'
          : eps <= 0.3
            ? `Below the shelf the moons shatter: the line is under the data's own internal spacing.`
            : 'Above the knee the gap between the moons is inside the radius, and the bridge appears.'
      }</div>`;
  }

  lineG.call(d3.drag().on('drag', (event) => {
    eps = Math.max(0.06, Math.min(ly.invert(event.y), ly.domain()[1]));
    render();
  }));
  render();
}
