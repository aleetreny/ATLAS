/* Why BIRCH exists: the measured bill for a full tree as n grows, against
 * summarise-first. Log-log time curves from the generator's benchmark, and
 * the honest other half: the ARI both methods get, because a speedup that
 * loses the answer is not a speedup.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

export function initBirchWidget(data) {
  const rows = data.scale;
  const big = data.scale_big;
  const { g, w, h } = makeChart('#birch-chart', {
    width: 660,
    height: 400,
    margin: { top: 30, right: 30, bottom: 56, left: 62 },
  });

  const x = d3.scaleLog().domain([rows[0].n * 0.8, big.n * 1.3]).range([0, w]);
  const allT = [...rows.map((r) => r.agg_seconds), ...rows.map((r) => r.birch_seconds), big.birch_seconds];
  const y = d3.scaleLog().domain([Math.max(d3.min(allT) * 0.6, 0.005), d3.max(allT) * 1.5]).range([h, 0]);
  /* a log axis given a tick count printed 9 and 10 on top of each other;
     decades cut to the domain, on grid and axis alike */
  const yVals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yValues: yVals });
  drawAxes(g, x, y, w, h, { xTicks: 5, yValues: yVals, xFmt: d3.format('~s'), yFmt: d3.format('~g') });
  axisLabels(g, w, h, { x: 'points clustered (log)', y: 'fit seconds (log)' });

  const line = (acc) => d3.line().x((r) => x(r.n)).y((r) => y(acc(r)));
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 3)
    .attr('d', line((r) => r.agg_seconds)(rows));
  const birchPts = [...rows.map((r) => ({ n: r.n, s: r.birch_seconds })), { n: big.n, s: big.birch_seconds }];
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3)
    .attr('d', d3.line().x((r) => x(r.n)).y((r) => y(r.s))(birchPts));
  for (const [set, colour, acc] of [[rows, 'var(--cosmos)', (r) => r.agg_seconds], [birchPts, 'var(--primary)', (r) => r.s]]) {
    g.selectAll(null).data(set).enter().append('circle')
      .attr('cx', (r) => x(r.n)).attr('cy', (r) => y(acc(r)))
      .attr('r', 4.5).attr('fill', colour).attr('stroke', 'white').attr('stroke-width', 1.4);
  }
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(rows[rows.length - 1].n)).attr('y', y(rows[rows.length - 1].agg_seconds) - 10)
    .attr('text-anchor', 'end').style('font-size', '0.56rem').style('text-transform', 'none')
    .attr('fill', 'var(--cosmos)').text('full ward tree');
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(big.n)).attr('y', y(big.birch_seconds) + 18)
    .attr('text-anchor', 'end').style('font-size', '0.56rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)').text('BIRCH');

  const readout = document.querySelector('#birch-readout');
  const last = rows[rows.length - 1];
  readout.innerHTML =
    `<table class="hier-table">
       <tr><th>n</th><th>ward</th><th>BIRCH</th><th>ward ARI</th><th>BIRCH ARI</th><th>subclusters</th></tr>
       ${rows.map((r) => `<tr><td>${r.n.toLocaleString('en-US')}</td><td>${r.agg_seconds.toFixed(2)}s</td>` +
         `<td>${r.birch_seconds.toFixed(2)}s</td><td>${r.agg_ari.toFixed(3)}</td>` +
         `<td>${r.birch_ari.toFixed(3)}</td><td>${r.birch_subclusters}</td></tr>`).join('')}
       <tr><td>${big.n.toLocaleString('en-US')}</td><td>out of reach</td><td>${big.birch_seconds.toFixed(2)}s</td>` +
         `<td></td><td>${big.birch_ari.toFixed(3)}</td><td>${big.birch_subclusters}</td></tr>
     </table>` +
    `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">The full tree pays for every ` +
    `pairwise distance and the bill curves upward: ${last.agg_seconds.toFixed(1)}s already at ` +
    `${last.n.toLocaleString('en-US')} points. BIRCH streams the data once into a tree of summary ` +
    `statistics (count, sum, sum of squares per node), then clusters the few hundred summaries instead ` +
    `of the points. Same answers on these blobs, a flat time curve, and one pass over data that never ` +
    `needs to fit in memory at once. The trade: those summaries are little spheres, so BIRCH inherits ` +
    `the round-cluster worldview this module keeps escaping. Summarise first, think later, works when ` +
    `the summary fits the shape.</div>`;
}
