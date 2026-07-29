/* What deletions do to a graph, and what rebuilding costs to undo it.
 *
 * A graph index cannot really delete: the removed node's edges are what holds
 * its neighbours together, so the usual answer is a tombstone, and the search
 * pays for visiting rows that will be thrown away. The two curves are the same
 * queries against a marked index and against one rebuilt from what is left, so
 * the gap between them is exactly what the tombstones cost.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initChurnWidget(data) {
  const C = data.churn;
  const readout = document.querySelector('#churn-readout');

  const chart = makeChart('#churn-chart', {
    width: 640, height: 350, margin: { top: 54, right: 66, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const rows = C.rows;
  const x = d3.scaleLinear().domain([0, rows[rows.length - 1].deleted]).range([0, w]);
  const y = d3.scaleLinear().domain([Math.min(...rows.map((r) => r.tombstone_recall)) - 0.05, 1.02])
    .range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.deleted), yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xValues: rows.map((r) => r.deleted), yTicks: 5,
    xFmt: (v) => `${(v * 100).toFixed(0)}%`, yFmt: d3.format('.2f'),
  });
  axisLabels(g, w, h, { x: 'share of the collection deleted', y: 'recall among what is left' });

  layer.append('path')
    .attr('d', d3.line().x((r) => x(r.deleted)).y((r) => y(r.tombstone_recall))(rows))
    .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
  rows.forEach((r) => layer.append('circle').attr('cx', x(r.deleted))
    .attr('cy', y(r.tombstone_recall)).attr('r', 4).attr('fill', 'var(--primary)'));

  const rebuilt = rows.filter((r) => r.rebuilt_recall != null);
  rebuilt.forEach((r) => {
    layer.append('path').attr('d', d3.symbol().type(d3.symbolDiamond).size(110)())
      .attr('transform', `translate(${x(r.deleted)},${y(r.rebuilt_recall)})`)
      .attr('fill', 'var(--anchor)');
  });

  [['marked as deleted and left in place', 'var(--primary)'],
    ['the index rebuilt from what is left', 'var(--anchor)']].forEach(([t, colour], i) => {
    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -40 + i * 15)
      .style('font-size', '11px').attr('fill', colour).text(t);
  });

  const worst = rows[rows.length - 1];
  const first = rows[0];
  const pair = rebuilt.length ? rebuilt[rebuilt.length - 1] : null;
  const tbl = rows.map((r) => `<tr><td>${(r.deleted * 100).toFixed(0)}%</td>`
    + `<td>${r.alive.toLocaleString('en-US')}</td>`
    + `<td>${r.tombstone_recall.toFixed(4)}</td>`
    + `<td>${r.rebuilt_recall == null ? '' : r.rebuilt_recall.toFixed(4)}</td>`
    + `<td>${Math.round(r.cost)}</td></tr>`).join('');
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>deleted</th><th>left</th>`
    + `<th>marked in place</th><th>rebuilt</th><th>distances per query</th></tr></thead>`
    + `<tbody>${tbl}</tbody></table>`
    + `<div class="gen-note">Deleting nothing gives ${first.tombstone_recall.toFixed(4)}; `
    + `deleting ${(worst.deleted * 100).toFixed(0)}% and marking them gives `
    + `${worst.tombstone_recall.toFixed(4)}. `
    + (pair
      ? `At ${(pair.deleted * 100).toFixed(0)}% deleted the rebuilt index reaches `
        + `${pair.rebuilt_recall.toFixed(4)} against ${pair.tombstone_recall.toFixed(4)} for `
        + `the marked one, so the tombstones cost `
        + `${(pair.rebuilt_recall - pair.tombstone_recall).toFixed(4)} of recall. `
      : '')
    + `The reason a rebuild is not the obvious answer is the last column of the previous `
    + `article: this index cost ${first.build_cost.toLocaleString('en-US')} distances to `
    + `build, which is ${(first.build_cost / (first.alive)).toFixed(0)} queries' worth of `
    + `exhaustive search, and it has to be paid again every time. That arithmetic, and not `
    + `the recall, is what makes a changing collection a different problem from a fixed `
    + `one.</div>`;

  return { render: () => {} };
}
