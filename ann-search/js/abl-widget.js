/* The two pieces of the graph index, removed one at a time.
 *
 * The hierarchy of layers is the part the name of the algorithm is about, and
 * it is the part with the least evidence behind it: later work found it buys
 * very little at moderate dimension. The edge pruning heuristic, which keeps a
 * neighbour only if it is closer to the node than to every neighbour already
 * kept, is the piece that almost never gets mentioned. Removing each of them
 * separately says which is carrying the algorithm.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const ARMS = [
  ['full', 'layers and pruning', 'var(--primary)'],
  ['one layer', 'no layers', 'var(--anchor)'],
  ['nearest m edges', 'no pruning', 'var(--cosmos)'],
  ['neither', 'neither', 'var(--squidink)'],
];

export function initAblWidget(data) {
  const A = data.ablation;
  const readout = document.querySelector('#abl-readout');

  const chart = makeChart('#abl-chart', {
    width: 640, height: 370, margin: { top: 62, right: 40, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const costs = A.rows.map((r) => r.cost);
  const x = d3.scaleLog().domain([Math.min(...costs) * 0.75, Math.max(...costs) * 1.4])
    .range([0, w]);
  const recalls = A.rows.map((r) => r.recall);
  const y = d3.scaleLinear().domain([Math.min(...recalls) - 0.03, 1.01]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xTicks: 5, yTicks: 5, yFmt: d3.format('.2f'),
    xFmt: (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v))),
  });
  axisLabels(g, w, h, { x: 'distances computed per query', y: 'recall at ten' });

  ARMS.forEach(([arm, name, colour], i) => {
    const rows = A.rows.filter((r) => r.arm === arm).sort((a, b) => a.cost - b.cost);
    layer.append('path').attr('d', d3.line().x((r) => x(r.cost)).y((r) => y(r.recall))(rows))
      .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
    rows.forEach((r) => layer.append('circle').attr('cx', x(r.cost)).attr('cy', y(r.recall))
      .attr('r', 4).attr('fill', colour));
    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -48 + i * 14)
      .style('font-size', '11px').attr('fill', colour).text(name);
  });

  const at = (arm, ef) => A.rows.find((r) => r.arm === arm && r.ef === ef);
  const efs = Array.from(new Set(A.rows.map((r) => r.ef))).sort((a, b) => a - b);
  const mid = efs[Math.floor(efs.length / 2)];
  const full = at('full', mid);
  const noL = at('one layer', mid);
  const noP = at('nearest m edges', mid);
  const layerCost = full.recall - noL.recall;
  const pruneCost = full.recall - noP.recall;
  const rows = ARMS.map(([arm, name]) => {
    const r = at(arm, mid);
    return `<tr><td>${name}</td><td>${r.recall.toFixed(4)}</td>`
      + `<td>${Math.round(r.cost)}</td><td>${r.layers}</td>`
      + `<td>${r.build.toLocaleString('en-US')}</td></tr>`;
  }).join('');
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>index</th>`
    + `<th>recall at the middle setting</th><th>distances per query</th><th>layers</th>`
    + `<th>distances to build</th></tr></thead><tbody>${rows}</tbody></table>`
    + `<div class="gen-note">On ${A.n.toLocaleString('en-US')} vectors, at the middle search `
    + `width. Removing the layers costs <span class="bold">${layerCost.toFixed(4)}</span> of `
    + `recall and removing the edge pruning costs `
    + `<span class="bold">${pruneCost.toFixed(4)}</span>, so on this data at this scale `
    + (Math.abs(pruneCost) > Math.abs(layerCost)
      ? `the piece nobody talks about is doing more than the piece the algorithm is named `
        + `after. That agrees with what later work found: the hierarchy earns its keep when `
        + `the intrinsic dimension is high, and these vectors do not have one.`
      : `the hierarchy is doing more than the pruning, which is the ordering the original `
        + `paper claims.`)
    + ` The build cost is the column that decides whether any of this is worth it for a `
    + `collection that changes: the full index pays `
    + `${full.build.toLocaleString('en-US')} distances before answering a single query, which `
    + `is ${(full.build / (A.n * A.n)).toFixed(2)} times what comparing everything to `
    + `everything would cost.</div>`;

  return { render: () => {} };
}
