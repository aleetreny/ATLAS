/* The dial that says when this whole family of models is the wrong idea.
 *
 * On a real graph you cannot ask this question, because the homophily is
 * whatever it is. These graphs are written by the generator with the classes
 * decided first, so the share of edges that join two nodes of the same class
 * is a knob, and the crossing point where averaging over neighbours stops
 * paying is a measurement rather than a warning.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

const SERIES = [
  ['mlp', 'no edges at all', 'var(--anchor)'],
  ['gcn', 'GCN', 'var(--primary)'],
  ['sage', 'GraphSAGE', 'var(--cosmos)'],
];

export function initHomoWidget(data) {
  const rows = data.homophily;
  const readout = document.querySelector('#homo-readout');

  const chart = makeChart('#homo-chart', {
    width: 640, height: 380, margin: { top: 44, right: 118, bottom: 58, left: 72 }, clip: true,
  });
  const { g, plot, w, h } = chart;

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([0.4, 1.02]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });

  SERIES.forEach(([key, label, colour]) => {
    plot.append('path').attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
      .attr('d', d3.line().x((d) => x(d.measured)).y((d) => y(d[key]))(rows));
    plot.selectAll(null).data(rows).enter().append('circle')
      .attr('cx', (d) => x(d.measured)).attr('cy', (d) => y(d[key])).attr('r', 3)
      .attr('fill', colour);
  });
  const ends = spread(SERIES.map(([key, label, colour]) => ({
    label, colour, y: y(rows[rows.length - 1][key]) + 3.5, yMax: h + 10,
  })), 18);
  ends.forEach((d) => {
    g.append('text').attr('class', 'chart-note').attr('x', w + 6).attr('y', d.y)
      .style('font-size', '11px').style('fill', d.colour).text(d.label);
  });

  /* where the graph model overtakes the one that ignores the graph */
  let cross = null;
  for (let i = 1; i < rows.length; i++) {
    const a = rows[i - 1].gcn - rows[i - 1].mlp;
    const b = rows[i].gcn - rows[i].mlp;
    if (a < 0 && b >= 0) {
      const t = -a / (b - a);
      cross = rows[i - 1].measured + t * (rows[i].measured - rows[i - 1].measured);
      break;
    }
  }
  if (cross !== null) {
    plot.append('line').attr('x1', x(cross)).attr('x2', x(cross)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3');
    g.append('text').attr('class', 'chart-note')
      .attr('x', x(cross) + 5).attr('y', 12).style('font-size', '11px')
      .text(`they cross at ${(100 * cross).toFixed(0)}%`);
  }
  if (data.meta.homophily <= 1) {
    plot.append('line').attr('x1', x(data.meta.homophily)).attr('x2', x(data.meta.homophily))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--smile)').attr('stroke-width', 1.6);
    g.append('text').attr('class', 'chart-note')
      .attr('x', x(data.meta.homophily) - 5).attr('y', h - 8).attr('text-anchor', 'end')
      .style('font-size', '11px').style('fill', 'var(--smile)')
      .text(`Cora sits here, ${(100 * data.meta.homophily).toFixed(0)}%`);
  }

  drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, xFmt: d3.format('.0%'), yFmt: d3.format('.0%') });
  axisLabels(g, w, h, { x: 'edges joining two nodes of the same class', y: 'test nodes correct' });
  g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px')
    .text('written graphs: 1,000 nodes, average degree 10, classes decided first');

  const first = rows[0];
  const last = rows[rows.length - 1];
  const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
  readout.innerHTML =
    `At the left, where only <span class="value">${pct(first.measured)}</span> of edges join two `
    + `nodes of the same class, the model that ignores the graph scores `
    + `<span class="value">${pct(first.mlp)}</span> and the GCN `
    + `<span class="value">${pct(first.gcn)}</span>: averaging over neighbours actively destroys `
    + `the features, because the neighbours are mostly of the other class. At the right, with `
    + `<span class="value">${pct(last.measured)}</span>, the same GCN gets `
    + `<span class="value">${pct(last.gcn)}</span> against `
    + `<span class="value">${pct(last.mlp)}</span>. `
    + `${cross !== null
      ? `The two lines cross at about <span class="value">${pct(cross, 0)}</span>, so on a graph `
        + `below that, using it costs accuracy rather than buying it.`
      : 'The two lines do not cross inside the range measured, so on these graphs the edges never '
        + 'hurt.'} `
    + `GraphSAGE keeps a separate weight for the node itself, and that shows up exactly where you `
    + `would expect: at the low end it scores <span class="value">${pct(first.sage)}</span>, `
    + `${first.sage > first.gcn ? 'above' : 'below'} the GCN, because it can learn to ignore what `
    + `the neighbours say.`;

  return { cross };
}
