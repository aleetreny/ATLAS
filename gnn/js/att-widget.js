/* What the attention weights came out as, against what they replaced.
 *
 * A GCN gives every neighbour the same weight, one over the degree. Attention
 * is supposed to learn better ones. The labels of every node are known here,
 * so "better" has a definition: an edge joining two papers of the same topic
 * should carry more weight than one joining two different topics. Both
 * quantities are on the chart, and the honest reading is the gap between the
 * two pairs of bars rather than either pair on its own.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initAttWidget(data) {
  const A = data.attention;
  const readout = document.querySelector('#att-readout');

  const chart = makeChart('#att-chart', {
    width: 640, height: 340, margin: { top: 44, right: 26, bottom: 66, left: 78 }, clip: true,
  });
  const { g, plot, w, h } = chart;

  const groups = [
    { label: 'same topic', learned: A.same, uniform: A.uniform_same },
    { label: 'different topics', learned: A.diff, uniform: A.uniform_diff },
  ];
  const x0 = d3.scaleBand().domain(groups.map((d) => d.label)).range([0, w]).padding(0.35);
  const x1 = d3.scaleBand().domain(['learned', 'uniform']).range([0, x0.bandwidth()]).padding(0.12);
  const y = d3.scaleLinear()
    .domain([0, d3.max(groups, (d) => Math.max(d.learned, d.uniform)) * 1.25]).range([h, 0]);
  drawGrid(g, x0, y, w, h, { xValues: [], yTicks: 5 });

  groups.forEach((grp) => {
    [['learned', 'var(--primary)'], ['uniform', 'var(--stone)']].forEach(([key, colour]) => {
      plot.append('rect')
        .attr('x', x0(grp.label) + x1(key)).attr('width', x1.bandwidth())
        .attr('y', y(grp[key])).attr('height', h - y(grp[key]))
        .attr('fill', colour);
      g.append('text').attr('class', 'chart-note')
        .attr('x', x0(grp.label) + x1(key) + x1.bandwidth() / 2).attr('y', y(grp[key]) - 8)
        .attr('text-anchor', 'middle').style('font-size', '11px')
        .text(grp[key].toFixed(4));
    });
    g.append('text').attr('class', 'chart-note')
      .attr('x', x0(grp.label) + x0.bandwidth() / 2).attr('y', h + 20)
      .attr('text-anchor', 'middle').style('font-size', '11.5px').text(grp.label);
  });

  /* The two bars of a group used to name themselves underneath, and even
     shortened to "attention" and "1 / degree" they sat a space apart and read
     as one run on phrase. Half a band is not enough room for two names; a
     legend has the whole width. */
  [['attention', 'var(--primary)'], ['1 / degree', 'var(--stone)']].forEach(([txt, colour], i) => {
    g.append('rect').attr('x', 6 + i * 104).attr('y', -12).attr('width', 10).attr('height', 10)
      .attr('fill', colour);
    g.append('text').attr('class', 'chart-note')
      .attr('x', 20 + i * 104).attr('y', -3).style('font-size', '11px').text(txt);
  });

  drawAxes(g, x0, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.2f') });
  axisLabels(g, w, h, { y: 'weight given to an edge' });
  g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px')
    .text(`${A.edges.toLocaleString('en-US')} directed edges, of which `
      + `${A.cross_edges.toLocaleString('en-US')} join two different topics`);

  const gap = A.ratio - A.uniform_ratio;
  readout.innerHTML =
    `The learned weights prefer same topic edges by a factor of `
    + `<span class="value">${A.ratio.toFixed(3)}</span>. Uniform weights, which is what a GCN uses `
    + `and which learn nothing, prefer them by `
    + `<span class="value">${A.uniform_ratio.toFixed(3)}</span>, because a paper with many `
    + `citations of the other topic has a larger degree and every one of its edges is worth less. `
    + `${Math.abs(gap) < 0.01
      ? `The difference between those two numbers is <span class="value">${gap.toFixed(4)}</span>. `
        + `On this graph the attention has not learned to tell the two kinds of edge apart: it has `
        + `learned the degree, which was already there for free.`
      : `The difference is <span class="value">${gap.toFixed(3)}</span> in favour of the learned `
        + `weights, so the attention has found something the normalisation did not.`} `
    + `Across every edge, the correlation between the attention weight and whether the two ends `
    + `share a topic is <span class="value">${A.corr.toFixed(4)}</span>. The self loop, which is the `
    + `node's own value, gets <span class="value">${A.self.toFixed(4)}</span>, and that is the one `
    + `place where the attention does something a fixed rule would not.`;

  return { groups };
}
