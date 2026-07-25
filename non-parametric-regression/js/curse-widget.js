/* The curse of dimensionality, in the only two numbers that matter for a
 * method built on the word "nearby":
 *
 *  - how big a neighbourhood must be to hold a fixed share of the data
 *  - how much closer the nearest neighbour is than a typical point
 *
 * Both are computed in the generator; this draws them. */
import { makeChart, axisLabels } from '../../assets/js/chart.js';

export function initCurseWidget(curse) {
  const dims = curse.edge.map((e) => e.d);

  const { g, w, h } = makeChart('#curse-chart', {
    width: 660,
    height: 340,
    margin: { top: 34, right: 130, bottom: 58, left: 60 },
  });
  const x = d3.scalePoint().domain(dims).range([0, w]).padding(0.5);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

  // The x scale here is a point scale, one slot per dimension count, and point
  // scales have no .ticks(), so the shared drawGrid helper cannot be used.
  const gridG = g.append('g').attr('class', 'grid');
  gridG.append('g').selectAll('line').data(dims).join('line')
    .attr('x1', (d) => x(d)).attr('x2', (d) => x(d)).attr('y1', 0).attr('y2', h);
  gridG.append('g').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', w).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));

  g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).tickSizeOuter(0));
  g.append('g').attr('class', 'axis y').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')).tickSizeOuter(0));
  axisLabels(g, w, h, { x: 'number of features', y: '' });

  const SERIES = [
    { key: 'edge10', label: 'neighbourhood needed to hold 10% of the data', color: 'var(--primary)',
      vals: curse.edge.map((e) => e.r10) },
    { key: 'ratio', label: 'nearest neighbour distance ÷ typical distance', color: 'var(--cosmos)',
      vals: curse.nn_distance.map((e) => e.ratio) },
  ];

  const line = d3.line().x((d, i) => x(dims[i])).y((d) => y(d)).curve(d3.curveMonotoneX);
  for (const s of SERIES) {
    g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5).attr('d', line(s.vals));
    g.append('path').attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 3.4).attr('d', line(s.vals));
    g.append('g').selectAll('circle').data(s.vals).join('circle')
      .attr('cx', (d, i) => x(dims[i])).attr('cy', (d) => y(d)).attr('r', 4)
      .attr('fill', s.color).attr('stroke', 'white').attr('stroke-width', 1.4);
    g.append('text').attr('class', 'chart-annotation')
      .attr('x', w + 8).attr('y', y(s.vals[s.vals.length - 1]) + 4)
      .style('font-size', '0.6rem').style('letter-spacing', '0.3px').style('text-transform', 'none')
      .attr('fill', s.color)
      .call((t) => {
        const words = s.label.split(' ');
        const lines = [];
        let cur = '';
        for (const word of words) {
          if ((cur + ' ' + word).trim().length > 18) { lines.push(cur.trim()); cur = word; }
          else cur += ' ' + word;
        }
        lines.push(cur.trim());
        t.selectAll('tspan').data(lines).join('tspan')
          .attr('x', w + 8).attr('dy', (d, i) => (i === 0 ? 0 : 11)).text((d) => d);
      });
  }

  const el = document.querySelector('#curse-stats');
  const e10 = curse.edge.find((e) => e.d === 10);
  const r10 = curse.nn_distance.find((e) => e.d === 10);
  const r100 = curse.nn_distance.find((e) => e.d === 100);
  el.innerHTML =
    `<div class="slider-label">In one dimension, the closest 10% of the data sits within ` +
    `<span class="value">${(curse.edge[0].r10 * 100).toFixed(0)}%</span> of the range. ` +
    `With ten features that neighbourhood has to span ` +
    `<span class="value">${(e10.r10 * 100).toFixed(0)}%</span> of every axis to hold the same 10%.</div>` +
    `<div class="slider-label" style="margin-top:.4rem">And "nearest" stops meaning much: with 500 points, ` +
    `the closest one is <span class="value">${(curse.nn_distance[0].ratio * 100).toFixed(1)}%</span> ` +
    `of a typical distance away in one dimension, ` +
    `<span class="value">${(r10.ratio * 100).toFixed(0)}%</span> with ten features, and ` +
    `<span class="value">${(r100.ratio * 100).toFixed(0)}%</span> with a hundred.</div>`;
}
