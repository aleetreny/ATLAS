/* Whether "nearest" is a meaningful word here, before asking how to find it.
 *
 * The quantity is the contrast: how much further the farthest point is than the
 * nearest one, as a fraction of the nearest. In enough dimensions of
 * independent noise it goes to zero, and once it does, an approximate answer
 * and an exact one are the same answer, which is either very good news or a
 * sign that the vectors are not carrying what you think.
 *
 * The two marks off the curve are the real data, and they sit far above the
 * gaussian line at their own dimension for a reason worth measuring rather than
 * asserting: their intrinsic dimension is much lower than their column count.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initConcWidget(data) {
  const C = data.concentration;
  const readout = document.querySelector('#conc-readout');

  const chart = makeChart('#conc-chart', {
    width: 640, height: 360, margin: { top: 48, right: 40, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const rows = C.gaussian;
  const x = d3.scaleLog().domain([rows[0].d * 0.8, rows[rows.length - 1].d * 1.3]).range([0, w]);
  const all = rows.map((r) => r.contrast).concat([C.reuters.contrast, C.words.contrast]);
  const y = d3.scaleLog().domain([d3.min(all) * 0.7, d3.max(all) * 1.5]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.d), yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xValues: rows.map((r) => r.d), yTicks: 5, xFmt: (v) => String(v), yFmt: d3.format('.2f'),
  });
  axisLabels(g, w, h, { x: 'dimensions', y: 'how much further the farthest point is' });

  layer.append('path').attr('d', d3.line().x((r) => x(r.d)).y((r) => y(r.contrast))(rows))
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.2);
  rows.forEach((r) => layer.append('circle').attr('cx', x(r.d)).attr('cy', y(r.contrast))
    .attr('r', 3.8).attr('fill', 'var(--squidink)'));
  layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '11px').text('independent gaussian noise');

  [[C.reuters, 'the newswire vectors', 'var(--primary)'],
    [C.words, 'the word vectors', 'var(--cosmos)']].forEach(([pt, name, colour], i) => {
    layer.append('path').attr('d', d3.symbol().type(d3.symbolDiamond).size(90)())
      .attr('transform', `translate(${x(pt.d)},${y(pt.contrast)})`).attr('fill', colour);
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(pt.d) - 10).attr('y', y(pt.contrast) + (i ? 18 : -10))
      .attr('text-anchor', 'end').style('font-size', '11px').attr('fill', colour)
      .text(`${name}, ${pt.d} columns`);
  });

  const at64 = rows.reduce((a, b) => (Math.abs(b.d - C.reuters.d) < Math.abs(a.d - C.reuters.d) ? b : a));
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>data</th><th>columns</th>`
    + `<th>intrinsic dimension</th><th>contrast</th><th>farthest over nearest</th>`
    + `</tr></thead><tbody>`
    + `<tr><td>gaussian noise</td><td>${at64.d}</td><td>${at64.d}</td>`
    + `<td>${at64.contrast.toFixed(3)}</td><td>${at64.ratio.toFixed(2)}</td></tr>`
    + `<tr><td>newswires</td><td>${C.reuters.d}</td>`
    + `<td>${C.reuters.intrinsic.toFixed(1)}</td><td>${C.reuters.contrast.toFixed(3)}</td>`
    + `<td>${C.reuters.ratio.toFixed(2)}</td></tr>`
    + `<tr><td>word vectors</td><td>${C.words.d}</td>`
    + `<td>${C.words.intrinsic.toFixed(1)}</td><td>${C.words.contrast.toFixed(3)}</td>`
    + `<td>${C.words.ratio.toFixed(2)}</td></tr>`
    + `</tbody></table>`
    + `<div class="gen-note">At ${at64.d} dimensions of pure noise the farthest of two `
    + `thousand points is only ${at64.ratio.toFixed(2)} times as far as the nearest, and the `
    + `gap keeps closing. Real vectors do not behave like that at the same column count: the `
    + `newswires have ${C.reuters.d} columns and a contrast of `
    + `${C.reuters.contrast.toFixed(3)} against ${at64.contrast.toFixed(3)} for noise, `
    + `because their intrinsic dimension, estimated from the ratio between each point's `
    + `first and second neighbour, is ${C.reuters.intrinsic.toFixed(1)} rather than `
    + `${C.reuters.d}. That is the whole reason approximate search works on real data and `
    + `not on noise: the vectors live on something much thinner than the box they are `
    + `written in. The word vectors are the ones the `
    + `<a href="../embeddings/">embeddings article</a> published, read from its own `
    + `file.</div>`;

  return { render: () => {} };
}
