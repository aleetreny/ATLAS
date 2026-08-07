/* The two coefficients, swept, against the boundary the theory draws.
 *
 * Each cell is ten runs. The curve is 24(1 - w^2) / (7 - 5w), recomputed here
 * rather than read from the file, and the interesting part of the picture is the
 * cells on the wrong side of it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { stabilityLimit } from './sakit.js';

export function initStabilityWidget(data) {
  const S = data.stability;
  const readout = document.querySelector('#stability-readout');
  const ws = [...new Set(S.rows.map((r) => r.w))];
  const cs = [...new Set(S.rows.map((r) => r.sum_c))];

  const chart = makeChart('#stability-chart', {
    width: 640, height: 380, margin: { top: 50, right: 128, bottom: 56, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scalePoint().domain(ws.map(String)).range([0, w]).padding(0.6);
  const y = d3.scalePoint().domain(cs.map(String)).range([h, 0]).padding(0.6);
  drawAxes(g, x, y, w, h, {});
  axisLabels(g, w, h, { x: 'inertia', y: 'the two pulls, added',
    leftBudget: margin.left });

  const cw = (w / ws.length) * 0.78;
  const ch = (h / cs.length) * 0.72;
  const colour = d3.scaleLog().domain([1e-6, 40]).range(['#ffffff', '#8e9aa6']).clamp(true);
  g.selectAll('rect').data(S.rows).join('rect')
    .attr('x', (d) => x(String(d.w)) - cw / 2).attr('y', (d) => y(String(d.sum_c)) - ch / 2)
    .attr('width', cw).attr('height', ch)
    .attr('fill', (d) => (d.diverged > 0 ? 'var(--cosmos)' : colour(Math.max(d.median, 1e-6))))
    .attr('fill-opacity', (d) => (d.diverged > 0 ? 0.75 : 1))
    .attr('stroke', '#ffffff').attr('stroke-width', 1);
  g.selectAll('text.v').data(S.rows).join('text').attr('class', 'chart-note v')
    .attr('x', (d) => x(String(d.w))).attr('y', (d) => y(String(d.sum_c)) + 4)
    .attr('text-anchor', 'middle').style('font-size', '10px')
    .attr('fill', (d) => (d.diverged > 0 ? '#ffffff' : 'var(--squidink)'))
    .attr('stroke', (d) => (d.diverged > 0 ? 'var(--cosmos)' : '#ffffff'))
    .text((d) => (d.diverged > 0 ? `${d.diverged}/${d.of}` : d.median.toExponential(0)));

  /* the predicted boundary, drawn as a curve through the same axes */
  const curve = [];
  for (let k = 0; k <= 60; k++) {
    const wv = ws[0] + ((ws[ws.length - 1] - ws[0]) * k) / 60;
    const lim = stabilityLimit(wv);
    if (lim < cs[0] || lim > cs[cs.length - 1]) continue;
    /* the axes are point scales, so a continuous value is placed by
       interpolating between the two neighbouring ticks it falls between */
    const place = (scale, vals, v) => {
      for (let i = 0; i < vals.length - 1; i++) {
        if (v >= vals[i] && v <= vals[i + 1]) {
          const t = (v - vals[i]) / (vals[i + 1] - vals[i]);
          return scale(String(vals[i])) + t * (scale(String(vals[i + 1])) - scale(String(vals[i])));
        }
      }
      return null;
    };
    const px = place(x, ws, wv);
    const py = place(y, cs, lim);
    if (px !== null && py !== null) curve.push([px, py]);
  }
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 2.4).attr('d', d3.line()(curve));

  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px').text('ten runs per cell, on the simplest function there is');
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px')
    .text(`the curve is where the theory says the particles stop settling`);

  const key = g.append('g').attr('transform', `translate(${w + 10}, 10)`);
  key.append('rect').attr('width', 12).attr('height', 12).attr('fill', 'var(--cosmos)')
    .attr('fill-opacity', 0.75);
  ['runs that', 'overflowed'].forEach((line, i) => {
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 28 + i * 16)
      .style('font-size', '10.5px').text(line);
  });
  key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 58).attr('y2', 58)
    .attr('stroke', 'var(--anchor)').attr('stroke-width', 2.4);
  ['the predicted', 'boundary'].forEach((line, i) => {
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 74 + i * 16)
      .style('font-size', '10.5px').text(line);
  });

  const wrong = S.rows.filter((d) => (d.diverged === 0) !== d.predicted_stable);
  const worstWrong = wrong.reduce((a, b) => (b.spread_ratio > a.spread_ratio ? b : a), wrong[0]);
  readout.innerHTML = `
    <table class="gen-table">
      <tr><th></th><th>predicted to settle</th><th>predicted not to</th></tr>
      <tr><td>cells</td>
          <td>${S.rows.filter((d) => d.predicted_stable).length}</td>
          <td>${S.rows.filter((d) => !d.predicted_stable).length}</td></tr>
      <tr><td>best value found, median</td>
          <td><span class="value">${S.stable_median}</span></td>
          <td>${S.unstable_median}</td></tr>
      <tr><td>how much the cloud grew</td>
          <td>${S.stable_spread.toExponential(1)}</td>
          <td>${S.unstable_spread.toExponential(1)}</td></tr>
    </table>
    <div class="gen-note">
      Every cell the theory calls stable settles, and none of them overflowed:
      the cloud ends <span class="value">${S.stable_spread.toExponential(1)}</span> times
      the size it started and the answers are ${S.stable_median}. On the other side the
      cloud ends ${S.unstable_spread.toExponential(1)} times its starting size
      and the answers are ${S.unstable_median}. But
      <span class="bold">${wrong.length} of the ${S.of} cells sit on the wrong
      side of the curve</span>, all of them just past it and all in the same
      direction: predicted to blow up, and they did not. The derivation holds the
      two attractors still, and in a real swarm the best position moves with the
      particle that found it, which pulls the whole thing back. The worst of the
      ${wrong.length} is inertia ${worstWrong.w} with pulls adding to
      ${worstWrong.sum_c} against a limit of ${worstWrong.limit}: no overflow,
      but the cloud still grew ${worstWrong.spread_ratio.toFixed(0)} times and its
      answer, ${worstWrong.median}, is far worse than anything inside the curve.
      The boundary is right about what it is about; it is about a swarm with the
      targets nailed down.
    </div>`;

  return { render: () => {} };
}
