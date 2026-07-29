/* The polytope, and a dial for the direction of the objective.
 *
 * The corners are enumerated here, in the browser, from the list of
 * constraints, and the winner is the corner with the largest dot product. Turn
 * the dial and the answer does not slide: it sits still and then jumps, which
 * is the whole first section drawn rather than asserted.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { enumerateVertices, ring } from './lpkit.js';

export function initDialWidget(data) {
  const P = data.polytope;
  const A = P.rows.map((r) => r.a).concat([[-1, 0], [0, -1]]);
  const b = P.rows.map((r) => r.b).concat([0, 0]);
  const names = P.rows.map((r) => r.name).concat(['x is not negative', 'y is not negative']);
  const pr = P.profit;                 // one source for the profits, not a literal
  const { vertices } = enumerateVertices(A, b);
  const R = ring(vertices);
  const slider = document.querySelector('#dial-angle');
  const label = document.querySelector('#dial-angle-label');
  const readout = document.querySelector('#dial-readout');

  const chart = makeChart('#dial-chart', {
    width: 640, height: 400, margin: { top: 44, right: 132, bottom: 52, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().domain([-0.4, 4.6]).range([0, w]);
  const y = d3.scaleLinear().domain([-0.3, 3.0]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'lathe product, x', y: 'press product, y', leftBudget: margin.left });

  g.append('path').attr('class', 'region')
    .attr('d', `M${R.map((p) => `${x(p.x)},${y(p.y)}`).join('L')}Z`)
    .attr('fill', 'var(--primary)').attr('fill-opacity', 0.1)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4);

  const contour = g.append('line').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
  const arrow = g.append('line').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.2);
  const arrowHead = g.append('path').attr('fill', 'var(--anchor)');

  g.selectAll('circle.v').data(R).join('circle').attr('class', 'v')
    .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 5)
    .attr('fill', '#ffffff').attr('stroke', 'var(--primary)').attr('stroke-width', 1.6);
  const win = g.append('circle').attr('r', 8).attr('fill', 'var(--primary)');
  const winLabel = g.append('text').attr('class', 'chart-note').style('font-size', '11.5px');

  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -26)
    .style('font-size', '12px')
    .text(`every plan that respects the four limits, and its ${R.length} corners`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -10)
    .style('font-size', '11.5px');

  /* the legend lives outside the plot, where no data can be */
  /* the glyph is the mark itself, not a square standing in for it */
  const key = g.append('g').attr('transform', `translate(${w + 12}, 6)`);
  key.append('circle').attr('cx', 6).attr('cy', 6).attr('r', 5.5).attr('fill', 'var(--primary)');
  key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 24)
    .style('font-size', '10.5px').text('the winning');
  key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 35)
    .style('font-size', '10.5px').text('corner');
  key.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 54).attr('y2', 54)
    .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
  key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 70)
    .style('font-size', '10.5px').text('equal profit');
  key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 81)
    .style('font-size', '10.5px').text('line');

  const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));

  function render() {
    const deg = Number(slider.value);
    const th = (deg * Math.PI) / 180;
    const dir = [Math.cos(th), Math.sin(th)];
    const scores = R.map((p) => p.x * dir[0] + p.y * dir[1]);
    const bestScore = Math.max(...scores);
    const winners = scores.map((s, i) => [s, i]).filter(([s]) => s > bestScore - 1e-9);
    const j = winners[0][1];
    const p = R[j];

    const mid = [(x.domain()[0] + x.domain()[1]) / 2, (y.domain()[0] + y.domain()[1]) / 2];
    const perp = [-dir[1], dir[0]];
    /* just long enough to cross the region: at 6 it ran to the corners of the
       canvas and read as a second axis */
    const L = 2.6;
    contour
      .attr('x1', x(p.x - perp[0] * L)).attr('y1', y(p.y - perp[1] * L))
      .attr('x2', x(p.x + perp[0] * L)).attr('y2', y(p.y + perp[1] * L));
    const ax = mid[0] - dir[0] * 0.9;
    const ay = mid[1] - dir[1] * 0.9;
    const bx = mid[0] + dir[0] * 0.9;
    const by = mid[1] + dir[1] * 0.9;
    arrow.attr('x1', x(ax)).attr('y1', y(ay)).attr('x2', x(bx)).attr('y2', y(by));
    const hx = x(bx);
    const hy = y(by);
    const ux = x(bx) - x(ax);
    const uy = y(by) - y(ay);
    const un = Math.hypot(ux, uy) || 1;
    const px = -uy / un;
    const py = ux / un;
    arrowHead.attr('d', `M${hx},${hy}L${hx - (ux / un) * 12 + px * 6},${hy - (uy / un) * 12 + py * 6}`
      + `L${hx - (ux / un) * 12 - px * 6},${hy - (uy / un) * 12 - py * 6}Z`);

    win.attr('cx', x(p.x)).attr('cy', y(p.y));
    winLabel.attr('x', x(p.x) + (p.x > 3 ? -10 : 12)).attr('y', y(p.y) - 10)
      .attr('text-anchor', p.x > 3 ? 'end' : 'start')
      .text(`(${fmt(p.x)}, ${fmt(p.y)})`);
    title.text(`direction ${deg}°: the best plan is the corner at `
      + `(${fmt(p.x)}, ${fmt(p.y)})`);
    label.textContent = `${deg}°`;

    const active = p.active.map((i) => names[i]);
    const profitHere = pr[0] * p.x + pr[1] * p.y;
    const tie = winners.length > 1;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>corner</th>${R.map((q) => `<th>(${fmt(q.x)}, ${fmt(q.y)})</th>`).join('')}</tr>
        <tr><td>score in this direction</td>
            ${scores.map((s, i) => `<td>${i === j ? '<span class="value">' : ''}${s.toFixed(3)}${i === j ? '</span>' : ''}</td>`).join('')}</tr>
        <tr><td>profit at ${pr[0]} and ${pr[1]}</td>
            ${R.map((q) => `<td>${(pr[0] * q.x + pr[1] * q.y).toFixed(1)}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        ${tie
    ? `Two corners tie here, by ${Math.abs(scores[winners[0][1]] - scores[winners[1][1]]).toExponential(1)}.
           This is the only way a linear problem has more than one answer: the
           objective is parallel to an edge, and then every point of that edge is
           just as good. Nudge the dial one degree either way and the tie breaks.`
    : `The winner is the corner where <span class="value">${active[0]}</span> and
           <span class="value">${active[1]}</span> are both used up to the last unit,
           which is what a corner is. At the profits this workshop actually has,
           ${pr[0]} and ${pr[1]}, that plan is worth
           <span class="value">${profitHere.toFixed(1)}</span>.`}
        Turning the dial through all ${data.dial.steps} directions the generator
        swept lands on <span class="value">${data.dial.distinct_owners}</span>
        different corners and on nothing else.
      </div>`;
  }

  slider.addEventListener('input', render);
  document.querySelector('#dial-profit').addEventListener('click', () => {
    slider.value = String(Math.round((Math.atan2(4, 5) * 180) / Math.PI));
    render();
  });
  render();
  return { render, vertices: R };
}
