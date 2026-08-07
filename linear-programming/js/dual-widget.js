/* What one more hour is worth, and for how long that stays true.
 *
 * The curve is not read out of the file: every point on it is a fresh solve of
 * the whole problem by the simplex in lpkit.js. The straight line is the
 * shadow price extended, and where the two come apart is the answer to the
 * question the price cannot answer on its own.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { simplex } from './lpkit.js';

export function initDualWidget(data) {
  const P = data.polytope;
  const A = P.rows.map((r) => r.a);
  const b0 = P.rows.map((r) => r.b);
  const c = P.profit;
  const names = P.rows.map((r) => r.name);
  const readout = document.querySelector('#dual-readout');
  const slider = document.querySelector('#dual-delta');
  const label = document.querySelector('#dual-delta-label');
  const controls = d3.select('#dual-controls');
  let which = 0;

  const solveWith = (i, delta) => {
    const b = b0.slice();
    b[i] = b0[i] + delta;
    if (b[i] < 0) return null;
    return simplex(c, A, b).value;
  };

  const shadow = (i) => {
    const hStep = 1e-6;
    const up = solveWith(i, hStep);
    const dn = solveWith(i, -hStep);
    return (up - dn) / (2 * hStep);
  };

  const chart = makeChart('#dual-chart', {
    width: 640, height: 360, margin: { top: 46, right: 118, bottom: 54, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().range([0, w]);
  const y = d3.scaleLinear().range([h, 0]);
  const curve = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2.4);
  const tangent = g.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
  const breakLine = g.append('line').attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.6);
  const breakText = g.append('text').attr('class', 'chart-note').style('font-size', '11px')
    .attr('fill', 'var(--cosmos)');
  const marker = g.append('circle').attr('r', 6).attr('fill', 'var(--primary)');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -28)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -11)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 10}, 8)`);
  [['var(--primary)', 'solved again', 'at every point'],
    ['var(--anchor)', 'the price', 'extended']].forEach(([col, a, bb], i) => {
    key.append('line').attr('x1', 0).attr('x2', 16).attr('y1', i * 48 + 6).attr('y2', i * 48 + 6)
      .attr('stroke', col).attr('stroke-width', 2)
      .attr('stroke-dasharray', i === 1 ? '5 4' : null);
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 48 + 20)
      .style('font-size', '10.5px').text(a);
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 48 + 38)
      .style('font-size', '10.5px').text(bb);
  });

  function setRange() {
    slider.min = String(-b0[which]);
    slider.max = '40';
    if (Number(slider.value) < -b0[which]) slider.value = String(-b0[which]);
  }

  function render() {
    const i = which;
    const lo = -b0[i];
    const hi = 40;
    const base = solveWith(i, 0);
    const price = shadow(i);
    const pts = [];
    const steps = 240;
    for (let k = 0; k <= steps; k++) {
      const d = lo + ((hi - lo) * k) / steps;
      pts.push([d, solveWith(i, d)]);
    }
    /* where the straight line stops telling the truth, measured on the curve */
    let breakAt = null;
    for (const [d, v] of pts) {
      if (d <= 0) continue;
      if (Math.abs(v - (base + price * d)) > 1e-7) { breakAt = d; break; }
    }
    if (breakAt !== null) {
      let loB = 0;
      let hiB = breakAt;
      for (let k = 0; k < 60; k++) {
        const mid = (loB + hiB) / 2;
        if (Math.abs(solveWith(i, mid) - (base + price * mid)) < 1e-7) loB = mid;
        else hiB = mid;
      }
      breakAt = (loB + hiB) / 2;
    }

    x.domain([lo, hi]);
    const top = Math.max(d3.max(pts, (p) => p[1]), base + price * hi);
    y.domain([Math.min(0, d3.min(pts, (p) => p[1])), top * 1.08]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: `extra ${names[i]}`, y: 'best profit', leftBudget: margin.left });

    curve.attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts));
    tangent.attr('d', `M${x(lo)},${y(base + price * lo)}L${x(hi)},${y(base + price * hi)}`);
    const delta = Number(slider.value);
    const here = solveWith(i, delta);
    marker.attr('cx', x(delta)).attr('cy', y(here));
    label.textContent = `${delta > 0 ? '+' : ''}${delta}`;

    if (breakAt !== null && breakAt < hi) {
      breakLine.style('display', null)
        .attr('x1', x(breakAt)).attr('x2', x(breakAt)).attr('y1', 0).attr('y2', h);
      /* short, and on whichever side has room: at 40 characters it ran into
         the legend, which lives outside the plot precisely to be safe */
      const right = x(breakAt) > w * 0.4;
      breakText.style('display', null)
        .attr('x', x(breakAt) + (right ? -6 : 6)).attr('y', 14)
        .attr('text-anchor', right ? 'end' : 'start')
        .text(`holds to +${breakAt.toFixed(2)}`);
    } else {
      breakLine.style('display', 'none');
      breakText.style('display', 'none');
    }

    title.text(`${names[i]}: every point is the whole problem solved again`);
    sub.text(`at ${delta > 0 ? '+' : ''}${delta}, the best profit is ${here.toFixed(3)}`);

    const row = data.duals.rows[i];
    const slack = b0[i] - (A[i][0] * simplex(c, A, b0).x[0] + A[i][1] * simplex(c, A, b0).x[1]);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>limit</th><th>amount</th><th>spare</th><th>price</th><th>holds to</th></tr>
        ${data.duals.rows.map((rr, k) => `
          <tr>
            <td>${k === i ? '<span class="value">' : ''}${rr.name}${k === i ? '</span>' : ''}</td>
            <td>${rr.b}</td>
            <td>${rr.slack}</td>
            <td>${rr.shadow}</td>
            <td>${rr.shadow > 0 ? `+${rr.valid_up_to}` : 'nothing changes'}</td>
          </tr>`).join('')}
      </table>
      <div class="gen-note">
        Solved here rather than read from the file: one more unit of
        <span class="value">${names[i]}</span> is worth
        <span class="value">${price.toFixed(4)}</span>, against the
        ${row.shadow} the solver's dual reports, a difference of
        ${Math.abs(price - row.shadow).toExponential(1)}.
        ${price > 1e-9
    ? `That price is only a local truth. It holds up to
           <span class="value">+${breakAt === null ? 'the end of this range' : breakAt.toFixed(2)}</span>,
           where a different pair of constraints takes over and the curve bends;
           past that, buying more of this is worth
           ${breakAt === null ? 'the same' : 'less'}.`
    : `This one has <span class="value">${slack.toFixed(2)}</span> left over at the
           optimum, so its price is zero: the workshop is not short of it, and
           buying more buys nothing. Push the slider left instead, and watch
           where the flat part ends.`}
      </div>`;
  }

  controls.selectAll('button').data(names).join('button')
    .attr('class', (_, k) => `atlas-btn${k === which ? '' : ' ghost'}`)
    .text((d) => d)
    .on('click', (_, d) => {
      which = names.indexOf(d);
      controls.selectAll('button')
        .attr('class', (_, k) => `atlas-btn${k === which ? '' : ' ghost'}`);
      setRange();
      render();
    });

  slider.addEventListener('input', render);
  setRange();
  render();
  return { render };
}
