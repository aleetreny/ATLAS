/* The term the derivation quietly drops.
 *
 * The training objective has three parts and everybody writes the first one as
 * "a constant that does not depend on the parameters, so ignore it". It is
 * KL(q(x_T) || N(0, I)): how much of the data is still there when the forward
 * process is supposed to have finished. Ignoring it is fine when it is zero.
 * Whether it is zero is a number, and in two dimensions that number can be
 * computed exactly rather than hoped for, which is what this chart is.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initPriorWidget(data) {
  const P = data.prior;
  const NAMES = ['linear', 'cosine', 'quadratic'];
  const COLOURS = { linear: 'var(--primary)', cosine: 'var(--anchor)', quadratic: 'var(--squidink)' };
  const buttons = document.querySelector('#prior-buttons');
  const readout = document.querySelector('#prior-readout');
  const st = { view: 'prior' };

  const chart = makeChart('#prior-chart', {
    width: 640, height: 350, margin: { top: 52, right: 34, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const btns = {};
  for (const [key, text] of [['prior', 'what is left at the end'],
    ['destroy', 'what is left along the way']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'prior') {
      const Ts = P.linear.rows.map((r) => r.T);
      const all = NAMES.flatMap((n) => P[n].rows.map((r) => Math.max(r.kl, 1e-17)));
      const x = d3.scaleLog().domain([80, 1300]).range([0, w]);
      const y = d3.scaleLog().domain([Math.min(...all) * 0.3, Math.max(...all) * 3]).range([h, 0]);
      const decades = d3.range(-17, 2).map((e) => 10 ** e)
        .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
      drawGrid(g, x, y, w, h, { xValues: Ts, yValues: decades });
      drawAxes(g, x, y, w, h, { xValues: Ts, yValues: decades, yFmt: d3.format('.0e') });
      axisLabels(g, w, h, { x: 'steps in the schedule', y: 'nats still there at the end' });

      /* the ceiling: the entropy of the truth. A prior term above this is
         larger than everything the model was trying to represent. */
      const ceil = Math.abs(data.meta.ceiling);
      if (ceil >= y.domain()[0] && ceil <= y.domain()[1]) {
        layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(ceil)).attr('y2', y(ceil))
          .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6).attr('stroke-dasharray', '7 4');
        layer.append('text').attr('class', 'chart-note')
          .attr('x', 2).attr('y', y(ceil) - 7).style('font-size', '11.5px')
          .attr('fill', 'var(--anchor)')
          .text(`the whole entropy of the truth, ${ceil} nats`);
      }
      NAMES.forEach((n, i) => {
        const rows = P[n].rows;
        layer.append('path')
          .attr('d', d3.line().x((r) => x(r.T)).y((r) => y(Math.max(r.kl, 1e-17)))(rows))
          .attr('fill', 'none').attr('stroke', COLOURS[n]).attr('stroke-width', 2);
        layer.selectAll(`circle.s${i}`).data(rows).join('circle').attr('class', `s${i}`)
          .attr('cx', (r) => x(r.T)).attr('cy', (r) => y(Math.max(r.kl, 1e-17))).attr('r', 4)
          .attr('fill', COLOURS[n]);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 4).attr('y', 12 + i * 16).attr('text-anchor', 'end')
          .style('font-size', '11.5px').attr('fill', COLOURS[n]).text(n);
      });
      wrapLabel(title, 'cut the schedule short and the term you dropped is the biggest one', w - 8);
    } else {
      const rows = data.destroy.rows;
      const x = d3.scaleLinear().domain([0, data.meta.steps]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.kl_normal) * 1.08]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      axisLabels(g, w, h, { x: 'noise level t', y: 'nats from a standard normal' });
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.t)).y((r) => y(r.kl_normal))(rows))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
      layer.selectAll('circle').data(rows).join('circle')
        .attr('cx', (r) => x(r.t)).attr('cy', (r) => y(r.kl_normal)).attr('r', 3.6)
        .attr('fill', 'var(--primary)');
      const half = rows.find((r) => r.t === data.destroy.half_t);
      if (half) {
        layer.append('line').attr('x1', x(half.t)).attr('x2', x(half.t))
          .attr('y1', y(half.kl_normal)).attr('y2', h)
          .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.3).attr('stroke-dasharray', '4 3');
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(half.t) + 6).attr('y', y(half.kl_normal) - 8).style('font-size', '11.5px')
          .attr('fill', 'var(--anchor)').text(`half gone by t = ${half.t}`);
      }
      wrapLabel(title, 'every point is the exact integral, not an average over draws', w - 8);
    }

    const best = P[data.best_prior];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>schedule</th>${P.linear.rows.map((r) => `<th>T = ${r.T}</th>`).join('')}</tr>
        ${NAMES.map((n) => `<tr>
          <td>${n === data.best_prior ? '<span class="bold">' + n + '</span>' : n}</td>
          ${P[n].rows.map((r) => `<td>${r.kl}</td>`).join('')}
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        Every cell is KL(q(x_T) &Vert; N(0, I)) computed exactly, and it is the term the
        derivation of the training loss drops as a constant. At the full
        ${data.meta.steps} steps that is fair: ${data.best_prior} leaves
        <span class="value">${best.kl_end}</span> nats behind, which is nothing.
        Cut the schedule to ${P.linear.rows[0].T} steps and it is
        <span class="value">${P.linear.rows[0].kl}</span> for the linear schedule, against a
        truth whose entire entropy is ${Math.abs(data.meta.ceiling)} nats: the term everyone
        ignores is then larger than the thing being modelled. The other reading is the second
        view, where the same integral is taken along the schedule rather than at its end, and
        half of the data is gone by t = ${data.destroy.half_t} of ${data.meta.steps}.
      </div>`;
  }

  render();
  return { render };
}
