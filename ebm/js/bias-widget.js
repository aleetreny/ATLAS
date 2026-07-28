/* The bias of the shortcut, against a gradient that is not a shortcut.
 *
 * Every arm here is the same architecture, the same initialisation and the same
 * number of updates; the only thing that changes is where the negative term of
 * the gradient comes from. One arm computes it by quadrature, which no energy
 * model on real data can do and which is the entire reason this module measures
 * everything in two dimensions.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initBiasWidget(data) {
  const A = data.arms;
  const chart = makeChart('#bias-chart', {
    width: 640, height: 360, margin: { top: 52, right: 28, bottom: 60, left: 76 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#bias-buttons');
  const readout = document.querySelector('#bias-readout');
  const st = { view: 'kl' };

  const btns = {};
  for (const [key, text] of [['kl', 'how far each arm ends from the truth'],
    ['trace', 'how they got there'],
    ['gradient', 'the gradient itself']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const exact = A.rows.find((r) => r.mode === 'exact');
  const cd = A.rows.filter((r) => r.mode === 'cd');
  const pcd = A.rows.filter((r) => r.mode === 'pcd');

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'kl') {
      const ks = cd.map((r) => r.k);
      const x = d3.scaleLog().domain([ks[0] * 0.7, ks[ks.length - 1] * 1.5]).range([0, w]);
      const all = A.rows.map((r) => r.kl);
      const y = d3.scaleLog().domain([Math.min(...all) * 0.6, Math.max(...all) * 1.8]).range([h, 0]);
      const decades = [1e-2, 1e-1, 1, 10, 100].filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
      drawGrid(g, x, y, w, h, { xValues: ks, yValues: decades });
      drawAxes(g, x, y, w, h, { xValues: ks, yValues: decades, xFmt: d3.format(',d'), yFmt: d3.format('.0e') });
      axisLabels(g, w, h, { x: 'sampling steps per update', y: 'exact distance from the truth, in nats' });

      [['from the data, k steps', cd, 'var(--primary)'],
        ['from a buffer that never restarts', pcd, 'var(--anchor)']]
        .forEach(([label, rows, colour], si) => {
          layer.append('path')
            .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r.kl))(rows))
            .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
          layer.selectAll(`circle.s${si}`).data(rows).join('circle')
            .attr('class', `s${si}`)
            .attr('cx', (r) => x(r.k)).attr('cy', (r) => y(r.kl)).attr('r', 4.5)
            .attr('fill', colour);
          rows.forEach((r) => {
            layer.append('line').attr('x1', x(r.k)).attr('x2', x(r.k))
              .attr('y1', y(Math.min(...r.kl_seeds))).attr('y2', y(Math.max(...r.kl_seeds)))
              .attr('stroke', colour).attr('stroke-width', 1.4).attr('opacity', 0.75);
          });
          layer.append('text').attr('class', 'chart-note')
            .attr('x', w - 4).attr('y', 12 + si * 16).attr('text-anchor', 'end')
            .style('font-size', '11.5px').attr('fill', colour).text(label);
        });
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(exact.kl)).attr('y2', y(exact.kl))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
      layer.append('text').attr('class', 'chart-note')
        .attr('x', 2).attr('y', y(exact.kl) - 7).style('font-size', '11.5px')
        .text('the arm with an exact gradient');
      wrapLabel(title, 'the same model, the same budget, a different negative term', w - 8);
    } else if (st.view === 'trace') {
      const rows = [exact, cd[0], cd[cd.length - 1], pcd[pcd.length - 1]];
      const upd = exact.trace.map((t) => t.update);
      const x = d3.scaleLinear().domain([0, upd[upd.length - 1]]).range([0, w]);
      const all = rows.flatMap((r) => r.trace.map((t) => t.kl));
      const y = d3.scaleLog().domain([Math.min(...all) * 0.7, Math.max(...all) * 1.5]).range([h, 0]);
      const decades = [1e-2, 1e-1, 1, 10, 100].filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yValues: decades });
      drawAxes(g, x, y, w, h, { xTicks: 5, yValues: decades, yFmt: d3.format('.0e') });
      axisLabels(g, w, h, { x: 'updates', y: 'exact distance from the truth, in nats' });
      const colours = ['var(--squidink)', 'var(--primary)', 'var(--cosmos)', 'var(--anchor)'];
      rows.forEach((r, i) => {
        layer.append('path')
          .attr('d', d3.line().x((t) => x(t.update)).y((t) => y(t.kl))(r.trace))
          .attr('fill', 'none').attr('stroke', colours[i]).attr('stroke-width', 2);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 4).attr('y', 12 + i * 15).attr('text-anchor', 'end')
          .style('font-size', '11px').attr('fill', colours[i]).text(r.arm);
      });
      wrapLabel(title, 'measured every hundred updates, by quadrature, on the model as it was', w - 8);
    } else {
      const rows = data.gradient.rows.filter((r) => r.batch === data.gradient.rows[0].batch);
      const ks = [...new Set(rows.map((r) => r.k))].sort((a, b) => a - b);
      const chks = [...new Set(rows.map((r) => r.update))];
      const x = d3.scaleLog().domain([ks[0] * 0.7, ks[ks.length - 1] * 1.5]).range([0, w]);
      const y = d3.scaleLinear().domain([-0.4, 1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: ks, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: ks, yTicks: 5, xFmt: d3.format(',d') });
      axisLabels(g, w, h, { x: 'sampling steps', y: 'agreement with the exact gradient' });
      const colours = ['var(--anchor)', 'var(--aqua)', 'var(--cosmos)', 'var(--primary)'];
      chks.forEach((c, i) => {
        const rs = rows.filter((r) => r.update === c).sort((a, b) => a.k - b.k);
        layer.append('path')
          .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r.cos_of_mean))(rs))
          .attr('fill', 'none').attr('stroke', colours[i % colours.length]).attr('stroke-width', 2);
        layer.selectAll(`circle.c${i}`).data(rs).join('circle')
          .attr('class', `c${i}`).attr('cx', (r) => x(r.k)).attr('cy', (r) => y(r.cos_of_mean))
          .attr('r', 3.6).attr('fill', colours[i % colours.length]);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 4).attr('y', 12 + i * 15).attr('text-anchor', 'end')
          .style('font-size', '11px').attr('fill', colours[i % colours.length])
          .text(`after ${c} updates`);
      });
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(1)).attr('y2', y(1))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('stroke-dasharray', '5 4');
      wrapLabel(title, 'the direction the shortcut points, against the direction it should', w - 8);
    }

    const cd1 = cd[0];
    const cdN = cd[cd.length - 1];
    const pcd1 = pcd[0];
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>arm</th><th>distance from the truth</th><th>seeds</th>
          <th>held out log density</th><th>energy evaluations it spent</th>
        </tr>
        ${A.rows.map((r) => `<tr>
          <td>${r.arm === exact.arm ? '<span class="bold">' + r.arm + '</span>' : r.arm}</td>
          <td><span class="value">${r.kl}</span></td>
          <td>${r.kl_seeds.join(', ')}</td>
          <td>${r.heldout}</td>
          <td>${r.negative_evals.toLocaleString('en-US')}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        The exact arm lands ${exact.kl} nats from the truth and everything else is measured
        against that. One sampling step from the data costs ${A.cd1_bias} nats on top of it and
        ${cdN.k} steps cost ${A.cd100_bias}; a persistent buffer at one step costs
        ${A.pcd1_bias}, which is not a bias, it is a failure.
        <span class="bold">And the seeds do not agree</span>: the median spread across arms is
        ${A.floor} nats, so ${Math.abs(A.cd1_bias) > A.floor
    ? `the cost of ${cd1.k} step is outside it and the cost of ${cdN.k} steps (${A.cd100_bias}) is not`
    : `even the cost of one step sits inside it`}. What this page can say without a threshold
        is the ordering, which holds on every seed, and the cost, which is
        ${(cdN.negative_evals / cd1.negative_evals).toFixed(0)} times the energy evaluations
        between the cheapest and the most expensive sampled arm, against
        ${(exact.negative_evals / cd1.negative_evals).toFixed(0)} for the arm that does the
        integral.
      </div>`;
  }

  render();
  return st;
}
