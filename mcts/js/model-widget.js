/* A plan made inside a learnt model, and how far into it the plan still
 * describes anything.
 *
 * The world is linear and so is the model, on purpose: with the shape of the
 * model right by construction, what the slider changes is only how much
 * experience it was fitted on, and the answer stops being about architecture.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initModelWidget(data) {
  const M = data.model;
  const readout = document.querySelector('#model-readout');
  const slider = document.querySelector('#model-slider');
  const value = document.querySelector('#model-value');
  slider.max = String(M.rows.length - 1);
  let idx = 1;

  const chart = makeChart('#model-chart', {
    width: 640, height: 340, margin: { top: 50, right: 128, bottom: 56, left: 74 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -32)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -13)
    .style('font-size', '11.5px');

  const ks = M.rows[0].horizon.map((d) => d.k);
  const allErr = M.rows.flatMap((r) => r.horizon.map((d) => d.error));
  const x = d3.scaleLog().domain([ks[0] * 0.85, ks[ks.length - 1] * 1.2]).range([0, w]);
  const y = d3.scaleLog().domain([Math.min(...allErr) * 0.7, Math.max(...allErr) * 1.5])
    .range([h, 0]);

  function render() {
    const row = M.rows[idx];
    value.textContent = row.samples.toLocaleString('en-US');
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    drawGrid(g, x, y, w, h, { xValues: ks, yValues: [0.0002, 0.002, 0.02, 0.2] });
    drawAxes(g, x, y, w, h, { xValues: ks, yValues: [0.0002, 0.002, 0.02, 0.2],
      xFmt: d3.format(',d'), yFmt: d3.format('.4f') });
    axisLabels(g, w, h, { x: 'steps imagined before looking',
      y: 'how far the imagined state is', leftBudget: margin.left });
    plot.selectAll('*').remove();
    M.rows.forEach((r, i) => {
      const on = i === idx;
      plot.append('path').attr('fill', 'none')
        .attr('stroke', on ? 'var(--primary)' : '#c9ced0')
        .attr('stroke-width', on ? 2.8 : 1.6)
        .attr('d', d3.line().x((d) => x(d.k)).y((d) => y(d.error))(r.horizon));
      if (on) {
        plot.selectAll(null).data(r.horizon).join('circle')
          .attr('cx', (d) => x(d.k)).attr('cy', (d) => y(d.error)).attr('r', 4)
          .attr('fill', 'var(--primary)');
      }
    });
    /* the one step error measured the way anybody measures it: against what
       actually happened, noise and all, which is why it does not move */
    plot.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(M.one_step_floor))
      .attr('y2', y(M.one_step_floor)).attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
    /* at the right end: on the left the selected curve crosses the line and the
       label sat on top of it */
    plot.append('text').attr('class', 'chart-note').attr('x', w - 2)
      .attr('y', y(M.one_step_floor) - 6).attr('text-anchor', 'end')
      .style('font-size', '10.5px').attr('fill', 'var(--anchor)')
      .text(`the world's own noise: ${M.one_step_floor}`);
    key.selectAll('*').remove();
    [['var(--primary)', ['this much', 'experience']], ['#c9ced0', ['the other', 'three']],
      ['var(--anchor)', ['what one step', 'cannot beat'], '5 4']]
      .forEach(([col, lines, dash], i) => {
        key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 42 + 6)
          .attr('y2', i * 42 + 6).attr('stroke', col).attr('stroke-width', 2.6)
          .attr('stroke-dasharray', dash || null);
        lines.forEach((t, j) => {
          key.append('text').attr('class', 'chart-note').attr('x', 0)
            .attr('y', i * 42 + 20 + j * 11).style('font-size', '10px').text(t);
        });
      });
    title.text(`fitted on ${row.samples.toLocaleString('en-US')} steps of experience`);
    sub.text('a cart with a position and a speed, pushed at random, noise every step');

    const first = M.rows[0];
    const last = M.rows[M.rows.length - 1];
    const far = (r) => r.horizon[r.horizon.length - 1];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>steps of experience</th>${M.rows.map((r) => `<th>${r.samples.toLocaleString('en-US')}</th>`).join('')}</tr>
        <tr><td>error after one step</td>
            ${M.rows.map((r) => `<td>${r.one_step}</td>`).join('')}</tr>
        <tr><td>error after ${far(first).k} steps</td>
            ${M.rows.map((r) => `<td>${far(r).error}</td>`).join('')}</tr>
        <tr><td>worst weight, off by</td>
            ${M.rows.map((r) => `<td>${r.weight_error}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Read the first row and then the second. Measured the way anyone would
        measure a model, one step ahead against what actually happened, going
        from ${first.samples.toLocaleString('en-US')} steps of experience to
        ${last.samples.toLocaleString('en-US')} buys nothing at all:
        ${first.one_step} becomes ${last.one_step}, because both are already
        sitting on the world's own noise, ${M.one_step_floor}. The model behind
        that flat number is meanwhile getting
        ${(first.weight_error / last.weight_error).toFixed(0)} times more
        accurate, and imagining ${far(first).k} steps says so:
        <span class="value">${far(first).error}</span> becomes
        ${far(last).error}, ${(far(first).error / far(last).error).toFixed(0)}
        times closer. A one step score cannot tell you whether a plan made in
        the model is worth anything. Only planning in it can.
      </div>`;
  }

  slider.addEventListener('input', () => { idx = Number(slider.value); render(); });
  slider.value = String(idx);
  render();
  return { render };
}
