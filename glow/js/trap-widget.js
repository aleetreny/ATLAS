/* What a continuous density does when it is fitted to discrete pixels.
 *
 * Pixel values are integers. A density has no obligation to be finite on a set
 * of measure zero, so it can put a spike on each of the 256 values a pixel can
 * take and report an arbitrarily large likelihood without having learned
 * anything about pictures. This widget is that arm, watched over training, with
 * the honest arm drawn as a line beside it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initTrapWidget(data) {
  const T = data.trap;
  const C = data.compress;
  const chart = makeChart('#trap-chart', {
    width: 640, height: 360, margin: { top: 52, right: 30, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#trap-readout');

  const rows = T.rows;
  const x = d3.scaleLinear().domain([0, rows[rows.length - 1].epochs * 1.05]).range([0, w]);
  const all = rows.map((r) => r.bpd).concat([T.honest, C.gzip, 0]);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const y = d3.scaleLinear().domain([lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1]).range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 6 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 6 });
  axisLabels(g, w, h, { x: 'epochs of training', y: 'bits per pixel it reports' });

  const layer = g.append('g');
  layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('opacity', 0.6);
  layer.append('text').attr('class', 'chart-note')
    .attr('x', w - 4).attr('y', y(0) - 7).attr('text-anchor', 'end')
    .style('font-size', '11.5px').text('zero bits, which is not a code length');

  [['with uniform dequantisation', T.honest, 'var(--anchor)'],
    ['gzip on the same bytes', C.gzip, 'var(--squidink)']].forEach(([label, v, colour], i) => {
    layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(v)).attr('y2', y(v))
      .attr('stroke', colour).attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y(v) + (i === 0 ? -7 : 15))
      .style('font-size', '11.5px').attr('fill', colour).text(label);
  });

  layer.append('path')
    .attr('d', d3.line().x((r) => x(r.epochs)).y((r) => y(r.bpd))(rows))
    .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
  layer.selectAll('circle').data(rows).join('circle')
    .attr('cx', (r) => x(r.epochs)).attr('cy', (r) => y(r.bpd)).attr('r', 4.5)
    .attr('fill', 'var(--primary)');
  layer.append('text').attr('class', 'chart-note')
    .attr('x', x(rows[rows.length - 1].epochs) - 6).attr('y', y(rows[rows.length - 1].bpd) + 18)
    .attr('text-anchor', 'end').style('font-size', '11.5px').attr('fill', 'var(--primary)')
    .text('with no dequantisation at all');

  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');
  wrapLabel(title, 'the same architecture, the same data, one line of noise removed', w - 8);

  const last = rows[rows.length - 1];
  readout.innerHTML = `
    <table class="gen-table">
      <tr>
        <th>epochs</th>
        ${rows.map((r) => `<th>${r.epochs}</th>`).join('')}
        <th>with the noise</th>
      </tr>
      <tr>
        <td>bits per pixel</td>
        ${rows.map((r) => `<td><span class="value">${r.bpd}</span></td>`).join('')}
        <td><span class="value">${T.honest}</span></td>
      </tr>
      <tr>
        <td>mean log density, nats</td>
        ${rows.map((r) => `<td>${r.logp}</td>`).join('')}
        <td></td>
      </tr>
    </table>
    <div class="gen-note">
      ${last.bpd < 0
    ? `The arm with no dequantisation reports <span class="value">${last.bpd}</span> bits per
       pixel, which is not a small number of bits, it is a negative number of bits, and there is
       no such thing. What it is measuring is a density in units of one over pixel value, free
       to be as tall as it likes on the 256 values a pixel can actually take, and it is still
       climbing when training stops: ${rows.map((r) => r.bpd).join(', ')} at
       ${rows.map((r) => r.epochs).join(', ')} epochs.`
    : `The arm with no dequantisation reports ${last.bpd} bits per pixel after
       ${last.epochs} epochs, against ${T.honest} for the same architecture with the noise. On
       this data at this size the divergence has not gone far enough to cross zero, and the page
       reports where it actually got to rather than where the argument says it goes.`}
      With uniform noise added before training, the same architecture reports ${T.honest}, and
      that number means something: the continuous model's expected log density is a lower bound
      on the discrete model's log likelihood, so the bits it reports are an upper bound on a real
      code length. That is why every flow paper that quotes bits per dimension quotes the
      dequantised number, and why the one above is not a result at all.
    </div>`;

  return { render: () => {} };
}
