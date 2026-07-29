/* The experiment that separates obeying from knowing.
 *
 * A second model was trained with two labels removed from its data entirely.
 * Then it was asked for them. If guidance conjured concepts, turning the dial
 * up would eventually produce something; if it only amplifies a token the model
 * already has a meaning for, it cannot.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initHeldWidget(data) {
  const H = data.held;
  const readout = document.querySelector('#held-readout');

  const chart = makeChart('#held-chart', {
    width: 640, height: 340, margin: { top: 52, right: 34, bottom: 58, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const scales = [...new Set(H.rows.map((r) => r.scale))].sort((a, b) => a - b);
  const x = d3.scalePoint().domain(scales.map(String)).range([0, w]).padding(0.6);
  const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.0%') });
  axisLabels(g, w, h, { x: 'guidance scale', y: 'makes what was asked for' });

  /* chance, which is the number the unseen labels should be compared against */
  layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0.1)).attr('y2', y(0.1))
    .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
  layer.append('text').attr('class', 'chart-note')
    .attr('x', 2).attr('y', y(0.1) - 7).style('font-size', '11.5px')
    .attr('fill', 'var(--anchor)').text('one label in ten, which is guessing');

  [[true, 'labels it was trained on', 'var(--primary)'],
    [false, 'labels removed from training', 'var(--squidink)']]
    .forEach(([seen, label, colour], si) => {
      const pts = scales.map((s) => {
        const rows = H.rows.filter((r) => r.scale === s && r.seen === seen);
        return { s, v: rows.reduce((a, r) => a + r.obeys, 0) / rows.length };
      });
      layer.append('path')
        .attr('d', d3.line().x((p) => x(String(p.s))).y((p) => y(p.v))(pts))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.4)
        .attr('stroke-dasharray', si === 1 ? '6 4' : null);
      layer.selectAll(`circle.s${si}`).data(pts).join('circle').attr('class', `s${si}`)
        .attr('cx', (p) => x(String(p.s))).attr('cy', (p) => y(p.v)).attr('r', 5)
        .attr('fill', colour);
      /* The unseen curve sits on the floor of the panel, so a label below it
         lands on the x axis ticks: both names go ABOVE their own last point,
         and they are far apart in y by construction. */
      const last = pts[pts.length - 1];
      layer.append('text').attr('class', 'chart-note')
        .attr('x', x(String(last.s)) - 8).attr('y', y(last.v) - 11)
        .attr('text-anchor', 'end')
        .style('font-size', '11.5px').attr('fill', colour).text(label);
    });

  wrapLabel(title, H.unseen < 0.1
    ? 'asking harder for something it never saw makes it less likely, not more'
    : 'what the dial does for a label the model never met', w - 8);

  readout.innerHTML = `
    <table class="gen-table">
      <tr>
        <th>guidance</th><th>label</th><th>was in training</th><th>makes what was asked</th>
      </tr>
      ${H.rows.map((r) => `<tr>
        <td>${r.scale}</td>
        <td>${r.label}</td>
        <td>${r.seen ? 'yes' : '<span class="bold">no</span>'}</td>
        <td><span class="value">${(r.obeys * 100).toFixed(1)}%</span></td>
      </tr>`).join('')}
    </table>
    <div class="gen-note">
      A second model, trained on ${H.n_train.toLocaleString('en-US')} digits with every
      ${H.labels.join(' and ')} taken out of its data completely. Averaged over the settings
      above it produces the requested digit <span class="value">${(H.seen * 100).toFixed(1)}%</span>
      of the time for labels it saw and <span class="value">${(H.unseen * 100).toFixed(1)}%</span>
      for the two it did not.
      ${H.unseen < 0.1
    ? `That second number is below guessing, and turning the dial up makes it worse rather than
       better: the guidance direction is "away from the average digit and towards this token", and
       for a token with no meaning attached that is a push into nowhere in particular.`
    : `That second number is around what guessing would give, so the dial is not conjuring the
       missing digit, but it is not actively pushing away from it either.`}
      The reading is narrow and worth keeping narrow. Guidance amplifies a correspondence the
      model was trained to have. It does not create one, and no setting of it will.
    </div>`;

  return { render: () => {} };
}
