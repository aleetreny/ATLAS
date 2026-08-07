/* The design problem, in one chart: two costs that pull opposite ways.
 *
 * Make the bottleneck smaller and the diffusion gets easy, but the ceiling
 * drops and no amount of diffusion can climb back through it. Make it larger
 * and the ceiling rises out of reach, but now the score network has a harder
 * distribution to learn and stops arriving at it. The crossover is not a matter
 * of taste: it is where the two curves swap which one is on top, and this chart
 * is that measurement.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initTradeWidget(data) {
  const T = data.trade;
  const readout = document.querySelector('#trade-readout');

  const chart = makeChart('#trade-chart', {
    width: 640, height: 360, margin: { top: 52, right: 34, bottom: 58, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const ks = T.map((r) => r.k);
  const xTicks = ks.filter((_, i) => i === 0 || i === ks.length - 1 || i % 2 === 0);
  const vals = T.flatMap((r) => [Math.abs(r.mmd2), Math.abs(r.ceiling_mmd2)])
    .concat([data.floor.mmd2_max]);
  const x = d3.scaleLog().base(2).domain([ks[0] * 0.75, ks[ks.length - 1] * 1.35]).range([0, w]);
  const y = d3.scaleLog().domain([Math.min(...vals) * 0.5, Math.max(...vals) * 2]).range([h, 0]);
  const decades = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
  drawGrid(g, x, y, w, h, { xValues: xTicks, yValues: decades });
  drawAxes(g, x, y, w, h, { xValues: ks, yValues: decades, yFmt: d3.format('.0e') });
  axisLabels(g, w, h, { x: 'numbers in the bottleneck', y: 'distance from real digits' });

  layer.append('rect').attr('x', 0).attr('y', y(data.floor.mmd2_max))
    .attr('width', w).attr('height', Math.max(0, h - y(data.floor.mmd2_max)))
    .attr('fill', 'var(--anchor)').attr('opacity', 0.1);
  layer.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', y(data.floor.mmd2_max)).attr('y2', y(data.floor.mmd2_max))
    .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
  layer.append('text').attr('class', 'chart-note')
    .attr('x', 2).attr('y', y(data.floor.mmd2_max) + 14).style('font-size', '11.5px')
    .attr('fill', 'var(--anchor)').text('real against real');

  /* the gap between the two curves is the thing to see, so it is shaded */
  layer.append('path')
    .attr('d', `${d3.line().x((r) => x(r.k)).y((r) => y(Math.abs(r.ceiling_mmd2)))(T)}L`
      + `${[...T].reverse().map((r) => `${x(r.k)},${y(Math.abs(r.mmd2))}`).join('L')}Z`)
    .attr('fill', 'var(--primary)').attr('opacity', 0.12);

  [['what the decoder allows', (r) => Math.abs(r.ceiling_mmd2), 'var(--squidink)', '6 4'],
    ['what the diffusion reaches', (r) => Math.abs(r.mmd2), 'var(--primary)', null]]
    .forEach(([label, f, colour, dash], si) => {
      const p = layer.append('path')
        .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(f(r)))(T))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
      if (dash) p.attr('stroke-dasharray', dash);
      layer.selectAll(`circle.s${si}`).data(T).join('circle').attr('class', `s${si}`)
        .attr('cx', (r) => x(r.k)).attr('cy', (r) => y(f(r))).attr('r', 4.5)
        .attr('fill', colour);
      /* At the LEFT end the two curves are almost on top of each other, so a
         label there sits beside the other line's point and reads as its name.
         At the right end they have separated, which is the whole finding, so
         that is where the names go: the lower curve labelled below itself and
         the upper one above, or both land in the gap between them. */
      const last = T[T.length - 1];
      layer.append('text').attr('class', 'chart-note')
        .attr('x', x(last.k) - 10).attr('y', y(f(last)) + (si === 0 ? 16 : -10))
        .attr('text-anchor', 'end')
        .style('font-size', '11.5px').attr('fill', colour).text(label);
    });

  const short = T.filter((r) => !r.at_ceiling);
  wrapLabel(title, short.length
    ? `the two meet up to ${T.filter((r) => r.at_ceiling).slice(-1)[0].k} numbers and separate after`
    : 'every size here reaches its own ceiling', w - 8);

  const best = T.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));
  readout.innerHTML = `
    <table class="gen-table">
      <tr>
        <th>bottleneck</th><th>ceiling</th><th>reached</th>
        <th>total weights</th><th>at its ceiling</th>
      </tr>
      ${T.map((r) => `<tr>
        <td>${r.k === best.k ? '<span class="bold">' + r.k + '</span>' : r.k}</td>
        <td>${r.ceiling_mmd2}</td>
        <td><span class="value">${r.mmd2}</span></td>
        <td>${r.params.toLocaleString('en-US')}</td>
        <td>${r.at_ceiling ? 'yes' : '<span class="bold">no</span>'}</td>
      </tr>`).join('')}
    </table>
    <div class="gen-note">
      Read the two columns in the middle against each other, because which one is binding tells
      you which half of the model to spend on.
      ${data.at_ceiling.length
    ? `At ${data.at_ceiling.join(', ')} numbers the sampler lands on its own ceiling, so the
       diffusion is already as good as it can usefully be and every extra weight spent on it is
       wasted: the autoencoder is what is holding the result down.`
    : 'No size here reaches its own ceiling, so on this data the diffusion is the binding half everywhere.'}
      ${short.length
    ? `At ${short.map((r) => r.k).join(', ')} it does not: the ceiling is
       ${short[0].ceiling_mmd2} and the sampler only gets to ${short[0].mmd2}, so now it is the
       diffusion that is behind and a bigger score network would pay. The crossover is between
       ${data.at_ceiling.length ? data.at_ceiling[data.at_ceiling.length - 1] : ks[0]} and
       ${short[0].k}.`
    : ''}
      The best of these is ${best.k} numbers at <span class="value">${best.mmd2}</span> on
      ${best.params.toLocaleString('en-US')} weights.
      ${data.ink.confounded
    ? `The shared judge cannot rank these: its confidence correlates ${data.ink.corr} with the
       ink of the samples, which is the failure the energy article measured on it.`
    : `The judge's opinion is usable here for once (its confidence correlates ${data.ink.corr}
       with ink, below the bar where it stops meaning anything), but the two sample statistic is
       the stronger measure and it is what this table reports.`}
    </div>`;

  return { render: () => {} };
}
