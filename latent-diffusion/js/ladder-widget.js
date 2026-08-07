/* Two ladders on one pair of axes, which is the only honest way to compare them.
 *
 * The latent model's cost is not just its score network: it cannot produce a
 * picture without its decoder, so the decoder's weights are part of the price
 * and are counted here. Even so the two curves do not overlap, and that gap is
 * the article.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initLadderWidget(data) {
  const buttons = document.querySelector('#ladder-buttons');
  const readout = document.querySelector('#ladder-readout');
  const st = { split: false };

  const chart = makeChart('#ladder-chart', {
    width: 640, height: 360, margin: { top: 52, right: 34, bottom: 58, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const b = document.createElement('button');
  b.className = 'atlas-btn ghost';
  b.textContent = 'count the decoder separately';
  b.addEventListener('click', () => {
    st.split = !st.split;
    b.textContent = st.split ? 'count the decoder in the total' : 'count the decoder separately';
    render();
  });
  buttons.appendChild(b);

  const bestLat = data.ladder.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));
  const bestPix = data.pixel_ladder.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));

  function render() {
    layer.selectAll('*').remove();
    const latX = (r) => (st.split ? r.diffusion_params : r.params);
    const all = data.ladder.map(latX).concat(data.pixel_ladder.map((r) => r.params));
    const vals = data.ladder.map((r) => Math.abs(r.mmd2))
      .concat(data.pixel_ladder.map((r) => Math.abs(r.mmd2)))
      .concat([data.floor.mmd2_max]);
    const x = d3.scaleLog().domain([Math.min(...all) * 0.6, Math.max(...all) * 1.6]).range([0, w]);
    const y = d3.scaleLog().domain([Math.min(...vals) * 0.5, Math.max(...vals) * 2]).range([h, 0]);
    const decades = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
      .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xTicks: 3, yValues: decades });
    drawAxes(g, x, y, w, h, { xTicks: 3, yValues: decades, yFmt: d3.format('.0e'), xFmt: d3.format('.2s') });
    axisLabels(g, w, h, { x: 'weights the model needs', y: 'distance from real digits' });

    /* the floor, as a band */
    layer.append('rect').attr('x', 0).attr('y', y(data.floor.mmd2_max))
      .attr('width', w).attr('height', Math.max(0, h - y(data.floor.mmd2_max)))
      .attr('fill', 'var(--anchor)').attr('opacity', 0.1);
    layer.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(data.floor.mmd2_max)).attr('y2', y(data.floor.mmd2_max))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y(data.floor.mmd2_max) + 14).style('font-size', '11.5px')
      .attr('fill', 'var(--anchor)').text('real against real: nothing below this');

    [['in a latent', data.ladder, latX, 'var(--primary)'],
      ['in pixels', data.pixel_ladder, (r) => r.params, 'var(--squidink)']]
      .forEach(([label, rows, fx, colour], si) => {
        layer.append('path')
          .attr('d', d3.line().x((r) => x(fx(r))).y((r) => y(Math.abs(r.mmd2)))(rows))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
        layer.selectAll(`circle.s${si}`).data(rows).join('circle').attr('class', `s${si}`)
          .attr('cx', (r) => x(fx(r))).attr('cy', (r) => y(Math.abs(r.mmd2))).attr('r', 4.5)
          .attr('fill', colour);
        const first = rows[0];
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(fx(first))).attr('y', y(Math.abs(first.mmd2)) - 11)
          .attr('text-anchor', 'middle')
          .style('font-size', '11.5px').attr('fill', colour).text(label);
      });

    wrapLabel(title, st.split
      ? 'the score network alone, which is not what a latent model costs to run'
      : 'both curves count everything the model needs to make a picture', w - 8);

    const ratio = bestPix.params / bestLat.params;
    const better = Math.abs(bestPix.mmd2) / Math.abs(bestLat.mmd2);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>where</th><th>score network</th><th>decoder</th>
          <th>total</th><th>distance from real</th>
        </tr>
        ${data.ladder.map((r) => `<tr>
          <td>latent</td>
          <td>${r.diffusion_params.toLocaleString('en-US')}</td>
          <td>${r.decoder_params.toLocaleString('en-US')}</td>
          <td>${r.params.toLocaleString('en-US')}</td>
          <td><span class="value">${r.mmd2}</span></td>
        </tr>`).join('')}
        ${data.pixel_ladder.map((r) => `<tr>
          <td>pixels</td>
          <td>${r.params.toLocaleString('en-US')}</td>
          <td>none</td>
          <td>${r.params.toLocaleString('en-US')}</td>
          <td>${r.mmd2}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        The decoder is counted, because a latent model cannot hand you a picture without it. Even
        so: the best latent row here reaches <span class="value">${bestLat.mmd2}</span> with
        <span class="value">${bestLat.params.toLocaleString('en-US')}</span> weights, and the best
        pixel row reaches ${bestPix.mmd2} with
        ${bestPix.params.toLocaleString('en-US')}. That is
        <span class="bold">${ratio.toFixed(0)} times fewer weights</span> for
        ${better.toFixed(1)} times the score.
        ${Math.abs(data.ladder[data.ladder.length - 1].mmd2) >= Math.abs(bestLat.mmd2)
    ? `And the latent ladder is flat: the widest score network here is no better than the
       narrowest, which is what happens when the thing limiting you is not the diffusion.
       The next section is about what it is instead.`
    : `The latent ladder is still improving at the widest network measured, so on this data the
       diffusion is still the binding constraint.`}
      </div>`;
  }

  render();
  return { render };
}
