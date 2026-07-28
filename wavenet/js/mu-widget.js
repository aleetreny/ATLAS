/* The compander, drawn and heard.
 *
 * Eight bits is 256 levels and the question is where to put them. Spread evenly
 * and the quiet parts of the signal get almost none; spread logarithmically and
 * the loud parts give some up. The curve is computed here from the same
 * formula the generator used, and the three clips are the difference.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, play } from '../../assets/js/dsp.js';

export function muEncode(x, mu) {
  const s = Math.sign(x);
  const y = (s * Math.log1p(mu * Math.abs(x))) / Math.log1p(mu);
  return Math.max(0, Math.min(mu, Math.floor(((y + 1) / 2) * mu + 0.5)));
}

export function muDecode(q, mu) {
  const y = 2 * (q / mu) - 1;
  return (Math.sign(y) * Math.expm1(Math.abs(y) * Math.log1p(mu))) / mu;
}

export function initMuWidget(data) {
  const Q = data.quantise;
  const mu = data.meta.mu;
  const CLIPS = [
    ['the original', 'audio_original'],
    ['eight bits, evenly spaced', 'audio_linear'],
    ['eight bits, mu-law', 'audio_mu'],
  ];
  let current = 0;
  const controls = d3.select('#mu-controls');
  const readout = document.querySelector('#mu-readout');
  const chart = makeChart('#mu-chart', {
    width: 640, height: 320, margin: { top: 44, right: 26, bottom: 50, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().domain([-1, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([-1.05, 1.05]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, {
    x: 'the sample', y: 'where its level goes', leftBudget: margin.left,
  });
  g.append('line').attr('x1', x(-1)).attr('y1', y(-1)).attr('x2', x(1)).attr('y2', y(1))
    .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '5 4').attr('opacity', 0.6);
  g.append('text').attr('class', 'chart-note').attr('x', x(0.55)).attr('y', y(0.42))
    .style('font-size', '11px').text('evenly spaced');
  /* recomputed here from the formula, not read from the file */
  const curve = d3.range(-1, 1.001, 0.01);
  g.append('path').datum(curve).attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2.4)
    .attr('d', d3.line().x((d) => x(d))
      .y((d) => y((Math.sign(d) * Math.log1p(mu * Math.abs(d))) / Math.log1p(mu))));
  g.append('text').attr('class', 'chart-note').attr('x', x(0.18)).attr('y', y(0.86))
    .style('font-size', '11px').attr('fill', 'var(--primary)').text('mu-law');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');
  wrapLabel(title, `${data.meta.levels} levels, and two ways to place them`, w - 8);

  function render() {
    const lin = Q.rows.find((r) => r.scheme.startsWith('linear, 8'));
    const law = Q.rows.find((r) => r.scheme.startsWith('mu-law, 8'));
    const linQ = Q.rows.find((r) => r.scheme.includes('quiet') && r.scheme.startsWith('linear'));
    const lawQ = Q.rows.find((r) => r.scheme.includes('quiet') && r.scheme.startsWith('mu-law'));
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>encoding</th><th>signal to noise, whole clip</th>
            <th>signal to noise, the quiet ${(Q.quiet_share * 100).toFixed(0)}%</th></tr>
        <tr><td>evenly spaced</td><td><span class="value">${lin.snr_db} dB</span></td>
            <td>${linQ.snr_db} dB</td></tr>
        <tr><td>mu-law</td><td><span class="value">${law.snr_db} dB</span></td>
            <td>${lawQ.snr_db} dB</td></tr>
      </table>
      <div class="gen-note">
        Two columns, and they are two different questions: the whole clip, and
        the ${(Q.quiet_share * 100).toFixed(0)}% of samples that sit below a
        tenth of full scale. Press the three buttons and listen before deciding
        which one you care about; the paragraph below reads them in that order.
        This is also why the model that comes next predicts a
        <span class="bold">category</span> rather than a number: with 256 levels
        placed like this, choosing one of them is a decision a softmax can make.
      </div>`;
  }

  controls.selectAll('button.clip').data(CLIPS).join('button')
    .attr('class', (d, i) => `atlas-btn clip${i === current ? '' : ' ghost'}`)
    .text((d) => d[0])
    .on('click', (evt, d) => {
      current = CLIPS.indexOf(d);
      controls.selectAll('button.clip')
        .attr('class', (dd) => `atlas-btn clip${dd === d ? '' : ' ghost'}`);
      play(decodeInt16(Q[d[1]]), data.meta.sr);
      render();
    });

  render();
  return { render };
}
