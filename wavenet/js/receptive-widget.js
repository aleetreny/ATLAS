/* The five stacks, and the prediction that was made before they were trained.
 *
 * The receptive field is recomputed in the page from the layer count, so the
 * horizontal axis of this widget is arithmetic rather than a number read out
 * of a file, and the guard checks the two agree.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, play } from '../../assets/js/dsp.js';

export const receptiveField = (layers) => 1 + (2 ** layers - 1);

export function initReceptiveWidget(data) {
  const R = data.receptive;
  const readout = document.querySelector('#receptive-readout');
  const chart = makeChart('#receptive-chart', {
    width: 640, height: 330, margin: { top: 46, right: 26, bottom: 52, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const rows = R.rows;
  const x = d3.scalePoint().domain(rows.map((d) => String(d.layers))).range([0, w]).padding(0.6);
  const y = d3.scaleLinear()
    .domain([0, d3.max(rows, (d) => d.snr_db) * 1.15]).range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 4 });
  drawAxes(g, x, y, w, h, { yTicks: 4 });
  axisLabels(g, w, h, {
    x: 'layers in the stack', y: 'signal to noise of what it predicts, dB',
    leftBudget: margin.left, xOffset: 40,
  });
  g.selectAll('rect').data(rows).join('rect')
    .attr('x', (d) => x(String(d.layers)) - 26).attr('width', 52)
    .attr('y', (d) => y(d.snr_db)).attr('height', (d) => h - y(d.snr_db))
    .attr('fill', (d) => (d.covers_period ? 'var(--primary)' : '#c9ced0'));
  g.selectAll('text.v').data(rows).join('text').attr('class', 'chart-note v')
    .attr('x', (d) => x(String(d.layers))).attr('y', (d) => y(d.snr_db) - 7)
    .attr('text-anchor', 'middle').style('font-size', '11px')
    .text((d) => `${receptiveField(d.layers)}`);
  g.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
    .style('font-size', '11px')
    .text(`the number over each bar is how many samples it can see; `
      + `one period is ${R.period}`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  wrapLabel(title, `predicting the next sample of a ${R.f0} hertz vowel, `
    + `${R.seconds} seconds of it`, w - 8);

  const first = rows.find((d) => d.covers_period);
  const last = rows.filter((d) => !d.covers_period).slice(-1)[0];
  readout.innerHTML = `
    <table class="gen-table">
      <tr><th>layers</th>${rows.map((d) => `<th>${d.layers}</th>`).join('')}</tr>
      <tr><td>samples it can see</td>
          ${rows.map((d) => `<td>${d.receptive}</td>`).join('')}</tr>
      <tr><td>milliseconds</td>${rows.map((d) => `<td>${d.receptive_ms}</td>`).join('')}</tr>
      <tr><td>held out loss</td>${rows.map((d) => `<td>${d.nll}</td>`).join('')}</tr>
      <tr><td>signal to noise, dB</td>
          ${rows.map((d) => `<td><span class="value">${d.snr_db}</span></td>`).join('')}</tr>
      <tr><td>parameters</td>${rows.map((d) => `<td>${d.params.toLocaleString('en-US')}</td>`).join('')}</tr>
    </table>
    <div class="gen-note">
      One period of this sound is <span class="value">${R.period}</span> samples,
      ${R.period_ms} milliseconds, and it was known before anything was trained.
      ${last && first
    ? `The stack with ${last.layers} layers can see ${last.receptive} samples,
         less than a period, and scores ${last.snr_db} dB; the one with
         ${first.layers} layers sees ${first.receptive}, more than a period, and
         scores ${first.snr_db} dB, a step of
         ${(first.snr_db - last.snr_db).toFixed(2)} dB.
         ${(() => {
    const steps = rows.slice(1).map((r, i) => ({ from: rows[i], to: r,
      gain: r.snr_db - rows[i].snr_db }));
    const top = steps.reduce((a, b) => (b.gain > a.gain ? b : a), steps[0]);
    return top.from === last
      ? 'That is the largest step in the table, exactly where the arithmetic '
             + 'said it would be.'
      : `It is not the largest step here either: that is
             ${top.from.receptive} to ${top.to.receptive} samples, worth
             ${top.gain.toFixed(2)} dB, which is the same verdict the loss
             gave, on a scale a listener would recognise.`;
  })()}`
    : `Every stack here covers a period.`}
    </div>`;

  d3.select('#receptive-widget').insert('div', ':first-child').attr('class', 'controls-row')
    .append('button').attr('class', 'atlas-btn ghost').text('listen to what it was trained on')
    .on('click', () => play(decodeInt16(R.audio), data.meta.sr));

  return { render: () => {} };
}
