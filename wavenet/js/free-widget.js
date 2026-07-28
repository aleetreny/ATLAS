/* Trained on the truth, run on itself.
 *
 * Every sample the model produced during training was preceded by real ones.
 * Every sample it produces now is preceded by its own, and after two thousand
 * of them the difference is audible. The three temperatures are the usual knob
 * and they fail in different directions, which is the reason to have all three
 * on one chart rather than one recommended setting.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, play } from '../../assets/js/dsp.js';

export function initFreeWidget(data) {
  const F = data.free;
  const ARMS = [
    ['what it was trained on', 'truth'],
    ['always the likeliest sample', '0'],
    ['sampled, temperature 0.7', '0.7'],
    ['sampled, temperature 1', '1'],
  ];
  let current = 0;
  const controls = d3.select('#free-controls');
  const readout = document.querySelector('#free-readout');
  const chart = makeChart('#free-chart', {
    width: 640, height: 300, margin: { top: 46, right: 26, bottom: 50, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');

  function render() {
    const key = ARMS[current][1];
    const arm = F.arms[key];
    const wave = arm.wave;
    layer.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, wave.length - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([-1.05, 1.05]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layer, x, y, w, h, {
      xTicks: 5, yTicks: 5, xFmt: (i) => `${(1000 * i / data.meta.sr).toFixed(0)}`,
    });
    axisLabels(layer, w, h, {
      x: 'milliseconds', y: 'amplitude', leftBudget: margin.left,
    });
    layer.append('path').datum(F.arms.truth.wave).attr('fill', 'none')
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.9).attr('opacity', 0.35)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    layer.append('path').datum(wave).attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 1.2)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
      .style('font-size', '11px').text('grey: what really came next');
    wrapLabel(title, `${ARMS[current][0]}, ${F.generated.toLocaleString('en-US')} samples after `
      + `${F.primed_with} real ones`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>distance from the real spectrum</th><th>loudness</th>
            <th>zero crossings</th></tr>
        <tr><td>what really came next</td><td>0</td>
            <td>${F.arms.truth.rms}</td><td>${F.arms.truth.zero_crossing_rate}</td></tr>
        ${ARMS.filter((a) => a[1] !== 'truth').map((a) => `<tr>
          <td>${a[0]}</td>
          <td><span class="value">${F.arms[a[1]].spec_distance}</span></td>
          <td>${F.arms[a[1]].rms}</td>
          <td>${F.arms[a[1]].zero_crossing_rate}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        The model was trained to answer "given these true samples, what comes
        next", and it is now being asked "given the samples you just invented,
        what comes next", which is a question nobody trained it on. The three
        arms fail differently: taking the likeliest sample every time
        ${F.arms['0'].rms < F.arms.truth.rms * 0.6
    ? `collapses towards silence (loudness ${F.arms['0'].rms} against
             ${F.arms.truth.rms})`
    : `gives loudness ${F.arms['0'].rms} against ${F.arms.truth.rms}`}, and
        sampling at temperature 1 injects the full noise of the model's own
        uncertainty at every step
        (${F.arms['1'].zero_crossing_rate} zero crossings per sample against
        ${F.arms.truth.zero_crossing_rate}). Press each button and listen to a
        second of it before reading the last column. The distances there are
        between spectra rather than between samples, for the reason the previous
        article measured: two periodic sounds one sample out of phase are far
        apart sample by sample and identical to the ear.
      </div>`;
  }

  controls.selectAll('button.arm').data(ARMS).join('button')
    .attr('class', (d, i) => `atlas-btn arm${i === current ? '' : ' ghost'}`)
    .text((d) => d[0])
    .on('click', (evt, d) => {
      current = ARMS.indexOf(d);
      controls.selectAll('button.arm')
        .attr('class', (dd) => `atlas-btn arm${dd === d ? '' : ' ghost'}`);
      render();
    });
  controls.append('button').attr('class', 'atlas-btn ghost').text('listen')
    .on('click', () => play(decodeInt16(F.arms[ARMS[current][1]].audio), data.meta.sr));

  render();
  return { render };
}
