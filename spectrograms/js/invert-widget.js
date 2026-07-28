/* Three reconstructions of the same utterance, and two rulers that disagree.
 *
 * Keeping the phase gives the sound back to the last bit of a double. Throwing
 * it away and guessing gives something that is recognisably the same words and
 * is, by waveform error, further from the original than silence. Both numbers
 * are here because the pair is the lesson: the ruler has to measure what the
 * transform preserved.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, stft, play } from '../../assets/js/dsp.js';

export function initInvertWidget(data) {
  const I = data.inversion;
  const sr = data.meta.sr;
  const CLIPS = [
    { key: 'audio_original', label: 'the original', gap: 0, snr: null },
    { key: 'audio_griffin', label: 'from the picture alone', gap: I.spec_gap, snr: I.griffin_snr_db },
    { key: 'audio_random', label: 'the picture, with random phase', gap: I.spec_gap_random,
      snr: I.random_phase_snr_db },
  ];
  const waves = CLIPS.map((c) => decodeInt16(I[c.key]));
  let current = 0;

  const controls = d3.select('#invert-controls');
  const readout = document.querySelector('#invert-readout');
  const chart = makeChart('#invert-chart', {
    width: 640, height: 320, margin: { top: 44, right: 22, bottom: 48, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');

  function render() {
    const x = waves[current];
    layer.selectAll('*').remove();
    const spec = stft(x, { win: I.win, hop: I.hop, sr, nfft: 512 });
    const cols = spec.frames.length;
    const rows = 90;
    const cw = w / cols;
    const ch = h / rows;
    let peak = -Infinity;
    const db = spec.frames.map((fr) => {
      const out = new Float64Array(rows);
      for (let ri = 0; ri < rows; ri++) {
        const k = Math.round((ri / (rows - 1)) * (fr.length - 1));
        out[ri] = 20 * Math.log10(fr[k] + 1e-8);
        if (out[ri] > peak) peak = out[ri];
      }
      return out;
    });
    const floorDb = peak - 50;
    const cells = [];
    db.forEach((col, ci) => {
      for (let ri = 0; ri < rows; ri++) {
        cells.push({ ci, ri, v: Math.max(0, Math.min(1, (col[ri] - floorDb) / (peak - floorDb))) });
      }
    });
    layer.selectAll('rect').data(cells).join('rect')
      .attr('x', (d) => d.ci * cw).attr('y', (d) => h - (d.ri + 1) * ch)
      .attr('width', cw + 0.6).attr('height', ch + 0.6)
      .attr('fill', (d) => d3.interpolateRgb('#f1f3f3', '#5c0e1d')(d.v));
    const xf = d3.scaleLinear().domain([0, spec.times[cols - 1]]).range([0, w]);
    const yf = d3.scaleLinear().domain([0, sr / 2]).range([h, 0]);
    drawAxes(layer, xf, yf, w, h, { xTicks: 5, yTicks: 4 });
    axisLabels(layer, w, h, {
      x: 'seconds', y: 'frequency, hertz', leftBudget: margin.left,
    });
    I.marks.forEach((m) => {
      layer.append('text').attr('class', 'chart-note')
        .attr('x', xf((m.start + m.end) / (2 * sr))).attr('y', -6)
        .attr('text-anchor', 'middle').style('font-size', '11px').text(m.sym);
    });
    wrapLabel(title, `${CLIPS[current].label}, and the picture it makes`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>distance between the two pictures</th>
            <th>waveform signal to noise</th></tr>
        <tr><td>keeping the phase, inverted exactly</td>
            <td><span class="value">0</span></td>
            <td>the samples come back to ${I.exact_error.toExponential(1)}</td></tr>
        <tr><td>from the magnitudes alone, ${I.iters} iterations</td>
            <td><span class="value">${I.spec_gap}</span></td>
            <td>${I.griffin_snr_db} dB</td></tr>
        <tr><td>the same magnitudes with random phase</td>
            <td><span class="value">${I.spec_gap_random}</span></td>
            <td>${I.random_phase_snr_db} dB</td></tr>
      </table>
      <div class="gen-note">
        Press the three buttons and listen, then read the last column again. By
        waveform error <span class="bold">both</span> reconstructions are worse
        than silence, which would score 0 dB, and the random phase one scores
        ${I.random_phase_snr_db} against ${I.griffin_snr_db} for the iterated
        one, so that ruler says the wash is the better reconstruction. Your ears
        disagree, and so does the ruler that matches what was kept: the iterated
        version lands ${I.spec_gap} away from the original picture and the random
        one ${I.spec_gap_random}, ${(I.spec_gap_random / I.spec_gap).toFixed(1)}
        times further. A spectrogram is not a lossless recording, and a metric
        that measures sample by sample is measuring the part that was thrown away
        on purpose.
      </div>`;
  }

  controls.selectAll('button.clip').data(CLIPS).join('button')
    .attr('class', (d, i) => `atlas-btn clip${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = CLIPS.indexOf(d);
      controls.selectAll('button.clip')
        .attr('class', (dd) => `atlas-btn clip${dd === d ? '' : ' ghost'}`);
      render();
    });
  controls.append('button').attr('class', 'atlas-btn ghost').text('listen')
    .on('click', () => play(waves[current], sr));

  render();
  return { render };
}
