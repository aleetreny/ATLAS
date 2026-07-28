/* The picture, the posteriors and the boundaries that were written down.
 *
 * The alignment is the part worth staring at: a connectionist temporal
 * classification head is trained without ever being told where the symbols are,
 * and the spikes it produces land near the phones anyway. How near is a number
 * here, in milliseconds, because these utterances were generated rather than
 * annotated.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, play } from '../../assets/js/dsp.js';

export function initAsrWidget(data) {
  const A = data.recognition;
  const D = A.demo;
  const syms = A.symbols;
  let view = 'post';
  const controls = d3.select('#asr-controls');
  const readout = document.querySelector('#asr-readout');
  const chart = makeChart('#asr-chart', {
    width: 640, height: 350, margin: { top: 48, right: 30, bottom: 52, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -28).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');
  const nFrames = D.posteriors.length;
  const secondsPerFrame = D.frame_ms / 1000;

  function drawPosteriors() {
    layer.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, nFrames - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
    drawAxes(layer, x, y, w, h, {
      xTicks: 5, yTicks: 4, xFmt: (i) => (i * secondsPerFrame).toFixed(2),
    });
    axisLabels(layer, w, h, {
      x: 'seconds', y: 'probability', leftBudget: margin.left,
    });
    const palette = ['var(--primary)', 'var(--anchor)', 'var(--aqua)', 'var(--galaxy)',
      'var(--cosmos)', '#7a6b52', '#2f6f5e', '#8a4b2a'];
    /* the blank is the last column and is drawn faintly: it is most of the
       picture and it is not a symbol */
    layer.append('path').datum(D.posteriors.map((p) => p[syms.length]))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('opacity', 0.3)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    const used = [...new Set(D.syms)];
    used.forEach((s) => {
      const k = syms.indexOf(s);
      layer.append('path').datum(D.posteriors.map((p) => p[k]))
        .attr('fill', 'none').attr('stroke', palette[k % palette.length])
        .attr('stroke-width', 1.8)
        .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    });
    D.marks.forEach((m) => {
      const x0 = x(m.start / data.meta.sr / secondsPerFrame);
      layer.append('line').attr('x1', x0).attr('x2', x0).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '3 3').attr('opacity', 0.4);
      layer.append('text').attr('class', 'chart-note')
        .attr('x', x((m.start + m.end) / 2 / data.meta.sr / secondsPerFrame)).attr('y', -8)
        .attr('text-anchor', 'middle').style('font-size', '11px').text(m.sym);
    });
    wrapLabel(title, `what the alignment free head believes, frame by frame`, w - 8);
  }

  function drawFeatures() {
    layer.selectAll('*').remove();
    const rows = D.feat[0].length;
    const cw = w / nFrames;
    const ch = h / rows;
    const flat = D.feat.flat();
    const lo = d3.min(flat);
    const hi = d3.max(flat);
    const cells = [];
    D.feat.forEach((col, ci) => {
      col.forEach((v, ri) => cells.push({ ci, ri, v: (v - lo) / (hi - lo) }));
    });
    layer.selectAll('rect').data(cells).join('rect')
      .attr('x', (d) => d.ci * cw).attr('y', (d) => h - (d.ri + 1) * ch)
      .attr('width', cw + 0.6).attr('height', ch + 0.6)
      .attr('fill', (d) => d3.interpolateRgb('#f1f3f3', '#5c3e29')(d.v));
    const x = d3.scaleLinear().domain([0, nFrames - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([0, rows]).range([h, 0]);
    drawAxes(layer, x, y, w, h, {
      xTicks: 5, yTicks: 4, xFmt: (i) => (i * secondsPerFrame).toFixed(2),
    });
    axisLabels(layer, w, h, {
      x: 'seconds', y: 'mel band', leftBudget: margin.left,
    });
    D.marks.forEach((m) => {
      layer.append('text').attr('class', 'chart-note')
        .attr('x', x((m.start + m.end) / 2 / data.meta.sr / secondsPerFrame)).attr('y', -8)
        .attr('text-anchor', 'middle').style('font-size', '11px').text(m.sym);
    });
    wrapLabel(title, `the only thing either decoder ever sees: ${rows} numbers, `
      + `${A.frames_per_second} times a second`, w - 8);
  }

  function render() {
    if (view === 'post') drawPosteriors(); else drawFeatures();
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>symbol error rate</th><th>what it does when it is wrong</th></tr>
        <tr><td>alignment free head</td>
            <td><span class="value">${(A.ctc.ser * 100).toFixed(1)}%</span></td>
            <td>${A.ctc.errors} errors over ${A.ctc.symbols} symbols, and it cannot emit a symbol
                out of order</td></tr>
        <tr><td>attention decoder</td>
            <td><span class="value">${(A.decoder.ser * 100).toFixed(1)}%</span></td>
            <td>${A.decoder.errors} errors over ${A.decoder.symbols}; ${A.decoder.too_long} of
                ${A.n_test} outputs are longer than the truth and ${A.decoder.runaway} repeat a
                symbol three times in a row</td></tr>
      </table>
      <div class="gen-note">
        The spikes in the first view are the whole trick of the alignment free
        loss: nobody ever told it where the phones were, and it puts a spike near
        each one anyway, because summing over every alignment makes the sharp
        ones the cheap ones. Measured against the boundaries this utterance was
        built from, the spikes land
        <span class="value">${A.alignment.median_ms} ms</span> from the middle of
        their phone in the median, on ${A.alignment.n} phones, against a frame
        step of ${A.alignment.frame_ms} ms: the alignment is good to about a
        frame, and it came free with the transcript.
        ${A.decoder.too_long > 0
    ? ` The decoder has the failure it is famous for: ${A.decoder.too_long} of
             ${A.n_test} test utterances come back longer than the truth, and
             ${A.decoder.runaway} of them repeat a symbol three times running.
             Nothing stops it: it emits an end symbol when it feels like it, and
             when it does not, it keeps going until the cap of ${A.decoder.cap}.`
    : ` On this corpus the decoder always stopped on time, which is the
             behaviour it is not famous for.`}
      </div>`;
  }

  controls.selectAll('button.view').data([['what it believes', 'post'], ['what it sees', 'feat']])
    .join('button')
    .attr('class', (d) => `atlas-btn view${d[1] === view ? '' : ' ghost'}`)
    .text((d) => d[0])
    .on('click', (evt, d) => {
      view = d[1];
      controls.selectAll('button.view')
        .attr('class', (dd) => `atlas-btn view${dd[1] === view ? '' : ' ghost'}`);
      render();
    });
  controls.append('button').attr('class', 'atlas-btn ghost').text('listen')
    .on('click', () => play(decodeInt16(D.audio), data.meta.sr));

  render();
  return { render };
}
