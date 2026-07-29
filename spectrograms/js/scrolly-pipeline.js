/* The four steps between a microphone and a matrix, drawn on real samples.
 *
 * The frame, the taper and the transform are all computed here, so the last
 * step is the same arithmetic the widget below runs and the guard checks
 * against the generator: nothing in this figure is an illustration of what the
 * page does elsewhere, it is the thing itself.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, hann, magSpectrum, stft, play } from '../../assets/js/dsp.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initPipelineScrolly(data) {
  const sr = data.meta.sr;
  const x = decodeInt16(data.inversion.audio_original);
  const marks = data.inversion.marks;
  const WIN = 256;
  const START = Math.round(marks[0].start + 0.25 * (marks[0].end - marks[0].start));

  const chart = makeChart('#pipeline-chart', {
    width: 640, height: 470, margin: { top: 44, right: 22, bottom: 50, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const topH = Math.round(h * 0.42);
  const botY = topH + 72;
  const botH = h - botY;
  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px');
  const subtitle = gBot.append('text').attr('class', 'chart-note')
    .attr('x', 0).attr('y', -14).style('font-size', '11.5px');

  /* the whole utterance, once, with the phone boundaries that were written
     down when it was made */
  const xs = d3.scaleLinear().domain([0, x.length - 1]).range([0, w]);
  const ys = d3.scaleLinear().domain([-1.05, 1.05]).range([topH, 0]);
  drawAxes(gTop, xs, ys, w, topH, {
    xTicks: 5, yTicks: 3, xFmt: (i) => `${(i / sr).toFixed(2)}s`,
  });
  axisLabels(gTop, w, topH, { y: 'amplitude', leftBudget: margin.left });
  const step = Math.max(1, Math.floor(x.length / 1600));
  gTop.append('path').datum(d3.range(0, x.length, step)).attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 0.7)
    .attr('d', d3.line().x((i) => xs(i)).y((i) => ys(x[i])));
  marks.forEach((m) => {
    gTop.append('line').attr('x1', xs(m.start)).attr('x2', xs(m.start))
      .attr('y1', 0).attr('y2', topH).attr('stroke', 'var(--squidink)')
      .attr('opacity', 0.35).attr('stroke-dasharray', '3 3');
    gTop.append('text').attr('class', 'chart-note')
      .attr('x', xs((m.start + m.end) / 2)).attr('y', 12).attr('text-anchor', 'middle')
      .style('font-size', '11px').text(m.sym);
  });
  const frameMark = gTop.append('g');
  const layer = gBot.append('g');

  function drawFrame(tapered) {
    layer.selectAll('*').remove();
    const xf = d3.scaleLinear().domain([0, WIN - 1]).range([0, w]);
    const yf = d3.scaleLinear().domain([-1.05, 1.05]).range([botH, 0]);
    drawGrid(layer, xf, yf, w, botH, { xTicks: 5, yTicks: 3 });
    drawAxes(layer, xf, yf, w, botH, { xTicks: 5, yTicks: 3 });
    axisLabels(layer, w, botH, {
      x: 'sample inside the slice', y: 'amplitude', leftBudget: margin.left, xOffset: 38,
    });
    const seg = Array.from(x.slice(START, START + WIN));
    layer.append('path').datum(seg).attr('fill', 'none')
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('opacity', tapered ? 0.35 : 1)
      .attr('d', d3.line().x((d, i) => xf(i)).y((d) => yf(d)));
    if (tapered) {
      const win = hann(WIN);
      layer.append('path').datum(seg.map((v, i) => v * win[i])).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4)
        .attr('d', d3.line().x((d, i) => xf(i)).y((d) => yf(d)));
      layer.append('path').datum(Array.from(win)).attr('fill', 'none')
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4')
        .attr('d', d3.line().x((d, i) => xf(i)).y((d) => yf(d)));
    }
  }

  function drawSpectrum() {
    layer.selectAll('*').remove();
    const win = hann(WIN);
    const frame = new Float64Array(512);
    for (let i = 0; i < WIN; i++) frame[i] = x[START + i] * win[i];
    const mag = magSpectrum(frame, 512);
    const freqs = d3.range(mag.length).map((k) => (k * sr) / 512);
    const db = Array.from(mag).map((v) => 20 * Math.log10(v + 1e-8));
    const xf = d3.scaleLinear().domain([0, sr / 2]).range([0, w]);
    const yf = d3.scaleLinear().domain([d3.min(db), d3.max(db) + 3]).range([botH, 0]);
    drawGrid(layer, xf, yf, w, botH, { xTicks: 5, yTicks: 4 });
    drawAxes(layer, xf, yf, w, botH, { xTicks: 5, yTicks: 4 });
    axisLabels(layer, w, botH, {
      x: 'frequency, hertz', y: 'decibels', leftBudget: margin.left, xOffset: 38,
    });
    layer.append('path').datum(db).attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 1.4)
      .attr('d', d3.line().x((d, i) => xf(freqs[i])).y((d) => yf(d)));
    const truth = data.meta.vowels[marks[0].sym];
    if (truth) {
      truth.forEach((f, i) => {
        layer.append('line').attr('x1', xf(f)).attr('x2', xf(f)).attr('y1', 0).attr('y2', botH)
          .attr('stroke', 'var(--anchor)').attr('stroke-dasharray', '4 4').attr('opacity', 0.75);
        layer.append('text').attr('class', 'chart-note').attr('x', xf(f) + 4).attr('y', 12)
          .style('font-size', '10.5px').attr('fill', 'var(--anchor)').text(`F${i + 1}`);
      });
    }
  }

  function drawSpectrogram() {
    layer.selectAll('*').remove();
    const spec = stft(x, { win: WIN, hop: 64, sr, nfft: 512 });
    const cols = spec.frames.length;
    const rows = 100;
    const cw = w / cols;
    const ch = botH / rows;
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
      .attr('x', (d) => d.ci * cw).attr('y', (d) => botH - (d.ri + 1) * ch)
      .attr('width', cw + 0.6).attr('height', ch + 0.6)
      .attr('fill', (d) => d3.interpolateRgb('#f1f3f3', '#5c0e1d')(d.v));
    const xf = d3.scaleLinear().domain([0, spec.times[cols - 1]]).range([0, w]);
    const yf = d3.scaleLinear().domain([0, sr / 2]).range([botH, 0]);
    drawAxes(layer, xf, yf, w, botH, { xTicks: 5, yTicks: 4 });
    axisLabels(layer, w, botH, {
      x: 'seconds', y: 'frequency, hertz', leftBudget: margin.left, xOffset: 38,
    });
  }

  const steps = {
    0: () => {
      frameMark.selectAll('*').remove();
      drawFrame(false);
      subtitle.text(`${WIN} of those samples, starting at `
        + `${(START / sr).toFixed(3)} seconds`);
      wrapLabel(title, `${x.length.toLocaleString('en-US')} samples of synthetic speech, `
        + `with the phone boundaries that were written down when it was made`, w - 8);
    },
    1: () => {
      frameMark.selectAll('rect').data([0]).join('rect')
        .attr('x', xs(START)).attr('width', Math.max(2, xs(START + WIN) - xs(START)))
        .attr('y', 0).attr('height', topH)
        .attr('fill', 'var(--anchor)').attr('opacity', 0.18);
      drawFrame(true);
      subtitle.text('grey: the raw slice. blue: the taper. accent: their product');
      wrapLabel(title, 'a slice, multiplied by a window that fades in and out', w - 8);
    },
    2: () => {
      drawSpectrum();
      subtitle.text('one column of the picture, and the resonances this vowel was built with');
      wrapLabel(title, 'the size of every frequency in that slice', w - 8);
    },
    3: () => {
      drawSpectrogram();
      subtitle.text(`${WIN} samples every 64, stacked left to right`);
      wrapLabel(title, 'slide the window along and stand the columns up next to each other',
        w - 8);
    },
  };

  d3.select('#pipeline-chart').append('div').attr('class', 'controls-row')
    .append('button').attr('class', 'atlas-btn ghost').text('listen to it')
    .on('click', () => play(x, sr));

  steps[0]();
  scrolly('#pipeline-scrolly .step', steps);
  return { render: (i) => steps[i]() };
}
