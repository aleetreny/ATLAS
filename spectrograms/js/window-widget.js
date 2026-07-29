/* The window length, and what it costs, on three sounds whose answer is known.
 *
 * Everything here is computed in the page from the samples the generator sent:
 * the transform, the peak track, and the error against the truth. The truth is
 * available because these sounds were written down rather than recorded, which
 * is the only reason a chart can show what the picture got wrong.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, stft, play } from '../../assets/js/dsp.js';

const WINS = [16, 32, 64, 128, 256, 512, 1024, 2048];

export function initWindowWidget(data) {
  const C = data.chirp;
  const sr = data.meta.sr;
  const chirpX = decodeInt16(C.audio);
  const vibX = decodeInt16(C.vibrato.audio);

  /* the two tones are three lines of arithmetic, so the page makes them rather
     than shipping them */
  const twoTones = (df) => {
    const n = Math.round(0.6 * sr);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = 0.5 * (Math.sin((2 * Math.PI * 1000 * i) / sr)
        + Math.sin((2 * Math.PI * (1000 + df) * i) / sr));
    }
    return out;
  };

  const SOUNDS = [
    {
      key: 'chirp',
      label: 'a straight sweep',
      x: chirpX,
      truth: (t) => C.f_start + ((C.f_end - C.f_start) * t) / C.duration,
      rows: C.rows,
      top: 3600,
    },
    {
      key: 'vibrato',
      label: 'a wobbling tone',
      x: vibX,
      truth: (t) => C.vibrato.centre_hz
        + C.vibrato.depth_hz * Math.sin(2 * Math.PI * C.vibrato.rate_hz * t),
      rows: C.vibrato.rows,
      top: 2000,
    },
    {
      key: 'tones',
      label: 'two tones, 40 Hz apart',
      x: twoTones(40),
      truth: null,
      rows: null,
      top: 1400,
      band: [900, 1120],
    },
  ];
  let current = 0;

  const controls = d3.select('#window-controls');
  const readout = document.querySelector('#window-readout');
  const slider = document.querySelector('#win-slider');
  const valueOut = document.querySelector('#win-value');
  const holder = d3.select('#window-canvas');
  const canvas = holder.append('canvas').attr('class', 'spec-canvas')
    .attr('width', 640).attr('height', 220).node();
  const ctx = canvas.getContext('2d');

  const chart = makeChart('#window-chart', {
    width: 640, height: 260, margin: { top: 40, right: 24, bottom: 48, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');

  function paint(S, win) {
    const hop = Math.max(1, Math.round(win / 8));
    const spec = stft(S.x, { win, hop, sr, nfft: Math.max(1024, win * 2) });
    const cols = spec.frames.length;
    const maxBin = Math.min(spec.freqs.length - 1,
      Math.round((S.top / (sr / 2)) * (spec.freqs.length - 1)));
    canvas.width = Math.max(160, Math.min(640, cols));
    canvas.height = 220;
    const img = ctx.createImageData(canvas.width, canvas.height);
    let peakDb = -Infinity;
    const db = spec.frames.map((fr) => {
      const out = new Float64Array(maxBin + 1);
      for (let k = 0; k <= maxBin; k++) {
        out[k] = 20 * Math.log10(fr[k] + 1e-8);
        if (out[k] > peakDb) peakDb = out[k];
      }
      return out;
    });
    const floorDb = peakDb - 55;
    for (let px = 0; px < canvas.width; px++) {
      const ci = Math.min(cols - 1, Math.floor((px * cols) / canvas.width));
      for (let py = 0; py < canvas.height; py++) {
        const k = Math.round(((canvas.height - 1 - py) / (canvas.height - 1)) * maxBin);
        const v = Math.max(0, Math.min(1, (db[ci][k] - floorDb) / (peakDb - floorDb)));
        const o = 4 * (py * canvas.width + px);
        /* paper to the accent, so a loud bin is the article's own colour */
        img.data[o] = Math.round(241 - v * (241 - 92));
        img.data[o + 1] = Math.round(243 - v * (243 - 14));
        img.data[o + 2] = Math.round(243 - v * (243 - 29));
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { spec, maxBin };
  }

  function render() {
    const S = SOUNDS[current];
    const win = WINS[+slider.value];
    const { spec } = paint(S, win);
    valueOut.textContent = `${win} samples, ${(1000 * win / sr).toFixed(1)} ms`;

    layer.selectAll('*').remove();
    if (S.truth) {
      const x = d3.scaleLinear().domain([0, spec.times[spec.times.length - 1]]).range([0, w]);
      const y = d3.scaleLinear().domain([0, S.top]).range([h, 0]);
      drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
      drawAxes(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
      axisLabels(layer, w, h, {
        x: 'seconds', y: 'frequency, hertz', leftBudget: margin.left,
      });
      const peaks = spec.frames.map((fr) => {
        let bi = 0;
        for (let k = 1; k < fr.length; k++) if (fr[k] > fr[bi]) bi = k;
        return spec.freqs[bi];
      });
      layer.append('path').datum(spec.times.map((t) => S.truth(t)))
        .attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.4)
        .attr('d', d3.line().x((d, i) => x(spec.times[i])).y((d) => y(d)));
      layer.append('path').datum(peaks)
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 1.6)
        .attr('d', d3.line().x((d, i) => x(spec.times[i])).y((d) => y(d)));
      layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
        .style('font-size', '11px').attr('fill', 'var(--anchor)')
        .text('blue: the frequency it really had. accent: the loudest bin');
      let se = 0;
      peaks.forEach((p, i) => { se += (p - S.truth(spec.times[i])) ** 2; });
      const rms = Math.sqrt(se / peaks.length);
      const row = S.rows.find((r) => r.win === win);
      const best = S.rows.reduce((a, b) => (b.rms_hz < a.rms_hz ? b : a), S.rows[0]);
      wrapLabel(title, `${S.label}: this page's tracking error is `
        + `${rms.toFixed(2)} hertz`, w - 8);
      readout.innerHTML = `
        <table class="gen-table">
          <tr><th>window</th>${S.rows.map((r) => `<th>${r.win}</th>`).join('')}</tr>
          <tr><td>milliseconds</td>${S.rows.map((r) => `<td>${r.ms}</td>`).join('')}</tr>
          <tr><td>tracking error, hertz</td>
              ${S.rows.map((r) => `<td${r.win === win ? ' style="font-weight:600"' : ''}>`
    + `${r.rms_hz}</td>`).join('')}</tr>
        </table>
        <div class="gen-note">
          ${row ? `The generator measured ${row.rms_hz} hertz at this window and this page
             computed ${rms.toFixed(2)} from the same samples.` : ''}
          ${S.key === 'chirp'
    ? `Read the row along: on a straight sweep the error just keeps falling as the
             window grows, all the way to ${best.win} samples (${best.rms_hz} hertz),
             with no penalty anywhere for a window a quarter of a second long. That is
             not what the tradeoff promises, and the paragraph below the widget is
             about why. Press the next button first.`
    : `Here the row turns around: down to ${best.rms_hz} hertz at ${best.win}
             samples and worse in both directions from there. The wobble is
             ${data.chirp.vibrato.rate_hz} times a second, so a window longer than
             about ${Math.round(1000 / (2 * data.chirp.vibrato.rate_hz))} milliseconds
             contains a whole cycle of it and averages it away, and a window shorter
             than a few of the tone's own cycles cannot place a frequency at all.
             That U is the tradeoff every audio pipeline is choosing when it picks a
             window length.`}
        </div>`;
    } else {
      const [lo, hi] = S.band;
      const kLo = Math.round((lo / (sr / 2)) * (spec.freqs.length - 1));
      const kHi = Math.round((hi / (sr / 2)) * (spec.freqs.length - 1));
      const avg = new Float64Array(kHi - kLo + 1);
      spec.frames.forEach((fr) => {
        for (let k = kLo; k <= kHi; k++) avg[k - kLo] += fr[k] / spec.frames.length;
      });
      const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(avg) * 1.1]).range([h, 0]);
      drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
      drawAxes(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
      axisLabels(layer, w, h, {
        x: 'frequency, hertz', y: 'average magnitude', leftBudget: margin.left,
      });
      layer.append('path').datum(Array.from(avg)).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2)
        .attr('d', d3.line().x((d, i) => x(spec.freqs[kLo + i])).y((d) => y(d)));
      [1000, 1040].forEach((f) => {
        layer.append('line').attr('x1', x(f)).attr('x2', x(f)).attr('y1', 0).attr('y2', h)
          .attr('stroke', 'var(--anchor)').attr('stroke-dasharray', '4 4').attr('opacity', 0.7);
      });
      layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
        .style('font-size', '11px').attr('fill', 'var(--anchor)')
        .text('dashed: where the two tones actually are');
      let peaks = 0;
      for (let i = 1; i < avg.length - 1; i++) {
        if (avg[i] > avg[i - 1] && avg[i] > avg[i + 1] && avg[i] > 0.25 * d3.max(avg)) peaks += 1;
      }
      const row = data.resolution.rows.find((r) => r.sep === 40);
      wrapLabel(title, `two tones 40 hertz apart: this window shows ${peaks} `
        + `peak${peaks === 1 ? '' : 's'}`, w - 8);
      readout.innerHTML = `
        <div class="gen-note">
          Two sine waves at 1000 and 1040 hertz, added. Below about
          ${row.needed} samples the picture has one peak and the answer "there are
          two tones here" is simply not in it: no amount of contrast or colour
          recovers it, because the transform never resolved them. The arithmetic
          says a window has to be longer than about the reciprocal of the
          separation, ${row.predicted} samples at this sample rate, and the
          measured threshold on this grid of powers of two is ${row.needed},
          a factor of ${(row.needed / row.predicted).toFixed(2)}: a tapered window
          buys freedom from leakage and pays for it by widening the main lobe.
        </div>`;
    }
  }

  controls.selectAll('button.sound').data(SOUNDS).join('button')
    .attr('class', (d, i) => `atlas-btn sound${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = SOUNDS.indexOf(d);
      controls.selectAll('button.sound')
        .attr('class', (dd) => `atlas-btn sound${dd === d ? '' : ' ghost'}`);
      render();
    });
  controls.append('button').attr('class', 'atlas-btn ghost').text('listen')
    .on('click', () => play(SOUNDS[current].x, sr));

  slider.addEventListener('input', render);
  render();
  return { render, paint };
}
