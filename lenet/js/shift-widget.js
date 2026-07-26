/* Drag the digit off centre and watch both networks fall over.
 *
 * The left half shows what the shift does to the picture and to a first-layer
 * feature map, which slides along with it and is the argument for convolution.
 * The right half shows what it does to the accuracy of the two trained
 * networks, which is the argument against believing that sliding is enough.
 */
import { loadImage, imageToGray, makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { convolveZeroPad, relu, unit, shiftImage } from './convkit.js';

const D = 28;
const ZOOM = 6;

export async function initShiftWidget(data, base) {
  const sprite = await loadImage(`${base}${data.digits.image}`);
  const strip = imageToGray(sprite, D * data.digits.count, D);
  const pickDigit = 3;
  const base3 = new Float32Array(D * D);
  for (let r = 0; r < D; r++) {
    for (let c = 0; c < D; c++) base3[r * D + c] = strip.gray[r * strip.w + pickDigit * D + c];
  }

  const { weights, bias, k } = data.conv1;
  const FILTER = 0;
  const flat = new Float32Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) flat[i * k + j] = weights[FILTER][i][j];

  /* ------------------------------------------------------------- pictures */
  const pad = 16;
  const panel = D * ZOOM;
  const cw = pad * 3 + panel * 2;
  const ch = panel + 46;
  const { ctx } = makeCanvas('#shift-pixels', cw, ch);
  const xL = pad;
  const xR = pad * 2 + panel;
  const y0 = 26;

  function paintPair(px) {
    const moved = shiftImage(base3, D, D, px, px);
    const map = relu(convolveZeroPad(moved, D, D, flat, k, bias[FILTER]));
    ctx.clearRect(0, 0, cw, ch);
    paintGray(ctx, moved, D, D, { zoom: ZOOM, x0: xL, y0 });
    paintGray(ctx, unit(map), D, D, { zoom: ZOOM, x0: xR, y0 });
    ctx.save();
    ctx.strokeStyle = 'rgba(35,47,62,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(xL + 0.5, y0 + 0.5, panel - 1, panel - 1);
    ctx.strokeRect(xR + 0.5, y0 + 0.5, panel - 1, panel - 1);
    ctx.restore();
    canvasLabel(ctx, `the digit, moved ${px} px`, xL + panel / 2, y0 - 10);
    canvasLabel(ctx, 'filter 1, moved with it', xR + panel / 2, y0 - 10);
  }

  /* ---------------------------------------------------------------- chart */
  const rows = data.shift;
  const W = 620;
  const H = 340;
  const margin = { top: 26, right: 158, bottom: 52, left: 62 };
  const { g, w, h } = makeChart('#shift-chart', { width: W, height: H, margin });

  const x = d3.scaleLinear().domain([0, d3.max(rows, (d) => d.shift)]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: rows.length, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: rows.length, yFmt: d3.format('.0%') });
  axisLabels(g, w, h, { x: 'the test digits, shifted by this many pixels', y: 'test accuracy' });

  /* Chance is the honest floor to compare against, and both curves go under
     it, which is a fact worth being able to see. */
  g.append('line')
    .attr('x1', 0).attr('x2', w)
    .attr('y1', y(0.1)).attr('y2', y(0.1))
    .attr('stroke', '#8a94a6')
    .attr('stroke-dasharray', '4 4')
    .attr('stroke-width', 1);
  g.append('text')
    .attr('x', 4).attr('y', y(0.1) - 6)
    .attr('class', 'annotation')
    .style('font-size', '11px')
    .style('fill', '#8a94a6')
    .text('guessing, 10%');

  const line = d3.line().x((d) => x(d.shift)).y((d) => y(d.v));
  const series = [
    { key: 'cnn', label: 'convolutional', colour: 'var(--primary)' },
    { key: 'mlp', label: 'fully connected', colour: '#232f3e' },
  ];

  series.forEach((s, si) => {
    const pts = rows.map((r) => ({ shift: r.shift, v: r[s.key] }));
    g.append('path')
      .datum(pts)
      .attr('fill', 'none')
      .attr('stroke', s.colour)
      .attr('stroke-width', 2.5)
      .attr('d', line);
    g.selectAll(`.dot-${s.key}`)
      .data(pts)
      .join('circle')
      .attr('class', `dot-${s.key}`)
      .attr('cx', (d) => x(d.shift))
      .attr('cy', (d) => y(d.v))
      .attr('r', 3)
      .attr('fill', s.colour);
    /* Both curves start within four points of each other, so the legend goes
       on fixed rows rather than next to the first point. */
    g.append('text')
      .attr('x', w + 12)
      .attr('y', 14 + si * 20)
      .attr('class', 'annotation')
      .style('font-size', '12px')
      .style('fill', s.colour)
      .text(s.label);
  });

  const marker = g.append('line')
    .attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--primary)')
    .attr('stroke-width', 1.5)
    .attr('opacity', 0.5);
  const hitsG = g.append('g');

  /* ------------------------------------------------------------- controls */
  const label = document.querySelector('#shift-value');
  const readout = document.querySelector('#shift-readout');
  const slider = document.querySelector('#shift-range');
  slider.max = String(d3.max(rows, (d) => d.shift));

  function update(px) {
    const row = rows.find((r) => r.shift === px);
    label.textContent = px === 0 ? 'not at all' : `${px} px down and right`;
    marker.attr('x1', x(px)).attr('x2', x(px));
    hitsG.selectAll('circle')
      .data([{ v: row.cnn, colour: 'var(--primary)' }, { v: row.mlp, colour: '#232f3e' }])
      .join('circle')
      .attr('cx', x(px))
      .attr('cy', (d) => y(d.v))
      .attr('r', 6)
      .attr('fill', 'none')
      .attr('stroke', (d) => d.colour)
      .attr('stroke-width', 2);
    paintPair(px);

    const ratio = row.mlp > 0 ? row.cnn / row.mlp : Infinity;
    let verdict;
    if (px === 0) {
      verdict = 'Centred digits, the way both networks were trained. The convolutional net is ahead by 3.5 points with 314 fewer parameters.';
    } else if (row.cnn >= 0.1 && row.cnn > row.mlp) {
      verdict = `The convolutional net is still ${ratio.toFixed(1)}× the fully connected one, and the feature map on the left is why: it slid along with the digit instead of landing on different weights.`;
    } else if (row.cnn > row.mlp) {
      verdict = 'The convolutional net has arrived at what guessing would give you, and the fully connected one is below it. Sliding filters bought a few pixels of grace, not immunity, because the classifier reading those maps still reads them position by position.';
    } else {
      verdict = 'Both are now far below the 10% that guessing would give, which means neither is uncertain: they are confidently wrong. The gap between them is real, but at this distance it measures which way each network fails rather than which of them recognises a digit.';
    }
    readout.innerHTML =
      `<span class="bold">Convolutional ${(row.cnn * 100).toFixed(2)}%, fully connected ${(row.mlp * 100).toFixed(2)}%.</span> ${verdict}`;
  }

  slider.addEventListener('input', () => update(+slider.value));
  update(+slider.value);
}
