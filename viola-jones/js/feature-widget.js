/* What boosting actually picked.
 *
 * Left: the learned feature drawn over the average face, its rectangles
 * shaded the way the arithmetic treats them. Right: the response of that one
 * feature on 100 faces and 100 background patches, computed live in the
 * browser from the sprite, with the learned threshold as a line. A single
 * stump is not a face detector, and this is what "not a detector but a
 * filter" looks like as a picture.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { loadImage, imageToGray, makeCanvas, canvasLabel } from '../../assets/js/imagery.js';
import { integralImage, featureValue } from './vjkit.js';

const ZOOM = 7;

export async function initFeatureWidget(data, base) {
  const N = data.n;
  const S = data.sprite;
  const meanImg = await loadImage(`${base}${data.mean_face}`);
  const spriteImg = await loadImage(`${base}${S.image}`);
  const { gray: meanGray } = imageToGray(meanImg, N, N);
  const { gray: spriteGray } = imageToGray(spriteImg, S.cols * S.tile, S.rows * S.tile);
  const spriteW = S.cols * S.tile;

  /* every tile's integral image, once */
  const tiles = [];
  for (let i = 0; i < S.labels.length; i++) {
    const tr = Math.floor(i / S.cols) * S.tile;
    const tc = (i % S.cols) * S.tile;
    const g = new Float32Array(N * N);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) g[r * N + c] = spriteGray[(tr + r) * spriteW + (tc + c)];
    }
    tiles.push({ ii: integralImage(g, N, N), label: S.labels[i] });
  }

  const panel = N * ZOOM;
  const { ctx } = makeCanvas('#feat-face', panel + 20, panel + 30);
  const chart = makeChart('#feat-hist', { width: 420, height: 300, margin: { top: 26, right: 20, bottom: 46, left: 52 } });

  const btnRow = document.querySelector('#feat-buttons');
  const readout = document.querySelector('#feat-readout');
  const state = { i: 0 };
  const btns = data.top_features.slice(0, 6).map((f, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = `#${i + 1}`;
    b.addEventListener('click', () => { state.i = i; render(); });
    btnRow.appendChild(b);
    return b;
  });

  function render() {
    const f = data.top_features[state.i];
    ctx.clearRect(0, 0, panel + 20, panel + 30);

    /* the average face, then the feature's rectangles over it */
    const off = document.createElement('canvas');
    off.width = N;
    off.height = N;
    const octx = off.getContext('2d');
    const img = octx.createImageData(N, N);
    for (let i = 0, p = 0; i < N * N; i++, p += 4) {
      const v = Math.round(meanGray[i] * 255);
      img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 10, 24, panel, panel);

    f.rects.forEach((rect, k) => {
      const [r0, c0, r1, c1] = rect;
      ctx.fillStyle = k % 2 === 0 ? 'rgba(185,28,28,0.55)' : 'rgba(255,255,255,0.62)';
      ctx.fillRect(10 + c0 * ZOOM, 24 + r0 * ZOOM, (c1 - c0 + 1) * ZOOM, (r1 - r0 + 1) * ZOOM);
      ctx.strokeStyle = '#232f3e';
      ctx.lineWidth = 1;
      ctx.strokeRect(10 + c0 * ZOOM + 0.5, 24 + r0 * ZOOM + 0.5,
        (c1 - c0 + 1) * ZOOM - 1, (r1 - r0 + 1) * ZOOM - 1);
    });
    /* Short enough to fit the panel: the red-against-white convention is
       spelled out in the readout below, where there is room for it. */
    canvasLabel(ctx, 'the feature on the average face', (panel + 20) / 2, 14, { size: 10 });

    /* the same feature's response on every tile, live */
    const vals = tiles.map((t) => featureValue(t.ii, N + 1, 0, 0, f.rects));
    const faceV = vals.filter((v, i) => tiles[i].label === 1);
    const negV = vals.filter((v, i) => tiles[i].label === 0);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const NB = 26;
    const bin = (v) => Math.max(0, Math.min(NB - 1, Math.floor(((v - lo) / (hi - lo + 1e-9)) * NB)));
    const hFace = new Array(NB).fill(0);
    const hNeg = new Array(NB).fill(0);
    faceV.forEach((v) => { hFace[bin(v)] += 1; });
    negV.forEach((v) => { hNeg[bin(v)] += 1; });

    chart.g.selectAll('*').remove();
    const x = d3.scaleLinear().domain([lo, hi]).range([0, chart.w]);
    const y = d3.scaleLinear().domain([0, Math.max(...hFace, ...hNeg) * 1.1]).range([chart.h, 0]);
    drawGrid(chart.g, x, y, chart.w, chart.h, { xTicks: 5, yTicks: 4 });
    drawAxes(chart.g, x, y, chart.w, chart.h, { xTicks: 5, yTicks: 4 });
    axisLabels(chart.g, chart.w, chart.h, { x: 'feature value', y: 'crops' });
    const bw = chart.w / NB;
    [[hNeg, '#6b7280'], [hFace, '#1e40af']].forEach(([hist, colour]) => {
      chart.g.selectAll(null).data(hist).enter().append('rect')
        .attr('x', (v, i) => i * bw + 1).attr('width', bw - 2)
        .attr('y', (v) => y(v)).attr('height', (v) => chart.h - y(v))
        .attr('fill', colour).attr('opacity', 0.62);
    });
    chart.g.append('line').attr('x1', x(f.threshold)).attr('x2', x(f.threshold))
      .attr('y1', 0).attr('y2', chart.h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    chart.g.append('text').attr('class', 'chart-annotation')
      .attr('x', chart.w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .style('font-size', '0.56rem').style('text-transform', 'none')
      /* the shared annotation style adds 2px between letters, which at this
         length pushes the caption past both edges of its own svg */
      .style('letter-spacing', 'normal')
      .text('faces in blue, background in grey, threshold dashed');

    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));

    /* what this one stump alone would achieve on these 200 crops */
    let ok = 0;
    tiles.forEach((t, i) => {
      const vote = f.polarity === 1 ? vals[i] > f.threshold : vals[i] <= f.threshold;
      if ((vote ? 1 : 0) === t.label) ok += 1;
    });
    readout.innerHTML =
      `<table class="vj-table">
         <tr><td>feature</td><td>${f.type}, ${f.rects.length} rectangles</td>
             <td>weighted error when chosen</td><td><span class="value">${f.weighted_error.toFixed(3)}</span></td>
             <td>vote weight</td><td><span class="value">${f.alpha.toFixed(2)}</span></td></tr>
         <tr><td>alone, on these 200 crops</td><td colspan="5"><span class="value">${(ok / tiles.length * 100).toFixed(1)}%</span> correct</td></tr>
         <tr><td>reading the panel</td><td colspan="5">red is the rectangle the feature subtracts, white the one it adds</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        state.i === 0
          ? `The first feature boosting reached for, out of ${data.pool.sampled.toLocaleString('en-US')} ` +
            `candidates. Look at the histograms: the two clouds overlap. This is not a face detector, it ` +
            `is a filter that is right more often than not, and the entire method is built on the ` +
            `observation that a pile of those, weighted, becomes one.`
          : `Every later feature is chosen to fix what the ones before it got wrong: that is why they look ` +
            `arbitrary on their own and why their individual accuracy stops improving. The weight in the ` +
            `final vote falls with the weighted error, so a feature that barely beats a coin flip barely ` +
            `speaks.`
      }</div>`;
  }

  render();
}
