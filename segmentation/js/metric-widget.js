/* What a perfect box is worth when the question is which pixels.
 *
 * The box drawn here is not an estimate and not a prediction: it is the exact
 * ground-truth box of the object, the same one the detection article measured
 * against. Read as a mask it is simply wrong about most of the rectangle, and
 * this widget counts by how much, object by object.
 *
 * Both numbers are counted rather than estimated, because the masks are exact:
 * the browser rebuilds them from the owner map the generator shipped, and
 * checks its own pixel counts against the generator's before printing anything.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { loadImage, makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { maskIou, maskDice } from './maskkit.js';

const ZOOM = 6;
const HEAD = 22;
const MASK_RGB = [192, 38, 211];

function stripBytes(img, tile, count) {
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, c.width, c.height).data;
  const out = [];
  for (let t = 0; t < count; t++) {
    const raw = new Uint8Array(tile * tile);
    for (let y = 0; y < tile; y++) {
      for (let x = 0; x < tile; x++) {
        raw[y * tile + x] = px[((y * c.width) + (t * tile + x)) * 4];
      }
    }
    out.push(raw);
  }
  return out;
}

export async function initMetricWidget(data, base) {
  const side = data.meta.canvas;
  const N = side * side;
  const count = data.images.count;
  const M = data.metric;

  const [sprite, ownerImg] = await Promise.all([
    loadImage(`${base}${data.images.sprite}`),
    loadImage(`${base}${data.images.owner}`),
  ]);
  const tiles = stripBytes(sprite, data.images.tile, count);
  const owners = stripBytes(ownerImg, data.images.tile, count);

  /* every object of the twelve canvases, flattened into one list the slider
     walks, so the reader never has to think about which canvas they are on */
  const objects = [];
  data.live_instances.forEach((row, n) => {
    row.objects.forEach((o, k) => {
      objects.push({ canvas: n, index: k, ...o });
    });
  });

  /* ---- the guard: the browser's masks must be the generator's masks ------- */
  let worstCount = 0;
  let worstBox = 0;
  for (const o of objects) {
    const m = maskOf(o.canvas, o.index + 1);
    let n = 0;
    let x0 = side;
    let y0 = side;
    let x1 = 0;
    let y1 = 0;
    for (let p = 0; p < N; p++) {
      if (!m[p]) continue;
      n += 1;
      const x = p % side;
      const y = (p / side) | 0;
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x + 1 > x1) x1 = x + 1;
      if (y + 1 > y1) y1 = y + 1;
    }
    worstCount = Math.max(worstCount, Math.abs(n - o.pixels));
    worstBox = Math.max(worstBox, Math.abs(x0 - o.mask_box[0]), Math.abs(y0 - o.mask_box[1]),
      Math.abs(x1 - o.mask_box[2]), Math.abs(y1 - o.mask_box[3]));
  }
  if (worstCount || worstBox) {
    console.warn(`the browser's masks differ from the generator's by ${worstCount} pixels `
      + `and ${worstBox} of box edge`);
  }

  function maskOf(i, id) {
    const m = new Uint8Array(N);
    for (let p = 0; p < N; p++) if (owners[i][p] === id) m[p] = 1;
    return m;
  }

  function boxMask(box) {
    const m = new Uint8Array(N);
    for (let y = box[1]; y < box[3]; y++) {
      for (let x = box[0]; x < box[2]; x++) m[y * side + x] = 1;
    }
    return m;
  }

  const state = { k: 0, view: 'curve' };
  const CW = side * ZOOM;
  const { ctx } = makeCanvas('#metric-canvas', CW, CW + HEAD);
  const chart = makeChart('#metric-chart', {
    width: 460, height: 360, margin: { top: 30, right: 24, bottom: 56, left: 64 },
  });

  function paint(obj, mask, bmask) {
    ctx.clearRect(0, 0, CW, CW + HEAD);
    const gray = new Float32Array(N);
    for (let p = 0; p < N; p++) gray[p] = tiles[obj.canvas][p] / 255;
    paintGray(ctx, gray, side, side, { zoom: ZOOM, y0: HEAD });

    const img = ctx.createImageData(side, side);
    for (let p = 0, q = 0; p < N; p++, q += 4) {
      const inMask = mask[p];
      const inBox = bmask[p];
      if (inMask && inBox) {
        img.data[q] = MASK_RGB[0];
        img.data[q + 1] = MASK_RGB[1];
        img.data[q + 2] = MASK_RGB[2];
        img.data[q + 3] = 150;
      } else if (inBox) {
        /* inside the box and not the object: this is what the box gets wrong */
        img.data[q] = 255;
        img.data[q + 1] = 255;
        img.data[q + 2] = 255;
        img.data[q + 3] = 60;
      } else {
        img.data[q + 3] = 0;
      }
    }
    const off = document.createElement('canvas');
    off.width = side;
    off.height = side;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, HEAD, CW, CW);

    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(obj.box[0] * ZOOM, HEAD + obj.box[1] * ZOOM,
      (obj.box[2] - obj.box[0]) * ZOOM, (obj.box[3] - obj.box[1]) * ZOOM);
    ctx.restore();

    ctx.save();
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    ctx.fillText(`canvas ${obj.canvas + 1}, ${data.meta.classes[obj.cls]}`, 0, 14);
    ctx.textAlign = 'right';
    ctx.fillText(`${obj.pixels} pixels of object in a ${(obj.box[2] - obj.box[0])
      * (obj.box[3] - obj.box[1])} pixel box`, CW, 14);
    ctx.restore();
  }

  function drawCurve(iou, dice) {
    chart.g.selectAll('*').remove();
    if (state.view === 'curve') {
      const x = d3.scaleLinear().domain([0, 1]).range([0, chart.w]);
      const y = d3.scaleLinear().domain([0, 1]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { xTicks: 5, yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { xTicks: 5, yTicks: 5 });
      axisLabels(chart.g, chart.w, chart.h, { x: 'intersection over union', y: 'Dice' });
      chart.g.append('line')
        .attr('x1', 0).attr('y1', chart.h).attr('x2', chart.w).attr('y2', 0)
        .attr('stroke', 'var(--stone)').attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
      chart.g.append('path').datum(M.curve).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6)
        .attr('d', d3.line().x((r) => x(r.iou)).y((r) => y(r.dice)));
      chart.g.append('circle').attr('cx', x(iou)).attr('cy', y(dice)).attr('r', 6)
        .attr('fill', 'var(--cosmos)').attr('stroke', 'var(--white)').attr('stroke-width', 2);
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .text('Dice is always the kinder of the two');
    } else {
      const rows = M.per_class;
      const x = d3.scaleBand().domain(rows.map((r) => r.cls)).range([0, chart.w]).padding(0.3);
      const y = d3.scaleLinear().domain([0, 1]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      axisLabels(chart.g, chart.w, chart.h, { y: 'IoU of a perfect box, as a mask' });
      const here = objects[state.k];
      chart.g.selectAll('.bar').data(rows).join('rect')
        .attr('class', 'bar')
        .attr('x', (r) => x(r.cls)).attr('width', x.bandwidth())
        .attr('y', (r) => y(r.box_as_mask_iou))
        .attr('height', (r) => chart.h - y(r.box_as_mask_iou))
        .attr('fill', (r) => (r.index === here.cls ? 'var(--cosmos)' : 'var(--primary)'))
        .attr('opacity', (r) => (r.index === here.cls ? 1 : 0.62));
      chart.g.selectAll('.val').data(rows).join('text')
        .attr('class', 'val')
        .attr('x', (r) => x(r.cls) + x.bandwidth() / 2)
        .attr('y', (r) => y(r.box_as_mask_iou) - 6)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
        .attr('fill', 'var(--squidink)')
        .text((r) => r.box_as_mask_iou.toFixed(3));
      chart.g.append('line')
        .attr('x1', 0).attr('x2', chart.w)
        .attr('y1', y(M.box_as_mask_iou_mean)).attr('y2', y(M.box_as_mask_iou_mean))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      /* The mean line used to be labelled beside itself, which put the label
         straight through the value of whichever bar sat nearest the mean. It
         goes in the title, where nothing else is. */
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .text(`dashed: all ${M.n_objects.toLocaleString('en-US')} objects`);
    }
  }

  function render() {
    const obj = objects[state.k];
    const mask = maskOf(obj.canvas, obj.index + 1);
    const bmask = boxMask(obj.box);
    const iou = maskIou(mask, bmask);
    const dice = maskDice(mask, bmask);
    paint(obj, mask, bmask);
    drawCurve(iou, dice);

    document.querySelector('#metric-object-value').textContent =
      `${state.k + 1} / ${objects.length}`;

    const boxPx = (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]);
    const cls = M.per_class[obj.cls];
    document.querySelector('#metric-readout').innerHTML =
      `<span style="display:inline-block;width:11px;height:11px;background:rgb(${MASK_RGB.join(',')});`
      + `border:1px solid var(--squidink);vertical-align:-1px"></span> the object `
      + `&nbsp;&nbsp;<span style="color:#5b6572">pale wash = inside the box and not the object`
      + `&nbsp;&nbsp;white outline = the exact ground-truth box</span>`
      + `<br /><br />`
      + `This box is <span class="bold">perfect</span>: it is the ground truth the detection `
      + `article scored against, not a prediction. Read as a mask it scores IoU `
      + `<span class="bold">${iou.toFixed(3)}</span> and Dice ${dice.toFixed(3)}, because `
      + `${obj.pixels} of its ${boxPx} pixels are the garment and the other `
      + `${boxPx - obj.pixels} are not.`
      + `<br /><br />`
      + `Across all ${M.n_objects.toLocaleString('en-US')} test objects a perfect box averages `
      + `<span class="bold">${M.box_as_mask_iou_mean.toFixed(4)}</span>, and the best single one `
      + `manages ${M.box_as_mask_iou_best.toFixed(4)}. A ${cls.cls} averages `
      + `${cls.box_as_mask_iou.toFixed(3)}: the more a garment differs from a rectangle, the `
      + `less its box is worth. That gap is the whole reason this article exists, and it is not `
      + `something a better detector can close.`
      + `<br /><br />`
      + `<span style="color:#5b6572">${M.identity}. Checked here on every object the widget `
      + `draws, and by the generator on ${M.n_objects.toLocaleString('en-US')} of them, where the `
      + `two sides ${M.identity_max_gap === 0 ? 'came out bit for bit equal every single time'
        : `disagreed by at most ${M.identity_max_gap.toExponential(1)}`}. The masks this `
      + `browser counted match the generator's to `
      + `${worstCount === 0 ? 'the pixel' : `${worstCount} pixels`} and their tight boxes to `
      + `${worstBox === 0 ? 'the edge' : `${worstBox} px`}.</span>`;
  }

  const slider = document.querySelector('#metric-object');
  slider.max = String(objects.length - 1);
  slider.addEventListener('input', () => { state.k = +slider.value; render(); });

  const row = document.querySelector('#metric-view-buttons');
  const views = [['curve', 'how Dice relates to IoU'], ['class', 'by garment shape']];
  const btns = views.map(([key, label]) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${key === state.view ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      state.view = key;
      btns.forEach((n, j) => {
        n.className = `atlas-btn${views[j][0] === key ? '' : ' ghost'}`;
      });
      render();
    });
    row.appendChild(b);
    return b;
  });

  render();
  return { objects: objects.length, worstCount, worstBox };
}
