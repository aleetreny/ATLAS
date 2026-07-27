/* The U-Net, and the one line of it that matters.
 *
 * An encoder throws away position on purpose: that is what pooling is for, and
 * it is why a classifier works. A segmenter has to give the position back, and
 * the U-Net's answer is to hand the encoder's own feature maps across to the
 * decoder untouched.
 *
 * The ablation behind this widget does not delete those connections, it
 * replaces them with zeros. Same parameters, same multiply-accumulates, same
 * layer shapes, same everything: the only difference is whether the detail can
 * get across. Flip between the two label maps and the difference is visible
 * exactly where the measurement says it is, on the edges.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { loadImage, makeCanvas, paintGray } from '../../assets/js/imagery.js';

const ZOOM = 6;
const HEAD = 22;
/* one colour per garment, chosen bright because they sit on black ink */
const CLASS_RGB = [null, [32, 116, 213], [223, 42, 93], [124, 232, 244], [255, 153, 0], [56, 239, 125]];

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

export async function initUnetWidget(data, base) {
  const side = data.meta.canvas;
  const N = side * side;
  const count = data.images.count;
  const U = data.unet;
  const skipArm = U.arms.find((a) => a.id === 'skips');
  const plainArm = U.arms.find((a) => a.id === 'no-skips');

  const [sprite, truthImg, skipImg, plainImg, ownerImg] = await Promise.all([
    loadImage(`${base}${data.images.sprite}`),
    loadImage(`${base}${data.images.truth}`),
    loadImage(`${base}${data.images.skips}`),
    loadImage(`${base}${data.images.noskips}`),
    loadImage(`${base}${data.images.owner}`),
  ]);
  const tiles = stripBytes(sprite, data.images.tile, count);
  const layers = {
    truth: stripBytes(truthImg, data.images.tile, count),
    skips: stripBytes(skipImg, data.images.tile, count),
    noskips: stripBytes(plainImg, data.images.tile, count),
  };
  const owners = stripBytes(ownerImg, data.images.tile, count);

  /* ---- the guard: rebuild the truth layer from the owner map -------------- */
  let truthGap = 0;
  let outOfRange = 0;
  data.live_instances.forEach((row, i) => {
    const rebuilt = new Uint8Array(N);
    row.objects.forEach((o, k) => {
      for (let p = 0; p < N; p++) if (owners[i][p] === k + 1) rebuilt[p] = o.cls + 1;
    });
    for (let p = 0; p < N; p++) {
      if (rebuilt[p] !== layers.truth[i][p]) truthGap += 1;
      for (const key of ['truth', 'skips', 'noskips']) {
        if (layers[key][i][p] > data.meta.classes.length) outOfRange += 1;
      }
    }
  });
  if (truthGap) {
    console.warn(`the shipped truth map and the owner map disagree on ${truthGap} pixels`);
  }
  if (outOfRange) console.warn(`${outOfRange} label pixels are outside 0..5`);

  const state = { i: 0, layer: 'truth', view: 'class' };
  const CW = side * ZOOM;
  const { ctx } = makeCanvas('#unet-canvas', CW, CW + HEAD);
  const chart = makeChart('#unet-chart', {
    width: 620, height: 360, margin: { top: 34, right: 24, bottom: 62, left: 70 },
  });

  function paint() {
    ctx.clearRect(0, 0, CW, CW + HEAD);
    const gray = new Float32Array(N);
    for (let p = 0; p < N; p++) gray[p] = tiles[state.i][p] / 255;
    paintGray(ctx, gray, side, side, { zoom: ZOOM, y0: HEAD });

    const lab = layers[state.layer][state.i];
    const img = ctx.createImageData(side, side);
    for (let p = 0, q = 0; p < N; p++, q += 4) {
      const c = lab[p];
      if (c > 0 && c < CLASS_RGB.length && CLASS_RGB[c]) {
        img.data[q] = CLASS_RGB[c][0];
        img.data[q + 1] = CLASS_RGB[c][1];
        img.data[q + 2] = CLASS_RGB[c][2];
        img.data[q + 3] = 165;
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

    const wrong = state.layer === 'truth' ? 0 : (() => {
      let n = 0;
      for (let p = 0; p < N; p++) if (lab[p] !== layers.truth[state.i][p]) n += 1;
      return n;
    })();

    ctx.save();
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    const name = { truth: 'the truth', skips: 'with skips', noskips: 'without skips' };
    ctx.fillText(`canvas ${state.i + 1} of ${count}, ${name[state.layer]}`, 0, 14);
    if (state.layer !== 'truth') {
      ctx.textAlign = 'right';
      ctx.fillText(`${wrong} pixels wrong`, CW, 14);
    }
    ctx.restore();
    return wrong;
  }

  function drawChart() {
    chart.g.selectAll('*').remove();
    if (state.view === 'class') {
      const names = skipArm.per_class.map((r) => r.cls);
      const x = d3.scaleBand().domain(names).range([0, chart.w]).padding(0.25);
      const inner = d3.scaleBand().domain(['no-skips', 'skips']).range([0, x.bandwidth()])
        .padding(0.12);
      const y = d3.scaleLinear().domain([0, 1]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      axisLabels(chart.g, chart.w, chart.h, { y: 'intersection over union' });
      [['no-skips', plainArm, 'var(--stone)'], ['skips', skipArm, 'var(--primary)']]
        .forEach(([key, arm, colour]) => {
          chart.g.selectAll(`.b-${key}`).data(arm.per_class).join('rect')
            .attr('class', `b-${key}`)
            .attr('x', (r) => x(r.cls) + inner(key))
            .attr('width', inner.bandwidth())
            .attr('y', (r) => y(r.iou))
            .attr('height', (r) => chart.h - y(r.iou))
            .attr('fill', colour);
        });
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -14).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .text('grey without skips, colour with them');
    } else {
      const rows = [
        { k: 'everywhere', a: plainArm.pixel_acc, b: skipArm.pixel_acc },
        { k: 'on the boundary', a: plainArm.acc_boundary, b: skipArm.acc_boundary },
        { k: 'off it', a: plainArm.acc_interior, b: skipArm.acc_interior },
      ];
      const lo = Math.min(...rows.flatMap((r) => [r.a, r.b])) - 0.03;
      const x = d3.scaleBand().domain(rows.map((r) => r.k)).range([0, chart.w]).padding(0.3);
      const inner = d3.scaleBand().domain(['a', 'b']).range([0, x.bandwidth()]).padding(0.12);
      const y = d3.scaleLinear().domain([Math.max(0, lo), 1]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5, yFmt: d3.format('.2f') });
      axisLabels(chart.g, chart.w, chart.h, { y: 'pixels labelled correctly' });
      rows.forEach((r) => {
        [['a', r.a, 'var(--stone)'], ['b', r.b, 'var(--primary)']].forEach(([k, v, colour]) => {
          chart.g.append('rect')
            .attr('x', x(r.k) + inner(k)).attr('width', inner.bandwidth())
            .attr('y', y(v)).attr('height', chart.h - y(v))
            .attr('fill', colour);
          chart.g.append('text')
            .attr('x', x(r.k) + inner(k) + inner.bandwidth() / 2).attr('y', y(v) - 6)
            .attr('text-anchor', 'middle')
            .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
            .attr('fill', 'var(--squidink)')
            .text(v.toFixed(3));
        });
      });
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -14).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .text(`the boundary is ${(U.band_share * 100).toFixed(1)}% of the pixels`);
    }
  }

  function render() {
    const wrong = paint();
    drawChart();
    document.querySelector('#unet-canvas-value').textContent = `${state.i + 1} / ${count}`;

    const legend = data.meta.classes.map((c, i) =>
      `<span style="display:inline-block;width:11px;height:11px;background:rgb(${CLASS_RGB[i + 1].join(',')});`
      + `border:1px solid var(--squidink);vertical-align:-1px"></span> ${c}`).join(' &nbsp; ');

    const band = U.regions.acc_boundary;
    const inside = U.regions.acc_interior;
    document.querySelector('#unet-readout').innerHTML =
      `${legend}`
      + `<br /><br />`
      + `Both arms are the same network to the last weight: ${U.control}. `
      + `On mean IoU that buys ${U.skip_delta >= 0 ? '+' : ''}${U.skip_delta.toFixed(4)} `
      + `(${skipArm.miou_fg.toFixed(4)} with the skips, ${plainArm.miou_fg.toFixed(4)} without) `
      + `against a seed spread of <span class="bold">${U.seed_spread.toFixed(4)}</span>. `
      + `<span class="bold">${U.skip_resolved
        ? 'That clears the noise floor.'
        : 'The noise is several times the effect, so on that metric this experiment resolves '
          + 'nothing at all.'}</span>`
      + `<br /><br />`
      + `Switch to the second view, where the canvas is split in two and each half gets its own `
      + `floor from the same pair of seeds. On the boundary, `
      + `${(U.band_share * 100).toFixed(1)}% of the pixels, the skips are worth `
      + `${band.delta_mean >= 0 ? '+' : ''}${band.delta_mean.toFixed(4)} against a floor of `
      + `${band.seed_spread.toFixed(4)}, ${band.same_sign ? 'the same sign at both seeds'
        : 'though the two seeds disagree in sign'}: `
      + `<span class="bold">${band.resolved ? 'resolved' : 'not resolved'}</span>. Off it they are `
      + `worth ${inside.delta_mean >= 0 ? '+' : ''}${inside.delta_mean.toFixed(4)} against `
      + `${inside.seed_spread.toFixed(4)}: `
      + `<span class="bold">${inside.resolved ? 'resolved' : 'not resolved'}</span>, and at one `
      + `seed the two arms agree to ${Math.abs(band.delta_seed19 === null ? 0
        : skipArm.acc_interior - plainArm.acc_interior).toFixed(4)}. `
      + `${band.resolved && !inside.resolved
        ? 'Skips are not making this network understand garments better. Away from the edges it '
          + 'is exactly as good without them. They are telling it where a garment stops.'
        : 'The split does not separate the two regions cleanly here either.'}`
      + `<br /><br />`
      + `<span style="color:#5b6572">Neither arm was ever taught the `
      + `${data.dataset.held_out.length} held-out classes, which are in these pictures too. `
      + `The skip arm labels ${(skipArm.distractor_claimed * 100).toFixed(1)}% of their pixels `
      + `as one of the ${data.meta.classes.length} garments it does know, against `
      + `${(plainArm.distractor_claimed * 100).toFixed(1)}% for the other. `
      + `${skipArm.params.toLocaleString('en-US')} parameters and `
      + `${(skipArm.macs / 1e6).toFixed(1)} million multiply-accumulates per canvas, identical `
      + `in both. The truth layer drawn here was rebuilt from the shipped owner map and `
      + `${truthGap === 0 ? 'matches the shipped label map on every pixel'
        : `DISAGREES with the shipped label map on ${truthGap} pixels`}.</span>`;
  }

  const slider = document.querySelector('#unet-canvas-slider');
  slider.max = String(count - 1);
  slider.addEventListener('input', () => { state.i = +slider.value; render(); });

  const layerRow = document.querySelector('#unet-layer-buttons');
  const layerViews = [['truth', 'the truth'], ['skips', 'with skips'], ['noskips', 'without skips']];
  const layerBtns = layerViews.map(([key, label]) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${key === state.layer ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      state.layer = key;
      layerBtns.forEach((n, j) => {
        n.className = `atlas-btn${layerViews[j][0] === key ? '' : ' ghost'}`;
      });
      render();
    });
    layerRow.appendChild(b);
    return b;
  });

  const viewRow = document.querySelector('#unet-view-buttons');
  const views = [['class', 'by garment'], ['band', 'boundary against interior']];
  const viewBtns = views.map(([key, label]) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${key === state.view ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      state.view = key;
      viewBtns.forEach((n, j) => {
        n.className = `atlas-btn${views[j][0] === key ? '' : ' ghost'}`;
      });
      render();
    });
    viewRow.appendChild(b);
    return b;
  });

  render();
  return { truthGap, outOfRange };
}
