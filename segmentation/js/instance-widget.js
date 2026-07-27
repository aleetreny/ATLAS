/* The question a per-pixel classifier cannot answer.
 *
 * A semantic map says what each pixel is. Ask it how many garments are in the
 * picture and it has no way to reply: two t-shirts that touch are one region of
 * t-shirt, and no amount of training fixes that, because the output has no
 * slot for "which one". This is a property of the representation, so it can be
 * counted on the ground truth without training anything, which is what the
 * census in the readout does.
 *
 * The fix is to go back to the previous article: find the objects first, then
 * ask for a mask inside each one. That is Mask R-CNN, and here it is literally
 * that article's detector with a mask head on the features it already computed.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { loadImage, makeCanvas, paintGray } from '../../assets/js/imagery.js';

const ZOOM = 6;
const HEAD = 22;
const CLASS_RGB = [null, [32, 116, 213], [223, 42, 93], [124, 232, 244], [255, 153, 0], [56, 239, 125]];
/* the instance palette is deliberately unrelated to the class palette: here the
   colour means "a different object", not "a different kind of thing" */
const INST_RGB = [[192, 38, 211], [56, 239, 125], [255, 153, 0], [124, 232, 244],
  [223, 42, 93], [32, 116, 213]];

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

export async function initInstanceWidget(data, base) {
  const side = data.meta.canvas;
  const N = side * side;
  const count = data.images.count;
  const I = data.instances;
  const R = data.mask_rcnn;

  const [sprite, ownerImg] = await Promise.all([
    loadImage(`${base}${data.images.sprite}`),
    loadImage(`${base}${data.images.owner}`),
  ]);
  const tiles = stripBytes(sprite, data.images.tile, count);
  const owners = stripBytes(ownerImg, data.images.tile, count);

  /* ---- the same census the generator ran, recomputed on what is on screen -- */
  function touchingPairs(i) {
    const objs = data.live_instances[i].objects;
    const own = owners[i];
    const pairs = [];
    for (let a = 0; a < objs.length; a++) {
      for (let b = a + 1; b < objs.length; b++) {
        if (objs[a].cls !== objs[b].cls) continue;
        let adj = false;
        for (let p = 0; p < N && !adj; p++) {
          if (own[p] !== a + 1) continue;
          const x = p % side;
          const y = (p / side) | 0;
          if ((x > 0 && own[p - 1] === b + 1) || (x < side - 1 && own[p + 1] === b + 1)
            || (y > 0 && own[p - side] === b + 1) || (y < side - 1 && own[p + side] === b + 1)) {
            adj = true;
          }
        }
        if (adj) pairs.push([a, b]);
      }
    }
    return pairs;
  }

  const liveCensus = { touching: 0, canvases: [] };
  for (let i = 0; i < count; i++) {
    const pairs = touchingPairs(i);
    liveCensus.touching += pairs.length;
    if (pairs.length) liveCensus.canvases.push(i);
  }

  const state = { i: liveCensus.canvases.length ? liveCensus.canvases[0] : 0, view: 'semantic' };
  const CW = side * ZOOM;
  const { ctx } = makeCanvas('#instance-canvas', CW, CW + HEAD);
  const chart = makeChart('#instance-chart', {
    width: 560, height: 300, margin: { top: 34, right: 24, bottom: 56, left: 74 },
  });

  function paint() {
    ctx.clearRect(0, 0, CW, CW + HEAD);
    const gray = new Float32Array(N);
    for (let p = 0; p < N; p++) gray[p] = tiles[state.i][p] / 255;
    paintGray(ctx, gray, side, side, { zoom: ZOOM, y0: HEAD });

    const objs = data.live_instances[state.i].objects;
    const own = owners[state.i];
    const img = ctx.createImageData(side, side);
    for (let p = 0, q = 0; p < N; p++, q += 4) {
      const id = own[p];
      let rgb = null;
      if (id > 0 && id <= objs.length) {
        rgb = state.view === 'semantic'
          ? CLASS_RGB[objs[id - 1].cls + 1]
          : INST_RGB[(id - 1) % INST_RGB.length];
      }
      if (rgb) {
        img.data[q] = rgb[0];
        img.data[q + 1] = rgb[1];
        img.data[q + 2] = rgb[2];
        img.data[q + 3] = 170;
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
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    ctx.fillText(`canvas ${state.i + 1} of ${count}, ${state.view === 'semantic'
      ? 'one colour per kind' : 'one colour per object'}`, 0, 14);
    ctx.textAlign = 'right';
    const pairs = touchingPairs(state.i);
    ctx.fillText(pairs.length
      ? `${pairs.length} same-kind pair${pairs.length === 1 ? '' : 's'} touching`
      : 'nothing of the same kind touches here', CW, 14);
    ctx.restore();
    return pairs;
  }

  function drawChart() {
    chart.g.selectAll('*').remove();
    const rows = R.variants.flatMap((v) => ([
      { k: `${v.variant}\nbox`, v: v.box_ap50, kind: 'box', label: v.variant },
      { k: `${v.variant}\nmask`, v: v.mask_ap50, kind: 'mask', label: v.variant },
    ]));
    const x = d3.scaleBand().domain(rows.map((r) => r.k)).range([0, chart.w]).padding(0.28);
    const hi = Math.max(...rows.map((r) => r.v)) * 1.25;
    const y = d3.scaleLinear().domain([0, hi]).range([chart.h, 0]);
    drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 4 });
    drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 4, xFmt: () => '' });
    axisLabels(chart.g, chart.w, chart.h, { y: 'average precision at 0.5' });
    chart.g.selectAll('.bar').data(rows).join('rect')
      .attr('class', 'bar')
      .attr('x', (r) => x(r.k)).attr('width', x.bandwidth())
      .attr('y', (r) => y(r.v)).attr('height', (r) => chart.h - y(r.v))
      .attr('fill', (r) => (r.kind === 'mask' ? 'var(--primary)' : 'var(--stone)'));
    chart.g.selectAll('.val').data(rows).join('text')
      .attr('class', 'val')
      .attr('x', (r) => x(r.k) + x.bandwidth() / 2).attr('y', (r) => y(r.v) - 6)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
      .attr('fill', 'var(--squidink)')
      .text((r) => r.v.toFixed(3));
    chart.g.selectAll('.lab').data(rows).join('text')
      .attr('class', 'lab')
      .attr('x', (r) => x(r.k) + x.bandwidth() / 2).attr('y', chart.h + 18)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
      .attr('fill', 'var(--squidink)')
      .text((r) => r.kind);
    chart.g.selectAll('.lab2').data(rows).join('text')
      .attr('class', 'lab2')
      .attr('x', (r) => x(r.k) + x.bandwidth() / 2).attr('y', chart.h + 34)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
      .attr('fill', '#5b6572')
      .text((r) => r.label);
  }

  function render() {
    const pairs = paint();
    drawChart();
    document.querySelector('#instance-canvas-value').textContent = `${state.i + 1} / ${count}`;

    const best = R.variants.reduce((a, b) => (b.mask_ap50 > a.mask_ap50 ? b : a));
    const perCls = R.variants.find((v) => v.variant === 'per-class');
    const agn = R.variants.find((v) => v.variant === 'class-agnostic');
    const gap = perCls && agn ? perCls.mask_ap50 - agn.mask_ap50 : 0;

    document.querySelector('#instance-readout').innerHTML =
      (pairs.length
        ? `On this canvas ${pairs.length} pair${pairs.length === 1 ? ' is' : 's are'} the same `
          + `kind of garment AND touching. In the left view they share a colour, so they are one `
          + `region and the map cannot say there are two. In the right view they do not.`
        : `Nothing of the same kind touches on this canvas, so both views agree. Move the slider: `
          + `${liveCensus.canvases.length} of these ${count} canvases have a touching same-kind `
          + `pair on them.`)
      + `<br /><br />`
      + `Over the whole test split, <span class="bold">${I.same_class_pairs_touching}</span> pairs `
      + `of same-class objects touch across ${I.n_test.toLocaleString('en-US')} canvases, and `
      + `<span class="bold">${I.merged_into_one}</span> of them are a single connected region in `
      + `the ground-truth semantic map. Not in a prediction: in the truth. `
      + `${I.merged_into_one > 0
        ? 'A per-pixel classifier that got every pixel exactly right would still report one '
          + 'object there, because the answer it is allowed to give has no room for two.'
        : 'On this dataset the two never quite fuse, so the argument here is structural rather '
          + 'than empirical.'}`
      + `<br /><br />`
      + `So find the objects first. The chart is the previous article's detector, unchanged, with `
      + `a mask head reading the features it had already computed: `
      + `${R.variants[0].bins}×${R.variants[0].bins} bins pooled per proposal, `
      + `${R.variants[0].params.toLocaleString('en-US')} extra parameters. Its boxes score `
      + `${best.box_ap50.toFixed(4)} and its masks <span class="bold">${best.mask_ap50.toFixed(4)}</span>, `
      + `with a mean overlap of ${best.mean_matched_mask_iou.toFixed(4)} on the objects it finds.`
      + `<br /><br />`
      + `<span style="color:#5b6572">Read those two against each other and against nothing else: `
      + `both come from the top ${R.topk} proposals per picture above a floor of `
      + `${R.score_floor} with no suppression, so they are not comparable with the `
      + `${R.detector_map50 ? R.detector_map50.toFixed(4) : 'mAP'} the detection article reported `
      + `for the same detector. The comparison that is clean is mask against box, because the `
      + `detections are identical and only the overlap function changed. `
      + (perCls && agn
        ? `Per class against one shared mask is ${gap >= 0 ? '+' : ''}${gap.toFixed(4)}, at one `
          + `seed, with no noise floor under it. `
        : '')
      + `And the ceiling was declared before the head was trained: ${R.declared}.</span>`;
  }

  const slider = document.querySelector('#instance-canvas-slider');
  slider.max = String(count - 1);
  slider.value = String(state.i);
  slider.addEventListener('input', () => { state.i = +slider.value; render(); });

  const row = document.querySelector('#instance-view-buttons');
  const views = [['semantic', 'one label per pixel'], ['instance', 'one label per object']];
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
  return liveCensus;
}
