/* The detector, running in your browser, on pictures it has never been shown
 * a label for.
 *
 * Nothing here is a recording. The page downloads 42,814 folded weights as
 * float16, decodes them, and runs the convolutions, the pooling and the box
 * decode in JavaScript over the pixels it reads out of a PNG. The generator
 * computed the same thing in float64 and shipped its answers, so the widget
 * can say how far apart the two are instead of asking to be trusted: the pixel
 * values it read, the checksum of the raw head, the first candidates in
 * confidence order, and the boxes that survive suppression.
 *
 * The two sliders are the two decisions that are made after the network has
 * finished thinking, and the readout compares whatever the reader chooses
 * against the best that is reachable on the same canvas with the same boxes.
 */
import { loadImage, makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { decodeWeights, forward, decode, iou, nms } from './detkit.js';

const ZOOM = 6;
const HEAD = 22;                       // strip above the picture for the caption
/* These boxes are drawn on black, so the class colours are the bright half of
   the site palette and each label carries whichever ink reads on top of it. */
const CLASS_COLOUR = ['#2074d5', '#df2a5d', '#7ce8f4', '#ff9900', '#38ef7d'];
const CLASS_INK = ['#ffffff', '#ffffff', '#161e2d', '#161e2d', '#161e2d'];
const NMS_STEPS = d3.range(0.05, 1.001, 0.05).map((v) => Math.round(v * 100) / 100);

/* The exact integers in the PNG, not a luma round trip: the generator wrote a
   single-channel image, so the red channel IS the pixel the browser must see
   before anything downstream can be believed. */
function spriteTiles(img, tile, count) {
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

/* Greedy matching at one IoU threshold, in confidence order, each ground-truth
   box consumed once: the same rule the generator's mAP uses. Returns the
   cumulative true positives after each detection, so one pass gives every
   confidence floor at once. */
function cumulativeTp(dets, gt, thr) {
  const used = new Array(gt.length).fill(false);
  const cum = [];
  let tp = 0;
  for (const d of dets) {
    let best = -1;
    let bestIou = thr;
    for (let j = 0; j < gt.length; j++) {
      if (used[j] || gt[j][0] !== d.cls) continue;
      const o = iou(d.box, gt[j].slice(1));
      if (o >= bestIou) { bestIou = o; best = j; }
    }
    if (best >= 0) { used[best] = true; tp += 1; }
    cum.push(tp);
  }
  return cum;
}

const f1 = (tp, kept, nGt) => (kept + nGt > 0 ? (2 * tp) / (kept + nGt) : 0);

export async function initDetectorWidget(det, base) {
  const arch = det.arch;
  const live = det.live;
  const side = arch.grid * arch.stride;

  const model = {
    layers: det.layers,
    arch,
    weights: decodeWeights(det.weights_b64, det.n_values),
  };

  const img = await loadImage(`${base}${live.sprite}`);
  const tiles = spriteTiles(img, live.tile, live.count);

  /* Every confidence the slider can be put at, read off the slider itself, so
     the reference below is a setting the reader can actually reach and no
     position can beat it. */
  const confSliderEl = document.querySelector('#det-conf');
  const CONF_STEPS = [];
  for (let v = +confSliderEl.min; v <= +confSliderEl.max; v += +confSliderEl.step) {
    CONF_STEPS.push(v / 100);
  }

  /* ---- guard one: the pixels ------------------------------------------- */
  const pc = live.pixel_check;
  let pixelBad = 0;
  for (let x = 0; x < pc.values.length; x++) {
    if (tiles[pc.index][pc.row * live.tile + x] !== pc.values[x]) pixelBad += 1;
  }
  if (pixelBad) console.warn(`the browser read ${pixelBad} of ${pc.values.length} pixels wrong`);

  /* ---- one canvas, computed once and kept ------------------------------- */
  const cache = new Map();
  function analyse(i) {
    if (cache.has(i)) return cache.get(i);
    const image = new Float32Array(side * side);
    for (let p = 0; p < image.length; p++) image[p] = tiles[i][p] / 255;
    const head = forward(model, image);
    let checksum = 0;
    for (let p = 0; p < head.d.length; p++) checksum += Math.abs(head.d[p]);
    /* decode() emits anchor-major, then row, then column, which is the slot
       numbering the generator recorded, so the index is the slot. */
    const all = decode(head, arch).map((d, slot) => ({ ...d, slot }));
    const pool = all
      .filter((d) => d.score >= arch.score_floor)
      .sort((a, b) => b.score - a.score)
      .slice(0, arch.pre_nms_topk);
    /* Suppression commutes with a confidence floor: a candidate below the floor
       can never suppress one above it, because greedy NMS walks the list in
       confidence order. So suppression is run once per threshold and the floor
       is applied to the survivors, which is what makes both sliders free. */
    const kept = new Map();
    for (const t of NMS_STEPS) kept.set(t, nms(pool, t));
    const gt = live.rows[i].gt;

    /* The reference the reader is measured against: the best F1 over every
       pair of slider positions on this canvas. Swept from the gentlest
       suppression and the lowest confidence upwards with a strict comparison,
       so where several settings tie the one reported is the least drastic. */
    let best = { f1: -1, t: NMS_STEPS[NMS_STEPS.length - 1], conf: CONF_STEPS[0], tp: 0, kept: 0 };
    for (let k = NMS_STEPS.length - 1; k >= 0; k--) {
      const t = NMS_STEPS[k];
      const list = kept.get(t);
      const cum = cumulativeTp(list, gt, 0.5);
      for (const conf of CONF_STEPS) {
        let n = 0;
        while (n < list.length && list[n].score >= conf) n += 1;
        const v = f1(n ? cum[n - 1] : 0, n, gt.length);
        if (v > best.f1) best = { f1: v, t, conf, tp: n ? cum[n - 1] : 0, kept: n };
      }
    }
    const res = { all, pool, kept, gt, checksum, best };
    cache.set(i, res);
    return res;
  }

  /* ---- guard two: the numbers, on every canvas -------------------------- */
  const agreement = {
    boxes: 0, scores: 0, checksum: 0, slots: 0, nmsSlots: 0, nmsBoxes: 0, done: 0,
  };
  function verify(i) {
    const row = live.rows[i];
    const r = analyse(i);
    agreement.checksum = Math.max(agreement.checksum, Math.abs(r.checksum - row.raw_checksum));
    row.decoded.forEach((ref, j) => {
      const mine = r.pool[j];
      if (!mine || mine.slot !== ref.slot || mine.cls !== ref.cls) { agreement.slots += 1; return; }
      agreement.scores = Math.max(agreement.scores, Math.abs(mine.score - ref.score));
      for (let k = 0; k < 4; k++) {
        agreement.boxes = Math.max(agreement.boxes, Math.abs(mine.box[k] - ref.box[k]));
      }
    });
    /* Suppression is compared as a SET, not as a sequence. The generator runs
       it per class and concatenates the groups, so its list comes back in class
       order; this one walks a single confidence-ordered list. Which boxes
       survive is the claim being checked, and the order they are handed back in
       is not part of it. */
    const mineNms = r.kept.get(arch.nms_threshold) || [];
    const mineSlots = mineNms.map((q) => q.slot).sort((p, q) => p - q).join(',');
    const refSlots = row.nms.map((q) => q.slot).sort((p, q) => p - q).join(',');
    if (mineSlots !== refSlots) {
      agreement.nmsSlots += 1;
      console.warn(`canvas ${i}: suppression kept ${mineNms.length} boxes, `
        + `the generator kept ${row.nms.length}`);
    }
    agreement.nmsBoxes += row.nms.length;
    agreement.done += 1;
  }
  verify(0);

  /* The remaining eleven canvases are a second of arithmetic; they finish the
     check while the reader is still reading the first paragraph. */
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
  let next = 1;
  function more() {
    if (next >= live.count) { render(); return; }
    verify(next);
    next += 1;
    idle(more);
  }
  idle(more);

  /* ---- canvas ---------------------------------------------------------- */
  const CW = side * ZOOM;
  const { ctx } = makeCanvas('#detector-canvas', CW, CW + HEAD);

  /* The sliders in the page are the single source for where the widget starts,
     so the labels rendered in the HTML and the state cannot drift apart. */
  const state = {
    i: +document.querySelector('#det-canvas-slider').value,
    conf: +document.querySelector('#det-conf').value / 100,
    t: +document.querySelector('#det-nms').value / 100,
    grid: false,
  };

  function currentDetections() {
    const r = analyse(state.i);
    const nearest = NMS_STEPS.reduce((a, b) =>
      (Math.abs(b - state.t) < Math.abs(a - state.t) ? b : a));
    return { r, list: r.kept.get(nearest).filter((d) => d.score >= state.conf) };
  }

  function paint(r, list) {
    ctx.clearRect(0, 0, CW, CW + HEAD);
    /* painted exactly as the network reads it, which is how every other page in
       this module paints Fashion-MNIST: ink bright, background black */
    const gray = new Float32Array(side * side);
    for (let p = 0; p < gray.length; p++) gray[p] = tiles[state.i][p] / 255;
    paintGray(ctx, gray, side, side, { zoom: ZOOM, y0: HEAD });

    if (state.grid) {
      ctx.save();
      ctx.strokeStyle = 'rgba(8,145,178,0.45)';
      ctx.lineWidth = 1;
      for (let c = 0; c <= arch.grid; c++) {
        const p = c * arch.stride * ZOOM + 0.5;
        ctx.beginPath();
        ctx.moveTo(p, HEAD); ctx.lineTo(p, HEAD + CW);
        ctx.moveTo(0, HEAD + p); ctx.lineTo(CW, HEAD + p);
        ctx.stroke();
      }
      ctx.restore();
    }

    /* truth first, so a detection sits on top of the box it is trying to be */
    ctx.save();
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.6;
    for (const g of r.gt) {
      ctx.strokeRect(g[1] * ZOOM, HEAD + g[2] * ZOOM, (g[3] - g[1]) * ZOOM, (g[4] - g[2]) * ZOOM);
    }
    ctx.restore();

    ctx.save();
    ctx.lineWidth = 2;
    ctx.font = '11px "IBM Plex Mono", monospace';
    for (const d of [...list].reverse()) {
      const [x1, y1, x2, y2] = d.box;
      ctx.strokeStyle = CLASS_COLOUR[d.cls];
      ctx.strokeRect(x1 * ZOOM, HEAD + y1 * ZOOM, (x2 - x1) * ZOOM, (y2 - y1) * ZOOM);
      if (d.score >= 0.3) {
        const label = `${arch.classes[d.cls]} ${d.score.toFixed(2)}`;
        const wpx = ctx.measureText(label).width + 6;
        /* a box against the right edge would otherwise have its label cut in
           half by the edge of the canvas */
        const lx = Math.max(0, Math.min(x1 * ZOOM, CW - wpx));
        const ly = Math.max(HEAD + 11, HEAD + y1 * ZOOM - 2);
        ctx.fillStyle = CLASS_COLOUR[d.cls];
        ctx.fillRect(lx, ly - 11, wpx, 13);
        ctx.fillStyle = CLASS_INK[d.cls];
        ctx.fillText(label, lx + 3, ly - 1);
      }
    }
    ctx.restore();

    ctx.save();
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    ctx.fillText(`canvas ${state.i + 1} of ${live.count}`, 0, 14);
    ctx.textAlign = 'right';
    ctx.fillText(`${list.length} box${list.length === 1 ? '' : 'es'} kept of ${r.all.length} the grid proposed`, CW, 14);
    ctx.restore();
  }

  /* ---- readout --------------------------------------------------------- */
  function render() {
    const { r, list } = currentDetections();
    paint(r, list);

    const cum = cumulativeTp(list, r.gt, 0.5);
    const tp = cum.length ? cum[cum.length - 1] : 0;
    const mine = f1(tp, list.length, r.gt.length);
    const gap = r.best.f1 - mine;
    const atBest = gap < 1e-9;

    /* The box colours are chosen to read on black, so on this white panel each
       swatch needs an outline of its own or the pale ones vanish. */
    const legend = arch.classes.map((c, i) =>
      `<span style="display:inline-block;width:11px;height:11px;background:${CLASS_COLOUR[i]};`
      + `border:1px solid var(--squidink);vertical-align:-1px"></span> ${c}`).join(' &nbsp; ');

    document.querySelector('#det-canvas-value').textContent = `${state.i + 1} / ${live.count}`;
    document.querySelector('#det-conf-value').textContent = state.conf.toFixed(2);
    document.querySelector('#det-nms-value').textContent =
      state.t >= 1 ? 'off' : state.t.toFixed(2);

    const verdict = atBest
      ? `<span class="bold">That is the best this canvas can do</span> with these boxes: no `
        + `confidence floor and no suppression threshold finds more of the objects without also `
        + `keeping more boxes that are not objects.`
      : `The best reachable here is <span class="bold">${r.best.f1.toFixed(3)}</span>, at a `
        + `confidence of ${r.best.conf.toFixed(2)} with suppression at `
        + `${r.best.t >= 1 ? 'off' : r.best.t.toFixed(2)} (${r.best.tp} of ${r.gt.length} found from `
        + `${r.best.kept} boxes). You are ${gap.toFixed(3)} below it.`;

    document.querySelector('#detector-readout').innerHTML =
      `${legend} &nbsp;&nbsp;<span style="color:#5b6572">dashed = the truth</span>`
      + `<br /><br />`
      + `The head proposes <span class="bold">${r.all.length}</span> boxes for this picture, three at `
      + `each of the ${arch.grid}×${arch.grid} cells, whatever is in it. `
      + `${r.pool.length} of them clear the ${arch.score_floor} floor the detector was measured with; `
      + `suppression at ${state.t >= 1 ? 'nothing' : state.t.toFixed(2)} and your floor of `
      + `${state.conf.toFixed(2)} leave <span class="bold">${list.length}</span>. `
      + `Of the ${r.gt.length} object${r.gt.length === 1 ? '' : 's'} really there, `
      + `<span class="bold">${tp}</span> ${tp === 1 ? 'is' : 'are'} found at IoU 0.5, which is an F1 of `
      + `<span class="bold">${mine.toFixed(3)}</span>. ${verdict}`
      + `<br /><br />`
      + `<span style="color:#5b6572">Checked against the generator on `
      + `${agreement.done} of ${live.count} canvases: `
      + `${pixelBad === 0 ? `all ${pc.values.length} pixels of the reference row read exactly` : `${pixelBad} PIXELS WRONG`}, `
      + `raw head checksums within ${agreement.checksum.toExponential(1)} of the generator's `
      + `(they are around ${Math.round(live.rows[0].raw_checksum).toLocaleString('en-US')}), every `
      + `shipped candidate in the same rank order (${agreement.slots} mismatches), confidences `
      + `within ${agreement.scores.toExponential(1)}, corners within `
      + `${agreement.boxes.toExponential(1)} px, and suppression at ${arch.nms_threshold} keeping `
      + `exactly the same ${agreement.nmsBoxes} boxes across the ${agreement.done} canvases `
      + `(${agreement.nmsSlots} canvases where the surviving set differs). `
      + `The weights were quantised to ${det.dtype} before the reference was computed, so what is `
      + `left in those gaps is float32 against float64 and nothing else. Quantising them cost `
      + `${det.errors.map50_delta_from_quantisation >= 0 ? '+' : ''}`
      + `${det.errors.map50_delta_from_quantisation.toFixed(5)} mAP@0.5 on the whole test split.</span>`;
  }

  /* ---- controls -------------------------------------------------------- */
  const canvasSlider = document.querySelector('#det-canvas-slider');
  canvasSlider.max = String(live.count - 1);
  canvasSlider.addEventListener('input', () => { state.i = +canvasSlider.value; render(); });

  const confSlider = document.querySelector('#det-conf');
  confSlider.addEventListener('input', () => { state.conf = +confSlider.value / 100; render(); });

  const nmsSlider = document.querySelector('#det-nms');
  nmsSlider.addEventListener('input', () => { state.t = +nmsSlider.value / 100; render(); });

  const gridRow = document.querySelector('#det-grid-buttons');
  const gridViews = [[false, 'just the picture'], [true, 'with the grid on it']];
  const gridBtns = gridViews.map(([key, label]) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${key === state.grid ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      state.grid = key;
      gridBtns.forEach((n, j) => {
        n.className = `atlas-btn${gridViews[j][0] === key ? '' : ' ghost'}`;
      });
      render();
    });
    gridRow.appendChild(b);
    return b;
  });

  render();
  return agreement;
}
