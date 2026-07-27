/* Point at something. The network segments it, here, now, in your browser.
 *
 * Nothing about this is a recording. The page downloads the folded float16
 * weights, decodes them, builds the prompt channel for wherever you clicked
 * and runs every convolution in JavaScript. The generator computed the same
 * masks in float64 with the same quantised weights, so the readout can say how
 * far apart the two are rather than asking to be trusted.
 *
 * The point of the widget is the second half of the readout: five of the ten
 * garment classes in these pictures were never shown to this network, not once,
 * and it was never told what a class is at all. It was only ever asked for the
 * thing under the point.
 */
import { loadImage, makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { decodeWeights, forward, promptChannel, sigmoid, maskIou, maskDice } from './maskkit.js';

const ZOOM = 6;
const HEAD = 22;
const MASK_RGB = [192, 38, 211];      // the article accent, over black ink
const TRUTH_RGB = [255, 255, 255];

/* The exact integers in a single-channel PNG: the red channel IS the value the
   generator wrote, so no luma round trip can quietly change it. */
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

export async function initPromptWidget(data, net, base) {
  const arch = net.arch;
  const live = net.live;
  const side = arch.canvas;
  const N = side * side;

  const model = {
    layers: net.layers,
    arch,
    weights: decodeWeights(net.weights_b64, net.n_values),
  };

  const [sprite, ownerImg] = await Promise.all([
    loadImage(`${base}${live.sprite}`),
    loadImage(`${base}${data.images.owner}`),
  ]);
  const tiles = stripBytes(sprite, live.tile, live.count);
  const owners = stripBytes(ownerImg, live.tile, live.count);

  /* ---- guard one: the pixels -------------------------------------------- */
  const pc = live.pixel_check;
  let pixelBad = 0;
  for (let x = 0; x < pc.values.length; x++) {
    if (tiles[pc.index][pc.row * live.tile + x] !== pc.values[x]) pixelBad += 1;
  }
  if (pixelBad) console.warn(`the browser read ${pixelBad} of ${pc.values.length} pixels wrong`);

  const cache = new Map();
  function run(i, px, py) {
    const key = `${i}:${px}:${py}`;
    if (cache.has(key)) return cache.get(key);
    const image = new Float32Array(N);
    for (let p = 0; p < N; p++) image[p] = tiles[i][p] / 255;
    const prompt = promptChannel(px, py, side, arch.sigma);
    const head = forward(model, image, prompt);
    const mask = new Uint8Array(N);
    let checksum = 0;
    let count = 0;
    for (let p = 0; p < N; p++) {
      checksum += Math.abs(head.d[p]);
      if (sigmoid(head.d[p]) >= 0.5) { mask[p] = 1; count += 1; }
    }
    const res = { mask, checksum, count };
    cache.set(key, res);
    return res;
  }

  /* ---- guard two: the generator's own prompts ---------------------------- */
  const agreement = { checksum: 0, pixels: 0, iou: 0, done: 0, prompts: 0 };
  function verify(i) {
    for (const ref of live.rows[i].prompts) {
      const r = run(i, ref.x, ref.y);
      agreement.checksum = Math.max(agreement.checksum,
        Math.abs(r.checksum - ref.logit_checksum) / Math.max(ref.logit_checksum, 1));
      agreement.pixels = Math.max(agreement.pixels, Math.abs(r.count - ref.pred_pixels));
      const truth = ownerMask(i, ownerAt(i, ref.x, ref.y));
      agreement.iou = Math.max(agreement.iou, Math.abs(maskIou(r.mask, truth) - ref.iou));
      agreement.prompts += 1;
    }
    agreement.done += 1;
  }

  /* The owner map, shipped as one byte per pixel: 0 is background, 1 to K are
     the garments the detector knows, and 200 and up are the five classes this
     network was never trained on. */
  const ownerAt = (i, x, y) => owners[i][y * side + x];
  function ownerMask(i, id) {
    const m = new Uint8Array(N);
    if (!id) return m;
    for (let p = 0; p < N; p++) if (owners[i][p] === id) m[p] = 1;
    return m;
  }

  const state = {
    i: 0,
    px: 32,
    py: 32,
    showTruth: true,
  };

  const CW = side * ZOOM;
  const { ctx, canvas } = makeCanvas('#prompt-canvas', CW, CW + HEAD);

  /* Start on a point that lands on something, so the first thing the reader
     sees is the widget working rather than an empty canvas. */
  function firstPromptOf(i) {
    const p = live.rows[i].prompts;
    return p.length ? { px: p[0].x, py: p[0].y } : { px: 32, py: 32 };
  }
  Object.assign(state, firstPromptOf(0));

  /* One forward pass is about a sixth of a second in JavaScript, and the
     reference set is a couple of dozen of them, so checking the whole thing on
     load would freeze the page. The first paint needs exactly one pass; the
     rest of the check runs in idle time and re-renders when it lands. */
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
  let next = 0;
  function more() {
    if (next >= live.count) { render(); return; }
    verify(next);
    next += 1;
    render();
    idle(more);
  }
  idle(more);

  function paint(res, truth) {
    ctx.clearRect(0, 0, CW, CW + HEAD);
    const gray = new Float32Array(N);
    for (let p = 0; p < N; p++) gray[p] = tiles[state.i][p] / 255;
    paintGray(ctx, gray, side, side, { zoom: ZOOM, y0: HEAD });

    /* the predicted mask as a wash, so the garment underneath stays visible */
    const img = ctx.createImageData(side, side);
    for (let p = 0, q = 0; p < N; p++, q += 4) {
      if (res.mask[p]) {
        img.data[q] = MASK_RGB[0];
        img.data[q + 1] = MASK_RGB[1];
        img.data[q + 2] = MASK_RGB[2];
        img.data[q + 3] = 130;
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

    if (state.showTruth && truth) {
      /* the true edge as a one-pixel outline, drawn from the shipped owner map */
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.95)';
      for (let y = 0; y < side; y++) {
        for (let x = 0; x < side; x++) {
          if (!truth[y * side + x]) continue;
          const edge = (x === 0 || !truth[y * side + x - 1])
            || (x === side - 1 || !truth[y * side + x + 1])
            || (y === 0 || !truth[(y - 1) * side + x])
            || (y === side - 1 || !truth[(y + 1) * side + x]);
          if (edge) ctx.fillRect(x * ZOOM, HEAD + y * ZOOM, ZOOM, ZOOM);
        }
      }
      ctx.restore();
    }

    /* the prompt itself */
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(state.px * ZOOM + ZOOM / 2, HEAD + state.py * ZOOM + ZOOM / 2, 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = `rgb(${MASK_RGB.join(',')})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(state.px * ZOOM + ZOOM / 2, HEAD + state.py * ZOOM + ZOOM / 2, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    ctx.fillText(`canvas ${state.i + 1} of ${live.count}`, 0, 14);
    ctx.textAlign = 'right';
    ctx.fillText(`${res.count} pixels claimed`, CW, 14);
    ctx.restore();
  }

  function render() {
    const res = run(state.i, state.px, state.py);
    const id = ownerAt(state.i, state.px, state.py);
    const truth = ownerMask(state.i, id);
    const hasTruth = id !== 0;
    paint(res, hasTruth ? truth : null);

    const iou = hasTruth ? maskIou(res.mask, truth) : null;
    const dice = hasTruth ? maskDice(res.mask, truth) : null;
    /* Above 200 the owner map says "a distractor", and the byte counts them
       within the canvas rather than naming a class: which of the five held-out
       kinds it is was never recorded, so the widget does not guess. */
    const held = id >= 200;
    const objs = data.live_instances[state.i].objects;
    const clsName = (!held && id > 0 && id <= objs.length)
      ? data.meta.classes[objs[id - 1].cls]
      : null;

    document.querySelector('#prompt-canvas-value').textContent =
      `${state.i + 1} / ${live.count}`;

    const P = data.promptable;
    const verdict = !hasTruth
      ? `You pointed at empty canvas, so there is nothing to compare against. The network `
        + `still answers, because it always answers: that is what a promptable model does `
        + `with a bad prompt.`
      : held
        ? `That is one of the ${data.dataset.held_out.length} kinds this network was `
          + `<span class="bold">never trained on</span> (${data.dataset.held_out.join(', ')}; `
          + `which of them is not recorded). IoU against its true mask `
          + `<span class="bold">${iou.toFixed(3)}</span>, Dice ${dice.toFixed(3)}.`
        : `That is a <span class="bold">${clsName}</span>, one of the `
          + `${data.meta.classes.length} classes it was trained on. IoU against its true `
          + `mask <span class="bold">${iou.toFixed(3)}</span>, Dice ${dice.toFixed(3)}.`;

    document.querySelector('#prompt-readout').innerHTML =
      `<span style="display:inline-block;width:11px;height:11px;background:rgb(${MASK_RGB.join(',')});`
      + `border:1px solid var(--squidink);vertical-align:-1px"></span> what the network claims `
      + `&nbsp;&nbsp;<span style="color:#5b6572">white outline = the true mask</span>`
      + `<br /><br />${verdict}`
      + `<br /><br />`
      + `Over the whole test split it scores <span class="bold">${P.known.iou.toFixed(4)}</span> `
      + `on the ${data.meta.classes.length} classes it was taught and `
      + `<span class="bold">${P.held_out.iou.toFixed(4)}</span> on the `
      + `${data.dataset.held_out.length} it was not, a gap of ${P.gap.toFixed(4)}. `
      + `${P.above_50_note || ''}`
      + `${(P.held_out.above_50 * 100).toFixed(0)}% of the never-seen objects come back above `
      + `IoU 0.5 against ${(P.known.above_50 * 100).toFixed(0)}% of the familiar ones.`
      + `<br /><br />`
      + `<span style="color:#5b6572">Checked against the generator on ${agreement.done} of `
      + `${live.count} canvases and ${agreement.prompts} reference prompts: `
      + `${pixelBad === 0 ? `all ${pc.values.length} pixels of the reference row read exactly`
        : `${pixelBad} PIXELS WRONG`}, logit checksums within `
      + `${agreement.checksum.toExponential(1)} relative, the pixel count of the mask within `
      + `${agreement.pixels}, and the IoU within ${agreement.iou.toExponential(1)}. `
      + `The weights were quantised to ${net.dtype} before the reference was computed, so what `
      + `is left in those gaps is float32 against float64. Folding batch normalisation away `
      + `cost ${net.errors.fold_abs.toExponential(1)} on the logits.</span>`;
  }

  /* ---- controls ---------------------------------------------------------- */
  canvas.style.cursor = 'crosshair';
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const sx = (ev.clientX - rect.left) / rect.width * CW;
    const sy = (ev.clientY - rect.top) / rect.height * (CW + HEAD) - HEAD;
    if (sy < 0) return;
    state.px = Math.max(0, Math.min(side - 1, Math.floor(sx / ZOOM)));
    state.py = Math.max(0, Math.min(side - 1, Math.floor(sy / ZOOM)));
    render();
  });

  const slider = document.querySelector('#prompt-canvas-slider');
  slider.max = String(live.count - 1);
  slider.addEventListener('input', () => {
    state.i = +slider.value;
    Object.assign(state, firstPromptOf(state.i));
    render();
  });

  const row = document.querySelector('#prompt-truth-buttons');
  const views = [[true, 'with the true edge'], [false, 'just the answer']];
  const btns = views.map(([key, label]) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${key === state.showTruth ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      state.showTruth = key;
      btns.forEach((n, j) => {
        n.className = `atlas-btn${views[j][0] === key ? '' : ' ghost'}`;
      });
      render();
    });
    row.appendChild(b);
    return b;
  });

  render();
  return agreement;
}
