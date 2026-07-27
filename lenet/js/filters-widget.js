/* The six filters the network chose for itself, and what they do to a digit.
 *
 * Nothing here is illustrative: the weights are the trained values exported by
 * the generator, and the maps are computed in this page by the same padded
 * cross-correlation torch runs, then checked against reference maps the
 * generator recorded straight out of the model.
 */
import { loadImage, imageToGray, makeCanvas, paintGray, pixelGrid, canvasLabel } from '../../assets/js/imagery.js';
import { convolveZeroPad, relu, unit, signedToUnit } from './convkit.js';

const D = 28;          // MNIST digits
const IN_ZOOM = 6;
const MAP_ZOOM = 4;
const W_ZOOM = 16;     // the 5x5 weights, blown up until they are readable

export async function initFiltersWidget(data, base) {
  const { weights, bias, n_filters: NF, k } = data.conv1;
  const sprite = await loadImage(`${base}${data.digits.image}`);
  const strip = imageToGray(sprite, D * data.digits.count, D);

  /* Slice the sprite, which lays the digits out side by side, into one array
     per digit. */
  const digits = [];
  for (let d = 0; d < data.digits.count; d++) {
    const g = new Float32Array(D * D);
    for (let r = 0; r < D; r++) {
      for (let c = 0; c < D; c++) g[r * D + c] = strip.gray[r * strip.w + d * D + c];
    }
    digits.push(g);
  }

  /* The digits were written to the PNG as raw MNIST intensities, so the
     browser's grayscale read is already the model's input scale. */
  const mapsFor = (g) => weights.map((w, f) => {
    const flat = new Float32Array(k * k);
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) flat[i * k + j] = w[i][j];
    return relu(convolveZeroPad(g, D, D, flat, k, bias[f]));
  });

  /* ------------------------------------------------- the self-check, first */
  const ref = data.ref_maps;
  const mine = mapsFor(digits[data.digits.labels.indexOf(ref.digit)]);
  let worst = 0;
  for (let f = 0; f < NF; f++) {
    for (let r = 0; r < D; r++) {
      for (let c = 0; c < D; c++) worst = Math.max(worst, Math.abs(mine[f][r * D + c] - ref.maps[f][r][c]));
    }
  }

  const pad = 16;
  const inW = D * IN_ZOOM;
  const mapW = D * MAP_ZOOM;
  const cols = 3;
  const gridW = cols * mapW + (cols - 1) * 12;
  const W = pad * 3 + inW + gridW;
  const H = Math.max(inW + k * W_ZOOM + 74, 2 * mapW + 12 + 46) + 20;
  const { ctx, canvas } = makeCanvas('#filters-chart', W, H);
  const xL = pad;
  const xR = pad * 2 + inW;
  const y0 = 26;
  const yW = y0 + inW + 44;      // where the enlarged weights go

  let digit = 0;
  let picked = 0;

  const INK = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#15803d';

  const mapBox = (f) => ({
    x: xR + (f % cols) * (mapW + 12),
    y: y0 + Math.floor(f / cols) * (mapW + 34),
  });

  function draw() {
    const g = digits[digit];
    const maps = mapsFor(g);
    ctx.clearRect(0, 0, W, H);

    /* the input */
    paintGray(ctx, g, D, D, { zoom: IN_ZOOM, x0: xL, y0 });
    ctx.save();
    ctx.strokeStyle = 'rgba(35,47,62,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(xL + 0.5, y0 + 0.5, inW - 1, inW - 1);
    ctx.restore();
    canvasLabel(ctx, `the digit, ${D}×${D}`, xL + inW / 2, y0 - 9);

    /* the chosen filter's weights, enlarged */
    const w = weights[picked];
    const flat = new Float32Array(k * k);
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) flat[i * k + j] = w[i][j];
    const wUnit = signedToUnit(flat);
    const wx = xL + (inW - k * W_ZOOM) / 2;
    paintGray(ctx, wUnit, k, k, { zoom: W_ZOOM, x0: wx, y0: yW });
    pixelGrid(ctx, k, k, W_ZOOM, { x0: wx, y0: yW, colour: 'rgba(35,47,62,0.4)' });
    canvasLabel(ctx, `filter ${picked + 1}, its ${k}×${k} weights`, xL + inW / 2, yW - 10);

    let wl = Infinity;
    let wh = -Infinity;
    let wsum = 0;
    for (let i = 0; i < flat.length; i++) {
      wsum += flat[i];
      if (flat[i] < wl) wl = flat[i];
      if (flat[i] > wh) wh = flat[i];
    }

    /* the six maps */
    let zeros = 0;
    maps.forEach((m, f) => {
      const { x, y } = mapBox(f);
      paintGray(ctx, unit(m), D, D, { zoom: MAP_ZOOM, x0: x, y0: y });
      ctx.save();
      ctx.strokeStyle = f === picked ? INK : 'rgba(35,47,62,0.3)';
      ctx.lineWidth = f === picked ? 2.5 : 1;
      ctx.strokeRect(x + 0.5, y + 0.5, mapW - 1, mapW - 1);
      ctx.restore();
      canvasLabel(ctx, `filter ${f + 1}`, x + mapW / 2, y + mapW + 16, {
        size: 11,
        colour: f === picked ? INK : '#232f3e',
      });
      if (f === picked) for (let i = 0; i < m.length; i++) if (m[i] === 0) zeros++;
    });

    /* The same reading the kernel editor taught: a sum near zero means flat
       regions cancel, which is what an edge detector does. */
    const kind = Math.abs(wsum) < 0.15
      ? 'near enough to zero that flat regions cancel, which is what makes it an edge detector'
      : 'far enough from zero that it answers to how much ink is under it, not only to how the ink is arranged';

    document.querySelector('#filters-readout').innerHTML =
      `<span class="bold">Filter ${picked + 1}</span> holds ${k * k} weights running from ${wl.toFixed(2)} to ${wh.toFixed(2)}, `
      + `summing to ${wsum.toFixed(2)}, ${kind}. On this digit its map comes out `
      + `${((zeros / (D * D)) * 100).toFixed(1)}% exactly zero: `
      + `that is the ReLU, discarding every place the filter disagreed with what it was looking for. `
      + `All ${NF} maps were computed in your browser just now, and on digit ${ref.digit} they match the values `
      + `exported straight from the trained model to ${worst.toExponential(1)}.`;
  }

  /* ------------------------------------------------------------- controls */
  const row = document.querySelector('#filters-digits');
  data.digits.labels.forEach((lab, i) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${i === 0 ? '' : ' ghost'}`;
    b.textContent = String(lab);
    b.addEventListener('click', () => {
      digit = i;
      row.querySelectorAll('.atlas-btn').forEach((n) => n.classList.add('ghost'));
      b.classList.remove('ghost');
      draw();
    });
    row.appendChild(b);
  });

  const pickRow = document.querySelector('#filters-pick');
  for (let f = 0; f < NF; f++) {
    const b = document.createElement('button');
    b.className = `atlas-btn${f === 0 ? '' : ' ghost'}`;
    b.textContent = `filter ${f + 1}`;
    b.addEventListener('click', () => {
      picked = f;
      pickRow.querySelectorAll('.atlas-btn').forEach((n) => n.classList.add('ghost'));
      b.classList.remove('ghost');
      draw();
    });
    pickRow.appendChild(b);
  }

  /* Clicking a map picks it too, since that is where the eye already is. */
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const scale = W / rect.width;
    const mx = (ev.clientX - rect.left) * scale;
    const my = (ev.clientY - rect.top) * scale;
    for (let f = 0; f < NF; f++) {
      const { x, y } = mapBox(f);
      if (mx >= x && mx <= x + mapW && my >= y && my <= y + mapW) {
        picked = f;
        pickRow.querySelectorAll('.atlas-btn').forEach((n, i) => n.classList.toggle('ghost', i !== f));
        draw();
        return;
      }
    }
  });
  canvas.style.cursor = 'pointer';

  if (worst > 1e-3) console.warn(`conv1 maps disagree with the reference by ${worst}`);
  draw();
}
