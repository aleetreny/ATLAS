/* Shared furniture for the widgets on this page: a labelled frame that a
 * renderer can paint into, a drag handler that turns a pointer into a camera,
 * and a contact sheet reader.
 *
 * Two widgets orbit the same scene with different renderers behind them, so
 * the control lives here once rather than twice: nothing that a control has to
 * do is deferred to an animation frame, because a frame is not composed when
 * the browser panel is hidden and the widget would look broken to anything
 * checking it.
 */
import { paintRGB } from '../../assets/js/imagery.js';

/* A canvas that shows one rendered frame, with a caption under it. */
export function makeFrame(parent, caption, { max = 220, cls = '' } = {}) {
  const cell = document.createElement('div');
  cell.className = `frame-cell ${cls}`.trim();
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.maxWidth = `${max}px`;
  canvas.style.height = 'auto';
  canvas.style.display = 'block';
  canvas.style.margin = '0 auto';
  canvas.style.imageRendering = 'pixelated';
  canvas.style.background = 'var(--paper)';
  const cap = document.createElement('div');
  cap.className = 'cap';
  cap.innerHTML = caption;
  cell.appendChild(canvas);
  cell.appendChild(cap);
  parent.appendChild(cell);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return {
    cell,
    canvas,
    ctx,
    /* Paint a res x res float image, scaled up to a fixed canvas size so that
       changing the resolution slider does not change the widget's layout. */
    draw(rgb, res) {
      const zoom = Math.max(1, Math.round(max / res));
      canvas.width = res * zoom;
      canvas.height = res * zoom;
      paintRGB(ctx, rgb, res, res, { zoom });
    },
    blank(res = 48, colour = '#e8ebeb') {
      canvas.width = res;
      canvas.height = res;
      ctx.fillStyle = colour;
      ctx.fillRect(0, 0, res, res);
    },
    cap(html) { cap.innerHTML = html; },
  };
}

/* Drag anywhere on `node` to move a camera around the scene. `state` carries
 * az and el in degrees; `onChange` gets called on every move and once on
 * release, with a flag saying whether the drag is still going, so an expensive
 * renderer can wait for the release and a cheap one need not. */
export function attachOrbit(node, state, onChange, { elMin = -80, elMax = 85 } = {}) {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  const move = (e) => {
    if (!dragging) return;
    const x = e.clientX ?? 0;
    const y = e.clientY ?? 0;
    state.az = (((state.az - (x - lastX) * 0.55) % 360) + 360) % 360;
    state.el = Math.max(elMin, Math.min(elMax, state.el + (y - lastY) * 0.4));
    lastX = x;
    lastY = y;
    onChange(true);
  };
  const up = () => {
    if (!dragging) return;
    dragging = false;
    node.classList.remove('dragging');
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    onChange(false);
  };
  node.classList.add('orbit-pad');
  node.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    node.classList.add('dragging');
    node.setPointerCapture?.(e.pointerId);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    e.preventDefault();
  });
  return { isDragging: () => dragging };
}

/* Buttons that name the view they are showing, which is this atlas's one
 * convention for a toggle: the label is the current state, not the one a
 * click would produce. */
export function buttonRow(container, items, current, onPick) {
  container.innerHTML = '';
  const btns = items.map((it) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn';
    b.type = 'button';
    b.textContent = it.label;
    b.addEventListener('click', () => onPick(it.key));
    container.appendChild(b);
    return { b, key: it.key };
  });
  const paint = (key) => btns.forEach(({ b, key: k }) => {
    b.className = k === key ? 'atlas-btn' : 'atlas-btn ghost';
  });
  paint(current);
  return paint;
}

/* Load a contact sheet PNG and hand back the tiles as float images, so the
 * page can put them anywhere and measure against them. */
export async function loadSheet(src, tile, cols) {
  const img = await new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(`could not load ${src}`));
    im.src = src;
  });
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const rows = Math.round(img.naturalHeight / tile);
  const n = rows * cols;
  const tiles = [];
  for (let k = 0; k < n; k++) {
    const r = Math.floor(k / cols);
    const cc = k % cols;
    const px = ctx.getImageData(cc * tile, r * tile, tile, tile).data;
    const f = new Float32Array(tile * tile * 3);
    for (let i = 0; i < tile * tile; i++) {
      f[i * 3] = px[i * 4] / 255;
      f[i * 3 + 1] = px[i * 4 + 1] / 255;
      f[i * 3 + 2] = px[i * 4 + 2] / 255;
    }
    tiles.push(f);
  }
  return { tiles, rows, cols, tile };
}

/* A row of tiles from a sheet, drawn at an integer zoom with captions. */
export function drawTiles(container, tiles, labels, tile, { max = 132, highlight = -1 } = {}) {
  container.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'frame-row';
  container.appendChild(row);
  tiles.forEach((t, i) => {
    const f = makeFrame(row, labels[i] ?? '', { max });
    f.draw(t, tile);
    if (i === highlight) {
      f.canvas.style.outline = '3px solid var(--primary)';
      f.canvas.style.outlineOffset = '2px';
    }
  });
  return row;
}

/* Amplified absolute difference between two float images, as the same
 * white-to-red map the generator writes into its contact sheets. */
export function errorImage(a, b, gain = 6) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length / 3; i++) {
    let e = 0;
    for (let k = 0; k < 3; k++) e += Math.abs(a[i * 3 + k] - b[i * 3 + k]);
    e = Math.min(1, (e / 3) * gain);
    out[i * 3] = 1;
    out[i * 3 + 1] = 1 - e;
    out[i * 3 + 2] = 1 - e * 0.92;
  }
  return out;
}

/* A direct label for a series or a marker.
 *
 * `annotate` from chart.js is styled for short shouted annotations: 0.8rem,
 * uppercase, two pixels of letter spacing. A four word series name in that
 * style is 165 units wide, which ran four labels off the right of two charts
 * on this page. Same family, smaller and tighter, so several of them fit in a
 * right margin without a legend box. */
export function seriesLabel(g, x, y, text, { colour = 'var(--squidink)', anchor = 'start' } = {}) {
  return g.append('text')
    .attr('class', 'chart-annotation')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', '0.64rem').style('letter-spacing', '0.5px')
    .attr('fill', colour)
    .text(text);
}

export const fmt = {
  db: (v) => `${v.toFixed(2)} dB`,
  /* a raster that finishes in a third of a millisecond should not print as
     "0 ms", which is the same family of bug as a probability printed as 0.000 */
  ms: (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} s`
    : v >= 10 ? `${Math.round(v)} ms` : `${v.toFixed(2)} ms`),
  n0: (v) => Math.round(v).toLocaleString('en-US'),
  pc: (v, d = 1) => `${(v * 100).toFixed(d)}%`,
};
