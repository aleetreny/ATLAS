/* ATLAS imagery helpers: the canvas side of the house, for module 1.3.
 *
 * Every other article in this atlas draws vectors with d3; a vision article
 * has to draw pixels, and pixels want a canvas. These helpers keep that
 * second rendering path as small and as shared as the chart helpers are:
 * load a PNG, read it as a grayscale Float32Array, paint arrays back out at
 * an integer zoom, and put a crisp grid over the result when the point is
 * that pixels are individually visible.
 *
 * Same-origin note: getImageData taints on file://, so every page that uses
 * this must be served over http (the repo's launch.json does that locally,
 * and GitHub Pages does it in production).
 */

/* Load an <img>, resolved when decoded. */
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${src}`));
    img.src = src;
  });
}

/* Draw an image into an offscreen canvas and read it back as grayscale in
 * [0, 1], optionally resampled to w x h. Rec. 601 luma, the same weights
 * skimage's rgb2gray uses, so the browser and the generator agree. */
export function imageToGray(img, w = img.naturalWidth, h = img.naturalHeight) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (0.2125 * px[p] + 0.7154 * px[p + 1] + 0.0721 * px[p + 2]) / 255;
  }
  return { gray, w, h };
}

/* A canvas sized in CSS pixels but backed at devicePixelRatio, appended to
 * `container`. Returns {canvas, ctx, w, h} in logical pixels. */
export function makeCanvas(container, w, h, { className = 'atlas-canvas' } = {}) {
  const node = typeof container === 'string' ? document.querySelector(container) : container;
  const canvas = document.createElement('canvas');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = '100%';
  canvas.style.maxWidth = `${w}px`;
  canvas.style.height = 'auto';
  canvas.style.display = 'block';
  canvas.style.margin = '0 auto';
  canvas.className = className;
  node.appendChild(canvas);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.scale(dpr, dpr);
  ctx.imageSmoothingEnabled = false;
  return { canvas, ctx, w, h, dpr };
}

/* Paint a Float32Array (values in [0, 1]) at an integer zoom, optionally
 * through a per-pixel colour function returning [r, g, b]. */
export function paintGray(ctx, gray, w, h, { zoom = 1, x0 = 0, y0 = 0, colour = null } = {}) {
  const img = ctx.createImageData(w, h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    if (colour) {
      const [r, g, b] = colour(gray[i], i);
      img.data[p] = r;
      img.data[p + 1] = g;
      img.data[p + 2] = b;
    } else {
      const v = Math.max(0, Math.min(255, Math.round(gray[i] * 255)));
      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
    }
    img.data[p + 3] = 255;
  }
  if (zoom === 1) {
    ctx.putImageData(img, x0, y0);
    return;
  }
  /* putImageData ignores transforms, so go through an offscreen bitmap. */
  const off = document.createElement('canvas');
  off.width = w;
  off.height = h;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x0, y0, w * zoom, h * zoom);
}

/* A hairline grid over a zoomed pixel block: only worth drawing when the
 * zoom is large enough that a grid line is thinner than a pixel cell. */
export function pixelGrid(ctx, w, h, zoom, { x0 = 0, y0 = 0, colour = 'rgba(35,47,62,0.25)' } = {}) {
  if (zoom < 6) return;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= w; i++) {
    ctx.moveTo(x0 + i * zoom + 0.5, y0);
    ctx.lineTo(x0 + i * zoom + 0.5, y0 + h * zoom);
  }
  for (let j = 0; j <= h; j++) {
    ctx.moveTo(x0, y0 + j * zoom + 0.5);
    ctx.lineTo(x0 + w * zoom, y0 + j * zoom + 0.5);
  }
  ctx.stroke();
  ctx.restore();
}

/* Signed values (filter responses, residuals) as blue-white-red, scaled by
 * the largest magnitude present so faint maps stay readable. */
export function divergingColour(maxAbs) {
  const m = maxAbs > 1e-12 ? maxAbs : 1;
  return (v) => {
    const t = Math.max(-1, Math.min(1, v / m));
    if (t >= 0) return [255 - 175 * t, 255 - 220 * t, 255 - 190 * t];   // white -> teal-ink
    return [255 + 32 * t, 255 + 190 * t, 255 + 150 * t];                // white -> crimson
  };
}

/* Label text drawn over a canvas panel, in the site's mono face. */
export function canvasLabel(ctx, text, x, y, { align = 'center', size = 11, colour = '#232f3e' } = {}) {
  ctx.save();
  ctx.font = `${size}px "IBM Plex Mono", monospace`;
  ctx.fillStyle = colour;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(text, x, y);
  ctx.restore();
}
