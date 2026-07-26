/* Convolution in the browser, written the way the operation is defined:
 * for every output position, multiply a small grid of weights by the pixels
 * under it and add. Slow and obvious on purpose; at these image sizes the
 * honest loop is fast enough, and the whole article is about what this loop
 * does rather than how to make it quick.
 */

/* Cross-correlation, which is what every deep learning library calls
 * "convolution": no kernel flip. Borders are handled by clamping to the
 * nearest edge pixel so the output is the same size as the input. */
export function convolve(gray, w, h, kernel, k) {
  const half = (k - 1) / 2;
  const out = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      let acc = 0;
      for (let i = 0; i < k; i++) {
        const rr = Math.min(h - 1, Math.max(0, r + i - half));
        for (let j = 0; j < k; j++) {
          const cc = Math.min(w - 1, Math.max(0, c + j - half));
          acc += kernel[i * k + j] * gray[rr * w + cc];
        }
      }
      out[r * w + c] = acc;
    }
  }
  return out;
}

/* The same thing with zero padding instead of edge clamping, which is what
 * torch's Conv2d(padding=2) does and therefore what the learned-filter widget
 * must do to land on the reference maps. */
export function convolveZeroPad(gray, w, h, kernel, k, bias = 0) {
  const half = (k - 1) / 2;
  const out = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      let acc = bias;
      for (let i = 0; i < k; i++) {
        const rr = r + i - half;
        if (rr < 0 || rr >= h) continue;
        for (let j = 0; j < k; j++) {
          const cc = c + j - half;
          if (cc < 0 || cc >= w) continue;
          acc += kernel[i * k + j] * gray[rr * w + cc];
        }
      }
      out[r * w + c] = acc;
    }
  }
  return out;
}

export const relu = (arr) => {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] > 0 ? arr[i] : 0;
  return out;
};

export function extent(arr) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < lo) lo = arr[i];
    if (arr[i] > hi) hi = arr[i];
  }
  return [lo, hi];
}

/* Map a signed array into [0,1] for painting, centred on zero so that a
 * filter response of zero is always the same shade of grey. */
export function signedToUnit(arr) {
  const [lo, hi] = extent(arr);
  const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-9);
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = 0.5 + 0.5 * (arr[i] / m);
  return out;
}

/* Positive-only arrays (a ReLU output) stretched to fill the range. */
export function unit(arr) {
  const [lo, hi] = extent(arr);
  const span = hi - lo || 1;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = (arr[i] - lo) / span;
  return out;
}

/* Translate an image, filling the vacated strip with background: the same
 * operation the generator used, so the numbers on the page match. */
export function shiftImage(gray, w, h, dr, dc) {
  const out = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    const sr = r - dr;
    if (sr < 0 || sr >= h) continue;
    for (let c = 0; c < w; c++) {
      const sc = c - dc;
      if (sc < 0 || sc >= w) continue;
      out[r * w + c] = gray[sr * w + sc];
    }
  }
  return out;
}

export const PRESETS = {
  identity: { label: 'nothing', k: [0, 0, 0, 0, 1, 0, 0, 0, 0] },
  edges: { label: 'all edges', k: [0, -1, 0, -1, 4, -1, 0, -1, 0] },
  vertical: { label: 'vertical edges', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  horizontal: { label: 'horizontal edges', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  blur: { label: 'blur', k: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9] },
  sharpen: { label: 'sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
};
