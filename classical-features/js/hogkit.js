/* HOG in the browser, deliberately identical to scikit-image's.
 *
 * The conventions that matter, all three inherited from skimage so the page
 * can be checked against it at load time:
 *   - gradients are centred differences (row[i+1] - row[i-1]) with the border
 *     rows and columns set to zero rather than reflected or replicated;
 *   - orientations are UNSIGNED, taken mod 180, because a light-to-dark edge
 *     and a dark-to-light edge describe the same piece of structure;
 *   - a cell's histogram bin is the sum of the magnitudes whose orientation
 *     falls in that bin, divided by the cell's area.
 * The generator verified this against skimage.feature.hog to 1.2e-7.
 */

export function gradients(gray, w, h) {
  const gr = new Float32Array(w * h);
  const gc = new Float32Array(w * h);
  for (let r = 1; r < h - 1; r++) {
    for (let c = 0; c < w; c++) gr[r * w + c] = gray[(r + 1) * w + c] - gray[(r - 1) * w + c];
  }
  for (let r = 0; r < h; r++) {
    for (let c = 1; c < w - 1; c++) gc[r * w + c] = gray[r * w + c + 1] - gray[r * w + c - 1];
  }
  const mag = new Float32Array(w * h);
  const ori = new Float32Array(w * h);
  for (let i = 0; i < mag.length; i++) {
    mag[i] = Math.hypot(gr[i], gc[i]);
    const deg = (Math.atan2(gr[i], gc[i]) * 180) / Math.PI;
    ori[i] = ((deg % 180) + 180) % 180;      // JS % keeps the sign; this does not
  }
  return { gr, gc, mag, ori };
}

/* Per-cell orientation histograms: [nRows][nCols][orientations], flat. */
export function cellHistograms(mag, ori, w, h, orientations, cell) {
  const nR = Math.floor(h / cell);
  const nC = Math.floor(w / cell);
  const out = new Float32Array(nR * nC * orientations);
  const width = 180 / orientations;
  for (let r = 0; r < nR; r++) {
    for (let c = 0; c < nC; c++) {
      const base = (r * nC + c) * orientations;
      for (let dy = 0; dy < cell; dy++) {
        const row = (r * cell + dy) * w;
        for (let dx = 0; dx < cell; dx++) {
          const i = row + c * cell + dx;
          const m = mag[i];
          if (m === 0) continue;
          let bin = Math.floor(ori[i] / width);
          if (bin >= orientations) bin = orientations - 1;   // exactly 180 lands home
          out[base + bin] += m;
        }
      }
      for (let k = 0; k < orientations; k++) out[base + k] /= cell * cell;
    }
  }
  return { hist: out, nR, nC, orientations };
}

/* The classic HOG picture: per cell, one line per orientation bin, its length
 * proportional to that bin's weight, drawn perpendicular to the gradient
 * because an edge runs across its own gradient. */
export function drawRose(ctx, cells, cell, zoom, { x0 = 0, y0 = 0, colour = '#f4f6f6', gain = 1 } = {}) {
  const { hist, nR, nC, orientations } = cells;
  let peak = 0;
  for (let i = 0; i < hist.length; i++) if (hist[i] > peak) peak = hist[i];
  if (peak <= 0) return;
  const half = (cell * zoom) / 2;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.strokeStyle = colour;
  for (let r = 0; r < nR; r++) {
    for (let c = 0; c < nC; c++) {
      const cx = x0 + (c * cell + cell / 2) * zoom;
      const cy = y0 + (r * cell + cell / 2) * zoom;
      const base = (r * nC + c) * orientations;
      for (let k = 0; k < orientations; k++) {
        const v = (hist[base + k] / peak) * gain;
        if (v < 0.02) continue;
        /* bin k spans [k, k+1) * 180/orientations degrees of GRADIENT angle;
         * the edge itself is perpendicular to that */
        const mid = ((k + 0.5) * 180) / orientations;
        const a = (mid * Math.PI) / 180 + Math.PI / 2;
        const len = v * half * 0.95;
        ctx.lineWidth = Math.max(0.6, v * 2.2);
        ctx.globalAlpha = Math.min(1, 0.25 + v);
        ctx.beginPath();
        ctx.moveTo(cx - Math.cos(a) * len, cy - Math.sin(a) * len);
        ctx.lineTo(cx + Math.cos(a) * len, cy + Math.sin(a) * len);
        ctx.stroke();
      }
    }
  }
  ctx.restore();
}

/* The descriptor length HOG reports for an image of this size: cells grouped
 * into overlapping 2x2 blocks, every block normalised and concatenated. This
 * is where HOG stops being a compression. */
export function descriptorLength(w, h, orientations, cell, block = 2) {
  const nR = Math.floor(h / cell);
  const nC = Math.floor(w / cell);
  const bR = Math.max(0, nR - block + 1);
  const bC = Math.max(0, nC - block + 1);
  return bR * bC * block * block * orientations;
}

export const maxAbs = (arr) => {
  let m = 0;
  for (let i = 0; i < arr.length; i++) if (Math.abs(arr[i]) > m) m = Math.abs(arr[i]);
  return m;
};
