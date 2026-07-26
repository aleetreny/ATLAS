/* The problem, stated as a count: one photograph, ten thousand questions.
 *
 * Steps 0-2 build up the sliding window; steps 3-4 let the trained cascade
 * loose on it, using the same live scan the widget below runs.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { loadImage, imageToGray, makeCanvas, canvasLabel } from '../../assets/js/imagery.js';
import { integralImage, runCascade } from './vjkit.js';

const ZOOM = 3;

export async function initScrollyScan(data, base) {
  const sc = data.scan;
  const S = sc.size;
  const win = sc.window;
  const img = await loadImage(`${base}${sc.image}`);
  const { gray } = imageToGray(img, S, S);

  const W = S * ZOOM + 20;
  const H = S * ZOOM + 40;
  const { ctx } = makeCanvas('#vj-scrolly-chart', W, H);
  const x0 = 10;
  const y0 = 28;

  const ii = integralImage(gray, S, S);
  const stride = S + 1;
  const results = [];
  for (let top = 0; top + win <= S; top++) {
    for (let left = 0; left + win <= S; left++) {
      results.push({ top, left, ...runCascade(ii, stride, top, left, data.learners, data.cascade.stages) });
    }
  }

  const photo = (dim) => {
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, x0, y0, S * ZOOM, S * ZOOM);
    if (dim) {
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fillRect(x0, y0, S * ZOOM, S * ZOOM);
      ctx.restore();
    }
  };

  const box = (top, left, colour, lw = 2) => {
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = lw;
    ctx.strokeRect(x0 + left * ZOOM, y0 + top * ZOOM, win * ZOOM, win * ZOOM);
    ctx.restore();
  };

  const dots = (subset, colour) => {
    ctx.save();
    ctx.fillStyle = colour;
    for (const r of subset) {
      ctx.fillRect(x0 + (r.left + win / 2) * ZOOM - 1, y0 + (r.top + win / 2) * ZOOM - 1, 2, 2);
    }
    ctx.restore();
  };

  const states = {
    0: () => {
      photo(false);
      canvasLabel(ctx, `one photograph, ${S} by ${S} pixels`, W / 2, 18, { size: 11 });
    },
    1: () => {
      photo(false);
      box(20, 46, '#1e40af');
      canvasLabel(ctx, `one ${win}-pixel window: is there a face in it?`, W / 2, 18, { size: 11 });
    },
    2: () => {
      photo(true);
      dots(results, 'rgba(30,64,175,0.35)');
      canvasLabel(ctx, `${results.length.toLocaleString('en-US')} windows, one per position`, W / 2, 18, { size: 11 });
    },
    3: () => {
      photo(true);
      const alive = results.filter((r) => r.stage >= 1);
      dots(alive, 'rgba(30,64,175,0.35)');
      canvasLabel(ctx, `after one stage of ${data.cascade.stages[0].size} features: ` +
        `${alive.length.toLocaleString('en-US')} left`, W / 2, 18, { size: 11 });
    },
    4: () => {
      photo(true);
      const alive = results.filter((r) => r.survived);
      dots(alive, 'rgba(30,64,175,0.5)');
      for (const r of alive) box(r.top, r.left, 'rgba(185,28,28,0.85)', 1.4);
      canvasLabel(ctx, `all ${data.cascade.stages.length} stages: ${alive.length} windows survive`,
        W / 2, 18, { size: 11 });
    },
  };

  scrolly('#vj-scrolly .step', states);
}
