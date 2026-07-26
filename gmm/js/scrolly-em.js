/* EM, told in five steps on the same 300 blobs the k-means article used.
 *
 * The trajectory is computed once from a seeded start whose first E-step is
 * visibly soft (a deliberately bad start would repeat the k-means scrolly;
 * what this one must show is softness, so the start places all three
 * components near the middle where every point is genuinely torn). Handlers
 * are idempotent so scrolling back replays cleanly.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { em, initFrom, eStep } from './gmmkit.js';
import { drawSoftPoints, drawMeans, drawEllipses } from './mixdraw.js';

export function initScrollyEm(data) {
  const pts = data.blobs.points;
  const { g, w, h } = makeChart('#em-chart', {
    width: 620,
    height: 560,
    margin: { top: 34, right: 26, bottom: 56, left: 60 },
  });

  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const ellG = g.append('g');
  const dotG = g.append('g');
  const meanG = g.append('g');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* A crowded start: three means close together at the data's centre, so the
   * first E-step paints genuine 30/30/40 confusion instead of a fait
   * accompli. Deterministic. */
  const cx = d3.mean(pts, (p) => p[0]);
  const cy = d3.mean(pts, (p) => p[1]);
  const start = initFrom(pts, [[cx - 0.7, cy + 0.4], [cx + 0.7, cy + 0.4], [cx, cy - 0.8]]);
  const path = em(pts, start);
  const last = path[path.length - 1];

  const state = (stage) => {
    /* 0 bare model, 1 first E-step, 2 first M-step, 3 midway, 4 converged */
    const frame = stage <= 1 ? path[0] : stage === 2 ? path[1] : stage === 3 ? path[Math.min(4, path.length - 1)] : last;
    drawSoftPoints(dotG, pts, stage >= 1 ? frame.resp : null, x, y);
    drawMeans(meanG, frame.model.means, x, y);
    drawEllipses(ellG, frame.model, x, y);
    caption.text(
      stage === 0 ? 'three Gaussians, guessed: means, shapes and weights'
        : stage === 1 ? 'E-step: every point gets a share in every cluster'
          : stage === 4 ? `converged: mean log-likelihood ${last.loglik.toFixed(4)} (scikit-learn: ${data.blobs.ref_loglik})`
            : `mean log-likelihood ${frame.loglik.toFixed(4)}, and it only ever climbs`
    );
  };

  scrolly('#em-scrolly .step', { 0: () => state(0), 1: () => state(1), 2: () => state(2), 3: () => state(3), 4: () => state(4) });

  /* Runtime check: this browser's EM must reach sklearn's optimum. */
  if (Math.abs(last.loglik - data.blobs.ref_loglik) > Math.abs(data.blobs.ref_loglik) * 0.005) {
    console.warn(`scrolly EM converged at ${last.loglik.toFixed(4)}, sklearn reference is ${data.blobs.ref_loglik}`);
  }
}
