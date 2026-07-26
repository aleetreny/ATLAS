/* The reader's own EM run: grab the three diamonds, drop them anywhere,
 * press run, and watch the mixture climb from that start, animated one
 * iteration at a time. The readout judges the landing against scikit-learn's
 * optimum, because the initialisation lottery of the previous article did
 * not go away when the assignments went soft.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { em, initFrom, eStep, seedMeans, rng } from './gmmkit.js';
import { COLOURS, drawSoftPoints, drawEllipses } from './mixdraw.js';

export function initDragWidget(data) {
  const pts = data.blobs.points;
  const ref = data.blobs.ref_loglik;
  const { g, w, h } = makeChart('#drag-chart', {
    width: 560,
    height: 500,
    margin: { top: 30, right: 24, bottom: 54, left: 58 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const ellG = g.append('g');
  const dotG = g.append('g');
  const handleG = g.append('g');

  const stats = document.querySelector('#drag-stats');
  const seeds = rng(31);
  let means = [[-4, 4.2], [-3.4, 4.6], [-4.4, 3.7]];   // a deliberately awful corner pile-up
  let timer = null;
  let ran = false;

  function drawHandles(enabled) {
    const sel = handleG.selectAll('g.hnd').data(means, (d, i) => i);
    const enter = sel.enter().append('g').attr('class', 'hnd').style('cursor', 'grab');
    enter.append('circle').attr('r', 14).attr('fill', 'transparent');
    enter.append('rect').attr('width', 15).attr('height', 15)
      .attr('transform', 'rotate(45 7.5 7.5)').attr('x', -7.5).attr('y', -7.5)
      .attr('stroke', 'white').attr('stroke-width', 2.2);
    const merged = enter.merge(sel);
    merged.select('rect').attr('fill', (m, i) => COLOURS[i % COLOURS.length]);
    merged.attr('transform', (m) => `translate(${x(m[0])},${y(m[1])})`);
    merged.call(d3.drag()
      .on('drag', function (event, m) {
        if (!enabled) return;
        m[0] = x.invert(event.x);
        m[1] = y.invert(event.y);
        d3.select(this).attr('transform', `translate(${event.x},${event.y})`);
        idle('Start moved. Press "run EM from here".');
      }));
  }

  function idle(msg) {
    if (timer) { timer.stop(); timer = null; }
    ran = false;
    const model = initFrom(pts, means, 8);
    const { resp, loglik } = eStep(pts, model);
    drawSoftPoints(dotG, pts, resp, x, y);
    drawEllipses(ellG, model, x, y);
    drawHandles(true);
    stats.innerHTML =
      `<div class="slider-label">start log-likelihood: <span class="value">${loglik.toFixed(4)}</span></div>` +
      `<div class="slider-label" style="margin-top:.4rem">sklearn's optimum: <span class="value">${ref}</span></div>` +
      `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">${msg}</div>`;
  }

  function run() {
    if (timer) { timer.stop(); timer = null; }
    const path = em(pts, initFrom(pts, means, 8));
    let i = 0;
    ran = true;
    /* Long runs speed up so no start takes more than ~12 seconds to watch. */
    const per = Math.min(180, 12000 / path.length);
    timer = d3.timer((elapsed) => {
      const step = Math.min(Math.floor(elapsed / per), path.length - 1);
      if (step !== i || elapsed === 0) {
        i = step;
        const f = path[i];
        drawSoftPoints(dotG, pts, f.resp, x, y);
        drawEllipses(ellG, f.model, x, y);
        means = f.model.means.map((m) => [...m]);
        drawHandles(false);
        stats.innerHTML =
          `<div class="slider-label">iteration <span class="value">${i}</span> of ${path.length - 1}</div>` +
          `<div class="slider-label" style="margin-top:.4rem">mean log-likelihood: <span class="value">${f.loglik.toFixed(4)}</span></div>`;
      }
      if (i >= path.length - 1) {
        timer.stop();
        timer = null;
        drawHandles(true);   // hand the diamonds back: play again from here
        const end = path[path.length - 1].loglik;
        const found = Math.abs(end - ref) < Math.abs(ref) * 0.005;
        stats.innerHTML +=
          `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">Converged at ` +
          `<span class="value">${end.toFixed(4)}</span> in ${path.length - 1} iterations. ` +
          `${found
            ? `That is scikit-learn's optimum: from your start, EM found the same answer.`
            : `sklearn's best is <span class="value">${ref}</span>: your start led EM into a worse local ` +
              `optimum, and no amount of further iterating will climb out. Soft assignments inherit ` +
              `k-means' lottery.`}</div>`;
      }
    });
  }

  document.querySelector('#drag-run').addEventListener('click', run);
  document.querySelector('#drag-shuffle').addEventListener('click', () => {
    means = seedMeans(pts, 3, seeds);
    idle('A k-means++-style random start. Drag the diamonds if you want it worse.');
  });
  document.querySelector('#drag-corner').addEventListener('click', () => {
    means = [[-4, 4.2], [-3.4, 4.6], [-4.4, 3.7]];
    idle('All three components piled into an empty corner. Surely this cannot work.');
  });

  idle('All three components piled into an empty corner. Drag them, or run as is: EM escapes more corners ' +
    'than you would guess. To see it genuinely strand, park all three inside the same cloud and run.');
}
