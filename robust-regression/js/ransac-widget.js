/* RANSAC, replayed. One frame per hypothesis: the two sampled stars, the line
 * they define, and who voted for it. Showing the search is the point; showing
 * only the answer hides what makes the method work. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { ransac, ols } from './mathkit.js';

export function initRansac(stars) {
  const pts = stars.points;
  const xs = pts.map((d) => d.x);
  const ys = pts.map((d) => d.y);

  const { g, plot, w, h } = makeChart('#ransac-chart', {
    width: 700,
    height: 430,
    margin: { top: 40, right: 26, bottom: 58, left: 60 },
    clip: true,
  });

  const x = d3.scaleLinear().domain([3.35, 4.75]).range([0, w]);
  const y = d3.scaleLinear().domain([3.7, 6.5]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'log surface temperature', y: 'log light intensity' });

  const bandG = plot.append('g');
  const candG = plot.append('g');
  const bestG = plot.append('g');
  const dotsG = g.append('g');

  const bandArea = bandG
    .append('path')
    .attr('fill', 'var(--primary)')
    .attr('fill-opacity', 0.12);
  const candLine = candG.append('line').attr('stroke', 'var(--squidink)').attr('stroke-width', 2).attr('opacity', 0);
  const bestHalo = bestG.append('line').attr('stroke', 'white').attr('stroke-width', 7).attr('opacity', 0);
  const bestLine = bestG.append('line').attr('stroke', 'var(--primary)').attr('stroke-width', 3.6).attr('opacity', 0);

  const dots = dotsG
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 5)
    .attr('fill', 'var(--stone)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1);

  const sampleRings = dotsG.append('g');
  const readout = g
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w / 2)
    .attr('y', -16)
    .attr('text-anchor', 'middle');

  const epsSlider = document.querySelector('#ransac-eps');
  const epsLabel = document.querySelector('#ransac-eps-label');
  const statsEl = document.querySelector('#ransac-stats');

  let run = null;
  let cursor = 0;
  let timer = null;

  function rebuild() {
    const eps = +epsSlider.value;
    epsLabel.textContent = eps.toFixed(2);
    run = ransac(xs, ys, { threshold: eps, iterations: 120, seed: 11 });
    cursor = 0;
    showFrame(0, true);
  }

  function lineCoords(fit) {
    const [x0, x1] = x.domain();
    return { x1: x(x0), y1: y(fit.intercept + fit.slope * x0), x2: x(x1), y2: y(fit.intercept + fit.slope * x1) };
  }

  function showFrame(i, reset = false) {
    if (!run || !run.history.length) return;
    const frame = run.history[Math.min(i, run.history.length - 1)];
    const eps = +epsSlider.value;

    // best-so-far, recomputed from the frames seen up to here
    let best = run.history[0];
    for (let k = 0; k <= i && k < run.history.length; k++) {
      if (run.history[k].count > best.count) best = run.history[k];
    }

    const c = lineCoords(frame);
    candLine.attr('x1', c.x1).attr('y1', c.y1).attr('x2', c.x2).attr('y2', c.y2).attr('opacity', 0.55);

    const bc = lineCoords(best);
    for (const l of [bestHalo, bestLine]) {
      l.attr('x1', bc.x1).attr('y1', bc.y1).attr('x2', bc.x2).attr('y2', bc.y2).attr('opacity', reset && i === 0 ? 0 : 1);
    }

    // the tolerance band around the best line
    const [x0, x1] = x.domain();
    const top = (v) => y(best.intercept + best.slope * v + eps);
    const bot = (v) => y(best.intercept + best.slope * v - eps);
    bandArea.attr('d', `M${x(x0)},${top(x0)}L${x(x1)},${top(x1)}L${x(x1)},${bot(x1)}L${x(x0)},${bot(x0)}Z`);

    const inSet = new Set(best.inliers);
    dots
      .attr('fill', (d, k) => (inSet.has(k) ? 'var(--primary)' : 'var(--stone)'))
      .attr('opacity', (d, k) => (inSet.has(k) ? 1 : 0.55));

    sampleRings
      .selectAll('circle')
      .data(frame.sample)
      .join('circle')
      .attr('cx', (k) => x(pts[k].x))
      .attr('cy', (k) => y(pts[k].y))
      .attr('r', 10)
      .attr('fill', 'none')
      .attr('stroke', 'var(--cosmos)')
      .attr('stroke-width', 2.5);

    readout.text(`guess ${frame.it + 1} of ${run.history.length} · this guess convinced ${frame.count}`);

    const giantsIn = best.inliers.filter((k) => pts[k].giant).length;
    const refined = ols(best.inliers.map((k) => xs[k]), best.inliers.map((k) => ys[k]));
    statsEl.innerHTML =
      `<div class="slider-label">best consensus so far: <span class="value">${best.count}</span> of ${pts.length} stars</div>` +
      `<div class="slider-label">red giants admitted: <span class="value" style="color:${giantsIn ? 'var(--cosmos)' : 'inherit'}">${giantsIn}</span> of 4</div>` +
      `<div class="slider-label" style="margin-top:.4rem">refit on the consensus: slope <span class="value">${refined.slope.toFixed(2)}</span>` +
      ` · least squares on everything: <span class="value" style="color:var(--cosmos)">${stars.ref.ols.slope.toFixed(2)}</span></div>`;
  }

  function stop() {
    if (timer) {
      timer.stop();
      timer = null;
    }
    document.querySelector('#ransac-play').textContent = 'Run the search';
  }

  document.querySelector('#ransac-play').addEventListener('click', () => {
    if (timer) {
      stop();
      return;
    }
    document.querySelector('#ransac-play').textContent = 'Pause';
    let last = 0;
    timer = d3.timer((elapsed) => {
      if (elapsed - last < 260) return;
      last = elapsed;
      cursor++;
      if (cursor >= run.history.length) {
        cursor = run.history.length - 1;
        showFrame(cursor);
        stop();
        return;
      }
      showFrame(cursor);
    });
  });

  document.querySelector('#ransac-step').addEventListener('click', () => {
    stop();
    cursor = Math.min(cursor + 1, run.history.length - 1);
    showFrame(cursor);
  });

  document.querySelector('#ransac-reset').addEventListener('click', () => {
    stop();
    rebuild();
  });

  epsSlider.addEventListener('input', () => {
    stop();
    rebuild();
  });

  rebuild();
}
