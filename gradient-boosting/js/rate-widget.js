/* Two knobs that trade against each other: how big a step each tree takes, and
 * how many trees you allow. Both the live one-dimensional fit and the measured
 * surface from the real diamonds data are shown, because the shape of that
 * trade is the single most useful thing to internalise about boosting. */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';
import { makeData, boost, ensemblePredict, stumpPredict, TRUTH, rmse } from './mathkit.js';

export function initRateWidget(dia) {
  const { xs, ys } = makeData();
  const grid = d3.range(0, 9.601, 0.05);
  const truth = grid.map(TRUTH);

  /* ---------------- live fit ---------------- */
  const A = makeChart('#rate-chart', {
    width: 620, height: 380, margin: { top: 26, right: 24, bottom: 52, left: 62 },
  });
  const x = d3.scaleLinear().domain([-0.2, 9.8]).range([0, A.w]);
  const y = d3.scaleLinear().domain([-4.5, 9]).range([A.h, 0]);
  drawGrid(A.g, x, y, A.w, A.h);
  drawAxes(A.g, x, y, A.w, A.h);
  axisLabels(A.g, A.w, A.h, { x: 'x', y: 'y' });

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
  const step = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveStepAfter);
  A.g.append('path').attr('d', line(grid.map((v) => [v, TRUTH(v)])))
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2)
    .attr('stroke-dasharray', '7 6').attr('opacity', 0.35);
  A.g.append('g').selectAll('circle').data(xs.map((v, i) => [v, ys[i]])).join('circle')
    .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.2)
    .attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.7);
  const halo = A.g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const fit = A.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);

  /* ---------------- measured surface from the diamonds ---------------- */
  const B = makeChart('#rate-surface', {
    width: 620, height: 300, margin: { top: 26, right: 116, bottom: 58, left: 62 },
  });
  const sx = d3.scaleLog().domain([d3.min(dia.n_grid), d3.max(dia.n_grid)]).range([0, B.w]);
  const flat = dia.lr_surface.flatMap((r) => r.values);
  const sy = d3.scaleLog().domain([d3.min(flat) * 0.92, d3.max(flat) * 1.05]).range([B.h, 0]);
  drawGrid(B.g, sx, sy, B.w, B.h);
  drawAxes(B.g, sx, sy, B.w, B.h, { xTicks: 5, yTicks: 4, yFmt: d3.format('~s') });
  axisLabels(B.g, B.w, B.h, { x: 'number of trees', y: 'validation RMSE ($)' });

  const COLORS = ['var(--anchor)', 'var(--aqua)', 'var(--primary)', 'var(--smile)', 'var(--galaxy)'];
  const sline = d3.line().x((d, i) => sx(dia.n_grid[i])).y((d) => sy(d));
  dia.lr_surface.forEach((row, i) => {
    B.g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6).attr('d', sline(row.values));
    B.g.append('path').attr('fill', 'none').attr('stroke', COLORS[i]).attr('stroke-width', 3).attr('d', sline(row.values));
  });
  // The whole point of this chart is that the rates converge, so their end
  // labels land on top of each other. Push them apart.
  const endLabels = spread(
    dia.lr_surface.map((row, i) => ({
      t: `lr ${row.lr}`, c: COLORS[i], y: sy(row.values[row.values.length - 1]),
    })),
    14
  );
  B.g.selectAll('text.lrlabel').data(endLabels).join('text')
    .attr('class', 'lrlabel chart-annotation')
    .attr('x', B.w + 8).attr('y', (d) => d.y + 4)
    .style('font-size', '0.62rem').style('letter-spacing', '0.4px').style('text-transform', 'none')
    .attr('fill', (d) => d.c).text((d) => d.t);

  const lrSlider = document.querySelector('#rate-lr');
  const nSlider = document.querySelector('#rate-n');
  const lrLabel = document.querySelector('#rate-lr-label');
  const nLabel = document.querySelector('#rate-n-label');
  const stats = document.querySelector('#rate-stats');

  const cache = new Map();
  function modelFor(lr) {
    if (!cache.has(lr)) cache.set(lr, boost(xs, ys, { rounds: 300, lr }));
    return cache.get(lr);
  }

  function render() {
    const lr = +lrSlider.value;
    const n = +nSlider.value;
    lrLabel.textContent = lr.toFixed(2);
    nLabel.textContent = n;

    const model = modelFor(lr);
    const pred = ensemblePredict(model, grid, n);
    const d = step(grid.map((v, i) => [v, pred[i]]));
    halo.attr('d', d);
    fit.attr('d', d);

    const errTruth = rmse(truth, pred);
    const effort = lr * n;

    // The verdict is read off the measured error, not guessed from the
    // settings. On a smooth, lightly-noised curve a large learning rate is not
    // automatically punished, and saying otherwise while the number on screen
    // disagrees would be worse than saying nothing. Until the background sweep
    // that establishes the floor has finished, the comparative half of the
    // readout simply is not shown, rather than being shown against a floor that
    // does not exist yet.
    const best = bestCache;
    let tail = '';
    let refNote = ' <span style="opacity:.7">(finding the floor of this curve…)</span>';
    if (best !== null) {
      const excess = errTruth / best - 1;
      const verdict =
        effort < 3
          ? 'Under-trained: the ensemble has barely moved off the mean. Take bigger steps or take more of them.'
          : excess < 0.06
            ? `Within ${(excess * 100).toFixed(0)}% of the best this curve allows. Notice how many different combinations of ν and trees land here: it is the product that matters far more than either one alone.`
            : excess < 0.25
              ? `About ${(excess * 100).toFixed(0)}% worse than the best reachable fit. Either direction will help.`
              : `${(excess * 100).toFixed(0)}% off the best reachable fit. ${lr > 0.6 ? 'Large steps overshoot and the later trees spend their time undoing it.' : 'Too few steps taken, at too small a size.'}`;
      tail = `<div class="slider-label" style="margin-top:.45rem;line-height:1.45">${verdict}</div>`;
      refNote = ` <span style="opacity:.7">(best reachable ${best.toFixed(3)})</span>`;
    }

    stats.innerHTML =
      `<div class="slider-label">error against the true curve: <span class="value">${errTruth.toFixed(3)}</span>` +
      refNote +
      `</div>` +
      `<div class="slider-label">learning rate × trees = <span class="value">${effort.toFixed(1)}</span>` +
      ` <span style="opacity:.7">(roughly how far the ensemble has travelled)</span></div>` +
      tail;
  }

  /* Sweep the two knobs once to find the floor, so "how good is this?" has a
   * reference rather than an adjective.
   *
   * The sweep has to be fine enough that the reader cannot beat it. A coarse
   * five-by-six grid reported 0.4565 as the floor while nu = 0.25 with 151
   * trees, a setting two drags away, reaches 0.451: the widget was congratulating
   * you for being "within 0% of the best" on a fit better than its own best.
   * The real floor over the sliders' full range is 0.4379, at nu = 0.72 and 48
   * trees.
   *
   * Fifty learning rates by three hundred tree counts is fifteen thousand
   * evaluations, affordable because the ensemble is accumulated one stump at a
   * time instead of re-summed from scratch at every n. It still costs a few
   * hundred milliseconds of model fitting, which is why it runs in idle time
   * after the first paint rather than inside it. */
  let bestCache = null;
  function bestError() {
    if (bestCache !== null) return bestCache;
    let b = Infinity;
    for (let lr = +lrSlider.min; lr <= +lrSlider.max + 1e-9; lr += 0.02) {
      const m = modelFor(+lr.toFixed(2));
      const acc = grid.map(() => m.base);
      for (let t = 0; t < m.stumps.length; t++) {
        for (let i = 0; i < grid.length; i++) acc[i] += m.lr * stumpPredict(m.stumps[t], grid[i]);
        b = Math.min(b, rmse(truth, acc));
      }
    }
    bestCache = b;
    return b;
  }

  lrSlider.addEventListener('input', render);
  nSlider.addEventListener('input', render);
  render();

  /* The floor sweep fits fifty ensembles and costs around 400ms, which was the
   * bulk of this page's load time when it ran synchronously in the first
   * render. Kicked to idle time instead: the widget appears at once with the
   * comparative verdict held back, and fills it in as soon as the sweep lands. */
  const idle = window.requestIdleCallback || ((f) => setTimeout(f, 80));
  idle(() => {
    bestError();
    render();
  });
}
