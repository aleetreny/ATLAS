/* Draggable-line least squares playground with residual-diagnostics view. */
import { makeChart, drawAxes, drawGrid, axisLabels, ols1d } from '../../assets/js/chart.js';
import { mse, r2Score } from './mathkit.js';

export function initMSEPlayground(mpg) {
  const pts = mpg.points;
  const xs = pts.map((d) => d.hp);
  const ys = pts.map((d) => d.mpg);
  const fit = ols1d(xs, ys);

  const { g, w, h } = makeChart('#mse-chart', {
    width: 640,
    height: 480,
    margin: { top: 24, right: 26, bottom: 60, left: 60 },
  });

  const x = d3.scaleLinear().domain([35, 235]).range([0, w]);
  const yData = d3.scaleLinear().domain([0, 50]).range([h, 0]);
  const yResid = d3.scaleLinear().domain([-22, 22]).range([h, 0]);

  const HP_A = 70;
  const HP_B = 200;
  const state = { b0: 45, b1: -0.05, resid: false };

  /* The grid is redrawn on every render, alongside the axis it belongs to.
   * Drawing it once here left MPG rules behind the residual axis, so the faint
   * lines named heights the labels disagreed with. */
  axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });

  const stemsG = g.append('g');
  const zeroLine = g
    .append('line')
    .attr('x1', 0).attr('x2', w)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 2)
    .attr('stroke-dasharray', '7 5').attr('opacity', 0);
  const dotsG = g.append('g');
  const halo = g.append('line').attr('stroke', 'white').attr('stroke-width', 7);
  const line = g.append('line').attr('stroke', 'var(--squidink)').attr('stroke-width', 3.5);
  const handlesG = g.append('g');

  const dots = dotsG
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('r', 3.5)
    .attr('fill', 'var(--stone)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.8)
    .attr('opacity', 0.85);

  const predict = (hp) => state.b0 + state.b1 * hp;

  const handles = handlesG
    .selectAll('circle')
    .data([{ hp: HP_A }, { hp: HP_B }])
    .join('circle')
    .attr('r', 9)
    .attr('fill', 'var(--primary)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2.5)
    .attr('cursor', 'ns-resize')
    .call(
      d3.drag().on('drag', (event, d) => {
        const yVal = yData.invert(Math.max(0, Math.min(h, event.y)));
        // move this handle's point; keep the other fixed; line passes through both
        const other = d.hp === HP_A ? { hp: HP_B, y: predict(HP_B) } : { hp: HP_A, y: predict(HP_A) };
        const b1 = (other.y - yVal) / (other.hp - d.hp);
        const b0 = yVal - b1 * d.hp;
        state.b0 = b0;
        state.b1 = b1;
        render(0);
      })
    );

  const eqLine = document.querySelector('#mse-eq-line');
  const eqStats = document.querySelector('#mse-eq-stats');

  function render(dur = 350) {
    const t = d3.transition().duration(dur).ease(d3.easeCubicOut);
    const yhat = xs.map(predict);
    const curMSE = mse(ys, yhat);
    const curR2 = r2Score(ys, yhat);

    if (!state.resid) {
      // ---- data view ----
      drawGrid(g, x, yData, w, h);
      drawAxes(g, x, yData, w, h);
      axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });
      zeroLine.transition(t).attr('opacity', 0);
      dots.transition(t).attr('cx', (d) => x(d.hp)).attr('cy', (d) => yData(d.mpg));
      const [x0, x1] = x.domain();
      for (const l of [halo, line]) {
        l.transition(t)
          .attr('x1', x(x0)).attr('y1', yData(predict(x0)))
          .attr('x2', x(x1)).attr('y2', yData(predict(x1)))
          .attr('opacity', 1);
      }
      handles
        .transition(t)
        .attr('cx', (d) => x(d.hp))
        .attr('cy', (d) => yData(predict(d.hp)))
        .attr('opacity', 1)
        .attr('pointer-events', 'all');
      stemsG
        .selectAll('line')
        .data(pts)
        .join('line')
        .transition(t)
        .attr('x1', (d) => x(d.hp)).attr('x2', (d) => x(d.hp))
        .attr('y1', (d) => yData(d.mpg)).attr('y2', (d) => yData(predict(d.hp)))
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1).attr('opacity', 0.35);
    } else {
      // ---- residual view ----
      /* Size the axis to the residuals the current line actually produces. A
       * fixed [-22, 22] window is fine for the fitted line but throws 81 of the
       * 392 points off the canvas the moment you drag the handles, and an
       * unclipped chart paints them over the axis labels. */
      const span = d3.max(pts, (d) => Math.abs(d.mpg - predict(d.hp))) * 1.06;
      yResid.domain([-span, span]);
      drawGrid(g, x, yResid, w, h);
      drawAxes(g, x, yResid, w, h);
      axisLabels(g, w, h, { x: 'horsepower', y: 'residual' });
      zeroLine.attr('y1', yResid(0)).attr('y2', yResid(0)).transition(t).attr('opacity', 1);
      dots
        .transition(t)
        .attr('cx', (d) => x(d.hp))
        .attr('cy', (d) => yResid(d.mpg - predict(d.hp)));
      for (const l of [halo, line]) l.transition(t).attr('opacity', 0);
      handles.transition(t).attr('opacity', 0).attr('pointer-events', 'none');
      stemsG
        .selectAll('line')
        .data(pts)
        .join('line')
        .transition(t)
        .attr('x1', (d) => x(d.hp)).attr('x2', (d) => x(d.hp))
        .attr('y1', yResid(0)).attr('y2', (d) => yResid(d.mpg - predict(d.hp)))
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1).attr('opacity', 0.35);
    }

    // ---- equation panel ----
    if (window.katex) {
      const sign = state.b1 < 0 ? '-' : '+';
      katex.render(
        `\\hat{y} = ${state.b0.toFixed(2)} ${sign} ${Math.abs(state.b1).toFixed(3)} \\cdot \\text{hp}`,
        eqLine,
        { displayMode: true, throwOnError: false }
      );
      const isBest = Math.abs(curMSE - mse(ys, xs.map((v) => fit.intercept + fit.slope * v))) < 0.05;
      eqStats.innerHTML =
        `<span class="slider-label">MSE = <span class="value">${curMSE.toFixed(1)}</span>` +
        ` &nbsp;·&nbsp; R² = <span class="value">${curR2.toFixed(3)}</span>` +
        (isBest ? ' &nbsp;·&nbsp; <span style="color:var(--primary)" class="bold">← the least squares line</span>' : '') +
        `</span>`;
    }
  }

  /* Animate the line into the least squares solution.
   *
   * This runs on a d3.timer rather than a d3.transition. A transition whose
   * tween calls render() does not work here: render() creates its own
   * transitions on the same nodes, and an unnamed d3.transition interrupts the
   * one already running, so the animation cancelled itself on the first tick
   * and the button appeared to do nothing at all. */
  let snapTimer = null;
  document.querySelector('#mse-fit-btn').addEventListener('click', () => {
    if (snapTimer) snapTimer.stop();
    const from = { b0: state.b0, b1: state.b1 };
    const to = { b0: fit.intercept, b1: fit.slope };
    const DURATION = 900;
    snapTimer = d3.timer((elapsed) => {
      const u = Math.min(1, elapsed / DURATION);
      const e = d3.easeCubicInOut(u);
      state.b0 = from.b0 + (to.b0 - from.b0) * e;
      state.b1 = from.b1 + (to.b1 - from.b1) * e;
      render(0);
      if (u >= 1) {
        snapTimer.stop();
        snapTimer = null;
      }
    });
  });

  document.querySelector('#mse-reset-btn').addEventListener('click', () => {
    state.b0 = 10 + Math.random() * 45;
    state.b1 = -0.35 + Math.random() * 0.45;
    render(400);
  });

  document.querySelector('#mse-resid-toggle').addEventListener('change', (e) => {
    state.resid = e.target.checked;
    render(600);
  });

  render(0);
}
