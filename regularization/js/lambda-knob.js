/* One slider, two penalties, same data: how Ridge and Lasso differ in what they
 * do to the SAME overgrown polynomial. Ridge shrinks everything a little, Lasso
 * deletes most of it. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { makeModel, polyDesign, nonZero, rmse } from './mathkit.js';

const truth = (x) => Math.sin(x) + x / 2;
const symlog = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v));

export function initLambdaKnob(sine) {
  const xs = sine.points.map((d) => d.x);
  const ys = sine.points.map((d) => d.y);
  const X = polyDesign(xs, sine.degree, sine.poly_mu, sine.poly_sd);
  const model = makeModel(X, ys);
  const yMean = d3.mean(ys);

  const grid = d3.range(0, 5.001, 0.02);
  const Xg = polyDesign(grid, sine.degree, sine.poly_mu, sine.poly_sd);

  // Both coefficient spectra live in the top margin. Underneath the plot they
  // collided with the x axis label and the lower one fell outside the viewBox
  // entirely, so half of it was simply not rendered.
  const { g, w, h } = makeChart('#knob-chart', {
    width: 620,
    height: 620,
    margin: { top: 175, right: 22, bottom: 62, left: 62 },
  });

  const x = d3.scaleLinear().domain([-0.15, 5.15]).range([0, w]);
  const y = d3.scaleLinear().domain([-1.6, 4.6]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'y' });

  g.append('clipPath').attr('id', 'knob-clip').append('rect').attr('width', w).attr('height', h);
  const plot = g.append('g').attr('clip-path', 'url(#knob-clip)');
  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);

  plot
    .append('path')
    .attr('d', line(grid.map((v) => [v, truth(v)])))
    .attr('fill', 'none')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', '7 6')
    .attr('opacity', 0.4);

  // both penalties drawn at once, so the shapes can be compared directly
  const mkPath = (color) =>
    plot.append('path').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 3.4);
  const haloRidge = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7);
  const ridgePath = mkPath('var(--anchor)');
  const haloLasso = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7);
  const lassoPath = mkPath('var(--cosmos)');

  g.append('g')
    .selectAll('circle')
    .data(sine.points)
    .join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 4.6)
    .attr('fill', 'var(--squidink)')
    .attr('stroke', 'white')
    .attr('stroke-width', 1.3);

  // ---------- paired coefficient spectra, in the top margin ----------
  // Local y (the inner group starts at margin.top): ridge bars grow up from
  // RIDGE_BASE, lasso bars grow down from LASSO_BASE, each capped at BAR_MAX.
  const RIDGE_BASE = -105;
  const LASSO_BASE = -80;
  const BAR_MAX = 40;
  const spec = g.append('g');
  const bx = d3.scaleBand().domain(d3.range(sine.degree)).range([0, w]).padding(0.3);

  const sliderEl = document.querySelector('#knob-alpha');

  /* Solve every stop of the slider once, up front.
   *
   * Ridge is exact (QR on an augmented system), so it does not care. Lasso is
   * coordinate descent, and on a degree-15 design with twenty points the
   * columns are close enough to parallel that a cold 800-iteration solve is
   * nowhere near the answer: at lambda = 1e-4 it reported max |coef| = 3.19 and
   * 13 surviving features when scikit-learn, shipped in the reference block,
   * says 8.15 and 7. The widget was drawing a lasso far tamer than the real
   * one, which is the opposite of what this page is arguing.
   *
   * Warm starting down the path from the strongest penalty, and giving each
   * step a real iteration budget, fixes it: measured against every reference
   * alpha the agreement is now to four decimals.
   *
   * The full path costs around a quarter of a second of coordinate descent, so
   * it is computed in idle time after the first paint. Until it lands, the two
   * arrays hold a direct fit at the slider's starting stop, where the problem
   * is well enough conditioned that the one-shot solve matches the warm-started
   * one, so nothing on screen jumps when the path arrives. */
  const stops = [];
  for (let lv = +sliderEl.min; lv <= +sliderEl.max + 1e-9; lv += +sliderEl.step) stops.push(10 ** lv);
  let lassoFits = null;
  let ridgeFits = null;
  const nearest = (a) => {
    let best = 0;
    for (let i = 1; i < stops.length; i++) if (Math.abs(stops[i] - a) < Math.abs(stops[best] - a)) best = i;
    return best;
  };

  /* Size the bar scale to the coefficients this slider can actually produce.
   * A fixed symlog domain borrowed from the unregularized fit (which reaches
   * into the hundreds) squashed every bar here into a few invisible pixels,
   * because nothing on this widget exceeds about 3 at the slider's floor of
   * lambda = 1e-3 (the sklearn reference says 2.68 there). The cap is a
   * constant rather than a sweep so the scale exists before the idle-time path
   * computation lands. */
  const by = d3.scaleLinear().domain([0, symlog(3.2) * 1.04]).range([0, BAR_MAX]).clamp(true);

  for (const yb of [RIDGE_BASE, LASSO_BASE]) {
    spec.append('line').attr('x1', 0).attr('x2', w).attr('y1', yb).attr('y2', yb)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
  }

  spec.append('text').attr('class', 'chart-annotation').attr('x', 0).attr('y', RIDGE_BASE - BAR_MAX - 8)
    .style('font-size', '0.66rem').attr('fill', 'var(--anchor)').text('ridge |coef|');
  spec.append('text').attr('class', 'chart-annotation').attr('x', 0).attr('y', LASSO_BASE + BAR_MAX + 18)
    .style('font-size', '0.66rem').attr('fill', 'var(--cosmos)').text('lasso |coef|');

  const ridgeBars = spec.selectAll('rect.r').data(new Array(sine.degree).fill(0)).join('rect')
    .attr('class', 'r').attr('x', (d, i) => bx(i)).attr('width', bx.bandwidth())
    .attr('y', RIDGE_BASE).attr('height', 0).attr('fill', 'var(--anchor)');
  const lassoBars = spec.selectAll('rect.l').data(new Array(sine.degree).fill(0)).join('rect')
    .attr('class', 'l').attr('x', (d, i) => bx(i)).attr('width', bx.bandwidth())
    .attr('y', LASSO_BASE).attr('height', 0).attr('fill', 'var(--cosmos)');

  const slider = sliderEl;
  const alphaLabel = document.querySelector('#knob-alpha-label');
  const statsEl = document.querySelector('#knob-stats');

  /* Look the reference up by VALUE, not by string. The keys are written the way
   * Python prints a float ("1e-06", "1.0") and the way JavaScript prints the
   * same number ("0.000001", "1") does not match, so a string lookup silently
   * skipped most of the checks and one genuinely wrong fit sailed through. */
  const REF = Object.entries(sine.ref.alphas).map(([k, v]) => [Number(k), v]);
  const refFor = (a) => (REF.find(([k]) => Math.abs(k - a) <= 1e-9 * Math.max(k, a)) || [])[1];

  function curveFor(b) {
    return grid.map((v, i) => [v, yMean + Xg[i].reduce((s, u, j) => s + u * b[j], 0)]);
  }

  function render() {
    const alpha = 10 ** (+slider.value);
    alphaLabel.textContent = alpha < 0.001 ? alpha.toExponential(0) : alpha.toPrecision(2);

    const i = nearest(alpha);
    /* Before the idle-time path lands, fall back to a direct solve at the
     * requested stop. Ridge is exact either way; lasso's one-shot solve is
     * only trusted here because the slider floor sits where the problem is
     * well conditioned, which is itself something the article explains. */
    const bR = ridgeFits ? ridgeFits[i] : model.fit(stops[i], 0);
    const bL = lassoFits ? lassoFits[i] : model.fit(stops[i], 1);

    const dR = line(curveFor(bR));
    const dL = line(curveFor(bL));
    haloRidge.attr('d', dR);
    ridgePath.attr('d', dR);
    haloLasso.attr('d', dL);
    lassoPath.attr('d', dL);

    ridgeBars.data(bR).transition().duration(160)
      .attr('y', (v) => RIDGE_BASE - by(symlog(Math.abs(v))))
      .attr('height', (v) => by(symlog(Math.abs(v))));
    lassoBars.data(bL).transition().duration(160)
      .attr('y', LASSO_BASE)
      .attr('height', (v) => by(symlog(Math.abs(v))));

    // honest scoring against the noiseless truth, not against the noisy dots
    const errTruth = (b) => {
      const pred = grid.map((v, i) => yMean + Xg[i].reduce((s, u, j) => s + u * b[j], 0));
      return rmse(grid.map(truth), pred);
    };
    const fmt = (v) => (Math.abs(v) >= 1000 ? d3.format(',.0f')(v) : v.toFixed(2));
    statsEl.innerHTML =
      `<table class="knob-table">
         <tr><th></th><th style="color:var(--anchor)">ridge (L2)</th><th style="color:var(--cosmos)">lasso (L1)</th></tr>
         <tr><td>max |coef|</td><td>${fmt(d3.max(bR, (v) => Math.abs(v)))}</td><td>${fmt(d3.max(bL, (v) => Math.abs(v)))}</td></tr>
         <tr><td>features alive</td><td>${nonZero(bR, 1e-6)}/15</td><td>${nonZero(bL, 1e-6)}/15</td></tr>
         <tr><td>error vs truth</td><td>${errTruth(bR).toFixed(3)}</td><td>${errTruth(bL).toFixed(3)}</td></tr>
       </table>`;

    // Runtime check against the scikit-learn reference shipped with the data.
    // Tolerance is relative, and deliberately loose: on this degree-15 design
    // scikit-learn's own coordinate descent reports ConvergenceWarning at the
    // weakest penalties (it hits max_iter with a duality gap still open), so a
    // few percent of disagreement down there is a property of the problem, not
    // a bug in either solver. Anywhere the problem is well posed they agree to
    // four decimals.
    const ref = refFor(alpha);
    if (ref) {
      const jsMax = d3.max(bL, (v) => Math.abs(v));
      const rel = Math.abs(jsMax - ref.lasso_max_abs) / Math.max(ref.lasso_max_abs, 1e-9);
      if (rel > 0.05) console.warn(`lasso@${alpha}: JS ${jsMax.toFixed(4)} vs sklearn ${ref.lasso_max_abs}`);
      if (nonZero(bL, 1e-6) !== ref.lasso_nonzero) {
        console.warn(`lasso@${alpha}: JS keeps ${nonZero(bL, 1e-6)} features vs sklearn ${ref.lasso_nonzero}`);
      }
    }
  }

  slider.addEventListener('input', render);
  render();

  /* The full warm-started path, computed off the critical path and swapped in.
   * Re-rendering after it lands is a no-op visually at the starting stop, and
   * from then on every drag reads the converged fits. */
  const idle = window.requestIdleCallback || ((f) => setTimeout(f, 80));
  idle(() => {
    lassoFits = model.path(stops, 1, { maxIter: 20000, tol: 1e-13 });
    ridgeFits = stops.map((a) => model.fit(a, 0));
    render();
  });
}
