/* Thirteen columns, and a batch of bottles that hides itself.
 *
 * The browser does the whole experiment: it takes the first m of the planted
 * rows the generator shipped, stacks them under the 178 real ones, estimates a
 * 13 by 13 covariance from the lot and scores every row against it. Nothing is
 * replayed, so the slider is running the argument rather than illustrating it,
 * and the guard can compare the result against the generator step by step.
 */
import { makeChart, drawAxes, axisLabels, drawGrid } from '../../assets/js/chart.js';
import { selfMahalanobis, chi2ppf, d2Ceiling } from './statkit.js';
import { swarm, marker, legend, seriesLabel } from './draw.js';

export function initDeepWidget(data) {
  const M = data.many_d;
  const X = data.wine.matrix;
  const PLANTED = M.masking.planted;
  const STEPS = M.masking.steps;
  const P = M.p;
  const CUT = chi2ppf(0.975, P);

  let idx = 0;
  const slider = d3.select('#deep-m')
    .attr('min', 0).attr('max', STEPS.length - 1).attr('step', 1).property('value', 0);
  const readout = d3.select('#deep-readout');

  const strip = makeChart('#deep-strip', {
    width: 640, height: 260, margin: { top: 46, right: 26, bottom: 56, left: 30 },
  });
  const curve = makeChart('#deep-curve', {
    width: 640, height: 280, margin: { top: 40, right: 104, bottom: 54, left: 62 },
  });

  /* every step, computed once, because the curve panel shows the whole road at
     every position rather than only the part already walked */
  const ROWS = STEPS.map((m) => {
    const rows = m ? X.concat(PLANTED.slice(0, m)) : X;
    const { d2 } = selfMahalanobis(rows);
    const planted = d2.slice(X.length);
    return {
      m,
      d2,
      plantedMean: m ? planted.reduce((a, b) => a + b, 0) / m : null,
      plantedFlagged: planted.filter((v) => v > CUT).length,
      realFlagged: d2.slice(0, X.length).filter((v) => v > CUT).length,
      worstReal: Math.max(...d2.slice(0, X.length)),
      ceiling: d2Ceiling(rows.length),
    };
  });

  function render() {
    idx = +slider.property('value');
    const R = ROWS[idx];
    const m = R.m;

    /* ------------------------------------------------------------- the strip */
    const sg = strip.g;
    sg.selectAll('*').remove();
    /* The domain covers the ceiling as well as the distances, for two reasons:
       the ceiling is a mark on this axis and a mark outside the domain is drawn
       outside the canvas, and a domain that does not move with the slider keeps
       the picture comparable between positions. */
    const maxD = Math.max(CUT * 1.3, ...ROWS.map((r) => Math.max(r.ceiling, ...r.d2)));
    const sx = d3.scaleSqrt().domain([0, maxD]).range([0, strip.w]);
    const ticks = [0, 5, 10, 25, 50, 100, 175, 225].filter((v) => v <= maxD);
    let gx = sg.append('g').attr('class', 'axis x').attr('transform', `translate(0,${strip.h})`);
    gx.call(d3.axisBottom(sx).tickValues(ticks).tickSizeOuter(0));
    axisLabels(sg, strip.w, strip.h,
      { x: 'squared distance to the sample centre (square root scale)' });

    const baseline = strip.h - 16;
    swarm(sg, R.d2, sx, baseline, {
      r: 3.2, tol: 6.4, maxLevel: 11,
      colour: (i) => (i >= X.length ? 'var(--primary)' : 'var(--stone)'),
      opacity: 0.9,
    });
    marker(sg, sx(CUT), -6, baseline + 12,
      `chi-squared cutoff ${CUT.toFixed(2)}`,
      { colour: 'var(--cosmos)', w: strip.w, size: 10.5 });
    marker(sg, sx(R.ceiling), -6, baseline + 12,
      `the ceiling ${R.ceiling.toFixed(1)}`,
      { colour: 'var(--squidink)', w: strip.w, size: 10.5, dash: '2 4', dy: -18 });
    legend(sg, [{ label: `the ${X.length} real bottles`, shape: 'dot', colour: 'var(--stone)' }]
      .concat(m ? [{ label: `the ${m} planted one${m === 1 ? '' : 's'}`,
        shape: 'dot', colour: 'var(--primary)' }] : []),
    { x0: 2, y0: -28, gap: 210, size: 10.5 });

    /* ------------------------------------------------------------ the curve */
    const cg = curve.g;
    cg.selectAll('*').remove();
    const withPlant = ROWS.filter((r) => r.m > 0);
    const cx = d3.scaleLinear().domain([1, STEPS[STEPS.length - 1]]).range([0, curve.w]);
    const cyMax = Math.max(CUT * 1.25, ...withPlant.map((r) => r.plantedMean));
    const cy = d3.scaleSqrt().domain([0, cyMax]).range([curve.h, 0]);
    drawGrid(cg, cx, cy, curve.w, curve.h, { xTicks: 6, yValues: [0, 10, 25, 50, 100, 160] });
    drawAxes(cg, cx, cy, curve.w, curve.h, { xTicks: 6, yValues: [0, 10, 25, 50, 100, 160] });
    axisLabels(cg, curve.w, curve.h,
      { x: 'how many bottles were planted', y: 'their distance, root scale' });

    cg.append('path')
      .attr('d', d3.line().x((r) => cx(r.m)).y((r) => cy(Math.min(r.plantedMean, cyMax)))(withPlant))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
    cg.selectAll('circle.step').data(withPlant).join('circle').attr('class', 'step')
      .attr('cx', (r) => cx(r.m)).attr('cy', (r) => cy(Math.min(r.plantedMean, cyMax)))
      .attr('r', (r) => (r.m === m ? 5.5 : 3))
      .attr('fill', (r) => (r.plantedFlagged ? 'var(--primary)' : 'var(--paper)'))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6);
    cg.append('line')
      .attr('x1', 0).attr('x2', curve.w).attr('y1', cy(CUT)).attr('y2', cy(CUT))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3');
    seriesLabel(cg, curve.w + 6, cy(CUT) + 4, 'the cutoff', { size: 10.5, opacity: 0.9 })
      .style('fill', 'var(--cosmos)');
    legend(cg, [
      { label: 'filled: still flagged', shape: 'dot', colour: 'var(--primary)' },
      { label: 'hollow: no longer flagged', shape: 'ring', colour: 'var(--primary)' },
    ], { x0: 2, y0: -24, gap: 216, size: 10.5 });

    /* ------------------------------------------------------------- readout */
    d3.select('#deep-m-label').text(m);
    const base = ROWS[0];
    readout.html(
      `<span class="slider-label">planted bottles <span class="value">${m}</span></span> `
      + (m === 0
        ? `Nothing planted yet: the ${X.length} real bottles score up to `
          + `${base.worstReal.toFixed(2)} against a cutoff of ${CUT.toFixed(2)}, and `
          + `${base.realFlagged} of them are past it. That count is the baseline every position `
          + `of this slider is compared against.`
        : `The ${m} planted bottle${m === 1 ? '' : 's'} sit${m === 1 ? 's' : ''} `
          + `${M.masking.shift.toFixed(0)} standard deviations off the cloud along its tightest `
          + `direction, so ${m === 1 ? 'it is' : 'they are'} not a subtle contamination. `
          + `${m === 1 ? 'Its' : 'Their'} own squared distance is `
          + `<span class="bold">${R.plantedMean.toFixed(2)}</span> against a cutoff of `
          + `${CUT.toFixed(2)}, and <span class="bold">${R.plantedFlagged} of the ${m}</span> `
          + `${R.plantedFlagged === 1 ? 'is' : 'are'} flagged. `
          + `Meanwhile ${R.realFlagged} of the ${X.length} real bottles are flagged, against `
          + `${base.realFlagged} with nothing planted`
          + (R.realFlagged > base.realFlagged
            ? `: the batch is not only hiding, it is pushing honest bottles over the line.`
            : R.realFlagged < base.realFlagged
              ? `: the batch has widened the cloud enough to pull honest bottles back inside it.`
              : `, which is the same count.`))
      + ` The sum of all ${R.d2.length} squared distances is `
      + `${R.d2.reduce((a, b) => a + b, 0).toFixed(1)}, which is `
      + `${P} times ${R.d2.length - 1} however the data is arranged.`
    );
  }

  slider.on('input', render);
  render();
  return { render, rows: ROWS };
}
