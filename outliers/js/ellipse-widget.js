/* Two columns, two rules, and a suspect the reader can put anywhere.
 *
 * The fit is redone from scratch on every move, including or excluding the
 * suspect according to the toggle, because that toggle is the article's whole
 * argument in one control: the honest default of every rule on this page is
 * that the suspect is in the fit, and what that costs is visible as soon as
 * the two are drawn on top of each other.
 */
import { makeChart, drawAxes, axisLabels, drawGrid } from '../../assets/js/chart.js';
import { colMeans, covariance, mahalanobis, ellipsePath, sd as sdOf } from './statkit.js';
import { seriesLabel, legend } from './draw.js';

export function initEllipseWidget(data) {
  const T = data.two_d;
  const A = T.points;
  const Z_CUT = T.z_cut;
  const CHI = T.chi_cut;

  /* Opens on the long axis of the cloud, where the two rules agree and there is
     nothing to see: the instruction is to move off it, and a widget that starts
     at the answer has nothing to ask for. */
  const start = [T.mu[0] + 1.8 * T.sd[0], T.mu[1] + 1.35 * T.sd[1]];
  let suspect = start.slice();
  let inFit = true;

  const chart = makeChart('#ell-chart', {
    width: 640, height: 490, margin: { top: 64, right: 26, bottom: 58, left: 62 },
  });
  const bars = makeChart('#ell-bars', {
    width: 640, height: 170, margin: { top: 30, right: 140, bottom: 34, left: 24 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#ell-readout');

  /* Wide enough for the blind-spot corner and the rectangle, and no wider: a
     flavanoid reading of minus one is not a place the reader should be invited
     to drag to. */
  const span = (vals, extra) => {
    const lo = Math.min(...vals, ...extra);
    const hi = Math.max(...vals, ...extra);
    const pad = (hi - lo) * 0.09;
    return [lo - pad, hi + pad];
  };
  const x = d3.scaleLinear()
    .domain(span(A.map((d) => d[0]), [start[0], T.corner.point[0]])).range([0, w]);
  const y = d3.scaleLinear()
    .domain(span(A.map((d) => d[1]), [start[1], T.corner.point[1]])).range([h, 0]);

  const gGrid = g.append('g');
  const gShapes = g.append('g');
  const gPts = g.append('g');
  const gSus = g.append('g');
  const gNotes = g.append('g');

  function fit() {
    const rows = inFit ? A.concat([suspect]) : A;
    const mu = colMeans(rows);
    const S = covariance(rows, mu);
    const sds = [sdOf(rows.map((r) => r[0])), sdOf(rows.map((r) => r[1]))];
    return { mu, S, sds };
  }

  function render() {
    const { mu, S, sds } = fit();
    const d2 = mahalanobis(A, mu, S);
    const susD2 = mahalanobis([suspect], mu, S)[0];
    const zOf = (pt) => [Math.abs((pt[0] - mu[0]) / sds[0]), Math.abs((pt[1] - mu[1]) / sds[1])];
    const susZ = zOf(suspect);

    drawGrid(gGrid, x, y, w, h, { xTicks: 6, yTicks: 5 });
    drawAxes(gGrid, x, y, w, h, { xTicks: 6, yTicks: 5 });
    axisLabels(gGrid, w, h, { x: T.cols[0].replace(/_/g, ' '), y: T.cols[1].replace(/_/g, ' ') });

    gShapes.selectAll('*').remove();
    /* the rectangle the two margins declare ordinary */
    gShapes.append('rect')
      .attr('x', x(mu[0] - Z_CUT * sds[0]))
      .attr('y', y(mu[1] + Z_CUT * sds[1]))
      .attr('width', Math.abs(x(mu[0] + Z_CUT * sds[0]) - x(mu[0] - Z_CUT * sds[0])))
      .attr('height', Math.abs(y(mu[1] - Z_CUT * sds[1]) - y(mu[1] + Z_CUT * sds[1])))
      .attr('fill', 'var(--cosmos)').attr('opacity', 0.08)
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.3).attr('stroke-opacity', 0.7);
    /* the ellipse the pair declares ordinary */
    const path = ellipsePath(mu, S, CHI);
    gShapes.append('path')
      .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(path) + 'Z')
      .attr('fill', 'var(--anchor)').attr('opacity', 0.1)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6).attr('stroke-opacity', 0.8);

    /* the bottles, tinted by which rule flags them */
    const kind = (i) => {
      const z = zOf(A[i]);
      const rect = z[0] < Z_CUT && z[1] < Z_CUT;
      const ell = d2[i] < CHI;
      if (rect && ell) return 0;
      if (rect && !ell) return 1;                 // margins fine, pair says no
      if (!rect && ell) return 2;                 // margins say no, pair fine
      return 3;                                   // both say no
    };
    const FILL = ['var(--stone)', 'var(--anchor)', 'var(--cosmos)', 'var(--galaxy)'];
    const sel = gPts.selectAll('circle.wine').data(A);
    sel.exit().remove();
    sel.enter().append('circle').attr('class', 'wine').merge(sel)
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', (p, i) => (kind(i) === 0 ? 3 : 4.4))
      .attr('fill', (p, i) => FILL[kind(i)])
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', (p, i) => (kind(i) === 0 ? 0.4 : 0.9))
      .attr('opacity', (p, i) => (kind(i) === 0 ? 0.5 : 0.95));

    gSus.selectAll('*').remove();
    gSus.append('circle')
      .attr('cx', x(suspect[0])).attr('cy', y(suspect[1])).attr('r', 9)
      .attr('fill', 'var(--primary)').attr('stroke', 'var(--paper)').attr('stroke-width', 2)
      .style('cursor', 'grab');
    gSus.append('circle')
      .attr('cx', x(suspect[0])).attr('cy', y(suspect[1])).attr('r', 15)
      .attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 1.2).attr('opacity', 0.55);

    gNotes.selectAll('*').remove();
    const counts = [0, 0, 0, 0];
    A.forEach((p, i) => { counts[kind(i)] += 1; });
    /* Four chips do not fit on one row of a 640 unit canvas, so they go in two,
       which is also the reading order the pairs want. */
    legend(gNotes, [
      { label: 'both rules content', shape: 'dot', colour: 'var(--stone)' },
      { label: `both object (${counts[3]})`, shape: 'dot', colour: 'var(--galaxy)' },
      { label: `margins content, pair not (${counts[1]})`, shape: 'dot', colour: 'var(--anchor)' },
      { label: `pair content, margins not (${counts[2]})`, shape: 'dot', colour: 'var(--cosmos)' },
    ], { x0: 2, y0: -46, gap: 300, size: 10 });
    gNotes.selectAll('g.legend-chip')
      .attr('transform', (d, i) => `translate(${2 + (i % 2) * 300},${i < 2 ? -46 : -28})`);

    /* ------------------------------------------------------- the two scores */
    const bg = bars.g;
    bg.selectAll('*').remove();
    const bx = d3.scaleLinear()
      .domain([0, Math.max(4, Math.abs(susZ[0]), Math.abs(susZ[1]), Math.sqrt(susD2)) * 1.12])
      .range([0, bars.w]);
    const rows = [
      { label: 'largest of the two z', v: Math.max(susZ[0], susZ[1]), cut: Z_CUT,
        colour: 'var(--cosmos)' },
      { label: 'distance on the pair', v: Math.sqrt(susD2), cut: Math.sqrt(CHI),
        colour: 'var(--anchor)' },
    ];
    /* The row's name goes above its bar rather than beside it: a left margin
       wide enough for "largest of the two z" is a left margin wider than the
       phone has, and the label was being cut by the svg element itself. */
    rows.forEach((r, i) => {
      const yy = 30 + i * 60;
      seriesLabel(bg, 0, yy - 6, r.label, { size: 11, opacity: 0.9 });
      bg.append('rect').attr('x', 0).attr('y', yy).attr('width', bx(r.v)).attr('height', 20)
        .attr('fill', r.colour).attr('opacity', r.v > r.cut ? 0.85 : 0.35);
      bg.append('line')
        .attr('x1', bx(r.cut)).attr('x2', bx(r.cut)).attr('y1', yy - 4).attr('y2', yy + 24)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '3 2');
      /* The value sits in the right margin rather than at the end of its own
         bar: at the end of the bar it lands on the cutoff's dashed line
         whenever the score is below the cutoff, which is most of the time. */
      seriesLabel(bg, bars.w + 10, yy + 14,
        `${r.v.toFixed(3)}  ${r.v > r.cut ? 'flagged' : 'ordinary'}`,
        { size: 11, opacity: 0.95 }).style('fill', r.v > r.cut ? r.colour : 'var(--squidink)');
      seriesLabel(bg, Math.min(bx(r.cut), bars.w - 4), yy + 36,
        `cutoff ${r.cut.toFixed(2)}`,
        { anchor: bx(r.cut) > 0.85 * bars.w ? 'end' : 'middle', size: 10, opacity: 0.75 });
    });

    /* --------------------------------------------------------------- prose */
    const flagRect = susZ[0] > Z_CUT || susZ[1] > Z_CUT;
    const flagEll = susD2 > CHI;
    const disagree = flagRect !== flagEll;
    readout.html(
      `<span class="slider-label">the suspect sits at <span class="value">`
      + `${suspect[0].toFixed(2)}</span> and <span class="value">${suspect[1].toFixed(2)}</span>`
      + `</span>. `
      + `Its two readings are ${susZ[0].toFixed(2)} and ${susZ[1].toFixed(2)} standard `
      + `deviations from their own column means, and ${Math.sqrt(susD2).toFixed(2)} from the `
      + `pair. `
      + (disagree
        ? (flagEll
          ? `<span class="bold">The two rules disagree.</span> Column by column it is an `
            + `unremarkable bottle; taken together it is not, because ${T.cols[0].replace(/_/g, ' ')} `
            + `and ${T.cols[1].replace(/_/g, ' ')} correlate at ${T.corr.toFixed(3)} in these `
            + `${A.length} wines and this one does not.`
          : `<span class="bold">The two rules disagree the other way.</span> One of the columns `
            + `is extreme on its own, but the pair is exactly where that reading would put it, `
            + `so the joint rule has nothing to complain about.`)
        : (flagEll
          ? `Both rules object, which is the easy case and the one people picture when they think `
            + `about outliers.`
          : `Both rules are content. Drag the suspect out along the short axis of the cloud, `
            + `across the correlation rather than along it, and they stop agreeing.`))
      + ` The fit ${inFit ? 'includes' : 'excludes'} the suspect: `
      + (inFit
        ? `the ellipse and the rectangle you see were both estimated with it in the sample, `
          + `which is what any of these rules does when it is pointed at a dataset.`
        : `both were estimated from the ${A.length} wines alone, which is what you would do if `
          + `you already knew which bottle was the suspect.`)
    );
  }

  /* -------------------------------------------------------------- the drag */
  /* No requestAnimationFrame anywhere: a control whose work is queued for the
     next frame is dead whenever the browser is not compositing, which is both
     a hidden tab and every automated sweep this site is checked with. Two
     covariances of 179 rows per move is nothing. */
  const clamp = (v, s) => Math.max(s.domain()[0], Math.min(s.domain()[1], v));
  function moveTo(event) {
    const p = d3.pointer(event.sourceEvent ?? event, g.node());
    suspect = [clamp(x.invert(p[0]), x), clamp(y.invert(p[1]), y)];
    render();
  }
  /* A transparent hit area over the whole plot, appended last so that clicking
     or dragging anywhere moves the suspect. The marks underneath stay visible;
     only the pointer belongs to this rectangle. */
  chart.svg.append('rect')
    .attr('x', chart.margin.left).attr('y', chart.margin.top)
    .attr('width', w).attr('height', h)
    .attr('fill', 'transparent')
    .style('cursor', 'crosshair')
    .call(d3.drag().on('start drag', moveTo))
    .on('click', moveTo);

  d3.select('#ell-fit').on('click', function toggle() {
    inFit = !inFit;
    d3.select(this).text(inFit ? 'the suspect is in the fit' : 'the suspect is out of the fit');
    render();
  }).text('the suspect is in the fit');
  d3.select('#ell-reset').on('click', () => {
    suspect = start.slice();
    render();
  });
  d3.select('#ell-blind').on('click', () => {
    suspect = T.corner.point.slice();
    render();
  });

  render();
  return { render, move: (p) => { suspect = p.slice(); render(); } };
}
