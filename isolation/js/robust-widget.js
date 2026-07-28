/* The concentration step, one step at a time.
 *
 * This is the previous article's ellipse with its estimator repaired, and the
 * repair is a loop of three lines: fit on the rows you currently trust, score
 * everybody against that fit, keep the h closest. Rousseeuw and Van Driessen's
 * theorem is that the determinant of the covariance never increases, so the
 * loop cannot wander, and the widget prints it at every step so that the claim
 * is on screen rather than in a footnote.
 *
 * The steps are run here from the supports the generator recorded, so the
 * ellipse drawn is the ellipse it measured.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { cStep, mahalanobis, averagePrecision } from './detkit.js';
import { drawPoints, legend, seriesLabel, accent } from './draw.js';

/* the same 2 by 2 ellipse the previous article drew, and for the same reason:
   it is the only thing any of these pictures needs */
function ellipsePath(mu, S, c, steps = 128) {
  const [a, b, d] = [S[0][0], S[0][1], S[1][1]];
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  let vx = 1;
  let vy = 0;
  if (Math.abs(b) > 1e-14) {
    vx = l1 - d;
    vy = b;
    const nrm = Math.hypot(vx, vy);
    vx /= nrm;
    vy /= nrm;
  }
  const r1 = Math.sqrt(Math.max(0, c * l1));
  const r2 = Math.sqrt(Math.max(0, c * l2));
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    const u = r1 * Math.cos(t);
    const v = r2 * Math.sin(t);
    pts.push([mu[0] + u * vx - v * vy, mu[1] + u * vy + v * vx]);
  }
  return pts;
}

export function initRobustWidget(data) {
  const S = data.sets['two densities'];
  const X = S.points;
  const y = S.labels;
  const STEPS = data.mcd.steps;
  const CHI = 5.991464547107979;            /* the 95% point of chi-squared on 2 */

  let idx = 0;
  const chart = makeChart('#mcd-chart', {
    width: 640, height: 470, margin: { top: 74, right: 26, bottom: 56, left: 46 },
  });
  const { g, w, h } = chart;
  const gShape = g.append('g');
  const gPts = g.append('g');
  const gNote = g.append('g');
  const readout = d3.select('#mcd-readout');
  const slider = d3.select('#mcd-step')
    .attr('min', 0).attr('max', STEPS.length - 1).attr('step', 1).property('value', 0);

  const xs = X.map((p) => p[0]);
  const ys = X.map((p) => p[1]);
  const x = d3.scaleLinear().domain([Math.min(...xs) - 0.6, Math.max(...xs) + 0.6]).range([0, w]);
  const yy = d3.scaleLinear().domain([Math.min(...ys) - 0.6, Math.max(...ys) + 0.6]).range([h, 0]);

  /* the full-sample ellipse, which is where the previous article stopped */
  const all = Array.from({ length: X.length }, (_, i) => i);
  const full = cStep(X, all);

  function render() {
    idx = +slider.property('value');
    const st = STEPS[idx];
    const r = cStep(X, st.support);
    const sup = new Set(st.support);

    drawAxes(g, x, yy, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, { x: 'first measurement', y: 'second' });

    gShape.selectAll('*').remove();
    const line = d3.line().x((p) => x(p[0])).y((p) => yy(p[1]));
    gShape.append('path')
      .attr('d', `${line(ellipsePath(full.mu, full.S, CHI))}Z`)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4').attr('opacity', 0.55);
    gShape.append('path')
      .attr('d', `${line(ellipsePath(r.mu, r.S, CHI))}Z`)
      .attr('fill', accent()).attr('fill-opacity', 0.1)
      .attr('stroke', accent()).attr('stroke-width', 2.2);

    drawPoints(gPts, X, y, x, yy, { r: 2.8, highlight: (i) => sup.has(i) });

    gNote.selectAll('*').remove();
    legend(gNote, [
      { label: 'planted anomaly', shape: 'ring', colour: 'var(--cosmos)' },
      { label: 'the rows it trusts', shape: 'dot', colour: 'var(--squidink)' },
      { label: 'the robust ellipse', shape: 'line', colour: accent() },
      { label: 'the whole-sample one', shape: 'dash', colour: 'var(--squidink)' },
    ], { x0: 2, y0: -50, gap: 178, cols: 2 });

    d3.select('#mcd-step-label').text(st.step);
    const d2 = mahalanobis(X, r.mu, r.S);
    const ap = averagePrecision(y, d2);
    const apFull = averagePrecision(y, full.d2);
    const prev = idx > 0 ? STEPS[idx - 1] : null;
    readout.html(
      `<span class="slider-label">concentration step <span class="value">${st.step}</span></span> `
      + `It trusts ${st.support.length} of the ${X.length} rows, the half whose distance to the `
      + `current fit is smallest. The determinant of the covariance is `
      + `<span class="bold">${r.det.toPrecision(6)}</span>, against `
      + `${full.det.toPrecision(6)} for the ellipse fitted to every row`
      + (prev
        ? ` and ${prev.det.toPrecision(6)} one step ago: it goes down at every step and never up, `
          + `which is the theorem the loop rests on. `
        : `: from here on it can only go down, which is the theorem the loop rests on. `)
      + (st.changed
        ? `${st.changed} row${st.changed === 1 ? '' : 's'} will change hands at the next step.`
        : `Nothing changes hands at the next step, so this is where it stops.`)
      + ` Average precision here is ${ap.toFixed(4)} against ${apFull.toFixed(4)} for the `
      + `whole-sample ellipse`
      + (ap > apFull + 0.005
        ? `: the repair is worth ${(ap - apFull).toFixed(4)} on this dataset.`
        : ap < apFull - 0.005
          ? `: <span class="bold">the repair costs ${(apFull - ap).toFixed(4)} here</span>, `
            + `because these are two clusters and no single ellipse, robust or otherwise, is the `
            + `right shape for them. The scoreboard below has the dataset where it pays.`
          : `, which is the same answer.`)
    );
  }

  slider.on('input', render);
  render();
  return { render };
}
