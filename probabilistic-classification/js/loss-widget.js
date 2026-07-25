/* Two objectives over the same two coefficients, drawn as contour maps.
 *
 * The claim being demonstrated is not aesthetic. Cross-entropy through a sigmoid
 * is convex in the coefficients, so every downhill path ends in the same place.
 * Squared error through a sigmoid is not: its gradient carries a p(1-p) factor
 * that vanishes wherever the model is confident, INCLUDING where it is
 * confidently wrong, so a descent started in the wrong corner has almost no
 * slope to follow and sits there.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { descend, lossAt } from './mathkit.js';

const B_LIM = 8;
const RES = 64;
const STEPS = 320;
const LR = 4;

export function initLossWidget(data) {
  /* One feature, standardised, so the two coefficients live on a comparable
   * scale and a square window is the right window. */
  const raw = data.one_d.points;
  const mx = d3.mean(raw, (d) => d.x);
  const sx = d3.deviation(raw, (d) => d.x);
  const xs = raw.map((d) => (d.x - mx) / sx);
  const ys = raw.map((d) => d.y);

  const { g, w, h } = makeChart('#loss-chart', {
    width: 560,
    height: 520,
    margin: { top: 34, right: 26, bottom: 58, left: 62 },
  });

  const x = d3.scaleLinear().domain([-B_LIM, B_LIM]).range([0, w]);
  const y = d3.scaleLinear().domain([-B_LIM, B_LIM]).range([h, 0]);
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'β₀  (intercept)', y: 'β₁  (slope)' });

  /* Sample both surfaces once. 64 x 64 x 398 sigmoids is about a tenth of a
   * second per surface and the values never change, so it is paid at load. */
  const grids = {};
  for (const kind of ['ce', 'mse']) {
    const vals = new Array(RES * RES);
    for (let j = 0; j < RES; j++) {
      const b1 = -B_LIM + (2 * B_LIM * j) / (RES - 1);
      for (let i = 0; i < RES; i++) {
        const b0 = -B_LIM + (2 * B_LIM * i) / (RES - 1);
        vals[j * RES + i] = lossAt(xs, ys, b0, b1, kind);
      }
    }
    grids[kind] = vals;
  }

  /* The reference optimum: a long cross-entropy run from the origin. Used to
   * say whether a given descent actually arrived, rather than eyeballing it. */
  const refPath = descend(xs, ys, { lr: LR, steps: 4000, loss: 'ce' });
  const ref = refPath[refPath.length - 1];
  const REF = { ce: lossAt(xs, ys, ref.b0, ref.b1, 'ce'), mse: lossAt(xs, ys, ref.b0, ref.b1, 'mse') };

  const surf = g.append('g');
  const pathG = g.append('g');
  const ballG = g.append('g');
  const optDot = g.append('circle').attr('r', 5.5)
    .attr('cx', x(ref.b0)).attr('cy', y(ref.b1))
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
  const optLabel = g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(ref.b0) + 10).attr('y', y(ref.b1) - 8)
    .style('font-size', '0.6rem').style('text-transform', 'none')
    .text('the answer');

  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const state = { kind: 'ce', start: { b0: -6, b1: -6 } };
  const statsEl = document.querySelector('#loss-stats');
  const btnCE = document.querySelector('#loss-ce');
  const btnMSE = document.querySelector('#loss-mse');

  /* The plot is not square, so a single uniform scale would stretch the
   * contours along one axis and they would stop lining up with the beta-1
   * ticks. Map each grid axis onto its own range, and flip y here rather than
   * mangling the coordinates by hand afterwards. */
  const geo = d3.geoPath(
    d3.geoTransform({
      point(a, b) {
        this.stream.point((a / (RES - 1)) * w, h - (b / (RES - 1)) * h);
      },
    })
  );

  function drawSurface() {
    const kind = state.kind;
    const vals = grids[kind];
    const ext = d3.extent(vals);
    /* Thresholds on a power scale: the interesting structure of both surfaces
     * is near the floor, and even steps would spend every contour on the walls. */
    const th = d3.range(0, 1.0001, 1 / 13).map((t) => ext[0] + (ext[1] - ext[0]) * t ** 2.1);
    const colour = d3.scaleSequential(d3.interpolateYlGnBu).domain([ext[1], ext[0]]);
    const contours = d3.contours().size([RES, RES]).thresholds(th)(vals);

    surf.selectAll('path').data(contours).join('path')
      .attr('d', geo)
      .attr('fill', (c) => colour(c.value))
      .attr('stroke', 'var(--white)')
      .attr('stroke-width', 0.35)
      .attr('opacity', 0.92);

    title.text(kind === 'ce'
      ? 'cross-entropy: one basin, and every path finds it'
      : 'squared error: a basin, and a plateau where the gradient dies');
    btnCE.classList.toggle('ghost', kind !== 'ce');
    btnMSE.classList.toggle('ghost', kind === 'ce');
  }

  let timer = null;
  function run() {
    if (timer) timer.stop();
    const kind = state.kind;
    const p = descend(xs, ys, { ...state.start, lr: LR, steps: STEPS, loss: kind });
    const line = d3.line().x((d) => x(d.b0)).y((d) => y(d.b1));

    const halo = pathG.selectAll('path.halo').data([p]).join('path').attr('class', 'halo')
      .attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 5.5);
    const trail = pathG.selectAll('path.trail').data([p]).join('path').attr('class', 'trail')
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.6);
    const ball = ballG.selectAll('circle').data([p[0]]).join('circle')
      .attr('r', 6).attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 2);

    const t0 = performance.now();
    const DUR = 1400;
    timer = d3.timer((elapsed) => {
      const k = Math.min(1, elapsed / DUR);
      const upto = Math.max(1, Math.round(k * STEPS));
      const seg = p.slice(0, upto + 1);
      halo.attr('d', line(seg));
      trail.attr('d', line(seg));
      ball.attr('cx', x(seg[seg.length - 1].b0)).attr('cy', y(seg[seg.length - 1].b1));
      if (k >= 1) {
        timer.stop();
        timer = null;
      }
    });
    void t0;

    const end = p[p.length - 1];
    const lossEnd = lossAt(xs, ys, end.b0, end.b1, kind);
    const lossStart = lossAt(xs, ys, state.start.b0, state.start.b1, kind);
    const gap = lossEnd - REF[kind];
    const moved = Math.hypot(end.b0 - state.start.b0, end.b1 - state.start.b1);

    /* The verdict is the measured distance to the reference optimum on the
     * SAME surface, not a guess from the picture. A run that ends 0.001 above
     * the floor arrived; one that ends a whole nat above it did not. */
    const arrived = gap < 0.02 * Math.max(REF[kind], 0.02) + 0.005;
    statsEl.innerHTML =
      `<table class="pc-table">
         <tr><th></th><th>start</th><th>after ${STEPS} steps</th></tr>
         <tr><td>β₀, β₁</td><td>${state.start.b0.toFixed(2)}, ${state.start.b1.toFixed(2)}</td>` +
      `<td>${end.b0.toFixed(2)}, ${end.b1.toFixed(2)}</td></tr>
         <tr><td>${kind === 'ce' ? 'cross-entropy' : 'squared error'}</td>` +
      `<td>${lossStart.toFixed(4)}</td><td>${lossEnd.toFixed(4)}</td></tr>
         <tr><td>best possible</td><td colspan="2">${REF[kind].toFixed(4)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.7rem;line-height:1.45">${
        arrived
          ? `Arrived: within ${gap.toFixed(4)} of the floor, having travelled ${moved.toFixed(1)} units.`
          : `Stalled ${gap.toFixed(4)} above the floor after moving only ${moved.toFixed(2)} units. ` +
            'The gradient here is a product with p(1−p) in it, and a confidently wrong model has p(1−p) near zero.'
      }</div>`;
  }

  btnCE.addEventListener('click', () => {
    state.kind = 'ce';
    drawSurface();
    run();
  });
  btnMSE.addEventListener('click', () => {
    state.kind = 'mse';
    drawSurface();
    run();
  });
  document.querySelector('#loss-run').addEventListener('click', () => {
    state.start = { b0: 0, b1: 0 };
    run();
  });
  document.querySelector('#loss-corner').addEventListener('click', () => {
    state.start = { b0: -6, b1: -6 };
    run();
  });

  d3.select('#loss-chart svg').style('cursor', 'crosshair').on('click', (event) => {
    const [px, py] = d3.pointer(event, g.node());
    if (px < 0 || px > w || py < 0 || py > h) return;
    state.start = { b0: x.invert(px), b1: y.invert(py) };
    run();
  });

  drawSurface();
  run();
}
