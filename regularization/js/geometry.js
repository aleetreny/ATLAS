/* THE picture: why L1 produces exact zeros and L2 does not.
 *
 * Two coefficients, so the whole optimization fits on a plane. Elliptical
 * contours are the squared error; the shaded region is the budget the penalty
 * allows. The solution is where the smallest affordable ellipse touches the
 * budget. A diamond has corners, and corners sit on the axes, which is the
 * entire reason Lasso deletes features.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { makeModel, eig2 } from './mathkit.js';

const FEAT_A = 2; // bmi
const FEAT_B = 3; // blood pressure

export function initGeometry(dia) {
  const nameA = dia.labels[FEAT_A];
  const nameB = dia.labels[FEAT_B];

  const X = dia.X.map((r) => [r[FEAT_A], r[FEAT_B]]);
  const ySd = Math.sqrt(d3.mean(dia.y, (v) => v * v));
  const y = dia.y.map((v) => v / ySd); // standardize the target: readable coefficients
  const model = makeModel(X, y);
  const bOls = model.fit(1e-12, 0);

  const { g, w, h } = makeChart('#geometry-chart', {
    width: 560,
    height: 560,
    margin: { top: 30, right: 30, bottom: 62, left: 62 },
  });

  // The L1 budget reaches |b1| + |b2| along each axis, which is always further
  // out than the solution itself, so the view has to be wide enough to contain
  // the whole diamond at the loosest penalty. At 0.46 it was clipped into an
  // octagon and the corners, the entire point of the picture, were off screen.
  const LIM = 0.82;
  const x = d3.scaleLinear().domain([-LIM, LIM]).range([0, w]);
  const yS = d3.scaleLinear().domain([-LIM, LIM]).range([h, 0]);
  const pxPerUnit = (x(1) - x(0));

  // axes through the origin, not around the edge: this is coefficient space
  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', yS(0)).attr('y2', yS(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('opacity', 0.5);
  g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('opacity', 0.5);

  drawAxes(g, x, yS, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: `coefficient of ${nameA}`, y: `coefficient of ${nameB}` });

  const region = g.append('path').attr('fill', 'var(--primary)').attr('fill-opacity', 0.11)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.5);

  const contoursG = g.append('g');
  const pathTrace = g.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 2).attr('stroke-dasharray', '4 4').attr('opacity', 0.5);

  const olsDot = g.append('g');
  olsDot.append('circle').attr('cx', x(bOls[0])).attr('cy', yS(bOls[1])).attr('r', 6)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.5);
  olsDot.append('text').attr('class', 'chart-annotation')
    .attr('x', x(bOls[0]) + 10).attr('y', yS(bOls[1]) - 10)
    .style('font-size', '0.7rem').text('OLS');

  const solDot = g.append('circle').attr('r', 8)
    .attr('fill', 'var(--primary)').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.5);
  const killNote = g.append('text').attr('class', 'chart-annotation').attr('text-anchor', 'middle');

  const A = model.gram();
  const { l1: lam1, l2: lam2, angle } = eig2(A);
  const quad = (b) => {
    const d0 = b[0] - bOls[0];
    const d1 = b[1] - bOls[1];
    return A[0][0] * d0 * d0 + 2 * A[0][1] * d0 * d1 + A[1][1] * d1 * d1;
  };

  /* One error contour: the set of coefficient pairs with identical training error. */
  function ellipse(k, opts = {}) {
    return {
      rx: Math.sqrt(k / lam1) * pxPerUnit,
      ry: Math.sqrt(k / lam2) * pxPerUnit,
      rot: (-angle * 180) / Math.PI, // screen y is flipped
      ...opts,
    };
  }

  function drawContours(kSolution) {
    const levels = [0.25, 0.5, 1, 1.75, 2.75].map((m) => kSolution * m);
    contoursG
      .selectAll('ellipse.bg')
      .data(levels)
      .join('ellipse')
      .attr('class', 'bg')
      .attr('cx', x(bOls[0]))
      .attr('cy', yS(bOls[1]))
      .attr('transform', `rotate(${(-angle * 180) / Math.PI},${x(bOls[0])},${yS(bOls[1])})`)
      .attr('rx', (k) => ellipse(k).rx)
      .attr('ry', (k) => ellipse(k).ry)
      .attr('fill', 'none')
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 1)
      .attr('opacity', 0.16);

    // the contour that actually touches the budget: the solution's error level
    contoursG
      .selectAll('ellipse.hit')
      .data([kSolution])
      .join('ellipse')
      .attr('class', 'hit')
      .attr('cx', x(bOls[0]))
      .attr('cy', yS(bOls[1]))
      .attr('transform', `rotate(${(-angle * 180) / Math.PI},${x(bOls[0])},${yS(bOls[1])})`)
      .attr('rx', (k) => ellipse(k).rx)
      .attr('ry', (k) => ellipse(k).ry)
      .attr('fill', 'none')
      .attr('stroke', 'var(--cosmos)')
      .attr('stroke-width', 2.5);
  }

  /* Budget boundary passing exactly through the solution. */
  function budgetPath(b, l1Ratio) {
    const raw =
      l1Ratio * (Math.abs(b[0]) + Math.abs(b[1])) +
      ((1 - l1Ratio) / 2) * (b[0] * b[0] + b[1] * b[1]);
    /* Once the penalty kills both coefficients the budget is exactly zero and
     * the region collapses to a point, so the shape the whole section is about
     * disappears at the very moment it has won the argument. Keep a sliver. */
    const t = Math.max(raw, 1e-3);
    const pts = [];
    for (let i = 0; i <= 240; i++) {
      const phi = (i / 240) * 2 * Math.PI;
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      const qa = (1 - l1Ratio) / 2;
      const qb = l1Ratio * (Math.abs(c) + Math.abs(s));
      let r;
      if (qa < 1e-12) r = t / qb;
      else r = (-qb + Math.sqrt(qb * qb + 4 * qa * t)) / (2 * qa);
      pts.push([x(r * c), yS(r * s)]);
    }
    return `M${pts.map((p) => p.join(',')).join('L')}Z`;
  }

  const slider = document.querySelector('#geo-alpha');
  const alphaLabel = document.querySelector('#geo-alpha-label');
  const statsEl = document.querySelector('#geo-stats');
  const buttons = {
    1: document.querySelector('#geo-l1'),
    0: document.querySelector('#geo-l2'),
    0.5: document.querySelector('#geo-en'),
  };
  const state = { l1Ratio: 1 };

  function drawTrace() {
    const alphas = d3.range(-3.2, 0.4, 0.02).map((v) => 10 ** v);
    const pts = alphas.map((a) => {
      const b = model.fit(a, state.l1Ratio);
      return [x(b[0]), yS(b[1])];
    });
    pathTrace.attr('d', `M${pts.map((p) => p.join(',')).join('L')}`);
  }

  function render() {
    const alpha = 10 ** +slider.value;
    alphaLabel.textContent = alpha.toPrecision(2);
    const b = model.fit(alpha, state.l1Ratio);

    region.attr('d', budgetPath(b, state.l1Ratio));
    drawContours(Math.max(quad(b), 1e-9));
    solDot.attr('cx', x(b[0])).attr('cy', yS(b[1]));

    const zeroA = Math.abs(b[0]) < 1e-9;
    const zeroB = Math.abs(b[1]) < 1e-9;
    if (zeroA || zeroB) {
      /* Both can go at once, and naming only the first one made the widget look
       * like it had missed half of what it had just done. */
      const gone = [zeroA && nameA, zeroB && nameB].filter(Boolean);
      killNote
        .attr('x', x(b[0]))
        .attr('y', yS(b[1]) + (zeroB ? 30 : 0))
        .attr('opacity', 1)
        .text(`${gone.join(' and ')} eliminated`);
    } else {
      killNote.attr('opacity', 0);
    }

    const kind = state.l1Ratio === 1 ? 'lasso (L1)' : state.l1Ratio === 0 ? 'ridge (L2)' : 'elasticnet';
    const shape = state.l1Ratio === 1 ? 'a diamond, corners on the axes' : state.l1Ratio === 0 ? 'a circle, no corners anywhere' : 'a rounded diamond, corners survive';
    statsEl.innerHTML =
      `<div class="slider-label" style="margin-bottom:.4rem"><span class="bold">${kind}</span>: budget is ${shape}.</div>` +
      `<div class="slider-label">${nameA} = <span class="value">${b[0].toFixed(3)}</span>` +
      `${zeroA ? ' <span class="bold" style="color:var(--cosmos)">← exactly zero</span>' : ''}</div>` +
      `<div class="slider-label">${nameB} = <span class="value">${b[1].toFixed(3)}</span>` +
      `${zeroB ? ' <span class="bold" style="color:var(--cosmos)">← exactly zero</span>' : ''}</div>`;
  }

  for (const [ratio, btn] of Object.entries(buttons)) {
    btn.addEventListener('click', () => {
      state.l1Ratio = +ratio;
      for (const [r2, b2] of Object.entries(buttons)) b2.classList.toggle('ghost', +r2 !== +ratio);
      drawTrace();
      render();
    });
  }
  slider.addEventListener('input', render);

  drawTrace();
  render();
}
