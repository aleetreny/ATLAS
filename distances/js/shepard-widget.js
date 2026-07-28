/* Metric against non-metric scaling, under a distortion that bends every
 * distance and moves none of their order.
 *
 * Two hundred SMACOF iterations on 178 points, twice, run here. The starting
 * configuration is the one the generator used, shipped in the file, so the two
 * runs are the same deterministic trajectory and can be compared to five
 * decimals rather than to within a shrug. Each exponent is cached after its
 * first visit, which is what makes the slider feel like a slider.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { smacof, upperOrder, pairwise, procrustes, stress1, toRows, zeros } from '../../assets/js/embed.js';

const CULTIVAR = ['var(--primary)', 'var(--aqua)', 'var(--smile)'];

/* The truth's own distance matrix, normalised so the largest is 1, which is
 * what makes raising it to a power a distortion rather than a rescaling. */
function targetMatrix(Dt, gamma) {
  const n = Dt.length;
  const D = zeros(n, n);
  let max = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = Dt[i][j] ** gamma;
      D[i][j] = v;
      D[j][i] = v;
      if (v > max) max = v;
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i][j] /= max;
  return D;
}

export function scalingRun(data, gamma, base) {
  const truth = base.truth;
  const init = base.init;
  const D = targetMatrix(base.Dtruth, gamma);
  const order = base.order;
  const iters = data.scaling.iters;
  const m = smacof(D, init, iters, { trace: gamma === 1 });
  const nm = smacof(D, init, iters, { nonmetric: true, order });
  return {
    D,
    metric: m.coords,
    nonmetric: nm.coords,
    metricStress: stress1(D, m.coords),
    nonmetricStress: stress1(nm.target, nm.coords),
    metricDisparity: procrustes(truth, m.coords).disparity,
    nonmetricDisparity: procrustes(truth, nm.coords).disparity,
    metricAligned: procrustes(truth, m.coords),
    nonmetricAligned: procrustes(truth, nm.coords),
    hat: nm.target,
  };
}

export function scalingBase(data) {
  const truth = toRows(data.scaling.truth);
  const Dt0 = pairwise(truth);
  let max = 0;
  for (let i = 0; i < Dt0.length; i++) for (let j = 0; j < Dt0.length; j++) max = Math.max(max, Dt0[i][j]);
  const Dtruth = Dt0.map((r) => Float64Array.from(r, (v) => v / max));
  return {
    truth,
    init: toRows(data.scaling.init),
    Dtruth,
    order: upperOrder(Dtruth),
  };
}

export function initShepardWidget(data, base, cache) {
  const gammas = data.scaling.gammas;
  const classes = data.wine.classes;
  const chart = makeChart('#shepard-chart', {
    width: 760, height: 400, margin: { top: 60, right: 22, bottom: 56, left: 60 },
  });
  const { g, w, h } = chart;
  const viewButtons = document.querySelector('#shepard-view-buttons');
  const slider = document.querySelector('#shepard-gamma');
  const gLabel = document.querySelector('#shepard-gamma-label');
  const readout = document.querySelector('#shepard-readout');
  const state = { view: 'maps', i: 4 };

  const btns = {};
  for (const [key, text] of [['maps', 'the two maps'], ['shepard', 'what each one is fitting']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === state.view ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state.view = key;
      render();
    });
    viewButtons.appendChild(b);
    btns[key] = b;
  }

  const panelA = g.append('g');
  const panelB = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  /* One panel of the two-map view: the recovered configuration after Procrustes
     has been allowed to rotate, reflect and rescale it, against faint rings for
     the truth. Without that alignment the reader would be asked to compare two
     pictures in different orientations, which is not the comparison. */
  function drawMap(panel, x0, width, aligned, label, colourByClass) {
    panel.selectAll('*').remove();
    const A = aligned.reference;
    const Bc = aligned.coords;
    const all = A.concat(Bc);
    const ex = d3.extent(all.map((d) => d[0]));
    const ey = d3.extent(all.map((d) => d[1]));
    const span = Math.max(ex[1] - ex[0], ey[1] - ey[0]) * 1.1 || 1;
    const cx = (ex[0] + ex[1]) / 2;
    const cy = (ey[0] + ey[1]) / 2;
    const unit = Math.min(width, h - 30) / span;
    const x = d3.scaleLinear().domain([cx - span / 2, cx + span / 2])
      .range([x0 + (width - span * unit) / 2, x0 + (width + span * unit) / 2]);
    const y = d3.scaleLinear().domain([cy - span / 2, cy + span / 2])
      .range([(h - 30 + span * unit) / 2 + 20, (h - 30 - span * unit) / 2 + 20]);
    panel.append('rect').attr('x', x0).attr('y', 6).attr('width', width).attr('height', h - 6)
      .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 1);
    panel.selectAll('circle.truth').data(A).join('circle').attr('class', 'truth')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 5.5)
      .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 1.2);
    panel.selectAll('circle.got').data(Bc).join('circle').attr('class', 'got')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.2)
      .attr('fill', (d, i) => (colourByClass ? CULTIVAR[classes[i]] : 'var(--squidink)'))
      .attr('opacity', 0.85);
    panel.append('text').attr('x', x0 + width / 2).attr('y', 0)
      .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
      .style('font-size', '12px').style('text-transform', 'none').text(label);
  }

  function render() {
    const gamma = gammas[state.i];
    if (!cache[gamma]) cache[gamma] = scalingRun(data, gamma, base);
    const R = cache[gamma];
    gLabel.textContent = String(gamma);
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== state.view);

    if (state.view === 'maps') {
      panelB.selectAll('*').remove();
      g.selectAll('g.axis, g.grid, text.axis-label').remove();
      const gap = 18;
      const half = (w - gap) / 2;
      drawMap(panelA.selectAll('g.left').data([0]).join('g').attr('class', 'left'),
        0, half, R.metricAligned, 'metric MDS', true);
      drawMap(panelA.selectAll('g.right').data([0]).join('g').attr('class', 'right'),
        half + gap, half, R.nonmetricAligned, 'non-metric MDS', true);
      wrapLabel(title, 'grey rings: where the bottles really are. dots: where each method put them', w - 8);
    } else {
      panelA.selectAll('*').remove();
      /* Shepard diagram: the dissimilarity handed in against the distance the
         map produced, plus the staircase the non-metric run is really chasing. */
      const n = R.D.length;
      const pts = [];
      const dM = pairwise(R.metric);
      const dN = pairwise(R.nonmetric);
      /* A non-metric solution has no scale: the disparities are renormalised
         to the current distances at every iteration, so the configuration can
         end up any size at all and here it ends up tiny. Its SHAPE is the whole
         content of the diagram, so the non-metric distances and their staircase
         are multiplied by one common factor that lines its largest distance up
         with the metric run's. Nothing about the relationship changes; without
         it, half the picture sits flat on the axis. */
      let maxM = 0;
      let maxN = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (dM[i][j] > maxM) maxM = dM[i][j];
          if (dN[i][j] > maxN) maxN = dN[i][j];
        }
      }
      const sN = maxN > 1e-12 ? maxM / maxN : 1;
      const step = 9;                       // every ninth pair: 1,751 dots, not 15,753
      let t = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (t++ % step) continue;
          pts.push({ d: R.D[i][j], m: dM[i][j], nm: dN[i][j] * sN, hat: R.hat[i][j] * sN });
        }
      }
      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const ymax = d3.max(pts, (p) => Math.max(p.m, p.nm, p.hat)) * 1.06;
      const y = d3.scaleLinear().domain([0, ymax]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      axisLabels(g, w, h, { x: 'dissimilarity handed in', y: 'distance in the map' });
      panelB.selectAll('circle.m').data(pts).join('circle').attr('class', 'm')
        .attr('cx', (p) => x(p.d)).attr('cy', (p) => y(p.m)).attr('r', 1.7)
        .attr('fill', 'var(--anchor)').attr('opacity', 0.4);
      /* The staircase goes down FIRST and the non-metric dots on top of it. The
         two coincide almost exactly, which is the finding, and with the
         staircase painted last the purple series the legend promises was simply
         not on the screen. */
      const stair = pts.slice().sort((a, b) => a.d - b.d);
      panelB.selectAll('path.stair').data([stair]).join('path').attr('class', 'stair')
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 3.2)
        .attr('d', d3.line().curve(d3.curveStepAfter).x((p) => x(p.d)).y((p) => y(p.hat)));
      panelB.selectAll('circle.n').data(pts).join('circle').attr('class', 'n')
        .attr('cx', (p) => x(p.d)).attr('cy', (p) => y(p.nm)).attr('r', 1.5)
        .attr('fill', 'var(--primary)').attr('opacity', 0.85);
      /* The line y = x, and it has to END at the same number in both scales:
         drawing to (1, ymax) instead of (e, e) tilted it and made a perfect
         metric fit look like a systematic overshoot. */
      const e = Math.min(1, ymax);
      panelB.selectAll('path.diag').data([0]).join('path').attr('class', 'diag')
        .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 4')
        .attr('d', `M${x(0)},${y(0)}L${x(e)},${y(e)}`);
      const key = panelB.selectAll('g.key').data([0]).join('g').attr('class', 'key');
      key.selectAll('*').remove();
      /* A panel behind it: whichever corner the legend sits in, one of the two
         distortion directions puts a curve through it. */
      key.append('rect').attr('x', 0).attr('y', 2).attr('width', 296).attr('height', 52)
        .attr('fill', 'var(--white)').attr('opacity', 0.88);
      [['var(--anchor)', 'metric: chases the dashed line'],
        ['var(--primary)', 'non-metric, rescaled: sits ON the staircase'],
        ['var(--cosmos)', 'the staircase, refitted every iteration']].forEach(([c, txt], i) => {
        key.append('circle').attr('cx', 12).attr('cy', 12 + i * 15).attr('r', 4).attr('fill', c);
        key.append('text').attr('x', 22).attr('y', 16 + i * 15)
          .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
          .attr('fill', 'var(--squidink)').text(txt);
      });
      wrapLabel(title, 'a perfect method would put every dot on a single rising curve', w - 8);
    }

    const ref = data.scaling.rows.find((r) => r.gamma === gamma);
    const worse = R.metricDisparity > R.nonmetricDisparity;
    readout.innerHTML = `
      <table class="dist-table">
        <tr><th>method</th><th>stress</th><th>distance from the truth</th><th>what it is fitting</th></tr>
        <tr><td>metric</td>
          <td><span class="value">${R.metricStress.toFixed(5)}</span></td>
          <td><span class="value">${R.metricDisparity.toFixed(5)}</span></td>
          <td>the numbers themselves</td></tr>
        <tr><td>non-metric</td>
          <td><span class="value">${R.nonmetricStress.toFixed(5)}</span></td>
          <td><span class="value">${R.nonmetricDisparity.toFixed(5)}</span></td>
          <td>their order alone</td></tr>
      </table>
      <div class="dist-note">
        ${gamma === 1
    ? `At an exponent of 1 nothing has been distorted, and metric scaling recovers the configuration `
          + `exactly: stress ${R.metricStress.toFixed(5)}, and a distance from the truth of `
          + `${R.metricDisparity.toFixed(5)}. Move the slider and only one of the two rows changes.`
    : `Every distance has been raised to the power ${gamma}, which ${gamma > 1 ? 'stretches the big '
          + 'gaps and squashes the small ones' : 'does the opposite'} and leaves every pair in the same `
          + `order as before. Metric scaling is now ${R.metricDisparity.toFixed(5)} from the truth, `
          + `${worse ? 'against' : 'and non-metric'} ${R.nonmetricDisparity.toFixed(5)} for non-metric. `
          + `${R.nonmetricDisparity < 1e-6
    ? 'The non-metric number is not small, it is unchanged: reading only the ordering means the '
              + 'distortion was never visible to it in the first place.'
    : 'The non-metric run has moved too, which is worth knowing about.'}`}
        ${ref ? '' : ''}
      </div>`;
  }

  slider.addEventListener('input', () => {
    state.i = +slider.value;
    render();
  });
  render();
}
