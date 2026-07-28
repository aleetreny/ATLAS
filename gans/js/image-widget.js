/* The image half: what the metric is worth before anything is judged by it,
 * what five objectives produce, and whether the number you can watch while
 * training tells you anything about the number you care about.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { drawTiles } from './panels.js';

export const ARM_COLOUR = {
  dcgan: 'var(--primary)',
  sat: 'var(--cosmos)',
  'wgan-clip': 'var(--smile)',
  'wgan-gp': 'var(--anchor)',
  mlp: 'var(--aqua)',
  real: 'var(--squidink)',
};
export const ARM_NAME = {
  dcgan: 'DCGAN',
  sat: 'saturating loss',
  'wgan-clip': 'WGAN, clipped',
  'wgan-gp': 'WGAN-GP',
  mlp: 'no convolutions',
  real: 'real MNIST',
};

const f2 = (v) => v.toFixed(2);

/* ------------------------------------------------------------------------ */
export function initMetricWidget(data) {
  const C = data.metric;
  const c = makeChart('#metric-chart', {
    width: 660, height: 340, margin: { top: 36, right: 30, bottom: 56, left: 74 },
  });
  const rows = C.blur;
  const marks = [
    { label: 'the floor: real against real', v: C.floor_train, colour: 'var(--squidink)' },
    { label: 'a whole class missing', v: C.class_drop, colour: 'var(--cosmos)' },
    { label: 'pixels shuffled', v: C.shuffled, colour: 'var(--galaxy)' },
    { label: 'uniform noise', v: C.noise, colour: 'var(--anchor)' },
  ];
  /* Three orders of magnitude between the floor and the ceiling, so a linear
     axis puts the floor and the dropped class on the same pixel and their two
     labels on top of each other. */
  const top = Math.max(d3.max(rows, (r) => r.fd), d3.max(marks, (m) => m.v)) * 1.6;
  const bottom = Math.min(d3.min(rows, (r) => r.fd), d3.min(marks, (m) => m.v)) * 0.7;
  const x = d3.scaleLinear().domain([1, d3.max(rows, (r) => r.k)]).range([0, c.w]);
  const y = d3.scaleLog().domain([Math.max(bottom, 1e-3), top]).range([c.h, 0]);
  const yv = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
  drawGrid(c.g, x, y, c.w, c.h, { xValues: rows.map((r) => r.k), yValues: yv });
  drawAxes(c.g, x, y, c.w, c.h, {
    xValues: rows.map((r) => r.k), yValues: yv, yFmt: (v) => (v < 1 ? v.toFixed(2) : d3.format('~s')(v)),
  });
  axisLabels(c.g, c.w, c.h, { x: 'width of a box blur, in pixels', y: 'distance from real data' });
  c.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4)
    .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r.fd))(rows));
  c.g.selectAll('circle.pt').data(rows).enter().append('circle').attr('class', 'pt')
    .attr('cx', (r) => x(r.k)).attr('cy', (r) => y(r.fd)).attr('r', 3.5)
    .attr('fill', 'var(--primary)');
  marks.forEach((m, i) => {
    c.g.append('line').attr('x1', 0).attr('x2', c.w).attr('y1', y(m.v)).attr('y2', y(m.v))
      .attr('stroke', m.colour).attr('stroke-width', 1.2).attr('stroke-dasharray', '4 4');
    c.g.append('text').attr('class', 'chart-annotation')
      .attr('x', 6).attr('y', y(m.v) - 5).style('font-size', '11px')
      .style('letter-spacing', '0.4px').style('fill', m.colour)
      .text(`${m.label}: ${f2(m.v)}`);
  });

  document.querySelector('#metric-readout').innerHTML =
    `<p>The yardstick is a small convolutional classifier trained on the real digits, which gets `
    + `<span class="bold">${(C.probe_acc * 100).toFixed(2)}%</span> on the MNIST test set. Its `
    + `sixty-four dimensional penultimate layer is the space distances are measured in, and its `
    + `predictions are used to count which digits a generator is producing. It is not Inception `
    + `and this is not FID; naming it that would be borrowing authority from a network that is `
    + `not in the room.</p>`
    + `<table class="atlas-table"><thead><tr><th>Two sets compared</th><th>Distance</th>`
    + `<th>What it establishes</th></tr></thead><tbody>`
    + `<tr><td>${C.n.toLocaleString('en-US')} real against ${C.n.toLocaleString('en-US')} other real</td>`
    + `<td>${f2(C.floor_train)}</td><td>the floor: this is what identical looks like</td></tr>`
    + `<tr><td>real train against real test</td><td>${f2(C.floor_test)}</td>`
    + `<td>a different sample of the same thing scores the same</td></tr>`
    + `<tr><td>real against real with three digits deleted</td><td>${f2(C.class_drop)}</td>`
    + `<td>it sees mode dropping, which is the failure this article is about</td></tr>`
    + `<tr><td>real against the same images blurred five pixels</td>`
    + `<td>${f2(C.blur[C.blur.length - 2].fd)}</td><td>it sees smearing</td></tr>`
    + `<tr><td>real against uniform noise</td><td>${f2(C.noise)}</td><td>the ceiling</td></tr>`
    + `</tbody></table>`;
}

/* ------------------------------------------------------------------------ */
export function initSampleWidget(data, sheets) {
  const M = data.meta;
  const order = M.sheet_order;
  const state = { arm: 'dcgan' };
  const hist = makeChart('#sample-hist', {
    width: 420, height: 300, margin: { top: 30, right: 20, bottom: 56, left: 58 },
  });

  function render() {
    const idx = order.indexOf(state.arm);
    const tiles = sheets.samples.tiles.slice(idx * 32, idx * 32 + 32);
    drawTiles(document.querySelector('#sample-sheet'), tiles, M.tile, 16, { zoom: 2, gap: 2 });

    const real = data.metric.real_hist;
    const h = state.arm === 'real' ? real : data.runs[`${state.arm}/1`].final.hist;
    hist.g.selectAll('*').remove();
    const x = d3.scaleBand().domain(d3.range(10)).range([0, hist.w]).padding(0.18);
    const y = d3.scaleLinear().domain([0, 0.35]).range([hist.h, 0]).clamp(true);
    drawAxes(hist.g, x, y, hist.w, hist.h, { yTicks: 5, yFmt: d3.format('.0%') });
    axisLabels(hist.g, hist.w, hist.h, { x: 'digit the classifier reads', y: 'share of samples' });
    hist.g.selectAll('rect').data(h).enter().append('rect')
      .attr('x', (d, i) => x(i)).attr('y', (d) => y(d))
      .attr('width', x.bandwidth()).attr('height', (d) => hist.h - y(d))
      .attr('fill', ARM_COLOUR[state.arm]);
    hist.g.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3')
      .attr('d', d3.line().x((d, i) => x(i) + x.bandwidth() / 2).y((d) => y(d))(real));
    hist.g.append('text').attr('class', 'chart-annotation').attr('x', 2).attr('y', -10)
      .style('font-size', '11px').style('letter-spacing', '0.4px')
      .text('dashed: what the classifier reads on real digits');

    if (state.arm === 'real') {
      document.querySelector('#sample-readout').innerHTML =
        `<p>Thirty-two real MNIST digits, drawn from the same twenty thousand every generator on `
        + `this page was fitted to. This is the row to keep looking back at.</p>`;
    } else {
      const a = data.runs[`${state.arm}/1`].final;
      const b = data.runs[`${state.arm}/2`].final;
      const spread = Math.abs(a.fd - b.fd);
      document.querySelector('#sample-readout').innerHTML =
        `<p><span class="bold">${ARM_NAME[state.arm]}</span>, after `
        + `${M.steps.toLocaleString('en-US')} generator updates. Distance from real: `
        + `<span class="bold">${f2(a.fd)}</span> on this seed and ${f2(b.fd)} on the other, so `
        + `anything smaller than <span class="bold">${f2(spread)}</span> on this arm is not a `
        + `difference. Precision ${a.precision.toFixed(3)}, recall ${a.recall.toFixed(3)}: how `
        + `much of what it makes looks real, and how much of what is real it can make. It puts `
        + `${(Math.max(...a.hist) * 100).toFixed(1)}% of its samples on one digit where the real `
        + `data's busiest is ${(Math.max(...data.metric.real_hist) * 100).toFixed(1)}%, and the `
        + `total mismatch across the ten digits is ${(a.tv * 100).toFixed(1)}%.</p>`;
    }
  }

  const row = document.querySelector('#sample-buttons');
  const buttons = order.map((arm) => {
    const b = document.createElement('button');
    b.className = arm === state.arm ? 'atlas-btn' : 'atlas-btn ghost';
    b.textContent = ARM_NAME[arm] || arm;
    b.addEventListener('click', () => {
      state.arm = arm;
      buttons.forEach((o, i) => {
        o.className = order[i] === arm ? 'atlas-btn' : 'atlas-btn ghost';
      });
      render();
    });
    row.appendChild(b);
    return b;
  });
  render();
  return { state, render };
}

/* ------------------------------------------------------------------------ */
export function initCurveWidget(data) {
  const M = data.meta;
  const c = makeChart('#curve-chart', {
    width: 700, height: 400, margin: { top: 34, right: 132, bottom: 56, left: 70 },
  });
  const all = [];
  M.arms.forEach((arm) => M.seeds.forEach((s) => data.runs[`${arm}/${s}`].curve
    .forEach((p) => all.push(p))));
  const x = d3.scaleLinear().domain([0, M.steps]).range([0, c.w]);
  /* the floor is a reference the reader is asked to compare against, so the
     domain has to reach it: drawn outside the domain its label fell out of the
     canvas entirely */
  const y = d3.scaleLog().domain([
    Math.max(0.02, Math.min(d3.min(all, (p) => p.fd), data.metric.floor_train) * 0.8),
    d3.max(all, (p) => p.fd) * 1.15]).range([c.h, 0]);
  const yv = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 5, yValues: yv });
  drawAxes(c.g, x, y, c.w, c.h, { xTicks: 5, yValues: yv, yFmt: (v) => (v < 1 ? v.toFixed(2) : v) });
  axisLabels(c.g, c.w, c.h, { x: 'generator updates', y: 'distance from real data' });
  c.g.append('line').attr('x1', 0).attr('x2', c.w)
    .attr('y1', y(data.metric.floor_train)).attr('y2', y(data.metric.floor_train))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('stroke-dasharray', '4 4');
  c.g.append('text').attr('class', 'chart-annotation').attr('x', 4)
    .attr('y', y(data.metric.floor_train) - 5).style('font-size', '11px')
    .style('letter-spacing', '0.4px').text('real against real');

  const labels = [];
  M.arms.forEach((arm) => {
    M.seeds.forEach((s, i) => {
      const rows = data.runs[`${arm}/${s}`].curve;
      c.g.append('path').attr('fill', 'none').attr('stroke', ARM_COLOUR[arm])
        .attr('stroke-width', i === 0 ? 2.2 : 1.2).attr('stroke-opacity', i === 0 ? 1 : 0.6)
        .attr('d', d3.line().x((p) => x(p.it)).y((p) => y(p.fd))(rows));
      if (i === 0) labels.push({ arm, y: y(rows[rows.length - 1].fd) });
    });
  });
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i - 1].y < 15) labels[i].y = labels[i - 1].y + 15;
  }
  labels.forEach((l) => {
    c.g.append('text').attr('class', 'chart-annotation').attr('x', c.w + 6).attr('y', l.y + 4)
      .style('font-size', '11.5px').style('letter-spacing', '0.4px')
      .style('fill', ARM_COLOUR[l.arm]).text(ARM_NAME[l.arm]);
  });
}

/* The same correlation over the second half of training only.
 *
 * The whole-run number is confounded by the critic's own warm-up: at step one
 * it has not learned anything, so its estimate is near zero while the samples
 * are at their worst, and that single corner can flip the sign of the fit for
 * a reason that has nothing to do with the claim being tested. */
export function lateR(data, arm) {
  const xs = [];
  const ys = [];
  data.meta.seeds.forEach((s) => data.runs[`${arm}/${s}`].curve.forEach((p) => {
    if (p.it <= data.meta.steps / 2) return;
    xs.push(p.d_stat);
    ys.push(p.fd);
  }));
  if (xs.length < 4) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  xs.forEach((v, i) => {
    sxy += (v - mx) * (ys[i] - my);
    sxx += (v - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  });
  if (sxx <= 0 || syy <= 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}

/* ------------------------------------------------------------------------ */
export function initTrackWidget(data) {
  const M = data.meta;
  const state = { arm: 'dcgan' };
  const c = makeChart('#track-chart', {
    width: 560, height: 360, margin: { top: 34, right: 24, bottom: 58, left: 74 },
  });

  function render() {
    const arm = state.arm;
    const pts = [];
    M.seeds.forEach((s) => data.runs[`${arm}/${s}`].curve.forEach((p) => pts.push(p)));
    c.g.selectAll('*').remove();
    const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p.d_stat)).nice().range([0, c.w]);
    const y = d3.scaleLinear().domain([0, d3.max(pts, (p) => p.fd) * 1.1]).range([c.h, 0]);
    drawGrid(c.g, x, y, c.w, c.h, { xTicks: 5, yTicks: 5 });
    drawAxes(c.g, x, y, c.w, c.h, { xTicks: 5, yTicks: 5 });
    axisLabels(c.g, c.w, c.h, {
      x: arm.startsWith('wgan') ? "the critic's estimate of the distance"
        : "the discriminator's loss",
      y: 'actual distance from real data',
    });
    c.g.selectAll('circle').data(pts).enter().append('circle')
      .attr('cx', (p) => x(p.d_stat)).attr('cy', (p) => y(p.fd)).attr('r', 4)
      .attr('fill', ARM_COLOUR[arm]).attr('fill-opacity', 0.75);
    const r = data.track[arm].r;
    c.g.append('text').attr('class', 'chart-annotation').attr('x', 2).attr('y', -10)
      .style('font-size', '11.5px').style('letter-spacing', '0.4px')
      .text(`correlation ${r === null ? 'undefined' : r.toFixed(3)} over ${data.track[arm].n} checkpoints`);

    const ranked = M.arms.map((a) => ({ a, r: data.track[a].r, late: lateR(data, a) }))
      .filter((o) => o.r !== null).sort((p, q) => Math.abs(q.r) - Math.abs(p.r));
    document.querySelector('#track-readout').innerHTML =
      `<table class="atlas-table"><thead><tr><th>Objective</th><th>What you can watch</th>`
      + `<th>Correlation, whole run</th><th>Second half only</th><th>Reads as</th>`
      + `</tr></thead><tbody>`
      + ranked.map((o) => `<tr><td>${ARM_NAME[o.a]}</td>`
        + `<td>${o.a.startsWith('wgan') ? "critic's distance estimate" : "discriminator's loss"}</td>`
        + `<td>${o.r.toFixed(3)}</td>`
        + `<td>${o.late === null ? 'too few points' : o.late.toFixed(3)}</td>`
        + `<td>${o.r > 0.3 ? 'lower is better, as advertised'
          : o.r < -0.3 ? 'higher is better, which is backwards' : 'no usable signal'}</td></tr>`).join('')
      + `</tbody></table>`
      + `<p style="margin-top:0.8rem">The second column is the number a practitioner watches. The `
      + `third is how well it predicts the number they care about, and the sign matters as much as `
      + `the size: the claim being tested is that a smaller critic estimate means better samples, `
      + `which is a <span class="bold">positive</span> correlation. A negative one means the `
      + `statistic still carries information and points the wrong way, which is worse than useless `
      + `if you are watching it to decide when to stop.</p>`;
  }

  const row = document.querySelector('#track-buttons');
  const buttons = M.arms.map((arm) => {
    const b = document.createElement('button');
    b.className = arm === state.arm ? 'atlas-btn' : 'atlas-btn ghost';
    b.textContent = ARM_NAME[arm];
    b.addEventListener('click', () => {
      state.arm = arm;
      buttons.forEach((o, i) => {
        o.className = M.arms[i] === arm ? 'atlas-btn' : 'atlas-btn ghost';
      });
      render();
    });
    row.appendChild(b);
    return b;
  });
  render();
  return { state, render };
}

/* ------------------------------------------------------------------------ */
export function initInterpWidget(data, sheets) {
  drawTiles(document.querySelector('#interp-sheet'), sheets.interp.tiles, data.meta.tile, 8,
    { zoom: 2, gap: 2 });
}
