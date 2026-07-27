/* Two voices, two microphones, and the one degree of freedom PCA leaves behind.
 *
 * The dial is the whole argument made physical. After whitening, the
 * covariance is the identity and stays the identity under every rotation, so
 * second-order statistics are finished and there is exactly one number left to
 * choose. Turn it and watch a curve that PCA cannot see rise to a peak.
 *
 * The FastICA button really runs the fixed-point iteration here rather than
 * jumping to a stored answer; the readout says how many steps it took and
 * whether it agrees with the brute-force sweep.
 */
import { makeChart } from '../../assets/js/chart.js';
import { whiten2, rotate, kurtosis, objectiveAt, recovery, mean } from './linalg.js';

const VIEWS = [
  ['sources', 'the two voices'],
  ['mixed', 'the two microphones'],
  ['white', 'whitened'],
  ['rotated', 'your rotation'],
];

/* Hyvarinen's fixed point with g(u) = u^3, on whitened 2-D data, in the
 * SYMMETRIC form: both units are updated together and then decorrelated
 * against each other, which is what scikit-learn runs by default.
 *
 * The one-unit version is shorter and is the wrong algorithm to put next to
 * this widget. It maximises the non-gaussianity of a SINGLE channel, and on
 * this data that peaks at 37.43 degrees, while the curve the reader is looking
 * at is the joint objective over both channels and peaks at 39.06. Both are
 * correct answers to their own question, they differ by 1.6 degrees here, and
 * showing one while drawing the other would have the button visibly miss the
 * top of its own curve.
 */
export function fastICA2(Z, { tol = 1e-11, maxIter = 200 } = {}) {
  let W = [[1, 0], [0, 1]];

  /* Symmetric decorrelation: W <- (W W')^(-1/2) W, in closed form at 2x2. */
  const decorrelate = (M) => {
    const a = M[0][0] * M[0][0] + M[0][1] * M[0][1];
    const b = M[0][0] * M[1][0] + M[0][1] * M[1][1];
    const c = M[1][0] * M[1][0] + M[1][1] * M[1][1];
    const tr = a + c;
    const disc = Math.sqrt(Math.max((a - c) * (a - c) / 4 + b * b, 0));
    const l1 = Math.max(tr / 2 + disc, 1e-12);
    const l2 = Math.max(tr / 2 - disc, 1e-12);
    const th = 0.5 * Math.atan2(2 * b, a - c);
    const cs = Math.cos(th);
    const sn = Math.sin(th);
    const i1 = 1 / Math.sqrt(l1);
    const i2 = 1 / Math.sqrt(l2);
    const s11 = cs * cs * i1 + sn * sn * i2;
    const s12 = cs * sn * (i1 - i2);
    const s22 = sn * sn * i1 + cs * cs * i2;
    return [
      [s11 * M[0][0] + s12 * M[1][0], s11 * M[0][1] + s12 * M[1][1]],
      [s12 * M[0][0] + s22 * M[1][0], s12 * M[0][1] + s22 * M[1][1]],
    ];
  };

  for (let it = 1; it <= maxIter; it++) {
    const next = [[0, 0], [0, 0]];
    for (let r = 0; r < 2; r++) {
      let e0 = 0;
      let e1 = 0;
      for (const [z0, z1] of Z) {
        const u = W[r][0] * z0 + W[r][1] * z1;
        const u3 = u * u * u;
        e0 += z0 * u3;
        e1 += z1 * u3;
      }
      /* E[z g(w'z)] - E[g'(w'z)] w, and for g(u) = u^3 the second term is 3w
         because the data is whitened and w is a unit vector. */
      next[r] = [e0 / Z.length - 3 * W[r][0], e1 / Z.length - 3 * W[r][1]];
    }
    const Wn = decorrelate(next);
    const moved = Math.max(
      Math.abs(Math.abs(Wn[0][0] * W[0][0] + Wn[0][1] * W[0][1]) - 1),
      Math.abs(Math.abs(Wn[1][0] * W[1][0] + Wn[1][1] * W[1][1]) - 1)
    );
    W = Wn;
    if (moved < tol) return { w: W[0], W, iters: it, converged: true };
  }
  return { w: W[0], W, iters: maxIter, converged: false };
}

export function initIcaWidget(data) {
  const I = data.ica;
  const W = 720;
  const H = 540;
  const { g } = makeChart('#ica-chart', {
    width: W, height: H, margin: { top: 0, right: 0, bottom: 0, left: 0 },
  });

  const slider = document.querySelector('#ica-angle');
  const angleLabel = document.querySelector('#ica-angle-label');
  const readout = document.querySelector('#ica-readout');
  const snap = document.querySelector('#ica-snap');
  const viewRow = document.querySelector('#ica-view-buttons');
  const srcRow = document.querySelector('#ica-source-buttons');

  const state = { deg: 0, view: 'rotated', voices: 'real' };

  /* Both worlds are prepared the same way, and the whitening is done here
     rather than read from the file, so the curve below is this page's. */
  const worlds = {};
  for (const [key, block] of [['real', I], ['gaussian', I.gaussian]]) {
    const sources = block.sources;
    const mixed = block.mixed;
    const white = whiten2(mixed);
    const z1 = white.map((d) => d[0]);
    const z2 = white.map((d) => d[1]);
    const m = [0, 0, 0, 0, 0];
    for (let i = 0; i < white.length; i++) {
      m[0] += z1[i] ** 4;
      m[1] += z1[i] ** 3 * z2[i];
      m[2] += z1[i] ** 2 * z2[i] ** 2;
      m[3] += z1[i] * z2[i] ** 3;
      m[4] += z2[i] ** 4;
    }
    for (let j = 0; j < 5; j++) m[j] /= white.length;
    const curve = d3.range(0, 180, I.dial_step).map((deg) => ({ deg, v: objectiveAt(m, deg).value }));
    const peak = curve.reduce((a, b) => (b.v > a.v ? b : a));
    worlds[key] = {
      sources, mixed, white, moments: m, curve, peak,
      lo: Math.min(...curve.map((d) => d.v)),
      hi: Math.max(...curve.map((d) => d.v)),
      fast: fastICA2(white),
      shipped: block,
    };
  }

  const btnV = {};
  for (const [key, text] of VIEWS) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === 'rotated' ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state.view = key;
      render();
    });
    viewRow.appendChild(b);
    btnV[key] = b;
  }
  const btnS = {};
  for (const [key, text] of [['real', 'real voices'], ['gaussian', 'gaussian voices']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === 'real' ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state.voices = key;
      render();
    });
    srcRow.appendChild(b);
    btnS[key] = b;
  }

  /* ---- panels: scatter top left, the two traces top right, the curve below */
  const SC = { x: 8, y: 34, s: 262 };
  const TR = { x: 322, y: 34, w: 390, h: 118, gap: 26 };
  const CV = { x: 62, y: 358, w: 636, h: 132 };

  const scatterG = g.append('g').attr('transform', `translate(${SC.x},${SC.y})`);
  const traceG = g.append('g').attr('transform', `translate(${TR.x},${TR.y})`);
  const curveG = g.append('g').attr('transform', `translate(${CV.x},${CV.y})`);

  scatterG.append('rect').attr('width', SC.s).attr('height', SC.s)
    .attr('fill', 'var(--white)').attr('stroke', 'var(--stone)');
  const dotsG = scatterG.append('g');
  const axesG = scatterG.append('g');
  /* Anchored at the panel's left edge, not centred on it: centred, a caption
     wider than the 262-unit scatter runs off the left of the whole canvas. */
  const scTitle = g.append('text').attr('class', 'chart-annotation')
    .attr('x', SC.x).attr('y', SC.y - 12).attr('text-anchor', 'start')
    .style('font-size', '11px').style('text-transform', 'none');
  const trTitle = g.append('text').attr('class', 'chart-annotation')
    .attr('x', TR.x + TR.w / 2).attr('y', SC.y - 12).attr('text-anchor', 'middle')
    .style('font-size', '11px').style('text-transform', 'none');

  curveG.append('rect').attr('width', CV.w).attr('height', CV.h)
    .attr('fill', 'var(--white)').attr('stroke', 'var(--stone)');
  const curvePath = curveG.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
  const peakLine = curveG.append('line').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4')
    .attr('y1', 0).attr('y2', CV.h);
  const markLine = curveG.append('line').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2).attr('y1', 0).attr('y2', CV.h);
  const markDot = curveG.append('circle').attr('r', 5)
    .attr('fill', 'var(--squidink)').attr('stroke', 'white').attr('stroke-width', 1.4);
  const curveAxis = curveG.append('g').attr('transform', `translate(0,${CV.h})`);
  const cvTitle = g.append('text').attr('class', 'chart-annotation')
    .attr('x', CV.x + CV.w / 2).attr('y', CV.y - 14).attr('text-anchor', 'middle')
    .style('font-size', '11px').style('text-transform', 'none');
  curveG.append('text').attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)').attr('x', -CV.h / 2).attr('y', -42)
    .attr('text-anchor', 'middle').style('font-size', '11px')
    .text('non-gaussianity');
  curveG.append('text').attr('class', 'axis-label')
    .attr('x', CV.w / 2).attr('y', CV.h + 38).attr('text-anchor', 'middle')
    .style('font-size', '11px').text('rotation of the whitened microphones');

  const peakLabel = curveG.append('text').attr('class', 'chart-annotation')
    .attr('y', -4).style('font-size', '11px').style('text-transform', 'none')
    .attr('fill', 'var(--anchor)');

  function currentPair(world) {
    if (state.view === 'sources') return world.sources;
    if (state.view === 'mixed') return world.mixed;
    if (state.view === 'white') return world.white;
    return rotate(world.white, state.deg);
  }

  function render() {
    const world = worlds[state.voices];
    const pair = currentPair(world);
    const isRot = state.view === 'rotated';

    /* ---- scatter, equal aspect so a right angle looks like a right angle */
    const all = pair.flat();
    const lim = Math.max(...all.map(Math.abs)) * 1.08;
    const sx = d3.scaleLinear().domain([-lim, lim]).range([0, SC.s]);
    const sy = d3.scaleLinear().domain([-lim, lim]).range([SC.s, 0]);
    dotsG.selectAll('circle').data(pair).join('circle')
      .attr('cx', (d) => sx(d[0])).attr('cy', (d) => sy(d[1]))
      .attr('r', 2).attr('fill', 'var(--squidink)').attr('opacity', 0.5);
    axesG.selectAll('line').data([[sx(0), 0, sx(0), SC.s], [0, sy(0), SC.s, sy(0)]])
      .join('line')
      .attr('x1', (d) => d[0]).attr('y1', (d) => d[1])
      .attr('x2', (d) => d[2]).attr('y2', (d) => d[3])
      .attr('stroke', 'var(--stone)').attr('stroke-width', 1);

    /* ---- the two traces */
    const n = pair.length;
    const tx = d3.scaleLinear().domain([0, n - 1]).range([0, TR.w]);
    for (let c = 0; c < 2; c++) {
      const vals = pair.map((d) => d[c]);
      const amp = Math.max(...vals.map(Math.abs)) || 1;
      const ty = d3.scaleLinear().domain([-amp * 1.1, amp * 1.1])
        .range([TR.h, 0]);
      const off = c * (TR.h + TR.gap);
      let row = traceG.select(`g.trace-${c}`);
      if (row.empty()) {
        row = traceG.append('g').attr('class', `trace-${c}`);
        row.append('rect').attr('width', TR.w).attr('height', TR.h)
          .attr('fill', 'var(--white)').attr('stroke', 'var(--stone)');
        row.append('path').attr('fill', 'none').attr('stroke-width', 1.6);
        row.append('text').attr('class', 'chart-annotation')
          .attr('x', 6).attr('y', 14).style('font-size', '11px')
          .style('text-transform', 'none');
      }
      row.attr('transform', `translate(0,${off})`);
      row.select('path')
        .attr('stroke', c === 0 ? 'var(--primary)' : 'var(--anchor)')
        .attr('d', d3.line().x((d, i) => tx(i)).y((d) => ty(d))(vals));
      row.select('text').text(`channel ${c + 1}   kurtosis ${kurtosis(vals).toFixed(3)}`);
    }

    /* ---- the objective curve */
    const pad = (world.hi - world.lo) * 0.15 || 0.1;
    const cx = d3.scaleLinear().domain([0, 180]).range([0, CV.w]);
    const cy = d3.scaleLinear().domain([world.lo - pad, world.hi + pad]).range([CV.h, 0]);
    curvePath.attr('d', d3.line().x((d) => cx(d.deg)).y((d) => cy(d.v))(world.curve));
    peakLine.attr('x1', cx(world.peak.deg)).attr('x2', cx(world.peak.deg));
    peakLabel.attr('x', Math.min(cx(world.peak.deg) + 6, CV.w - 90))
      .text(`peak ${world.peak.deg}°`);
    const here = objectiveAt(world.moments, state.deg).value;
    markLine.attr('x1', cx(state.deg)).attr('x2', cx(state.deg))
      .attr('opacity', isRot ? 1 : 0.25);
    markDot.attr('cx', cx(state.deg)).attr('cy', cy(here))
      .attr('opacity', isRot ? 1 : 0.25);
    curveAxis.call(d3.axisBottom(cx).ticks(7).tickFormat((d) => `${d}°`).tickSizeOuter(0))
      .attr('class', 'axis x');
    curveG.select('g.axis.x').selectAll('text').style('font-size', '11px');

    /* ---- titles */
    const label = VIEWS.find((v) => v[0] === state.view)[1];
    scTitle.text(state.view === 'rotated' ? `whitened, turned ${state.deg.toFixed(1)}°` : label);
    trTitle.text(state.view === 'sources'
      ? 'what the algorithm is never shown'
      : state.view === 'mixed' ? 'what it is given' : 'the two channels, as signals');
    cvTitle.text(state.voices === 'real'
      ? 'the same curve at every view: it is a property of the data, not of the dial'
      : 'gaussian voices: the same axes, and almost no curve left to climb');

    angleLabel.textContent = `${state.deg.toFixed(1)}°`;
    for (const [k, b] of Object.entries(btnV)) b.classList.toggle('ghost', k !== state.view);
    for (const [k, b] of Object.entries(btnS)) b.classList.toggle('ghost', k !== state.voices);

    /* ---- readout: every claim measured on what is drawn */
    const rotated = rotate(world.white, state.deg);
    const mine = recovery(rotated, world.sources);
    const pcaOnly = recovery(world.white, world.sources);
    const atPeak = recovery(rotate(world.white, world.peak.deg), world.sources);
    const span = world.hi - world.lo;
    const gapDeg = Math.min(Math.abs(state.deg - world.peak.deg),
      180 - Math.abs(state.deg - world.peak.deg));
    const fastDeg = ((Math.atan2(world.fast.w[1], world.fast.w[0]) * 180) / Math.PI + 180) % 180;
    const fastAgrees = Math.min(Math.abs(fastDeg - world.peak.deg) % 90,
      90 - (Math.abs(fastDeg - world.peak.deg) % 90));

    let verdict;
    if (state.voices === 'gaussian') {
      verdict = `Two gaussian voices, the same microphones, the same whitening. The curve is `
        + `<span class="bold">flat</span>: it spans ${span.toFixed(4)} from end to end against `
        + `${(worlds.real.hi - worlds.real.lo).toFixed(4)} for the real voices, so there is no peak `
        + `to climb and nothing to choose between one rotation and another. That is not a failure of `
        + `the algorithm: a rotation of independent gaussians is independent gaussians, so the `
        + `rotated data is genuinely indistinguishable from the original and no method that exists `
        + `could tell them apart. The best rotation here still leaves `
        + `${atPeak.leak.toFixed(4)} of the other voice in the answer.`;
    } else if (gapDeg < 0.6) {
      verdict = `That is the peak. The worse-recovered voice now matches its source at `
        + `<span class="bold">${mine.match.toFixed(4)}</span> with only ${mine.leak.toFixed(4)} of `
        + `the other one leaking in, against ${pcaOnly.match.toFixed(4)} and `
        + `${pcaOnly.leak.toFixed(4)} for whitening alone. Nothing was added to the data to get `
        + `here: PCA had already done everything second-order statistics can do, and the whole `
        + `remaining job was choosing one angle by a criterion PCA does not have.`;
    } else {
      verdict = `${gapDeg.toFixed(1)}° from the peak at ${world.peak.deg}°. Non-gaussianity here is `
        + `${here.toFixed(3)} against ${world.hi.toFixed(3)} at the peak, and the worse-recovered `
        + `voice matches at ${mine.match.toFixed(4)} with ${mine.leak.toFixed(4)} of the other one `
        + `mixed into it. For comparison, stopping at whitening (which is exactly what PCA hands `
        + `you) gives ${pcaOnly.match.toFixed(4)} and ${pcaOnly.leak.toFixed(4)}.`;
    }

    readout.innerHTML = `
      <table class="dir-table">
        <tr><th>rotation</th><th>non-gaussianity</th><th>worse voice recovered</th><th>other voice leaking in</th></tr>
        <tr>
          <td>${state.deg.toFixed(1)}°</td>
          <td><span class="value">${here.toFixed(3)}</span></td>
          <td><span class="value">${mine.match.toFixed(4)}</span></td>
          <td><span class="value">${mine.leak.toFixed(4)}</span></td>
        </tr>
        <tr>
          <td>whitening alone</td><td>${objectiveAt(world.moments, 0).value.toFixed(3)}</td>
          <td>${pcaOnly.match.toFixed(4)}</td><td>${pcaOnly.leak.toFixed(4)}</td>
        </tr>
        <tr>
          <td>the peak, ${world.peak.deg}°</td><td>${world.hi.toFixed(3)}</td>
          <td>${atPeak.match.toFixed(4)}</td><td>${atPeak.leak.toFixed(4)}</td>
        </tr>
      </table>
      <div class="dir-note">${verdict}
        <br /><br />
        FastICA's fixed point, run in this page on these ${n} samples, converged in
        <span class="bold">${world.fast.iters}</span> step${world.fast.iters === 1 ? '' : 's'} to
        ${fastDeg.toFixed(2)}°, which is the swept peak to within ${fastAgrees.toFixed(2)}°.
        It never sweeps anything: it starts anywhere and climbs.
      </div>`;
  }

  slider.addEventListener('input', () => {
    state.deg = +slider.value;
    /* A dial that does nothing in three of the four views is a dead control,
       so moving it puts you in the view where it means something. */
    state.view = 'rotated';
    render();
  });
  snap.addEventListener('click', () => {
    const world = worlds[state.voices];
    const deg = ((Math.atan2(world.fast.w[1], world.fast.w[0]) * 180) / Math.PI + 180) % 180;
    /* Land on the dial's own grid, or the reader is shown a setting the
       slider cannot represent and the thumb jumps somewhere else. */
    state.deg = Math.round(deg / I.dial_step) * I.dial_step;
    state.view = 'rotated';
    slider.value = String(state.deg);
    render();
  });

  render();
  return worlds;
}
