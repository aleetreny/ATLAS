/* One ray, taken apart.
 *
 * The scrolly reveals the three quantities in the rendering integral in the
 * order they depend on each other, and the widget underneath lets the reader
 * move the ray and starve the sum of steps until it stops being right. Both
 * draw the same chart, because they are about the same arithmetic.
 *
 * The ray the scrolly opens on is not hand picked. The widget scans the frame
 * for the ray whose density profile has two clearly separated humps, so the
 * transmittance panel has something to destroy between them; if the scene ever
 * changes, the illustration follows it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { rayDir } from './truthkit.js';
import { makeFrame, buttonRow, seriesLabel, fmt } from './panels.js';

const REF_STEPS = 1024;
const STEP_CHOICES = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384];

/* How many separated humps the density has along this ray, and how wide the
   gap between the first two is. Two humps means the reader can watch the
   transmittance kill the second one. */
function humps(prof, steps) {
  const runs = [];
  let start = -1;
  for (let i = 0; i < steps; i++) {
    const on = prof.sigma[i] > 1;
    if (on && start < 0) start = i;
    if (!on && start >= 0) { runs.push([start, i - 1]); start = -1; }
  }
  if (start >= 0) runs.push([start, steps - 1]);
  return runs;
}

/* Coarse on purpose: this runs five times at page load and the answer only has
   to be a good ray, not the best one, so it looks at a ninth of the pixels at
   two thirds of the step count. */
function pickRay(scene, cam, res, fov) {
  let best = null;
  const STEPS = 64;
  for (let i = 2; i < res; i += 3) {
    for (let j = 2; j < res; j += 3) {
      const d = rayDir(cam, res, fov, i, j);
      const r = scene.march(cam.pos, d, STEPS, { profile: true, stopAt: 1e-6 });
      if (!r.prof) continue;
      const hs = humps(r.prof, STEPS);
      if (hs.length < 2) continue;
      const gap = hs[1][0] - hs[0][1];
      const mass = hs.reduce((s, h) => s + (h[1] - h[0] + 1), 0);
      const score = gap * 2 + mass;
      if (!best || score > best.score) best = { i, j, score, gap };
    }
  }
  return best || { i: Math.floor(res / 2), j: Math.floor(res / 2) };
}

/* The chart both the scrolly and the widget draw. `show` says which of the
   three curves are visible, which is what the scrolly steps change. */
export function drawRayChart(node, state) {
  const { prof, steps, tn, tf, show, marks, rgb } = state;
  d3.select(node).selectAll('*').remove();
  const c = makeChart(node, { width: 660, height: 400, margin: { top: 34, right: 58, bottom: 54, left: 62 } });
  const x = d3.scaleLinear().domain([tn, tf]).range([0, c.w]);
  const sigMax = Math.max(10, d3.max(prof.sigma) * 1.12);
  const ys = d3.scaleLinear().domain([0, sigMax]).range([c.h, 0]);
  const yu = d3.scaleLinear().domain([0, 1.04]).range([c.h, 0]);
  drawGrid(c.g, x, yu, c.w, c.h, { xTicks: 6, yTicks: 5 });
  drawAxes(c.g, x, ys, c.w, c.h, { xTicks: 6, yTicks: 5, yFmt: d3.format('~s') });
  axisLabels(c.g, c.w, c.h, { x: 'depth along the ray', y: 'density' });

  /* the right hand axis carries transmittance and weight, both of which live
     in [0, 1] and neither of which is a density */
  const gr = c.g.append('g').attr('class', 'axis y-right').attr('transform', `translate(${c.w},0)`);
  gr.call(d3.axisRight(yu).ticks(5).tickSizeOuter(0));
  c.g.append('text').attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)').attr('x', -c.h / 2).attr('y', c.w + 46)
    .attr('text-anchor', 'middle').text('transmittance / weight');

  if (marks && marks.length) {
    c.g.append('g').selectAll('line').data(marks).join('line')
      .attr('x1', (d) => x(d)).attr('x2', (d) => x(d))
      .attr('y1', c.h).attr('y2', c.h - 9)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.5);
  }

  const line = (acc, scaleY) => d3.line()
    .x((_, i) => x(prof.t[i]))
    .y((_, i) => scaleY(acc(i)))
    .defined((_, i) => prof.t[i] > 0);

  if (show.sigma) {
    const area = d3.area()
      .x((_, i) => x(prof.t[i]))
      .y0(c.h)
      .y1((_, i) => ys(prof.sigma[i]));
    c.g.append('path').datum(Array.from(prof.sigma))
      .attr('d', area).attr('fill', 'var(--primary)').attr('opacity', 0.18);
    c.g.append('path').datum(Array.from(prof.sigma))
      .attr('d', line((i) => prof.sigma[i], ys))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
  }
  if (show.T) {
    c.g.append('path').datum(Array.from(prof.T))
      .attr('d', line((i) => prof.T[i], yu))
      .attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.2);
  }
  if (show.w) {
    /* drawn on the same [0, 1] axis as the transmittance and not rescaled: the
       weights add up to how opaque the ray turned out to be, so a step count
       that halves each weight is telling the truth about itself */
    const area = d3.area()
      .x((_, i) => x(prof.t[i]))
      .y0(c.h)
      .y1((_, i) => yu(prof.w[i]));
    c.g.append('path').datum(Array.from(prof.w))
      .attr('d', area).attr('fill', 'var(--cosmos)').attr('opacity', 0.3);
    c.g.append('path').datum(Array.from(prof.w))
      .attr('d', line((i) => prof.w[i], yu))
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);
  }

  /* direct labels beat a legend, and they go where the curve is */
  /* No greek in these labels. The annotation face is uppercased by the
     stylesheet, so "density σ" came out as "DENSITY Σ", which reads as a sum
     and is a different symbol from the one meant. The formula above the chart
     is where the letters belong. */
  const labels = [];
  if (show.sigma) labels.push({ text: 'density', colour: 'var(--primary)', y: 14 });
  if (show.T) labels.push({ text: 'transmittance', colour: 'var(--anchor)', y: 32 });
  if (show.w) labels.push({ text: 'weight', colour: 'var(--cosmos)', y: 50 });
  labels.forEach((L) => {
    seriesLabel(c.g, 8, L.y, L.text, { colour: L.colour });
  });

  if (rgb && show.w) {
    const s = 26;
    c.g.append('rect').attr('x', c.w - s - 2).attr('y', 4).attr('width', s).attr('height', s)
      .attr('fill', `rgb(${rgb.map((v) => Math.round(v * 255)).join(',')})`)
      .attr('stroke', 'var(--squidink)');
    seriesLabel(c.g, c.w - s - 8, 20, 'pixel', { anchor: 'end' });
  }
  c.svg.append('text').attr('class', 'chart-title')
    .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
    .text(`steps: ${steps}`);
  return c;
}

function profileFor(scene, cam, res, fov, i, j, steps) {
  const d = rayDir(cam, res, fov, i, j);
  const r = scene.march(cam.pos, d, steps, { profile: true, stopAt: 1e-7 });
  return r;
}

export function initRayScrolly(data, scene) {
  const meta = data.meta;
  const cam = data.cameras.train[4];
  const res = meta.res;
  const pick = pickRay(scene, cam, res, meta.fov_deg);
  const r = profileFor(scene, cam, res, meta.fov_deg, pick.i, pick.j, 128);
  const shows = [
    { sigma: false, T: false, w: false },
    { sigma: true, T: false, w: false },
    { sigma: true, T: true, w: false },
    { sigma: true, T: true, w: true },
  ];
  const node = document.querySelector('#ray-chart');
  const render = (k) => drawRayChart(node, {
    prof: r.prof, steps: 128, tn: r.tn, tf: r.tf, show: shows[k], marks: null, rgb: r.rgb,
  });
  render(0);
  scrolly('#ray-scrolly .step', shows.map((_, k) => () => render(k)));
}

export function initQuadWidget(data, scene, sheets) {
  const meta = data.meta;
  const res = meta.res;
  const picks = [0, 4, 10, 18];
  const state = { camIdx: 0, i: 0, j: 0, stepIdx: 8 };
  const frameHost = document.querySelector('#quad-canvas');
  frameHost.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'frame-row';
  frameHost.appendChild(row);
  const frame = makeFrame(row, 'click a pixel', { max: 260 });
  const chart = document.querySelector('#quad-chart');
  const readout = document.querySelector('#quad-readout');
  const stepsVal = document.querySelector('#quad-steps-value');
  const slider = document.querySelector('#quad-steps');
  slider.max = String(STEP_CHOICES.length - 1);

  const cams = picks.map((p) => data.cameras.train[p]);
  const tiles = picks.map((p) => sheets.train.tiles[p]);
  const chosen = cams.map((c) => pickRay(scene, c, res, meta.fov_deg));

  const paintFrame = () => {
    const t = tiles[state.camIdx];
    const marked = new Float32Array(t);
    /* a crosshair drawn into the pixels themselves, so it survives the zoom */
    for (let k = 0; k < res; k++) {
      for (const [ii, jj] of [[state.i, k], [k, state.j]]) {
        if (Math.abs(ii - state.i) + Math.abs(jj - state.j) < 2) continue;
        const p = (ii * res + jj) * 3;
        marked[p] = 0.13;
        marked[p + 1] = 0.18;
        marked[p + 2] = 0.24;
      }
    }
    frame.draw(marked, res);
  };

  const render = () => {
    const cam = cams[state.camIdx];
    const steps = STEP_CHOICES[state.stepIdx];
    stepsVal.textContent = String(steps);
    const ref = profileFor(scene, cam, res, meta.fov_deg, state.i, state.j, REF_STEPS);
    const got = profileFor(scene, cam, res, meta.fov_deg, state.i, state.j, steps);
    const marks = [];
    const dt = (got.tf - got.tn) / steps;
    for (let s = 0; s < steps && s < 260; s++) marks.push(got.tn + (s + 0.5) * dt);
    drawRayChart(chart, {
      prof: got.prof, steps, tn: got.tn, tf: got.tf,
      show: { sigma: true, T: true, w: true }, marks, rgb: got.rgb,
    });
    paintFrame();
    const err = Math.max(...got.rgb.map((v, k) => Math.abs(v - ref.rgb[k])));
    const q = data.quadrature.rows.find((x) => x.steps === steps);
    const swatch = (c) => `<span style="display:inline-block;width:0.9em;height:0.9em;`
      + `border:1px solid var(--squidink);vertical-align:-0.1em;`
      + `background:rgb(${c.map((v) => Math.round(v * 255)).join(',')})"></span>`;
    readout.innerHTML = `pixel (${state.i}, ${state.j}) of camera at azimuth `
      + `${cam.az.toFixed(0)}°, elevation ${cam.el.toFixed(0)}°. `
      + `<span class="bold">${steps}</span> steps ${swatch(got.rgb)} against `
      + `${fmt.n0(REF_STEPS)} steps ${swatch(ref.rgb)}: off by `
      + `<span class="bold">${(err * 255).toFixed(1)}</span> of 255 on the worst channel`
      + (q ? `, and over the whole frame that step count scores <span class="bold">${fmt.db(q.psnr)}</span> `
        + `against the same reference.` : '.');
  };

  frame.canvas.addEventListener('click', (e) => {
    const b = frame.canvas.getBoundingClientRect();
    state.j = Math.max(0, Math.min(res - 1, Math.floor(((e.clientX - b.left) / b.width) * res)));
    state.i = Math.max(0, Math.min(res - 1, Math.floor(((e.clientY - b.top) / b.height) * res)));
    render();
  });
  slider.addEventListener('input', () => { state.stepIdx = +slider.value; render(); });

  const host = document.querySelector('#quad-view-buttons');
  host.innerHTML = '<span class="slider-label">camera:</span>';
  const btnBox = document.createElement('span');
  btnBox.style.display = 'contents';
  host.appendChild(btnBox);
  const paint = buttonRow(btnBox,
    cams.map((c, k) => ({ key: k, label: `${c.az.toFixed(0)}°, ${c.el.toFixed(0)}°` })),
    0, (k) => {
      state.camIdx = k;
      state.i = chosen[k].i;
      state.j = chosen[k].j;
      paint(k);
      render();
    });
  state.i = chosen[0].i;
  state.j = chosen[0].j;
  state.stepIdx = STEP_CHOICES.indexOf(64) >= 0 ? STEP_CHOICES.indexOf(64) : 8;
  slider.value = String(state.stepIdx);
  render();
}
