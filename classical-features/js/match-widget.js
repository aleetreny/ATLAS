/* Invariance, drawn and measured.
 *
 * Top: the original and the warped image, with the matches a descriptor
 * actually proposed drawn between them, green when the match lands within 3
 * pixels of where the known transform says it must and crimson when it does
 * not. Bottom: the full measured sweep, correct matches and precision, for
 * SIFT, ORB and the naive raw-patch control, with the displayed case marked.
 *
 * Nothing here is recomputed in the browser: these are scikit-image's matches
 * against a ground truth the generator could check because it chose the
 * transform itself.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { loadImage, makeCanvas } from '../../assets/js/imagery.js';

const COL = { sift: '#78350f', orb: '#1d4ed8', patch: '#6b7280' };
const OKC = '#0e8a55';
const BADC = '#b91c1c';
const PANEL_ORDER = ['rot0', 'rot45', 'rot90', 'scale50'];
const PANEL_LABEL = { rot0: 'unchanged', rot45: 'rotated 45', rot90: 'rotated 90', scale50: 'half size' };

export async function initMatchWidget(data, base) {
  const M = data.match;
  const N = M.size;
  const src = await loadImage(`${base}${M.image}`);
  const warped = {};
  for (const key of PANEL_ORDER) {
    warped[key] = await loadImage(`${base}${M.panels[key].image}`);
  }

  const gap = 26;
  const W = N * 2 + gap;
  const H = N + 26;
  const { ctx } = makeCanvas('#match-images', W, H);
  const y0 = 22;

  /* the two measured sweeps, side by side */
  const wrap = d3.select('#match-curve');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const leftDiv = wrap.append('div').node();
  const rightDiv = wrap.append('div').node();
  const L = makeChart(leftDiv, { width: 340, height: 300, margin: { top: 28, right: 14, bottom: 44, left: 52 } });
  const R = makeChart(rightDiv, { width: 340, height: 300, margin: { top: 28, right: 14, bottom: 44, left: 44 } });

  const state = { panel: 'rot45', desc: 'sift' };

  const descRow = document.querySelector('#match-desc');
  const caseRow = document.querySelector('#match-case');
  const readout = document.querySelector('#match-readout');

  const descBtns = {};
  for (const [key, label] of [['sift', 'SIFT'], ['orb', 'ORB'], ['patch', 'raw 16x16 patches']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === state.desc ? '' : ' ghost');
    b.textContent = label;
    b.addEventListener('click', () => { state.desc = key; render(); });
    descRow.appendChild(b);
    descBtns[key] = b;
  }
  const caseBtns = {};
  for (const key of PANEL_ORDER) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === state.panel ? '' : ' ghost');
    b.textContent = PANEL_LABEL[key];
    b.addEventListener('click', () => { state.panel = key; render(); });
    caseRow.appendChild(b);
    caseBtns[key] = b;
  }

  function drawImages() {
    const p = M.panels[state.panel];
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(src, 0, y0, N, N);
    ctx.drawImage(warped[state.panel], N + gap, y0, N, N);

    ctx.save();
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#232f3e';
    ctx.textAlign = 'center';
    ctx.fillText('the original', N / 2, 14);
    ctx.fillText(p.theta ? `rotated ${p.theta} degrees` : `scaled to ${p.scale}x`,
      N + gap + N / 2, 14);
    ctx.restore();

    const pairs = state.desc === 'sift' ? p.sift_pairs : state.desc === 'orb' ? p.orb_pairs : [];
    ctx.save();
    ctx.lineWidth = 1;
    for (const [x1, y1, x2, y2, ok] of pairs) {
      ctx.strokeStyle = ok ? OKC : BADC;
      ctx.globalAlpha = ok ? 0.42 : 0.9;
      ctx.beginPath();
      ctx.moveTo(x1, y0 + y1);
      ctx.lineTo(N + gap + x2, y0 + y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (const [x1, y1, x2, y2, ok] of pairs) {
      ctx.fillStyle = ok ? OKC : BADC;
      ctx.beginPath();
      ctx.arc(x1, y0 + y1, 1.7, 0, 6.284);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(N + gap + x2, y0 + y2, 1.7, 0, 6.284);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawSweeps() {
    const isRot = state.panel !== 'scale50';
    const rows = isRot ? M.rotation : M.scale;
    const xVal = (r) => (isRot ? r.theta : r.scale);
    const xs = rows.map(xVal);

    for (const [chart, acc, title, dom] of [
      [L, (r, k) => r[k].n_correct, 'correct matches', null],
      [R, (r, k) => r[k].precision, 'precision of the proposals', [0, 1.05]],
    ]) {
      chart.g.selectAll('*').remove();
      const x = isRot
        ? d3.scaleLinear().domain([0, 180]).range([0, chart.w])
        : d3.scaleLinear().domain([d3.min(xs), 1]).range([0, chart.w]);
      const maxY = dom ? dom[1] : d3.max(rows, (r) => Math.max(acc(r, 'sift'), acc(r, 'orb'), acc(r, 'patch'))) * 1.1;
      const y = d3.scaleLinear().domain(dom || [0, maxY]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { xTicks: 5, yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { xTicks: 5, yTicks: 5 });
      axisLabels(chart.g, chart.w, chart.h, { x: isRot ? 'rotation, degrees' : 'scale factor' });
      /* two 340-unit panels sit side by side on a phone, so the render scale
         halves: 0.56rem fell to 4.4px there, 0.68rem stays above the floor */
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '0.68rem').style('text-transform', 'none')
        .text(title);

      for (const key of ['sift', 'orb', 'patch']) {
        chart.g.append('path').attr('fill', 'none').attr('stroke', COL[key])
          .attr('stroke-width', key === state.desc ? 3.2 : 1.6)
          .attr('opacity', key === state.desc ? 1 : 0.45)
          .attr('stroke-dasharray', key === 'patch' ? '5 4' : null)
          .attr('d', d3.line().x((r) => x(xVal(r))).y((r) => y(acc(r, key)))(rows));
      }
      /* where the panel above sits on these curves */
      const p = M.panels[state.panel];
      const px = isRot ? p.theta : p.scale;
      chart.g.append('line').attr('x1', x(px)).attr('x2', x(px)).attr('y1', 0).attr('y2', chart.h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
      chart.g.append('circle').attr('r', 4.5).attr('cx', x(px))
        .attr('cy', y(acc(rows.find((r) => Math.abs(xVal(r) - px) < 1e-6) || rows[0], state.desc)))
        .attr('fill', COL[state.desc]).attr('stroke', 'white').attr('stroke-width', 1.4);
    }
  }

  function render() {
    for (const [k, b] of Object.entries(descBtns)) b.classList.toggle('ghost', k !== state.desc);
    for (const [k, b] of Object.entries(caseBtns)) b.classList.toggle('ghost', k !== state.panel);
    drawImages();
    drawSweeps();

    const p = M.panels[state.panel];
    const d = p[state.desc];
    const other = state.desc === 'sift' ? 'orb' : 'sift';
    readout.innerHTML =
      `<table class="feat-table">
         <tr><th>${state.desc === 'patch' ? 'raw patches' : state.desc.toUpperCase()}</th>
             <th>proposed</th><th>correct within ${M.tol_px} px</th><th>precision</th></tr>
         <tr><td>${PANEL_LABEL[state.panel]}</td><td><span class="value">${d.n_matches}</span></td>
             <td><span class="value">${d.n_correct}</span></td>
             <td><span class="value">${d.precision.toFixed(2)}</span></td></tr>
         <tr><td>${other === 'sift' ? 'SIFT' : 'ORB'}, same case</td><td>${p[other].n_matches}</td>
             <td>${p[other].n_correct}</td><td>${p[other].precision.toFixed(2)}</td></tr>
         <tr><td>raw patches, same case</td><td>${p.patch.n_matches}</td>
             <td>${p.patch.n_correct}</td><td>${p.patch.precision.toFixed(2)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${verdict()}</div>`;
  }

  function verdict() {
    const p = M.panels[state.panel];
    const d = p[state.desc];
    if (state.desc === 'patch') {
      return p.theta === 0 && p.scale === 1
        ? `Unchanged image, so the naive descriptor is perfect: ${p.patch.n_correct} of ` +
          `${p.patch.n_matches}. Comparing raw pixel patches works beautifully right up until ` +
          `anything moves. Press a rotated case.`
        : `Here is the control that makes the point: the raw patch descriptor sits on the very same ` +
          `SIFT keypoints, so the detector is not what fails. It proposed ${p.patch.n_matches} matches ` +
          `and got <span class="bold">${p.patch.n_correct}</span> right. A rotated patch of pixels ` +
          `simply is not the same list of numbers, and no amount of good keypoints saves it. That gap ` +
          `is the descriptor's entire job.`;
    }
    if (state.desc === 'orb' && p.theta >= 90) {
      return `ORB stays cheap and stays useful, but look at the precision curve: it drifts down as the ` +
        `rotation grows, ${p.orb.precision.toFixed(2)} here against SIFT's ${p.sift.precision.toFixed(2)}. ` +
        `ORB rotates its sampling pattern to the keypoint's measured angle, which is a discretised ` +
        `approximation of what SIFT does with its full gradient histogram. You get most of the ` +
        `invariance for a binary descriptor that matches by counting bit differences.`;
    }
    if (state.panel === 'scale50') {
      return `At half size the detectors simply find fewer things: SIFT keeps ${p.sift.n_correct} correct ` +
        `matches and ORB ${p.orb.n_correct}, against ${M.n_sift_kp} and ${M.n_orb_kp} keypoints in the ` +
        `original. What does <i>not</i> happen is a collapse in precision: what survives is still right. ` +
        `SIFT searches over a stack of blurred copies precisely so that a feature seen small is still ` +
        `the same feature.`;
    }
    if (p.theta === 0 && p.scale === 1) {
      return `The sanity check, and it matters: with no transform at all, every descriptor should be ` +
        `perfect, and all three are. If this case were not clean, nothing measured on the rotated ones ` +
        `could be trusted. Now rotate.`;
    }
    const label = state.desc.toUpperCase();
    return `${d.n_correct} of ${d.n_matches} ${label} matches survive a ${p.theta}-degree rotation, ` +
      `a precision of ${d.precision.toFixed(2)}. ${label} measures a local orientation at each ` +
      `keypoint and describes the patch relative to it, so turning the picture can turn the ` +
      `description with it. The green lines are the ones a ground truth confirms; crimson ones ` +
      `are honest mistakes.`;
  }

  render();
}
