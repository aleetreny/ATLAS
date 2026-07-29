/* What counts as a right answer when the answer is a box.
 *
 * Classification has accuracy because a label is either the label or it is
 * not. A box is never exactly the box, so the metric has to be continuous,
 * and intersection over union is the one everybody settled on. The point of
 * this widget is not that IoU exists: it is how fast it falls. Drag the box
 * a few pixels and watch a number that started at 1 arrive at the threshold
 * that decides whether the detection counted at all.
 *
 * Everything drawn here is checked against the generator before it is shown:
 * the same iou() the detector uses is run over the exported reference pairs
 * and over every row of the closed-form tables, and the worst disagreement is
 * printed in the readout rather than assumed away.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { iou } from './detkit.js';

const K = 5.4;             // units to px in the field panel

export function initIouWidget(data) {
  const geom = data.iou_geometry;
  const sizes = geom.sizes;
  const FIELD = data.meta.canvas;   // the canvas these boxes live on

  /* ---- the guard, run before anything is drawn ------------------------- */
  let refGap = 0;
  for (const r of data.iou_reference) {
    refGap = Math.max(refGap, Math.abs(iou(r.a, r.b) - r.iou));
  }
  let geomGap = 0;
  for (const { L, rows } of geom.shift) {
    for (const r of rows) {
      geomGap = Math.max(geomGap, Math.abs(iou([0, 0, L, L], [r.d, 0, L + r.d, L]) - r.axis));
      geomGap = Math.max(geomGap,
        Math.abs(iou([0, 0, L, L], [r.d, r.d, L + r.d, L + r.d]) - r.diag));
    }
  }
  for (const { L, rows } of geom.scale) {
    for (const r of rows) {
      const s = r.f * L;
      const b = [(L - s) / 2, (L - s) / 2, (L + s) / 2, (L + s) / 2];
      geomGap = Math.max(geomGap, Math.abs(iou([0, 0, L, L], b) - r.iou));
    }
  }
  /* The reference pairs carry eight decimals and the closed-form tables five,
     so those two bounds are what the browser has to land inside. A tolerance
     wider than the rounding would swallow a real implementation error. */
  if (refGap > 1e-7) console.warn(`iou() disagrees with the reference pairs by ${refGap}`);
  if (geomGap > 6e-6) console.warn(`iou() disagrees with the closed form by ${geomGap}`);

  /* The half-IoU distances are solved here for whichever object size is
     selected, so they are checked against the one size the generator solved
     them for rather than trusted. */
  const halfAxisOf = (L) => L / 3;                        // (L-d)/(L+d) = 1/2
  const halfDiagOf = (L) => L * (1 - Math.sqrt(2 / 3));   // the diagonal case
  const solveGap = Math.max(
    Math.abs(halfAxisOf(sizes[1]) - geom.iou50_shift_axis_px),
    Math.abs(halfDiagOf(sizes[1]) - geom.iou50_shift_diag_px),
  );
  if (solveGap > 1e-4) {
    console.warn(`the half-IoU distances disagree with the generator by ${solveGap}`);
  }

  /* ---- state ----------------------------------------------------------- */
  const state = {
    L: 28,                   // truth box side
    f: 1.0,                  // prediction size, as a factor of the truth
    dx: 6,                   // prediction centre offset from the truth centre
    dy: 4,
    view: 'shift',
  };

  const fieldWrap = document.querySelector('#iou-field');
  const svg = d3.select(fieldWrap).append('svg')
    .attr('viewBox', `0 0 ${FIELD * K} ${FIELD * K}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('max-width', `${FIELD * K}px`)
    .style('margin', '0 auto')
    .style('display', 'block')
    .style('touch-action', 'none');

  svg.append('rect')
    .attr('width', FIELD * K).attr('height', FIELD * K)
    .attr('fill', 'var(--white)').attr('stroke', 'var(--stone)');

  const interRect = svg.append('rect').attr('fill', 'var(--primary)').attr('opacity', 0.28);
  const truth = svg.append('rect')
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
  const pred = svg.append('rect')
    .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2)
    .attr('stroke-dasharray', '6 3').style('cursor', 'grab');
  const handle = svg.append('circle')
    .attr('r', 7).attr('fill', 'var(--cosmos)').attr('stroke', 'var(--white)')
    .attr('stroke-width', 2).style('cursor', 'grab');
  const labels = svg.append('g');

  const chart = makeChart('#iou-chart', {
    width: 460, height: 340, margin: { top: 30, right: 22, bottom: 56, left: 62 },
  });

  /* ---- geometry -------------------------------------------------------- */
  function boxes() {
    const L = state.L;
    const s = L * state.f;
    const c = FIELD / 2;
    const a = [c - L / 2, c - L / 2, c + L / 2, c + L / 2];
    const b = [c + state.dx - s / 2, c + state.dy - s / 2,
      c + state.dx + s / 2, c + state.dy + s / 2];
    return { a, b };
  }

  function areas(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const ar = (a[2] - a[0]) * (a[3] - a[1]);
    const br = (b[2] - b[0]) * (b[3] - b[1]);
    return { inter, union: ar + br - inter, rect: [x1, y1, x2, y2] };
  }

  /* ---- drag ------------------------------------------------------------ */
  function dragged(event) {
    const L = state.L;
    const s = L * state.f;
    const half = s / 2;
    const cx = Math.max(half, Math.min(FIELD - half, event.x / K));
    const cy = Math.max(half, Math.min(FIELD - half, event.y / K));
    state.dx = cx - FIELD / 2;
    state.dy = cy - FIELD / 2;
    render();
  }
  const drag = d3.drag().on('drag', dragged);
  pred.call(drag);
  handle.call(drag);

  /* ---- the reference curve panel --------------------------------------- */
  function drawCurve(cur) {
    chart.g.selectAll('*').remove();
    const shiftRows = geom.shift.find((r) => r.L === state.L).rows;
    const scaleRows = geom.scale.find((r) => r.L === state.L).rows;

    if (state.view === 'shift') {
      const x = d3.scaleLinear().domain([0, state.L]).range([0, chart.w]);
      const y = d3.scaleLinear().domain([0, 1]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5, xTicks: 5 });
      axisLabels(chart.g, chart.w, chart.h, {
        x: 'centres this many pixels apart', y: 'intersection over union',
      });
      const line = d3.line().x((r) => x(r.d)).y((r) => y(r.axis));
      chart.g.append('path').datum(shiftRows).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6).attr('d', line);
      chart.g.append('path').datum(shiftRows).attr('fill', 'none')
        .attr('stroke', 'var(--galaxy)').attr('stroke-width', 2.6)
        .attr('stroke-dasharray', '5 3')
        .attr('d', d3.line().x((r) => x(r.d)).y((r) => y(r.diag)));
      chart.g.append('line')
        .attr('x1', 0).attr('x2', chart.w).attr('y1', y(0.5)).attr('y2', y(0.5))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '2 3');
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w - 4).attr('y', y(0.5) - 6).attr('text-anchor', 'end')
        .style('font-size', '0.52rem').style('letter-spacing', '1px')
        .text('the usual threshold');
      chart.g.append('text').attr('x', x(state.L * 0.42)).attr('y', y(0.62))
        .attr('class', 'chart-annotation').style('font-size', '0.52rem')
        .style('letter-spacing', '1px').attr('fill', 'var(--primary)')
        .text('along one axis');
      chart.g.append('text').attr('x', x(state.L * 0.2)).attr('y', y(0.22))
        .attr('class', 'chart-annotation').style('font-size', '0.52rem')
        .style('letter-spacing', '1px').attr('fill', 'var(--galaxy)')
        .text('diagonally');
      const dist = Math.hypot(state.dx, state.dy);
      chart.g.append('circle')
        .attr('cx', x(Math.min(dist, state.L))).attr('cy', y(cur))
        .attr('r', 6).attr('fill', 'var(--cosmos)')
        .attr('stroke', 'var(--white)').attr('stroke-width', 2);
    } else {
      const x = d3.scaleLinear().domain([0.5, 2.0]).range([0, chart.w]);
      const y = d3.scaleLinear().domain([0, 1]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5, xTicks: 6 });
      axisLabels(chart.g, chart.w, chart.h, {
        x: 'predicted side, as a factor of the true one', y: 'intersection over union',
      });
      chart.g.append('path').datum(scaleRows).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6)
        .attr('d', d3.line().x((r) => x(r.f)).y((r) => y(r.iou)));
      chart.g.append('line')
        .attr('x1', 0).attr('x2', chart.w).attr('y1', y(0.5)).attr('y2', y(0.5))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '2 3');
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w - 4).attr('y', y(0.5) - 6).attr('text-anchor', 'end')
        .style('font-size', '0.52rem').style('letter-spacing', '1px')
        .text('the usual threshold');
      chart.g.append('circle')
        .attr('cx', x(state.f)).attr('cy', y(cur))
        .attr('r', 6).attr('fill', 'var(--cosmos)')
        .attr('stroke', 'var(--white)').attr('stroke-width', 2);
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '0.52rem').style('letter-spacing', '1px')
        .text('curve drawn with the centres together');
    }
  }

  /* ---- render ---------------------------------------------------------- */
  function render() {
    const { a, b } = boxes();
    const { inter, union, rect } = areas(a, b);
    const cur = union > 0 ? inter / union : 0;

    truth.attr('x', a[0] * K).attr('y', a[1] * K)
      .attr('width', (a[2] - a[0]) * K).attr('height', (a[3] - a[1]) * K);
    pred.attr('x', b[0] * K).attr('y', b[1] * K)
      .attr('width', (b[2] - b[0]) * K).attr('height', (b[3] - b[1]) * K);
    handle.attr('cx', ((b[0] + b[2]) / 2) * K).attr('cy', ((b[1] + b[3]) / 2) * K);
    if (inter > 0) {
      interRect.attr('x', rect[0] * K).attr('y', rect[1] * K)
        .attr('width', (rect[2] - rect[0]) * K).attr('height', (rect[3] - rect[1]) * K)
        .attr('opacity', 0.28);
    } else {
      interRect.attr('width', 0).attr('height', 0);
    }

    labels.selectAll('*').remove();
    const truthY = a[1] * K - 6;
    const truthLab = labels.append('text').attr('x', a[0] * K).attr('y', truthY)
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 12)
      .attr('fill', 'var(--squidink)').text('truth');
    /* Under the box unless there is no room under the box: a prediction dragged
       to the bottom edge would otherwise take its label off the canvas, where
       it is cut off rather than merely hidden. */
    const below = b[3] * K + 15;
    const predX = Math.min(b[0] * K, FIELD * K - 150);
    let predY = below > FIELD * K - 4 ? Math.max(12, b[1] * K - 6) : below;
    /* And once it has flipped above, it can land on "truth", which is also
       above its own box. Push it clear, upward if there is room and downward
       if there is not, measuring the width that is actually there. */
    const truthW = truthLab.node().getComputedTextLength();
    const overlapsX = predX < a[0] * K + truthW && a[0] * K < predX + 150;
    if (overlapsX && Math.abs(predY - truthY) < 14) {
      predY = truthY - 15 >= 12 ? truthY - 15 : truthY + 15;
    }
    labels.append('text')
      .attr('x', predX).attr('y', predY)
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 12)
      .attr('fill', 'var(--cosmos)').text('prediction (drag me)');

    document.querySelector('#iou-f-value').textContent = `${state.f.toFixed(2)}×`;
    drawCurve(cur);

    const shiftRows = geom.shift.find((r) => r.L === state.L).rows;
    const halfAxis = halfAxisOf(state.L);
    const halfDiag = halfDiagOf(state.L);
    const dist = Math.hypot(state.dx, state.dy);
    const verdict = cur >= 0.75
      ? 'counted at both thresholds this article reports'
      : cur >= 0.5
        ? 'counted at 0.5 and thrown away at 0.75'
        : 'not a detection at all: at 0.5 this is a false positive AND a miss';

    document.querySelector('#iou-readout').innerHTML =
      `<span class="bold">IoU ${cur.toFixed(3)}</span> &middot; intersection `
      + `${inter.toFixed(1)} px², union ${union.toFixed(1)} px², centres `
      + `${dist.toFixed(1)} px apart. That is ${verdict}.`
      + `<br /><br />`
      + `A ${state.L} px box loses half its IoU at <span class="bold">${halfAxis.toFixed(2)} px</span> `
      + `along one axis and at <span class="bold">${halfDiag.toFixed(2)} px</span> diagonally, `
      + `and the two are different because a diagonal error moves both edges at once. `
      + `Shrink the prediction to ${scaleAt(0.5)}× the true side and the IoU is already `
      + `0.5 with the centres perfectly aligned: `
      + `<span class="bold">getting the position exactly right does not save a box of the wrong size</span>. `
      + `That is the whole reason a detector predicts a shape as well as a place.`
      + `<br /><br />`
      + `<span style="color:#5b6572">Both boxes are the tight rectangle around ink at or above `
      + `${geom.ink_threshold}, which is what makes them exact. This browser's IoU against the `
      + `generator's ${data.iou_reference.length} reference pairs: worst disagreement `
      + `${refGap.toExponential(1)}. Against every row of the closed-form tables `
      + `(${shiftRows.length} shifts and ${geom.scale[0].rows.length} scales at each of `
      + `${sizes.length} sizes): ${geomGap.toExponential(1)}, which is the rounding the numbers were `
      + `written with. The generator ran the same check in full precision against rasterised boxes `
      + `and the two ${geom.max_closed_gap === 0
        ? 'agreed to the last of the twelve decimals it recorded'
        : `differed by ${geom.max_closed_gap.toExponential(1)}`}, so the curves on the right are the `
      + `closed form and the closed form is the geometry.</span>`;
  }

  /* The factor at which a perfectly centred prediction reaches a given IoU:
     read off the exported table rather than re-derived, so the sentence and
     the curve cannot drift apart. */
  function scaleAt(target) {
    const rows = geom.scale.find((r) => r.L === state.L).rows.filter((r) => r.f <= 1);
    const hit = rows.reduce((best, r) =>
      (Math.abs(r.iou - target) < Math.abs(best.iou - target) ? r : best));
    return hit.f.toFixed(2);
  }

  /* ---- controls -------------------------------------------------------- */
  /* A prediction wider than the canvas cannot be drawn, let alone dragged, so
     the size slider stops where the box stops fitting. The exported table runs
     to 2.00x and only the largest object is limited by the canvas. */
  const maxF = (L) => Math.min(200, Math.floor((FIELD / L) * 20) * 5);

  const sizeRow = document.querySelector('#iou-size-buttons');
  const sizeBtns = sizes.map((L) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${L === state.L ? '' : ' ghost'}`;
    b.textContent = `${L} px object`;
    b.addEventListener('click', () => {
      state.L = L;
      const cap = maxF(L);
      const slider = document.querySelector('#iou-f');
      slider.max = String(cap);
      if (+slider.value > cap) slider.value = String(cap);
      state.f = +slider.value / 100;
      const half = (state.L * state.f) / 2;
      state.dx = Math.max(half - FIELD / 2, Math.min(FIELD / 2 - half, state.dx));
      state.dy = Math.max(half - FIELD / 2, Math.min(FIELD / 2 - half, state.dy));
      sizeBtns.forEach((n, j) => {
        n.className = `atlas-btn${sizes[j] === L ? '' : ' ghost'}`;
      });
      render();
    });
    sizeRow.appendChild(b);
    return b;
  });

  const viewRow = document.querySelector('#iou-view-buttons');
  const views = [['shift', 'as it slides'], ['scale', 'as it grows']];
  const viewBtns = views.map(([key, label]) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${key === state.view ? '' : ' ghost'}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      state.view = key;
      viewBtns.forEach((n, j) => {
        n.className = `atlas-btn${views[j][0] === key ? '' : ' ghost'}`;
      });
      render();
    });
    viewRow.appendChild(b);
    return b;
  });

  const slider = document.querySelector('#iou-f');
  slider.max = String(maxF(state.L));
  slider.addEventListener('input', () => {
    state.f = +slider.value / 100;
    /* Growing the box must not push it off the field, or the reader loses the
       corner they were dragging. */
    const half = (state.L * state.f) / 2;
    state.dx = Math.max(half - FIELD / 2, Math.min(FIELD / 2 - half, state.dx));
    state.dy = Math.max(half - FIELD / 2, Math.min(FIELD / 2 - half, state.dy));
    render();
  });

  render();
  return { refGap, geomGap };
}
