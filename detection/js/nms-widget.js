/* What a badly set suppression threshold costs.
 *
 * Suppression is the one part of a detector that is not learned and not
 * evaluated: it happens after the network has finished, it has one number in
 * it, and that number is usually copied from whichever repository the code
 * came from. The detections below never change. Only which of them survive
 * changes, and the score moves by more than most of the architecture decisions
 * in this article.
 *
 * The lines are measured over the whole test split. The dashed line is
 * recomputed in this browser, right now, over the ranked detections the
 * generator shipped, so the shape of the curve is something the reader can
 * watch being produced rather than something to take on faith.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { iou, nms } from './detkit.js';

/* The generator's all-point average precision, transcribed. */
function allPointAp(rec, prec) {
  const mrec = [0, ...rec, 1];
  const mpre = [0, ...prec, 0];
  for (let i = mpre.length - 2; i >= 0; i--) mpre[i] = Math.max(mpre[i], mpre[i + 1]);
  let ap = 0;
  for (let i = 0; i + 1 < mrec.length; i++) {
    if (mrec[i + 1] !== mrec[i]) ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
  }
  return ap;
}

export function initNmsWidget(data) {
  const N = data.nms;
  const grid = N.grid;
  const W = data.eval.widget;
  const nClasses = data.meta.classes.length;

  /* The rows are unpacked positionally, so the column order shipped with them
     is checked rather than assumed: a reordered export would otherwise turn
     into wrong boxes that still look like boxes. */
  const EXPECTED = ['img', 'cls', 'score', 'x1', 'y1', 'x2', 'y2'];
  const columnsOk = JSON.stringify(W.columns) === JSON.stringify(EXPECTED);
  if (!columnsOk) {
    console.warn(`the shipped detections are in column order ${W.columns.join(', ')}, `
      + `not ${EXPECTED.join(', ')}`);
  }

  /* ---- the browser's own copy of the split, unpacked once ---------------- */
  const byImage = new Map();
  for (const [img, cls, score, x1, y1, x2, y2] of W.detections) {
    if (!byImage.has(img)) byImage.set(img, []);
    byImage.get(img).push({
      img,
      cls,
      score: score / W.score_scale,
      box: [x1 / W.box_scale, y1 / W.box_scale, x2 / W.box_scale, y2 / W.box_scale],
    });
  }
  const gtByImage = new Map();
  for (const [img, cls, x1, y1, x2, y2] of W.gt) {
    if (!gtByImage.has(img)) gtByImage.set(img, []);
    gtByImage.get(img).push({
      cls, box: [x1 / W.box_scale, y1 / W.box_scale, x2 / W.box_scale, y2 / W.box_scale],
    });
  }
  const nPos = new Array(nClasses).fill(0);
  for (const rows of gtByImage.values()) for (const g of rows) nPos[g.cls] += 1;

  /* mAP over the shipped subset at one suppression threshold, judged at the
     first threshold of the exported sweep, which is the one every headline
     number on the page is quoted at. The matching rule is the generator's:
     confidence order, each ground-truth box consumed once, matched only within
     its own image and its own class. */
  const MATCH_IOU = data.eval.iou_sweep[0];
  function liveMap(t) {
    const kept = [];
    for (const dets of byImage.values()) {
      for (const d of nms(dets, t)) kept.push(d);
    }
    const aps = [];
    for (let c = 0; c < nClasses; c++) {
      if (nPos[c] === 0) continue;
      const mine = kept.filter((d) => d.cls === c)
        .sort((a, b) => (b.score - a.score) || (a.img - b.img));
      const used = new Map();
      let tp = 0;
      let fp = 0;
      const rec = [];
      const prec = [];
      for (const d of mine) {
        const gts = (gtByImage.get(d.img) || []);
        if (!used.has(d.img)) used.set(d.img, new Array(gts.length).fill(false));
        const flags = used.get(d.img);
        let best = -1;
        let bestIou = MATCH_IOU;
        for (let j = 0; j < gts.length; j++) {
          if (flags[j] || gts[j].cls !== c) continue;
          const o = iou(d.box, gts[j].box);
          if (o >= bestIou) { bestIou = o; best = j; }
        }
        if (best >= 0) { flags[best] = true; tp += 1; } else { fp += 1; }
        rec.push(tp / nPos[c]);
        prec.push(tp / (tp + fp));
      }
      aps.push(rec.length ? allPointAp(rec, prec) : 0);
    }
    return {
      map50: aps.reduce((a, b) => a + b, 0) / Math.max(aps.length, 1),
      kept: kept.length,
    };
  }

  const liveCurve = grid.map((t) => ({ t, ...liveMap(t) }));

  /* ---- the guard the subset can actually fail ---------------------------- */
  const bestFull = N.all.reduce((a, b) => (b.map50 > a.map50 ? b : a));
  const bestLive = liveCurve.reduce((a, b) => (b.map50 > a.map50 ? b : a));
  const argmaxAgrees = Math.abs(bestLive.t - bestFull.t) < 1e-9;
  if (!argmaxAgrees) {
    console.warn(`the ${W.n_images}-image subset peaks at ${bestLive.t}, `
      + `the full split at ${bestFull.t}`);
  }

  const state = { t: N.isolated.chosen, view: 'all' };
  const chart = makeChart('#nms-chart', {
    width: 700, height: 380, margin: { top: 34, right: 24, bottom: 58, left: 74 },
  });
  const hist = makeChart('#nms-hist', {
    width: 700, height: 190, margin: { top: 26, right: 24, bottom: 50, left: 74 },
  });

  const rowAt = (rows, t) => rows.reduce((a, b) =>
    (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));

  /* ---- the overlap in the data, which is what suppression is guessing at -- */
  function drawHist() {
    const bins = data.dataset.pair_iou_hist;
    const hi = data.dataset.pair_iou_hist_max;   // the range the generator binned over
    const wBin = hi / bins.length;
    const x = d3.scaleLinear().domain([0, hi]).range([0, hist.w]);
    const y = d3.scaleLinear().domain([0, d3.max(bins)]).range([hist.h, 0]);
    hist.g.selectAll('*').remove();
    drawGrid(hist.g, x, y, hist.w, hist.h, { yTicks: 3 });
    drawAxes(hist.g, x, y, hist.w, hist.h, { yTicks: 3, xTicks: 7, yFmt: d3.format('~s') });
    axisLabels(hist.g, hist.w, hist.h, {
      x: 'the most overlapped pair of objects in a test canvas, IoU', y: 'canvases',
    });
    hist.g.selectAll('.bin').data(bins).join('rect')
      .attr('class', 'bin')
      .attr('x', (v, i) => x(i * wBin) + 1)
      .attr('width', Math.max(1, x(wBin) - x(0) - 2))
      .attr('y', (v) => y(v))
      .attr('height', (v) => hist.h - y(v))
      .attr('fill', 'var(--primary)').attr('opacity', 0.75);
    [[0.05, `isolated: ${data.dataset.isolated_test.toLocaleString('en-US')} canvases below`],
      [0.2, `crowded: ${data.dataset.crowded_test.toLocaleString('en-US')} above`]]
      .forEach(([v, label], i) => {
        hist.g.append('line')
          .attr('x1', x(v)).attr('x2', x(v)).attr('y1', 0).attr('y2', hist.h)
          .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4)
          .attr('stroke-dasharray', '4 3');
        hist.g.append('text').attr('class', 'chart-annotation')
          .attr('x', x(v) + 6).attr('y', 14 + i * 15)
          .style('font-size', '11px').style('letter-spacing', '1px')
          .text(label);
      });
    /* Where the reader's threshold sits against the overlap that is really
       there: a threshold below the bulk of it suppresses real objects. */
    hist.g.append('line')
      .attr('x1', x(Math.min(state.t, hi))).attr('x2', x(Math.min(state.t, hi)))
      .attr('y1', -6).attr('y2', hist.h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
  }

  function drawCurve() {
    chart.g.selectAll('*').remove();
    /* Each label is anchored to a different point along its own curve: four
       labels parked at the same right-hand end sit on top of each other. */
    const series = state.view === 'all'
      ? [
        { rows: N.all, colour: 'var(--primary)', dash: null, at: 13, label: 'the whole test split' },
        { rows: liveCurve, colour: 'var(--cosmos)', dash: '5 3', at: 4, label: `computed here, ${W.n_images} canvases` },
      ]
      : [
        { rows: N.isolated.test, colour: 'var(--primary)', dash: null, at: 13, label: 'isolated scenes, test' },
        { rows: N.isolated.val, colour: 'var(--primary)', dash: '4 3', at: 3, label: 'isolated, validation' },
        { rows: N.crowded.test, colour: 'var(--galaxy)', dash: null, at: 13, label: 'crowded scenes, test' },
        { rows: N.crowded.val, colour: 'var(--galaxy)', dash: '4 3', at: 3, label: 'crowded, validation' },
      ];

    const vals = series.flatMap((s) => s.rows.map((r) => r.map50));
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12;
    const x = d3.scaleLinear().domain([grid[0], grid[grid.length - 1]]).range([0, chart.w]);
    const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([chart.h, 0]);
    drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
    drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5, xTicks: 7, yFmt: d3.format('.2f') });
    axisLabels(chart.g, chart.w, chart.h, {
      x: 'suppression threshold: boxes overlapping more than this are dropped',
      y: 'mAP@0.5',
    });

    /* the interval inside which the test optimum itself cannot be located */
    if (state.view !== 'all') {
      [['isolated', 'var(--primary)'], ['crowded', 'var(--galaxy)']].forEach(([k, colour]) => {
        const ci = N[k].optimum_ci;
        chart.g.append('rect')
          .attr('x', x(ci[0])).attr('width', Math.max(2, x(ci[1]) - x(ci[0])))
          .attr('y', 0).attr('height', chart.h)
          .attr('fill', colour).attr('opacity', 0.08);
      });
    }

    series.forEach((s) => {
      chart.g.append('path').datum(s.rows).attr('fill', 'none')
        .attr('stroke', s.colour).attr('stroke-width', 2.4)
        .attr('stroke-dasharray', s.dash)
        .attr('d', d3.line().x((r) => x(r.t)).y((r) => y(r.map50)));
      const anchor = s.rows[Math.min(s.at, s.rows.length - 1)];
      chart.g.append('text')
        .attr('x', x(anchor.t)).attr('y', y(anchor.map50) - 9)
        .attr('text-anchor', 'middle')
        .attr('class', 'chart-annotation')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .attr('fill', s.colour)
        .text(s.label);
    });

    chart.g.append('line')
      .attr('x1', x(state.t)).attr('x2', x(state.t)).attr('y1', 0).attr('y2', chart.h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
    series.forEach((s) => {
      const r = rowAt(s.rows, state.t);
      chart.g.append('circle')
        .attr('cx', x(r.t)).attr('cy', y(r.map50)).attr('r', 5)
        .attr('fill', s.colour).attr('stroke', 'var(--white)').attr('stroke-width', 2);
    });

    const chosen = N.isolated.chosen;
    chart.g.append('line')
      .attr('x1', x(chosen)).attr('x2', x(chosen)).attr('y1', 0).attr('y2', chart.h)
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '2 4');
    chart.g.append('text').attr('class', 'chart-annotation')
      .attr('x', x(chosen) + 6).attr('y', 12)
      .style('font-size', '11px').style('letter-spacing', '1px')
      .text('chosen on validation');
  }

  function render() {
    document.querySelector('#nms-t-value').textContent = state.t.toFixed(2);
    drawCurve();
    drawHist();

    const here = rowAt(N.all, state.t);
    const best = bestFull;
    const worst = N.all.reduce((a, b) => (b.map50 < a.map50 ? b : a));
    const hereLive = rowAt(liveCurve, state.t);
    const iso = rowAt(N.isolated.test, state.t);
    const crowd = rowAt(N.crowded.test, state.t);
    const chosenIso = rowAt(N.isolated.test, N.isolated.chosen);
    const chosenCrowd = rowAt(N.crowded.test, N.crowded.chosen);
    const behind = best.map50 - here.map50;

    const standing = behind < 1e-9
      ? `<span class="bold">That is the best threshold on this split.</span>`
      : `That is <span class="bold">${behind.toFixed(4)}</span> mAP behind the best threshold on `
        + `this split, which is ${best.t.toFixed(2)} at ${best.map50.toFixed(4)}.`;

    document.querySelector('#nms-readout').innerHTML =
      `<span class="bold">Threshold ${state.t.toFixed(2)}: mAP@0.5 `
      + `${here.map50.toFixed(4)}</span> over ${data.dataset.n_test.toLocaleString('en-US')} canvases. `
      + `${standing} The worst setting on the same detections is ${worst.t.toFixed(2)} at `
      + `${worst.map50.toFixed(4)}, so the whole range of this one number is worth `
      + `<span class="bold">${(best.map50 - worst.map50).toFixed(4)}</span> mAP, and not one weight `
      + `changed anywhere.`
      + `<br /><br />`
      + `At this setting ${here.kept.toLocaleString('en-US')} boxes survive, at a precision of `
      + `${here.precision.toFixed(4)} and a recall of ${here.recall.toFixed(4)}: `
      + `${here.duplicates_surviving.toLocaleString('en-US')} of them are second copies of an object `
      + `that was already found, while ${here.objects_suppressed.toLocaleString('en-US')} genuine `
      + `objects were suppressed by a neighbour. Those two counts move in opposite directions and `
      + `there is no setting where both are zero: that is the trade, and it is why the answer depends `
      + `on the scene. Isolated canvases score ${iso.map50.toFixed(4)} here and crowded ones `
      + `${crowd.map50.toFixed(4)}.`
      + `<br /><br />`
      + `Chosen honestly, on validation, the threshold is ${N.isolated.chosen.toFixed(2)} for isolated `
      + `scenes and ${N.crowded.chosen.toFixed(2)} for crowded ones, and on test those score `
      + `${chosenIso.map50.toFixed(4)} and ${chosenCrowd.map50.toFixed(4)}. `
      + (N.isolated.chosen === N.crowded.chosen
        ? `The two strata chose the <span class="bold">same</span> threshold, which is the more `
          + `interesting outcome: the curves are flat enough that the test optimum itself cannot be `
          + `pinned down more precisely than [${N.isolated.optimum_ci.join(', ')}] for isolated scenes `
          + `and [${N.crowded.optimum_ci.join(', ')}] for crowded ones. Tuning this number past the `
          + `first decimal is fitting the noise.`
        : `The two strata chose different thresholds, which is the case for tuning suppression per `
          + `scene rather than per repository.`)
      + `<br /><br />`
      + `<span style="color:#5b6572">Soft suppression instead of hard: `
      + `${N.variants.map((v) => `${v.mode} ${v.map50.toFixed(4)}`).join(', ')}. `
      + `Suppression also costs arithmetic: ${N.cost_note}, `
      + `${here.pair_tests.toLocaleString('en-US')} of them here, and the class-agnostic variant `
      + `pays ${N.variants.find((v) => v.mode === 'class-agnostic').pair_tests.toLocaleString('en-US')} `
      + `because every box has to be compared with every other one rather than only with its own `
      + `class. The dashed line is this browser running the same suppression over the `
      + `${W.n_images} canvases of ranked detections shipped with the page: `
      + `${hereLive.kept.toLocaleString('en-US')} boxes survive here and it scores `
      + `${hereLive.map50.toFixed(4)}, `
      + `${hereLive.map50 < here.map50
        ? `below the full split's ${here.map50.toFixed(4)} because only the `
          + `${Math.round(W.detections.length / W.n_images)} best detections per canvas were shipped `
          + `and the rest of the ranking, which average precision needs in order to reach full `
          + `recall, is not there`
        : `above the full split's ${here.map50.toFixed(4)}, which the truncation to the `
          + `${Math.round(W.detections.length / W.n_images)} best detections per canvas can do by `
          + `removing low-ranked false positives`}. `
      + `What it is here to show is the shape: it peaks at ${bestLive.t.toFixed(2)}, `
      + `${argmaxAgrees ? 'the same threshold as' : 'a DIFFERENT threshold from'} the full split.</span>`;
  }

  /* ---- controls -------------------------------------------------------- */
  const slider = document.querySelector('#nms-t');
  slider.min = '0';
  slider.max = String(grid.length - 1);
  /* the chosen threshold need not be a grid value to the bit, so index by
     proximity rather than by identity */
  let startIdx = 0;
  grid.forEach((v, i) => {
    if (Math.abs(v - state.t) < Math.abs(grid[startIdx] - state.t)) startIdx = i;
  });
  state.t = grid[startIdx];
  slider.value = String(startIdx);
  slider.addEventListener('input', () => { state.t = grid[+slider.value]; render(); });

  const viewRow = document.querySelector('#nms-view-buttons');
  const views = [['all', 'every scene together'], ['split', 'isolated against crowded']];
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

  render();
  return { argmaxAgrees, bestLive: bestLive.t, bestFull: bestFull.t };
}
