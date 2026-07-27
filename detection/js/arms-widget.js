/* Who is responsible for each object, and what the answer is worth.
 *
 * Every arm here trains the same trunk on the same data with the same schedule
 * and changes exactly one thing, which is the only way an ablation means
 * anything. Two of the arms were also trained at three seeds, and the spread
 * that produced is drawn as a band around whichever arm is selected: anything
 * inside it is a result this experiment cannot tell apart from re-rolling the
 * dice, and saying so is the point.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const SHOWN = ['slots1', 'slots3-same', 'slots3-kmeans', 'slots3-square',
  'stride16', 'stride4', 'boxparam-giou'];

const CHANGED = {
  slots1: 'one slot per cell',
  'slots3-same': 'three slots, identical priors',
  'slots3-kmeans': 'three slots, priors from k-means',
  'slots3-square': 'three slots, identical squares',
  stride16: 'the same net, grid at stride 16',
  stride4: 'the same net, grid at stride 4',
  'boxparam-giou': 'GIoU on the box instead of smooth L1',
};

export function initArmsWidget(data) {
  const arms = SHOWN.map((id) => data.arms.find((a) => a.id === id)).filter(Boolean);
  const spread = Math.max(...data.verdicts.map((v) => v.seed_spread));
  const ref = arms.find((a) => a.id === 'slots3-kmeans') || arms[0];

  const state = { view: 'map50', picked: arms.indexOf(ref) };
  const chart = makeChart('#arms-chart', {
    width: 700, height: 400, margin: { top: 36, right: 26, bottom: 74, left: 76 },
  });

  /* ---- the 8x8 map of which cell ends up answering ---------------------- */
  function drawCells() {
    const cm = data.grid.cell_map;
    const n = cm.length;
    const cell = 26;
    const pad = 34;
    const wrap = d3.select('#arms-cells');
    wrap.selectAll('*').remove();
    const svg = wrap.append('svg')
      .attr('viewBox', `0 0 ${n * cell + pad * 2} ${n * cell + 62}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('max-width', `${n * cell + pad * 2}px`)
      .style('margin', '0 auto').style('display', 'block');
    const flat = cm.flat();
    const hi = Math.max(...flat);
    const colour = d3.scaleLinear().domain([0, hi])
      .range(['#f1f3f3', 'var(--primary)']);
    const g = svg.append('g').attr('transform', `translate(${pad},26)`);
    cm.forEach((row, r) => row.forEach((v, c) => {
      g.append('rect')
        .attr('x', c * cell).attr('y', r * cell)
        .attr('width', cell - 1).attr('height', cell - 1)
        .attr('fill', colour(v));
      g.append('text')
        .attr('x', c * cell + cell / 2).attr('y', r * cell + cell / 2 + 4)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 9)
        .attr('fill', v > hi * 0.55 ? '#ffffff' : '#5b6572')
        .text(v);
    }));
    /* This panel is only 276 units wide, so the caption has to be short enough
       to fit inside it at a size that is still legible on a phone. */
    svg.append('text').attr('x', (n * cell + pad * 2) / 2).attr('y', 14)
      .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
      .style('font-size', '11px').style('letter-spacing', '0.5px')
      .text(`test objects per cell, stride ${data.arms.find((q) => q.id === 'slots3-kmeans').stride}`);
    /* A heatmap without a legend is decoration, so the scale is drawn. */
    const lw = n * cell;
    const bar = svg.append('g').attr('transform', `translate(${pad},${26 + n * cell + 12})`);
    const gradId = 'arms-cell-grad';
    const defs = svg.append('defs');
    const grad = defs.append('linearGradient').attr('id', gradId);
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#f1f3f3');
    grad.append('stop').attr('offset', '100%').attr('stop-color', 'var(--primary)');
    bar.append('rect').attr('width', lw).attr('height', 8)
      .attr('fill', `url(#${gradId})`).attr('stroke', 'var(--stone)');
    bar.append('text').attr('x', 0).attr('y', 22)
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 10)
      .attr('fill', '#5b6572').text('0');
    bar.append('text').attr('x', lw).attr('y', 22).attr('text-anchor', 'end')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 10)
      .attr('fill', '#5b6572').text(`${hi} objects`);
  }

  /* ---- the arms ---------------------------------------------------------- */
  function drawArms() {
    chart.g.selectAll('*').remove();
    const x = d3.scaleBand().domain(arms.map((a) => a.id)).range([0, chart.w]).padding(0.35);

    if (state.view === 'map50') {
      const los = arms.map((a) => a.map50_ci[0]);
      const his = arms.map((a) => a.map50_ci[1]);
      const y = d3.scaleLinear()
        .domain([Math.min(...los) - 0.02, Math.max(...his) + 0.015]).range([chart.h, 0]);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5, yFmt: d3.format('.2f'), xFmt: () => '' });
      axisLabels(chart.g, chart.w, chart.h, { y: 'mAP@0.5 on the test split' });

      const sel = arms[state.picked];
      chart.g.append('rect')
        .attr('x', 0).attr('width', chart.w)
        .attr('y', y(sel.map50 + spread))
        .attr('height', Math.max(2, y(sel.map50 - spread) - y(sel.map50 + spread)))
        .attr('fill', '#8a94a6').attr('opacity', 0.18);
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w - 4).attr('y', y(sel.map50 + spread) - 6).attr('text-anchor', 'end')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .text('inside this band is a different random seed');

      chart.g.selectAll('.whisk').data(arms).join('line')
        .attr('class', 'whisk')
        .attr('x1', (a) => x(a.id) + x.bandwidth() / 2)
        .attr('x2', (a) => x(a.id) + x.bandwidth() / 2)
        .attr('y1', (a) => y(a.map50_ci[0]))
        .attr('y2', (a) => y(a.map50_ci[1]))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);

      /* every seed this arm was trained at, where there is more than one */
      const reps = new Map(data.seed_spread.map((s) => [s.arm, s.map50]));
      arms.forEach((a) => {
        const vals = reps.get(a.id);
        if (!vals) return;
        vals.forEach((v) => {
          chart.g.append('line')
            .attr('x1', x(a.id) + x.bandwidth() * 0.26)
            .attr('x2', x(a.id) + x.bandwidth() * 0.74)
            .attr('y1', y(v)).attr('y2', y(v))
            .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.1)
            .attr('opacity', 0.75);
        });
      });

      chart.g.selectAll('.dot').data(arms).join('circle')
        .attr('class', 'dot')
        .attr('cx', (a) => x(a.id) + x.bandwidth() / 2)
        .attr('cy', (a) => y(a.map50))
        .attr('r', (a, i) => (i === state.picked ? 8 : 6))
        .attr('fill', (a, i) => (i === state.picked ? 'var(--cosmos)' : 'var(--primary)'))
        .attr('stroke', 'var(--white)').attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('click', (ev, a) => { state.picked = arms.indexOf(a); render(); });
    } else if (state.view === 'iou') {
      const sweep = data.eval.iou_sweep;
      const y = d3.scaleLinear().domain([0, Math.max(...arms.map((a) => a.map_by_iou[0])) * 1.06])
        .range([chart.h, 0]);
      const xl = d3.scaleLinear().domain([sweep[0], sweep[sweep.length - 1]]).range([0, chart.w]);
      drawGrid(chart.g, xl, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, xl, y, chart.w, chart.h, { yTicks: 5, xTicks: 6, yFmt: d3.format('.2f') });
      axisLabels(chart.g, chart.w, chart.h, {
        x: 'how much overlap counts as a hit', y: 'mean average precision',
      });
      arms.forEach((a, i) => {
        const on = i === state.picked;
        chart.g.append('path')
          .datum(a.map_by_iou.map((v, j) => [sweep[j], v]))
          .attr('fill', 'none')
          .attr('stroke', on ? 'var(--cosmos)' : 'var(--primary)')
          .attr('stroke-width', on ? 3 : 1.4)
          .attr('opacity', on ? 1 : 0.45)
          .attr('d', d3.line().x((p) => xl(p[0])).y((p) => y(p[1])))
          .style('cursor', 'pointer')
          .on('click', () => { state.picked = i; render(); });
      });
      const sel = arms[state.picked];
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', xl(sweep[2])).attr('y', y(sel.map_by_iou[2]) - 10)
        .style('font-size', '11px').style('letter-spacing', '1px')
        .attr('fill', 'var(--cosmos)').text(sel.id);
    } else {
      const y = d3.scaleLinear().domain([0, 1]).range([chart.h, 0]);
      const inner = d3.scaleBand().domain([0, 1, 2])
        .range([0, x.bandwidth()]).padding(0.12);
      drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
      drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 5, yFmt: d3.format('.1f'), xFmt: () => '' });
      axisLabels(chart.g, chart.w, chart.h, { y: 'objects found at IoU 0.5' });
      const tone = ['0.35', '0.65', '1'];
      arms.forEach((a, i) => {
        a.recall_by_size.forEach((v, k) => {
          chart.g.append('rect')
            .attr('x', x(a.id) + inner(k)).attr('width', inner.bandwidth())
            .attr('y', y(v)).attr('height', chart.h - y(v))
            .attr('fill', i === state.picked ? 'var(--cosmos)' : 'var(--primary)')
            .attr('opacity', tone[k])
            .style('cursor', 'pointer')
            .on('click', () => { state.picked = i; render(); });
        });
      });
      chart.g.append('text').attr('class', 'chart-annotation')
        .attr('x', chart.w / 2).attr('y', -14).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('letter-spacing', '1px')
        .text(`smallest third, middle third, largest third (cut at ${ref.size_terciles.join(' and ')} px²)`);
    }

    chart.g.selectAll('.armlabel').data(arms).join('text')
      .attr('class', 'armlabel')
      .attr('x', (a) => x(a.id) + x.bandwidth() / 2)
      .attr('y', chart.h + 18)
      .attr('text-anchor', 'end')
      .attr('transform', (a) => `rotate(-28,${x(a.id) + x.bandwidth() / 2},${chart.h + 18})`)
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
      .attr('fill', (a, i) => (i === state.picked ? 'var(--cosmos)' : 'var(--squidink)'))
      .style('cursor', 'pointer')
      .style('display', state.view === 'iou' ? 'none' : null)
      .text((a) => a.id)
      .on('click', (ev, a) => { state.picked = arms.indexOf(a); render(); });
  }

  function render() {
    drawArms();
    drawCells();
    const a = arms[state.picked];
    const rep = data.seed_spread.find((s) => s.arm === a.id);
    const verdict = data.verdicts.find((v) => v.pair[0] === a.id);
    const against = verdict ? data.arms.find((q) => q.id === verdict.pair[1]) : null;
    /* what the pair was built to isolate, in the generator's own words */
    const isolates = (data.anchors.comparisons.find((c) =>
      verdict && c.pair.includes(a.id) && c.pair.includes(verdict.pair[1])) || {}).isolates;
    const valBest = a.nms_val_curve.reduce((p, q) => (q.map50 > p.map50 ? q : p));
    const resolvedCount = data.verdicts.filter((v) => v.resolved).length;
    const classes = data.meta.classes;
    const bestCls = a.ap_per_class.indexOf(Math.max(...a.ap_per_class));
    const worstCls = a.ap_per_class.indexOf(Math.min(...a.ap_per_class));

    document.querySelector('#arms-readout').innerHTML =
      `<span class="bold">${a.id}: ${CHANGED[a.id]}.</span> mAP@0.5 `
      + `<span class="bold">${a.map50.toFixed(4)}</span> `
      + `[${a.map50_ci[0].toFixed(4)}, ${a.map50_ci[1].toFixed(4)}], mAP@0.75 ${a.map75.toFixed(4)}, `
      + `mAP@[.5:.95] ${a.map5095.toFixed(4)}, mean IoU of a matched box `
      + `${a.mean_matched_iou.toFixed(4)}. ${a.params.toLocaleString('en-US')} parameters, `
      + `${(a.macs / 1e6).toFixed(2)} MMAC per canvas, ${a.seconds.toFixed(0)}s to train at seed `
      + `${a.seed} with a ${a.box_mode} box loss, training loss from `
      + `${a.history[0].toFixed(2)} to ${a.history[a.history.length - 1].toFixed(2)}. `
      + `Easiest class ${classes[bestCls]} at ${a.ap_per_class[bestCls].toFixed(4)}, hardest `
      + `${classes[worstCls]} at ${a.ap_per_class[worstCls].toFixed(4)}. `
      + `Priors: ${a.anchors.map((s) => `${s[0]}×${s[1]}`).join(', ')}; the `
      + `${a.assignment.boxes.toLocaleString('en-US')} training boxes landed on them `
      + `${a.positives_per_slot.join(', ')} times, with ${a.assignment.collisions} collisions and `
      + `${a.assignment.unassignable} left with nowhere to go. Its suppression threshold was picked `
      + `on validation, where ${valBest.t.toFixed(2)} scored ${valBest.map50.toFixed(4)}, and it kept `
      + `${a.detections_kept.toLocaleString('en-US')} detections on test.`
      + `<br /><br />`
      + (verdict && against
        ? `Against <span class="bold">${verdict.pair[1]}</span>, the arm it is meant to be compared `
          + `with${isolates ? `, which isolates ${isolates}` : ''}: `
          + `${verdict.delta >= 0 ? '+' : ''}${verdict.delta.toFixed(4)} mAP, against a seed `
          + `spread of ${verdict.seed_spread.toFixed(4)} measured over ${verdict.seeds} seeds. `
          + `<span class="bold">${verdict.resolved
            ? 'That clears the noise floor.'
            : 'That does not clear the noise floor, so this experiment does not resolve it.'}</span> `
        : a.id === ref.id
          ? `This arm is the reference the others are compared against. `
          /* an arm with no verdict of its own still gets compared, to the
             reference, rather than left standing on a number with no scale */
          : `It has no verdict row of its own, so against the reference arm ${ref.id}: `
            + `${a.map50 - ref.map50 >= 0 ? '+' : ''}${(a.map50 - ref.map50).toFixed(4)} at IoU 0.5, `
            + `<span class="bold">${Math.abs(a.map50 - ref.map50) > spread
              ? 'outside the seed spread' : 'inside the seed spread'}</span>, but `
            + `${a.map75 - ref.map75 >= 0 ? '+' : ''}${(a.map75 - ref.map75).toFixed(4)} at IoU 0.75, `
            + `${Math.abs(a.map75 - ref.map75) > spread ? 'which is not noise either' : 'inside it'}. `
            + `${a.map75 > ref.map75 && a.map50 < ref.map50
              ? 'A loss that optimises overlap directly buys precision of placement rather than more '
                + 'detections, which is exactly the trade its name promises. '
              : ''}`)
      + (rep
        ? `Trained at ${rep.n_seeds} seeds it scored ${rep.map50.map((v) => v.toFixed(4)).join(', ')}, `
          + `a spread of ${rep.spread.toFixed(4)}. `
        : `Trained at one seed; the band comes from the arms that were replicated. `)
      + `Everything else was held fixed: ${data.arm_control}.`
      + `<br /><br />`
      + `<span style="color:#5b6572">Of the ${data.verdicts.length} comparisons this factorial was `
      + `built to make, ${resolvedCount} clear the seed spread and `
      + `${data.verdicts.length - resolvedCount} do not. The grid on the right is where the `
      + `responsibility lands: at stride ${data.arms.find((q) => q.id === 'slots3-kmeans').stride} the `
      + `outer ring is ${(data.grid.border_share * 100).toFixed(2)}% of the assignments, because no `
      + `object in these canvases is ever cut off by the edge of the picture. A real photograph would `
      + `light that ring up, and that is a property of this dataset rather than of detection.</span>`;
  }

  const viewRow = document.querySelector('#arms-view-buttons');
  const views = [['map50', 'the headline score'], ['iou', 'how strict the overlap has to be'],
    ['size', 'by how big the object is']];
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
  return { spread, arms: arms.length };
}
