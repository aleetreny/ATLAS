/* The window bill, and the shapes you would have had to guess.
 *
 * Article 19 slid one 25-pixel window across a photograph and counted the
 * evaluations. Nothing about that argument changes when the classifier becomes
 * a convolutional network, except the size of the bill: an object can be
 * anywhere, at any size, at any shape, so the honest version of "slide a
 * classifier" enumerates all three.
 *
 * The counts here are recomputed in the browser from the reader's settings and
 * land exactly on the five the generator exported, which is the check that the
 * arithmetic on screen is the arithmetic that was measured.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const CLASS_COLOUR = ['var(--anchor)', 'var(--cosmos)', 'var(--aqua)', 'var(--galaxy)', 'var(--smile)'];

/* The generator's window enumeration, re-derived rather than copied: a size
   and an aspect give a shape, and a shape and a stride give a count. The canvas
   size comes from the export, so a different dataset moves these counts instead
   of quietly breaking the check below. */
const shapeFor = (s, a) => [Math.round(s * Math.sqrt(a)), Math.round(s / Math.sqrt(a))];
const positions = (canvas, w, h, stride) =>
  Math.ceil((canvas - w + 1) / stride) * Math.ceil((canvas - h + 1) / stride);

/* n sizes means the n most central of the exported list, so that one size is
   28 px and the count matches the exported single-size figure. */
function pickSizes(all, n) {
  const mid = (all.length - 1) / 2;
  return [...all].sort((p, q) => Math.abs(all.indexOf(p) - mid) - Math.abs(all.indexOf(q) - mid))
    .slice(0, n).sort((p, q) => p - q);
}
const pickAspects = (all, n) => (n === 1 ? [1.0] : n === 2 ? [0.5, 1.0] : [...all]);

export function initWindowsWidget(data) {
  const W = data.windows;
  const CANVAS = data.meta.canvas;
  const MACW = W.macs.window_classifier;
  const STRIDES = [1, 2, 4, 8];

  /* ---- the guard: the live arithmetic must land on the exported counts --- */
  const bill = (nS, nA, stride) => {
    const shapes = [];
    for (const s of pickSizes(W.sizes, nS)) {
      for (const a of pickAspects(W.aspects, nA)) shapes.push(shapeFor(s, a));
    }
    return {
      shapes,
      windows: shapes.reduce((t, [w, h]) => t + positions(CANVAS, w, h, stride), 0),
    };
  };
  const full = bill(5, 3, 1);
  const checks = [
    ['15 shapes, stride 1', full.windows, W.exhaustive_five_by_three],
    ['15 shapes, stride 4', bill(5, 3, 4).windows, W.exhaustive_five_by_three_stride4],
    ['one shape, stride 1', bill(1, 1, 1).windows, W.exhaustive_one_size],
    ['one shape, stride 4', bill(1, 1, 4).windows, W.exhaustive_one_size_stride4],
    ['five sizes, one aspect', bill(5, 1, 1).windows, W.exhaustive_five_sizes],
  ];
  const shapesMatch = JSON.stringify(full.shapes) === JSON.stringify(W.shapes);
  const failed = checks.filter(([, mine, theirs]) => mine !== theirs);
  failed.forEach(([name, mine, theirs]) =>
    console.warn(`window count "${name}": browser ${mine}, generator ${theirs}`));
  if (!shapesMatch) console.warn('the browser enumerated different window shapes');

  const state = { view: 'bill', nS: 5, nA: 3, strideIdx: 0, k: 3 };

  const chart = makeChart('#windows-chart', {
    width: 700, height: 400, margin: { top: 34, right: 26, bottom: 62, left: 78 },
  });

  /* ---- view one: what an exhaustive scan costs -------------------------- */
  function drawBill() {
    const stride = STRIDES[state.strideIdx];
    const mine = bill(state.nS, state.nA, stride);
    const rows = [
      { key: 'yours', label: 'your settings', windows: mine.windows, macs: mine.windows * MACW, live: true },
      { key: 's1', label: '15 shapes, stride 1', windows: W.exhaustive_five_by_three, macs: W.macs.naive_scan },
      { key: 's4', label: '15 shapes, stride 4', windows: W.exhaustive_five_by_three_stride4, macs: W.macs.conv_scan_equivalent },
      { key: 'det', label: 'one pass of the detector', windows: W.detector_candidates, macs: W.macs.detector_s8 },
    ];

    chart.g.selectAll('*').remove();
    const x = d3.scaleBand().domain(rows.map((r) => r.key)).range([0, chart.w]).padding(0.28);
    const y = d3.scaleLog()
      .domain([W.macs.detector_s8 / 3e6, (W.macs.naive_scan / 1e6) * 2.2]).range([chart.h, 0]);
    drawGrid(chart.g, x, y, chart.w, chart.h, { yTicks: 5 });
    drawAxes(chart.g, x, y, chart.w, chart.h, {
      yTicks: 5,
      yFmt: (v) => (v >= 1 ? d3.format('~s')(v) : d3.format('.2f')(v)),
      xFmt: () => '',
    });
    axisLabels(chart.g, chart.w, chart.h, { y: 'million multiply-accumulates (log)' });

    chart.g.selectAll('.bar').data(rows).join('rect')
      .attr('class', 'bar')
      .attr('x', (r) => x(r.key))
      .attr('width', x.bandwidth())
      .attr('y', (r) => y(r.macs / 1e6))
      .attr('height', (r) => chart.h - y(r.macs / 1e6))
      .attr('fill', (r) => (r.live ? 'var(--cosmos)' : 'var(--primary)'))
      .attr('opacity', (r) => (r.live ? 1 : 0.62));

    chart.g.selectAll('.blabel').data(rows).join('text')
      .attr('class', 'blabel')
      .attr('x', (r) => x(r.key) + x.bandwidth() / 2)
      .attr('y', (r) => y(r.macs / 1e6) - 8)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 12)
      .attr('fill', 'var(--squidink)')
      .text((r) => `${(r.macs / 1e6).toFixed(r.macs / 1e6 < 10 ? 2 : 0)} MMAC`);

    chart.g.selectAll('.xlabel').data(rows).join('text')
      .attr('class', 'xlabel')
      .attr('x', (r) => x(r.key) + x.bandwidth() / 2)
      .attr('y', chart.h + 20)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
      .attr('fill', 'var(--squidink)')
      .text((r) => r.label);
    chart.g.selectAll('.xcount').data(rows).join('text')
      .attr('class', 'xcount')
      .attr('x', (r) => x(r.key) + x.bandwidth() / 2)
      .attr('y', chart.h + 36)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 11)
      .attr('fill', '#5b6572')
      .text((r) => `${r.windows.toLocaleString('en-US')} windows`);

    const t = W.timing;
    const eq = W.equivalence;
    const sh = W.sharing;
    document.querySelector('#windows-readout').innerHTML =
      `<span class="bold">${mine.shapes.length} shape${mine.shapes.length === 1 ? '' : 's'} `
      + `at stride ${stride}: ${mine.windows.toLocaleString('en-US')} windows</span>, `
      + `${(mine.windows * MACW / 1e6).toFixed(1)} million multiply-accumulates per canvas, `
      + `against <span class="bold">${(W.macs.detector_s8 / 1e6).toFixed(2)}</span> for one pass of the `
      + `detector over the same picture. The detector scores `
      + `${W.detector_candidates} candidates, three at each of its ${W.detector_cells} cells.`
      + `<br /><br />`
      + `Where the difference goes, in three factors whose product is the whole ratio of `
      + `${W.total_ratio.toFixed(1)}×: ${W.factors.map((f) =>
        `<span class="bold">${f.value.toFixed(2)}×</span> for ${f.name}`).join(', ')}. `
      + `Multiplied out here in the browser they come to `
      + `${W.factors.reduce((t, f) => t * f.value, 1).toFixed(1)}×, against the `
      + `${W.factor_product.toFixed(1)}× the generator recorded. `
      + `${W.attribution_note.charAt(0).toUpperCase()}${W.attribution_note.slice(1)}.`
      + `<br /><br />`
      + `Multiply-accumulates are not seconds, so both were timed over ${t.n_images} canvases: `
      + `<span class="bold">${t.naive_seconds.toFixed(2)}s</span> to crop and resize every window `
      + `and run them one by one, <span class="bold">${t.conv_seconds.toFixed(2)}s</span> to get the `
      + `same answers by resizing the picture instead of the window `
      + `(${t.speedup.toFixed(1)}× faster), and ${t.detector_seconds.toFixed(3)}s for the detector.`
      + `<br /><br />`
      + `<span style="color:#5b6572">The counts above are recomputed here from the sizes, aspect `
      + `ratios and stride; ${checks.length - failed.length} of ${checks.length} land exactly on the `
      + `generator's figures${failed.length ? ` (${failed.length} do NOT: see the console)` : ''}, and `
      + `the ${W.shapes.length} enumerated shapes ${shapesMatch ? 'match' : 'DO NOT match'} the exported `
      + `list. Sharing: ${W.detector_cells} crops of 28×28 through this trunk cost `
      + `${(sh.crop_macs / 1e6).toFixed(2)} MMAC against ${(sh.full_macs / 1e6).toFixed(2)} for one `
      + `64×64 pass, a factor of ${sh.factor.toFixed(2)}, and those crops cover `
      + `${sh.area_redundancy.toFixed(2)} times the area of the canvas. `
      + `${sh.note.charAt(0).toUpperCase()}${sh.note.slice(1)}: the ratio between them is `
      + `${sh.residual.toFixed(3)}, and the ${(100 * (1 - sh.residual)).toFixed(1)}% that is missing is `
      + `the padding at the edges. Run the window classifier on a crop and on the matching cell of a `
      + `full-picture pass and the two answers differ by `
      + `<span class="bold">${eq.winclf_gap === 0
        ? 'exactly nothing' : eq.winclf_gap.toExponential(3)}</span>.</span>`;
  }

  /* ---- view two: the shapes that are actually in the data ---------------- */
  function drawShapes() {
    const pts = data.anchors.box_shapes;
    const cls = data.anchors.box_classes;
    const kRow = data.anchors.k_curve.find((r) => r.k === state.k);
    const lim = 52;

    chart.g.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, lim]).range([0, chart.w]);
    const y = d3.scaleLinear().domain([0, lim]).range([chart.h, 0]);
    drawGrid(chart.g, x, y, chart.w, chart.h, { xTicks: 6, yTicks: 6 });
    drawAxes(chart.g, x, y, chart.w, chart.h, { xTicks: 6, yTicks: 6 });
    axisLabels(chart.g, chart.w, chart.h, { x: 'box width, px', y: 'box height, px' });

    chart.g.append('g').selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 2.4)
      .attr('fill', (p, i) => CLASS_COLOUR[cls[i]])
      .attr('opacity', 0.32);

    /* the 15 shapes an exhaustive scan would enumerate, blind to the data */
    chart.g.append('g').selectAll('path').data(W.shapes).join('path')
      .attr('d', d3.symbol().type(d3.symbolCross).size(70))
      .attr('transform', (s) => `translate(${x(s[0])},${y(s[1])}) rotate(45)`)
      .attr('fill', '#5b6572').attr('opacity', 0.8);

    /* the k shapes fitted to the data */
    chart.g.append('g').selectAll('circle').data(kRow.shapes).join('circle')
      .attr('cx', (s) => x(s[0])).attr('cy', (s) => y(s[1]))
      .attr('r', 9)
      .attr('fill', 'var(--primary)').attr('stroke', 'var(--white)').attr('stroke-width', 2.5);

    chart.g.append('text').attr('class', 'chart-annotation')
      .attr('x', chart.w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('letter-spacing', '1px')
      .text(`${pts.length} test boxes, ${W.shapes.length} enumerated shapes (grey), ${state.k} fitted (filled)`);

    const used = data.anchors.sets['slots3-kmeans'];
    const usedK = data.anchors.k_curve.find((r) => r.k === used.length);
    document.querySelector('#windows-readout').innerHTML =
      `<span class="bold">${state.k} shape${state.k === 1 ? '' : 's'} fitted to the training boxes `
      + `cover ${kRow.test_iou.toFixed(4)}</span> of a held-out box on average, measured as the IoU `
      + `between each test box and the closest fitted shape placed on top of it. Fitted on train, `
      + `read on test: ${kRow.train_iou.toFixed(4)} against ${kRow.test_iou.toFixed(4)}, so this is a `
      + `property of the data and not of the fit.`
      + `<br /><br />`
      + `The grey crosses are the ${W.shapes.length} shapes an exhaustive scan enumerates: `
      + `${W.sizes.length} sizes times ${W.aspects.length} aspect ratios, chosen before anyone looked `
      + `at a garment. The filled circles are what k-means on the training boxes returns. `
      + `The detector in this article uses <span class="bold">${used.length}</span> of them, covering `
      + `${usedK.test_iou.toFixed(4)}, and the reason it stops at ${used.length} rather than `
      + `${data.anchors.k_curve[data.anchors.k_curve.length - 1].k} is that every extra shape is `
      + `another ${(5 + data.meta.classes.length)} channels on the head at every cell, `
      + `paid for on every image forever.`
      + `<br /><br />`
      + `<span style="color:#5b6572">A prior is not a constraint here: the head predicts a size as `
      + `<i>anchor × exp(t)</i>, so any box is reachable from any prior. What a prior buys is the `
      + `starting point, and with all three slots the detector scores `
      + `${data.arms.find((a) => a.id === 'slots3-kmeans').map50.toFixed(4)} against `
      + `${data.anchors.slot0_only.map50.toFixed(4)} when only the first slot is allowed to answer. `
      + `The three slots are not decoration.</span>`;
  }

  function render() {
    document.querySelector('#windows-bill-controls')
      .style.display = state.view === 'bill' ? '' : 'none';
    document.querySelector('#windows-shape-controls')
      .style.display = state.view === 'shapes' ? '' : 'none';
    if (state.view === 'bill') drawBill(); else drawShapes();
  }

  /* ---- controls -------------------------------------------------------- */
  const viewRow = document.querySelector('#windows-view-buttons');
  const views = [['bill', 'the bill for guessing'], ['shapes', 'the shapes really there']];
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

  const sSlider = document.querySelector('#win-sizes');
  const aSlider = document.querySelector('#win-aspects');
  const stSlider = document.querySelector('#win-stride');
  const kSlider = document.querySelector('#win-k');
  kSlider.max = String(data.anchors.k_curve.length);

  function labelControls() {
    const sizes = pickSizes(W.sizes, state.nS);
    document.querySelector('#win-sizes-value').textContent =
      `${state.nS} (${sizes.join(', ')} px)`;
    document.querySelector('#win-aspects-value').textContent =
      `${state.nA} (${pickAspects(W.aspects, state.nA).map((a) => a.toFixed(1)).join(', ')})`;
    document.querySelector('#win-stride-value').textContent = String(STRIDES[state.strideIdx]);
    document.querySelector('#win-k-value').textContent = String(state.k);
  }

  sSlider.addEventListener('input', () => { state.nS = +sSlider.value; labelControls(); render(); });
  aSlider.addEventListener('input', () => { state.nA = +aSlider.value; labelControls(); render(); });
  stSlider.addEventListener('input', () => { state.strideIdx = +stSlider.value; labelControls(); render(); });
  kSlider.addEventListener('input', () => { state.k = +kSlider.value; labelControls(); render(); });

  labelControls();
  render();
  return { failed: failed.length, shapesMatch };
}
