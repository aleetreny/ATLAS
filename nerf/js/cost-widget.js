/* What a frame costs, twice: in arithmetic and on your machine.
 *
 * The bars are multiply-accumulates counted off the two algorithms, which is
 * the same number on every machine and does not move when something else on
 * the laptop wakes up. The clock is the reader's own, measured here, as the
 * best of several runs, because a wall clock measures contention as much as it
 * measures work: the same row of the same table in the previous article came
 * out three times slower with a browser auditing next to it.
 */
import { renderTruth, camera, rayDir, psnr } from './truthkit.js';
import { renderField } from './nerfkit.js';
import { renderSplats } from './splatkit.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { makeFrame, drawTiles, seriesLabel, fmt } from './panels.js';

/* The two models side by side at a camera inside the photographed band and at
   one below it, with the difference maps under the same amplification. Drawn
   from the generator's own renders rather than live, because these two
   cameras are the ones every number in this section refers to. */
function drawCompare(data, sheets) {
  const cfg = data.sheets.compare;
  const host = document.querySelector('#compare-sheet');
  if (!host || !sheets.compare) return;
  host.innerHTML = '';
  cfg.rows.forEach((title, r) => {
    const h = document.createElement('div');
    h.className = 'cap';
    h.style.margin = '0.8rem 0 0.2rem';
    h.innerHTML = `<b>${title}</b>`;
    host.appendChild(h);
    const wrap = document.createElement('div');
    host.appendChild(wrap);
    drawTiles(wrap, sheets.compare.tiles.slice(r * cfg.cols, (r + 1) * cfg.cols),
      cfg.labels, cfg.tile, { max: 120 });
  });
}

export function initCostWidget(data, scene, field, cloud, sheets) {
  const meta = data.meta;
  const cost = data.cost;
  const res = cost.res;
  const steps = cost.steps;
  const cam = camera(data.cameras.inside[0].az, data.cameras.inside[0].el, meta);

  const host = document.querySelector('#cost-frames');
  host.innerHTML = '';
  const truthF = makeFrame(host, 'the scene', { max: 200 });
  const fieldF = makeFrame(host, 'the field', { max: 200 });
  const blobF = makeFrame(host, 'the blobs', { max: 200 });
  const readout = document.querySelector('#cost-readout');
  const buttons = document.querySelector('#cost-buttons');
  const measured = { field: null, blob: null, truth: null };

  const bars = [
    { key: 'field', label: 'field: network', colour: 'var(--primary)',
      value: cost.nerf.macs_per_frame,
      note: `${fmt.n0(cost.pixels)} px × ${steps} samples × ${fmt.pc(cost.nerf.live_fraction, 0)} kept `
        + `× ${fmt.n0(cost.nerf.macs_per_point)}` },
    { key: 'proj', label: 'blobs: projection', colour: 'var(--anchor)',
      value: cost.splat.blobs * cost.splat.project_macs,
      note: `${fmt.n0(cost.splat.blobs)} blobs × ${cost.splat.project_macs}` },
    { key: 'raster', label: 'blobs: per pixel', colour: 'var(--cosmos)',
      value: Math.round(cost.pixels * cost.splat.touched_per_pixel * cost.splat.touch_macs),
      note: `${fmt.n0(cost.pixels)} px × ${cost.splat.touched_per_pixel} touched × ${cost.splat.touch_macs}` },
  ];

  const drawChart = () => {
    const node = document.querySelector('#cost-chart');
    d3.select(node).selectAll('*').remove();
    const c = makeChart(node, { width: 640, height: 340, margin: { top: 36, right: 34, bottom: 58, left: 150 } });
    const lo = Math.min(...bars.map((b) => b.value));
    const hi = Math.max(...bars.map((b) => b.value));
    const x = d3.scaleLog().domain([lo / 3, hi * 3.4]).range([0, c.w]);
    const y = d3.scaleBand().domain(bars.map((b) => b.label)).range([0, c.h]).padding(0.42);
    const xv = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10].filter((v) => v >= lo / 3 && v <= hi * 3.4);
    drawGrid(c.g, x, y, c.w, c.h, { xValues: xv });
    drawAxes(c.g, x, y, c.w, c.h, { xValues: xv, xFmt: d3.format('~s') });
    axisLabels(c.g, c.w, c.h, { x: 'multiply-accumulates for one frame' });
    c.g.selectAll('rect.b').data(bars).join('rect').attr('class', 'b')
      .attr('x', 0).attr('y', (d) => y(d.label)).attr('height', y.bandwidth())
      .attr('width', (d) => x(d.value)).attr('fill', (d) => d.colour);
    /* The count at the end of the bar, abbreviated, and the arithmetic that
       produced it in the gap underneath. Both on one line and unabbreviated ran
       the longest of them 488 units off the edge of the canvas; the exact
       counts are in the readout, where there is room for them. */
    const si = d3.format('.3~s');
    bars.forEach((d) => {
      seriesLabel(c.g, x(d.value) + 8, y(d.label) + y.bandwidth() / 2 + 4, si(d.value));
      seriesLabel(c.g, 2, y(d.label) + y.bandwidth() + 15, d.note, { colour: 'var(--squidink)' })
        .attr('opacity', 0.75);
    });
    c.svg.append('text').attr('class', 'chart-title')
      .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
      .text(`one ${res} by ${res} frame, counted rather than timed`);
  };

  const say = (extra) => {
    const blobTotal = bars[1].value + bars[2].value;
    readout.innerHTML = `In arithmetic the field needs `
      + `<span class="bold">${fmt.n0(cost.nerf.macs_per_frame)}</span> multiply-accumulates for this frame `
      + `and the blobs need <span class="bold">${fmt.n0(blobTotal)}</span>, a factor of `
      + `<span class="bold">${cost.ratio.toFixed(1)}×</span>. Per pixel that is `
      + `${fmt.n0(cost.nerf.macs_per_pixel)} against ${fmt.n0(cost.splat.macs_per_pixel)}, plus one sort of `
      + `${fmt.n0(cost.splat.blobs)} depths a frame. ` + (extra || '');
  };

  const sleep = () => new Promise((r) => { setTimeout(r, 0); });

  const run = async () => {
    readout.innerHTML = 'rendering both, best of several runs...';
    const dirFn = (i, j) => rayDir(cam, res, meta.fov_deg, i, j);
    await sleep();
    const truth = renderTruth(scene, cam, res, 128);
    truthF.draw(truth, res);
    await sleep();
    let f = null;
    for (let k = 0; k < 2; k++) {
      const got = renderField(field, cam, res, steps, dirFn, { bg: meta.bg, skip: true });
      if (!f || got.ms < f.ms) f = got;
      await sleep();
    }
    fieldF.draw(f.rgb, res);
    let b = null;
    for (let k = 0; k < 5; k++) {
      const got = renderSplats(cloud, cam, res, meta.fov_deg, { bg: meta.bg });
      if (!b || got.ms < b.ms) b = got;
      await sleep();
    }
    blobF.draw(b.rgb, res);
    measured.field = f.ms;
    measured.blob = b.ms;
    /* A ratio whose divisor is a fraction of a millisecond is not a
       measurement, it is a clock resolution, so it is reported as a floor
       rather than as a number with a decimal point in it. */
    const ratio = b.ms >= 0.5
      ? `<span class="bold">${(f.ms / b.ms).toFixed(1)}×</span>`
      : `over <span class="bold">${Math.round(f.ms / 0.5)}×</span>, though the blob raster is too `
        + `fast here for the clock to divide by honestly`;
    say(`Here, now, in this tab: <span class="bold">${fmt.ms(f.ms)}</span> for the field and `
      + `<span class="bold">${fmt.ms(b.ms)}</span> for the blobs, the best of two and five runs, a factor of `
      + `${ratio}. They score `
      + `${fmt.db(psnr(f.rgb, truth))} and ${fmt.db(psnr(b.rgb, truth))} against the closed form at this `
      + `camera. A clock measures how busy your machine is as well as how much work there was, which is `
      + `why the bars above are counted instead.`);
  };

  buttons.innerHTML = '';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'atlas-btn';
  btn.textContent = 'render both, here';
  btn.addEventListener('click', () => { run(); });
  buttons.appendChild(btn);

  truthF.blank(res);
  fieldF.blank(res);
  blobF.blank(res);
  drawChart();
  drawCompare(data, sheets);
  say('Press the button and the same two renderers will do it on your machine.');
  return { run, measured };
}
