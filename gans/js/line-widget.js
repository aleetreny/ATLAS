/* Which distribution each divergence picks out of the same one parameter
 * family, computed live.
 *
 * The data is a two component mixture and the model is a single gaussian, so
 * the model cannot be right and every objective has to decide how to be wrong.
 * Nothing here is a lookup: move the two sliders and all four numbers are
 * integrated on the spot, on the same grid the generator used, which is why
 * the article can promise that the reader's machine reproduces the figures.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import {
  makeGrid, mixturePdf, normalPdf, divergences, optimalD, valueAtOptimum, bestAccuracy,
} from './divkit.js';

const KEYS = ['fwd', 'rev', 'js', 'w1'];
const NAMES = {
  fwd: 'forward KL',
  rev: 'reverse KL',
  js: 'Jensen-Shannon',
  w1: 'Wasserstein-1',
};
const MATH = {
  fwd: 'KL(data || model)',
  rev: 'KL(model || data)',
  js: 'JS(data, model)',
  w1: 'W(data, model)',
};

export function initLineWidget(data) {
  const M = data.meta;
  const grid = makeGrid(M.grid);
  const p = mixturePdf(grid.x, M.mixture);

  const state = { mu: 1.2, sd: 0.7 };
  const muStops = [];
  for (let v = -3; v <= 3 + 1e-9; v += 0.025) muStops.push(Math.round(v * 1000) / 1000);

  /* ---------------------------------------------------------------- panels */
  const left = makeChart('#line-density', {
    width: 620, height: 380, margin: { top: 30, right: 58, bottom: 52, left: 58 },
  });
  const right = makeChart('#line-curves', {
    width: 620, height: 380, margin: { top: 44, right: 20, bottom: 52, left: 62 },
  });

  const xScale = d3.scaleLinear().domain([-5, 5]).range([0, left.w]);
  const yScale = d3.scaleLinear().domain([0, 0.85]).range([left.h, 0]);
  const dScale = d3.scaleLinear().domain([0, 1]).range([left.h, 0]);
  const area = d3.area().x((d, i) => xScale(grid.x[i])).y0(left.h).y1((d) => yScale(d));
  const line = d3.line().x((d, i) => xScale(grid.x[i])).y((d) => yScale(d));
  const dLine = d3.line().x((d, i) => xScale(grid.x[i])).y((d) => dScale(d));

  drawGrid(left.g, xScale, yScale, left.w, left.h, { xTicks: 6, yTicks: 5 });
  const pathData = left.g.append('path').attr('fill', 'var(--squidink)').attr('fill-opacity', 0.13)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5);
  const pathModel = left.g.append('path').attr('fill', 'var(--primary)').attr('fill-opacity', 0.18)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2);
  const pathD = left.g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)')
    .attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
  drawAxes(left.g, xScale, yScale, left.w, left.h, { xTicks: 6, yTicks: 5 });
  /* The discriminator rides a second scale, so it gets a second axis: a series
     on an undrawn scale is decoration. */
  left.g.append('g').attr('class', 'axis y right')
    .attr('transform', `translate(${left.w},0)`)
    .call(d3.axisRight(dScale).ticks(5))
    .call((s) => s.selectAll('text').style('fill', 'var(--cosmos)'));
  axisLabels(left.g, left.w, left.h, { x: 'x', y: 'density' });
  left.g.append('text').attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)').attr('x', -left.h / 2).attr('y', left.w + 44)
    .attr('text-anchor', 'middle').style('fill', 'var(--cosmos)')
    .text('D*(x), the best discriminator');
  const legend = left.g.append('g');
  [['data, a mixture of two', 'var(--squidink)'], ['model, one gaussian', 'var(--primary)']]
    .forEach(([t, c], i) => {
      legend.append('line').attr('x1', 6).attr('x2', 24).attr('y1', -18 + i * 0).attr('y2', -18)
        .attr('stroke', c).attr('stroke-width', 2.5)
        .attr('transform', `translate(${i * 210},0)`);
      legend.append('text').attr('class', 'chart-annotation').attr('x', 28 + i * 210).attr('y', -14)
        .style('font-size', '11.5px').style('letter-spacing', '0.5px').text(t);
    });

  /* the four curves, each over mu at the current sd, in its own small panel */
  const cols = 2;
  const cw = (right.w - 26) / cols;
  const ch = (right.h - 30) / 2;
  const panels = KEYS.map((k, i) => {
    const gx = (i % cols) * (cw + 26);
    const gy = Math.floor(i / cols) * (ch + 30);
    const g = right.g.append('g').attr('transform', `translate(${gx},${gy})`);
    /* the title is the one node render() must not wipe, so it carries a class
       of its own: selecting on `chart-annotation` alone kept the value labels
       too and they stacked up one per render */
    g.append('text').attr('class', 'chart-annotation panel-title').attr('x', 0).attr('y', -8)
      .style('font-size', '11.5px').style('letter-spacing', '0.5px').text(NAMES[k]);
    return { k, g, cw, ch };
  });
  right.g.append('text').attr('class', 'axis-label')
    .attr('x', right.w / 2).attr('y', right.h + 42).attr('text-anchor', 'middle')
    .text('model mean, at the current width');

  /* ------------------------------------------------------------- the maths */
  let curves = null;
  let curveSd = null;

  function recomputeCurves() {
    if (curveSd === state.sd) return;
    curveSd = state.sd;
    const out = { fwd: [], rev: [], js: [], w1: [] };
    muStops.forEach((mu, i) => {
      if (i % 2) return;                       // every other stop is plenty to draw
      const q = normalPdf(grid.x, mu, state.sd);
      const dv = divergences(p, q, grid.dx);
      KEYS.forEach((k) => out[k].push([mu, dv[k]]));
    });
    curves = out;
  }

  function render() {
    const q = normalPdf(grid.x, state.mu, state.sd);
    const dv = divergences(p, q, grid.dx);
    const value = valueAtOptimum(p, q, grid.dx);
    const acc = bestAccuracy(p, q, grid.dx);
    const dstar = optimalD(p, q);

    pathData.attr('d', area(Array.from(p)));
    pathModel.attr('d', area(Array.from(q)));
    pathD.attr('d', dLine(Array.from(dstar)));

    recomputeCurves();
    panels.forEach(({ k, g }) => {
      const rows = curves[k];
      const finite = rows.filter((r) => Number.isFinite(r[1]));
      const hi = d3.quantile(finite.map((r) => r[1]).sort(d3.ascending), 0.92);
      const lo = Math.min(...finite.map((r) => r[1]));
      const sx = d3.scaleLinear().domain([-3, 3]).range([0, cw]);
      const sy = d3.scaleLinear().domain([lo, Math.max(hi, lo + 1e-6)]).range([ch, 0]).clamp(true);
      g.selectAll('*:not(text.panel-title)').remove();
      g.append('line').attr('x1', 0).attr('x2', cw).attr('y1', ch).attr('y2', ch)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
      g.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2)
        .attr('d', d3.line().x((r) => sx(r[0])).y((r) => sy(r[1]))(rows));
      const best = data.best[k];
      g.append('line').attr('x1', sx(best.mu)).attr('x2', sx(best.mu)).attr('y1', 0).attr('y2', ch)
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
      g.append('circle').attr('cx', sx(state.mu)).attr('cy', sy(dv[k])).attr('r', 4)
        .attr('fill', 'var(--primary)');
      g.append('text').attr('class', 'chart-annotation').attr('x', cw).attr('y', -8)
        .attr('text-anchor', 'end').style('font-size', '11px').style('letter-spacing', '0.4px')
        .text(fmt(dv[k]));
      /* the horizontal range, on the bottom row only, so the reader knows what
         the curves are drawn over without four repeated axes */
      if (k === 'js' || k === 'w1') {
        [[-3, 'start'], [0, 'middle'], [3, 'end']].forEach(([v, anchor]) => {
          g.append('text').attr('class', 'chart-annotation').attr('x', sx(v))
            .attr('y', ch + 13).attr('text-anchor', anchor).style('font-size', '10.5px')
            .style('letter-spacing', '0.3px').text(v);
        });
      }
    });

    const rows = KEYS.map((k) => {
      const best = data.best[k];
      const here = dv[k];
      const gap = here - best.value;
      return `<tr><td>${NAMES[k]}</td><td>${MATH[k]}</td><td>${fmt(here)}</td>`
        + `<td>${fmt(best.value)} at mean ${best.mu.toFixed(3)}, width ${best.sd.toFixed(2)}</td>`
        + `<td>${gap < 1e-4 ? 'you are on it' : `${fmt(gap)} above its best`}</td></tr>`;
    }).join('');
    document.querySelector('#line-readout').innerHTML =
      `<table class="atlas-table"><thead><tr><th>Objective</th><th>Written</th>`
      + `<th>Here</th><th>Its own best</th><th>Distance to it</th></tr></thead>`
      + `<tbody>${rows}</tbody></table>`
      + `<p style="margin-top:0.8rem">At this setting the best possible discriminator is right `
      + `<span class="bold">${(acc * 100).toFixed(1)}%</span> of the time and the game is worth `
      + `<span class="bold">${value.toFixed(4)}</span>, which is `
      + `2 &times; ${dv.js.toFixed(4)} &minus; 2 log 2 to `
      + `${Math.abs(value - (2 * dv.js - 2 * Math.log(2))).toExponential(1)}. `
      + `The value of the game and the Jensen-Shannon divergence are the same number wearing `
      + `different clothes, which is the whole reason a classifier can stand in for a divergence.</p>`;
  }

  const fmt = (v) => (!Number.isFinite(v) ? 'infinite'
    : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));

  /* --------------------------------------------------------------- controls */
  const muInput = document.querySelector('#line-mu');
  const sdInput = document.querySelector('#line-sd');
  const sync = () => {
    document.querySelector('#line-mu-value').textContent = state.mu.toFixed(3);
    document.querySelector('#line-sd-value').textContent = state.sd.toFixed(2);
    muInput.value = state.mu;
    sdInput.value = state.sd;
  };
  muInput.addEventListener('input', () => { state.mu = +muInput.value; sync(); render(); });
  sdInput.addEventListener('input', () => { state.sd = +sdInput.value; sync(); render(); });

  const row = document.querySelector('#line-buttons');
  KEYS.forEach((k) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = `best for ${NAMES[k]}`;
    b.addEventListener('click', () => {
      state.mu = data.best[k].mu;
      state.sd = data.best[k].sd;
      sync();
      render();
    });
    row.appendChild(b);
  });

  sync();
  render();
  return { state, render };
}
