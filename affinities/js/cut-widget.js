/* Every possible cut of sixteen points, and where the eigenvector lands.
 *
 * The histogram is not a sample and not a sketch: the page enumerates all
 * 32,767 ways of splitting sixteen points into two non-empty groups, scores
 * each with the normalised cut, and draws the distribution. The reader's
 * threshold slides along the eigenvector, and every position picks out one of
 * those splits, so the two panels are the same object seen twice.
 *
 * This is the only place in the article where "the exact answer" is available
 * at all, which is exactly why it is worth sixteen points.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { rbfGraph, jacobiEigen, ncut } from './graphkit.js';
import { drawPoints, drawEdges, legend } from './draw.js';

export function initCutWidget(data) {
  const C = data.cut;
  const pts = C.points;
  const n = pts.length;
  const W = rbfGraph(pts, C.gamma);

  /* the eigenvector, from the whole spectrum: at sixteen points there is no
     reason to iterate for a few of them */
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += W[i * n + j];
    deg[i] = s;
  }
  const M = Array.from({ length: n }, (_, i) => Float64Array.from(
    { length: n }, (_, j) => W[i * n + j] / Math.sqrt(deg[i] * deg[j])
  ));
  const { values, vectors } = jacobiEigen(M);
  const lambda = values.map((v) => 1 - v);                   // of the Laplacian, ascending
  let f = Array.from(vectors[1], (v, i) => v / Math.sqrt(deg[i]));
  if (f.reduce((s, v) => s + v, 0) < 0) f = f.map((v) => -v);  // sign is arbitrary; fix it

  /* every split, scored */
  const order = f.map((v, i) => i).sort((a, b) => f[a] - f[b]);
  const all = [];
  /* Every subset that contains point 0 except the whole set: each split is
     counted once, and bits = 0 is the perfectly legitimate one where point 0
     stands alone. Starting the loop at 1 silently drops it, which shows up as
     a total of 32,766 next to the generator's 32,767. */
  for (let bits = 0; bits < 2 ** (n - 1); bits++) {
    const mask = new Array(n).fill(false);
    mask[0] = true;
    for (let i = 1; i < n; i++) if (bits & (1 << (i - 1))) mask[i] = true;
    if (mask.every(Boolean)) continue;
    all.push(ncut(W, n, mask));
  }
  all.sort((a, b) => a - b);
  const optimum = all[0];

  /* the n-1 splits the eigenvector's ordering allows */
  const thresholds = [];
  for (let t = 1; t < n; t++) {
    const mask = new Array(n).fill(false);
    for (let i = 0; i < t; i++) mask[order[i]] = true;
    thresholds.push({
      t,
      at: (f[order[t - 1]] + f[order[t]]) / 2,
      ncut: ncut(W, n, mask),
      mask,
    });
  }
  const bestT = thresholds.reduce((a, b) => (b.ncut < a.ncut ? b : a));
  const zeroT = thresholds.reduce((a, b) => (Math.abs(b.at) < Math.abs(a.at) ? b : a));

  /* ---- the picture: the graph, split ---- */
  const main = makeChart('#cut-chart', {
    width: 620, height: 380, margin: { top: 40, right: 24, bottom: 76, left: 50 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, main.w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([main.h, 0]);
  drawGrid(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
  drawAxes(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
  const edgeG = main.g.append('g');
  const dotG = main.g.append('g');
  const title = main.g.append('text').attr('class', 'chart-annotation')
    .attr('x', main.w / 2).attr('y', -10).attr('text-anchor', 'middle').style('font-size', '0.64rem');
  legend(main.g, [
    { colour: 'var(--primary)', shape: 'line', label: 'edge cut by this split' },
    { colour: 'var(--squidink)', shape: 'line', label: 'edge kept' },
  ], { x0: 4, y0: -26, gap: 190 });

  /* the eigenvector as its own axis under the picture, which is what the
     threshold actually slides along */
  const fy = main.h + 44;
  const fx = d3.scaleLinear().domain(d3.extent(f)).nice().range([0, main.w]);
  const strip = main.g.append('g');
  strip.append('line').attr('x1', 0).attr('x2', main.w).attr('y1', fy).attr('y2', fy)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.5);
  strip.append('text').attr('class', 'chart-annotation')
    .attr('x', 0).attr('y', fy + 22).style('font-size', '0.6rem')
    .text('the second eigenvector, one tick per point');
  const stripDots = strip.append('g');
  const cutLine = strip.append('line').attr('y1', fy - 14).attr('y2', fy + 14)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.5);

  /* ---- the histogram of every split ---- */
  const hist = makeChart('#cut-hist', {
    width: 620, height: 240, margin: { top: 34, right: 26, bottom: 52, left: 62 },
  });
  const hx = d3.scaleLinear().domain([Math.min(...all) * 0.98, Math.max(...all) * 1.01])
    .range([0, hist.w]);
  const bins = d3.bin().domain(hx.domain()).thresholds(46)(all);
  const hy = d3.scaleLog().domain([0.7, d3.max(bins, (b) => b.length) * 1.3]).range([hist.h, 0]);
  const HT = [1, 10, 100, 1000, 10000].filter((v) => v <= d3.max(bins, (b) => b.length) * 1.3);
  drawGrid(hist.g, hx, hy, hist.w, hist.h, { xTicks: 6, yValues: HT });
  drawAxes(hist.g, hx, hy, hist.w, hist.h, { xTicks: 6, yValues: HT, yFmt: d3.format('d') });
  axisLabels(hist.g, hist.w, hist.h, { x: 'normalised cut', y: 'splits (log)' });
  hist.g.append('g').selectAll('rect').data(bins).join('rect')
    .attr('x', (b) => hx(b.x0))
    .attr('width', (b) => Math.max(1, hx(b.x1) - hx(b.x0) - 1))
    .attr('y', (b) => hy(Math.max(b.length, 0.7)))
    .attr('height', (b) => hist.h - hy(Math.max(b.length, 0.7)))
    .attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6);
  const markLine = (val, colour, label, dy) => {
    hist.g.append('line').attr('x1', hx(val)).attr('x2', hx(val))
      .attr('y1', 0).attr('y2', hist.h)
      .attr('stroke', colour).attr('stroke-width', 1.6).attr('stroke-dasharray', '4 3');
    hist.g.append('text').attr('class', 'chart-annotation')
      .attr('x', hx(val) + 5).attr('y', dy).style('font-size', '0.62rem')
      .style('fill', colour).text(label);
  };
  markLine(lambda[1], 'var(--aqua)', `the relaxed value, ${lambda[1].toFixed(4)}`, 14);
  markLine(optimum, 'var(--primary)', `the best of all ${all.length.toLocaleString('en-US')}, ${optimum.toFixed(4)}`, 30);
  const here = hist.g.append('line').attr('y1', 0).attr('y2', hist.h)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.5);

  const slider = document.querySelector('#cut-t');
  const readout = document.querySelector('#cut-readout');
  slider.min = 0;
  slider.max = thresholds.length - 1;
  slider.step = 1;
  /* Opens at the textbook shortcut, cutting the eigenvector at zero, so that
     both buttons still have somewhere to go and the first thing on screen is a
     split somebody would actually take. */
  slider.value = zeroT.t - 1;

  function render() {
    const cur = thresholds[+slider.value];
    const { mask } = cur;
    const labels = mask.map((v) => (v ? 0 : 1));
    const items = [];
    let hi = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = W[i * n + j];
        items.push({ a: i, b: j, w, cut: mask[i] !== mask[j] });
        if (w > hi) hi = w;
      }
    }
    edgeG.selectAll('line').remove();
    drawEdges(edgeG, items.filter((d) => !d.cut).map((d) => ({ ...d, w: d.w / hi })), pts, x, y,
      { colour: 'var(--squidink)', maxWidth: 2.4 });
    const cutSel = edgeG.append('g');
    /* The cut edges are the cheap ones, which is the whole point of the
       objective and also why they nearly vanish: they get an opacity floor so
       that "few edges crossed, and light ones" is legible rather than blank. */
    drawEdges(cutSel, items.filter((d) => d.cut).map((d) => ({ ...d, w: d.w / hi })), pts, x, y,
      { colour: 'var(--primary)', maxWidth: 3.4, floor: 0.34 });
    drawPoints(dotG, pts, labels, x, y, { r: 6 });

    stripDots.selectAll('line').data(f).join('line')
      .attr('x1', (d) => fx(d)).attr('x2', (d) => fx(d))
      .attr('y1', fy - 7).attr('y2', fy + 7)
      .attr('stroke', (d, i) => (mask[i] ? 'var(--aqua)' : 'var(--cosmos)'))
      .attr('stroke-width', 2);
    cutLine.attr('x1', fx(cur.at)).attr('x2', fx(cur.at));
    here.attr('x1', hx(cur.ncut)).attr('x2', hx(cur.ncut));
    document.querySelector('#cut-t-label').textContent = cur.at.toFixed(3);

    const rank = all.filter((v) => v < cur.ncut - 1e-12).length + 1;
    const cutWeight = items.filter((d) => d.cut).reduce((s, d) => s + d.w, 0);
    const side = mask.filter(Boolean).length;
    title.text(`${side} point${side === 1 ? '' : 's'} on one side, ${n - side} on the other, `
      + `cutting ${cutWeight.toFixed(3)} of edge weight`);

    readout.innerHTML =
      `<table class="aff-table">
        <tr><th></th><th>normalised cut</th><th>rank among all ${all.length.toLocaleString('en-US')}</th></tr>
        <tr><td>this threshold</td><td><span class="value">${cur.ncut.toFixed(6)}</span></td>
            <td><span class="value">${rank.toLocaleString('en-US')}</span></td></tr>
        <tr><td>the best the eigenvector can do</td><td>${bestT.ncut.toFixed(6)}</td>
            <td>${(all.filter((v) => v < bestT.ncut - 1e-12).length + 1).toLocaleString('en-US')}</td></tr>
        <tr><td>the best split there is</td><td>${optimum.toFixed(6)}</td><td>1</td></tr>
        <tr><td>the relaxed problem's answer</td><td>${lambda[1].toFixed(6)}</td>
            <td>a lower bound, reached by no split at all</td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `${rank === 1
        ? `This is the best split of the sixteen points, out of ${all.length.toLocaleString('en-US')}.`
        : `Better splits than this one: ${(rank - 1).toLocaleString('en-US')}.`}`
      + ` The relaxed value below every split is ${lambda[1].toFixed(6)}, which is `
      + `${((optimum - lambda[1]) / optimum * 100).toFixed(1)}% under the best real cut: that `
      + `distance is what allowing a yes-or-no to become a real number bought, and it is never `
      + `paid back.`
      + `</div>`;
  }

  document.querySelector('#cut-best').addEventListener('click', () => {
    slider.value = bestT.t - 1;
    render();
  });
  document.querySelector('#cut-zero').addEventListener('click', () => {
    slider.value = zeroT.t - 1;
    render();
  });
  slider.addEventListener('input', render);
  render();
  return { thresholds, all, optimum, lambda, bestT, zeroT };
}
