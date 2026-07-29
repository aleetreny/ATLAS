/* The cleanest demonstration of leakage there is, because there is nothing to
 * learn: the labels are coin flips and the columns are independent noise, so
 * the only honest answer is one half.
 *
 * Two histograms over two hundred datasets. Choosing the twenty columns that
 * separate the classes best while looking at every row, and then cross
 * validating on those twenty, gives an answer far above one half. Choosing them
 * again inside each fold, from that fold's training rows only, gives one half.
 * Same data, same model, same folds: the only difference is where the column
 * selection stood.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initNoiseWidget(data) {
  const N = data.noise;
  const readout = document.querySelector('#noise-readout');

  const chart = makeChart('#noise-chart', {
    width: 640, height: 340, margin: { top: 56, right: 30, bottom: 58, left: 74 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const all = N.outside_all.concat(N.inside_all);
  const lo = Math.min(0.35, Math.min(...all) - 0.02);
  const hi = Math.max(0.75, Math.max(...all) + 0.02);
  const bins = 32;
  const count = (arr) => {
    const out = new Array(bins).fill(0);
    arr.forEach((v) => {
      const k = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / (hi - lo) * bins)));
      out[k] += 1;
    });
    return out;
  };
  const ho = count(N.outside_all);
  const hi2 = count(N.inside_all);

  const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
  const y = d3.scaleLinear().domain([0, Math.max(...ho, ...hi2) * 1.15]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 4 });
  drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 4, xFmt: d3.format('.2f') });
  axisLabels(g, w, h, { x: 'cross validated accuracy', y: 'datasets' });

  const bw = w / bins;
  [[ho, 'var(--cosmos)'], [hi2, 'var(--primary)']].forEach(([hh, colour]) => {
    hh.forEach((c, k) => {
      if (!c) return;
      layer.append('rect').attr('x', x(lo + (hi - lo) * k / bins)).attr('width', bw + 0.4)
        .attr('y', y(c)).attr('height', h - y(c)).attr('fill', colour).attr('opacity', 0.72);
    });
  });

  layer.append('line').attr('x1', x(0.5)).attr('x2', x(0.5)).attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 3');
  layer.append('text').attr('class', 'chart-note').attr('x', x(0.5) + 6).attr('y', 12)
    .style('font-size', '11px').text('the only correct answer');

  [['columns chosen while looking at everything', 'var(--cosmos)'],
    ['columns chosen inside each fold', 'var(--primary)']].forEach(([t, colour], i) => {
    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -42 + i * 15)
      .style('font-size', '11px').attr('fill', colour).text(t);
  });

  readout.innerHTML = `<div class="gen-note">${N.seeds} datasets of ${N.n} rows and `
    + `${N.p.toLocaleString('en-US')} columns, every column independent noise and every label `
    + `a coin flip. Keep the ${N.keep} columns that separate the classes best, looking at all `
    + `${N.n} rows to choose them, then cross validate on those columns: `
    + `<span class="bold">${N.outside.toFixed(4)}</span> with a spread of `
    + `${N.outside_sd.toFixed(4)}. Choose the ${N.keep} again inside each fold, from that `
    + `fold's training rows only: <span class="bold">${N.inside.toFixed(4)}</span>. The `
    + `second is the honest answer and the first is the same experiment with the selection `
    + `standing one line higher up. Nothing about the model changed, and nothing about the `
    + `folds changed. For scale, fitting on all the rows and scoring on the same rows gives `
    + `${N.resub.toFixed(4)}, which nobody would report; the point of this section is that `
    + `${N.outside.toFixed(4)} looks like a result and is the same mistake.</div>`;

  return { render: () => {} };
}
