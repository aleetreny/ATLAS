/* The interval of effects the data cannot narrow.
 *
 * Drawn as an interval rather than as a pair of worlds, because the pair is an
 * illustration and the interval is the result: it is the whole set of answers
 * that reproduce this covariance matrix exactly, and the estimate everybody
 * would report is one point inside it with nothing to prefer it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { wald, ols, fitLine } from './ivkit.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);

export function initWorldsWidget(data) {
  const host = document.querySelector('#worlds-widget');
  if (!host) return;
  const M = data.meta;
  const W = data.worlds;
  const chart = makeChart('#worlds-chart', {
    width: 640, height: 400, margin: { top: 44, right: 30, bottom: 60, left: 64 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#worlds-readout');
  const views = d3.select('#worlds-views');
  let view = 'interval';

  const buttons = [
    ['interval', 'every effect that fits'],
    ['a', 'world one: no direct path'],
    ['b', 'world two: a direct path'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();

    if (view === 'interval') {
      const pad = 0.35;
      const x = d3.scaleLinear()
        .domain([W.beta_lo - pad, W.beta_hi + pad]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { yValues: [] });
      drawAxes(g, x, y, w, h, { yValues: [] });
      axisLabels(g, w, h, { x: 'effect of the treatment on the outcome' });
      g.append('rect')
        .attr('x', x(W.beta_lo)).attr('y', y(0.72))
        .attr('width', x(W.beta_hi) - x(W.beta_lo)).attr('height', y(0.28) - y(0.72))
        .attr('fill', 'var(--primary)').attr('fill-opacity', 0.18)
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6);
      [[W.beta_lo, 'lowest that fits'], [W.beta_hi, 'highest that fits']].forEach(([v, t], k) => {
        g.append('line').attr('x1', x(v)).attr('x2', x(v))
          .attr('y1', y(0.72)).attr('y2', y(0.28))
          .attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(v)).attr('y', k === 0 ? y(0.2) : y(0.12))
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(`${t}: ${f3(v)}`);
      });
      [[M.beta, 'the true effect', 'var(--anchor)', 0.88],
        [W.iv_a, 'what the estimator reports', 'var(--cosmos)', 0.06]].forEach(([v, t, c, yy]) => {
        g.append('line').attr('x1', x(v)).attr('x2', x(v))
          .attr('y1', 0).attr('y2', h)
          .attr('stroke', c).attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(v) + 6).attr('y', y(yy))
          .attr('text-anchor', 'start').style('font-size', '12px')
          .attr('fill', c)
          .text(`${t}: ${f3(v)}`);
      });
      readout.innerHTML =
        `<p>The shaded band is every effect for which a complete world exists that reproduces `
        + `this covariance matrix exactly, with no negative variances anywhere. It runs from `
        + `<span class="bold">${f3(W.beta_lo)}</span> to `
        + `<span class="bold">${f3(W.beta_hi)}</span>. The estimator reports `
        + `${f3(W.iv_a)} and it is right, in the sense that it is the answer under the `
        + `exclusion restriction; the band is what the data alone say, which is a range of `
        + `${f3(W.width)}. Narrowing it is not a matter of more rows. Every point in there `
        + `fits these rows perfectly, and choosing among them is done by assumption.</p>`;
      return;
    }

    const P = W.points;
    const zs = view === 'a' ? P.za : P.zb;
    const ys = view === 'a' ? P.ya : P.yb;
    const xsv = view === 'a' ? P.xa : P.xb;
    const x = d3.scaleLinear().domain(d3.extent(zs)).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(ys)).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'the lever Z', y: 'outcome Y' });
    g.append('g').selectAll('circle')
      .data(zs.map((v, k) => [v, ys[k]]))
      .join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.6);
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);
    g.append('path')
      .attr('d', line(fitLine(zs, ys).map((p) => [x(p[0]), y(p[1])])))
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
    const iv = wald(zs, xsv, ys);
    annotate(g, w - 4, -16, `the ratio on these points: ${f3(iv)}`, { anchor: 'end' })
      .style('font-size', '12px');

    const world = view === 'a'
      ? { truth: W.beta_a, kappa: 0, f: W.f_a }
      : { truth: W.beta_b, kappa: W.kappa_b, f: W.f_b };
    readout.innerHTML =
      `<p>In this world the effect is <span class="bold">${f3(world.truth)}</span> and the `
      + `direct path from the lever to the outcome is `
      + `<span class="bold">${f3(world.kappa)}</span>. The estimator computed on these very `
      + `points returns ${f3(iv)}, and the first stage F is ${world.f.toFixed(1)}. Switch to `
      + `the other world and every one of those diagnostics stays where it is, because the `
      + `two worlds were solved to share their covariance matrix to `
      + `${W.algebra_max_diff.toExponential(1)}. The only difference between them is an `
      + `arrow, and the arrow is not in the file.</p>`;
  }

  render();
}
