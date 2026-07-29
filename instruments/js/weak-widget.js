/* What a weak lever does to the shape of the estimate.
 *
 * Two views, and the second one exists because of a rule this atlas keeps
 * relearning: a summary statistic is a claim about a distribution, and the
 * mean of a ratio with a denominator near zero is a claim about a distribution
 * that has no mean. Drawing the six seeds is cheaper than an integral and
 * harder to argue with.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f1 = (v) => v.toFixed(1);
const pc = (v) => `${(100 * v).toFixed(1)}%`;

export function initWeakWidget(data) {
  const host = document.querySelector('#weak-widget');
  if (!host) return;
  const M = data.meta;
  const S = data.strength;
  const Z = data.seeds;
  const chart = makeChart('#weak-chart', {
    width: 640, height: 400, margin: { top: 44, right: 30, bottom: 56, left: 70 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#weak-readout');
  const views = d3.select('#weak-views');
  let view = 'tails';

  const buttons = [
    ['tails', 'how far the worst sample goes'],
    ['seeds', 'the mean and the median, six times'],
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
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    if (view === 'tails') {
      const x = d3.scaleLog().domain([0.02, 1.5]).range([0, w]);
      const y = d3.scaleLog()
        .domain([Math.max(1e-3, d3.min(S, (r) => Math.abs(r.iv_median_bias)) / 2),
          d3.max(S, (r) => r.iv_max_abs) * 1.4])
        .range([h, 0]);
      const xt = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5];
      const yt = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
      drawGrid(g, x, y, w, h, { xValues: xt, yValues: yt.filter((v) => v >= y.domain()[0] && v <= y.domain()[1]) });
      drawAxes(g, x, y, w, h, {
        xValues: xt,
        yValues: yt.filter((v) => v >= y.domain()[0] && v <= y.domain()[1]),
      });
      axisLabels(g, w, h, { x: 'how hard the lever pushes', y: 'distance from the truth' });
      g.append('path')
        .attr('d', line(S.map((r) => [x(r.pi), y(Math.max(Math.abs(r.iv_median_bias), y.domain()[0]))])))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      g.append('path')
        .attr('d', line(S.map((r) => [x(r.pi), y(r.iv_max_abs)])))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      S.forEach((r) => {
        g.append('circle').attr('cx', x(r.pi))
          .attr('cy', y(Math.max(Math.abs(r.iv_median_bias), y.domain()[0]))).attr('r', 3.6)
          .attr('fill', 'var(--primary)');
        g.append('circle').attr('cx', x(r.pi)).attr('cy', y(r.iv_max_abs)).attr('r', 3.6)
          .attr('fill', 'var(--cosmos)');
      });
      annotate(g, 4, -26, 'the worst of 400 samples', { anchor: 'start' })
        .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      annotate(g, 4, -10, 'the median of the same 400', { anchor: 'start' })
        .style('font-size', '11.5px').attr('fill', 'var(--primary)');
      const weakest = S[S.length - 1];
      const strong = S[0];
      readout.innerHTML =
        `<p>Both axes are logarithmic, which is the only way these two curves fit on one `
        + `picture. With a strong lever the median is ${f3(Math.abs(strong.iv_median_bias))} `
        + `from the truth and the worst of ${M.reps} samples is `
        + `${f2(strong.iv_max_abs)} away. With the weakest lever measured, the median is `
        + `${f3(Math.abs(weakest.iv_median_bias))} away and the worst sample is `
        + `<span class="bold">${f1(weakest.iv_max_abs)}</span> away, on an effect of `
        + `${f2(M.beta)}. The gap between the two curves is what a heavy tail looks like: `
        + `the typical sample is fine and the occasional one is not, and nothing in a single `
        + `dataset says which kind you have.</p>`;
    } else {
      const x = d3.scalePoint().domain(Z.map((s, k) => `seed ${k + 1}`)).range([0, w]).padding(0.6);
      const lo = Math.min(M.beta, d3.min(Z, (s) => Math.min(s.mean, s.median)));
      const hi = Math.max(M.beta, d3.max(Z, (s) => Math.max(s.mean, s.median)));
      const y = d3.scaleLinear().domain([lo - 0.3, hi + 0.3]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { y: 'summary of 400 estimates' });
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(M.beta)).attr('y2', y(M.beta))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6)
        .attr('stroke-dasharray', '4 4');
      annotate(g, 4, y(M.beta) - 8, `the effect, ${f2(M.beta)}`, { anchor: 'start' })
        .style('font-size', '11.5px');
      Z.forEach((s, k) => {
        const cx = x(`seed ${k + 1}`);
        g.append('line').attr('x1', cx).attr('x2', cx)
          .attr('y1', y(s.mean)).attr('y2', y(s.median))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 2);
        g.append('circle').attr('cx', cx).attr('cy', y(s.mean)).attr('r', 5)
          .attr('fill', 'var(--cosmos)');
        g.append('circle').attr('cx', cx).attr('cy', y(s.median)).attr('r', 5)
          .attr('fill', 'var(--primary)');
      });
      annotate(g, w - 4, -26, 'the mean of the 400', { anchor: 'end' })
        .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      annotate(g, w - 4, -10, 'the median of the same 400', { anchor: 'end' })
        .style('font-size', '11.5px').attr('fill', 'var(--primary)');
      const means = Z.map((s) => s.mean);
      const meds = Z.map((s) => s.median);
      readout.innerHTML =
        `<p>Six independent repetitions of the same experiment at a lever of 0.05, each one `
        + `${M.reps} samples. The medians land between ${f3(Math.min(...meds))} and `
        + `${f3(Math.max(...meds))}, a spread of `
        + `<span class="bold">${f3(Math.max(...meds) - Math.min(...meds))}</span>. The means `
        + `land between ${f2(Math.min(...means))} and ${f2(Math.max(...means))}, a spread of `
        + `<span class="bold">${f2(Math.max(...means) - Math.min(...means))}</span>. More `
        + `data will not fix the second column, because a ratio whose denominator can be `
        + `arbitrarily close to zero has no expected value for a sample mean to converge to. `
        + `The estimator is fine. The summary is the thing that does not exist.</p>`;
    }
  }

  render();
}
