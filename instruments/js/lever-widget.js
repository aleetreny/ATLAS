/* The two halves of the fraction, drawn.
 *
 * The slider does not move this cloud: it moves the sweep marker and the
 * numbers, because the cloud is a real sample from one setting and redrawing
 * it at every notch would suggest the estimator's trouble is about sampling
 * noise, which is the one thing the next section says it is not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { ols, wald, firstStageF, fitLine } from './ivkit.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const f1 = (v) => v.toFixed(1);

export function initLeverWidget(data) {
  const host = document.querySelector('#lever-widget');
  if (!host) return;
  const M = data.meta;
  const C = data.cloud;
  const S = data.strength;
  const chart = makeChart('#lever-chart', {
    width: 640, height: 400, margin: { top: 40, right: 26, bottom: 56, left: 64 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#lever-readout');
  const label = document.querySelector('#lever-label');
  const slider = document.querySelector('#lever-slider');
  const views = d3.select('#lever-views');
  let view = 'first';
  let i = 2;

  const buttons = [
    ['first', 'the lever against the treatment'],
    ['reduced', 'the lever against the outcome'],
    ['sweep', 'both estimators against strength'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const est = { ols: ols(C.x, C.y), iv: wald(C.z, C.x, C.y), f: firstStageF(C.z, C.x) };

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const row = S[i];
    label.textContent = f2(row.pi);
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    if (view === 'sweep') {
      const x = d3.scaleLog().domain([0.02, 1.5]).range([0, w]);
      const y = d3.scaleLinear()
        .domain([M.beta - 0.15, d3.max(S, (r) => Math.max(r.iv_median, M.beta + r.ols_bias)) + 0.2])
        .nice().range([h, 0]);
      const ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5];
      drawGrid(g, x, y, w, h, { xValues: ticks });
      drawAxes(g, x, y, w, h, { xValues: ticks });
      axisLabels(g, w, h, { x: 'how hard the lever pushes', y: 'estimate' });
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(M.beta)).attr('y2', y(M.beta))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6)
        .attr('stroke-dasharray', '4 4');
      annotate(g, 4, y(M.beta) - 8, `the effect, ${f2(M.beta)}`, { anchor: 'start' })
        .style('font-size', '11.5px');
      g.append('path')
        .attr('d', line(S.map((r) => [x(r.pi), y(M.beta + r.ols_bias)])))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      g.append('path')
        .attr('d', line(S.map((r) => [x(r.pi), y(r.iv_median)])))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      S.forEach((r) => {
        g.append('line')
          .attr('x1', x(r.pi)).attr('x2', x(r.pi))
          .attr('y1', y(Math.min(r.iv_q3, y.domain()[1]))).attr('y2', y(Math.max(r.iv_q1, y.domain()[0])))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 5)
          .attr('stroke-linecap', 'round');
        g.append('circle').attr('cx', x(r.pi)).attr('cy', y(r.iv_median)).attr('r', 3.8)
          .attr('fill', 'var(--primary)');
      });
      g.append('line').attr('x1', x(row.pi)).attr('x2', x(row.pi))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--stone)').attr('stroke-width', 1.4);
      annotate(g, 4, -22, 'the ratio, median of 400 samples', { anchor: 'start' })
        .style('font-size', '11.5px').attr('fill', 'var(--primary)');
      annotate(g, 4, -6, 'least squares', { anchor: 'start' })
        .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      readout.innerHTML =
        `<p>At a lever of <span class="bold">${f2(row.pi)}</span> the first stage F is `
        + `<span class="bold">${f1(row.f_median)}</span> and the median of ${data.meta.reps} `
        + `estimates is <span class="bold">${f4(row.iv_median)}</span> against a truth of `
        + `${f2(M.beta)}, with the middle half of them between ${f3(row.iv_q1)} and `
        + `${f3(row.iv_q3)}. Least squares sits at ${f4(M.beta + row.ols_bias)} and barely `
        + `moves across the whole sweep, because its bias depends on the confounder rather `
        + `than on the lever. The two curves meeting at the left is the weak instrument `
        + `problem in one picture: as the lever weakens, the estimator that was supposed to `
        + `fix the bias converges to the one that has it.</p>`;
      return;
    }

    const xs = C.z;
    const ys = view === 'first' ? C.x : C.y;
    const x = d3.scaleLinear().domain(d3.extent(xs)).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(ys)).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, {
      x: 'the lever Z', y: view === 'first' ? 'treatment X' : 'outcome Y',
    });
    g.append('g').selectAll('circle')
      .data(xs.map((v, k) => [v, ys[k]]))
      .join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.6);
    const fit = fitLine(xs, ys);
    g.append('path')
      .attr('d', line(fit.map((p) => [x(p[0]), y(p[1])])))
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.6);
    const slope = ols(xs, ys);
    annotate(g, w - 4, -16, `slope ${f3(slope)}`, { anchor: 'end' })
      .style('font-size', '12px');

    readout.innerHTML = view === 'first'
      ? `<p>The bottom of the fraction. Pushing the lever by one unit moves the treatment by `
        + `<span class="bold">${f3(slope)}</span> on these ${xs.length} rows, and the first `
        + `stage F is ${f1(est.f)}. This is the half everybody reports, and it is the only `
        + `one of the four requirements in the table below that the data can speak to.</p>`
      : `<p>The top of the fraction. The same push moves the outcome by `
        + `<span class="bold">${f3(slope)}</span>. Divide one by the other and the confounder `
        + `cancels, because whatever it is doing it is not responding to the lever: `
        + `${f3(slope)} over ${f3(ols(C.z, C.x))} is <span class="bold">${f3(est.iv)}</span>, `
        + `against a truth of ${f2(M.beta)}. Least squares on the same rows gives `
        + `${f3(est.ols)}. All three numbers were computed in this page from the points `
        + `above.</p>`;
  }

  slider.addEventListener('input', (e) => { i = +e.target.value; render(); });
  render();
}
