/* The confounding machine, with the reader holding the knob.
 *
 * The page draws its own sample rather than replaying one the generator sent,
 * because the whole point of the section is that the bias is a function of a
 * parameter and the reader should be able to move the parameter. What the
 * sample is checked against is not another sample: it is the closed form,
 * which is exact at every setting of the slider.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { draw, slope, ols2, millsGap } from './statkit.js';

const N = 420;
const ALPHAS = [0, 0.2, 0.4, 0.7, 1.0, 1.2, 1.6, 2.0, 3.0, 5.0];
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);

export function initEngineWidget(data) {
  const host = document.querySelector('#engine-widget');
  if (!host) return;
  const M = data.meta;
  const chart = makeChart('#engine-chart', {
    width: 640, height: 400, margin: { top: 34, right: 26, bottom: 54, left: 62 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#engine-readout');
  const label = document.querySelector('#engine-alpha-label');
  const slider = document.querySelector('#engine-alpha');
  const views = d3.select('#engine-views');

  let view = 'cloud';
  let ai = 5;

  const buttons = [
    ['cloud', 'one sample'],
    ['curve', 'bias against strength'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  function render() {
    const alpha = ALPHAS[ai];
    label.textContent = f2(alpha);
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const sample = draw(N, alpha, M, 20240719 + ai);
    const naive = d3.mean(sample.y.filter((_, i) => sample.x[i] === 1))
      - d3.mean(sample.y.filter((_, i) => sample.x[i] === 0));
    const adjusted = ols2(sample.x, sample.u, sample.y);
    const closed = M.beta + M.gamma * millsGap(alpha);

    if (view === 'cloud') drawCloud(sample, alpha, naive, adjusted);
    else drawCurve(alpha, naive);

    const overshoot = naive / M.beta;
    readout.innerHTML =
      `<p>With the hidden variable pushing at <span class="bold">${f2(alpha)}</span>, this `
      + `sample of ${N} gives a difference in means of `
      + `<span class="bold">${f3(naive)}</span> where the effect written into the world is `
      + `<span class="bold">${f2(M.beta)}</span>, so it overstates it by a factor of `
      + `${f2(overshoot)}. The closed form says to expect ${f3(closed)} on average. `
      + `Adjusting for the hidden variable on this same sample returns `
      + `<span class="bold">${f3(adjusted)}</span>. At the far left of the slider nothing `
      + `decides the treatment, the two agree, and the comparison you can make without any `
      + `causal assumption is the one you wanted all along.</p>`;
  }

  function drawCloud(sample, alpha, naive, adjusted) {
    const x = d3.scaleLinear().domain(d3.extent(sample.u)).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(sample.y)).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'the hidden variable U', y: 'outcome Y' });

    [0, 1].forEach((t) => {
      g.append('g')
        .selectAll('circle')
        .data(sample.u.map((u, i) => [u, sample.y[i], sample.x[i]]).filter((d) => d[2] === t))
        .join('circle')
        .attr('cx', (d) => x(d[0]))
        .attr('cy', (d) => y(d[1]))
        .attr('r', 2.6)
        .attr('fill', t ? 'var(--primary)' : 'var(--smile)')
        .attr('fill-opacity', 0.68);
      const ys = sample.y.filter((_, i) => sample.x[i] === t);
      const us = sample.u.filter((_, i) => sample.x[i] === t);
      g.append('line')
        .attr('x1', x(d3.min(us))).attr('x2', x(d3.max(us)))
        .attr('y1', y(d3.mean(ys))).attr('y2', y(d3.mean(ys)))
        .attr('stroke', t ? 'var(--primary)' : 'var(--smile)')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5 4');
    });
    annotate(g, 6, -14, 'treated', { anchor: 'start' })
      .style('font-size', '12px').attr('fill', 'var(--primary)');
    annotate(g, 86, -14, 'untreated', { anchor: 'start' })
      .style('font-size', '12px').attr('fill', 'var(--smile)');
    annotate(g, w, -14, `the two dashed lines are ${f3(naive)} apart`, { anchor: 'end' })
      .style('font-size', '12px');
  }

  function drawCurve(alpha, naive) {
    const x = d3.scaleLinear().domain([0, 5.2]).range([0, w]);
    const rows = data.alpha_sweep;
    const y = d3.scaleLinear()
      .domain([M.beta - 0.4, d3.max(rows, (r) => r.closed) + 0.4]).nice().range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'how strongly U decides treatment', y: 'difference in means' });

    const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
    const dense = d3.range(0, 5.21, 0.02)
      .map((a) => [a, M.beta + M.gamma * millsGap(a)]);
    g.append('path')
      .attr('d', line(dense))
      .attr('fill', 'none')
      .attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.2);
    g.append('g')
      .selectAll('circle')
      .data(rows)
      .join('circle')
      .attr('cx', (d) => x(d.alpha))
      .attr('cy', (d) => y(d.naive))
      .attr('r', 4)
      .attr('fill', 'var(--cosmos)');
    g.append('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', y(M.beta)).attr('y2', y(M.beta))
      .attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 1.6)
      .attr('stroke-dasharray', '4 4');
    g.append('line')
      .attr('x1', x(alpha)).attr('x2', x(alpha))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--stone)')
      .attr('stroke-width', 1.4);
    annotate(g, 6, y(M.beta) - 8, `the effect, ${f2(M.beta)}`, { anchor: 'start' })
      .style('font-size', '12px');
    annotate(g, w - 4, 14, 'line: closed form', { anchor: 'end' })
      .style('font-size', '11.5px');
    annotate(g, w - 4, 30, 'dots: four hundred thousand draws each', { anchor: 'end' })
      .style('font-size', '11.5px');
  }

  slider.addEventListener('input', (e) => {
    ai = +e.target.value;
    render();
  });
  render();
}
