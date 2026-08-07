/* What the chain is actually sampling, against what it is supposed to sample.
 *
 * The bars are a chain run in this page; the line is exp(-E/T) normalised over
 * the ring, which on sixty four states can be written down exactly. The second
 * view answers the obvious objection: the two never match perfectly, so is the
 * difference a bias or is it the sample size? It falls like the square root of
 * the chain length, which is what a sample does and a bias does not.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { walk, boltzmann, totalVariation } from './sakit.js';

const VIEWS = [
  { key: 'hist', label: 'what the chain visits' },
  { key: 'conv', label: 'is the gap a bias' },
];

export function initBoltzmannWidget(data) {
  const B = data.boltzmann;
  const E = B.energy;
  const temps = B.rows.map((r) => r.T);
  const lengths = B.convergence.steps;
  const readout = document.querySelector('#boltzmann-readout');
  const controls = d3.select('#boltzmann-controls');
  const tSlider = document.querySelector('#boltz-t');
  const sSlider = document.querySelector('#boltz-steps');
  const tLabel = document.querySelector('#boltz-t-label');
  const sLabel = document.querySelector('#boltz-steps-label');
  let view = 'hist';

  const chart = makeChart('#boltzmann-chart', {
    width: 640, height: 350, margin: { top: 48, right: 118, bottom: 54, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function renderHist() {
    const T = temps[Number(tSlider.value)];
    const steps = lengths[Number(sSlider.value)];
    const run = walk(E, { T, steps, seed: 991 });
    const exact = boltzmann(E, T);
    const tv = totalVariation(run.visits, exact);
    const x = d3.scaleLinear().domain([0, E.length - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([0, Math.max(...exact, ...run.visits) * 1.15])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 4 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 4 });
    axisLabels(g, w, h, { x: 'state around the ring', y: 'share of the time',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    plot.selectAll('rect').data(run.visits).join('rect')
      .attr('x', (_, i) => x(i) - (w / E.length) * 0.42).attr('width', (w / E.length) * 0.84)
      .attr('y', (v) => y(v)).attr('height', (v) => h - y(v))
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.55);
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 2.2)
      .attr('d', d3.line().x((_, i) => x(i)).y((d) => y(d))(exact));
    key.selectAll('*').remove();
    key.append('rect').attr('width', 12).attr('height', 12)
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.55);
    ['a chain run', 'in this page'].forEach((line, i) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 26 + i * 16)
        .style('font-size', '10.5px').text(line);
    });
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 58).attr('y2', 58)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 2.2);
    ['the formula,', 'exactly'].forEach((line, i) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 74 + i * 16)
        .style('font-size', '10.5px').text(line);
    });
    title.text(`temperature ${T}, ${steps.toLocaleString('en-US')} steps`);
    sub.text(`distance between the two: ${tv.toFixed(4)}`);
    tLabel.textContent = `${T}`;
    sLabel.textContent = steps.toLocaleString('en-US');

    const row = B.rows.find((r) => r.T === T);
    const share = exact.map((p, i) => [p, E[i]]).filter(([, e]) => e < 0.5)
      .reduce((a, [p]) => a + p, 0);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>temperature</th>${B.rows.map((r) => `<th>${r.T}</th>`).join('')}</tr>
        <tr><td>distance to the formula</td>
            ${B.rows.map((r) => `<td>${r.T === T ? '<span class="value">' : ''}${r.tv_distance}${r.T === T ? '</span>' : ''}</td>`).join('')}</tr>
        <tr><td>uphill moves accepted</td>
            ${B.rows.map((r) => `<td>${(r.accept_measured * 100).toFixed(1)}%</td>`).join('')}</tr>
        <tr><td>the rule predicts</td>
            ${B.rows.map((r) => `<td>${(r.accept_predicted * 100).toFixed(1)}%</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        At temperature ${T} the formula puts <span class="value">${(share * 100).toFixed(1)}%</span>
        of the time in the low ground, and the chain run in this page agrees to
        <span class="value">${tv.toFixed(4)}</span>. Nothing in the walk knows
        that formula: it only ever compares two neighbouring states and tosses a
        coin. The row underneath is the check with no sampling in it at all: how
        often an uphill move is accepted, measured
        (${(row.accept_measured * 100).toFixed(1)}%) against what the rule
        predicts once each edge is weighted by how much time the chain spends at
        its start (${(row.accept_predicted * 100).toFixed(1)}%). The unweighted
        average, which is the tempting thing to compute, is the wrong number and
        misses by several points at the cold end.
      </div>`;
  }

  function renderConv() {
    const C = B.convergence;
    const pts = C.steps.map((s, i) => [s, C.distance[i]]);
    const x = d3.scaleLog().domain([C.steps[0] * 0.8, C.steps[C.steps.length - 1] * 1.25])
      .range([0, w]);
    const y = d3.scaleLog().domain([Math.min(...C.distance) * 0.7, Math.max(...C.distance) * 1.4])
      .range([h, 0]);
    const xt = [25000, 100000, 400000].filter((v) => v >= x.domain()[0] && v <= x.domain()[1]);
    const yt = [0.01, 0.02, 0.05, 0.1].filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xValues: xt, yValues: yt });
    drawAxes(g, x, y, w, h, { xValues: xt, yValues: yt, xFmt: d3.format(',d'),
      yFmt: d3.format('.3f') });
    axisLabels(g, w, h, { x: 'steps in the chain', y: 'distance to the formula',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.6)
      .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts));
    plot.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 4)
      .attr('fill', 'var(--primary)');
    /* the line a pure sampling error would follow, anchored at the first point */
    const ref = pts.map(([s]) => [s, C.distance[0] * (s / C.steps[0]) ** -0.5]);
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4')
      .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(ref));
    key.selectAll('*').remove();
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 6).attr('y2', 6)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
    ['measured'].forEach((line, i) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 22 + i * 16)
        .style('font-size', '10.5px').text(line);
    });
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 46).attr('y2', 46)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
    ['pure sampling,', 'one over', 'the square root'].forEach((line, i) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 62 + i * 16)
        .style('font-size', '10.5px').text(line);
    });
    title.text('the gap between the bars and the line, as the chain gets longer');
    sub.text(`slope ${C.slope} against the ${C.predicted_slope} of pure sampling`);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>steps</th>${C.steps.map((s) => `<th>${(s / 1000)}k</th>`).join('')}</tr>
        <tr><td>distance</td>
            ${C.distance.map((v) => `<td>${v}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Over a ${C.steps[C.steps.length - 1] / C.steps[0]} fold range of chain
        lengths the distance falls with a slope of
        <span class="value">${C.slope}</span> on these axes, against the
        ${C.predicted_slope} that pure sampling noise would give. A bias would
        have flattened out at whatever it was worth. That is the difference
        between "these two are close" and "these two are the same thing measured
        twice", and it is the whole reason the acceptance rule is written the way
        it is rather than any other way that also prefers downhill.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    d3.selectAll('#boltz-t, #boltz-steps').attr('disabled', view === 'conv' ? 'disabled' : null);
    if (view === 'hist') renderHist(); else renderConv();
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  tSlider.addEventListener('input', render);
  sSlider.addEventListener('input', render);
  render();
  return { render };
}
