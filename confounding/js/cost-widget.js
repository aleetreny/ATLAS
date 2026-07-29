/* Three ways the correction fails, on one pair of axes each.
 *
 * They share a widget because they share a moral: every one of them produces a
 * number that looks exactly like a corrected estimate, with no diagnostic
 * anywhere on the output that says which fraction of the bias survived.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v) => `${(100 * v).toFixed(1)}%`;

export function initCostWidget(data) {
  const host = document.querySelector('#cost-widget');
  if (!host) return;
  const M = data.meta;
  const chart = makeChart('#cost-chart', {
    width: 640, height: 400, margin: { top: 34, right: 30, bottom: 56, left: 66 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#cost-readout');
  const views = d3.select('#cost-views');
  let view = 'strata';

  const buttons = [
    ['strata', 'groups too coarse'],
    ['proxy', 'confounder measured with noise'],
    ['collider', 'the wrong variable entirely'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  function frame(x, y, xlab, ylab, xValues) {
    drawGrid(g, x, y, w, h, { xValues });
    drawAxes(g, x, y, w, h, { xValues });
    axisLabels(g, w, h, { x: xlab, y: ylab });
  }

  function truthLine(y) {
    g.append('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 1.6)
      .attr('stroke-dasharray', '4 4');
    annotate(g, 6, y(0) - 8, 'no bias left', { anchor: 'start' })
      .style('font-size', '12px');
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    if (view === 'strata') {
      const rows = data.strata;
      const x = d3.scaleLog().domain([1, 100]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.bias) * 1.08])
        .range([h, 0]);
      frame(x, y, 'how many groups the confounder is cut into',
        'bias still in the answer', [1, 2, 3, 5, 10, 20, 50, 100]);
      truthLine(y);
      g.append('path')
        .attr('d', line(rows.map((r) => [x(r.k), y(r.bias)])))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      g.append('g').selectAll('circle').data(rows).join('circle')
        .attr('cx', (d) => x(d.k)).attr('cy', (d) => y(d.bias)).attr('r', 4)
        .attr('fill', 'var(--primary)');
      const five = rows.find((r) => r.k === 5);
      annotate(g, x(five.k) + 8, y(five.bias) - 10,
        `five groups still leaves ${f3(five.bias)}`, { anchor: 'start' })
        .style('font-size', '12px');
      readout.innerHTML =
        `<p>The confounder is measured perfectly here. The only thing being lost is `
        + `resolution: inside a group it still varies, so inside a group the treated and `
        + `the untreated are still different people. Cutting it into `
        + `${five.k} groups leaves <span class="bold">${f3(five.bias)}</span> of bias on an `
        + `effect of ${f2(M.beta)}, which is ${pc(five.bias / M.beta)} of what is being `
        + `estimated, and it takes ${rows[rows.length - 1].k} groups to get under `
        + `${f3(rows[rows.length - 1].bias)}. The curve has no cliff and no warning on it.</p>`;
    } else if (view === 'proxy') {
      const rows = data.proxy;
      const x = d3.scaleLinear().domain([0, 3]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.bias) * 1.08])
        .range([h, 0]);
      frame(x, y, 'noise in the recorded confounder, in its own units',
        'bias still in the answer');
      truthLine(y);
      g.append('path')
        .attr('d', line(rows.map((r) => [x(r.noise), y(r.bias)])))
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      g.append('g').selectAll('circle').data(rows).join('circle')
        .attr('cx', (d) => x(d.noise)).attr('cy', (d) => y(d.bias)).attr('r', 4)
        .attr('fill', 'var(--cosmos)');
      g.append('line')
        .attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.engine.bias_measured)).attr('y2', y(data.engine.bias_measured))
        .attr('stroke', 'var(--stone)').attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '3 5');
      annotate(g, w - 4, y(data.engine.bias_measured) - 8,
        'adjusting for nothing at all', { anchor: 'end' }).style('font-size', '12px');
      const half = rows.find((r) => r.noise === 0.5);
      readout.innerHTML =
        `<p>Now the confounder is recorded with error, which is the normal case for anything `
        + `a human wrote on a form. The correction only removes the part of the confounder `
        + `that the recorded copy explains, so the leftover bias grows with the noise. At `
        + `noise ${f2(half.noise)}, half of the confounder's own spread, the answer is still `
        + `<span class="bold">${f4(half.bias)}</span> away from the truth, and the recorded `
        + `copy still correlates with the real thing at `
        + `${f2(Math.sqrt(half.reliability))}. Adjusting helped. It did not finish.</p>`;
    } else {
      const C = data.collider;
      const P = C.points;
      const x = d3.scaleLinear().domain(d3.extent(P.c)).nice().range([0, w]);
      const y = d3.scaleLinear().domain(d3.extent(P.y)).nice().range([h, 0]);
      frame(x, y, 'the variable caused by both, C', 'outcome Y');
      [0, 1].forEach((t) => {
        g.append('g').selectAll('circle')
          .data(P.c.map((c, i) => [c, P.y[i], P.x[i]]).filter((d) => d[2] === t))
          .join('circle')
          .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
          .attr('fill', t ? 'var(--primary)' : 'var(--smile)')
          .attr('fill-opacity', 0.72);
      });
      P.edges.forEach((e) => {
        g.append('line')
          .attr('x1', x(e)).attr('x2', x(e)).attr('y1', 0).attr('y2', h)
          .attr('stroke', 'var(--stone)').attr('stroke-width', 1.2)
          .attr('stroke-dasharray', '3 4');
      });
      annotate(g, 6, -14, 'treated', { anchor: 'start' })
        .style('font-size', '12px').attr('fill', 'var(--primary)');
      annotate(g, 86, -14, 'untreated', { anchor: 'start' })
        .style('font-size', '12px').attr('fill', 'var(--smile)');
      annotate(g, w, -14, 'the dashed lines cut C into thirds', { anchor: 'end' })
        .style('font-size', '12px');
      readout.innerHTML =
        `<p>There is no confounding in this picture at all: the treatment was decided by a `
        + `coin, so the plain difference in means is already right at `
        + `<span class="bold">${f4(C.naive)}</span> against a truth of ${f2(C.truth)}. `
        + `C is caused by the treatment and by the outcome. Inside any vertical slice of it, `
        + `a treated unit with a given C had to get there with a lower Y than an untreated `
        + `one, which is why the estimates inside the three slices are `
        + `${C.strata.map((s) => f3(s.estimate)).join(', ')} and the pooled version is `
        + `<span class="bold">${f4(C.stratified)}</span>. Putting C into a regression gives `
        + `<span class="bold">${f4(C.with_collider)}</span>. The correction is what broke `
        + `it.</p>`;
    }
  }

  render();
}
