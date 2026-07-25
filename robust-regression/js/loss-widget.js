/* Loss curves and, more importantly, their derivatives: the force a residual
 * exerts on the fit. Robustness is entirely a statement about that second
 * panel being bounded. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { losses, pulls } from './mathkit.js';

const R = 4;

export function initLossWidget() {
  const host = d3.select('#loss-chart');
  const width = 620;
  const height = 520;

  const svg = host
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const m = { top: 34, right: 20, bottom: 46, left: 54 };
  const panelH = (height - m.top - m.bottom - 46) / 2;
  const w = width - m.left - m.right;

  const panels = [
    { key: 'loss', title: 'what a residual costs', fns: losses, yMax: 8 },
    { key: 'pull', title: 'how hard it pulls (the derivative)', fns: pulls, yMax: 4.2 },
  ].map((p, i) => {
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top + i * (panelH + 46)})`);
    const x = d3.scaleLinear().domain([-R, R]).range([0, w]);
    const y = d3.scaleLinear().domain([0, p.yMax]).range([panelH, 0]);
    drawGrid(g, x, y, w, panelH, { xTicks: 5, yTicks: 4 });
    drawAxes(g, x, y, w, panelH, { xTicks: 5, yTicks: 4 });
    axisLabels(g, w, panelH, { x: i === 1 ? 'residual r' : '' });
    g.append('text')
      .attr('class', 'chart-annotation')
      .attr('x', 0)
      .attr('y', -12)
      .style('font-size', '0.66rem')
      .style('letter-spacing', '0.5px')
      .style('text-transform', 'none')
      .text(p.title);
    return { ...p, g, x, y };
  });

  const grid = d3.range(-R, R + 0.001, 0.02);
  const SERIES = [
    { id: 'squared', label: 'squared', color: 'var(--cosmos)', dash: null },
    { id: 'absolute', label: 'absolute', color: 'var(--stone)', dash: '6 5' },
    { id: 'huber', label: 'huber', color: 'var(--primary)', dash: null },
  ];

  for (const p of panels) {
    p.paths = {};
    for (const s of SERIES) {
      p.paths[s.id] = p.g
        .append('path')
        .attr('fill', 'none')
        .attr('stroke', s.color)
        .attr('stroke-width', s.id === 'huber' ? 3.6 : 2.4)
        .attr('stroke-dasharray', s.dash);
    }
    // the two delta markers, showing where huber switches behaviour
    p.deltaLines = p.g.append('g');
  }

  const slider = document.querySelector('#loss-delta');
  const label = document.querySelector('#loss-delta-label');
  const stats = document.querySelector('#loss-stats');

  function render() {
    const d = +slider.value;
    label.textContent = d.toFixed(2);

    for (const p of panels) {
      const line = d3.line().x((r) => p.x(r)).y((v) => p.y(Math.min(v, p.yMax * 1.4)));
      for (const s of SERIES) {
        const vals = grid.map((r) => p.fns[s.id](r, d));
        p.paths[s.id].attr('d', line.y((v, i) => p.y(Math.min(p.fns[s.id](grid[i], d), p.yMax * 1.4)))(grid));
      }
      p.deltaLines.selectAll('line').data([-d, d]).join('line')
        .attr('x1', (v) => p.x(v)).attr('x2', (v) => p.x(v))
        .attr('y1', 0).attr('y2', panelH)
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6)
        .attr('stroke-dasharray', '4 4').attr('opacity', 0.55);
    }

    const at = 4;
    stats.innerHTML =
      `<table class="method-table">
         <tr><th></th><th>cost of r = ${at}</th><th>pull at r = ${at}</th></tr>
         <tr><td style="color:var(--cosmos)">squared</td><td>${losses.squared(at).toFixed(1)}</td><td>${pulls.squared(at).toFixed(2)}</td></tr>
         <tr><td style="opacity:.65">absolute</td><td>${losses.absolute(at).toFixed(1)}</td><td>${pulls.absolute(at).toFixed(2)}</td></tr>
         <tr><td style="color:var(--primary)">huber</td><td>${losses.huber(at, d).toFixed(1)}</td><td>${pulls.huber(at, d).toFixed(2)}</td></tr>
       </table>`;
  }

  slider.addEventListener('input', render);
  render();
}
