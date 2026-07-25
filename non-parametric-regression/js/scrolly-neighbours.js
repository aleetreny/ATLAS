/* Scrollytelling: the same Auto MPG cloud the first article fitted a line to,
 * now approached without assuming any equation at all. */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { knnPredict, knnNeighbours } from './mathkit.js';

const DUR = 650;

export function initScrollyNeighbours(mpg) {
  const pts = mpg.points;
  const xs = pts.map((d) => d.x);
  const ys = pts.map((d) => d.y);
  const grid = mpg.grid;

  const { g, w, h } = makeChart('#neighbours-chart', {
    width: 620,
    height: 540,
    margin: { top: 46, right: 26, bottom: 62, left: 62 },
  });

  const x = d3.scaleLinear().domain([35, 235]).range([0, w]);
  const y = d3.scaleLinear().domain([5, 50]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });

  const bandG = g.append('g');
  const dotsG = g.append('g');
  const lineG = g.append('g');
  const annG = g.append('g');

  const dots = dotsG
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 4)
    .attr('fill', 'var(--stone)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.8);

  const halo = lineG.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7).attr('opacity', 0);
  const curve = lineG.append('path').attr('fill', 'none').attr('stroke-width', 3.5).attr('opacity', 0);
  const olsLine = lineG.append('line').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.5)
    .attr('stroke-dasharray', '8 6').attr('opacity', 0);

  const readout = annG.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -18).attr('text-anchor', 'middle');

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);
  const t = () => d3.transition().duration(DUR).ease(d3.easeCubicOut);

  function setCurve(k, weights, color, visible) {
    if (!visible) {
      halo.transition(t()).attr('opacity', 0);
      curve.transition(t()).attr('opacity', 0);
      return;
    }
    const pred = knnPredict(xs, ys, grid, k, weights);
    const d = line(grid.map((v, i) => [v, pred[i]]));
    halo.transition(t()).attr('d', d).attr('opacity', 1);
    curve.transition(t()).attr('d', d).attr('stroke', color).attr('opacity', 1);
  }

  /* highlight the neighbourhood one query actually consults */
  function setQuery(q, k) {
    bandG.selectAll('*').remove();
    dots.transition(t()).attr('fill', 'var(--stone)').attr('r', 4);
    if (q === null) return;
    const nb = new Set(knnNeighbours(xs, q, k));
    const near = [...nb].map((i) => xs[i]);
    const lo = Math.min(...near);
    const hi = Math.max(...near);
    // Horsepower is recorded in whole numbers and repeats heavily, so in the
    // dense part of the range the k nearest cars can share one exact value and
    // the neighbourhood collapses to a line. The query used here sits in a
    // sparser region precisely so the band has width to see.
    bandG.append('rect')
      .attr('x', x(lo)).attr('width', Math.max(3, x(hi) - x(lo)))
      .attr('y', 0).attr('height', h)
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.14)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
    bandG.append('line')
      .attr('x1', x(q)).attr('x2', x(q)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2.5);
    dots.transition(t())
      .attr('fill', (d, i) => (nb.has(i) ? 'var(--primary)' : 'var(--stone)'))
      .attr('r', (d, i) => (nb.has(i) ? 6 : 4));
  }

  const handlers = {
    0: () => {
      setCurve(0, 'uniform', '', false);
      olsLine.transition(t()).attr('opacity', 0);
      setQuery(null);
      readout.text('auto mpg · the same 392 cars');
    },
    1: () => {
      setCurve(0, 'uniform', '', false);
      const [x0, x1] = x.domain();
      olsLine.transition(t())
        .attr('x1', x(x0)).attr('y1', y(39.94 - 0.158 * x0))
        .attr('x2', x(x1)).attr('y2', y(39.94 - 0.158 * x1))
        .attr('opacity', 1);
      setQuery(null);
      readout.text('a parametric model assumes the shape first');
    },
    2: () => {
      setCurve(0, 'uniform', '', false);
      olsLine.transition(t()).attr('opacity', 0.35);
      setQuery(190, 15);
      readout.text('ask the 15 nearest cars instead');
    },
    3: () => {
      setCurve(1, 'uniform', 'var(--cosmos)', true);
      olsLine.transition(t()).attr('opacity', 0);
      setQuery(null);
      readout.text('k = 1 · every point is its own answer');
    },
    4: () => {
      setCurve(120, 'uniform', 'var(--anchor)', true);
      olsLine.transition(t()).attr('opacity', 0);
      setQuery(null);
      readout.text('k = 120 · almost the global average');
    },
    5: () => {
      setCurve(15, 'uniform', 'var(--primary)', true);
      olsLine.transition(t()).attr('opacity', 0.35);
      setQuery(null);
      readout.text('k = 15 · the curve nobody had to specify');
    },
  };

  handlers[0]();
  scrolly('#neighbours-scrolly .step', handlers);
}
