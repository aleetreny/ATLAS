/* The plane, warped, one layer at a time, computed here.
 *
 * The lines are a regular grid drawn in the latent space and pushed through the
 * model into the data space, which is the direction that generates. Nothing is
 * precomputed: every vertex is run through the layers the reader has reached,
 * so the picture is the model's output rather than a drawing of it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

const LINES = 15;          // grid lines each way
const STEPS = 46;          // vertices along each line
const SPAN = 2.6;          // how far into the gaussian the grid reaches

export function initScrollyFlow(data, flow) {
  const chart = makeChart('#flow-chart', {
    width: 540, height: 470, margin: { top: 52, right: 26, bottom: 54, left: 58 },
  });
  const { g, w, h } = chart;
  const view = Math.max(Math.abs(data.picture.lo), Math.abs(data.picture.hi));
  const x = d3.scaleLinear().domain([-view, view]).range([0, w]);
  const y = d3.scaleLinear().domain([-view, view]).range([h, 0]);

  const mesh = g.append('g');
  const dots = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -30).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const SHOWN = [0, 1, 2, flow.depth];
  const sample = data.picture.sample;

  function meshPaths(upto) {
    const paths = [];
    for (let i = 0; i < LINES; i++) {
      const u = -SPAN + (2 * SPAN * i) / (LINES - 1);
      for (const dir of [0, 1]) {
        const pts = [];
        for (let j = 0; j < STEPS; j++) {
          const v = -SPAN + (2 * SPAN * j) / (STEPS - 1);
          const z = dir === 0 ? [u, v] : [v, u];
          pts.push(flow.inverseFrom(z, upto));
        }
        paths.push(pts);
      }
    }
    return paths;
  }

  function render(step) {
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: 'first coordinate', y: 'second coordinate' });

    const upto = SHOWN[Math.min(step, SHOWN.length - 1)];
    const line = d3.line().x((p) => x(p[0])).y((p) => y(p[1]));
    mesh.selectAll('path').data(meshPaths(upto)).join('path')
      .attr('d', line).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 0.9).attr('opacity', 0.55);

    dots.selectAll('circle').data(step >= 3 ? sample : []).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 1.6)
      .attr('fill', 'var(--squidink)').attr('opacity', 0.6);

    const labels = [
      'a square grid in the gaussian the flow starts from',
      'after one coupling layer: one coordinate has moved and the other has not',
      'after two: the mask has swapped, so now the other one moves',
      `after all ${flow.depth}, with the data it was fitted to on top`,
      'and however far it is pushed, the grid never crosses itself',
    ];
    wrapLabel(title, labels[step], w - 8);
  }

  render(0);
  const handlers = {};
  for (let i = 0; i < 5; i++) handlers[i] = () => render(i);
  const ctl = scrolly('#flow-scrolly .step', handlers);
  return { render, goTo: ctl.goTo };
}
