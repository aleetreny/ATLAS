/* The walk itself, on the landscape it can be checked against.
 *
 * Three chains are run here, in the browser, with the same generator the file
 * used: one hot, one cold, and the reference one that cools. What is drawn is
 * where each of them spent its time.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { walk, boltzmann } from './sakit.js';

export function initMetropolisScrolly(data) {
  const E = data.reference.energy;
  const n = E.length;
  const cfg = data.reference.config;
  const runs = {
    hot: walk(E, { T: cfg.T0, steps: cfg.steps, seed: cfg.seed, keepPath: true }),
    cold: walk(E, { T: cfg.T1, steps: cfg.steps, seed: cfg.seed, keepPath: true }),
    cooling: walk(E, { T: cfg.T0, T1: cfg.T1, steps: cfg.steps, seed: cfg.seed,
      keepPath: true }),
  };

  const chart = makeChart('#metropolis-chart', {
    width: 640, height: 420, margin: { top: 50, right: 30, bottom: 54, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().domain([0, n - 1]).range([0, w]);
  const y = d3.scaleLinear().domain([0, Math.max(...E) * 1.12]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 4 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 4 });
  axisLabels(g, w, h, { x: 'state around the ring', y: 'energy', leftBudget: margin.left });

  /* how often the walk sat on each state, drawn as bars under the curve */
  const bars = g.append('g');
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2.4)
    .attr('d', d3.line().x((_, i) => x(i)).y((d) => y(d))(E));
  const dot = g.append('circle').attr('r', 6).attr('fill', 'var(--cosmos)').style('opacity', 0);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function draw(key, note) {
    const run = runs[key];
    const top = Math.max(...run.visits);
    bars.selectAll('rect').data(run.visits).join('rect')
      .attr('x', (_, i) => x(i) - (w / n) * 0.4).attr('width', (w / n) * 0.8)
      .transition('bars').duration(500)
      .attr('y', (v) => y(0) - (v / top) * h * 0.92)
      .attr('height', (v) => (v / top) * h * 0.92)
      .attr('fill', 'var(--primary)').attr('fill-opacity', 0.22);
    dot.style('opacity', 1).transition('dot').duration(400).attr('cx', x(run.final))
      .attr('cy', y(E[run.final]));
    sub.text(note);
  }

  const lowest = E.indexOf(Math.min(...E));
  const handlers = {
    0: () => {
      bars.selectAll('rect').data([]).join('rect');
      dot.style('opacity', 0);
      title.text(`${n} states in a ring, and the energy of each one`);
      sub.text(`the deepest is state ${lowest}; there is a second valley and a `
        + 'barrier between them');
    },
    1: () => {
      title.text(`${cfg.steps.toLocaleString('en-US')} steps at temperature ${cfg.T0}`);
      draw('hot', `${((runs.hot.accepted / cfg.steps) * 100).toFixed(0)}% of moves accepted: `
        + 'it goes everywhere');
    },
    2: () => {
      title.text(`the same ${cfg.steps.toLocaleString('en-US')} steps at temperature ${cfg.T1}`);
      draw('cold', `${runs.cold.uphillAccepted} of ${runs.cold.uphillProposed} uphill moves `
        + 'accepted: wherever it landed, it stays');
    },
    3: () => {
      title.text(`cooling from ${cfg.T0} to ${cfg.T1} over the run`);
      draw('cooling', `${runs.cooling.uphillAccepted} of ${runs.cooling.uphillProposed} uphill `
        + `moves accepted, and it ends on state ${runs.cooling.final}`);
    },
  };
  handlers[0]();
  const ctrl = scrolly('#metropolis-scrolly .step', handlers);
  return { ...ctrl, runs };
}
