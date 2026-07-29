/* The world, its exact answer, and the two paths.
 *
 * Value iteration is rerun here rather than read, so the values shaded into the
 * squares are computed in the page from the same rules the generator used.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { makeWorld, valueIteration, greedyPath } from './gridkit.js';
import { drawGrid, paintSpecials, paintValues, paintArrows, paintPath } from './gridview.js';

export function initCliffScrolly(data) {
  const T = data.truth;
  const world = makeWorld({ rows: T.rows, cols: T.cols, start: T.start, goal: T.goal });
  world.cliffCells = T.cliff;
  const solved = valueIteration(world);
  const V = solved.Q.map((row) => Math.max(...row));

  const chart = makeChart('#cliff-chart', {
    width: 640, height: 320, margin: { top: 54, right: 20, bottom: 40, left: 20 },
  });
  const { g, w, h } = chart;
  const geo = drawGrid(g, world, { w, h });
  const specials = paintSpecials(g, world, geo);
  const shade = d3.scaleLinear().domain([d3.min(V), d3.max(V)])
    .range(['#e8e2f2', '#ffffff']);
  const overlay = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -14)
    .style('font-size', '11.5px');

  function reset() {
    overlay.selectAll('*').remove();
    geo.layer.selectAll('rect').attr('fill', '#ffffff');
    specials.raise();
  }

  const handlers = {
    0: () => {
      reset();
      title.text(`${T.rows} by ${T.cols} squares, four moves, every step costs one`);
      sub.text(`the red band costs ${Math.abs(-100)} and puts you back at the start`);
    },
    1: () => {
      reset();
      paintValues(geo.layer, world, geo, V, shade);
      specials.raise();
      paintArrows(overlay, world, geo, solved.Q, null);
      title.text(`solved exactly in ${solved.sweeps} sweeps`);
      sub.text('the shade is what a square is worth; two arrows means two equally good moves');
    },
    2: () => {
      reset();
      paintValues(geo.layer, world, geo, V, shade);
      specials.raise();
      const p = greedyPath(world, solved.Q);
      paintPath(overlay, geo, p.path, 'var(--primary)');
      title.text(`the best route: ${p.path.length - 1} steps, worth ${p.total}`);
      sub.text('it runs along the edge of the drop, because that is the short way');
    },
    3: () => {
      reset();
      paintValues(geo.layer, world, geo, V, shade);
      specials.raise();
      paintPath(overlay, geo, data.learners.sarsa.path, 'var(--anchor)');
      paintPath(overlay, geo, data.learners.qlearning.path, 'var(--primary)');
      title.text('what the two learners come back with');
      sub.text(`${data.learners.qlearning.path_return} along the edge against `
        + `${data.learners.sarsa.path_return} one row up`);
    },
  };
  handlers[0]();
  const ctrl = scrolly('#cliff-scrolly .step', handlers);
  return { ...ctrl, solved };
}
