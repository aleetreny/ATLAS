/* The same board four times: the answer, what the network believes, where it is
 * wrong, and what it does about it.
 *
 * The network is run in this page from its exported weights, so the second
 * panel is its output rather than a picture of its output.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { makeWorld, valueIteration, greedyPath } from './gridkit.js';
import { drawGrid, paintSpecials, paintValues, paintPath } from './gridview.js';
import { forward, observe } from './netkit.js';

export function initNetScrolly(data) {
  const M = data.meta;
  const world = makeWorld({ rows: M.rows, cols: M.cols, start: M.start, goal: M.goal });
  world.cliffCells = M.cliff;
  const solved = valueIteration(world);
  const truth = solved.Q.map((row) => Math.max(...row));
  const W = data.weights.weights;
  const netQ = [];
  for (let s = 0; s < world.nStates; s++) {
    const rc = [Math.floor(s / M.cols), s % M.cols];
    netQ.push(forward(W, observe(rc, M.rows, M.cols)));
  }
  const netV = netQ.map((row) => Math.max(...row));
  const err = netV.map((v, i) => Math.abs(v - truth[i]));

  const chart = makeChart('#net-chart', {
    width: 640, height: 320, margin: { top: 54, right: 20, bottom: 40, left: 20 },
  });
  const { g, w, h } = chart;
  const geo = drawGrid(g, world, { w, h });
  const specials = paintSpecials(g, world, geo);
  const overlay = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -14)
    .style('font-size', '11.5px');
  const lo = d3.min([...truth, ...netV]);
  const hi = d3.max([...truth, ...netV]);
  const shade = d3.scaleLinear().domain([lo, hi]).range(['#e7dff0', '#ffffff']);
  const errShade = d3.scaleLinear().domain([0, d3.max(err)]).range(['#ffffff', 'var(--cosmos)']);

  const handlers = {
    0: () => {
      overlay.selectAll('*').remove();
      paintValues(geo.layer, world, geo, truth, shade);
      specials.raise();
      title.text('the exact answer, from the same value iteration as before');
      sub.text(`${world.nStates} squares, four actions, ${world.nStates * 4} numbers stored`);
    },
    1: () => {
      overlay.selectAll('*').remove();
      paintValues(geo.layer, world, geo, netV, shade);
      specials.raise();
      const nW = Object.values(W).reduce((a, m) => a
        + (Array.isArray(m[0]) ? m.length * m[0].length : m.length), 0);
      title.text('what the network believes, run here from its weights');
      sub.text(`two inputs, ${data.weights.hidden} hidden units, ${nW} numbers stored`);
    },
    2: () => {
      overlay.selectAll('*').remove();
      paintValues(geo.layer, world, geo, err, errShade);
      specials.raise();
      title.text(`where it is wrong: worst square off by ${d3.max(err).toFixed(2)}`);
      sub.text(`on the squares its own route visits it is off by `
        + `${data.weights.gap_on_path}`);
    },
    3: () => {
      overlay.selectAll('*').remove();
      paintValues(geo.layer, world, geo, netV, shade);
      specials.raise();
      paintPath(overlay, geo, greedyPath(world, solved.Q).path, '#8e9aa6');
      paintPath(overlay, geo, data.weights.path, 'var(--primary)');
      title.text(`its route: ${data.weights.path.length - 1} steps, worth ${data.weights.return}`);
      sub.text(`against ${M.truth_return} for the exact answer, drawn in grey`);
    },
  };
  handlers[0]();
  const ctrl = scrolly('#net-scrolly .step', handlers);
  return { ...ctrl, netQ, truth };
}
