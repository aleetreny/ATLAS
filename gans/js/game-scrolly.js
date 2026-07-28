/* The game, one piece at a time, on the same picture the widget below uses.
 *
 * Four idempotent states: the data on its own, a model that cannot contain it,
 * the best discriminator that could ever be trained against that pair, and the
 * quantity the game is worth. Everything drawn is integrated live from the
 * grid in game.json rather than stored, so the last panel's number is the same
 * number the generator measured.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import {
  makeGrid, mixturePdf, normalPdf, divergences, optimalD, valueAtOptimum,
} from './divkit.js';

export function initGameScrolly(data) {
  const M = data.meta;
  const grid = makeGrid(M.grid);
  const p = mixturePdf(grid.x, M.mixture);
  const MU = 1.2;
  const SD = 0.7;
  const q = normalPdf(grid.x, MU, SD);
  const dstar = optimalD(p, q);
  const dv = divergences(p, q, grid.dx);
  const value = valueAtOptimum(p, q, grid.dx);

  const c = makeChart('#game-chart', {
    width: 700, height: 440, margin: { top: 36, right: 62, bottom: 56, left: 62 },
  });
  const x = d3.scaleLinear().domain([-5, 5]).range([0, c.w]);
  const y = d3.scaleLinear().domain([0, 0.85]).range([c.h, 0]);
  const dScale = d3.scaleLinear().domain([0, 1]).range([c.h, 0]);
  const area = d3.area().x((d, i) => x(grid.x[i])).y0(c.h).y1((d) => y(d));
  const dLine = d3.line().x((d, i) => x(grid.x[i])).y((d) => dScale(d));

  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 5 });
  const pathData = c.g.append('path').attr('d', area(Array.from(p)))
    .attr('fill', 'var(--squidink)').attr('fill-opacity', 0.13)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5);
  const pathModel = c.g.append('path').attr('d', area(Array.from(q)))
    .attr('fill', 'var(--primary)').attr('fill-opacity', 0.18)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2).attr('opacity', 0);
  const pathD = c.g.append('path').attr('d', dLine(Array.from(dstar)))
    .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2)
    .attr('stroke-dasharray', '5 3').attr('opacity', 0);
  drawAxes(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 5 });
  const rightAxis = c.g.append('g').attr('class', 'axis y right')
    .attr('transform', `translate(${c.w},0)`).attr('opacity', 0)
    .call(d3.axisRight(dScale).ticks(5))
    .call((s) => s.selectAll('text').style('fill', 'var(--cosmos)'));
  axisLabels(c.g, c.w, c.h, { x: 'x', y: 'density' });
  const note = c.g.append('text').attr('class', 'chart-annotation')
    .attr('x', 2).attr('y', -12).style('font-size', '11.5px').style('letter-spacing', '0.5px');

  const show = (sel, on) => sel.transition('fade').duration(420).attr('opacity', on ? 1 : 0);

  const states = {
    0: () => {
      show(pathModel, false);
      show(pathD, false);
      show(rightAxis, false);
      note.text('the data');
    },
    1: () => {
      show(pathModel, true);
      show(pathD, false);
      show(rightAxis, false);
      note.text('the data, and a model that cannot be right');
    },
    2: () => {
      show(pathModel, true);
      show(pathD, true);
      show(rightAxis, true);
      note.text('and the best discriminator that will ever exist for this pair');
    },
    3: () => {
      show(pathModel, true);
      show(pathD, true);
      show(rightAxis, true);
      note.text(`the game at that discriminator is worth ${value.toFixed(4)}`);
    },
  };
  const sc = scrolly('#game-scrolly .step', states);
  states[0]();

  document.querySelector('#game-caption').innerHTML =
    `At this model, the best discriminator scores the game at `
    + `<span class="bold">${value.toFixed(4)}</span> and the Jensen-Shannon divergence between the `
    + `two densities is <span class="bold">${dv.js.toFixed(4)}</span>. `
    + `2 &times; ${dv.js.toFixed(4)} &minus; 2 log 2 = ${(2 * dv.js - 2 * Math.log(2)).toFixed(4)}: `
    + `the same number. Both are integrated in your browser as this page loads.`;
  return sc;
}
