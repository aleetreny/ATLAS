/* Drawing a grid world: the squares, a value in each one, the arrow the policy
 * would follow, and a path over the top. Shared by the scrolly and the widget so
 * that both draw the same world the same way.
 */
import { ACTIONS } from './gridkit.js';

export function drawGrid(g, world, { w, h }) {
  const cw = w / world.cols;
  const ch = h / world.rows;
  const cells = [];
  for (let r = 0; r < world.rows; r++) {
    for (let c = 0; c < world.cols; c++) cells.push([r, c]);
  }
  const layer = g.append('g');
  layer.selectAll('rect').data(cells).join('rect')
    .attr('x', (d) => d[1] * cw).attr('y', (d) => d[0] * ch)
    .attr('width', cw - 1).attr('height', ch - 1)
    .attr('fill', '#ffffff').attr('stroke', '#d4dada').attr('stroke-width', 1);
  return { cw, ch, layer, cells,
    cx: (rc) => rc[1] * cw + cw / 2, cy: (rc) => rc[0] * ch + ch / 2 };
}

export function paintSpecials(g, world, geo) {
  const marks = g.append('g');
  world.cliffCells.forEach((rc) => {
    marks.append('rect').attr('x', rc[1] * geo.cw).attr('y', rc[0] * geo.ch)
      .attr('width', geo.cw - 1).attr('height', geo.ch - 1)
      .attr('fill', 'var(--cosmos)').attr('fill-opacity', 0.55);
  });
  [[world.start, 'start'], [world.goal, 'goal']].forEach(([rc, label]) => {
    marks.append('text').attr('class', 'chart-note')
      .attr('x', geo.cx(rc)).attr('y', geo.cy(rc) + 4).attr('text-anchor', 'middle')
      .style('font-size', '10px').text(label);
  });
  return marks;
}

export function paintValues(layer, world, geo, V, scale) {
  layer.selectAll('rect').attr('fill', (d) => scale(V[world.sid(d)]));
}

export function paintArrows(g, world, geo, Q, counts) {
  const arrows = g.append('g');
  for (let r = 0; r < world.rows; r++) {
    for (let c = 0; c < world.cols; c++) {
      const rc = [r, c];
      if (world.isCliff(rc)) continue;
      if (rc[0] === world.goal[0] && rc[1] === world.goal[1]) continue;
      const s = world.sid(rc);
      const usable = [];
      for (let a = 0; a < world.nActions; a++) {
        const seen = !counts || counts[s].every((n) => n === 0) || counts[s][a] > 0;
        if (seen) usable.push([a, Q[s][a]]);
      }
      if (!usable.length) continue;
      const bestV = Math.max(...usable.map(([, v]) => v));
      /* every move within a whisker of the best is drawn, not just the first of
         them: on this board a lot of squares have two equally good moves, and an
         arrow that hides that makes the policy look more decided than it is */
      usable.filter(([, v]) => v >= bestV - 1e-9).forEach(([a]) => {
        const [dr, dc] = ACTIONS[a];
        const len = Math.min(geo.cw, geo.ch) * 0.3;
        arrows.append('line')
          .attr('x1', geo.cx(rc)).attr('y1', geo.cy(rc))
          .attr('x2', geo.cx(rc) + dc * len).attr('y2', geo.cy(rc) + dr * len)
          .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
        arrows.append('circle').attr('cx', geo.cx(rc) + dc * len)
          .attr('cy', geo.cy(rc) + dr * len).attr('r', 2.4).attr('fill', 'var(--squidink)');
      });
    }
  }
  return arrows;
}

export function paintPath(g, geo, path, colour = 'var(--primary)') {
  return g.append('path').attr('fill', 'none').attr('stroke', colour)
    .attr('stroke-width', 3.4).attr('stroke-linejoin', 'round').attr('opacity', 0.85)
    .attr('d', `M${path.map((rc) => `${geo.cx(rc)},${geo.cy(rc)}`).join('L')}`);
}
