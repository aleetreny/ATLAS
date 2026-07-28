/* Six steps from a rolled up rectangle to an unrolled one.
 *
 * Colour is the ONE thing to watch: every point is tinted by where it really
 * sits along the sheet, a quantity nothing in any of these methods is ever
 * told. A method that unrolled the sheet leaves a smooth ramp behind; a method
 * that flattened it leaves the ramp folded over itself. It is the only way to
 * read these pictures without already knowing the answer.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';

export function initScrollyRoll(data, state, computed) {
  const R = data.roll;
  const chart = makeChart('#roll-chart', {
    width: 640, height: 520, margin: { top: 62, right: 26, bottom: 88, left: 62 },
  });
  const { g, w, h } = chart;

  const sVals = state.chart.map((c) => c[0]);
  const sExt = [Math.min(...sVals), Math.max(...sVals)];
  const ramp = d3.scaleSequential(d3.interpolateTurbo).domain(sExt);

  const layers = {
    roll: g.append('g').attr('opacity', 0),
    flat: g.append('g').attr('opacity', 0),
  };
  const edgesG = layers.roll.append('g');
  const pathG = layers.roll.append('g');
  const dotsG = layers.roll.append('g');
  const markG = layers.roll.append('g');
  const flatDots = layers.flat.append('g');

  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', h + 64).attr('text-anchor', 'middle')
    .style('font-size', '11.5px').style('text-transform', 'none');

  /* ------------------------------------------------------------- the 3-D view */
  const proj = state.proj;
  const { x: rx, y: ry } = equalScales(proj, w, h, { pad: 1.06 });
  const depthOrder = proj.map((p, i) => i).sort((a, b) => proj[a][2] - proj[b][2]);
  dotsG.selectAll('circle').data(depthOrder).join('circle')
    .attr('cx', (i) => rx(proj[i][0])).attr('cy', (i) => ry(proj[i][1]))
    .attr('r', 3.4)
    .attr('fill', (i) => ramp(state.chart[i][0]))
    .attr('opacity', 0.92);

  const [wi, wj] = R.worst_pair;
  markG.append('line').attr('class', 'straight')
    .attr('x1', rx(proj[wi][0])).attr('y1', ry(proj[wi][1]))
    .attr('x2', rx(proj[wj][0])).attr('y2', ry(proj[wj][1]))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 2.4)
    .attr('stroke-dasharray', '6 4').attr('opacity', 0);
  [wi, wj].forEach((i) => {
    markG.append('circle').attr('class', 'end')
      .attr('cx', rx(proj[i][0])).attr('cy', ry(proj[i][1])).attr('r', 8)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.4)
      .attr('opacity', 0);
  });

  function setEdges(show) {
    const e = show ? computed.iso.edges : [];
    edgesG.selectAll('line').data(e).join('line')
      .attr('x1', (d) => rx(proj[d[0]][0])).attr('y1', (d) => ry(proj[d[0]][1]))
      .attr('x2', (d) => rx(proj[d[1]][0])).attr('y2', (d) => ry(proj[d[1]][1]))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6).attr('opacity', 0.35);
  }

  /* The actual shortest path between the two ends, walked through the graph the
     widget below builds, so the drawing is the algorithm and not an artist's
     impression of it. */
  function shortestPathNodes(adj, from, to) {
    const n = adj.length;
    const dist = new Float64Array(n).fill(Infinity);
    const prev = new Int32Array(n).fill(-1);
    const done = new Uint8Array(n);
    dist[from] = 0;
    for (;;) {
      let u = -1;
      let best = Infinity;
      for (let i = 0; i < n; i++) if (!done[i] && dist[i] < best) { best = dist[i]; u = i; }
      if (u === -1 || u === to) break;
      done[u] = 1;
      for (const [v, wt] of adj[u]) {
        if (dist[u] + wt < dist[v]) {
          dist[v] = dist[u] + wt;
          prev[v] = u;
        }
      }
    }
    const path = [];
    for (let v = to; v !== -1; v = prev[v]) path.push(v);
    return path.reverse();
  }
  const walk = shortestPathNodes(computed.iso.adj, wi, wj);

  function setPath(show) {
    pathG.selectAll('path').data(show ? [walk] : []).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3)
      .attr('stroke-linejoin', 'round')
      .attr('d', d3.line().x((i) => rx(proj[i][0])).y((i) => ry(proj[i][1])));
  }

  /* -------------------------------------------------------------- the 2-D view */
  function drawFlat(Z) {
    const { x, y } = equalScales(Z, w, h, { pad: 1.08 });
    drawGrid(layers.flat, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layers.flat, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(layers.flat, w, h, { x: 'first coordinate', y: 'second coordinate' });
    flatDots.selectAll('circle').data(Z).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.4)
      .attr('fill', (d, i) => ramp(state.chart[i][0])).attr('opacity', 0.9);
  }

  function show(which) {
    Object.entries(layers).forEach(([k, layer]) => {
      layer.transition(`fade-${k}`).duration(450).attr('opacity', k === which ? 1 : 0);
    });
  }

  const steps = [
    () => {
      show('roll');
      setEdges(false);
      setPath(false);
      markG.selectAll('*').attr('opacity', 0);
      wrapLabel(title, `${state.points.length} points on a sheet that has been rolled up, `
        + 'coloured by where they sit along it', w - 8);
      wrapLabel(caption, 'the colour is the answer, and no method below is ever shown it', w - 8);
    },
    () => {
      show('roll');
      setEdges(false);
      setPath(false);
      markG.selectAll('*').attr('opacity', 1);
      wrapLabel(title, 'two points a hand apart in the room', w - 8);
      wrapLabel(caption, 'and most of a corridor apart along the sheet, which their colours say',
        w - 8);
    },
    () => {
      show('flat');
      drawFlat(computed.pca);
      wrapLabel(title, 'the best flat projection of the same points', w - 8);
      wrapLabel(caption, 'the ramp is folded over itself: far ends of the sheet land on top of '
        + 'each other', w - 8);
    },
    () => {
      show('roll');
      setEdges(true);
      setPath(false);
      markG.selectAll('*').attr('opacity', 1);
      wrapLabel(title, `each point joined to its ${computed.iso.k} nearest neighbours`, w - 8);
      wrapLabel(caption, 'short hops are the only distances a curved sheet lets you trust', w - 8);
    },
    () => {
      show('roll');
      setEdges(true);
      setPath(true);
      markG.selectAll('*').attr('opacity', 1);
      wrapLabel(title, `the shortest way between the two ends, along ${walk.length - 1} edges`,
        w - 8);
      wrapLabel(caption, 'add up the hops and you have an estimate of the distance along the sheet',
        w - 8);
    },
    () => {
      show('flat');
      drawFlat(computed.iso.coords);
      const pairs = (state.points.length * (state.points.length - 1)) / 2;
      wrapLabel(title, `do that for all ${pairs.toLocaleString('en-US')} pairs and hand the `
        + 'table to classical scaling', w - 8);
      wrapLabel(caption, 'the ramp comes back in one piece: the sheet is unrolled', w - 8);
    },
  ];

  steps[0]();
  return scrolly('#roll-scrolly .step', steps);
}
