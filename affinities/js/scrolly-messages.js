/* Affinity propagation, watched.
 *
 * Thirty of the blobs, which are the same thirty the generator enumerated
 * every subset of, so the last step can say something stronger than "it looks
 * right": on these points the message passing lands on the exact optimum of
 * its own objective.
 *
 * Every annotation is read off the run itself. The messages are real messages:
 * the arrows are entries of R and A copied out of the loop at named iterations,
 * not a drawing of what messages would look like.
 */
import { makeChart, drawAxes, drawGrid } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { similarity, affinityPropagation } from './graphkit.js';
import { drawPoints, drawExemplars, colourOf, legend } from './draw.js';

const SHOTS = [1, 2, 8];

export function initScrollyMessages(data) {
  const stride = data.ap.objective.brute ? data.ap.objective.sub_stride || 10 : 10;
  const pts = data.datasets.blobs.points.filter((_, i) => i % stride === 0);
  const n = pts.length;
  const S = similarity(pts);
  const pref = data.ap.objective.brute.chosen.preference;
  const run = affinityPropagation(S, n, {
    preference: pref, damping: 0.9, snapshots: SHOTS, trace: true,
  });

  /* Three bands live in the top margin (title, subtitle, legend); at 44px the
     legend's row collided with a centred title long enough to reach the left
     edge, so each band gets its own 14 to 16px lane. */
  const { g, w, h } = makeChart('#messages-chart', {
    width: 640, height: 470, margin: { top: 60, right: 26, bottom: 46, left: 52 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });

  const edgeG = g.append('g');
  const dotG = g.append('g');
  const exG = g.append('g');
  const markG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -44).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  const sub = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -28).attr('text-anchor', 'middle').style('font-size', '0.6rem');
  const key = legend(g, [], { x0: 4, y0: -10, gap: 150 });

  /* The point whose messages get drawn: the one that ends up an exemplar with
     the most members, so the arrows are about a decision that matters. */
  const sizes = run.exemplars.map((e, c) => run.labels.filter((l) => l === c).length);
  const focusC = sizes.indexOf(Math.max(...sizes));
  const focus = run.exemplars[focusC];
  /* and a talker: the point closest to the middle of the whole cloud. A point
     out at the edge has an obvious answer and draws all eight of its arrows on
     top of each other; one in the middle has a real choice to make, which is
     what the step is about. */
  const mid = [0, 1].map((c) => pts.reduce((s2, p) => s2 + p[c], 0) / n);
  let talker = 0;
  let near = Infinity;
  for (let i = 0; i < n; i++) {
    if (run.exemplars.includes(i)) continue;
    const d = (pts[i][0] - mid[0]) ** 2 + (pts[i][1] - mid[1]) ** 2;
    if (d < near) {
      near = d;
      talker = i;
    }
  }

  const arrows = (rows, colour) => {
    edgeG.selectAll('line').data(rows).join('line')
      .attr('x1', (d) => x(pts[d.from][0])).attr('y1', (d) => y(pts[d.from][1]))
      .attr('x2', (d) => x(pts[d.to][0])).attr('y2', (d) => y(pts[d.to][1]))
      .attr('stroke', colour)
      .attr('stroke-width', (d) => 0.4 + 3.6 * d.w)
      .attr('opacity', (d) => 0.15 + 0.65 * d.w);
  };

  const ring = (idx, colour) => {
    markG.selectAll('circle.focus').data(idx).join('circle').attr('class', 'focus')
      .attr('cx', (i) => x(pts[i][0])).attr('cy', (i) => y(pts[i][1]))
      .attr('r', 9).attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
  };

  const clear = () => {
    edgeG.selectAll('line').remove();
    markG.selectAll('circle.focus').remove();
    exG.selectAll('circle.exemplar').remove();
  };

  const f2 = (v) => v.toFixed(2);
  const handlers = {
    0: () => {
      clear();
      drawPoints(dotG, pts, null, x, y, { r: 5 });
      key.remove();
      legend(g, [{ colour: 'var(--squidink)', shape: 'dot', label: 'candidate exemplar' }],
        { x0: 4, y0: -10, gap: 150 });
      title.text(`${n} points, ${n * n} similarities, and ${n} candidates`);
      sub.text('no k anywhere: every point is applying for the job');
    },
    1: () => {
      clear();
      drawPoints(dotG, pts, null, x, y, { r: 5, faded: (i) => i !== talker });
      const shot = run.shots[1];
      const vals = [];
      for (let j = 0; j < n; j++) vals.push({ from: talker, to: j, v: shot.R[talker * n + j] });
      const top = vals.filter((d) => d.to !== talker).sort((a, b) => b.v - a.v).slice(0, 8);
      /* Width across the range of the eight drawn, not clipped at zero: all but
         the winner are negative here, and clipping made seven of the eight
         invisible. */
      const hi = top[0].v;
      const lo = top[top.length - 1].v;
      arrows(top.map((d) => ({ ...d, w: (d.v - lo) / (hi - lo || 1) })), 'var(--primary)');
      ring([talker], 'var(--primary)');
      title.text(`responsibilities out of one point, after one iteration`);
      sub.text(`its best candidate gets r = ${f2(top[0].v)}, the eighth gets ${f2(top[7].v)}`);
      key.remove();
      legend(g, [{ colour: 'var(--primary)', shape: 'line', label: 'r(i,k), thicker is stronger' }],
        { x0: 4, y0: -10, gap: 150 });
    },
    2: () => {
      clear();
      drawPoints(dotG, pts, null, x, y, { r: 5, faded: (i) => i === focus });
      const shot = run.shots[2];
      const vals = [];
      for (let i = 0; i < n; i++) {
        if (i === focus) continue;
        vals.push({ from: i, to: focus, v: shot.A[i * n + focus] });
      }
      const top = vals.sort((a, b) => b.v - a.v).slice(0, 10);
      const lo = Math.min(...top.map((d) => d.v));
      arrows(top.map((d) => ({ ...d, w: lo < 0 ? 1 - d.v / lo : 1 })), 'var(--aqua)');
      ring([focus], 'var(--aqua)');
      title.text('availabilities arriving at one candidate, after two iterations');
      sub.text(`it is offering itself at a = ${f2(shot.A[focus * n + focus])} to its own account`);
      key.remove();
      legend(g, [{ colour: 'var(--aqua)', shape: 'line', label: 'a(i,k), thicker is closer to zero' }],
        { x0: 4, y0: -10, gap: 190 });
    },
    3: () => {
      clear();
      const shot = run.shots[8];
      const selfScore = [];
      for (let i = 0; i < n; i++) selfScore.push(shot.A[i * n + i] + shot.R[i * n + i]);
      const elected = selfScore.map((v) => v > 0);
      drawPoints(dotG, pts, null, x, y, { r: 5, faded: (i) => !elected[i] });
      ring(elected.map((v, i) => (v ? i : -1)).filter((i) => i >= 0), 'var(--primary)');
      title.text(`after eight iterations, ${elected.filter(Boolean).length} points elect themselves`);
      sub.text(`the count settles at ${run.exemplars.length}, and the run stops when it holds still`);
      key.remove();
      legend(g, [{ colour: 'var(--primary)', shape: 'ring', label: 'r(k,k) + a(k,k) > 0' }],
        { x0: 4, y0: -10, gap: 150 });
    },
    4: () => {
      clear();
      drawPoints(dotG, pts, run.labels, x, y, { r: 5 });
      const rows = [];
      for (let i = 0; i < n; i++) {
        const e = run.exemplars[run.labels[i]];
        if (e !== i) rows.push({ from: i, to: e, w: 0.35, c: run.labels[i] });
      }
      edgeG.selectAll('line').data(rows).join('line')
        .attr('x1', (d) => x(pts[d.from][0])).attr('y1', (d) => y(pts[d.from][1]))
        .attr('x2', (d) => x(pts[d.to][0])).attr('y2', (d) => y(pts[d.to][1]))
        .attr('stroke', (d) => colourOf(d.c)).attr('stroke-width', 1).attr('opacity', 0.45);
      drawExemplars(exG, run.exemplars.map((i) => pts[i]), x, y, { r: 12 });
      title.text(`${run.exemplars.length} exemplars after ${run.iters} iterations, nobody having said 3`);
      sub.text('rings are exemplars, and every one of them is one of the points');
      key.remove();
      legend(g, [{ colour: 'var(--squidink)', shape: 'ring', label: 'exemplars' }],
        { x0: 4, y0: -10, gap: 150 });
    },
  };
  const s = scrolly('#messages-scrolly .step', handlers);
  handlers[0]();
  return { scrolly: s, run, n, pref };
}
