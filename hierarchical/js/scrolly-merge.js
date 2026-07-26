/* Agglomeration watched happening: twenty-four points, and the tree they
 * become, growing side by side in one chart. Left panel: the points, with
 * every completed merge drawn as a link between the pair that fused. Right
 * panel: the dendrogram so far. The whole run is this page's own ward
 * agglomeration, computed once.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { agglomerate, cutTree, dendrogram } from './hierkit.js';

const COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)'];

/* A small seeded dataset with three obvious clumps. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const r = rng(7);
const gauss = () => (r() + r() + r() + r() - 2) * 1.15;
const CENTRES = [[-2.4, 1.5], [2.2, 1.9], [0.2, -2.0]];
const PTS = [];
CENTRES.forEach((c) => {
  for (let i = 0; i < 8; i++) PTS.push([c[0] + gauss() * 0.62, c[1] + gauss() * 0.62]);
});
const N = PTS.length;

export function initScrollyMerge() {
  const { g, w, h } = makeChart('#agg-chart', {
    width: 640,
    height: 560,
    margin: { top: 34, right: 20, bottom: 30, left: 20 },
  });

  /* Left: scatter. Right: growing dendrogram. */
  const sw = w * 0.52;
  const dw = w * 0.40;
  const dx0 = w - dw;
  const x = d3.scaleLinear().domain(d3.extent(PTS, (p) => p[0])).nice().range([0, sw]);
  const y = d3.scaleLinear().domain(d3.extent(PTS, (p) => p[1])).nice().range([h * 0.9, h * 0.08]);

  const Z = agglomerate(PTS, 'ward');
  const layout = dendrogram(Z, N);
  const maxH = Z[Z.length - 1][2];
  const dxs = d3.scaleLinear().domain([0, N - 1]).range([dx0, dx0 + dw]);
  const dys = d3.scaleSqrt().domain([0, maxH * 1.05]).range([h * 0.92, h * 0.06]);

  const linkG = g.append('g');
  const dotG = g.append('g');
  const treeG = g.append('g');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.64rem');
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', dx0 + dw / 2).attr('y', h).attr('text-anchor', 'middle')
    .style('font-size', '0.54rem').text('the tree so far');

  /* Which merge joined which two POINTS: for drawing the links, use the
   * closest cross-pair between the two merged groups' members. */
  const members = [];
  for (let i = 0; i < N; i++) members.push([i]);
  const mergeLinks = Z.map((row) => {
    const A = row[0] < N ? [row[0]] : members[row[0]];
    const B = row[1] < N ? [row[1]] : members[row[1]];
    let best = [A[0], B[0]];
    let bd = Infinity;
    for (const a of A) {
      for (const b of B) {
        const d = (PTS[a][0] - PTS[b][0]) ** 2 + (PTS[a][1] - PTS[b][1]) ** 2;
        if (d < bd) { bd = d; best = [a, b]; }
      }
    }
    members.push([...A, ...B]);
    return best;
  });

  function state(nMerges, colour) {
    const k = N - nMerges;
    const labels = cutTree(Z, N, Math.max(k, 1));
    const showColours = colour && k <= 3;
    dotG.selectAll('circle').data(PTS).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 4.5)
      .attr('fill', (p, i) => (showColours ? COLOURS[labels[i] % 3] : 'var(--stone)'))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8).attr('opacity', 0.9);
    linkG.selectAll('line').data(mergeLinks.slice(0, nMerges)).join('line')
      .attr('x1', ([a]) => x(PTS[a][0])).attr('y1', ([a]) => y(PTS[a][1]))
      .attr('x2', ([, b]) => x(PTS[b][0])).attr('y2', ([, b]) => y(PTS[b][1]))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6).attr('opacity', 0.55);
    treeG.selectAll('path').data(layout.merges.slice(0, nMerges)).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 1.6)
      .attr('d', (m) =>
        `M${dxs(m.xa)},${dys(m.ha)} L${dxs(m.xa)},${dys(m.h)} L${dxs(m.xb)},${dys(m.h)} L${dxs(m.xb)},${dys(m.hb)}`);
    caption.text(
      nMerges === 0 ? `${N} points, ${N} clusters: everyone starts alone`
        : nMerges >= N - 1 ? 'one cluster, and the whole history is the tree'
          : `${nMerges} merge${nMerges === 1 ? '' : 's'} done, ${N - nMerges} clusters left`
    );
  }

  scrolly('#agg-scrolly .step', {
    0: () => state(0, false),
    1: () => state(1, false),
    2: () => state(12, false),
    3: () => state(21, true),
    4: () => state(N - 1, false),
  });
}
