/* Lloyd's loop, told in six steps, run live.
 *
 * The whole trajectory is computed once from a seeded bad start (chosen so the
 * story of steps 1-4 is actually visible: one cluster starts with two centres)
 * and each scrolly step shows one stage. Handlers are idempotent so scrolling
 * back replays cleanly.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { assign, lloyd, initRandom, rng } from './kmeanskit.js';
import { CLUSTER_COLOURS, drawClusteredPoints, drawCenters, drawBorders } from './clusterdraw.js';

export function initScrollyLloyd(data) {
  const pts = data.blobs.points;
  const { g, w, h } = makeChart('#lloyd-chart', {
    width: 620,
    height: 560,
    margin: { top: 34, right: 26, bottom: 56, left: 60 },
  });

  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const borderG = g.append('g');
  const dotG = g.append('g');
  const centreG = g.append('g');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* Hunt for a seed whose random start physically places exactly two centres
   * inside one true cluster and the third in another, leaving one cluster with
   * no centre at all: the picture the step-2 caption describes. Deterministic:
   * the same seed wins every load. */
  const homeOf = (c) => {
    let bi = 0;
    let bd = Infinity;
    pts.forEach((p, i) => {
      const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;
      if (d < bd) { bd = d; bi = i; }
    });
    return data.blobs.true_labels[bi];
  };
  let start = null;
  for (let s = 1; s < 200 && !start; s++) {
    const cand = initRandom(pts, 3, rng(s));
    if (new Set(cand.map(homeOf)).size === 2) start = cand;
  }
  if (!start) start = initRandom(pts, 3, rng(1));
  const path = lloyd(pts, start);
  const last = path[path.length - 1];

  const state = (stage) => {
    /* stage: 0 bare, 1 centres only, 2 first assign, 3 first update,
     * 4 midway, 5 converged */
    const frame = stage <= 2 ? path[0] : stage === 3 ? path[1] : stage === 4 ? path[Math.min(3, path.length - 1)] : last;
    const showLabels = stage >= 2;
    const showCentres = stage >= 1;
    drawClusteredPoints(dotG, pts, showLabels ? frame.labels : null, x, y, { unlabelled: !showLabels });
    drawCenters(centreG, showCentres ? frame.centers : [], x, y);
    drawBorders(borderG, showCentres && stage >= 2 ? frame.centers : [], x, y, w, h);
    const J = assign(pts, frame.centers).inertia;
    caption.text(
      stage === 0 ? `${pts.length} points, no labels anywhere`
        : stage === 1 ? 'three centres, dropped at random data points'
          : stage === 5 ? `converged: inertia ${last.inertia.toFixed(1)} (scikit-learn's reference: ${data.blobs.ref_inertia})`
            : `inertia ${J.toFixed(1)}, and it only ever falls`
    );
  };

  scrolly('#lloyd-scrolly .step', {
    0: () => state(0),
    1: () => state(1),
    2: () => state(2),
    3: () => state(3),
    4: () => state(4),
    5: () => state(5),
  });

  /* Runtime check on the claim in step 6: this run's converged inertia matches
   * the sklearn reference (same data, both at the good optimum). */
  if (Math.abs(last.inertia - data.blobs.ref_inertia) / data.blobs.ref_inertia > 0.01) {
    console.warn(`scrolly run converged at ${last.inertia.toFixed(1)}, reference is ${data.blobs.ref_inertia}`);
  }
}
