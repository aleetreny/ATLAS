/* Density clustering, told in five steps on the two moons.
 *
 * Everything on screen is computed live by this page's DBSCAN: the eps-ball
 * demonstrations pick their exhibit points deterministically (the core with
 * the most neighbours, and the loneliest non-noise point), and the growth
 * step replays the BFS discovery order of the first cluster in three phases.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { dbscan, neighbours, ari } from './dbkit.js';
import { COLOURS, drawLabelled } from './densdraw.js';

export function initScrollyDensity(data) {
  const pts = data.datasets.moons.points;
  const truth = data.datasets.moons.true_labels;
  const cross = data.cross.moons;

  const { g, w, h } = makeChart('#den-chart', {
    width: 620,
    height: 560,
    margin: { top: 34, right: 26, bottom: 56, left: 60 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const ringG = g.append('g');
  const dotG = g.append('g');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* Live runs at the two epsilons the story uses. */
  const EPS = 0.3;
  const MINPTS = 5;
  const anatomy = dbscan(pts, EPS, MINPTS);
  const final = dbscan(pts, 0.45, 5);
  const nb = neighbours(pts, EPS);

  /* Exhibits: the best-connected core, and the loneliest border point. */
  let star = 0;
  let lone = -1;
  nb.forEach((list, i) => {
    if (list.length > nb[star].length) star = i;
    if (!anatomy.core[i] && anatomy.labels[i] !== -1 && (lone === -1 || list.length < nb[lone].length)) lone = i;
  });

  function balls(show) {
    const data2 = show
      ? [{ i: star, cls: 'core' }, ...(lone >= 0 ? [{ i: lone, cls: 'border' }] : [])]
      : [];
    ringG.selectAll('circle.ball').data(data2).join('circle').attr('class', 'ball')
      .attr('cx', (d) => x(pts[d.i][0])).attr('cy', (d) => y(pts[d.i][1]))
      .attr('r', Math.abs(x(EPS) - x(0)))
      .attr('fill', (d) => (d.cls === 'core' ? 'var(--primary)' : 'var(--cosmos)'))
      .attr('fill-opacity', 0.10)
      .attr('stroke', (d) => (d.cls === 'core' ? 'var(--primary)' : 'var(--cosmos)'))
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', (d) => (d.cls === 'core' ? null : '6 4'));
  }

  const states = {
    0: () => {
      balls(false);
      drawLabelled(dotG, pts, null, null, x, y);
      caption.text(`two moons: k-means scored ARI ${cross.km}, the mixture ${cross.gmm}`);
    },
    1: () => {
      balls(true);
      drawLabelled(dotG, pts, null, null, x, y);
      caption.text(`radius ${EPS}: one crowded ball (${nb[star].length + 1} points), one that holds only ${nb[lone].length + 1}`);
    },
    2: () => {
      balls(false);
      drawLabelled(dotG, pts, anatomy.labels.map((l, i) => (anatomy.core[i] ? l : -1)), anatomy.role, x, y);
      caption.text(`the cores: every point whose ball holds at least ${MINPTS}`);
    },
    3: () => {
      balls(false);
      drawLabelled(dotG, pts, anatomy.labels, anatomy.role, x, y);
      caption.text(`cores chained, borders adopted, ${anatomy.nNoise} points called noise (eps ${EPS})`);
    },
    4: () => {
      balls(false);
      drawLabelled(dotG, pts, final.labels, final.role, x, y);
      caption.text(`eps 0.45: two moons, ARI ${ari(final.labels, truth).toFixed(3)}`);
    },
  };

  scrolly('#den-scrolly .step', states);
}
