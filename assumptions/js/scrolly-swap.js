/* PAM, watched. Twenty-five of the blobs' points, the same seed as every other
 * article in this module, and the two phases of the algorithm one at a time:
 * BUILD collects k medoids greedily, SWAP exchanges the one pair that helps
 * most, until no exchange helps at all.
 *
 * Every number in the annotations comes out of the run itself, including
 * whether a swap improved anything: if BUILD lands on the answer, the page
 * says so rather than pretending there was a dramatic exchange.
 */
import { makeChart, drawAxes, drawGrid } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { distanceMatrix, pam, kmeans } from './relaxkit.js';
import { drawPoints, drawMedoids, drawMeans, colourOf } from './draw.js';

const STRIDE = 12;

export function initScrollySwap(data) {
  const pts = data.datasets.blobs.points.filter((_, i) => i % STRIDE === 0);
  const D = distanceMatrix(pts);
  const run = pam(D, 3, 100, { trace: true });
  const km = kmeans(pts, 3, { restarts: 20 });
  const build = run.steps.filter((s) => s.phase === 'build');
  const swaps = run.steps.filter((s) => s.phase === 'swap');
  const done = run.steps.find((s) => s.phase === 'done');

  const { g, w, h } = makeChart('#swap-chart', {
    width: 640, height: 470, margin: { top: 40, right: 26, bottom: 46, left: 52 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });

  const spokeG = g.append('g');
  const dotG = g.append('g');
  const medG = g.append('g');
  const meanG = g.append('g');
  const arrowG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -18).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  const sub = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -2).attr('text-anchor', 'middle').style('font-size', '0.6rem');

  /* Which medoid every point currently answers to, and the lines that make the
   * quantity being minimised visible: the sum of the lengths of these spokes. */
  const assign = (med) => pts.map((p, i) => {
    let b = Infinity;
    let a = 0;
    med.forEach((m, c) => {
      if (D[m][i] < b) {
        b = D[m][i];
        a = c;
      }
    });
    return a;
  });

  function show(med, { spokes = true, labelled = true, arrow = null, means = false } = {}) {
    const labels = med.length && labelled ? assign(med) : null;
    drawPoints(dotG, pts, labels, x, y, { r: 4.4 });
    drawMedoids(medG, med.map((m) => pts[m]), x, y, { r: 10 });
    drawMeans(meanG, means ? km.centers : [], x, y, { size: 13 });

    const lines = spokes && med.length
      ? pts.map((p, i) => ({ p, m: pts[med[labels ? labels[i] : 0]], c: labels ? labels[i] : 0 }))
      : [];
    spokeG.selectAll('line').data(lines).join('line')
      .attr('x1', (d) => x(d.p[0])).attr('y1', (d) => y(d.p[1]))
      .attr('x2', (d) => x(d.m[0])).attr('y2', (d) => y(d.m[1]))
      .attr('stroke', (d) => colourOf(d.c))
      .attr('stroke-width', 1).attr('opacity', 0.45);

    const arr = arrow ? [arrow] : [];
    arrowG.selectAll('path').data(arr).join('path')
      .attr('d', (d) => {
        const a = pts[d.from];
        const b = pts[d.to];
        return `M${x(a[0])},${y(a[1])} L${x(b[0])},${y(b[1])}`;
      })
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2.5)
      .attr('stroke-dasharray', '7 4').attr('fill', 'none');
    arrowG.selectAll('circle').data(arr).join('circle')
      .attr('cx', (d) => x(pts[d.to][0])).attr('cy', (d) => y(pts[d.to][1]))
      .attr('r', 14).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2);
  }

  const fmt = (v) => v.toFixed(2);
  const handlers = {
    0: () => {
      show([], { spokes: false });
      const triples = (pts.length * (pts.length - 1) * (pts.length - 2)) / 6;
      title.text(`${pts.length} points, ${triples.toLocaleString('en-US')} possible sets of three medoids`);
      sub.text('a centre has to be one of them');
    },
    1: () => {
      show([build[0].medoids[0]], { labelled: false });
      title.text(`one medoid: total distance ${fmt(build[0].loss)}`);
      sub.text('the point with the smallest sum of distances to everyone else');
    },
    2: () => {
      const last = build[build.length - 1];
      show(last.medoids);
      title.text(`build done, three medoids: total distance ${fmt(last.loss)}`);
      sub.text('each one added for the largest reduction it could buy');
    },
    3: () => {
      if (swaps.length) {
        const s = swaps[0];
        show(s.medoids, { arrow: { from: s.from, to: s.to } });
        title.text(`swap: ${fmt(s.was)} down to ${fmt(s.loss)}`);
        sub.text('the best of every medoid-for-point exchange there is');
      } else {
        const last = build[build.length - 1];
        show(last.medoids);
        title.text(`no swap helps: the best one on offer costs ${fmt(done.rejected.cost)} against ${fmt(last.loss)}`);
        sub.text('build had already arrived, and the search says so');
      }
    },
    4: () => {
      show(run.medoids, { means: true });
      title.text(`settled at ${fmt(run.loss)} after ${swaps.length} swap${swaps.length === 1 ? '' : 's'}`);
      sub.text('rings are medoids, real points; diamonds are k-means, which are not');
    },
  };
  const s = scrolly('#swap-scrolly .step', handlers);
  handlers[0]();
  return s;
}
