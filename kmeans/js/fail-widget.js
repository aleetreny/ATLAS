/* The failure gallery: three datasets, each violating one assumption.
 *
 * Both views keep the same points and only recolour, so the eye can hold the
 * shape fixed while the story changes: this is what is there, this is what
 * k-means says is there, and the gap between the two is the lesson.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { drawClusteredPoints, drawCenters, drawBorders } from './clusterdraw.js';

const BLURBS = {
  anisotropic: {
    label: 'stretched',
    what: 'Three clusters sheared into parallel cigars. Distance now means different things in different directions, and the straight Voronoi borders cut across every cigar.',
    broke: 'the assumption that distance treats all directions alike',
  },
  rings: {
    label: 'rings',
    what: 'One ring inside another. The natural groups are not blobs at all, and both mean-centres land in the middle where neither cluster lives. K-means cuts the picture into two half-moons.',
    broke: 'the assumption that a cluster has a centre inside it',
  },
  unequal: {
    label: 'unequal',
    what: 'A small tight cluster beside a huge diffuse one. Minimising summed squared distance makes stealing the giant’s edge irresistible, because a mean-based border cannot know one cluster is wider than the other.',
    broke: 'the assumption that clusters have similar spread',
  },
};

export function initFailWidget(data) {
  const gallery = data.gallery;
  const { g, w, h } = makeChart('#fail-chart', {
    width: 660,
    height: 460,
    margin: { top: 34, right: 26, bottom: 50, left: 54 },
  });

  const borderG = g.append('g');
  const dotG = g.append('g');
  const centreG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const btnRow = document.querySelector('#fail-buttons');
  const readout = document.querySelector('#fail-readout');
  const btnKm = document.querySelector('#fail-km');
  const btnTrue = document.querySelector('#fail-true');
  const state = { i: 0, view: 'km' };

  const caseBtns = gallery.map((c, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = BLURBS[c.name].label;
    b.addEventListener('click', () => {
      state.i = i;
      render();
    });
    btnRow.appendChild(b);
    return b;
  });

  function render() {
    const c = gallery[state.i];
    const pts = c.points;
    const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });

    const km = state.view === 'km';
    drawClusteredPoints(dotG, pts, km ? c.km_labels : c.true_labels, x, y, { r: 3.4 });
    drawCenters(centreG, km ? c.centers : [], x, y);
    drawBorders(borderG, km ? c.centers : [], x, y, w, h);

    caseBtns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    btnKm.classList.toggle('ghost', !km);
    btnTrue.classList.toggle('ghost', km);
    title.text(km
      ? `k-means, given the right k = ${c.k}: agreement with the truth, ARI ${c.ari.toFixed(2)}`
      : 'the actual groups the data was built from');

    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">${BLURBS[c.name].what}</div>` +
      `<div class="slider-label" style="margin-top:.5rem;line-height:1.5">` +
      `What broke: <span class="bold">${BLURBS[c.name].broke}</span>. ` +
      `Adjusted Rand index against the truth: <span class="value">${c.ari.toFixed(3)}</span>, where 1 is ` +
      `perfect recovery and 0 is what random guessing scores on average.</div>`;
  }

  btnKm.addEventListener('click', () => {
    state.view = 'km';
    render();
  });
  btnTrue.addEventListener('click', () => {
    state.view = 'true';
    render();
  });
  render();
}
