/* The 178 bottles again, this time clustered around exemplars.
 *
 * Nothing here needs coordinates: PAM only ever looks at a table of
 * dissimilarities, so the metric is a free choice and three of them are on
 * offer. The picture is the same PCA projection the earlier articles used, and
 * because that projection is linear, the mean of the projected points IS the
 * projection of the mean, so k-means' centres can be drawn honestly beside
 * medoids that are bottles.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { distanceMatrix, pam, kmeans, ari } from './relaxkit.js';
import { drawPoints, drawMeans, drawMedoids } from './draw.js';

const METHODS = [
  { key: 'kmeans', label: 'k-means', centre: 'the average of a cluster: a bottle that does not exist' },
  { key: 'euclidean', label: 'medoids, euclidean', centre: 'the bottle with the smallest total distance to its cluster' },
  { key: 'manhattan', label: 'medoids, manhattan', centre: 'the same, adding up differences instead of squaring them' },
  { key: 'cosine', label: 'medoids, cosine', centre: 'the same, with distance meaning angle rather than length' },
];

export function initWineWidget(data) {
  const W = data.wine;
  const proj = W.proj;
  const truth = W.true;
  const cache = {};

  const { g, w, h } = makeChart('#wine-chart', {
    width: 640, height: 430, margin: { top: 34, right: 26, bottom: 50, left: 56 },
  });
  const x = d3.scaleLinear().domain(d3.extent(proj, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(proj, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'first principal component', y: 'second' });
  const dotG = g.append('g');
  const medG = g.append('g');
  const meanG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const btnRow = document.querySelector('#wine-buttons');
  const readout = document.querySelector('#wine-readout');
  const state = { i: 1 };

  const btns = METHODS.map((mth, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = mth.label;
    b.addEventListener('click', () => {
      state.i = i;
      render();
    });
    btnRow.appendChild(b);
    return b;
  });

  /* Every run happens here, in the page, and is cached because the reader will
     click back and forth and a 178 by 178 matrix is worth keeping. */
  function runOf(key) {
    if (cache[key]) return cache[key];
    if (key === 'kmeans') {
      const km = kmeans(W.features, 3, { restarts: 10 });
      cache[key] = { labels: km.labels, medoids: null };
    } else {
      const D = distanceMatrix(W.features, key);
      const r = pam(D, 3);
      cache[key] = { labels: r.labels, medoids: r.medoids, loss: r.loss };
    }
    return cache[key];
  }

  function render() {
    const mth = METHODS[state.i];
    const run = runOf(mth.key);
    const a = ari(truth, run.labels);

    drawPoints(dotG, proj, run.labels, x, y, { r: 3.8 });
    drawMedoids(medG, run.medoids ? run.medoids.map((i) => proj[i]) : [], x, y, { r: 9 });

    /* the projected means, computed from the projected points: identical to
       projecting the 13-dimensional centroid, because the projection is linear */
    let means = [];
    if (!run.medoids) {
      const sums = [0, 1, 2].map(() => [0, 0, 0]);
      run.labels.forEach((l, i) => {
        sums[l][0] += proj[i][0];
        sums[l][1] += proj[i][1];
        sums[l][2] += 1;
      });
      means = sums.map((s) => [s[0] / s[2], s[1] / s[2]]);
    }
    drawMeans(meanG, means, x, y, { size: 14 });

    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    title.text(`${mth.label}: ARI ${a.toFixed(3)} against the three cultivars`);

    const ref = W.metrics.find((m) => m.key === mth.key);
    const exemplars = run.medoids
      ? run.medoids.map((i) => `#${i} (cultivar ${truth[i] + 1})`).join(', ')
      : null;
    readout.innerHTML =
      `<table class="rel-table">
        <tr><th>what the centre is</th><th>ARI</th></tr>
        <tr><td>${mth.centre}</td><td><span class="value">${a.toFixed(3)}</span></td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      (exemplars
        ? `The three exemplars are bottles ${exemplars}: one from each cultivar, and each of `
          + `them is a wine somebody could actually open. `
        : `There is nothing to point at: the centre of this cluster is an average of `
          + `13 standardised measurements, and no bottle in the cellar has them. `) +
      `On these same 178 bottles k-means scores ${W.km_ari.toFixed(3)}, the mixture `
      + `${W.gmm_ari.toFixed(3)} and ward's tree ${W.ward_ari.toFixed(3)}` +
      (ref ? `, so this costs ${(W.km_ari - ref.ari).toFixed(3)} of agreement against k-means `
        + `and buys an exemplar.` : `.`) +
      `</div>`;
  }

  render();
}
