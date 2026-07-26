/* The failure gallery, revisited with the model that learns shape.
 *
 * Same three datasets as the k-means article, same layout, and the ARI pair
 * km -> gmm in every readout. Two redemptions and one honest refusal: a
 * mixture of blobs still cannot describe a ring, which is the next article's
 * opening argument.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { COLOURS, drawMeans, drawEllipses } from './mixdraw.js';

const BLURBS = {
  anisotropic: 'The sheared cigars. Full covariance learns each tilt and recovers the truth outright: ' +
    'the failure was never in the data, it was in a model that could only draw circles.',
  rings: 'The rings, still broken, and honestly so: a Gaussian is a blob, a weighted sum of blobs is ' +
    'blobs, and no amount of covariance can bend one into an annulus. Density, not shape, is what ' +
    'defines these clusters, and density is the next article.',
  unequal: 'The giant and the dwarf. The mixture learns one huge ellipse and one tight one, and the ' +
    'weights learn that four points in five belong to the giant: exactly the two facts k-means had no ' +
    'vocabulary for.',
};

export function initRedeemWidget(data) {
  const gallery = data.gallery;
  const { g, w, h } = makeChart('#redeem-chart', {
    width: 660,
    height: 460,
    margin: { top: 34, right: 26, bottom: 50, left: 54 },
  });

  const dotG = g.append('g');
  const ellG = g.append('g');
  const meanG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const btnRow = document.querySelector('#redeem-buttons');
  const readout = document.querySelector('#redeem-readout');
  const btnGmm = document.querySelector('#redeem-gmm');
  const btnTrue = document.querySelector('#redeem-true');
  const state = { i: 0, view: 'gmm' };

  const caseBtns = gallery.map((c, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = c.name === 'anisotropic' ? 'stretched' : c.name;
    b.addEventListener('click', () => { state.i = i; render(); });
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

    const gmm = state.view === 'gmm';
    dotG.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 3.2)
      .attr('fill', (p, i) => COLOURS[(gmm ? c.labels[i] : c.true_labels[i]) % COLOURS.length])
      .attr('stroke', 'white').attr('stroke-width', 0.5).attr('opacity', 0.82);
    drawEllipses(ellG, gmm ? c : { means: [], covs: [] }, x, y, { sigmas: [2] });
    drawMeans(meanG, gmm ? c.means : [], x, y);

    caseBtns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    btnGmm.classList.toggle('ghost', !gmm);
    btnTrue.classList.toggle('ghost', gmm);
    title.text(gmm
      ? `full-covariance mixture, k = ${c.k}: ARI ${c.ari.toFixed(3)}`
      : 'the actual groups the data was built from');

    readout.innerHTML =
      `<table class="mix-table">
         <tr><td>k-means (last article)</td><td><span class="value">${c.km_ari.toFixed(3)}</span></td>
             <td>mixture, full covariance</td><td><span class="value">${c.ari.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${BLURBS[c.name]}</div>`;
  }

  btnGmm.addEventListener('click', () => { state.view = 'gmm'; render(); });
  btnTrue.addEventListener('click', () => { state.view = 'true'; render(); });
  render();
}
