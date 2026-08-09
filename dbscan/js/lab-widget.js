/* The two-dial laboratory: every dataset of the module, live DBSCAN, and
 * nothing precomputed. Drag eps and minPts and read what happens to the
 * cluster count, the noise count and the agreement with the truth. The
 * verdict lines know each dataset's measured story and say where the reader
 * currently is in it.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { dbscan, ari } from './dbkit.js';
import { drawLabelled } from './densdraw.js';

const CASES = [
  {
    key: 'moons', label: 'moons', epsMax: 1.2, eps0: 0.45, minPts0: 5,
    story: (r, a) => (r.nClusters === 2 && a > 0.9
      ? `Two moons, found by chaining: no centre, no shape model, ARI ${a.toFixed(3)}. This is the picture k-means (0.275) and the mixture (0.434) could not draw.`
      : r.nClusters === 1
        ? `One cluster: the radius is long enough to bridge the gap between the moons, and chaining happily walks across it. The bridge is DBSCAN's characteristic failure.`
        : `Fragments and noise: the radius is too short for the moons' own thickness, so each moon shatters.`),
  },
  {
    key: 'rings', label: 'rings', epsMax: 1.6, eps0: 0.65, minPts0: 5,
    story: (r, a) => (r.nClusters === 2 && a > 0.99
      ? `ARI ${a.toFixed(3)}: the two-article curse is lifted. K-means scored -0.003 here, the mixture 0.002, and density gets it exactly right, because each ring is dense inside and empty around, which is all DBSCAN asks.`
      : r.nClusters > 4
        ? `${r.nClusters} fragments: below roughly 0.6 the radius is shorter than the sparse outer ring's own gaps, so the chain keeps breaking. The rings are only connected at the right scale.`
        : 'Watch for the sweet range: wide enough to chain each ring, narrower than the 2-unit moat between them.'),
  },
  {
    key: 'blobs_noise', label: 'blobs + impostors', epsMax: 1.6, eps0: 0.8, minPts0: 8,
    story: (r, a, extra) => `${extra.caught} of the 30 uniform impostors are called noise` +
      (r.nClusters === 3
        ? `, with the three blobs intact (ARI ${a.toFixed(3)}). No other model in this module can say "this point belongs to nothing": k-means dealt all 30 impostors into the nearest cluster (ARI 0.832).`
        : `. Tune until exactly three clusters survive: the blobs are std 0.85-1.05, so the radius has to live between a blob's internal spacing and the impostors' isolation.`),
  },
  {
    key: 'varying', label: 'the trap', epsMax: 1.4, eps0: 0.44, minPts0: 5,
    story: (r, a) => `ARI ${a.toFixed(3)}. This dataset defeats every single radius, and the next section proves it with a sweep.`,
  },
];

export function initLabWidget(data) {
  const { g, w, h } = makeChart('#lab-chart', {
    width: 640,
    height: 470,
    margin: { top: 34, right: 26, bottom: 50, left: 54 },
  });
  const dotG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const btnRow = document.querySelector('#lab-buttons');
  const readout = document.querySelector('#lab-readout');
  const epsSlider = document.querySelector('#lab-eps');
  const mpSlider = document.querySelector('#lab-minpts');
  const state = { i: 0 };

  const btns = CASES.map((c, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = c.label;
    b.addEventListener('click', () => {
      state.i = i;
      epsSlider.max = c.epsMax;
      epsSlider.value = c.eps0;
      mpSlider.value = c.minPts0;
      render();
    });
    btnRow.appendChild(b);
    return b;
  });

  function render() {
    const c = CASES[state.i];
    const ds = data.datasets[c.key];
    const pts = ds.points;
    const eps = +epsSlider.value;
    const minPts = +mpSlider.value;
    const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });

    const r = dbscan(pts, eps, minPts);
    const a = ari(r.labels, ds.true_labels);
    drawLabelled(dotG, pts, r.labels, r.role, x, y);

    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    document.querySelector('#lab-eps-label').textContent = eps.toFixed(2);
    document.querySelector('#lab-mp-label').textContent = minPts;
    title.text(`${r.nClusters} cluster${r.nClusters === 1 ? '' : 's'}, ${r.nNoise} noise point${r.nNoise === 1 ? '' : 's'}`);

    const extra = {};
    if (c.key === 'blobs_noise') {
      extra.caught = r.labels.filter((l, i) => l === -1 && ds.true_labels[i] === -1).length;
    }
    readout.innerHTML =
      `<table class="den-table">
         <tr><td>clusters found</td><td><span class="value">${r.nClusters}</span></td>
             <td>noise</td><td><span class="value">${r.nNoise}</span></td>
             <td>ARI vs truth</td><td><span class="value">${a.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${c.story(r, a, extra)}</div>`;
  }

  epsSlider.addEventListener('input', render);
  mpSlider.addEventListener('input', render);
  epsSlider.max = CASES[0].epsMax;
  epsSlider.value = CASES[0].eps0;
  render();
}
