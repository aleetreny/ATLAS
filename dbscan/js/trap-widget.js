/* The trap: one radius cannot serve two densities, proven by exhaustive
 * sweep, and then HDBSCAN's answer.
 *
 * Left: the data, clustered live at the slider's epsilon (or by HDBSCAN's
 * offline labels in the second view). Right: the whole measured sweep, with
 * the two story curves: "are the tight pair still separate?" as a shaded
 * band, "how much of the diffuse cluster survives in one piece?" as a line.
 * They never succeed together, and the chart is the proof.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { dbscan, ari } from './dbkit.js';
import { drawLabelled } from './densdraw.js';

export function initTrapWidget(data) {
  const ds = data.datasets.varying;
  const pts = ds.points;
  const sweep = data.trap_sweep;
  const best = data.trap_best;
  const hdb = data.hdbscan;

  const wrap = d3.select('#trap-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  const L = makeChart(left, { width: 360, height: 360, margin: { top: 26, right: 14, bottom: 44, left: 44 } });
  const lx = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, L.w]);
  const ly = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([L.h, 0]);
  drawGrid(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 4 });
  drawAxes(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 4 });
  const dotG = L.g.append('g');
  /* two side-by-side panels halve the render scale on a phone: 0.7rem keeps
     the title above the 5px floor, and the short form keeps it inside 360 */
  const lTitle = L.g.append('text').attr('class', 'chart-annotation')
    .attr('x', L.w / 2).attr('y', -10).attr('text-anchor', 'middle').style('font-size', '0.7rem');

  const R = makeChart(right, { width: 360, height: 360, margin: { top: 26, right: 16, bottom: 46, left: 50 } });
  const sx = d3.scaleLinear().domain(d3.extent(sweep, (s) => s.eps)).range([0, R.w]);
  const sy = d3.scaleLinear().domain([0, 1.05]).range([R.h, 0]);
  drawGrid(R.g, sx, sy, R.w, R.h, { xTicks: 5, yTicks: 5 });
  drawAxes(R.g, sx, sy, R.w, R.h, { xTicks: 5, yTicks: 5 });
  axisLabels(R.g, R.w, R.h, { x: 'epsilon', y: 'the two goals' });

  /* Shaded band where the tight pair stays separated. */
  const sepRegions = [];
  let open = null;
  sweep.forEach((s, i) => {
    if (s.separated && open === null) open = s.eps;
    if ((!s.separated || i === sweep.length - 1) && open !== null) {
      sepRegions.push([open, s.separated ? s.eps : sweep[i - 1].eps]);
      open = null;
    }
  });
  for (const [a, b] of sepRegions) {
    R.g.append('rect').attr('x', sx(a)).attr('width', Math.max(sx(b) - sx(a), 2))
      .attr('y', 0).attr('height', R.h)
      .attr('fill', 'var(--primary)').attr('opacity', 0.12);
  }
  R.g.append('text').attr('class', 'chart-annotation')
    .attr('x', sx((sepRegions[0][0] + sepRegions[0][1]) / 2)).attr('y', 14)
    .attr('text-anchor', 'middle').style('font-size', '0.66rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)').text('tight pair still separate');

  R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 3)
    .attr('d', d3.line().x((s) => sx(s.eps)).y((s) => sy(s.recovered))(sweep));
  R.g.append('text').attr('class', 'chart-annotation')
    .attr('x', sx(sweep[sweep.length - 1].eps)).attr('y', sy(sweep[sweep.length - 1].recovered) - 8)
    .attr('text-anchor', 'end').style('font-size', '0.66rem').style('text-transform', 'none')
    .attr('fill', 'var(--cosmos)').text('diffuse cluster held together');

  const marker = R.g.append('line').attr('y1', 0).attr('y2', R.h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');

  const slider = document.querySelector('#trap-eps');
  const readout = document.querySelector('#trap-readout');
  const btnDb = document.querySelector('#trap-db');
  const btnHdb = document.querySelector('#trap-hdb');
  const mcsRow = document.querySelector('#trap-mcs');
  const state = { view: 'db', mcs: '10' };

  const mcsBtns = {};
  for (const mcs of Object.keys(hdb)) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = `min size ${mcs}`;
    b.addEventListener('click', () => { state.view = 'hdb'; state.mcs = mcs; render(); });
    mcsRow.appendChild(b);
    mcsBtns[mcs] = b;
  }

  function render() {
    const eps = +slider.value;
    document.querySelector('#trap-eps-label').textContent = eps.toFixed(2);
    btnDb.classList.toggle('ghost', state.view !== 'db');
    btnHdb.classList.toggle('ghost', state.view !== 'hdb');
    for (const [m, b] of Object.entries(mcsBtns)) {
      b.style.display = state.view === 'hdb' ? '' : 'none';
      b.classList.toggle('ghost', !(state.view === 'hdb' && state.mcs === m));
    }
    marker.attr('x1', sx(eps)).attr('x2', sx(eps)).attr('opacity', state.view === 'db' ? 1 : 0.25);

    if (state.view === 'db') {
      const r = dbscan(pts, eps, 5);
      const a = ari(r.labels, ds.true_labels);
      drawLabelled(dotG, pts, r.labels, r.role, lx, ly, { r: 2.8 });
      lTitle.text(`eps ${eps.toFixed(2)}: ${r.nClusters} cluster${r.nClusters === 1 ? '' : 's'}, ${r.nNoise} noise`);
      const row = sweep.reduce((p, s) => (Math.abs(s.eps - eps) < Math.abs(p.eps - eps) ? s : p));
      readout.innerHTML =
        `<table class="den-table">
           <tr><td>tight pair separate</td><td><span class="value">${row.separated ? 'yes' : 'NO'}</span></td>
               <td>diffuse cluster in one piece</td><td><span class="value">${Math.round(row.recovered * 100)}%</span></td>
               <td>ARI</td><td><span class="value">${a.toFixed(3)}</span></td></tr>
         </table>` +
        `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
          row.separated && row.recovered < 0.6
            ? `The radius protects the tight pair but shreds the diffuse cluster: its points sit farther ` +
              `than ${eps.toFixed(2)} apart, so the chain keeps snapping. Push eps right and watch the ` +
              `shaded band end before the crimson curve gets anywhere.`
            : !row.separated
              ? `The pair has fused: their gap is inside the radius now, while the diffuse cluster ` +
                `${row.recovered >= 0.9 ? 'finally holds together' : 'is still not whole'}. The sweep's best ` +
                `compromise is ARI ${best.ari} at eps ${best.eps}, with the diffuse cluster at ` +
                `${Math.round(best.recovered * 100)}%. There is no good epsilon. That is the theorem this ` +
                `dataset exists to state.`
              : `Both goals partially met: this is as good as one radius gets here.`
        }</div>`;
    } else {
      const hv = hdb[state.mcs];
      drawLabelled(dotG, pts, hv.labels, null, lx, ly, { r: 2.8 });
      lTitle.text(`min size ${state.mcs}: ${hv.n_clusters} cluster${hv.n_clusters === 1 ? '' : 's'}, ${hv.n_noise} noise`);
      readout.innerHTML =
        `<table class="den-table">
           <tr><td>tight pair separate</td><td><span class="value">${hv.separated ? 'yes' : 'NO'}</span></td>
               <td>diffuse cluster in one piece</td><td><span class="value">${Math.round(hv.recovered * 100)}%</span></td>
               <td>ARI</td><td><span class="value">${hv.ari.toFixed(3)}</span></td></tr>
         </table>` +
        `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">All three clusters at once, ` +
        `which no epsilon on the left slider can do, because HDBSCAN never fixes a radius: it asks how ` +
        `long each cluster survives as the radius grows, and keeps the ones that live longest. Each ` +
        `cluster is judged at its own density. The price is visible too: the diffuse cluster's thinnest ` +
        `fringe is pruned as noise, more aggressively as the minimum size rises.</div>`;
    }
  }

  btnDb.addEventListener('click', () => { state.view = 'db'; render(); });
  btnHdb.addEventListener('click', () => { state.view = 'hdb'; render(); });
  slider.addEventListener('input', () => { state.view = 'db'; render(); });
  render();
}
