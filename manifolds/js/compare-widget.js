/* Four methods, the same five hundred points, one score they can all be given.
 *
 * PCA, Isomap and locally linear embedding produce coordinates, so they can be
 * rotated onto the true chart and what is left over measured. A self-organising
 * map does not: it produces a grid of prototypes and a cell number per point,
 * so what gets compared is where on the grid each point landed. The grid is
 * trained here, in the browser, from the same starting prototypes and the same
 * sample order the generator drew, which is why the button says train and not
 * replay.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { procrustes, pairwise, trustworthiness, corr } from '../../assets/js/embed.js';
import { isomapAt, lleAt, pca2, somTrain, somScores } from './roll.js';

export function computeMethods(data, state, cache) {
  const out = {};
  out.pca = { coords: pca2(state.points) };
  const kIso = data.meta.k_show;
  out.isomap = { coords: (cache[kIso] || (cache[kIso] = isomapAt(state, kIso))).coords, k: kIso };
  const kLle = data.lle.k;
  out.lle = { coords: lleAt(state, kLle).coords, k: kLle };
  const cfg = {
    rows: data.som.rows,
    cols: data.som.cols,
    epochs: data.som.epochs,
    orders: data.som.orders,
    initIndex: data.som.init_index,
    sigma0: data.som.sigma0,
    lr0: data.som.lr0,
  };
  const som = somTrain(state.scaled, cfg);
  const sc = somScores(state.scaled, som);
  out.som = { coords: sc.coords, som, scores: sc, cfg };
  const Dscaled = pairwise(state.scaled);
  Object.entries(out).forEach(([key, m]) => {
    m.aligned = procrustes(state.chart, m.coords);
    m.disparity = m.aligned.disparity;
    const Dz = pairwise(m.coords);
    m.trust = trustworthiness(key === 'som' ? Dscaled : state.D, Dz, 12);
    const a = [];
    const b = [];
    for (let i = 0; i < Dz.length; i++) {
      for (let j = i + 1; j < Dz.length; j++) {
        a.push(Dz[i][j]);
        b.push(state.GEO[i][j]);
      }
    }
    m.geoCorr = corr(a, b);
  });
  return out;
}

const NAMES = {
  pca: 'PCA', isomap: 'Isomap', lle: 'locally linear embedding', som: 'self-organising map',
};
/* The table has four columns of figures and nowrap cells, so the long names
   push the last one off the edge of the readout. Short ones in the table, full
   ones on the buttons and in the prose. */
const SHORT = { pca: 'PCA', isomap: 'Isomap', lle: 'LLE', som: 'SOM' };

export function initCompareWidget(data, state, methods) {
  const chart = makeChart('#compare-chart', {
    width: 780, height: 330, margin: { top: 84, right: 22, bottom: 30, left: 40 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#compare-buttons');
  const readout = document.querySelector('#compare-readout');
  const st = { key: 'isomap' };

  const sVals = state.chart.map((c) => c[0]);
  const ramp = d3.scaleSequential(d3.interpolateTurbo)
    .domain([Math.min(...sVals), Math.max(...sVals)]);

  const btns = {};
  ['pca', 'isomap', 'lle', 'som'].forEach((key) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.key ? '' : ' ghost');
    b.textContent = NAMES[key];
    b.addEventListener('click', () => {
      st.key = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  });

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -66).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const legend = g.append('g');

  function render() {
    const M = methods[st.key];
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.key);
    body.selectAll('*').remove();
    legend.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    const Z = M.aligned.coords;
    const ref = M.aligned.reference;
    /* No axes here on purpose. Procrustes has scaled both clouds to the same
       total scatter, so the coordinates are in arbitrary units and printing
       "0.05" invites the reader to read something into a number that carries
       nothing. The rings are the reference frame. */
    const { x, y } = equalScales(ref.concat(Z), w, h, { pad: 1.1 });
    body.selectAll('circle.t').data(ref).join('circle').attr('class', 't')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 5)
      .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 1);
    body.selectAll('circle.z').data(Z).join('circle').attr('class', 'z')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
      .attr('fill', (d, i) => ramp(state.chart[i][0])).attr('opacity', 0.9);
    [['var(--stone)', 'where the points really are, unrolled', 'ring'],
      ['var(--primary)', 'where this method put them, tinted by the truth', 'dot']]
      .forEach(([c, t, kind], i) => {
        if (kind === 'ring') {
          legend.append('circle').attr('cx', 9 + i * 320).attr('cy', -24).attr('r', 5)
            .attr('fill', 'none').attr('stroke', c).attr('stroke-width', 1.4);
        } else {
          legend.append('circle').attr('cx', 9 + i * 320).attr('cy', -24).attr('r', 3.4)
            .attr('fill', c);
        }
        legend.append('text').attr('x', 20 + i * 320).attr('y', -20)
          .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
          .attr('fill', 'var(--squidink)').text(t);
      });
    wrapLabel(title, st.key === 'som'
      ? `${NAMES[st.key]}: each point drawn at the grid cell it fell into, `
        + `${data.som.rows} by ${data.som.cols}, rescaled onto the true chart`
      : `${NAMES[st.key]}, rotated and rescaled onto the true chart`, w - 8);

    const iso = methods.isomap;
    const rows = ['pca', 'isomap', 'lle', 'som'].map((k) => {
      const m = methods[k];
      const mark = k === st.key ? ' style="font-weight:700"' : '';
      return `<tr${mark}><td>${SHORT[k]}</td>`
        + `<td>${m.disparity.toFixed(5)}</td>`
        + `<td>${m.trust.toFixed(5)}</td>`
        + `<td>${m.geoCorr.toFixed(5)}</td></tr>`;
    }).join('');
    const ratio = M.disparity / Math.max(iso.disparity, 1e-12);
    readout.innerHTML = `
      <table class="man-table">
        <tr><th>method</th><th>distance from the truth</th><th>trustworthiness</th><th>geodesic agreement</th></tr>
        ${rows}
      </table>
      <div class="man-note">
        ${st.key === 'isomap'
    ? `Isomap is the only one of the four that puts the sheet back: ${M.disparity.toFixed(5)} away `
      + `from the true chart, with ${M.geoCorr.toFixed(5)} agreement between its distances and the `
      + `real ones along the sheet.`
    : `${NAMES[st.key]} lands <span class="bold">${ratio.toFixed(0)} times</span> further from the `
      + `true chart than Isomap does (${M.disparity.toFixed(5)} against `
      + `${iso.disparity.toFixed(5)}). ${M.trust > 0.85
    ? `Its trustworthiness is ${M.trust.toFixed(5)}, which is high: it keeps neighbours together `
        + `perfectly well. Keeping neighbours together and recovering the sheet are two different `
        + `jobs, and this is what it looks like when a method does the first and not the second.`
    : `And its trustworthiness, ${M.trust.toFixed(5)}, says it is not keeping neighbours together `
        + `either.`}`}
        ${st.key === 'som'
    ? ` Its own two diagnostics are ${methods.som.scores.quantisation.toFixed(4)} of quantisation `
      + `error and ${methods.som.scores.topographic.toFixed(4)} of topographic error, both of which `
      + `say the fit went beautifully.`
    : ''}
      </div>`;
  }

  render();
}
