/* The spectrum: all thirteen wine measurements, with one button that does
 * nothing but divide each column by its own standard deviation.
 *
 * The eigendecomposition runs here rather than arriving precomputed, because
 * the point of the button is that it produces a different model, and a widget
 * that swapped between two shipped answers would be asking to be believed. It
 * checks itself against scikit-learn's numbers at load time (see main.js).
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { prepare, covariance, jacobiEigen } from './linalg.js';

/* sklearn's column names are machine-readable rather than legible. */
const SHORT = {
  'od280/od315_of_diluted_wines': 'od280/od315',
  nonflavanoid_phenols: 'nonflavanoid phenols',
  color_intensity: 'colour intensity',
  alcalinity_of_ash: 'alcalinity of ash',
  malic_acid: 'malic acid',
  total_phenols: 'total phenols',
};
const label = (n) => SHORT[n] ?? n.replace(/_/g, ' ');

/* The eigendecomposition of both versions of the matrix, computed once. */
export function decompose(data) {
  const S = data.spectrum;
  const out = {};
  for (const key of ['raw', 'std']) {
    const X = prepare(S.matrix, key === 'std');
    const eig = jacobiEigen(covariance(X));
    const total = eig.values.reduce((s, v) => s + v, 0);
    const evr = eig.values.map((v) => v / total);
    const cum = [];
    evr.reduce((s, v) => {
      cum.push(s + v);
      return s + v;
    }, 0);
    out[key] = { evr, cum, vectors: eig.vectors, values: eig.values, total, X };
  }
  return out;
}

export function initSpectrumWidget(data, computed) {
  const S = data.spectrum;
  const names = S.features;
  const P = names.length;

  const A = makeChart('#spectrum-chart', {
    width: 720, height: 290, margin: { top: 30, right: 26, bottom: 52, left: 62 },
  });
  const B = makeChart('#spectrum-chart', {
    width: 720, height: 340, margin: { top: 34, right: 26, bottom: 40, left: 168 },
  });

  const buttons = document.querySelector('#spectrum-buttons');
  const slider = document.querySelector('#spectrum-k');
  const kLabel = document.querySelector('#spectrum-k-label');
  const readout = document.querySelector('#spectrum-readout');
  const state = { scale: 'std', k: 2 };

  const btns = {};
  for (const [key, text] of [['std', 'standardised'], ['raw', 'original units']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === 'std' ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state.scale = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const xA = d3.scalePoint().domain(d3.range(1, P + 1)).range([26, A.w - 26]);
  const yA = d3.scaleLinear().domain([0, 1]).range([A.h, 0]);
  const barW = Math.min(30, (A.w - 52) / P - 6);
  const barsG = A.g.append('g');
  const cumG = A.g.append('g');
  const titleA = A.g.append('text').attr('class', 'chart-annotation')
    .attr('x', A.w / 2).attr('y', -12).attr('text-anchor', 'middle')
    .style('font-size', '11px').style('text-transform', 'none');

  const yB = d3.scalePoint().domain(names).range([0, B.h]).padding(0.5);
  const loadG = B.g.append('g');
  /* Centred on the SVG, not on the plot area. The loadings panel needs a
     168-unit left margin for the measurement names, so a title centred on the
     drawing area sits 84 units right of centre and runs off the edge. */
  const titleB = B.g.append('text').attr('class', 'chart-annotation')
    .attr('x', B.width / 2 - B.margin.left).attr('y', -16).attr('text-anchor', 'middle')
    .style('font-size', '11px').style('text-transform', 'none');

  function render() {
    const cur = computed[state.scale];
    const k = state.k;

    drawGrid(A.g, xA, yA, A.w, A.h, { xTicks: 0, yTicks: 5 });
    drawAxes(A.g, xA, yA, A.w, A.h, { xTicks: P, yTicks: 5, yFmt: d3.format('.0%') });
    axisLabels(A.g, A.w, A.h, { x: 'component', y: 'share of variance' });

    barsG.selectAll('rect').data(cur.evr).join('rect')
      .attr('x', (d, i) => xA(i + 1) - barW / 2)
      .attr('width', barW)
      .attr('y', (d) => yA(d))
      .attr('height', (d) => A.h - yA(d))
      .attr('fill', (d, i) => (i < k ? 'var(--primary)' : 'var(--stone)'))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6);

    const lineGen = d3.line().x((d, i) => xA(i + 1)).y((d) => yA(d));
    let path = cumG.select('path');
    if (path.empty()) {
      path = cumG.append('path').attr('fill', 'none')
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2)
        .attr('stroke-dasharray', '6 4');
    }
    path.attr('d', lineGen(cur.cum));
    cumG.selectAll('circle').data(cur.cum).join('circle')
      .attr('cx', (d, i) => xA(i + 1)).attr('cy', (d) => yA(d))
      .attr('r', (d, i) => (i === k - 1 ? 5 : 2.6))
      .attr('fill', (d, i) => (i === k - 1 ? 'var(--anchor)' : 'var(--anchor)'))
      .attr('stroke', 'white').attr('stroke-width', (d, i) => (i === k - 1 ? 1.4 : 0.6));

    /* Short enough to fit the 720-unit canvas. A title wider than its own SVG
       is simply cut off at both ends, and the overlap sweep is what noticed. */
    titleA.text(state.scale === 'std'
      ? 'bars: each component. dashed line: the running total'
      : 'the measurements in their own units');

    /* The loadings of the component the slider just added. */
    const v = cur.vectors[k - 1];
    const maxAbs = Math.max(...v.map(Math.abs), 1e-9);
    const xB = d3.scaleLinear().domain([-maxAbs * 1.12, maxAbs * 1.12]).range([0, B.w]);
    drawAxes(B.g, xB, yB, B.w, B.h, { xTicks: 5, yTicks: P });
    B.g.select('g.axis.y').selectAll('text')
      .style('font-size', '12px')
      .text((d) => label(d));

    loadG.selectAll('rect').data(names.map((n, i) => ({ n, v: v[i] }))).join('rect')
      .attr('x', (d) => (d.v >= 0 ? xB(0) : xB(d.v)))
      .attr('width', (d) => Math.abs(xB(d.v) - xB(0)))
      .attr('y', (d) => yB(d.n) - 9)
      .attr('height', 18)
      .attr('fill', (d) => (d.v >= 0 ? 'var(--primary)' : 'var(--anchor)'))
      .attr('opacity', 0.9);
    let zero = loadG.select('line.zero');
    if (zero.empty()) {
      zero = loadG.append('line').attr('class', 'zero')
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
    }
    zero.attr('x1', xB(0)).attr('x2', xB(0)).attr('y1', 0).attr('y2', B.h);

    const top = names[v.map(Math.abs).indexOf(Math.max(...v.map(Math.abs)))];
    titleB.text(`component ${k}: its ${P} weights, ${(cur.evr[k - 1] * 100).toFixed(2)}% `
      + `of the variance, mostly ${label(top)}`);

    kLabel.textContent = String(k);
    for (const [key, b] of Object.entries(btns)) b.classList.toggle('ghost', key !== state.scale);

    /* Readout. Every comparison is written as a comparison, so a rerun that
       moves a number moves the sentence with it. */
    const kept = cur.cum[k - 1];
    const ari = S.kmeans.by_k[k - 1].ari;
    const all = S.kmeans.all_features;
    const f = S.separation.f[k - 1];
    const k90 = cur.cum.findIndex((c) => c >= 0.9) + 1;
    const concentration = Math.max(...cur.vectors[0].map(Math.abs));

    const clusterLine = state.scale === 'std'
      ? (ari > all
        ? `And k-means on just these ${k} coordinate${k === 1 ? '' : 's'} scores ARI `
          + `<span class="bold">${ari.toFixed(4)}</span> against the cultivars, `
          + `<span class="bold">better</span> than the ${all.toFixed(4)} it gets on all thirteen. `
          + `Throwing away ${P - k} of ${P} dimensions did not cost accuracy, it bought some: what `
          + `went out with them was noise.`
        : `k-means on these ${k} coordinate${k === 1 ? '' : 's'} scores ARI ${ari.toFixed(4)}, `
          + `against ${all.toFixed(4)} on all thirteen.`)
      : `Clustering numbers are not quoted for the unstandardised version, because "the first `
        + `component" here is one measurement wearing a hat and the comparison would be with `
        + `itself.`;

    readout.innerHTML = `
      <table class="dir-table">
        <tr><th>keeping</th><th>variance kept</th><th>thrown away</th><th>90% needs</th><th>PC1's biggest weight</th></tr>
        <tr>
          <td>${k} of ${P}</td>
          <td><span class="value">${(kept * 100).toFixed(2)}%</span></td>
          <td>${((1 - kept) * 100).toFixed(2)}%</td>
          <td>${k90} component${k90 === 1 ? '' : 's'}</td>
          <td><span class="value">${concentration.toFixed(4)}</span></td>
        </tr>
      </table>
      <div class="dir-note">
        ${state.scale === 'raw'
    ? `Look at the weight in the last column. A component is a unit vector, so with thirteen `
          + `measurements an even mixture would put about ${(1 / Math.sqrt(P)).toFixed(4)} on each. `
          + `Unstandardised, the first component puts <span class="bold">${concentration.toFixed(4)}</span> `
          + `on ${label(S.raw.top_feature)} and shares the remainder among the other twelve. It is not a `
          + `mixture of anything. Proline is measured in the hundreds and everything else in single `
          + `digits, so proline has more variance than the rest of the dataset combined, and "the `
          + `direction of greatest variance" is the direction of the largest unit.`
    : `Component ${k} carries ${(cur.evr[k - 1] * 100).toFixed(2)}% of the variance and separates the `
          + `three cultivars with an F of ${f.toFixed(2)}. ${clusterLine}`}
      </div>`;
  }

  slider.addEventListener('input', () => {
    state.k = +slider.value;
    render();
  });
  render();
}
