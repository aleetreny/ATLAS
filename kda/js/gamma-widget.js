/* The kernel dial: seven measured stops from "secretly linear" to "memorised
 * the training set". The boundary field and the projection histogram are both
 * evaluated live in the browser from the shipped dual coefficients; the
 * accuracies are the generator's, and main.js re-derives them as a check.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { kdaScores, histo } from './kdakit.js';

const C0 = 'var(--aqua)';
const C1 = 'var(--cosmos)';

export function initGammaWidget(data) {
  const X = data.x_train;
  const y = data.y_train;

  const wrap = d3.select('#gam-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  const L = makeChart(left, { width: 360, height: 340, margin: { top: 24, right: 14, bottom: 40, left: 42 } });
  const lx = d3.scaleLinear().domain(d3.extent(X, (p) => p[0])).nice().range([0, L.w]);
  const ly = d3.scaleLinear().domain(d3.extent(X, (p) => p[1])).nice().range([L.h, 0]);
  drawGrid(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 4 });
  drawAxes(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 4 });
  const fieldG = L.g.append('g');
  const dotL = L.g.append('g');

  const R = makeChart(right, { width: 360, height: 340, margin: { top: 24, right: 14, bottom: 40, left: 20 } });
  const histG = R.g.append('g');
  const rTitle = R.g.append('text').attr('class', 'chart-annotation')
    .attr('x', R.w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '0.58rem');

  const slider = document.querySelector('#gam-slider');
  const readout = document.querySelector('#gam-readout');

  function render() {
    const row = data.kda[+slider.value];
    const gamma = row.gamma;
    document.querySelector('#gam-label').textContent = gamma;

    /* boundary field, evaluated live */
    const NG = 55;
    const xs = d3.range(NG).map((i) => lx.domain()[0] + (i + 0.5) / NG * (lx.domain()[1] - lx.domain()[0]));
    const yv = d3.range(NG).map((j) => ly.domain()[0] + (j + 0.5) / NG * (ly.domain()[1] - ly.domain()[0]));
    const pts = [];
    for (const yy of yv) for (const xx of xs) pts.push([xx, yy]);
    const z = kdaScores(pts, X, row.alpha, gamma);
    const cw = L.w / NG;
    const ch = L.h / NG;
    fieldG.selectAll('rect').data(pts).join('rect')
      .attr('x', (p) => lx(p[0]) - cw / 2).attr('y', (p) => ly(p[1]) - ch / 2)
      .attr('width', cw + 0.5).attr('height', ch + 0.5)
      .attr('fill', (p, i) => (z[i] > row.b ? C1 : C0))
      .attr('opacity', 0.14);
    dotL.selectAll('circle').data(X).join('circle')
      .attr('cx', (p) => lx(p[0])).attr('cy', (p) => ly(p[1]))
      .attr('r', 2.8).attr('fill', (p, i) => (y[i] ? C1 : C0))
      .attr('stroke', 'white').attr('stroke-width', 0.5).attr('opacity', 0.85);

    /* projection histogram, live */
    const zt = Array.from(kdaScores(X, X, row.alpha, gamma));
    const lo = Math.min(...zt);
    const hi = Math.max(...zt) + 1e-9;
    const NB = 30;
    const { b0, b1 } = histo(zt, y, lo, hi, NB);
    const hx = d3.scaleLinear().domain([0, NB]).range([0, R.w]);
    const maxC = Math.max(...b0, ...b1, 1);
    const hyy = d3.scaleLinear().domain([0, maxC]).range([R.h, 0]);
    const bw = R.w / NB;
    histG.selectAll('rect.c0').data(b0).join('rect').attr('class', 'c0')
      .attr('x', (v, i) => hx(i) + 1).attr('width', bw - 2)
      .attr('y', (v) => hyy(v)).attr('height', (v) => R.h - hyy(v))
      .attr('fill', C0).attr('opacity', 0.62);
    histG.selectAll('rect.c1').data(b1).join('rect').attr('class', 'c1')
      .attr('x', (v, i) => hx(i) + 1).attr('width', bw - 2)
      .attr('y', (v) => hyy(v)).attr('height', (v) => R.h - hyy(v))
      .attr('fill', C1).attr('opacity', 0.62);
    rTitle.text('the kernel projection of the training points');

    readout.innerHTML =
      `<table class="kda-table">
         <tr><td>train accuracy</td><td><span class="value">${row.acc_train.toFixed(3)}</span></td>
             <td>held-out accuracy</td><td><span class="value">${row.acc_test.toFixed(3)}</span></td>
             <td>linear ceiling</td><td>${data.best_line.acc_test.toFixed(3)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        gamma <= 0.01
          ? `At this width every kernel bump spans the whole dataset, the feature space is effectively ` +
            `flat, and KDA lands on the linear ceiling to the decimal: ${row.acc_test.toFixed(3)} against ` +
            `the dial's ${data.best_line.acc_test.toFixed(3)}. A kernel machine with a huge bandwidth IS ` +
            `the linear machine, which is the safety net under this whole family.`
          : gamma >= 300
            ? `The bumps are now narrower than the gaps between training points: the boundary is a ring ` +
              `around each crimson point, train accuracy is a perfect ${row.acc_train.toFixed(3)}, and the ` +
              `held-out score has collapsed to ${row.acc_test.toFixed(3)}. Article 8's overfitting story, ` +
              `re-enacted by a discriminant.`
            : row.acc_test > 0.98
              ? `The plateau: from 0.1 to 30, two orders of magnitude of gamma, the test score does not ` +
                `move. Kernel discriminants are forgiving in the middle, treacherous only at the ends, ` +
                `and the histogram shows why: two clean islands with open water between them.`
              : `The bend is forming: watch the histogram's overlap drain away as the boundary learns ` +
                `to curve.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
