/* How many Gaussians: the answer inertia could never give.
 *
 * Left panel: the mixture fitted at the slider's k, ellipses and all,
 * measured offline. Right panel: BIC and AIC across k, and unlike the
 * inertia curve of the previous article, BIC has a genuine minimum, because
 * the likelihood's reward for fitting is charged for every parameter spent.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { COLOURS, drawMeans, drawEllipses } from './mixdraw.js';

export function initBicWidget(data) {
  const pts = data.blobs.points;
  const rows = data.em_k;

  const wrap = d3.select('#bic-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  const L = makeChart(left, { width: 360, height: 360, margin: { top: 26, right: 14, bottom: 44, left: 44 } });
  const xs = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, L.w]);
  const ys = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([L.h, 0]);
  drawGrid(L.g, xs, ys, L.w, L.h, { xTicks: 4, yTicks: 4 });
  drawAxes(L.g, xs, ys, L.w, L.h, { xTicks: 4, yTicks: 4 });
  const dotG = L.g.append('g');
  const ellG = L.g.append('g');
  const meanG = L.g.append('g');

  const R = makeChart(right, { width: 360, height: 360, margin: { top: 26, right: 20, bottom: 44, left: 62 } });
  const kx = d3.scalePoint().domain(rows.map((r) => r.k)).range([0, R.w]).padding(0.4);
  const all = rows.flatMap((r) => [r.bic, r.aic]);
  const cy = d3.scaleLinear().domain([d3.min(all) * 0.99, d3.max(all) * 1.01]).range([R.h, 0]);
  drawAxes(R.g, kx, cy, R.w, R.h, { yTicks: 5, yFmt: d3.format('~s') });
  axisLabels(R.g, R.w, R.h, { x: 'number of components k' });
  const lb = d3.line().x((r) => kx(r.k)).y((r) => cy(r.bic));
  const la = d3.line().x((r) => kx(r.k)).y((r) => cy(r.aic));
  R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('d', lb(rows));
  R.g.append('path').attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 3).attr('d', la(rows));
  R.g.append('text').attr('class', 'chart-annotation').attr('x', kx(rows[rows.length - 1].k))
    .attr('y', cy(rows[rows.length - 1].bic) - 10).attr('text-anchor', 'end')
    .style('font-size', '0.56rem').style('text-transform', 'none').attr('fill', 'var(--primary)').text('BIC');
  R.g.append('text').attr('class', 'chart-annotation').attr('x', kx(rows[rows.length - 1].k))
    .attr('y', cy(rows[rows.length - 1].aic) + 16).attr('text-anchor', 'end')
    .style('font-size', '0.56rem').style('text-transform', 'none').attr('fill', 'var(--squidink)').attr('opacity', 0.6).text('AIC');
  const marker = R.g.append('line').attr('y1', 0).attr('y2', R.h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
  const dotB = R.g.append('circle').attr('r', 4.5).attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.4);

  const slider = document.querySelector('#bic-slider');
  const readout = document.querySelector('#bic-readout');
  const best = rows.reduce((a, b) => (b.bic < a.bic ? b : a));

  function render() {
    const k = +slider.value;
    const row = rows.find((r) => r.k === k);
    document.querySelector('#bic-label').textContent = k;
    dotG.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => xs(p[0])).attr('cy', (p) => ys(p[1]))
      .attr('r', 2.8)
      .attr('fill', (p, i) => COLOURS[row.labels[i] % COLOURS.length])
      .attr('stroke', 'white').attr('stroke-width', 0.5).attr('opacity', 0.8);
    drawEllipses(ellG, row, xs, ys, { sigmas: [2] });
    drawMeans(meanG, row.means, xs, ys);
    marker.attr('x1', kx(k)).attr('x2', kx(k));
    dotB.attr('cx', kx(k)).attr('cy', cy(row.bic));

    readout.innerHTML =
      `<table class="mix-table">
         <tr><td>BIC at k = ${k}</td><td><span class="value">${row.bic.toLocaleString('en-US')}</span></td>
             <td>minimum</td><td><span class="value">${best.bic.toLocaleString('en-US')}</span> at k = ${best.k}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        k === best.k
          ? `The bottom of the valley. Fewer components leave likelihood on the table; more buy their ` +
            `extra likelihood at full parameter price and BIC refuses the deal. Compare this with the ` +
            `elbow squint of the last article: a curve with a minimum needs no squinting.`
          : k < best.k
            ? `Under-fitted: adding a component would buy more likelihood than its parameters cost, so ` +
              `the curve is still falling.`
            : `Past the valley: the ${k} ellipses on the left are slicing real clusters to shave residual ` +
              `likelihood, and the parameter bill now exceeds the win. Note the picture still looks ` +
              `plausible. The criterion knows; the eye alone would negotiate.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
