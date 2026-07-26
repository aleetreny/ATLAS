/* The covariance wardrobe: four constraints on the same sheared data.
 *
 * All four fits were measured offline with scikit-learn; the widget draws
 * each one's ellipses and lets the reader watch the ARI move as the model is
 * allowed to learn shape. The subtle star is "tied": the three blobs share
 * one shear, tied spends 6 fewer parameters saying exactly that, and BIC
 * rewards it over full.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { COLOURS, drawMeans, drawEllipses } from './mixdraw.js';

const STORY = {
  spherical: 'Circles only: every cluster forced round and even. This is k-means with extra arithmetic, ' +
    'and it scores like it: the sheared cigars get carved across the grain again.',
  diag: 'Ellipses allowed, but only axis-aligned. The cigars here are tilted, so the freedom is spent ' +
    'in the wrong directions and buys nothing.',
  tied: 'Any tilt, but one shared shape for all clusters. These three blobs really were sheared by the ' +
    'same matrix, so this is the true model of the data: perfect recovery, and the cheapest of the ' +
    'shape-aware options.',
  full: 'Every cluster its own free ellipse. Also perfect here, at the price of six more parameters ' +
    'than tied, and BIC notices the waste.',
};

export function initShapeWidget(data) {
  const S = data.shapes;
  const pts = S.points;
  const { g, w, h } = makeChart('#shape-chart', {
    width: 640,
    height: 460,
    margin: { top: 30, right: 26, bottom: 50, left: 54 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });

  const dotG = g.append('g');
  const ellG = g.append('g');
  const meanG = g.append('g');

  const btnRow = document.querySelector('#shape-buttons');
  const readout = document.querySelector('#shape-readout');
  const state = { i: 0 };

  const btns = S.variants.map((v, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = v.type;
    b.addEventListener('click', () => { state.i = i; render(); });
    btnRow.appendChild(b);
    return b;
  });

  function render() {
    const v = S.variants[state.i];
    dotG.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 3)
      .attr('fill', (p, i) => COLOURS[v.labels[i] % COLOURS.length])
      .attr('stroke', 'white').attr('stroke-width', 0.5).attr('opacity', 0.8);
    drawEllipses(ellG, v, x, y);
    drawMeans(meanG, v.means, x, y);
    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));

    const best = S.variants.reduce((a, b2) => (b2.bic < a.bic ? b2 : a));
    readout.innerHTML =
      `<table class="mix-table">
         <tr><th>constraint</th><th>ARI</th><th>parameters</th><th>BIC</th></tr>
         ${S.variants.map((v2) =>
           `<tr${v2.type === v.type ? ' style="font-weight:700"' : ''}>` +
           `<td>${v2.type}</td><td>${v2.ari.toFixed(3)}</td><td>${v2.n_params}</td>` +
           `<td>${v2.bic.toLocaleString('en-US')}${v2.type === best.type ? ' &#9666; BIC&#39;s pick' : ''}</td></tr>`).join('')}
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${STORY[v.type]} ` +
      `(k-means scored ARI 0.529 on this data.)</div>`;
  }

  render();
}
