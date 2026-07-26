/* The thermostat: one slider between "mixture" and "k-means".
 *
 * Raising every responsibility to the power 1/T and renormalising is the
 * standard temperature trick: at T = 1 this is exactly the converged EM
 * posterior, and as T falls toward zero every point snaps to its nearest
 * component and the tints become the flat colours of the previous article.
 * The readout counts what freezing destroys: the points whose membership was
 * genuinely shared.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { em, initFrom } from './gmmkit.js';
import { drawSoftPoints, drawMeans, drawEllipses } from './mixdraw.js';

export function initTempWidget(data) {
  const pts = data.blobs.points;
  const { g, w, h } = makeChart('#temp-chart', {
    width: 660,
    height: 440,
    margin: { top: 30, right: 26, bottom: 54, left: 58 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const ellG = g.append('g');
  const dotG = g.append('g');
  const meanG = g.append('g');

  /* The converged mixture, computed once from sklearn's own answer as the
   * start (one E-step to get responsibilities). */
  const model = {
    means: data.blobs.ref.means.map((m) => [...m]),
    covs: data.blobs.ref.covs.map((c) => [...c]),
    weights: [...data.blobs.ref.weights],
  };
  const path = em(pts, model, 50);
  const R = path[path.length - 1].resp;
  drawMeans(meanG, model.means, x, y);
  drawEllipses(ellG, model, x, y);

  const shared = R.filter((r) => Math.max(...r) < 0.9).length;
  const slider = document.querySelector('#temp-slider');
  const readout = document.querySelector('#temp-readout');

  function render() {
    /* slider 0..100 -> T from 1 down to ~0.02 */
    const t = +slider.value / 100;
    const T = Math.pow(0.02, t);          // 1 at left end, 0.02 at right end
    const RT = R.map((r) => {
      const p = r.map((v) => Math.pow(Math.max(v, 1e-12), 1 / T));
      const s = p.reduce((a, b) => a + b, 0);
      return p.map((v) => v / s);
    });
    drawSoftPoints(dotG, pts, RT, x, y);
    const sharedT = RT.filter((r) => Math.max(...r) < 0.9).length;
    document.querySelector('#temp-label').textContent = T.toFixed(2);
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">At T = 1 the mixture reports what it actually ` +
      `believes: <span class="value">${shared}</span> of 300 points hold less than 90% membership in any ` +
      `single cluster, and their blended tints say which neighbours are pulling. At this temperature ` +
      `<span class="value">${sharedT}</span> point${sharedT === 1 ? ' keeps' : 's keep'} that nuance` +
      `${sharedT === 0
        ? ': the freeze is complete, every point belongs totally to one colour, and the picture reads ' +
          'like the previous article again. Hard assignment is not a different philosophy. It is a ' +
          'mixture with the thermostat at zero.'
        : '.'
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
