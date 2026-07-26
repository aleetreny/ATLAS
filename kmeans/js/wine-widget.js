/* The blind test: the wines, clustered without their labels, then revealed.
 *
 * The projection is PCA to two components for DRAWING only; the clustering
 * happened in the full 13 dimensions offline, which the readout says, because
 * "we clustered the picture" and "we clustered the chemistry" are different
 * claims and only the second one is impressive.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { CLUSTER_COLOURS, drawCenters } from './clusterdraw.js';

export function initWineWidget(data) {
  const W = data.wine;
  const { g, w, h } = makeChart('#wine-chart', {
    width: 660,
    height: 460,
    margin: { top: 34, right: 26, bottom: 56, left: 58 },
  });

  const x = d3.scaleLinear().domain(d3.extent(W.proj, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(W.proj, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'first principal component', y: 'second principal component' });

  const dotG = g.append('g');
  const centreG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const readout = document.querySelector('#wine-readout');
  const btns = {
    km: document.querySelector('#wine-km'),
    true: document.querySelector('#wine-true'),
    diff: document.querySelector('#wine-diff'),
  };
  const state = { view: 'km' };

  const disagreements = W.proj.map((_, i) => W.km[i] !== W.true[i]);
  const nDiff = disagreements.filter(Boolean).length;

  function render() {
    const v = state.view;
    dotG.selectAll('circle').data(W.proj).join('circle')
      .attr('cx', (p) => x(p[0]))
      .attr('cy', (p) => y(p[1]))
      .attr('r', (p, i) => (v === 'diff' && disagreements[i] ? 6.5 : 4))
      .attr('fill', (p, i) =>
        v === 'diff'
          ? disagreements[i] ? 'var(--cosmos)' : 'var(--stone)'
          : CLUSTER_COLOURS[(v === 'km' ? W.km[i] : W.true[i]) % CLUSTER_COLOURS.length])
      .attr('stroke', (p, i) => (v === 'diff' && disagreements[i] ? 'var(--squidink)' : 'white'))
      .attr('stroke-width', (p, i) => (v === 'diff' && disagreements[i] ? 1.8 : 0.7))
      .attr('opacity', 0.88);
    drawCenters(centreG, v === 'km' ? W.centers : [], x, y);

    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== v);
    title.text(
      v === 'km' ? 'clusters found blind in 13 dimensions, drawn in 2 for viewing'
        : v === 'true' ? `the cultivars: ${W.classes.join(', ')}`
          : `${nDiff} bottles where cluster and cultivar disagree`
    );

    readout.innerHTML =
      `<table class="km-table">
         <tr><td>clustered on</td><td>all 13 standardised features, labels deleted</td></tr>
         <tr><td>agreement after relabelling</td><td><span class="value">${W.agree}</span> of ${W.n} bottles</td></tr>
         <tr><td>adjusted Rand index</td><td><span class="value">${W.ari.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        v === 'diff'
          ? `Every disagreement sits in the border zone between two clouds, which is exactly where an ` +
            `assignment with no concept of "maybe" has to guess. A model with soft boundaries would hand ` +
            `these six a probability instead: that model is the next article.`
          : `The clustering ran offline in the full 13-dimensional space; this plot is only a projection ` +
            `for human eyes. Flip between the two views: the algorithm was never shown the thing the ` +
            `second view is coloured by.`
      }</div>`;
  }

  for (const [k, b] of Object.entries(btns)) {
    b.addEventListener('click', () => {
      state.view = k;
      render();
    });
  }
  render();
}
