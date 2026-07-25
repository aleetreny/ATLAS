/* PCA, PLS and LDA on the same 178 wines: three answers to "which two axes?".
 *
 * All three projections are standardised per axis before plotting, so the
 * panels are directly comparable and no method wins on units alone. The
 * readout leads with the cross-validated accuracy of a logistic model on the
 * two coordinates, which is scale-free, and quotes Fisher's ratio second with
 * the caveat that it is LDA's own objective.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

const CLASS_COLOURS = ['var(--anchor)', 'var(--primary)', 'var(--cosmos)'];
const VIEWS = {
  pca: { btn: '#ax-pca', title: 'PCA: the directions of most variance, label never consulted' },
  pls: { btn: '#ax-pls', title: 'PLS: most covariance with the target, and the target here is a class index' },
  lda: { btn: '#ax-lda', title: 'LDA: the directions that push the class means apart' },
};

export function initAxesWidget(data) {
  const names = data.wine_classes;
  const { g, w, h } = makeChart('#axes-chart', {
    width: 660,
    height: 440,
    margin: { top: 34, right: 26, bottom: 56, left: 60 },
  });

  const x = d3.scaleLinear().domain([-3.2, 3.2]).range([0, w]);
  const y = d3.scaleLinear().domain([-3.2, 3.2]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'axis 1, standardised', y: 'axis 2, standardised' });

  const hullG = g.append('g');
  const dotsG = g.append('g');
  const tip = tooltip();
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.64rem');

  const legend = g.append('g').attr('transform', `translate(${w - 116},6)`);
  names.forEach((n, i) => {
    const row = legend.append('g').attr('transform', `translate(0,${i * 17})`);
    row.append('circle').attr('r', 5).attr('fill', CLASS_COLOURS[i]).attr('stroke', 'white').attr('stroke-width', 1);
    row.append('text').attr('x', 11).attr('y', 4).attr('class', 'chart-annotation')
      .style('font-size', '0.6rem').style('text-transform', 'none')
      .attr('fill', CLASS_COLOURS[i]).text(n);
  });

  const readout = document.querySelector('#axes-readout');
  const state = { view: 'pca' };

  function render() {
    const v = data.axes[state.view];
    const pts = v.points;

    dotsG.selectAll('circle').data(pts).join('circle')
      .attr('r', 4.6)
      .attr('fill', (p) => CLASS_COLOURS[p.c])
      .attr('stroke', 'white').attr('stroke-width', 1.1)
      .attr('opacity', 0.88)
      .on('mousemove', (event, p) => tip.show(names[p.c], event))
      .on('mouseleave', () => tip.hide())
      .transition('move').duration(650)
      .attr('cx', (p) => x(p.x))
      .attr('cy', (p) => y(p.y));

    /* The convex hull of each class makes overlap visible without asking the
     * reader to trace scattered dots by eye. */
    const hulls = [0, 1, 2].map((c) => {
      const P = pts.filter((p) => p.c === c).map((p) => [x(p.x), y(p.y)]);
      return { c, hull: d3.polygonHull(P) };
    }).filter((d) => d.hull);
    hullG.selectAll('path').data(hulls, (d) => d.c).join('path')
      .attr('fill', (d) => CLASS_COLOURS[d.c])
      .attr('stroke', (d) => CLASS_COLOURS[d.c])
      .attr('stroke-width', 1.4)
      .attr('opacity', 0.1)
      .transition('move').duration(650)
      .attr('d', (d) => `M${d.hull.join('L')}Z`);

    title.text(VIEWS[state.view].title);
    for (const [k, cfg] of Object.entries(VIEWS)) {
      document.querySelector(cfg.btn).classList.toggle('ghost', k !== state.view);
    }

    const best = Object.entries(data.axes).reduce((a, b) => (b[1].cv_acc > a[1].cv_acc ? b : a));
    readout.innerHTML =
      `<table class="gc-table">
         <tr><th></th>${Object.keys(data.axes).map((k) => `<th>${k.toUpperCase()}</th>`).join('')}</tr>
         <tr><td>logistic on these 2 axes, 5-fold</td>` +
      Object.values(data.axes).map((a) => `<td>${a.cv_acc.toFixed(4)}</td>`).join('') + `</tr>
         <tr><td>Fisher separation</td>` +
      Object.values(data.axes).map((a) => `<td>${a.separation.toFixed(2)}</td>`).join('') + `</tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">` +
      `<span class="bold">${best[0].toUpperCase()}</span> wins on both, and on the second row it should: ` +
      `Fisher separation is the ratio LDA is defined to maximise, so quoting it is naming the objective ` +
      `rather than discovering something. The first row is the fair one, and it agrees. ` +
      `Thirteen dimensions down to two with nothing lost, because the two that were kept were chosen ` +
      `by asking about the labels.</div>`;
  }

  for (const [k, cfg] of Object.entries(VIEWS)) {
    document.querySelector(cfg.btn).addEventListener('click', () => {
      state.view = k;
      render();
    });
  }
  render();
}
