/* Two features, two models, and the same picture.
 *
 * This widget exists to make an honest concession before the benchmark takes it
 * away: on features that really are close to independent, naive Bayes is not
 * worse than logistic regression at anything. The third view subtracts one
 * probability surface from the other so the size of the disagreement is a
 * measured quantity rather than an impression.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

export function initBoundaryWidget(data) {
  const D = data.two_d;
  const { g, w, h } = makeChart('#bnd-chart', {
    width: 660,
    height: 480,
    margin: { top: 34, right: 26, bottom: 58, left: 66 },
  });

  const x = d3.scaleLinear().domain(d3.extent(D.gx)).range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(D.gy)).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, {
    x: `${D.features[0].replace(/_/g, ' ')}, standardised`,
    y: `${D.features[1].replace(/_/g, ' ')}, standardised`,
  });

  const nx = D.gx.length;
  const ny = D.gy.length;
  const cellW = w / (nx - 1);
  const cellH = h / (ny - 1);

  const heat = g.append('g');
  const boundary = g.append('g');
  const dotsG = g.append('g');
  const tip = tooltip();

  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* d3.contours works on a row-major grid and returns coordinates in grid
   * units, so the path has to be rescaled to the plot and flipped in y, since
   * the grid counts upward and the screen counts downward. */
  const geo = (cells, threshold) => {
    const c = d3.contours().size([nx, ny]).contour(cells, threshold);
    const px = d3.geoIdentity().scale(1).translate([0, 0]);
    void px;
    const path = d3.geoPath(
      d3.geoTransform({
        point(a, b) {
          this.stream.point((a / (nx - 1)) * w, h - (b / (ny - 1)) * h);
        },
      })
    );
    return path(c);
  };

  const state = { mode: 'log', showTest: false };
  const readout = document.querySelector('#bnd-readout');
  const btns = {
    log: document.querySelector('#bnd-log'),
    nb: document.querySelector('#bnd-nb'),
    both: document.querySelector('#bnd-both'),
  };

  function render() {
    const { mode } = state;
    const A = D.logistic;
    const B = D.naive_bayes;
    const diff = A.map((v, i) => v - B[i]);
    const field = mode === 'both' ? diff : mode === 'log' ? A : B;

    const colour = mode === 'both'
      ? d3.scaleDiverging(d3.interpolateRdBu).domain([0.35, 0, -0.35])
      : d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]);

    const cells = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) cells.push({ i, j, v: field[j * nx + i] });
    }
    heat.selectAll('rect').data(cells).join('rect')
      .attr('x', (c) => x(D.gx[c.i]) - cellW / 2)
      .attr('y', (c) => y(D.gy[c.j]) - cellH / 2)
      .attr('width', cellW + 1)
      .attr('height', cellH + 1)
      .attr('fill', (c) => colour(c.v))
      .attr('opacity', 0.62);

    /* Both decision boundaries stay on screen in every view: the point of the
     * disagreement view is where they part company, and it cannot be seen if
     * only one of them is drawn. */
    boundary.selectAll('path').data([
      { d: geo(A, 0.5), c: 'var(--primary)', dash: null },
      { d: geo(B, 0.5), c: 'var(--squidink)', dash: '7 5' },
    ]).join('path')
      .attr('d', (d) => d.d)
      .attr('fill', 'none')
      .attr('stroke', (d) => d.c)
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', (d) => d.dash);

    const pts = state.showTest ? D.train.concat(D.test.map((p) => ({ ...p, held: 1 }))) : D.train;
    dotsG.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p.x))
      .attr('cy', (p) => y(p.y))
      .attr('r', (p) => (p.held ? 5 : 3.4))
      .attr('fill', (p) => (p.held ? 'none' : p.c ? 'var(--cosmos)' : 'var(--aqua)'))
      .attr('stroke', (p) => (p.held ? (p.c ? 'var(--cosmos)' : 'var(--aqua)') : 'white'))
      .attr('stroke-width', (p) => (p.held ? 2 : 0.7))
      .attr('opacity', 0.9)
      .on('mousemove', (event, p) => tip.show(p.c ? 'malignant' : 'benign', event))
      .on('mouseleave', () => tip.hide());

    title.text(
      mode === 'both'
        ? 'red where logistic is more worried, blue where naive Bayes is'
        : mode === 'log'
          ? 'logistic regression: the boundary is a straight line, always'
          : 'naive Bayes: two Gaussians per class make the boundary a conic'
    );
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== mode);

    const maxDiff = d3.max(diff, Math.abs);
    const meanDiff = d3.mean(diff, Math.abs);
    const flips = diff.filter((v, i) => (A[i] >= 0.5) !== (B[i] >= 0.5)).length;
    readout.innerHTML =
      `<table class="pc-table">
         <tr><th></th><th style="color:var(--primary)">logistic</th><th>naive Bayes</th></tr>
         <tr><td>accuracy</td><td>${D.logistic_test.acc.toFixed(4)}</td><td>${D.nb_test.acc.toFixed(4)}</td></tr>
         <tr><td>roc auc</td><td>${D.logistic_test.auc.toFixed(4)}</td><td>${D.nb_test.auc.toFixed(4)}</td></tr>
         <tr><td>brier</td><td>${D.logistic_test.brier.toFixed(5)}</td><td>${D.nb_test.brier.toFixed(5)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">` +
      `These two features correlate at 0.38, which is about as close to the naive assumption as real data gets, ` +
      `and the two models behave almost identically: they disagree about the label on ` +
      `<span class="value">${((flips / diff.length) * 100).toFixed(1)}%</span> of the plane, ` +
      `and their probabilities differ by <span class="value">${meanDiff.toFixed(3)}</span> on average ` +
      `(<span class="value">${maxDiff.toFixed(2)}</span> at worst). Hold on to that. ` +
      `The next section hands them all thirty features.</div>`;
  }

  for (const [k, b] of Object.entries(btns)) {
    b.addEventListener('click', () => {
      state.mode = k;
      render();
    });
  }
  document.querySelector('#bnd-test').addEventListener('change', (e) => {
    state.showTest = e.target.checked;
    render();
  });

  render();
}
