/* The kernel trick as a morph.
 *
 * At lift 0 the chart is the rings in their own plane, where no straight line
 * works. At lift 1 the vertical axis has become the squared radius, the classes
 * are two horizontal bands, and a flat cut separates them. Everything in between
 * is the same points on their way up, which is the only part of the kernel trick
 * that is genuinely hard to picture from a formula.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initLiftWidget(data) {
  const R = data.rings;
  const pts = R.points;

  const { g, w, h } = makeChart('#lift-chart', {
    width: 560,
    height: 470,
    margin: { top: 32, right: 24, bottom: 56, left: 62 },
  });

  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p.x)).nice().range([0, w]);
  const yFlat = d3.extent(pts, (p) => p.y);
  const yLift = d3.extent(pts, (p) => p.r2);
  const y = d3.scaleLinear().range([h, 0]);

  const clip = 'lift-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -1).attr('y', -1)
    .attr('width', w + 2).attr('height', h + 2);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  const cut = plot.append('line').attr('x1', 0).attr('x2', w)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('opacity', 0);
  const cutLabel = plot.append('text').attr('class', 'chart-annotation')
    .attr('x', 8).style('font-size', '0.62rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)').attr('opacity', 0).text('one flat cut, and it is done');
  /* The same boundary, drawn in the original plane. The section claims the
   * circle and the line are one object seen from two angles, and that claim is
   * worth showing rather than asserting: this fades out as the straight cut
   * fades in, so both ends of the slider have the boundary on screen. */
  const ring = plot.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 3).attr('opacity', 0);
  const ringLabel = plot.append('text').attr('class', 'chart-annotation')
    .style('font-size', '0.62rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)').attr('text-anchor', 'middle').attr('opacity', 0)
    .text('the same boundary, back in the plane');
  const dotsG = plot.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* The threshold that separates the two rings in squared radius: midway
   * between the largest inner radius and the smallest outer one, measured
   * rather than guessed, so the label under the cut is honest even if the
   * noise level of the generated rings changes. */
  const inner = pts.filter((p) => p.c === 1).map((p) => p.r2);
  const outer = pts.filter((p) => p.c === 0).map((p) => p.r2);
  const CUT = (d3.quantile(inner.sort(d3.ascending), 0.99) + d3.quantile(outer.sort(d3.ascending), 0.01)) / 2;
  const wrong = pts.filter((p) => (p.r2 > CUT ? 0 : 1) !== p.c).length;

  const slider = document.querySelector('#lift-slider');
  const btn = document.querySelector('#lift-plane');
  const statsEl = document.querySelector('#lift-stats');
  const state = { showCut: false };

  function render() {
    const t = +slider.value;
    document.querySelector('#lift-label').textContent = t.toFixed(2);

    const lo = yFlat[0] * (1 - t) + yLift[0] * t;
    const hi = yFlat[1] * (1 - t) + yLift[1] * t;
    const pad = (hi - lo) * 0.08;
    y.domain([lo - pad, hi + pad]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, {
      x: 'x₁',
      y: t < 0.02 ? 'x₂' : t > 0.98 ? 'x₁² + x₂²' : `x₂ lifting towards x₁² + x₂²`,
    });

    dotsG.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p.x))
      .attr('cy', (p) => y(p.y * (1 - t) + p.r2 * t))
      .attr('r', 4)
      .attr('fill', (p) => (p.c ? 'var(--cosmos)' : 'var(--aqua)'))
      .attr('stroke', 'white').attr('stroke-width', 0.8)
      .attr('opacity', 0.88);

    /* The cut only means anything once the vertical axis is mostly the squared
     * radius, so it fades in with the lift rather than sitting there being
     * wrong at the bottom of the slider. */
    const vis = state.showCut ? Math.max(0, (t - 0.55) / 0.35) : 0;
    const cy = y(yFlat[0] * 0 + CUT * 1 * t + (1 - t) * 0);
    cut.attr('y1', y(CUT * t + (1 - t) * d3.mean(pts, (p) => p.y)))
      .attr('y2', y(CUT * t + (1 - t) * d3.mean(pts, (p) => p.y)))
      .attr('opacity', Math.min(1, vis));
    cutLabel.attr('y', y(CUT * t + (1 - t) * d3.mean(pts, (p) => p.y)) - 8).attr('opacity', Math.min(1, vis));
    void cy;

    const R0 = Math.sqrt(CUT);
    const circ = d3.range(0, 361, 3).map((deg) => {
      const a = (deg * Math.PI) / 180;
      return [x(R0 * Math.cos(a)), y(R0 * Math.sin(a))];
    });
    const ringVis = state.showCut ? Math.max(0, 1 - t / 0.3) : 0;
    ring.attr('d', `M${circ.map((p) => p.join(',')).join('L')}Z`).attr('opacity', ringVis);
    ringLabel.attr('x', x(0)).attr('y', y(R0) - 17).attr('opacity', ringVis);

    /* The best horizontal cut AT THE CURRENT LIFT, so the panel answers the
     * slider instead of sitting there. This number climbing from a coin flip to
     * perfect as the points rise is the demonstration, and it is measured on
     * every frame rather than quoted at the two ends.
     *
     * Both polarities are scored as the sweep runs. Taking n - best at the end
     * instead reads the complement of the BEST threshold rather than the best
     * complement, which is a different and usually wrong number. */
    const coord = pts.map((p) => ({ v: p.y * (1 - t) + p.r2 * t, c: p.c }))
      .sort((a, b) => a.v - b.v);
    const totalInner = coord.filter((p) => p.c === 1).length;
    let below = 0;
    let bestRight = 0;
    let worstRight = coord.length;
    for (let i = 0; i <= coord.length; i++) {
      // everything below the cut called class 1, everything above called class 0
      const right = below + (coord.length - i - (totalInner - below));
      if (right > bestRight) bestRight = right;
      if (right < worstRight) worstRight = right;
      if (i < coord.length && coord[i].c === 1) below++;
    }
    const bestAcc = Math.max(bestRight, coord.length - worstRight) / coord.length;

    btn.textContent = state.showCut ? 'Hide the boundary' : 'Show the boundary';
    title.text(
      t < 0.05
        ? `two rings: no flat answer gets past ${bestAcc.toFixed(2)}`
        : t > 0.95
          ? 'the same points, one coordinate added: two flat bands'
          : 'lifting'
    );

    statsEl.innerHTML =
      `<table class="gc-table">
         <tr><td>linear SVM, in the plane</td><td><span class="value">${R.linear_acc.toFixed(3)}</span></td></tr>
         <tr><td>RBF SVM, in the plane</td><td><span class="value">${R.rbf_acc.toFixed(3)}</span></td></tr>
         <tr><td>best flat cut at this lift</td><td><span class="value">${bestAcc.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.7rem;line-height:1.45">${
        t < 0.05
          ? `In the plane, the best horizontal line anyone can draw gets ` +
            `<span class="value">${bestAcc.toFixed(3)}</span> of the points right. There is no flat answer here.`
          : t > 0.95
            ? `A single threshold on x₁² + x₂² gets <span class="value">${pts.length - wrong}</span> ` +
              `of ${pts.length} right. Drawn back in the original plane that same boundary is a circle. ` +
              `Nothing about the data changed: the only thing that moved was the space it was measured in.`
            : `Halfway up, the best flat cut is already at <span class="value">${bestAcc.toFixed(3)}</span>. ` +
              `The separation is not appearing at the top, it is arriving continuously as the coordinate changes.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  /* The button now shows the boundary at whichever end of the slider you are
   * standing at: the circle down in the plane, the straight cut up in the
   * lifted space. Jumping the slider would have hidden the very thing the
   * section claims, that these are one object. */
  btn.addEventListener('click', () => {
    state.showCut = !state.showCut;
    render();
  });
  render();
}
