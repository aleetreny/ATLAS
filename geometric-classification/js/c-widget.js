/* The C knob, across five orders of magnitude.
 *
 * The checkbox is the argument the section is making, so it is a control rather
 * than a sentence: delete every point that is not a support vector and the
 * boundary is unchanged, because the model never depended on them.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { lineInBox, streetPolygon, margin } from './marginkit.js';

export function initCWidget(data) {
  const M = data.margin;
  const pts = M.points;
  const runs = M.runs;

  const { g, w, h } = makeChart('#c-chart', {
    width: 560,
    height: 470,
    /* 56 on the left clipped the rotated y label, which axisLabels parks at
     * -44 and which needs its own descenders on top of that. */
    margin: { top: 32, right: 24, bottom: 54, left: 64 },
  });

  const xe = d3.extent(pts, (p) => p.x);
  const ye = d3.extent(pts, (p) => p.y);
  const px = (xe[1] - xe[0]) * 0.1;
  const py = (ye[1] - ye[0]) * 0.1;
  const x = d3.scaleLinear().domain([xe[0] - px, xe[1] + px]).range([0, w]);
  const y = d3.scaleLinear().domain([ye[0] - py, ye[1] + py]).range([h, 0]);
  const BOX = [x.domain(), y.domain()];

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const clip = 'c-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -1).attr('y', -1)
    .attr('width', w + 2).attr('height', h + 2);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  const street = plot.append('path').attr('fill', 'var(--primary)').attr('opacity', 0.13);
  const kerbs = plot.append('g');
  const bnd = plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);
  const dotsG = g.append('g');
  const ringG = g.append('g');

  const seg = (d) => (d ? `M${x(d[0][0])},${y(d[0][1])}L${x(d[1][0])},${y(d[1][1])}` : null);
  const poly = (p) => (p ? `M${p.map((q) => `${x(q[0])},${y(q[1])}`).join('L')}Z` : null);

  const slider = document.querySelector('#c-slider');
  const onlySv = document.querySelector('#c-only-sv');
  const statsEl = document.querySelector('#c-stats');

  function render() {
    const r = runs[+slider.value];
    document.querySelector('#c-label').textContent = r.C >= 1 ? r.C : r.C.toString();
    const sv = new Set(r.support);

    street.attr('d', poly(streetPolygon(r.w, r.b, BOX)));
    kerbs.selectAll('path').data([1, -1]).join('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4').attr('opacity', 0.85)
      .attr('d', (c) => seg(lineInBox(r.w, r.b, c, BOX[0], BOX[1])));
    bnd.attr('d', seg(lineInBox(r.w, r.b, 0, BOX[0], BOX[1])));

    const shown = onlySv.checked ? pts.filter((p, i) => sv.has(i)) : pts;
    dotsG.selectAll('circle').data(shown, (p) => `${p.x},${p.y}`).join(
      (e) => e.append('circle').attr('stroke', 'white').attr('stroke-width', 1)
    )
      .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y))
      .attr('r', 4.4)
      .attr('fill', (p) => (p.c ? 'var(--cosmos)' : 'var(--aqua)'))
      .attr('opacity', 0.9);
    ringG.selectAll('circle').data(pts.filter((p, i) => sv.has(i))).join('circle')
      .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 8.5)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.8);

    /* How far the widest and narrowest settings actually move the boundary, so
     * "the street breathes" is a number rather than an impression. */
    const hardest = runs[runs.length - 1];
    const tilt = Math.acos(
      Math.min(1, Math.abs(r.w[0] * hardest.w[0] + r.w[1] * hardest.w[1]) /
        (Math.hypot(...r.w) * Math.hypot(...hardest.w)))
    ) * (180 / Math.PI);
    const inside = pts.filter((p) => Math.abs(margin(r.w, r.b, [p.x, p.y])) < r.half_width - 1e-9).length;

    statsEl.innerHTML =
      `<table class="gc-table">
         <tr><td>half-width of the street</td><td><span class="value">${r.half_width.toFixed(3)}</span></td></tr>
         <tr><td>support vectors</td><td><span class="value">${r.n_support}</span> of ${pts.length}</td></tr>
         <tr><td>points inside the street</td><td><span class="value">${inside}</span></td></tr>
         <tr><td>training accuracy</td><td><span class="value">${r.train_acc.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.7rem;line-height:1.45">` +
      `Against the hardest margin on the slider this boundary is tilted by ` +
      `<span class="value">${tilt.toFixed(1)}°</span> and its street is ` +
      `<span class="value">${(r.half_width / hardest.half_width).toFixed(1)}×</span> as wide. ` +
      `Training accuracy has not moved anywhere on the slider: what C buys is not fit, ` +
      `it is how many observations the answer rests on.</div>`;
  }

  slider.addEventListener('input', render);
  onlySv.addEventListener('change', render);
  render();
}
