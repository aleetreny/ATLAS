/* The threshold, and what it costs.
 *
 * The cost slider is the point of the widget: it turns "which threshold is
 * best?" into "what do you think a missed cancer is worth?", which is the honest
 * form of the question. The cheapest threshold is recomputed from the sweep
 * every time that ratio changes, so the verdict is always a measurement.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initThreshWidget(data) {
  const sweep = data.threshold_sweep;
  const { g, w, h } = makeChart('#thresh-chart', {
    width: 620,
    height: 430,
    /* The right margin has to hold "missed cancers" at the end of its line. */
    margin: { top: 32, right: 122, bottom: 56, left: 62 },
  });

  /* A hair of padding at both ends of both axes: a domain starting at exactly
   * zero stacks the "0.0" of one axis on the "0" of the other, right where they
   * meet. */
  const x = d3.scaleLinear().domain([-0.02, 1.02]).range([0, w]);
  const maxErr = d3.max(sweep, (s) => Math.max(s.fn, s.fp));
  const y = d3.scaleLinear().domain([-maxErr * 0.03, maxErr * 1.1]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('d') });
  axisLabels(g, w, h, { x: 'decision threshold', y: 'patients, out of 171 held out' });

  const line = d3.line().x((s) => x(s.t)).y((s) => y(s.v));
  const SERIES = [
    { key: 'fn', label: 'missed cancers', colour: 'var(--cosmos)' },
    { key: 'fp', label: 'false alarms', colour: 'var(--aqua)' },
  ];
  for (const s of SERIES) {
    const pts = sweep.map((r) => ({ t: r.t, v: r[s.key] }));
    g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6).attr('d', line(pts));
    g.append('path').attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 3).attr('d', line(pts));
    g.append('text').attr('class', 'chart-annotation')
      .attr('x', w + 8)
      .attr('y', y(pts[pts.length - 1].v) + 4)
      .style('font-size', '0.62rem').style('text-transform', 'none')
      .attr('fill', s.colour)
      .text(s.label);
  }

  /* The cheapest threshold moves with the cost ratio, so it gets its own
   * marker: a faint rule the reader can watch slide as they drag the cost. */
  const bestRule = g.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4').attr('opacity', 0.55);
  const bestLabel = g.append('text').attr('class', 'chart-annotation')
    .attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.6rem').style('text-transform', 'none');
  const rule = g.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
  const dots = g.append('g');

  const tEl = document.querySelector('#thresh-t');
  const cEl = document.querySelector('#thresh-c');
  const statsEl = document.querySelector('#thresh-stats');

  const at = (t) => sweep.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
  const cheapest = (cost) => sweep.reduce((a, b) => (b.fn * cost + b.fp < a.fn * cost + a.fp ? b : a));

  /* This model is good enough that the cost calculus has exactly one tipping
   * point over the whole slider, and saying where it is beats letting the
   * reader drag a control that mostly returns the same answer. Found by sweep
   * rather than quoted, so it stays true if the data is regenerated. */
  const cheapAt = [];
  for (let c = +cEl.min; c <= +cEl.max; c++) cheapAt.push({ c, t: cheapest(c).t });
  const flip = cheapAt.find((r, i) => i > 0 && r.t !== cheapAt[i - 1].t);

  function render() {
    const t = +tEl.value;
    const cost = +cEl.value;
    document.querySelector('#thresh-t-label').textContent = t.toFixed(2);
    document.querySelector('#thresh-c-label').textContent = cost;

    const s = at(t);
    const best = cheapest(cost);
    rule.attr('x1', x(s.t)).attr('x2', x(s.t));
    bestRule.attr('x1', x(best.t)).attr('x2', x(best.t));
    bestLabel.attr('x', x(best.t)).text(`cheapest at ${best.t.toFixed(2)}`);
    dots.selectAll('circle').data(SERIES).join('circle')
      .attr('cx', x(s.t))
      .attr('cy', (d) => y(s[d.key]))
      .attr('r', 5.5)
      .attr('fill', (d) => d.colour)
      .attr('stroke', 'white')
      .attr('stroke-width', 1.8);

    const total = s.fn * cost + s.fp;
    const bestTotal = best.fn * cost + best.fp;
    statsEl.innerHTML =
      `<table class="pc-table">
         <tr><th></th><th>said malignant</th><th>said benign</th></tr>
         <tr><td>was malignant</td><td><span class="value">${s.tp}</span></td>` +
      `<td><span class="value" style="color:var(--cosmos)">${s.fn}</span></td></tr>
         <tr><td>was benign</td><td><span class="value" style="color:var(--aqua)">${s.fp}</span></td>` +
      `<td><span class="value">${s.tn}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.7rem">recall <span class="value">${s.recall.toFixed(3)}</span> ` +
      `· precision <span class="value">${s.precision.toFixed(3)}</span></div>` +
      `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">` +
      `At ${cost}× the cost, this threshold runs up <span class="value">${total}</span> units of regret. ` +
      `${
        s.t === best.t
          ? 'Nothing on the sweep is cheaper.'
          : `Threshold <span class="value">${best.t.toFixed(2)}</span> costs ` +
            `<span class="value">${bestTotal}</span>, trading ${best.fp - s.fp > 0 ? best.fp - s.fp : 0} extra ` +
            `false alarms for ${s.fn - best.fn > 0 ? s.fn - best.fn : 0} fewer missed cancers.`
      }</div>` +
      (flip
        ? `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">` +
          `This model is good enough that the whole argument has one tipping point. ` +
          `Below <span class="value">${flip.c}×</span> the cheapest threshold is ` +
          `<span class="value">${cheapAt[0].t.toFixed(2)}</span>; from ` +
          `<span class="value">${flip.c}×</span> upward it becomes ` +
          `<span class="value">${flip.t.toFixed(2)}</span>, and it never moves again. ` +
          `Everything between those two settings is the same ` +
          `<span class="value">${Math.max(0, at(flip.t).fp - at(cheapAt[0].t).fp)}</span> extra false alarms ` +
          `bought for the same ` +
          `<span class="value">${Math.max(0, at(cheapAt[0].t).fn - at(flip.t).fn)}</span> cancers.</div>`
        : '');
  }

  tEl.addEventListener('input', render);
  cEl.addEventListener('input', render);
  document.querySelector('#thresh-best').addEventListener('click', () => {
    tEl.value = cheapest(+cEl.value).t;
    render();
  });
  render();
}
