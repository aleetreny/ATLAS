/* What skip-gram is actually computing, plotted against what it computes.
 *
 * Levy and Goldberg proved that the objective skip-gram with negative sampling
 * minimises is optimised when the dot product of a word vector and a context
 * vector equals the pointwise mutual information of that pair minus the log of
 * the number of negative samples. That is not an analogy or an intuition. It is
 * an equality, so it can be plotted as one: the horizontal axis is the quantity
 * the theorem names, computed from the co-occurrence counts and nothing else,
 * and the vertical axis is the dot product the training actually produced.
 *
 * The diagonal is therefore not a fit. It is what the theorem predicts, and the
 * interesting number is not the correlation but the slope: a tight correlation
 * with a slope of a third would mean the model has learned to rank pairs like
 * PMI without learning to equal it, which is a different and much weaker claim.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend } from '../../assets/js/textdraw.js';

export function initPmiWidget(data) {
  const P = data.pmi;
  const chart = makeChart('#pmi-chart', {
    width: 640, height: 420, margin: { top: 44, right: 26, bottom: 58, left: 66 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#pmi-readout');
  let showFit = false;

  d3.select('#pmi-views').selectAll('button')
    .data([{ k: false, label: 'what the theorem predicts' },
      { k: true, label: 'and the line that was fitted' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.k === showFit ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      showFit = d.k;
      d3.select('#pmi-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.k === showFit ? '' : ' ghost'}`);
      render();
    });

  function render() {
    g.selectAll('*').remove();
    const xs = P.cloud.map((p) => p[0]);
    const ys = P.cloud.map((p) => p[1]);
    const lo = Math.min(...xs, ...ys);
    const hi = Math.max(...xs, ...ys);
    const pad = 0.06 * (hi - lo);
    const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, w]);
    const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([h, 0]);

    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, y, w, h, { xTicks: 6, yTicks: 6 }));
    const axes = g.append('g');
    drawAxes(axes, x, y, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(axes, w, h, {
      x: 'mutual information of the pair, minus log of the negatives',
      y: 'the dot product the training produced',
    });

    /* the identity, which is the prediction */
    g.append('line')
      .attr('x1', x(lo - pad)).attr('y1', y(lo - pad))
      .attr('x2', x(hi + pad)).attr('y2', y(hi + pad))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2)
      .attr('stroke-dasharray', '7 4');

    const pts = g.append('g');
    P.cloud.forEach(([a, b]) => {
      pts.append('circle').attr('cx', x(a)).attr('cy', y(b)).attr('r', 1.9)
        .attr('fill', 'var(--primary)').attr('opacity', 0.35);
    });

    if (showFit) {
      const a0 = lo - pad;
      const a1 = hi + pad;
      g.append('line')
        .attr('x1', x(a0)).attr('y1', y(P.intercept + P.slope * a0))
        .attr('x2', x(a1)).attr('y2', y(P.intercept + P.slope * a1))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2);
    }

    legend(g, [
      { label: 'one word-context pair', colour: 'var(--primary)', shape: 'dot' },
      { label: 'what the theorem says', colour: 'var(--cosmos)', shape: 'dash' },
    ].concat(showFit
      ? [{ label: 'the line actually fitted', colour: 'var(--anchor)', shape: 'line' }]
      : []), { x0: 0, y0: -24, gap: 200, size: 10.5, cols: 2 });

    const close = Math.abs(P.slope - 1) < 0.15;
    readout.html(
      `<div>Over <span class="value">${P.pairs.toLocaleString('en-US')}</span> word-context `
      + `pairs seen at least <span class="value">${P.min_count}</span> times, the correlation `
      + `between the two is <span class="bold">${P.pearson.toFixed(4)}</span> and the fitted `
      + `slope is <span class="bold">${P.slope.toFixed(4)}</span>.</div>`
      + `<div>The slope is the number that matters. `
      + (close
        ? 'At close to one, the model is not merely ranking pairs the way mutual information '
          + 'would, it is reproducing its values, which is what the theorem claims.'
        : `At ${P.slope.toFixed(2)} rather than 1, the model orders pairs much the way mutual `
          + 'information does but does not reproduce its scale. The correlation alone would have '
          + 'hidden that, which is why the diagonal is drawn and not just the cloud: five epochs '
          + 'on a corpus this size is nowhere near the optimum the theorem describes, and the '
          + 'theorem is about the optimum.')
      + `</div>`
      + `<div>Pairs seen fewer than ${P.min_count} times are left out. The mutual information of `
      + `a pair observed twice is mostly noise with a formula around it, and including it would `
      + `measure the noise rather than the identity.</div>`
    );
  }

  render();
  return { render };
}
