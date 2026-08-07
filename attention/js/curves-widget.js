/* Two curves that explain two lines of the formula, behind one button.
 *
 * The first is why the scores are divided by the square root of the head size.
 * With independent components of unit variance, a dot product of dimension d
 * has standard deviation the root of d, so without the division the softmax
 * inputs grow with the dimension and the distribution collapses onto one
 * element. A collapsed softmax has a gradient of essentially zero, so the layer
 * stops learning. Measured on random vectors, because that is the regime the
 * claim is about: initialisation.
 *
 * The second is the reach of the gradient, measured with the same probe and
 * the same normalisation the sequences article used on its recurrent cells, so
 * the curves can share an axis with theirs honestly. Attention has a path of
 * length one between any two positions, and the question is whether that shows
 * up as a flat line or whether the layers in between put a decay back.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend, logTicks } from '../../assets/js/textdraw.js';

const VIEWS = [
  { key: 'scaling', label: 'why it is divided by the root of d' },
  { key: 'reach', label: 'how far the gradient reaches' },
];

export function initCurvesWidget(data) {
  let view = 'scaling';
  const chart = makeChart('#curves-chart', {
    width: 640, height: 400, margin: { top: 46, right: 30, bottom: 58, left: 72 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#curves-readout');

  d3.select('#curves-views').selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      view = d.key;
      d3.select('#curves-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });

  function drawScaling() {
    const rows = data.scaling.rows;
    const x = d3.scaleLog().domain([rows[0].d, rows[rows.length - 1].d]).range([0, w]);
    const top = Math.max(data.scaling.rows[0].uniform_entropy,
      ...rows.map((r) => Math.max(r.entropy_raw, r.entropy_scaled)));
    const y = d3.scaleLinear().domain([0, top * 1.08]).range([h, 0]);
    const xv = rows.map((r) => r.d);
    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, y, w, h, { xValues: xv, yTicks: 5 }));
    const axes = g.append('g');
    drawAxes(axes, x, y, w, h, { xValues: xv, yTicks: 5, yFmt: (v) => v.toFixed(1) });
    axisLabels(axes, w, h, {
      x: 'size of each head',
      y: 'entropy of the attention, in nats',
    });

    const uni = rows[0].uniform_entropy;
    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(uni)).attr('y2', y(uni))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3').attr('opacity', 0.5);
    seriesLabel(g, w, y(uni) - 6, 'looking everywhere equally',
      { anchor: 'end', size: 10.5, opacity: 0.65 });

    [['entropy_scaled', 'var(--primary)', 'divided by the root of the head size'],
      ['entropy_raw', 'var(--cosmos)', 'not divided']].forEach(([key, colour, label]) => {
      g.append('path').datum(rows)
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
        .attr('d', d3.line().x((r) => x(r.d)).y((r) => y(r[key])));
      rows.forEach((r) => {
        g.append('circle').attr('cx', x(r.d)).attr('cy', y(r[key])).attr('r', 3.4)
          .attr('fill', colour);
      });
    });
    legend(g, [
      { label: 'divided by the root of the head size', colour: 'var(--primary)', shape: 'line' },
      { label: 'not divided', colour: 'var(--cosmos)', shape: 'line' },
    ], { x0: 0, y0: -24, gap: 300, size: 10.5, cols: 2 });

    const last = rows[rows.length - 1];
    const first = rows[0];
    readout.html(
      `<div>At a head size of ${first.d} the two are close (`
      + `<span class="value">${first.entropy_raw.toFixed(3)}</span> against `
      + `<span class="value">${first.entropy_scaled.toFixed(3)}</span> nats). At `
      + `${last.d} the undivided version has fallen to `
      + `<span class="bold">${last.entropy_raw.toFixed(3)}</span> while the divided one holds at `
      + `<span class="bold">${last.entropy_scaled.toFixed(3)}</span>.</div>`
      + `<div>In the undivided version the average largest weight in a row reaches `
      + `<span class="value">${last.max_raw.toFixed(3)}</span>, which is a distribution that has `
      + `chosen one word out of ${data.scaling.length} and has almost no gradient left to give `
      + `about any of the others. That is what the division prevents, and it is why a `
      + `one-character constant in the formula is not a detail.</div>`
      + `<div>Measured on random vectors over ${data.scaling.samples} samples, because the claim `
      + `is about what happens at initialisation, before any of it has been trained.</div>`
    );
  }

  function drawReach() {
    const R = data.reach;
    const pos = R.relative.filter((v) => v > 0);
    const lo = Math.max(1e-6, Math.min(...pos));
    const hi = Math.max(...pos);
    /* The reach is noisy and, unlike a recurrence, never decays: it rides around
     * one and spikes above three. A fixed top of 1.4 clipped most of the curve
     * off the canvas, so the domain is fitted to the data it has to show. */
    const yBot = Math.min(lo, 0.5);
    const yTop = Math.max(hi * 1.1, 1.4);
    const x = d3.scaleLinear().domain([1, R.distances.length]).range([0, w]);
    const y = d3.scaleLog().domain([yBot, yTop]).range([h, 0]);
    const yv = logTicks(yBot, yTop);
    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, y, w, h, { xTicks: 6, yValues: yv }));
    const axes = g.append('g');
    drawAxes(axes, x, y, w, h, { xTicks: 6, yValues: yv,
      yFmt: (v) => (v >= 0.01 ? String(v) : v.toExponential(0)) });
    axisLabels(axes, w, h, {
      x: 'steps back from the end of the sentence',
      y: 'how much that input still moves the state',
    });

    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0.5)).attr('y2', y(0.5))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3').attr('opacity', 0.45);
    seriesLabel(g, w, y(0.5) - 6, 'half', { anchor: 'end', size: 10.5, opacity: 0.6 });

    const pts = R.distances.map((d, i) => ({ d, v: R.relative[i] })).filter((p) => p.v > 0);
    g.append('path').datum(pts)
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4)
      .attr('d', d3.line().x((p) => x(p.d)).y((p) => y(p.v)));
    legend(g, [{ label: 'self-attention', colour: 'var(--primary)', shape: 'line' }],
      { x0: 0, y0: -24, gap: 160, size: 10.5 });

    readout.html(
      `<div>The gradient `
      + (R.half_life
        ? `falls to half of its one-step value <span class="bold">${R.half_life} steps</span> back`
        : `<span class="bold">never falls to half</span> of its one-step value anywhere in the `
          + `sentence`)
      + `, and thirty steps back it still carries `
      + `<span class="value">${R.at_30 !== null && R.at_30 !== undefined ? R.at_30.toFixed(3) : 'n/a'}</span> `
      + `of what it carries one step back.</div>`
      + `<div>Same probe, same normalisation and the same ${R.sentences} long sentences the `
      + `previous article used on its recurrent cells, so this curve and those go on one axis `
      + `without anybody having to take it on trust. The reason for the difference is structural `
      + `rather than a matter of degree: in a recurrence the path from a word to the end is as `
      + `long as the distance between them, and here every path is one step.</div>`
    );
  }

  function render() {
    g.selectAll('*').remove();
    if (view === 'scaling') drawScaling();
    else drawReach();
  }

  render();
  return { render };
}
