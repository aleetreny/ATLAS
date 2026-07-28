/* Two pictures of the same failure: what the generator is told when the
 * discriminator is winning.
 *
 * The first is the mixture from the previous widget with the model slid away
 * from it, and the gradient the two candidate losses hand back. The second is
 * the case the theory says is hopeless and image data always sits in, two
 * distributions with no overlap at all, where the Jensen-Shannon divergence is
 * a constant and a constant has no slope. Both are read off measurements: the
 * first from exact integrals, the second through networks that were actually
 * trained, because an argument about training is only interesting if the
 * trained thing behaves the way the closed form says.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const abs = Math.abs;

export function initGradWidget(data) {
  const rows = data.grads;
  const c = makeChart('#grad-chart', {
    width: 700, height: 420, margin: { top: 40, right: 112, bottom: 56, left: 74 },
  });
  const lo = Math.max(1e-8, d3.min(rows, (r) => Math.min(abs(r.sat) || 1, abs(r.ns) || 1)));
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.sep)]).range([0, c.w]);
  const y = d3.scaleLog().domain([lo, d3.max(rows, (r) => Math.max(abs(r.sat), abs(r.ns))) * 1.6])
    .range([c.h, 0]);
  const ticks = [];
  for (let e = -8; e <= 2; e++) ticks.push(10 ** e);
  const shown = ticks.filter((t) => t >= y.domain()[0] && t <= y.domain()[1]);
  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 7, yValues: shown });
  drawAxes(c.g, x, y, c.w, c.h, {
    xTicks: 7, yValues: shown, yFmt: (v) => (v >= 1 ? d3.format('~s')(v) : d3.format('.0e')(v)),
  });
  axisLabels(c.g, c.w, c.h, {
    x: 'where the model sits, as a mean on the line',
    y: 'size of the gradient it receives',
  });

  const series = [
    { k: 'ns', label: 'non-saturating', colour: 'var(--primary)' },
    { k: 'sat', label: 'saturating', colour: 'var(--cosmos)' },
  ];
  series.forEach((s) => {
    const pts = rows.map((r) => [r.sep, Math.max(abs(r[s.k]), lo)]);
    c.g.append('path').attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 2.4)
      .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts));
    const last = pts[pts.length - 1];
    c.g.append('text').attr('class', 'chart-annotation')
      .attr('x', c.w + 6).attr('y', y(last[1]) + 4).style('font-size', '11.5px')
      .style('letter-spacing', '0.4px').style('fill', s.colour).text(s.label);
  });

  /* the two component means, because the sign of both gradients flips there
     and a reader will otherwise think the wiggle is noise */
  data.meta.mixture.forEach((m) => {
    if (m.mu < 0) return;
    c.g.append('line').attr('x1', x(m.mu)).attr('x2', x(m.mu)).attr('y1', 0).attr('y2', c.h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('stroke-dasharray', '3 4');
    c.g.append('text').attr('class', 'chart-annotation').attr('x', x(m.mu) + 5).attr('y', 14)
      .style('font-size', '11px').style('letter-spacing', '0.4px').text('a data mode');
  });

  const pick = (s) => rows.reduce((b, r) => (abs(r.sep - s) < abs(b.sep - s) ? r : b), rows[0]);
  const at = [3, 4.5, 6].map(pick);
  const cells = at.map((r) => `<tr><td>${r.sep.toFixed(1)}</td>`
    + `<td>${(r.acc * 100).toFixed(2)}%</td>`
    + `<td>${abs(r.sat).toExponential(2)}</td><td>${abs(r.ns).toExponential(2)}</td>`
    + `<td>${(abs(r.ns) / Math.max(abs(r.sat), 1e-300)).toExponential(1)}&times;</td></tr>`).join('');
  document.querySelector('#grad-readout').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Model mean</th><th>Best discriminator is right</th>`
    + `<th>Saturating gradient</th><th>Non-saturating gradient</th><th>Ratio</th></tr></thead>`
    + `<tbody>${cells}</tbody></table>`;
}

export function initDisjointWidget(data) {
  const rows = data.disjoint.rows;
  const c = makeChart('#disjoint-chart', {
    width: 700, height: 400, margin: { top: 40, right: 118, bottom: 56, left: 76 },
  });
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.theta)]).range([0, c.w]);
  const y = d3.scaleLinear().domain([0, 2.2]).range([c.h, 0]);
  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 6 });
  drawAxes(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 6 });
  axisLabels(c.g, c.w, c.h, { x: 'distance between the two supports', y: 'divergence' });

  const draw = (key, colour, dash, label) => {
    const pts = rows.map((r) => [r.theta, r[key]]);
    c.g.append('path').attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.4)
      .attr('stroke-dasharray', dash).attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts));
    const last = pts[pts.length - 1];
    c.g.append('text').attr('class', 'chart-annotation').attr('x', c.w + 6)
      .attr('y', y(last[1]) + 4).style('font-size', '11.5px').style('letter-spacing', '0.4px')
      .style('fill', colour).text(label);
  };
  draw('w1', 'var(--primary)', null, 'Wasserstein');
  draw('w_est', 'var(--anchor)', '5 3', 'the critic');
  draw('js', 'var(--cosmos)', null, 'Jensen-Shannon');

  const state = { i: rows.length - 3 };
  const pic = makeChart('#disjoint-pic', {
    width: 300, height: 400, margin: { top: 40, right: 20, bottom: 56, left: 44 },
  });
  const px = d3.scaleLinear().domain([-0.4, 2.4]).range([0, pic.w]);
  const py = d3.scaleLinear().domain([-0.1, 1.1]).range([pic.h, 0]);
  drawAxes(pic.g, px, py, pic.w, pic.h, { xTicks: 4, yTicks: 3 });
  axisLabels(pic.g, pic.w, pic.h, { x: 'x', y: 'y' });
  const barA = pic.g.append('line').attr('stroke', 'var(--squidink)').attr('stroke-width', 5)
    .attr('x1', px(0)).attr('x2', px(0)).attr('y1', py(0)).attr('y2', py(1));
  const barB = pic.g.append('line').attr('stroke', 'var(--primary)').attr('stroke-width', 5);
  const capA = pic.g.append('text').attr('class', 'chart-annotation')
    .attr('x', px(0)).attr('y', -10)
    .attr('text-anchor', 'start').style('font-size', '11px').style('letter-spacing', '0.4px')
    .text('data');
  const capB = pic.g.append('text').attr('class', 'chart-annotation')
    .attr('text-anchor', 'start').style('font-size', '11px').style('letter-spacing', '0.4px')
    .style('fill', 'var(--primary)').text('model');

  const marker = c.g.append('circle').attr('r', 5).attr('fill', 'var(--squidink)');

  function render() {
    const r = rows[state.i];
    barB.attr('x1', px(r.theta)).attr('x2', px(r.theta)).attr('y1', py(0)).attr('y2', py(1));
    /* at zero the two segments are the same segment, so two labels would sit on
       top of each other saying something the picture already says */
    /* when the two segments are close their two labels would sit on top of
       each other, so the second one drops to its own line; at zero they are the
       same segment and one label says so */
    const near = px(r.theta) - px(0) < 52;
    capB.attr('x', Math.min(px(r.theta), pic.w - 40))
      .attr('y', near ? -24 : -10)
      .attr('opacity', r.theta === 0 ? 0 : 1);
    capA.text(r.theta === 0 ? 'data and model, the same' : 'data');
    marker.attr('cx', x(r.theta)).attr('cy', y(r.w1));
    document.querySelector('#disjoint-theta-value').textContent = r.theta.toFixed(2);
    const ratio = abs(r.g_crit) / Math.max(abs(r.g_sat), 1e-300);
    document.querySelector('#disjoint-readout').innerHTML =
      `<p>Two segments of the same length, <span class="bold">${r.theta.toFixed(2)}</span> apart. `
      + (r.theta === 0
        ? `They are the same distribution, so there is nothing for either method to measure and `
          + `both correctly say zero.`
        : `They share not one point, so the Jensen-Shannon divergence is `
          + `<span class="bold">log 2 = ${Math.log(2).toFixed(4)}</span> here and at every other `
          + `non-zero distance: as a function of where the model is, it is flat, and a flat `
          + `function has no downhill. The Wasserstein distance is `
          + `<span class="bold">${r.w1.toFixed(2)}</span>, which is the distance itself.`)
      + `</p><table class="atlas-table"><thead><tr><th>Measured through a trained network</th>`
      + `<th>Value</th><th>What it means</th></tr></thead><tbody>`
      + `<tr><td>discriminator on real, on generated</td>`
      + `<td>${r.d_real.toFixed(4)} / ${r.d_fake.toFixed(4)}</td>`
      + `<td>it separates them ${r.theta === 0 ? 'not at all, which is correct' : 'perfectly'}</td></tr>`
      + `<tr><td>gradient from the saturating loss</td><td>${abs(r.g_sat).toExponential(2)}</td>`
      + `<td>what the generator is told to do about it</td></tr>`
      + `<tr><td>gradient from a gradient-penalised critic</td><td>${abs(r.g_crit).toFixed(4)}</td>`
      + `<td>${r.theta === 0 ? 'nothing to say, and it says nothing'
        : `${ratio >= 10 ? `${d3.format('.3~s')(ratio)} times more to work with` : 'the same order of size'}`}</td></tr>`
      + `<tr><td>the critic's own estimate of the distance</td><td>${r.w_est.toFixed(4)}</td>`
      + `<td>against a true ${r.w1.toFixed(2)}</td></tr></tbody></table>`;
  }

  const slider = document.querySelector('#disjoint-theta');
  slider.min = 0;
  slider.max = rows.length - 1;
  slider.step = 1;
  slider.value = state.i;
  slider.addEventListener('input', () => { state.i = +slider.value; render(); });
  render();
}
