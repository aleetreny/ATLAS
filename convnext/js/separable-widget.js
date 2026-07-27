/* Depthwise separable convolutions: the arithmetic, and then the stopwatch.
 *
 * The closed form is the part everyone quotes, so the sliders let you drive
 * it: the ratio tends to 1/k squared, which means a 3x3 separable convolution
 * can never be more than nine times cheaper than a dense one, however wide
 * the network gets. The saving is capped by the kernel, not by the width, and
 * that is visible the moment you move the channel slider and watch the number
 * stop moving. Then the measured wall clock disagrees with all of it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initSeparableWidget(data) {
  const kEl = document.querySelector('#sep-k');
  const cEl = document.querySelector('#sep-c');
  const kLab = document.querySelector('#sep-k-value');
  const cLab = document.querySelector('#sep-c-value');
  const readout = document.querySelector('#sep-readout');

  const W = 640;
  const H = 340;
  const margin = { top: 30, right: 120, bottom: 54, left: 74 };
  const { g, w, h } = makeChart('#sep-chart', { width: W, height: H, margin });

  const CH = [8, 16, 32, 64, 128, 256, 512];
  const x = d3.scaleLog().domain([CH[0], CH[CH.length - 1]]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('d'), yFmt: d3.format('.0%') });
  axisLabels(g, w, h, { x: 'channels in and out', y: 'separable cost, as a share of dense' });

  const curves = g.append('g');
  const marker = g.append('circle').attr('r', 6).attr('fill', 'var(--primary)');
  const asym = g.append('g');

  /* dense: k*k*Cin*Cout. separable: k*k*Cin (depthwise) + Cin*Cout (pointwise) */
  const ratio = (k, C) => (k * k * C + C * C) / (k * k * C * C);

  function draw() {
    const k = +kEl.value;
    const C = +cEl.value;
    kLab.textContent = `${k}×${k}`;
    cLab.textContent = String(C);

    const line = d3.line().x((c) => x(c)).y((c) => y(ratio(k, c)));
    curves.selectAll('path')
      .data([3, 5, 7, 9].map((kk) => ({ kk, pts: CH })))
      .join('path')
      .attr('fill', 'none')
      .attr('stroke', (d) => (d.kk === k ? 'var(--primary)' : '#c3c9d2'))
      .attr('stroke-width', (d) => (d.kk === k ? 2.6 : 1.4))
      .attr('d', (d) => d3.line().x((c) => x(c)).y((c) => y(ratio(d.kk, c)))(d.pts));

    curves.selectAll('.klab')
      .data([3, 5, 7, 9])
      .join('text')
      .attr('class', 'klab')
      .attr('x', w + 8)
      .attr('y', (kk) => y(ratio(kk, CH[CH.length - 1])) + 4)
      .attr('class', 'annotation')
      .style('font-size', '11px')
      .style('fill', (kk) => (kk === k ? 'var(--primary)' : '#8a94a6'))
      .text((kk) => `${kk}×${kk}, floor 1/${kk * kk}`);

    marker.attr('cx', x(C)).attr('cy', y(ratio(k, C)));

    asym.selectAll('line')
      .data([1 / (k * k)])
      .join('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', (v) => y(v)).attr('y2', (v) => y(v))
      .attr('stroke', 'var(--primary)').attr('stroke-dasharray', '4 4').attr('opacity', 0.7);

    const r = ratio(k, C);
    const floor = 1 / (k * k);
    const stop = (data.separable.stopwatch || [])
      .filter((s) => s.threads === (data.meta.threads_requested || 12));
    const measured = stop.length
      ? stop.reduce((a, b) => (Math.abs(b.channels - C) < Math.abs(a.channels - C) ? b : a))
      : null;

    readout.innerHTML =
      `<span class="bold">At ${k}×${k} and ${C} channels, a separable convolution costs `
      + `${(r * 100).toFixed(1)}% of a dense one.</span> Push the channel slider and watch it stop `
      + `falling: the limit is 1/${k * k} = ${(floor * 100).toFixed(1)}%, set entirely by the kernel. `
      + `Width cannot buy you past it, which is why the arithmetic in the papers is always quoted `
      + `per kernel size.`
      + (measured
        ? ` <br /><br />And then the stopwatch. On this machine, at ${measured.channels} channels and `
          + `${measured.spatial}×${measured.spatial}, the multiply-accumulate ratio is `
          + `${measured.mac_ratio.toFixed(3)} and the measured wall-clock ratio is `
          + `<span class="bold">${measured.time_ratio.toFixed(2)}</span>. Fewer arithmetic operations, `
          + `more seconds. ${data.separable.finding}`
        : '');
  }

  kEl.addEventListener('input', draw);
  cEl.addEventListener('input', draw);
  draw();
}
