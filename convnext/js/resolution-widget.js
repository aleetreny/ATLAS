/* Accuracy against input resolution, with the two controls that stop the
 * obvious reading from being the wrong one.
 *
 * Bigger inputs score better, and there are two different reasons that could
 * be true: more information reaching the network, or more computation spent
 * on it. The restride arm keeps the resolution and throws away the compute;
 * the information arm keeps the compute and throws away the information.
 * Between them they say which of the two was doing the work.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initResolutionWidget(data) {
  const rows = [...data.resolution.ladder].sort((a, b) => a.res - b.res);
  const W = 700;
  const H = 390;
  const margin = { top: 32, right: 186, bottom: 56, left: 72 };
  const { g, w, h } = makeChart('#resolution-chart', { width: W, height: H, margin });

  const restride = data.resolution.restride;
  const info = data.resolution.information_control;

  const x = d3.scaleLinear()
    .domain([Math.min(...rows.map((r) => r.res)) - 1, Math.max(...rows.map((r) => r.res)) + 1])
    .range([0, w]);
  const accs = rows.map((r) => r.test_acc_mean);
  const extra = [restride, info].flatMap((c) => (c ? Object.values(c).filter((v) => typeof v === 'number' && v > 0.2 && v < 1) : []));
  const lo = Math.min(...accs, ...extra);
  const hi = Math.max(...accs, ...extra);
  const pad = (hi - lo) * 0.2 + 0.004;
  const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: rows.length, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: rows.length, yTicks: 5, xFmt: d3.format('d'), yFmt: d3.format('.1%') });
  axisLabels(g, w, h, { x: 'input resolution, pixels square', y: 'test accuracy' });

  const line = d3.line().x((r) => x(r.res)).y((r) => y(r.test_acc_mean));
  g.append('path').datum(rows).attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6).attr('d', line);
  g.selectAll('.pt').data(rows).join('circle')
    .attr('class', 'pt')
    .attr('cx', (r) => x(r.res)).attr('cy', (r) => y(r.test_acc_mean))
    .attr('r', 4.5).attr('fill', 'var(--primary)');
  g.append('text').attr('x', w + 10).attr('y', 16)
    .attr('class', 'annotation').style('font-size', '12px').style('fill', 'var(--primary)')
    .text('the resolution ladder');

  /* Both controls sit at 24 pixels, so they are drawn on the same abscissa as
     the rung they are arguing with. */
  const marks = [];
  if (restride && typeof restride.test_acc_mean === 'number') {
    marks.push({ res: restride.res || 24, acc: restride.test_acc_mean,
      label: 'same size, quarter the compute', colour: '#232f3e' });
  }
  if (info && typeof info.test_acc_mean === 'number') {
    marks.push({ res: info.res || 24, acc: info.test_acc_mean,
      label: 'same compute, blurred first', colour: '#b45309' });
  }
  marks.forEach((m, i) => {
    g.append('circle').attr('cx', x(m.res)).attr('cy', y(m.acc)).attr('r', 6)
      .attr('fill', 'none').attr('stroke', m.colour).attr('stroke-width', 2.4);
    g.append('text').attr('x', w + 10).attr('y', 40 + i * 20)
      .attr('class', 'annotation').style('font-size', '12px').style('fill', m.colour)
      .text(m.label);
  });

  const first = rows[0];
  const last = rows[rows.length - 1];
  const gain = (last.test_acc_mean - first.test_acc_mean) * 100;

  let controls = '';
  if (restride && typeof restride.test_acc_mean === 'number') {
    controls += ` Keeping the input at ${restride.res || 24} pixels but halving the stem's stride, `
      + `so the network sees the same picture on a quarter of the compute, gives `
      + `${(restride.test_acc_mean * 100).toFixed(2)}%.`;
  }
  if (info && typeof info.test_acc_mean === 'number') {
    controls += ` Feeding it a ${info.res || 24}-pixel input that was built by blowing up a `
      + `12-pixel thumbnail, so the compute is identical and the information is not, gives `
      + `${(info.test_acc_mean * 100).toFixed(2)}%.`;
  }

  document.querySelector('#resolution-readout').innerHTML =
    `<span class="bold">From ${first.res} to ${last.res} pixels, ${gain >= 0 ? '+' : ''}${gain.toFixed(2)} points`
    + `</span>, for ${(last.macs / first.macs).toFixed(1)}× the multiply-accumulates and not one extra `
    + `parameter: resolution is the axis you can scale without making the model bigger.${controls} `
    + `Those two controls are the point of this chart. A ladder on its own cannot tell you whether `
    + `bigger inputs help because of what they contain or because of what they cost, and the answer `
    + `changes what you would do about it.`;
}
