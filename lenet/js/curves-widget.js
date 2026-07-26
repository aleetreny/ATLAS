/* The three training runs, drawn from the losses they actually recorded.
 *
 * This chart exists because of an inconvenient measurement: on a network this
 * small, tanh beat ReLU. Drawing the curves rather than asserting a result
 * lets the reader see how far apart they are and decide how much to make of
 * a gap of six tenths of a point.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initCurvesWidget(data) {
  const cnn = data.models.cnn.history;
  const mlp = data.models.mlp.history;
  const tan = data.models.tanh_history;

  /* The two greens of the theme are too close to tell apart on a line chart,
     so the tanh run gets a warm accent as well as a dashed stroke. */
  const TANH = '#b45309';

  const series = [
    { key: 'cnn', label: 'convolutional, ReLU', hist: cnn, colour: 'var(--primary)', dash: null },
    { key: 'tanh', label: 'convolutional, tanh', hist: tan, colour: TANH, dash: '5 4' },
    { key: 'mlp', label: 'fully connected, ReLU', hist: mlp, colour: '#232f3e', dash: null },
  ];

  const W = 700;
  const H = 380;
  const margin = { top: 26, right: 168, bottom: 52, left: 66 };
  const { g, w, h } = makeChart('#curves-chart', { width: W, height: H, margin });

  const x = d3.scaleLinear().domain([1, cnn.length]).range([0, w]);
  const y = d3.scaleLinear()
    .domain([0, d3.max([...cnn, ...mlp, ...tan]) * 1.04])
    .range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: cnn.length, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: cnn.length, xFmt: d3.format('d'), yFmt: d3.format('.2f') });
  axisLabels(g, w, h, { x: 'epoch', y: 'training loss' });

  const line = d3.line().x((d, i) => x(i + 1)).y((d) => y(d));

  series.forEach((s, i) => {
    g.append('path')
      .datum(s.hist)
      .attr('fill', 'none')
      .attr('stroke', s.colour)
      .attr('stroke-width', 2.5)
      .attr('stroke-dasharray', s.dash)
      .attr('d', line);
    g.selectAll(`.pt-${s.key}`)
      .data(s.hist)
      .join('circle')
      .attr('class', `pt-${s.key}`)
      .attr('cx', (d, j) => x(j + 1))
      .attr('cy', (d) => y(d))
      .attr('r', 3)
      .attr('fill', s.colour);
    g.append('text')
      .attr('x', w + 10)
      .attr('y', 16 + i * 20)
      .attr('class', 'annotation')
      .style('font-size', '12px')
      .style('fill', s.colour)
      .text(s.label);
  });

  /* The shared finish line the generator used: 1.5x the worse of the two
     convolutional runs' final losses, so neither activation gets a target
     tuned to its own trajectory. */
  const target = 1.5 * Math.max(cnn[cnn.length - 1], tan[tan.length - 1]);
  g.append('line')
    .attr('x1', 0).attr('x2', w)
    .attr('y1', y(target)).attr('y2', y(target))
    .attr('stroke', '#8a94a6')
    .attr('stroke-dasharray', '4 4')
    .attr('stroke-width', 1);
  g.append('text')
    .attr('x', 4).attr('y', y(target) - 11)
    .attr('class', 'annotation')
    .style('font-size', '11px')
    .style('fill', '#8a94a6')
    /* Short, because the tanh curve crosses this height around epoch 3 and a
       longer label would run straight into it. */
    .text(`finish line ${target.toFixed(3)}`);

  const firstAt = (hist) => hist.findIndex((v) => v <= target) + 1;
  [
    { hist: cnn, colour: 'var(--primary)' },
    { hist: tan, colour: TANH },
  ].forEach((s) => {
    const e = firstAt(s.hist);
    if (e < 1) return;
    g.append('circle')
      .attr('cx', x(e))
      .attr('cy', y(s.hist[e - 1]))
      .attr('r', 6.5)
      .attr('fill', 'none')
      .attr('stroke', s.colour)
      .attr('stroke-width', 2);
  });

  document.querySelector('#curves-readout').innerHTML =
    `The two convolutional runs are the same network, the same seed and the same data; only the activation differs. `
    + `Crossing the shared finish line at ${target.toFixed(3)} took ReLU ${firstAt(cnn)} epochs and tanh ${firstAt(tan)}, `
    + `and after eight epochs tanh sat at ${tan[tan.length - 1].toFixed(4)} against ReLU's ${cnn[cnn.length - 1].toFixed(4)}, `
    + `scoring ${(data.ingredients[0].baseline * 100).toFixed(2)}% on the test set against ${(data.ingredients[0].acc * 100).toFixed(2)}%.`;
}
