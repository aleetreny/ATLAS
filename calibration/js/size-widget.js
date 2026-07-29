/* How much data does honesty cost: test ECE against calibration-set size.
 *
 * Both axes log. Every point is the mean of eleven seeded draws, measured
 * offline; the widget only draws and narrates. The patient is the SMOTE
 * logistic, the model with the largest raw distortion, so the repairs have
 * something real to earn at every size.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

export function initSizeWidget(data) {
  const sc = data.size_curve;
  const { g, w, h } = makeChart('#size-chart', {
    width: 660,
    height: 420,
    margin: { top: 30, right: 30, bottom: 58, left: 66 },
  });

  const all = [...sc.platt, ...sc.isotonic];
  const x = d3.scaleLog().domain([sc.sizes[0] * 0.8, sc.sizes[sc.sizes.length - 1] * 1.25]).range([0, w]);
  const y = d3.scaleLog().domain([d3.min(all) * 0.7, Math.max(d3.max(all), data.models[sc.model].raw.ece) * 1.4]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('~s'), yFmt: d3.format('.4~f') });
  axisLabels(g, w, h, { x: 'calibration cells used (log)', y: 'test ECE (log)' });

  const raw = data.models[sc.model].raw.ece;
  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(raw)).attr('y2', y(raw))
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  g.append('text').attr('class', 'chart-annotation').attr('x', w - 4).attr('y', y(raw) - 6)
    .attr('text-anchor', 'end').style('font-size', '0.66rem').style('text-transform', 'none')
    .attr('fill', 'var(--cosmos)').text(`no repair at all, ${raw.toFixed(4)}`);

  const series = [
    { name: 'Platt', vals: sc.platt, colour: 'var(--primary)' },
    { name: 'isotonic', vals: sc.isotonic, colour: 'var(--squidink)' },
  ];
  for (const s of series) {
    const pts = sc.sizes.map((n, i) => [n, s.vals[i]]);
    g.append('path').attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 3)
      .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts));
    g.selectAll(null).data(pts).enter().append('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 4.5).attr('fill', s.colour).attr('stroke', 'white').attr('stroke-width', 1.4);
    g.append('text').attr('class', 'chart-annotation')
      .attr('x', x(pts[pts.length - 1][0])).attr('y', y(pts[pts.length - 1][1]) + (s.name === 'Platt' ? 18 : -12))
      .attr('text-anchor', 'end').style('font-size', '0.56rem').style('text-transform', 'none')
      .attr('fill', s.colour).text(s.name);
  }

  const readout = document.querySelector('#size-readout');
  const n0 = sc.sizes[0];
  const last = sc.sizes.length - 1;
  const iBeat = sc.sizes.findIndex((n, i) => sc.isotonic[i] <= sc.platt[i]);
  readout.innerHTML =
    `<div class="slider-label" style="line-height:1.5">Repairing a raw ECE of ${raw.toFixed(4)}. At ` +
    `${n0} cells, carrying a single positive on average, Platt already reaches ` +
    `${sc.platt[0].toFixed(4)}, ${(raw / sc.platt[0]).toFixed(0)} times better than no repair, while ` +
    `isotonic trails at ${sc.isotonic[0].toFixed(4)}: a staircase fitted to a handful of outcomes ` +
    `memorises their noise. From there the two curves tell the whole story. Platt improves quickly and ` +
    `then flattens near ${sc.platt[last].toFixed(4)}: once its two parameters are right there is nothing ` +
    `left for it to learn, and whatever part of the distortion is not sigmoid-shaped becomes its ` +
    `permanent floor. Isotonic keeps converting cells into accuracy and ` +
    `${iBeat > 0
      ? `takes the lead at ${sc.sizes[iBeat].toLocaleString('en-US')} cells, ending at ` +
        `${sc.isotonic[last].toFixed(4)}: freedom loses small and wins big.`
      : `matches it once the data catches up with its freedom.`
    } The practical rule: with hundreds of calibration points, take the two parameters; ` +
    `with ten thousand, take the staircase; in between, either repair beats none by an order of ` +
    `magnitude.</div>`;
}
