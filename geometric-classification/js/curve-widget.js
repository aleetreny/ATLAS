/* How much data each discriminant needs before its extra freedom pays.
 *
 * The left end of this chart is the interesting part and it is a hole: QDA
 * cannot be fitted at all below fourteen rows per class on thirteen features,
 * because the covariance it has to invert is singular. Drawing that as a gap in
 * the line, with the reason stated, is more honest than dropping the rows and
 * letting the chart start where the method happens to work.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

const SERIES = [
  { key: 'lda', label: 'LDA', colour: 'var(--primary)' },
  { key: 'qda', label: 'QDA', colour: 'var(--cosmos)' },
  { key: 'gnb', label: 'naive Bayes', colour: 'var(--aqua)' },
];

export function initCurveWidget(data) {
  const rows = data.discriminant_curve;
  const { g, w, h } = makeChart('#curve-chart', {
    width: 660,
    height: 400,
    margin: { top: 34, right: 116, bottom: 58, left: 66 },
  });

  const x = d3.scalePoint().domain(rows.map((r) => r.n_per_class)).range([0, w]).padding(0.5);
  const lo = d3.min(rows.flatMap((r) => SERIES.map((s) => r[s.key]).filter((v) => v != null)));
  const hi = d3.max(rows.flatMap((r) => SERIES.map((s) => r[s.key]).filter((v) => v != null)));
  const y = d3.scaleLinear().domain([lo - 0.03, Math.min(1, hi + 0.02)]).range([h, 0]);

  drawGrid(g, x, y, w, h, { yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.2f') });
  axisLabels(g, w, h, { x: 'training rows per class', y: 'accuracy on everything held out' });

  const tip = tooltip();
  const line = d3.line().x((r) => x(r.n_per_class)).y((r) => y(r[line.key]));

  /* The band where QDA could not be fitted, shaded once rather than explained
   * twice. Its right edge is the first sample size at which it ran. */
  const firstQda = rows.find((r) => r.qda != null);
  if (firstQda) {
    const edge = x(firstQda.n_per_class) - (x(rows[1].n_per_class) - x(rows[0].n_per_class)) / 2;
    g.insert('rect', ':first-child')
      .attr('x', 0).attr('y', 0).attr('width', Math.max(0, edge)).attr('height', h)
      .attr('fill', 'var(--cosmos)').attr('opacity', 0.07);
    g.append('text').attr('class', 'chart-annotation')
      .attr('x', Math.max(0, edge) / 2).attr('y', 16).attr('text-anchor', 'middle')
      .style('font-size', '0.6rem').style('text-transform', 'none')
      .attr('fill', 'var(--cosmos)')
      .text('QDA cannot be fitted here');
  }

  for (const s of SERIES) {
    const pts = rows.filter((r) => r[s.key] != null);
    line.key = s.key;
    g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6).attr('d', line(pts));
    g.append('path').attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 3).attr('d', line(pts));
    g.append('g').selectAll('circle').data(pts).join('circle')
      .attr('cx', (r) => x(r.n_per_class)).attr('cy', (r) => y(r[s.key])).attr('r', 4.2)
      .attr('fill', s.colour).attr('stroke', 'white').attr('stroke-width', 1.4)
      .on('mousemove', (event, r) => tip.show(
        `<span style="color:${s.colour}">${s.label}</span><br/>${r.n_per_class} per class<br/>` +
        `accuracy ${r[s.key].toFixed(3)} ± ${r[s.key + '_sd'].toFixed(3)}`,
        event
      ))
      .on('mouseleave', () => tip.hide());
    g.append('text').attr('class', 'chart-annotation')
      .attr('x', w + 8).attr('y', y(pts[pts.length - 1][s.key]) + 4)
      .style('font-size', '0.62rem').style('text-transform', 'none')
      .attr('fill', s.colour).text(s.label);
  }

  /* Where QDA overtakes LDA, found rather than quoted. */
  const cross = rows.find((r) => r.qda != null && r.lda != null && r.qda > r.lda);
  const last = rows[rows.length - 1];
  document.querySelector('#curve-readout').innerHTML =
    `<div class="slider-label" style="line-height:1.5">` +
    `Thirteen features, three classes, and 24 random draws at every sample size. ` +
    `Below <span class="value">${firstQda ? firstQda.n_per_class : '?'}</span> rows per class QDA has no answer at all: ` +
    `a covariance estimated from fewer rows than it has dimensions is singular and cannot be inverted. ` +
    `${
      cross
        ? `From <span class="value">${cross.n_per_class}</span> rows per class it overtakes LDA, and by ` +
          `<span class="value">${last.n_per_class}</span> it is ahead by ` +
          `<span class="value">${((last.qda - last.lda) * 100).toFixed(1)}</span> points. ` +
          `That crossover is the whole practical difference between the two: the extra freedom is real, and you ` +
          `have to buy it with rows.`
        : `Over this range it never overtakes LDA, which is the same lesson from the other side.`
    }` +
    ` Naive Bayes, which estimates no covariance at all, is the flattest line on the chart: worst once data is ` +
    `plentiful, and the only one of the three that never has to invert a matrix, so the sample sizes where QDA ` +
    `has no answer never trouble it.</div>`;
}
