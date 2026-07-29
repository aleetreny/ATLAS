/* Two ways of choosing how many topics there are, plotted against each other.
 *
 * Held-out perplexity is the criterion the model itself defines. Coherence is
 * a criterion built from what the top words of a topic look like next to each
 * other in real documents. They are both curves over the same fits, and the
 * question is not which is better but whether they pick the same answer.
 *
 * Both series share one panel and they are on different scales, so the second
 * one gets its own axis, on the right, with its own ticks. A second scale
 * without an axis is decoration: two series and one readable ruler means the
 * reader is being asked to trust the shapes rather than read them.
 *
 * The coherence axis carries a shaded band for the floor, which is what
 * randomly chosen words score under the same measure. Without it, "+0.23 of
 * coherence" is a number with no zero.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend } from '../../assets/js/textdraw.js';

const SERIES = [
  { key: 'coherence', label: 'coherence of the topics', colour: 'var(--primary)' },
  { key: 'perplexity', label: 'held-out perplexity', colour: 'var(--cosmos)' },
  { key: 'ari', label: 'agreement with the answer key', colour: 'var(--anchor)' },
];

export function initKSweepWidget(data) {
  const rows = data.ksweep.rows;
  const chart = makeChart('#ksweep-chart', {
    width: 640, height: 400, margin: { top: 46, right: 72, bottom: 58, left: 68 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#ksweep-readout');
  let right = 'perplexity';

  d3.select('#ksweep-right').selectAll('button')
    .data(SERIES.filter((s) => s.key !== 'coherence'))
    .join('button')
    .attr('class', (d) => `atlas-btn${d.key === right ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      right = d.key;
      d3.select('#ksweep-right').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === right ? '' : ' ghost'}`);
      render();
    });

  function render() {
    g.selectAll('*').remove();
    const ks = rows.map((r) => r.k);
    const x = d3.scalePoint().domain(ks).range([0, w]).padding(0.5);

    const coh = rows.map((r) => r.coherence);
    const fl = rows.map((r) => r.coherence_floor);
    const yL = d3.scaleLinear()
      .domain([Math.min(...fl) - 0.03, Math.max(...coh) + 0.03]).range([h, 0]);
    const other = rows.map((r) => r[right]);
    const yR = d3.scaleLinear()
      .domain(right === 'perplexity'
        ? [Math.min(...other) * 0.96, Math.max(...other) * 1.02]
        : [0, Math.max(...other) * 1.1])
      .range([h, 0]);

    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, yL, w, h, { xTicks: 0, yTicks: 5 }));
    const axes = g.append('g');
    drawAxes(axes, x, yL, w, h, { yTicks: 5, yFmt: (v) => v.toFixed(2) });
    axisLabels(axes, w, h, { x: 'number of topics asked for' });

    /* the right-hand axis, drawn by hand because chart.js owns only one */
    const rightAxis = g.append('g').attr('class', 'axis y')
      .attr('transform', `translate(${w},0)`);
    rightAxis.call(d3.axisRight(yR).ticks(5)
      .tickFormat(right === 'perplexity' ? d3.format(',.0f') : ((v) => v.toFixed(2))));
    rightAxis.selectAll('text').style('fill', SERIES.find((s) => s.key === right).colour);

    /* the coherence floor: randomly chosen words under the same measure */
    g.append('path')
      .datum(rows)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2)
      .attr('stroke-dasharray', '4 4').attr('opacity', 0.6)
      .attr('d', d3.line().x((d) => x(d.k)).y((d) => yL(d.coherence_floor)));

    const line = (acc, scale, colour) => {
      g.append('path').datum(rows)
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
        .attr('d', d3.line().x((d) => x(d.k)).y((d) => scale(acc(d))));
      rows.forEach((d) => {
        g.append('circle').attr('cx', x(d.k)).attr('cy', scale(acc(d))).attr('r', 3.4)
          .attr('fill', colour);
      });
    };
    line((d) => d[right], yR, SERIES.find((s) => s.key === right).colour);
    line((d) => d.coherence, yL, 'var(--primary)');

    /* the two argmaxima, marked, because they are the finding */
    const bestCoh = rows.reduce((p, c) => (c.coherence > p.coherence ? c : p), rows[0]);
    const bestOther = right === 'perplexity'
      ? rows.reduce((p, c) => (c.perplexity < p.perplexity ? c : p), rows[0])
      : rows.reduce((p, c) => (c[right] > p[right] ? c : p), rows[0]);
    [[bestCoh, 'var(--primary)', 'best coherence'],
      [bestOther, SERIES.find((s) => s.key === right).colour,
        right === 'perplexity' ? 'best perplexity' : 'best agreement']]
      .forEach(([r, colour, text], i) => {
        g.append('line').attr('x1', x(r.k)).attr('x2', x(r.k)).attr('y1', 0).attr('y2', h)
          .attr('stroke', colour).attr('stroke-width', 1).attr('stroke-dasharray', '3 3')
          .attr('opacity', 0.55);
        const t = seriesLabel(g, x(r.k) + 6, 14 + i * 15, `${text}: k = ${r.k}`,
          { anchor: 'start', size: 10.5, colour });
        if (x(r.k) + 6 + t.node().getComputedTextLength() > w) {
          t.attr('x', x(r.k) - 6).attr('text-anchor', 'end');
        }
      });

    legend(g, [
      { label: 'coherence, left axis', colour: 'var(--primary)', shape: 'line' },
      { label: `${SERIES.find((s) => s.key === right).label}, right axis`,
        colour: SERIES.find((s) => s.key === right).colour, shape: 'line' },
      { label: 'coherence of random words', colour: 'var(--squidink)', shape: 'dash' },
    ], { x0: 0, y0: -26, gap: 210, size: 10.5, cols: 2 });

    const agree = bestCoh.k === bestOther.k;
    readout.html(
      `<div>Coherence peaks at <span class="bold">k = ${bestCoh.k}</span> `
      + `(<span class="value">${bestCoh.coherence.toFixed(4)}</span>, against a floor of `
      + `<span class="value">${bestCoh.coherence_floor.toFixed(4)}</span> for randomly chosen `
      + `words). Perplexity is best at <span class="bold">k = `
      + `${rows.reduce((p, c) => (c.perplexity < p.perplexity ? c : p), rows[0]).k}</span>.</div>`
      + `<div>${agree
        ? 'On this corpus the two criteria happen to agree, which is worth saying because they '
          + 'are measuring different things and often do not.'
        : 'The two criteria <span class="bold">do not pick the same model</span>. Perplexity is '
          + 'the likelihood of held-out words, and adding topics almost always buys some; '
          + 'coherence asks whether the words at the top of a topic belong together, and it '
          + 'stops improving long before perplexity does.'}</div>`
      + `<div>Against the eight hand-written categories, agreement peaks at `
      + `<span class="bold">k = ${rows.reduce((p, c) => (c.ari > p.ari ? c : p), rows[0]).k}</span> `
      + `with an adjusted Rand index of `
      + `<span class="value">${Math.max(...rows.map((r) => r.ari)).toFixed(4)}</span>, where 1 `
      + `would be a perfect recovery and 0 is what shuffling gives.</div>`
    );
  }

  render();
  return { render };
}
