/* How empty the table is, and what a threshold costs.
 *
 * The catalogue sorted by how many people rated each book, on log axes, with
 * the running share of all ratings on top of it. The slider is the decision
 * every recommender makes before any modelling: how many ratings a book needs
 * before it is allowed to exist. Both consequences are on the chart, because
 * the cheap version of this decision only ever reports one of them.
 *
 * The counts arrive whole, so everything here is computed rather than read.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initHoleWidget(data) {
  const counts = data.shape.counts_sorted;
  const total = counts.reduce((a, b) => a + b, 0);
  const cum = new Float64Array(counts.length);
  let run = 0;
  counts.forEach((c, i) => { run += c; cum[i] = run / total; });

  const slider = document.querySelector('#hole-slider');
  const valueLabel = document.querySelector('#hole-value');
  const readout = document.querySelector('#hole-readout');

  const chart = makeChart('#hole-chart', {
    width: 640, height: 380, margin: { top: 42, right: 62, bottom: 58, left: 66 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const x = d3.scaleLog().domain([1, counts.length]).range([0, w]);
  const y = d3.scaleLog().domain([Math.max(1, counts[counts.length - 1]),
    counts[0] * 1.3]).range([h, 0]);
  const y2 = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  const xTicks = [1, 10, 100, 1000, 10000].filter((v) => v <= counts.length);
  const yTicks = [1, 10, 100, 1000, 10000].filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);

  /* the curve is 10,000 points and only 640 of them can land on separate
     pixels, so it is thinned on a log grid: same shape, a tenth of the nodes */
  const sample = [];
  for (let e = 0; e <= 400; e++) {
    const k = Math.round(10 ** (e / 400 * Math.log10(counts.length)));
    if (k >= 1 && (sample.length === 0 || sample[sample.length - 1] !== k)) sample.push(k);
  }

  function stats(t) {
    /* counts is sorted descending, so the last index with count >= t is found
       by walking from the end of the sorted array: a binary search on a
       reversed comparison, written out because it is three lines */
    let lo = 0;
    let hi = counts.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (counts[mid] >= t) lo = mid + 1; else hi = mid;
    }
    const kept = lo;
    return { kept, share: kept ? cum[kept - 1] : 0 };
  }

  function render() {
    const t = Number(slider.value);
    const { kept, share } = stats(t);
    valueLabel.textContent = t === 0 ? 'one rating' : `${t} ratings`;
    clear();
    drawGrid(g, x, y, w, h, { xValues: xTicks, yValues: yTicks });

    plot.append('path')
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2)
      .attr('d', d3.line().x((k) => x(k)).y((k) => y(Math.max(counts[k - 1], 1)))(sample));

    plot.append('path')
      .attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6)
      .attr('stroke-dasharray', '5 3')
      .attr('d', d3.line().x((k) => x(k)).y((k) => y2(cum[k - 1]))(sample));

    if (kept > 0 && kept < counts.length) {
      plot.append('line').attr('x1', x(kept)).attr('x2', x(kept))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
    }

    drawAxes(g, x, y, w, h, { xValues: xTicks, yValues: yTicks, xFmt: d3.format(','), yFmt: d3.format(',') });
    axisLabels(g, w, h, { x: 'books, most rated first', y: 'ratings the book has' });

    /* the right hand axis exists because the dashed series lives on it: a
       second scale without its own axis is decoration */
    const ax2 = g.append('g').attr('class', 'axis y2')
      .attr('transform', `translate(${w},0)`)
      .call(d3.axisRight(y2).ticks(5).tickFormat(d3.format('.0%')));
    ax2.selectAll('text').style('fill', 'var(--anchor)');
    g.append('text').attr('class', 'chart-note')
      .attr('transform', `translate(${w + 52},${h / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle').style('font-size', '11.5px')
      .style('fill', 'var(--anchor)')
      .text('share of all ratings');

    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text(`${counts.length.toLocaleString('en-US')} books, ${total.toLocaleString('en-US')} ratings between them`);

    const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
    readout.innerHTML = t === 0
      ? `Nothing is thrown away: all <span class="value">${counts.length.toLocaleString('en-US')}</span> `
        + `books, <span class="value">100%</span> of the ratings. The thinnest book in the catalogue `
        + `has <span class="value">${counts[counts.length - 1]}</span> ratings, and the fattest has `
        + `<span class="value">${counts[0].toLocaleString('en-US')}</span>.`
      : `Books with at least <span class="value">${t}</span> ratings: `
        + `<span class="value">${kept.toLocaleString('en-US')}</span> of `
        + `${counts.length.toLocaleString('en-US')}, which is <span class="value">${pct(kept / counts.length)}</span> `
        + `of the catalogue and <span class="value">${pct(share, 2)}</span> of the ratings. `
        + `Throwing out ${pct(1 - kept / counts.length)} of the books costs `
        + `<span class="value">${pct(1 - share, 2)}</span> of the data, and that asymmetry is the `
        + `whole shape of this problem.`;
  }

  slider.addEventListener('input', render);
  render();
  return { stats, render };
}
