/* Four views of one series, and the correlogram that goes with each.
 *
 * The point of the panel underneath is that it disagrees with the tests. A
 * unit root test asks whether the level wanders off; a fixed twelve month
 * cycle does not wander off, so a seasonal series can pass both tests on this
 * page while its correlation with itself a year ago is still enormous, and the
 * only way to see that is to draw it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { acf } from '../../assets/js/series.js';
import { scrolly } from '../../assets/js/scrolly.js';

const ORDER = ['raw', 'd1', 'dS', 'd1dS'];

export function initStationarityScrolly(data) {
  const C = data.stationarity;
  const rows = Object.fromEntries(C.co2.map((r) => [r.name, r]));
  const total = data.meta.co2_n;
  const labels = data.series.co2.labels;
  const NLAG = 36;

  const chart = makeChart('#stationarity-chart', {
    width: 640, height: 500, margin: { top: 46, right: 20, bottom: 52, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const topH = Math.round(h * 0.50);
  const botY = topH + 78;
  const botH = h - botY;

  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle')
    .style('font-size', '12px');
  const subtitle = gBot.append('text').attr('class', 'chart-note')
    .attr('x', 0).attr('y', -14).style('font-size', '11.5px');

  function render(name) {
    const cur = C.curves[name];
    const y = cur.y;
    const off = total - y.length;
    const x = d3.scaleLinear().domain([0, y.length - 1]).range([0, w]);
    const lo = d3.min(y);
    const hi = d3.max(y);
    const pad = (hi - lo) * 0.08 || 1;
    const ys = d3.scaleLinear().domain([lo - pad, hi + pad]).range([topH, 0]);

    const years = [1960, 1970, 1980, 1990, 2000]
      .map((yr) => labels.findIndex((l) => l === `${yr}-01`) - off)
      .filter((i) => i >= 0 && i <= y.length - 1);

    drawGrid(gTop, x, ys, w, topH, { xValues: years, yTicks: 5 });
    drawAxes(gTop, x, ys, w, topH, {
      xValues: years, yTicks: 5,
      xFmt: (i) => labels[i + off].slice(0, 4),
    });
    axisLabels(gTop, w, topH, { y: 'parts per million', leftBudget: margin.left });

    let path = gTop.select('path.series');
    if (path.empty()) {
      path = gTop.append('path').attr('class', 'series').attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4);
    }
    path.datum(y).transition('shape').duration(650)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => ys(d)));

    /* the correlogram, computed here rather than read from the file: the guard
       compares it against the generator's, so it has to be this page's own */
    const a = acf(y, NLAG);
    const band = 1.96 / Math.sqrt(y.length);
    const xb = d3.scaleLinear().domain([0.4, NLAG + 0.6]).range([0, w]);
    const yb = d3.scaleLinear().domain([Math.min(-0.35, d3.min(a) - 0.05), 1]).range([botH, 0]);

    drawAxes(gBot, xb, yb, w, botH, {
      xValues: [1, 6, 12, 18, 24, 30, 36], yValues: [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
      yFmt: d3.format('.1f'),
    });
    axisLabels(gBot, w, botH, {
      x: 'lag, in months', y: 'autocorrelation', leftBudget: margin.left, xOffset: 38,
    });

    const bars = gBot.selectAll('line.bar').data(a.slice(1));
    bars.join(
      (enter) => enter.append('line').attr('class', 'bar').attr('stroke-width', 3.4),
      (update) => update
    )
      .attr('x1', (d, i) => xb(i + 1)).attr('x2', (d, i) => xb(i + 1))
      .attr('y1', yb(0))
      .attr('stroke', (d, i) => ((i + 1) % 12 === 0 ? 'var(--cosmos)' : 'var(--anchor)'))
      .transition('bars').duration(650)
      .attr('y2', (d) => yb(d));

    let bandG = gBot.select('g.band');
    if (bandG.empty()) bandG = gBot.append('g').attr('class', 'band');
    bandG.selectAll('line').data([band, -band]).join('line')
      .attr('x1', 0).attr('x2', w)
      .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '4 4').attr('opacity', 0.55)
      .transition('bandmove').duration(650)
      .attr('y1', (d) => yb(d)).attr('y2', (d) => yb(d));

    const row = rows[name];
    wrapLabel(title, `${row.label}: ${row.n} observations, `
      + `standard deviation ${row.sd}`, w - 8);
    /* both facts go in the one line above the panel: inside it, the only empty
       band is along zero, and that is where the bars are anchored */
    subtitle.text(`lag 12 in red: ${a[12].toFixed(4)}. `
      + `dashed: the +-1.96/sqrt(n) band, +-${band.toFixed(3)}`);
  }

  render('raw');
  const handlers = {};
  ORDER.forEach((name, i) => {
    handlers[String(i)] = () => render(name);
  });
  scrolly('#stationarity-scrolly .step', handlers);

  return { render, order: ORDER };
}
