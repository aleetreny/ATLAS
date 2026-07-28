/* Twenty years of daily returns, and four questions asked of them.
 *
 * The top panel never changes: it is the series, and the clustering is visible
 * without any statistics at all. The panel underneath is where the argument
 * happens, and it deliberately shows four different things, because the point
 * is that the same data answers "can you forecast this" with no and with yes
 * depending on which function of it you ask about.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initReturnsScrolly(data) {
  const T = data.two_r2;
  const thin = data.returns.thin;
  const chart = makeChart('#returns-chart', {
    width: 640, height: 500, margin: { top: 44, right: 20, bottom: 52, left: 64 },
  });
  const { g, w, h, margin } = chart;
  const topH = Math.round(h * 0.44);
  const botY = topH + 84;
  const botH = h - botY;
  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');
  /* two lines of room above the lower panel, because one of the four captions
     is a sentence and a single line of it ran 95 units past the canvas */
  const subtitle = gBot.append('text').attr('class', 'chart-note')
    .attr('x', 0).attr('y', -32).style('font-size', '11.5px');

  /* the series, once */
  const xs = d3.scaleLinear().domain([0, thin.length - 1]).range([0, w]);
  const yv = d3.scaleLinear()
    .domain([d3.min(thin, (d) => d.r) * 1.05, d3.max(thin, (d) => d.r) * 1.05])
    .range([topH, 0]);
  const years = ['2000-01', '2004-01', '2008-01', '2012-01', '2016-01'].map((p) => {
    let bestI = 0;
    thin.forEach((d, i) => { if (d.t.slice(0, 7) <= p) bestI = i; });
    return bestI;
  });
  drawGrid(gTop, xs, yv, w, topH, { xValues: years, yTicks: 5 });
  drawAxes(gTop, xs, yv, w, topH, {
    xValues: years, yTicks: 5, xFmt: (i) => thin[i].t.slice(0, 4),
  });
  axisLabels(gTop, w, topH, { y: 'daily return, percent', leftBudget: margin.left });
  gTop.append('path').datum(thin).attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 0.7).attr('opacity', 0.9)
    .attr('d', d3.line().x((d, i) => xs(i)).y((d) => yv(d.r)));
  const mark = gTop.append('g');

  function histogram() {
    const vals = data.returns.y;
    const lo = -6;
    const hi = 6;
    const bins = 60;
    const counts = new Array(bins).fill(0);
    let outside = 0;
    vals.forEach((v) => {
      const k = Math.floor(((v - lo) / (hi - lo)) * bins);
      if (k < 0 || k >= bins) outside += 1;
      else counts[k] += 1;
    });
    const sd = Math.sqrt(vals.reduce((a, b) => a + b * b, 0) / vals.length);
    const width = (hi - lo) / bins;
    const normal = counts.map((_, i) => {
      const c = lo + (i + 0.5) * width;
      return (vals.length * width) * Math.exp(-(c * c) / (2 * sd * sd))
        / (sd * Math.sqrt(2 * Math.PI));
    });
    const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
    const y = d3.scaleLinear().domain([0.5, Math.max(...counts) * 1.15])
      .range([botH, 0]).clamp(true);
    gBot.selectAll('*:not(text.chart-note)').remove();
    drawAxes(gBot, x, y, w, botH, { xValues: [-6, -3, 0, 3, 6], yTicks: 4 });
    axisLabels(gBot, w, botH, {
      x: 'daily return, percent', y: 'days', leftBudget: margin.left, xOffset: 38,
    });
    gBot.selectAll('rect.bin').data(counts).join('rect').attr('class', 'bin')
      .attr('x', (d, i) => x(lo + i * width) + 0.5)
      .attr('width', Math.max(1, w / bins - 1))
      .attr('y', (d) => y(Math.max(0.5, d)))
      .attr('height', (d) => botH - y(Math.max(0.5, d)))
      .attr('fill', 'var(--anchor)').attr('opacity', 0.75);
    gBot.append('path').datum(normal).attr('fill', 'none').attr('stroke', 'var(--cosmos)')
      .attr('stroke-width', 2)
      .attr('d', d3.line().x((d, i) => x(lo + (i + 0.5) * width)).y((d) => y(Math.max(0.5, d))));
    wrapLabel(subtitle, `in red, the normal with the same standard deviation. `
      + `${outside} days fall outside this axis entirely`, w - 8);
    subtitle.raise();
  }

  function correlogram(which) {
    const series = {
      returns: [['the return itself', T.acf_return, 'var(--anchor)']],
      abs: [['the absolute return', T.acf_abs, 'var(--primary)']],
      both: [['the return itself', T.acf_return, 'var(--anchor)'],
        ['the absolute return', T.acf_abs, 'var(--primary)']],
    }[which];
    const n = T.acf_abs.length - 1;
    const x = d3.scaleLinear().domain([0.4, n + 0.6]).range([0, w]);
    const y = d3.scaleLinear().domain([-0.12, 0.42]).range([botH, 0]);
    gBot.selectAll('*:not(text.chart-note)').remove();
    drawGrid(gBot, x, y, w, botH, { xValues: [1, 10, 20, 30, 40], yValues: [0, 0.2, 0.4] });
    drawAxes(gBot, x, y, w, botH, {
      xValues: [1, 10, 20, 30, 40], yValues: [-0.1, 0, 0.1, 0.2, 0.3, 0.4],
      yFmt: d3.format('.1f'),
    });
    axisLabels(gBot, w, botH, {
      x: 'lag, in trading days', y: 'autocorrelation', leftBudget: margin.left, xOffset: 38,
    });
    const bw = series.length > 1 ? 2.6 : 5;
    series.forEach(([, vals, colour], si) => {
      gBot.selectAll(`line.bar${si}`).data(vals.slice(1)).join('line')
        .attr('class', `bar${si}`).attr('stroke-width', bw).attr('stroke', colour)
        .attr('x1', (d, i) => x(i + 1) + (series.length > 1 ? (si ? 1.6 : -1.6) : 0))
        .attr('x2', (d, i) => x(i + 1) + (series.length > 1 ? (si ? 1.6 : -1.6) : 0))
        .attr('y1', y(0)).attr('y2', (d) => y(d));
    });
    [T.band, -T.band].forEach((b) => {
      gBot.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(b)).attr('y2', y(b))
        .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '4 4').attr('opacity', 0.55);
    });
    wrapLabel(subtitle, `${series.map(([name]) => name).join(' against ')}; `
      + `dashed is the +-${T.band.toFixed(4)} band`, w - 8);
    subtitle.raise();
  }

  const steps = {
    0: () => {
      mark.selectAll('*').remove();
      const worst = data.garch.worst_day;
      let idx = 0;
      thin.forEach((d, i) => { if (d.t <= worst.label) idx = i; });
      mark.append('line').attr('x1', xs(idx)).attr('x2', xs(idx))
        .attr('y1', 0).attr('y2', topH).attr('stroke', 'var(--cosmos)')
        .attr('stroke-width', 1.4).attr('opacity', 0.8);
      mark.append('text').attr('class', 'chart-note').attr('x', xs(idx) - 6).attr('y', 12)
        .attr('text-anchor', 'end').style('font-size', '11px')
        .text(`${worst.label}: ${worst.ret}%`);
      histogram();
      wrapLabel(title, `${data.meta.ret_n.toLocaleString('en-US')} daily returns, `
        + `${data.meta.sp500.start} to ${data.meta.sp500.end}`, w - 8);
    },
    1: () => { correlogram('returns'); wrapLabel(title, 'is the sign of tomorrow in the past?', w - 8); },
    2: () => { correlogram('abs'); wrapLabel(title, 'is the size of tomorrow in the past?', w - 8); },
    3: () => { correlogram('both'); wrapLabel(title, 'the same forty lags, two questions', w - 8); },
  };

  steps[0]();
  scrolly('#returns-scrolly .step', steps);
  return { render: (i) => steps[i]() };
}
