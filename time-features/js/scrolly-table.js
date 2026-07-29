/* The series turning into a table, one group of columns at a time.
 *
 * Every column here is a function of observations strictly before its own row,
 * which is the only rule that matters and the easiest one to break by accident.
 * The heatmap is drawn from the same arithmetic the generator used, computed in
 * the page, so the picture and the fitted numbers cannot drift apart.
 */
import { makeChart, drawAxes, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

const GROUPS = {
  lags: ['lag1', 'lag2', 'lag3', 'lag12'],
  calendar: ['sin1', 'cos1', 'sin2', 'cos2', 'month'],
  rolling: ['mean3', 'mean12', 'sd12', 'diff1', 'diff12'],
  trend: ['t'],
};
const SHOW = 16;

function mean(a) {
  return a.reduce((s, v) => s + v, 0) / a.length;
}

export function buildRow(y, i, month) {
  const f = {};
  f.lag1 = y[i - 1];
  f.lag2 = y[i - 2];
  f.lag3 = y[i - 3];
  f.lag12 = y[i - 12];
  f.month = month;
  f.sin1 = Math.sin((2 * Math.PI * month) / 12);
  f.cos1 = Math.cos((2 * Math.PI * month) / 12);
  f.sin2 = Math.sin((4 * Math.PI * month) / 12);
  f.cos2 = Math.cos((4 * Math.PI * month) / 12);
  const w3 = y.slice(i - 3, i);
  const w12 = y.slice(i - 12, i);
  f.mean3 = mean(w3);
  f.mean12 = mean(w12);
  const m12 = f.mean12;
  f.sd12 = Math.sqrt(mean(w12.map((v) => (v - m12) ** 2)));
  f.diff1 = y[i - 1] - y[i - 2];
  f.diff12 = y[i - 1] - y[i - 13];
  f.t = i;
  return f;
}

export function initTableScrolly(data) {
  const y = data.series.y;
  const labels = data.series.labels;
  const chart = makeChart('#table-chart', {
    width: 640, height: 470, margin: { top: 40, right: 22, bottom: 30, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const topH = 120;
  const botY = topH + 74;
  const botH = h - botY;
  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px');

  const start = y.length - SHOW;
  const rows = d3.range(start, y.length).map((i) => ({
    i, label: labels[i], f: buildRow(y, i, +labels[i].slice(5, 7) - 1), target: y[i],
  }));

  /* the tail of the series, with the rows of the table marked on it */
  const xs = d3.scaleLinear().domain([start - 26, y.length - 1]).range([0, w]);
  const ys = d3.scaleLinear()
    .domain([d3.min(y.slice(start - 26)) - 1, d3.max(y.slice(start - 26)) + 1])
    .range([topH, 0]);
  drawAxes(gTop, xs, ys, w, topH, {
    xValues: d3.range(start - 24, y.length, 12), yTicks: 3,
    xFmt: (i) => labels[i].slice(0, 4),
  });
  axisLabels(gTop, w, topH, { y: 'ppm', leftBudget: margin.left });
  gTop.append('path').datum(d3.range(start - 26, y.length)).attr('fill', 'none')
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
    .attr('d', d3.line().x((i) => xs(i)).y((i) => ys(y[i])));
  gTop.selectAll('circle').data(rows).join('circle')
    .attr('cx', (d) => xs(d.i)).attr('cy', (d) => ys(d.target)).attr('r', 3)
    .attr('fill', 'var(--primary)');

  const cellH = Math.floor(botH / (SHOW + 1));

  function render(cols, note) {
    gBot.selectAll('*').remove();
    const names = cols.concat(['target']);
    const cw = Math.min(46, (w - 58) / names.length);
    const x0 = 58;
    const scales = {};
    names.forEach((c) => {
      const vals = rows.map((r) => (c === 'target' ? r.target : r.f[c]));
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      scales[c] = d3.scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([0, 1]);
    });
    gBot.selectAll('text.rowlab').data(rows).join('text')
      .attr('class', 'chart-note rowlab')
      .attr('x', x0 - 6).attr('y', (d, i) => (i + 1) * cellH + cellH * 0.72)
      .attr('text-anchor', 'end').style('font-size', '9.5px').text((d) => d.label);
    names.forEach((c, ci) => {
      gBot.append('text').attr('class', 'chart-note')
        .attr('x', x0 + ci * cw + cw / 2).attr('y', cellH * 0.75)
        .attr('text-anchor', 'middle').style('font-size', '9.5px')
        .attr('fill', c === 'target' ? 'var(--cosmos)' : 'var(--squidink)')
        .text(c === 'target' ? 'y' : c);
      gBot.selectAll(`rect.c${ci}`).data(rows).join('rect').attr('class', `c${ci}`)
        .attr('x', x0 + ci * cw + 1).attr('y', (d, i) => (i + 1) * cellH + 1)
        .attr('width', cw - 2).attr('height', cellH - 2)
        /* real hex at both ends: d3.interpolateRgb cannot parse a
           `var(--cosmos)`, it fails silently, and every cell comes out the same
           pale colour, which looks like a scale with no contrast rather than a
           broken one */
        .attr('fill', (d) => {
          const v = scales[c](c === 'target' ? d.target : d.f[c]);
          return c === 'target'
            ? d3.interpolateRgb('#f7dfe2', '#a02040')(v)
            : d3.interpolateRgb('#eef0f0', '#853c43')(v);
        });
    });
    gBot.append('text').attr('class', 'chart-note')
      .attr('x', 0).attr('y', -12).style('font-size', '11.5px').text(note);
  }

  const steps = {
    0: () => {
      render([], `the last ${SHOW} months, one row each, and nothing to learn from yet`);
      wrapLabel(title, 'one row per month, and the target is that month', w - 8);
    },
    1: () => {
      render(GROUPS.lags, 'four columns, each one an earlier observation');
      wrapLabel(title, 'the past becomes columns', w - 8);
    },
    2: () => {
      render(GROUPS.lags.concat(GROUPS.calendar), 'plus the month, and the first two '
        + 'harmonics of the year');
      wrapLabel(title, 'the calendar becomes columns too', w - 8);
    },
    3: () => {
      render(GROUPS.lags.concat(GROUPS.calendar, GROUPS.rolling, GROUPS.trend),
        `${GROUPS.lags.length + GROUPS.calendar.length + GROUPS.rolling.length + 1} columns, `
        + 'every one of them computed from strictly before its own row');
      wrapLabel(title, 'and anything else you can compute without looking down', w - 8);
    },
  };

  steps[0]();
  scrolly('#table-scrolly .step', steps);
  return { render: (i) => steps[i](), rows };
}
