/* One window through the interpretable stack, and the answer key underneath.
 *
 * The trend blocks and the seasonal blocks are drawn separately because that is
 * the only claim the interpretable basis makes: not that it forecasts better,
 * but that its two halves mean something. Whether they do is a question with an
 * answer here, since the series was generated from components that were written
 * down before it existed, and both are on the chart.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initStackScrolly(data) {
  const ex = data.bases.examples[0];
  const L = data.meta.lookback;
  const H = data.meta.horizon;
  const chart = makeChart('#stack-chart', {
    width: 640, height: 460, margin: { top: 46, right: 24, bottom: 50, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const topH = Math.round(h * 0.54);
  const botY = topH + 74;
  const botH = h - botY;
  const gTop = g.append('g');
  const gBot = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const subtitle = gBot.append('text').attr('class', 'chart-note')
    .attr('x', 0).attr('y', -14).style('font-size', '11.5px');

  const x = d3.scaleLinear().domain([0, L + H - 1]).range([0, w]);
  const all = ex.context.concat(ex.truth, ex.forecast);
  const yTop = d3.scaleLinear()
    .domain([d3.min(all) - 1, d3.max(all) + 1]).range([topH, 0]);
  drawGrid(gTop, x, yTop, w, topH, { xValues: [0, 12, 24, L, L + H - 1], yTicks: 4 });
  drawAxes(gTop, x, yTop, w, topH, { xValues: [0, 12, 24, L, L + H - 1], yTicks: 4 });
  axisLabels(gTop, w, topH, { y: 'the series', leftBudget: margin.left });
  gTop.append('path').datum(ex.context).attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.5)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => yTop(d)));
  gTop.selectAll('circle.truth').data(ex.truth).join('circle').attr('class', 'truth')
    .attr('cx', (d, i) => x(L + i)).attr('cy', (d) => yTop(d)).attr('r', 3)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
  gTop.append('line').attr('x1', x(L - 0.5)).attr('x2', x(L - 0.5))
    .attr('y1', 0).attr('y2', topH).attr('stroke', 'var(--squidink)')
    .attr('stroke-dasharray', '4 4').attr('opacity', 0.6);
  gTop.append('text').attr('class', 'chart-note').attr('x', x(L - 0.5) - 5).attr('y', 12)
    .attr('text-anchor', 'end').style('font-size', '11px').text('the lookback ends');
  const layerTop = gTop.append('g');

  const parts = ex.trend_hat.concat(ex.season_hat, ex.trend_true, ex.season_true);
  const yBot = d3.scaleLinear()
    .domain([d3.min(parts) * 1.1 - 1, d3.max(parts) * 1.1 + 1]).range([botH, 0]);
  const layerBot = gBot.append('g');

  function bottom(series, note) {
    layerBot.selectAll('*').remove();
    drawGrid(gBot, x, yBot, w, botH, { xValues: [0, 12, 24, L, L + H - 1], yTicks: 3 });
    drawAxes(gBot, x, yBot, w, botH, { xValues: [0, 12, 24, L, L + H - 1], yTicks: 3 });
    axisLabels(gBot, w, botH, {
      x: 'months, the last twelve of them unseen', y: 'the part',
      leftBudget: margin.left, xOffset: 36,
    });
    series.forEach(([vals, colour, dash]) => {
      layerBot.append('path').datum(vals).attr('fill', 'none').attr('stroke', colour)
        .attr('stroke-width', dash ? 1.6 : 2.4).attr('stroke-dasharray', dash || null)
        .attr('d', d3.line().x((d, i) => x(L + i)).y((d) => yBot(d)));
    });
    subtitle.text(note);
    subtitle.raise();
  }

  const steps = {
    0: () => {
      layerTop.selectAll('*').remove();
      bottom([], 'the network has not been asked anything yet');
      wrapLabel(title, `one held out window from the "${ex.family.replace(/_/g, ' ')}" `
        + `family: ${L} months in, ${H} months to produce`, w - 8);
    },
    1: () => {
      layerTop.selectAll('*').remove();
      bottom([[ex.trend_hat, 'var(--primary)', null]],
        'what the polynomial blocks produced for the next twelve months');
      wrapLabel(title, 'the first blocks can only make polynomials', w - 8);
    },
    2: () => {
      layerTop.selectAll('*').remove();
      bottom([[ex.trend_hat, 'var(--primary)', null],
        [ex.season_hat, 'var(--aqua)', null]],
      'and what the Fourier blocks made of the residual left behind');
      wrapLabel(title, 'the later blocks only see what is left over', w - 8);
    },
    3: () => {
      layerTop.selectAll('*').remove();
      layerTop.append('path').datum(ex.forecast).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6)
        .attr('d', d3.line().x((d, i) => x(L + i)).y((d) => yTop(d)));
      layerTop.selectAll('circle').data(ex.forecast).join('circle')
        .attr('cx', (d, i) => x(L + i)).attr('cy', (d) => yTop(d)).attr('r', 3)
        .attr('fill', 'var(--primary)');
      bottom([[ex.trend_hat, 'var(--primary)', null],
        [ex.season_hat, 'var(--aqua)', null],
        [ex.trend_true, 'var(--primary)', '5 4'],
        [ex.season_true, 'var(--aqua)', '5 4']],
      'solid: what the blocks produced. dashed: the parts this series was built from');
      wrapLabel(title, 'the sum is the forecast, and the parts have an answer key', w - 8);
    },
  };

  steps[0]();
  scrolly('#stack-scrolly .step', steps);
  return { render: (i) => steps[i]() };
}
