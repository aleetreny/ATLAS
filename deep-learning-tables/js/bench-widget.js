/* The benchmark, plotted the way it should be read: cost against error.
 *
 * A leaderboard hides the axis that matters. Here fit time runs left to right
 * and held-out error runs bottom to top, so "better" is down and "cheaper" is
 * left, and the winner is whatever sits closest to the bottom-left corner. On
 * this data that corner is occupied by a gradient boosting model that finishes
 * in under a second.
 */
import { makeChart, drawGrid, axisLabels, tooltip, spread } from '../../assets/js/chart.js';

const NOTE = {
  ridge: 'A straight line. The floor everything else is measured against.',
  'mlp (256-128-64)': 'A plain feed-forward network on standardised columns. The strongest neural entry, and it is not close to boosting.',
  'neural additive model': 'One small network per feature, summed. Pays for readability with error.',
  tabnet: 'Attentive feature selection, purpose-built for tables. Note the spread across seeds.',
  'explainable boosting, additive': 'Shape functions fitted by boosting instead of by gradient descent. Same additive restriction as the NAM, much better fit.',
  'explainable boosting, +pairs': 'The same model allowed a budget of pairwise terms. Nearly closes the gap while staying readable.',
  'gradient boosting': 'Ordinary histogram boosting, scikit-learn defaults, 300 trees.',
};

/* Direct labels have to fit beside the plot, and "explainable boosting,
 * additive" does not. Abbreviate on the chart, keep the full name in the
 * tooltip and in the table below. */
const SHORT = {
  ridge: 'ridge',
  'mlp (256-128-64)': 'MLP',
  'neural additive model': 'NAM',
  tabnet: 'TabNet',
  'explainable boosting, additive': 'EBM, additive',
  'explainable boosting, +pairs': 'EBM, + pairs',
  'gradient boosting': 'boosting',
};

const FAMILY = (n) =>
  n.startsWith('gradient') ? 'trees' : n.startsWith('explainable') ? 'glass box' : n === 'ridge' ? 'linear' : 'neural';

const COLOUR = {
  trees: 'var(--cosmos)',
  'glass box': 'var(--seablue)',
  neural: 'var(--primary)',
  linear: 'var(--stone)',
};

export function initBenchWidget(data) {
  const rows = data.benchmark.map((r) => ({ ...r, family: FAMILY(r.name) }));
  let logScale = true;

  const { g, w, h } = makeChart('#bench-chart', {
    width: 700,
    height: 430,
    margin: { top: 26, right: 132, bottom: 56, left: 80 },
  });
  const tip = tooltip();

  const yMax = d3.max(rows, (r) => r.rmse + r.rmse_sd) * 1.06;
  const y = d3.scaleLinear().domain([400, yMax]).range([h, 0]);

  const dotsG = g.append('g');
  const labelsG = g.append('g');

  function xScale() {
    /* Timings run from 0.02s to 49s. On a linear axis only ridge and boosting
     * sit on the left edge, three models bunch around ten to sixteen seconds,
     * and TabNet and the pairwise EBM trail off alone, which is itself worth
     * seeing once. */
    return logScale
      ? d3.scaleLog().domain([0.015, 60]).range([0, w])
      /* A domain starting at exactly 0 puts the "0ms" label directly under the
       * y axis, where it crosses the bottom y tick ("$400"). A sliver of
       * negative padding moves it clear without moving any data point
       * perceptibly. */
      : d3.scaleLinear().domain([-1.6, d3.max(rows, (r) => r.seconds) * 1.08]).range([0, w]);
  }

  function render() {
    const x = xScale();
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });

    g.selectAll('g.axis.x')
      .data([0])
      .join((e) => e.append('g').attr('class', 'axis x'))
      .attr('transform', `translate(0,${h})`)
      .transition()
      .duration(400)
      .call(
        d3
          .axisBottom(x)
          /* A log scale ignores the tick count and emits every decade plus its
           * submultiples, which at this width collided into an unreadable
           * smear. Name the ticks explicitly instead. */
          .tickValues(logScale ? [0.02, 0.1, 0.5, 2, 10, 40] : null)
          .ticks(logScale ? undefined : 5)
          .tickFormat((v) => (v >= 1 ? `${v}s` : `${Math.round(v * 1000)}ms`))
          .tickSizeOuter(0)
      );
    g.selectAll('g.axis.y')
      .data([0])
      .join((e) => e.append('g').attr('class', 'axis y'))
      .call(d3.axisLeft(y).ticks(5).tickFormat((v) => '$' + d3.format(',')(v)).tickSizeOuter(0));
    axisLabels(g, w, h, { x: 'fit time, wall clock', y: 'held-out RMSE' });
    /* axisLabels parks the y label 44px out, which is inside a "$1,000" tick.
     * Push it clear of the widest label this axis can produce. */
    g.select('text.axis-label.y').attr('y', -60);

    const px = (r) => x(Math.max(logScale ? 0.02 : 0, r.seconds));

    dotsG
      .selectAll('line.err')
      .data(rows, (r) => r.name)
      .join((e) => e.append('line').attr('class', 'err').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-opacity', 0.5))
      .transition()
      .duration(400)
      .attr('x1', px)
      .attr('x2', px)
      .attr('y1', (r) => y(r.rmse - r.rmse_sd))
      .attr('y2', (r) => y(r.rmse + r.rmse_sd));

    dotsG
      .selectAll('circle')
      .data(rows, (r) => r.name)
      .join((e) =>
        e
          .append('circle')
          .attr('r', 7)
          .attr('stroke', 'var(--squidink)')
          .attr('stroke-width', 1.4)
          .attr('cy', (r) => y(r.rmse))
          .attr('cx', px)
          .on('mousemove', (event, r) =>
            tip.show(
              `<span style="color:var(--smile)">${r.name}</span><br/>` +
                `RMSE $${r.rmse.toLocaleString('en-US')} ± ${r.rmse_sd}<br/>` +
                `${r.seconds < 0.02 ? 'under 20ms to fit, drawn at the axis minimum' : `${r.seconds}s to fit`}<br/>` +
                `<span style="opacity:.75">${NOTE[r.name] ?? ''}</span>`,
              event
            )
          )
          .on('mouseleave', () => tip.hide())
      )
      .attr('fill', (r) => COLOUR[r.family])
      .transition()
      .duration(400)
      .attr('cx', px)
      .attr('cy', (r) => y(r.rmse));

    /* Direct labels beat a legend, but only once they stop colliding. */
    const placed = spread(
      rows.map((r) => ({ name: r.name, y: y(r.rmse), colour: COLOUR[r.family] })),
      15
    );
    labelsG
      .selectAll('text')
      .data(placed, (d) => d.name)
      .join((e) =>
        e
          .append('text')
          .attr('class', 'chart-annotation')
          .attr('x', w + 12)
          .style('font-size', '0.6rem')
          .style('text-transform', 'none')
      )
      /* The linear family is drawn in --stone so its dot reads as a baseline,
       * but --stone on --paper is 1.3:1 and the direct label vanished. Ink the
       * text even when the mark stays pale. */
      .attr('fill', (d) => (d.colour === 'var(--stone)' ? 'var(--squidink)' : d.colour))
      .transition()
      .duration(400)
      .attr('y', (d) => d.y + 4)
      .text((d) => SHORT[d.name] ?? d.name);

    labelsG
      .selectAll('line.leader')
      .data(placed, (d) => d.name)
      .join((e) => e.append('line').attr('class', 'leader').attr('stroke', 'var(--stone)').attr('stroke-width', 1))
      .transition()
      .duration(400)
      .attr('x1', (d) => px(rows.find((r) => r.name === d.name)) + 9)
      .attr('x2', w + 8)
      .attr('y1', (d) => y(rows.find((r) => r.name === d.name).rmse))
      .attr('y2', (d) => d.y);
  }

  const best = rows.reduce((a, b) => (a.rmse <= b.rmse ? a : b));
  const bestNeural = rows.filter((r) => r.family === 'neural').reduce((a, b) => (a.rmse <= b.rmse ? a : b));
  const glass = rows.find((r) => r.name === 'explainable boosting, +pairs');

  document.querySelector('#bench-readout').innerHTML =
    `<div class="slider-label">Best of all: <span class="value" style="color:var(--cosmos)">${best.name}</span> at ` +
    `$${best.rmse.toLocaleString('en-US')} in ${best.seconds}s. Best neural: ` +
    `<span class="value" style="color:var(--primary)">${bestNeural.name}</span> at $${bestNeural.rmse.toLocaleString('en-US')}, ` +
    `<span class="value">${Math.round(((bestNeural.rmse - best.rmse) / best.rmse) * 100)}%</span> worse and ` +
    `<span class="value">${Math.round(bestNeural.seconds / best.seconds)}×</span> slower.</div>` +
    `<div class="slider-label" style="margin-top:.5rem">The interesting point is the one in between: ` +
    `<span class="value" style="color:var(--seablue)">${glass.name}</span> lands at $${glass.rmse.toLocaleString('en-US')}, ` +
    `within <span class="value">${Math.round(((glass.rmse - best.rmse) / best.rmse) * 100)}%</span> of the winner while still being a model you can read end to end.</div>`;

  const btn = document.querySelector('#bench-scale');
  btn.addEventListener('click', () => {
    logScale = !logScale;
    btn.textContent = logScale ? 'log time' : 'linear time';
    render();
  });

  render();
}
