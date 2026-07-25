/* What happens when you bury the signal in junk.
 *
 * Nine real columns, then 10, 30 and 60 columns of pure standard normal noise
 * added on. The nine real ones never change, so a model that could ignore the
 * new columns entirely would draw a flat line. Boosting nearly does. The network
 * does not, because a dense first layer has to actively learn to zero out
 * hundreds of weights, and it never fully manages it.
 */
import { makeChart, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initJunkWidget(data) {
  const rows = data.junk;
  const SERIES = [
    { key: 'gbm', label: 'gradient boosting', colour: 'var(--cosmos)' },
    { key: 'mlp', label: 'neural network', colour: 'var(--primary)' },
  ];

  const { g, w, h } = makeChart('#junk-chart', {
    width: 660,
    height: 380,
    margin: { top: 30, right: 168, bottom: 56, left: 70 },
  });

  const x = d3.scalePoint().domain(rows.map((r) => r.extra)).range([0, w]).padding(0.12);
  const y = d3
    .scaleLinear()
    .domain([450, d3.max(rows, (r) => Math.max(r.gbm, r.mlp)) * 1.07])
    .range([h, 0]);

  /* scalePoint has no .ticks(), so the vertical rules are drawn by hand. */
  const grid = g.insert('g', ':first-child').attr('class', 'grid');
  grid
    .selectAll('line.v')
    .data(rows)
    .join('line')
    .attr('class', 'v')
    .attr('x1', (r) => x(r.extra))
    .attr('x2', (r) => x(r.extra))
    .attr('y1', 0)
    .attr('y2', h);
  grid
    .selectAll('line.hz')
    .data(y.ticks(5))
    .join('line')
    .attr('class', 'hz')
    .attr('x1', 0)
    .attr('x2', w)
    .attr('y1', (d) => y(d))
    .attr('y2', (d) => y(d));

  g.append('g')
    .attr('class', 'axis x')
    .attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickFormat((d) => (d === 0 ? 'none' : `+${d}`)).tickSizeOuter(0));
  g.append('g')
    .attr('class', 'axis y')
    .call(d3.axisLeft(y).ticks(5).tickFormat((v) => '$' + d3.format(',')(v)).tickSizeOuter(0));
  axisLabels(g, w, h, { x: 'columns of pure noise added to the nine real ones', y: 'held-out RMSE' });

  const marker = g
    .append('line')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', '4 4')
    .attr('y1', -6)
    .attr('y2', h);

  const line = d3
    .line()
    .x((r) => x(r.extra))
    .y((r) => y(r.value));

  SERIES.forEach((s) => {
    const pts = rows.map((r) => ({ extra: r.extra, value: r[s.key] }));
    g.append('path')
      .datum(pts)
      .attr('fill', 'none')
      .attr('stroke', s.colour)
      .attr('stroke-width', 3)
      .attr('d', line);
    g.append('g')
      .selectAll('circle')
      .data(pts)
      .join('circle')
      .attr('class', `dot-${s.key}`)
      .attr('cx', (p) => x(p.extra))
      .attr('cy', (p) => y(p.value))
      .attr('r', 5)
      .attr('fill', s.colour)
      .attr('stroke', 'var(--white)')
      .attr('stroke-width', 1.5);
    g.append('text')
      .attr('class', 'chart-annotation')
      .attr('x', w + 12)
      .attr('y', y(pts[pts.length - 1].value) + 4)
      .attr('fill', s.colour)
      .style('font-size', '0.62rem')
      .style('text-transform', 'none')
      .text(s.label);
  });

  const slider = document.querySelector('#junk-slider');
  const readout = document.querySelector('#junk-readout');

  function render() {
    const i = +slider.value;
    const r = rows[i];
    marker.attr('x1', x(r.extra)).attr('x2', x(r.extra));
    SERIES.forEach((s) => {
      g.selectAll(`circle.dot-${s.key}`).attr('r', (p) => (p.extra === r.extra ? 8 : 5));
    });

    const base = rows[0];
    const gPct = ((r.gbm - base.gbm) / base.gbm) * 100;
    const mPct = ((r.mlp - base.mlp) / base.mlp) * 100;
    readout.innerHTML =
      `<div class="slider-label">With <span class="value">${r.extra === 0 ? 'no' : r.extra}</span> junk column${r.extra === 1 ? '' : 's'}: ` +
      `boosting <span class="value" style="color:var(--cosmos)">$${r.gbm.toLocaleString('en-US')}</span> ` +
      `(${gPct >= 0 ? '+' : ''}${gPct.toFixed(1)}%), network ` +
      `<span class="value" style="color:var(--primary)">$${r.mlp.toLocaleString('en-US')}</span> ` +
      `(${mPct >= 0 ? '+' : ''}${mPct.toFixed(1)}%).</div>` +
      (r.extra === 0
        ? `<div class="slider-label" style="margin-top:.5rem">Move the slider. Every column added from here on is drawn from a standard normal and has nothing to do with price.</div>`
        : `<div class="slider-label" style="margin-top:.5rem">A split is chosen by scanning every column and keeping the best one, so a column that explains nothing simply never wins. A dense layer has no such veto: it starts by mixing all ${9 + r.extra} columns together and has to learn its way back out.</div>`);
  }

  slider.addEventListener('input', render);
  render();
}
