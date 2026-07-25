/* The rotation test on the real diamonds, not on a quilt.
 *
 * Same 43,136 training rows, same held-out set, same models. The only change is
 * that the nine standardised columns are multiplied by a random orthogonal
 * matrix, so every row keeps its length and every pair of rows keeps its
 * distance. Nothing is lost that a rotation-invariant model could have used.
 *
 * Two views, because they say different things. The percentage view isolates
 * the inductive bias: only the tree ensemble is hurt. The absolute view is the
 * corrective: boosting gives up 18% and is still the most accurate model on the
 * chart, which is why "trees are not rotation invariant" is an explanation
 * rather than a reason to switch.
 */
import { makeChart, axisLabels } from '../../assets/js/chart.js';

export function initRotationWidget(data) {
  const rows = data.rotation;
  let mode = 'penalty';

  const { g, w, h } = makeChart('#rot-real-chart', {
    width: 640,
    height: 320,
    margin: { top: 24, right: 92, bottom: 52, left: 132 },
  });

  const y = d3.scaleBand().domain(rows.map((r) => r.name)).range([0, h]).padding(0.34);
  g.append('g').attr('class', 'axis y').call(d3.axisLeft(y).tickSizeOuter(0));
  const gx = g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${h})`);

  const barsG = g.append('g');
  const labelG = g.append('g');
  const zero = g.append('line').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5).attr('y1', 0).attr('y2', h);

  /* Only the absolute view has two bars per model, so only it needs a key. */
  const legend = g.append('g').attr('transform', 'translate(0,-10)').attr('opacity', 0);
  [
    { label: 'before rotation', colour: 'var(--stone)', dx: 0 },
    { label: 'after', colour: 'var(--primary)', dx: 132 },
  ].forEach((d) => {
    legend.append('rect').attr('x', d.dx).attr('y', -9).attr('width', 10).attr('height', 10)
      .attr('fill', d.colour).attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8);
    legend.append('text').attr('class', 'chart-annotation').attr('x', d.dx + 15).attr('y', 0)
      .style('font-size', '0.58rem').style('text-transform', 'none').text(d.label);
  });

  function render() {
    const penalty = mode === 'penalty';
    const x = penalty
      ? d3.scaleLinear().domain([-10, 22]).range([0, w])
      : d3.scaleLinear().domain([0, d3.max(rows, (r) => Math.max(r.plain, r.rotated)) * 1.08]).range([0, w]);

    gx.transition()
      .duration(360)
      .call(
        d3
          .axisBottom(x)
          .ticks(6)
          .tickFormat((v) => (penalty ? (v > 0 ? `+${v}%` : `${v}%`) : '$' + d3.format(',')(v)))
          .tickSizeOuter(0)
      );
    axisLabels(g, w, h, { x: penalty ? 'change in error when the axes are rotated' : 'held-out RMSE' });
    zero.transition().duration(360).attr('x1', x(0)).attr('x2', x(0)).attr('opacity', penalty ? 1 : 0);
    legend.transition().duration(360).attr('opacity', penalty ? 0 : 1);

    /* In penalty mode one bar per model; in absolute mode two, before and
     * after, so the pair can be compared directly. */
    const items = penalty
      ? rows.map((r) => ({ key: r.name, name: r.name, from: 0, to: r.penalty, colour: r.penalty > 1 ? 'var(--cosmos)' : 'var(--seablue)', band: 1, slot: 0, text: `${r.penalty > 0 ? '+' : ''}${r.penalty}%` }))
      : rows.flatMap((r) => [
          { key: `${r.name}-a`, name: r.name, from: 0, to: r.plain, colour: 'var(--stone)', band: 2, slot: 0, text: '$' + r.plain.toLocaleString('en-US') },
          { key: `${r.name}-b`, name: r.name, from: 0, to: r.rotated, colour: 'var(--primary)', band: 2, slot: 1, text: '$' + r.rotated.toLocaleString('en-US') },
        ]);

    barsG
      .selectAll('rect')
      .data(items, (d) => d.key)
      .join((e) => e.append('rect').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.9).attr('height', 0))
      .attr('fill', (d) => d.colour)
      .transition()
      .duration(360)
      .attr('x', (d) => Math.min(x(d.from), x(d.to)))
      .attr('width', (d) => Math.max(1.5, Math.abs(x(d.to) - x(d.from))))
      .attr('y', (d) => y(d.name) + (d.slot * y.bandwidth()) / d.band)
      .attr('height', (d) => y.bandwidth() / d.band - (d.band > 1 ? 3 : 0));

    labelG
      .selectAll('text')
      .data(items, (d) => d.key)
      .join((e) => e.append('text').attr('class', 'chart-annotation').style('font-size', '0.6rem').style('text-transform', 'none'))
      .attr('fill', (d) => (d.colour === 'var(--stone)' ? 'var(--squidink)' : d.colour))
      .transition()
      .duration(360)
      .attr('x', (d) => Math.max(x(d.from), x(d.to)) + 7)
      .attr('y', (d) => y(d.name) + (d.slot * y.bandwidth()) / d.band + y.bandwidth() / (2 * d.band) + 3.5)
      .text((d) => d.text);
  }

  const gbm = rows.find((r) => r.name === 'gradient boosting');
  const mlp = rows.find((r) => r.name === 'mlp');
  document.querySelector('#rot-real-readout').innerHTML =
    `<div class="slider-label">Boosting pays <span class="value" style="color:var(--cosmos)">+${gbm.penalty}%</span> for the rotation; ` +
    `the network pays <span class="value" style="color:var(--seablue)">${mlp.penalty}%</span>, which is to say nothing at all, ` +
    `and ridge is unmoved to the dollar because a rotation is exactly the kind of transformation a linear model absorbs into its coefficients.</div>` +
    `<div class="slider-label" style="margin-top:.5rem">Now switch the toggle to <span class="bold">absolute error</span>. Boosting after the rotation is ` +
    `$${gbm.rotated.toLocaleString('en-US')}, still better than the network before it. The rotation explains where the advantage comes from; it does not take the advantage away.</div>`;

  const btn = document.querySelector('#rot-real-mode');
  btn.addEventListener('click', () => {
    mode = mode === 'penalty' ? 'absolute' : 'penalty';
    btn.textContent = mode === 'penalty' ? '% change' : 'absolute error';
    render();
  });

  render();
}
