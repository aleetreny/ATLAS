/* Feature importance, and the trap underneath it.
 *
 * On this data the highest-gain feature is not carat but y, one of the physical
 * dimensions, which correlates with carat at 0.95. Gain-based importance has to
 * split credit between near-duplicate columns somehow, and how it splits it is
 * close to arbitrary. Showing the correlation next to the importance is the
 * honest way to present the chart. */
import { makeChart, drawAxes, axisLabels, tooltip } from '../../assets/js/chart.js';

export function initImportanceWidget(dia) {
  const rows = dia.importance;
  const corr = dia.corr_with_carat;

  const { g, w, h } = makeChart('#importance-chart', {
    width: 640, height: 340, margin: { top: 30, right: 24, bottom: 46, left: 96 },
  });

  const y = d3.scaleBand().domain(rows.map((r) => r.feature)).range([0, h]).padding(0.25);
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.gain) * 1.12]).range([0, w]);

  g.append('g').attr('class', 'axis y').call(d3.axisLeft(y).tickSizeOuter(0));
  g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.0%')).tickSizeOuter(0));
  axisLabels(g, w, h, { x: 'share of total gain' });

  const tip = tooltip();
  const isProxy = (f) => Math.abs(corr[f] ?? 0) > 0.9 && f !== 'carat';

  g.append('g').selectAll('rect').data(rows).join('rect')
    .attr('x', 0).attr('y', (r) => y(r.feature)).attr('height', y.bandwidth())
    .attr('width', (r) => x(r.gain))
    .attr('fill', (r) => (r.feature === 'carat' ? 'var(--primary)' : isProxy(r.feature) ? 'var(--cosmos)' : 'var(--stone)'))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8)
    .on('mousemove', (event, r) => {
      const c = corr[r.feature];
      tip.show(
        `<span style="color:var(--smile)">${r.feature}</span><br/>` +
        `gain ${(r.gain * 100).toFixed(1)}%<br/>` +
        `correlation with carat ${c === undefined ? 'n/a' : c.toFixed(2)}`,
        event
      );
    })
    .on('mouseleave', () => tip.hide());

  g.append('g').selectAll('text').data(rows).join('text')
    .attr('class', 'chart-annotation')
    .attr('x', (r) => x(r.gain) + 7).attr('y', (r) => y(r.feature) + y.bandwidth() / 2 + 4)
    .style('font-size', '0.62rem').style('letter-spacing', '0.3px').style('text-transform', 'none')
    .text((r) => `${(r.gain * 100).toFixed(1)}%`);

  const proxies = rows.filter((r) => isProxy(r.feature));
  const proxyShare = proxies.reduce((s, r) => s + r.gain, 0);
  const caratShare = rows.find((r) => r.feature === 'carat')?.gain ?? 0;

  document.querySelector('#importance-stats').innerHTML =
    `<div class="slider-label">The three physical dimensions correlate with carat at ` +
    proxies.map((p) => `<span class="value">${corr[p.feature].toFixed(2)}</span>`).join(', ') +
    `, so they are not really separate variables. Together they take ` +
    `<span class="value" style="color:var(--cosmos)">${(proxyShare * 100).toFixed(0)}%</span> of the gain, ` +
    `against <span class="value" style="color:var(--primary)">${(caratShare * 100).toFixed(0)}%</span> for carat itself.</div>` +
    `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">Read naively, this chart says size in millimetres matters more than weight. What it actually says is that one underlying quantity was measured four times, and the algorithm had to give the credit to somebody.</div>`;
}
