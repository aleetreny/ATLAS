/* Feature importance, and the five planted columns that expose its bias.
 *
 * The chart shows the top of the ranking plus every planted column wherever it
 * lands, so the noise is always visible even when it sits well down the list.
 * Cutting the chart at the top twelve and quietly dropping the evidence would
 * be a strange way to make this particular argument.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

const TOP = 12;

export function initImportanceWidget(data) {
  const rows = data.importance;
  const meta = data.importance_meta;
  const planted = new Set(meta.planted.map((p) => p.feature));

  const { g, w, h } = makeChart('#imp-chart', {
    width: 700,
    height: 520,
    margin: { top: 34, right: 80, bottom: 60, left: 168 },
  });

  const y = d3.scaleBand().range([0, h]).padding(0.24);
  const x = d3.scaleLinear().range([0, w]);
  const tip = tooltip();

  const barG = g.append('g');
  const labG = g.append('g');
  g.append('g').attr('class', 'axis y');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const readout = document.querySelector('#imp-readout');
  const state = { metric: 'gini', highlight: true };

  function render() {
    const gini = state.metric === 'gini';
    const key = gini ? 'gini' : 'perm';
    const sorted = [...rows].sort((a, b) => b[key] - a[key]);
    /* Top of the ranking, plus every planted column wherever it fell. */
    const shown = sorted.slice(0, TOP).concat(sorted.slice(TOP).filter((r) => planted.has(r.feature)));

    y.domain(shown.map((r) => r.feature));
    x.domain([Math.min(0, d3.min(shown, (r) => r[key])), d3.max(shown, (r) => r[key]) * 1.14]);

    drawGrid(g, x, y, w, h, { xTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, xFmt: d3.format('.3f') });
    axisLabels(g, w, h, {
      x: gini ? 'mean decrease in impurity' : 'accuracy lost when the column is shuffled',
    });
    g.select('g.axis.y').call(d3.axisLeft(y).tickSize(0))
      .selectAll('text')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', '10.5px')
      .style('fill', (f) => (state.highlight && planted.has(f) ? 'var(--cosmos)' : 'var(--squidink)'))
      .text((f) => f.replace(/_/g, ' '));

    barG.selectAll('rect').data(shown, (r) => r.feature).join('rect')
      .attr('y', (r) => y(r.feature))
      .attr('height', y.bandwidth())
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 0.8)
      .attr('fill', (r) => (state.highlight && planted.has(r.feature) ? 'var(--cosmos)' : 'var(--primary)'))
      .on('mousemove', (event, r) => tip.show(
        `<span style="color:${planted.has(r.feature) ? '#df2a5d' : '#86198f'}">${r.feature.replace(/_/g, ' ')}</span><br/>` +
        `${r.distinct} distinct values<br/>impurity ${r.gini.toFixed(5)}<br/>permutation ${r.perm.toFixed(5)}`,
        event
      ))
      .on('mouseleave', () => tip.hide())
      .transition('bar').duration(400)
      .attr('x', x(Math.min(0, x.domain()[0])))
      .attr('width', (r) => Math.max(1, x(r[key]) - x(Math.min(0, x.domain()[0]))));

    labG.selectAll('text').data(shown, (r) => r.feature).join('text')
      .attr('class', 'chart-annotation')
      .attr('y', (r) => y(r.feature) + y.bandwidth() / 2 + 4)
      .style('font-size', '0.6rem').style('text-transform', 'none')
      .attr('fill', 'var(--squidink)')
      .transition('bar').duration(400)
      .attr('x', (r) => x(r[key]) + 7)
      .text((r) => r[key].toFixed(4));

    title.text(gini
      ? 'the default ranking: what the model computed while fitting'
      : 'the honest one: what actually breaks when you shuffle the column');
    document.querySelector('#imp-gini').classList.toggle('ghost', !gini);
    document.querySelector('#imp-perm').classList.toggle('ghost', gini);

    const p = meta.planted;
    const worstNoise = Math.max(...p.map((q) => q[key]));
    readout.innerHTML =
      `<table class="tr-table">
         <tr><th>planted noise column</th><th>distinct values</th><th>impurity</th><th>permutation</th></tr>
         ${p.map((q) => `<tr><td>${q.feature.replace(/_/g, ' ')}</td><td>${q.distinct}</td>` +
           `<td>${q.gini.toFixed(5)}</td><td>${q.perm.toFixed(5)}</td></tr>`).join('')}
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        gini
          ? `All five columns are random noise, so all five deserve exactly zero. The finest-grained one is ` +
            `rated <span class="value">${meta.ratio}×</span> the coarsest, purely because more distinct ` +
            `values means more candidate thresholds and more chances for one of them to look good by luck. ` +
            `In fairness to the method: on this data none of them beats the weakest real feature, which ` +
            `scores <span class="value">${meta.weakest_real.toFixed(5)}</span>.`
          : `Shuffle a column on held-out data and measure what breaks. The largest score any of the five ` +
            `planted columns manages is <span class="value">${worstNoise.toFixed(5)}</span>, which is the ` +
            `right answer for a column that contains nothing. The cardinality bias is gone because the ` +
            `question changed: not "where did the model look" but "what happens if I take this away".`
      }</div>`;
  }

  document.querySelector('#imp-gini').addEventListener('click', () => {
    state.metric = 'gini';
    render();
  });
  document.querySelector('#imp-perm').addEventListener('click', () => {
    state.metric = 'perm';
    render();
  });
  document.querySelector('#imp-noise').addEventListener('change', (e) => {
    state.highlight = e.target.checked;
    render();
  });
  render();
}
