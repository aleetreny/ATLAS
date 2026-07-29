/* Six schedules on the forty cities of the previous article.
 *
 * Two lines per schedule: the temperature it was at, and the tour length it was
 * carrying. Reading them together is the point, because a schedule is only ever
 * wrong in one of two ways, too hot at the end or too cold at the start, and
 * both are visible.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initScheduleWidget(data) {
  const S = data.schedules;
  const readout = document.querySelector('#schedule-readout');
  const controls = d3.select('#schedule-controls');
  let pick = 'geometric_fast';

  const chart = makeChart('#schedule-chart', {
    width: 640, height: 380, margin: { top: 50, right: 126, bottom: 54, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().domain([0, S.budget]).range([0, w]);
  const yLen = d3.scaleLinear()
    .domain([S.reference.two_opt * 0.9, Math.max(...S.rows.map((r) => r.mean)) * 1.5])
    .range([h, 0]);
  const yT = d3.scaleLog().domain([S.T1 * 0.5, S.T0 * 1.5]).range([h, 0]);
  drawGrid(g, x, yLen, w, h, { xTicks: 4, yTicks: 4 });
  drawAxes(g, x, yLen, w, h, { xTicks: 4, yTicks: 4, xFmt: d3.format('.2s') });
  axisLabels(g, w, h, { x: 'moves tried', y: 'tour length', leftBudget: margin.left });

  /* the temperature travels on its own scale, so it gets its own axis on the
     right: a second scale without an axis is decoration */
  const gT = g.append('g').attr('transform', `translate(${w},0)`);
  gT.call(d3.axisRight(yT).ticks(3, '~g').tickSizeOuter(0)).attr('class', 'axis y');
  g.append('text').attr('class', 'chart-note').attr('x', w + 8).attr('y', -6)
    .style('font-size', '10.5px').text('temperature');

  const ref2opt = g.append('line').attr('stroke', '#8e9aa6').attr('stroke-width', 1.4)
    .attr('x1', 0).attr('x2', w).attr('y1', yLen(S.reference.two_opt))
    .attr('y2', yLen(S.reference.two_opt));
  const refGa = g.append('line').attr('stroke', '#8e9aa6').attr('stroke-width', 1.4)
    .attr('stroke-dasharray', '4 4')
    .attr('x1', 0).attr('x2', w).attr('y1', yLen(S.reference.ga))
    .attr('y2', yLen(S.reference.ga));
  /* The two references are 0.14 apart on an axis nine units tall, so a fixed
     offset stacks them on top of each other. They are sorted by value and the
     upper one is labelled above its line and the lower one below it, which
     keeps them apart whatever the numbers turn out to be. */
  [[S.reference.two_opt, `two opt, ${S.reference.two_opt}`],
    [S.reference.ga, `the genetic algorithm, ${S.reference.ga}`]]
    .sort((a, b) => b[0] - a[0])
    .forEach(([v, text], i) => {
      g.append('text').attr('class', 'chart-note').attr('x', 4)
        .attr('y', yLen(v) + (i === 0 ? -6 : 13)).style('font-size', '10.5px')
        .text(text);
    });

  const lenPath = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2.4);
  const tempPath = g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)')
    .attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function render() {
    const trace = S.traces[pick];
    const row = S.rows.find((r) => r.schedule === pick);
    lenPath.attr('d', d3.line().x((p) => x(p[0])).y((p) => yLen(Math.min(p[1], yLen.domain()[1])))(trace));
    tempPath.attr('d', d3.line().x((p) => x(p[0]))
      .y((p) => yT(Math.max(Math.min(p[2], yT.domain()[1]), yT.domain()[0])))(trace));
    title.text(`${row.label}: ${row.mean} on average over ${S.seeds} runs`);
    sub.text(`${(row.uphill_share * 100).toFixed(1)}% of the uphill moves it tried `
      + 'were accepted');

    const best = S.rows.reduce((a, b) => (b.mean < a.mean ? b : a), S.rows[0]);
    const hot = S.rows.find((r) => r.schedule === 'hot');
    const cold = S.rows.find((r) => r.schedule === 'cold');
    const cooled = S.rows.filter((r) => !['hot', 'cold'].includes(r.schedule));
    const spread = Math.max(...cooled.map((r) => r.mean)) - Math.min(...cooled.map((r) => r.mean));
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>schedule</th><th>mean</th><th>best</th><th>spread</th><th>uphill taken</th></tr>
        ${S.rows.map((r) => `
          <tr><td>${r.schedule === pick ? '<span class="value">' : ''}${r.label}${r.schedule === pick ? '</span>' : ''}</td>
              <td>${r.mean}</td><td>${r.best}</td><td>${r.spread}</td>
              <td>${(r.uphill_share * 100).toFixed(1)}%</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The two controls are the ends of the dial. Never cooling gives
        <span class="value">${hot.mean}</span>, which is a random walk that
        happens to remember its best moment; never heating gives
        <span class="value">${cold.mean}</span>, which is the downhill only
        search from the last article under another name. Everything that actually
        cools lands between ${Math.min(...cooled.map((r) => r.mean))} and
        ${Math.max(...cooled.map((r) => r.mean))}, a spread of
        ${spread.toFixed(4)}, and the best of them (${best.label}, ${best.mean})
        beats the genetic algorithm's ${S.reference.ga} on the same cities and
        the same budget while staying above two opt's ${S.reference.two_opt}.
        Which schedule you pick matters far less than that you pick one.
      </div>`;
  }

  controls.selectAll('button').data(S.rows).join('button')
    .attr('class', (d) => `atlas-btn${d.schedule === pick ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      pick = d.schedule;
      controls.selectAll('button')
        .attr('class', (q) => `atlas-btn${q.schedule === pick ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
