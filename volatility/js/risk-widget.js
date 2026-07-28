/* Exceedances against their nominal rate.
 *
 * This is the only score in the article that a risk desk would recognise, and
 * it is a count: how often did the loss go past the line the model drew? A
 * model can win the likelihood comparison and still fail here, and one of them
 * fails in both directions at once, which is the row worth looking at.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initRiskWidget(data) {
  const R = data.risk;
  let level = 1;
  const readout = document.querySelector('#risk-readout');
  const controls = d3.select('#risk-controls');
  const chart = makeChart('#risk-chart', {
    width: 640, height: 350, margin: { top: 46, right: 104, bottom: 74, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const names = R.rows.map((r) => r.model);
  const x = d3.scalePoint().domain(names).range([0, w]).padding(0.6);
  const y = d3.scaleLinear().domain([0, 0.022]).range([h, 0]);
  const grid = g.append('g');
  const bars = g.append('g');
  const nominal = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');

  function render() {
    const key = `obs${level}`;
    const nom = level / 100;
    const hi = Math.max(nom * 2.2, d3.max(R.rows, (r) => r[key]) * 1.2);
    y.domain([0, hi]);
    drawGrid(grid, x, y, w, h, { yTicks: 4 });
    drawAxes(g, x, y, w, h, { yTicks: 4, yFmt: d3.format('.1%') });
    axisLabels(g, w, h, { y: 'days past the line', leftBudget: margin.left, xOffset: 56 });
    g.selectAll('g.axis.x text').style('font-size', '10px')
      .attr('transform', 'rotate(-12)').attr('text-anchor', 'end');
    bars.selectAll('rect').data(R.rows).join('rect')
      .attr('x', (d) => x(d.model) - 22).attr('width', 44)
      .attr('fill', (d) => (d[`kupiec${level}`] < 0.05 ? 'var(--cosmos)' : 'var(--primary)'))
      .transition('bar').duration(400)
      .attr('y', (d) => y(d[key])).attr('height', (d) => h - y(d[key]));
    nominal.selectAll('*').remove();
    nominal.append('line').attr('x1', 0).attr('x2', w + 6).attr('y1', y(nom)).attr('y2', y(nom))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.3).attr('stroke-dasharray', '5 4');
    nominal.append('text').attr('class', 'chart-note').attr('x', w + 10).attr('y', y(nom) + 4)
      .style('font-size', '11.5px').text(`${level}% promised`);
    wrapLabel(title, `the ${level}% loss line, over ${R.n_days.toLocaleString('en-US')} days: `
      + `red is rejected by the count`, w - 8);

    const rows = R.rows;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>model</th><th>days past the line</th><th>promised</th><th>ratio</th>
            <th>Kupiec p</th></tr>
        ${rows.map((r) => `<tr>
          <td>${r.model}</td>
          <td><span class="value">${(r[key] * 100).toFixed(2)}%</span></td>
          <td>${level}%</td>
          <td>${r[`ratio${level}`]}</td>
          <td>${r[`kupiec${level}`] < 0.001 ? '&lt;0.001' : r[`kupiec${level}`]}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        ${level === 1
    ? `At one percent every model on this page is wrong in the same direction:
           the loss goes past the line more often than promised, from
           ${(d3.min(rows, (r) => r.obs1) * 100).toFixed(2)}% to
           ${(d3.max(rows, (r) => r.obs1) * 100).toFixed(2)}% where 1% was
           claimed. The tail is the part a fitted variance does not fix, and the
           fat tailed innovation is the closest: ${(rows.find((r) => r.model === 'GARCH t').obs1 * 100).toFixed(2)}%.`
    : `At five percent the two GARCH models are indistinguishable from their
           promise (Kupiec p of ${rows.find((r) => r.model === 'GARCH normal').kupiec5}
           and ${rows.find((r) => r.model === 'GARCH t').kupiec5}), and the
           constant variance model is rejected for the opposite reason to the
           one percent case: it draws its line too far out
           (${(rows.find((r) => r.model === 'normal constant').obs5 * 100).toFixed(2)}%
           against 5%) because it is averaging calm years with 2008.`}
        Switch levels and read the constant variance row both ways: it is the
        one model that is too cautious in the body of the distribution and not
        cautious enough in the tail, which is what a single number for the
        spread buys you.
      </div>`;
  }

  controls.selectAll('button').data([1, 5]).join('button')
    .attr('class', (d) => `atlas-btn${d === level ? '' : ' ghost'}`)
    .text((d) => `the ${d}% line`)
    .on('click', (evt, d) => {
      level = d;
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd === level ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
