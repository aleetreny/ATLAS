/* The same fitted model, the same shock, two answers.
 *
 * An impulse response needs the shocks to be uncorrelated with each other, and
 * they are not, so they get orthogonalised, and orthogonalising requires an
 * order. The order is a modelling assumption about which variable is allowed to
 * react within the same quarter, it is almost never discussed, and here it is a
 * button.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

const NICE = {
  realgdp: 'GDP', realcons: 'consumption', realinv: 'investment', unemp: 'unemployment',
};

export function initIrfWidget(data) {
  const I = data.irf;
  const orders = Object.keys(I.orders);
  let current = 0;
  const readout = document.querySelector('#irf-readout');
  const controls = d3.select('#irf-controls');
  const chart = makeChart('#irf-chart', {
    width: 640, height: 380, margin: { top: 48, right: 122, bottom: 50, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const vars = data.var.vars;
  const all = orders.flatMap((o) => vars.flatMap((v) => I.orders[o].resp[v]));
  const x = d3.scaleLinear().domain([0, I.steps]).range([0, w]);
  const y = d3.scaleLinear().domain([d3.min(all) * 1.1, d3.max(all) * 1.1]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: [0, 3, 6, 9, 12], yTicks: 5 });
  drawAxes(g, x, y, w, h, { xValues: [0, 3, 6, 9, 12], yTicks: 5 });
  axisLabels(g, w, h, {
    x: 'quarters after the shock', y: 'response, percent', leftBudget: margin.left,
  });
  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', 'var(--squidink)').attr('opacity', 0.5);
  const colours = {
    realgdp: 'var(--primary)', realcons: 'var(--anchor)',
    realinv: 'var(--aqua)', unemp: 'var(--galaxy)',
  };
  const lines = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');

  function render() {
    const name = orders[current];
    const R = I.orders[name].resp;
    const line = d3.line().x((d, i) => x(i)).y((d) => y(d));
    lines.selectAll('path').data(vars).join('path')
      .attr('fill', 'none').attr('stroke', (v) => colours[v]).attr('stroke-width', 2)
      .transition('irf').duration(450)
      .attr('d', (v) => line(R[v]));
    const labs = vars.map((v) => ({ v, y: y(R[v][I.steps]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labs.length; i++) {
      if (labs[i].y - labs[i - 1].y < 15) labs[i].y = labs[i - 1].y + 15;
    }
    lines.selectAll('text.lab').data(labs, (d) => d.v).join('text')
      .attr('class', 'chart-note lab').attr('x', w + 6).style('font-size', '11.5px')
      .attr('fill', (d) => colours[d.v]).text((d) => NICE[d.v])
      .transition('irflab').duration(450)
      .attr('y', (d) => d.y + 4);
    wrapLabel(title, `a one standard deviation shock to investment, with the variables `
      + `ordered ${I.orders[name].order.map((v) => NICE[v]).join(' then ')}`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>response of</th><th>peak, GDP first</th><th>peak, investment first</th>
            <th>ratio</th></tr>
        ${vars.map((v) => `<tr>
          <td>${NICE[v]}</td>
          <td>${I.peak[v].gdp_first}</td>
          <td>${I.peak[v].inv_first}</td>
          <td><span class="value">${I.peak[v].ratio}</span></td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        Nothing about the fitted model changed between the two buttons: same
        data, same lag length, same coefficients. What changed is which variable
        is permitted to respond inside the same quarter, and the largest single
        difference in the whole picture is
        <span class="value">${I.max_gap}</span> percent, on
        ${NICE[I.worst_var]}. Press the two buttons back and forth a few times
        and watch the top row of the table rather than the curves; the paragraph
        below is about what that row means for a chart like this one.
      </div>`;
  }

  controls.selectAll('button').data(orders).join('button')
    .attr('class', (d, i) => `atlas-btn${i === current ? '' : ' ghost'}`)
    .text((d) => `ordered: ${d}`)
    .on('click', (evt, d) => {
      current = orders.indexOf(d);
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd === d ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
