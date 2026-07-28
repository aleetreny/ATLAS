/* Two rankings of the same four inputs.
 *
 * One is the model's own variable selection weight, which is what gets printed
 * in a paper as the interpretable part. The other is what actually happens to
 * the error when that input is destroyed. They are different questions and the
 * second one is the only one with a unit.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initAttentionWidget(data) {
  const A = data.attention;
  let view = 'weight';
  const readout = document.querySelector('#attention-readout');
  const controls = d3.select('#attention-controls');
  const chart = makeChart('#attention-chart', {
    width: 640, height: 340, margin: { top: 46, right: 26, bottom: 84, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const names = A.rows.map((r) => r.name);
  const x = d3.scalePoint().domain(names).range([0, w]).padding(0.55);
  const grid = g.append('g');
  const bars = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');

  function render() {
    const key = view === 'weight' ? 'weight' : 'ratio';
    const vals = A.rows.map((r) => (view === 'weight' ? r.weight : r.ratio));
    const y = d3.scaleLinear()
      .domain([view === 'weight' ? 0 : 0.95, Math.max(...vals) * 1.15]).range([h, 0]);
    drawGrid(grid, x, y, w, h, { yTicks: 4 });
    drawAxes(g, x, y, w, h, { yTicks: 4, yFmt: view === 'weight' ? d3.format('.2f') : d3.format('.2f') });
    axisLabels(g, w, h, {
      y: view === 'weight' ? 'the weight it gives that input' : 'error when that input is broken',
      leftBudget: margin.left, xOffset: 66,
    });
    g.selectAll('g.axis.x text').style('font-size', '10px')
      .attr('transform', 'rotate(-14)').attr('text-anchor', 'end');
    bars.selectAll('rect').data(A.rows).join('rect')
      .attr('x', (d) => x(d.name) - 26).attr('width', 52)
      .attr('fill', (d) => (view === 'weight'
        ? (d.weight > 0.01 ? 'var(--primary)' : '#c9ced0')
        : (d.ratio > 1.02 ? 'var(--primary)' : '#c9ced0')))
      .transition('bar').duration(400)
      .attr('y', (d) => y(view === 'weight' ? d.weight : d.ratio))
      .attr('height', (d) => h - y(view === 'weight' ? d.weight : d.ratio));
    bars.selectAll('text').data(A.rows).join('text').attr('class', 'chart-note')
      .attr('x', (d) => x(d.name)).attr('text-anchor', 'middle').style('font-size', '11px')
      .text((d) => (view === 'weight' ? d.weight.toFixed(3) : `${d.ratio.toFixed(3)}x`))
      .transition('lab').duration(400)
      .attr('y', (d) => y(view === 'weight' ? d.weight : d.ratio) - 7);
    if (view === 'weight') {
      wrapLabel(title, `the model's own selection weights, averaged over `
        + `${A.n_windows.toLocaleString('en-US')} windows (the history itself takes `
        + `${A.history.weight})`, w - 8);
    } else {
      wrapLabel(title, `what the error does when each input is shuffled between windows, `
        + `relative to ${A.base_loss}`, w - 8);
    }

    const driver = A.rows[0];
    const copy = A.rows[1];
    const echo = A.rows[2];
    const noise = A.rows[3];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>input</th><th>weight the model gives it</th><th>error when it is broken</th>
            <th>ratio</th></tr>
        ${A.rows.map((r) => `<tr>
          <td>${r.name}</td>
          <td><span class="value">${r.weight}</span></td>
          <td>${r.loss_when_permuted}</td>
          <td><span class="value">${r.ratio}</span></td>
        </tr>`).join('')}
        <tr><td>both copies at once</td><td>${(driver.weight + copy.weight).toFixed(4)}</td>
            <td>${A.both_copies.loss_when_permuted}</td>
            <td><span class="value">${A.both_copies.ratio}</span></td></tr>
      </table>
      <div class="gen-note">
        The second and third columns are two different questions. The last row is
        where they come apart: the promotion and its exact duplicate carry the
        same information, the model puts
        ${driver.weight > copy.weight ? `${driver.weight} on one and ${copy.weight} on the other`
    : `${copy.weight} on one and ${driver.weight} on the other`},
        and breaking either one alone costs
        ${Math.min(driver.ratio, copy.ratio).toFixed(3)}x to
        ${Math.max(driver.ratio, copy.ratio).toFixed(3)}x, while breaking both
        costs <span class="value">${A.both_copies.ratio}x</span>.
        ${Math.min(driver.ratio, copy.ratio) < 1.05
    ? `So permutation says one of the two copies does not matter, which is
           true only because the other one is still there.`
    : `So permutation finds both of them, which happens when the model has
           spread its reliance across the pair.`}
        And ${echo.name} gets a weight of ${echo.weight} while breaking it still
        costs ${echo.ratio}x: a weight of zero does not mean no effect, because
        the weights themselves are computed from all the inputs, so destroying
        an input the model claims to ignore still changes what it decides to
        attend to. Noise gets ${noise.weight} and costs ${noise.ratio}x, which is
        the one row where the two agree completely.
      </div>`;
  }

  controls.selectAll('button').data([['what the model says', 'weight'],
    ['what breaking it costs', 'cost']]).join('button')
    .attr('class', (d) => `atlas-btn${d[1] === view ? '' : ' ghost'}`)
    .text((d) => d[0])
    .on('click', (evt, d) => {
      view = d[1];
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd[1] === view ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
