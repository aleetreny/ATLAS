/* The three settings, each swept, each with the thing it is usually chosen by
 * drawn next to what it actually does.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { etcFormula } from './banditkit.js';

const VIEWS = [
  { key: 'eps', label: 'how often to explore' },
  { key: 'etc', label: 'how long to explore first' },
  { key: 'decay', label: 'how fast to stop exploring' },
];

export function initDialsWidget(data) {
  const readout = document.querySelector('#dials-readout');
  const controls = d3.select('#dials-controls');
  let view = 'eps';

  const chart = makeChart('#dials-chart', {
    width: 640, height: 360, margin: { top: 50, right: 122, bottom: 56, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function renderEps() {
    const rows = data.epsilon.rows;
    const x = d3.scalePoint().domain(rows.map((d) => String(d.eps))).range([0, w]).padding(0.5);
    const y = d3.scaleLinear().domain([0, Math.max(...rows.map((d) => d.regret)) * 1.15])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5 });
    axisLabels(g, w, h, { x: 'share of pulls spent exploring', y: 'regret',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    const best = rows.reduce((a, b) => (b.regret < a.regret ? b : a), rows[0]);
    plot.selectAll('rect').data(rows).join('rect')
      .attr('x', (d) => x(String(d.eps)) - 18).attr('width', 36)
      .attr('y', (d) => y(d.regret)).attr('height', (d) => h - y(d.regret))
      .attr('fill', (d) => (d === best ? 'var(--primary)' : '#c9ced0'));
    key.selectAll('*').remove();
    title.text(`the exploring rate, swept, at ${data.meta.horizon.toLocaleString('en-US')} pulls`);
    sub.text(`the least regret is at ${best.eps}, worth ${best.regret}`);
    const linear = rows.filter((d) => d.eps > 0);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>rate</th><th>regret</th><th>per pull, late</th><th>the tax it predicts</th></tr>
        ${rows.map((d) => `
          <tr><td>${d === best ? '<span class="value">' : ''}${d.eps}${d === best ? '</span>' : ''}</td>
              <td>${d.regret}</td><td>${d.measured_slope}</td>
              <td>${d.predicted_slope}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The last two rows are the argument against a fixed rate, and they are the
        same number twice. Exploring a share of the time means paying that share
        of the average gap forever, whatever you have already learnt: predicted
        ${linear[linear.length - 1].predicted_slope} per pull at rate
        ${linear[linear.length - 1].eps}, measured
        ${linear[linear.length - 1].measured_slope} over the second half of the
        run. So the regret of a fixed rate is a straight line, and no straight
        line is ever the best you can do for long. The measured minimum here,
        ${best.eps}, is the best compromise for
        <span class="value">this horizon</span> and would move if the horizon did.
      </div>`;
  }

  function renderEtc() {
    const sweeps = data.etc.sweeps;
    const all = sweeps.flatMap((s) => s.rows);
    const x = d3.scaleLog().domain([1, Math.max(...all.map((d) => d.m)) * 1.2]).range([0, w]);
    const y = d3.scaleLinear().domain([0, Math.max(...all.map((d) => d.regret)) * 1.1])
      .range([h, 0]);
    const xt = [1, 10, 100, 640];
    drawGrid(g, x, y, w, h, { xValues: xt, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: xt, yTicks: 5, xFmt: d3.format(',d') });
    axisLabels(g, w, h, { x: 'pulls given to each machine first', y: 'regret',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    const shades = ['var(--primary)', 'var(--anchor)', 'var(--cosmos)'];
    sweeps.forEach((s, i) => {
      plot.append('path').attr('fill', 'none').attr('stroke', shades[i])
        .attr('stroke-width', 2.2)
        .attr('d', d3.line().x((d) => x(d.m)).y((d) => y(d.regret))(s.rows));
      const bestRow = s.rows.find((d) => d.m === s.best_m);
      plot.append('circle').attr('cx', x(bestRow.m)).attr('cy', y(bestRow.regret))
        .attr('r', 5).attr('fill', shades[i]);
      const predRow = s.rows.find((d) => d.m === s.predicted_m);
      if (predRow) {
        plot.append('circle').attr('cx', x(predRow.m)).attr('cy', y(predRow.regret))
          .attr('r', 6).attr('fill', 'none').attr('stroke', shades[i]).attr('stroke-width', 2);
      }
    });
    key.selectAll('*').remove();
    sweeps.forEach((s, i) => {
      key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 30 + 6).attr('y2', i * 30 + 6)
        .attr('stroke', shades[i]).attr('stroke-width', 2.2);
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 30 + 20)
        .style('font-size', '10.5px').text(`gap ${s.delta}`);
    });
    key.append('circle').attr('cx', 6).attr('cy', 100).attr('r', 5).attr('fill', '#8e9aa6');
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 116)
      .style('font-size', '10.5px').text('measured best');
    key.append('circle').attr('cx', 6).attr('cy', 134).attr('r', 6).attr('fill', 'none')
      .attr('stroke', '#8e9aa6').attr('stroke-width', 2);
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 150)
      .style('font-size', '10.5px').text('what the');
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 166)
      .style('font-size', '10.5px').text('formula picks');
    title.text('two machines, and how long to try both before choosing');
    sub.text(`${data.etc.horizon.toLocaleString('en-US')} pulls in total, `
      + `${sweeps[0].rows[0].of.toLocaleString('en-US')} runs per point`);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>gap</th><th>best length, measured</th><th>its regret</th>
            <th>the formula says</th><th>its regret</th></tr>
        ${sweeps.map((s) => `
          <tr><td>${s.delta}</td>
              <td><span class="value">${s.best_m}</span></td><td>${s.best_regret}</td>
              <td>${s.predicted_m}</td><td>${s.regret_at_predicted ?? 'off the grid'}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        The formula everybody quotes for how long to explore first is
        ${sweeps.map((s) => s.ratio).join(', ')} times longer than what actually
        works here, and following it costs
        ${sweeps.map((s) => (s.regret_at_predicted / s.best_regret).toFixed(1)).join(', ')}
        times the regret. That is not a mistake in the formula. It minimises an
        <span class="bold">upper bound</span> on the regret rather than the
        regret, and an upper bound that is loose by a constant produces a setting
        that is wrong by that constant. It is the honest way to read every
        closed form in this branch: the shape is right and the number is a
        starting point for a sweep.
      </div>`;
  }

  function renderDecay() {
    const rows = data.decay.rows;
    const x = d3.scaleLog().domain([rows[0].c * 0.7, rows[rows.length - 1].c * 1.4]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: rows.map((d) => d.c), yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: rows.map((d) => d.c), yTicks: 5,
      xFmt: d3.format(',g') });
    axisLabels(g, w, h, { x: 'the constant in the decaying rate',
      y: 'how well each shape fits', leftBudget: margin.left });
    plot.selectAll('*').remove();
    [['linear_r2', 'var(--cosmos)'], ['log_r2', 'var(--primary)']].forEach(([f, col]) => {
      plot.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.4)
        .attr('d', d3.line().x((d) => x(d.c)).y((d) => y(d[f]))(rows));
      plot.selectAll(`circle.${f}`).data(rows).join('circle').attr('class', f)
        .attr('cx', (d) => x(d.c)).attr('cy', (d) => y(d[f])).attr('r', 4).attr('fill', col);
    });
    key.selectAll('*').remove();
    [['var(--cosmos)', 'a straight line'], ['var(--primary)', 'a logarithm']]
      .forEach(([col, text], i) => {
        key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 34 + 6)
          .attr('y2', i * 34 + 6).attr('stroke', col).attr('stroke-width', 2.4);
        text.split(' ').forEach((word, j) => {
          key.append('text').attr('class', 'chart-note').attr('x', 0)
            .attr('y', i * 34 + 20 + j * 11).style('font-size', '10.5px')
            .text(j === 0 ? text.split(' ').slice(0, 2).join(' ') : text.split(' ').slice(2).join(' '));
        });
      });
    title.text('a rate that decays like one over the pull count');
    sub.text('which shape the regret follows, over five horizons');
    const flips = rows.find((d) => d.log_r2 > d.linear_r2);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>constant</th>${rows.map((d) => `<th>${d.c}</th>`).join('')}</tr>
        <tr><td>fits a straight line</td>
            ${rows.map((d) => `<td>${d.linear_r2}</td>`).join('')}</tr>
        <tr><td>fits a logarithm</td>
            ${rows.map((d) => `<td>${d.log_r2}</td>`).join('')}</tr>
        <tr><td>regret at the longest run</td>
            ${rows.map((d) => `<td>${d.regret[d.regret.length - 1]}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        A rate that decays is the textbook fix for the straight line above, and it
        works, but only past a constant. Below
        <span class="value">${flips ? flips.c : 'any value here'}</span> the regret
        still fits a straight line better than a logarithm; at and above it, the
        logarithm wins. The theory says the constant has to scale like one over
        the square of the smallest gap, which for these machines
        (${data.decay.smallest_gap}) means about
        <span class="value">${data.decay.required_c.toLocaleString('en-US')}</span>,
        far above where the fit flips: the requirement is sufficient, not
        necessary. And it is written in terms of a gap that nobody knows before
        they start, which is the flaw the next article is built to remove.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    plot.selectAll('*').remove();
    if (view === 'eps') renderEps();
    else if (view === 'etc') renderEtc();
    else renderDecay();
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
