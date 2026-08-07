/* Is the uncertainty honest? Two checks, one per policy.
 *
 * A confidence bound is a promise: the truth is below it, except this often.
 * A posterior is a promise too: the truth is inside the middle ninety percent,
 * nine times in ten. Both promises are counted here against what happened.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const VIEWS = [
  { key: 'bound', label: 'how often the bound is wrong' },
  { key: 'post', label: 'how often the truth is inside' },
  { key: 'const', label: 'what the constant costs' },
];

export function initHonestWidget(data) {
  const readout = document.querySelector('#honest-readout');
  const controls = d3.select('#honest-controls');
  let view = 'bound';

  const chart = makeChart('#honest-chart', {
    width: 640, height: 350, margin: { top: 48, right: 122, bottom: 56, left: 68 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function renderBound() {
    const C = data.coverage;
    const rows = C.by_t;
    const x = d3.scaleLinear().domain([0, C.horizon]).range([0, w]);
    const y = d3.scaleLinear().domain([0, Math.max(...rows.map((d) => d[2])) * 1.15])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, yFmt: d3.format('.3f') });
    axisLabels(g, w, h, { x: 'pulls', y: 'share of arms above their bound',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    plot.append('path').attr('fill', 'none').attr('stroke', '#8e9aa6')
      .attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
      .attr('d', d3.line().x((d) => x(d[0])).y((d) => y(d[2]))(rows));
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(rows));
    key.selectAll('*').remove();
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 6).attr('y2', 6)
      .attr('stroke', '#8e9aa6').attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    ['what the', 'inequality', 'allows'].forEach((t, i) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 20 + i * 16)
        .style('font-size', '10px').text(t);
    });
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 66).attr('y2', 66)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
    ['what actually', 'happened'].forEach((t, i) => {
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 80 + i * 16)
        .style('font-size', '10px').text(t);
    });
    title.text(`${C.runs.toLocaleString('en-US')} runs, every arm checked at every pull`);
    sub.text(`the truth was above the bound in ${(C.share_outside * 100).toFixed(4)}% of checks`);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>pulls</th>${rows.filter((_, i) => i % 2 === 0).map((d) => `<th>${d[0]}</th>`).join('')}</tr>
        <tr><td>allowed to be wrong</td>
            ${rows.filter((_, i) => i % 2 === 0).map((d) => `<td>${d[2]}</td>`).join('')}</tr>
        <tr><td>actually wrong</td>
            ${rows.filter((_, i) => i % 2 === 0).map((d) => `<td>${d[1]}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Over ${C.runs.toLocaleString('en-US')} runs of ${C.horizon} pulls the true
        mean of an arm was above that arm's bound in
        <span class="value">${(C.share_outside * 100).toFixed(4)}%</span> of all
        the checks, against the ${(C.promised * 100).toFixed(2)}% the inequality
        allows on average. The bound is not merely correct, it is
        <span class="bold">never wrong</span>, and that is the problem: an
        interval that is never wrong is wider than it needs to be, and every unit
        of extra width is a pull spent on an arm that did not deserve it. The
        third view puts a number on what that costs.
      </div>`;
  }

  function renderPost() {
    const C = data.calibration;
    const x = d3.scalePoint().domain(C.per_arm.map((_, i) => `machine ${i + 1}`))
      .range([0, w]).padding(0.6);
    const y = d3.scaleLinear().domain([0.7, 1]).range([h, 0]);
    drawGrid(g, x, y, w, h, { yTicks: 4 });
    drawAxes(g, x, y, w, h, { yTicks: 4 });
    axisLabels(g, w, h, { y: 'times the truth was inside', leftBudget: margin.left });
    plot.selectAll('*').remove();
    plot.selectAll('rect').data(C.per_arm).join('rect')
      .attr('x', (_, i) => x(`machine ${i + 1}`) - 24).attr('width', 48)
      .attr('y', (d) => y(d)).attr('height', (d) => h - y(d))
      .attr('fill', 'var(--primary)');
    plot.selectAll('text').data(C.per_arm).join('text').attr('class', 'chart-note')
      .attr('x', (_, i) => x(`machine ${i + 1}`)).attr('y', (d) => y(d) - 6)
      .attr('text-anchor', 'middle').style('font-size', '10.5px')
      .text((d) => d.toFixed(3));
    plot.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(C.level)).attr('y2', y(C.level))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);
    /* at the left edge it landed on the first bar's own label */
    plot.append('text').attr('class', 'chart-note').attr('x', w - 2).attr('y', y(C.level) - 6)
      .attr('text-anchor', 'end').style('font-size', '10.5px').attr('fill', 'var(--cosmos)')
      .text(`promised ${C.level}`);
    key.selectAll('*').remove();
    title.text(`${C.runs} runs, the middle ${C.level * 100}% of each posterior`);
    sub.text(`checked after ${C.checked_at.join(', ')} pulls`);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>machine</th>${C.per_arm.map((_, i) => `<th>${i + 1}</th>`).join('')}</tr>
        <tr><td>truth inside the interval</td>
            ${C.per_arm.map((d) => `<td>${d}</td>`).join('')}</tr>
        <tr><td>pulls it tends to get</td>
            ${data.meta.arms.map((a, i) => `<td>${a === Math.max(...data.meta.arms) ? 'most' : 'few'}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Across every arm and every checkpoint the truth sits inside the middle
        ${C.level * 100}% of the posterior <span class="value">${(C.inside * 100).toFixed(2)}%</span>
        of the time, against a promise of ${C.level * 100}%. That is close and it
        is not exact, and the gap has a cause worth naming: the data is not an
        independent sample. The policy chooses what to look at next based on what
        it has already seen, so the posterior of an arm is conditioned on the
        history that made it stop pulling that arm. Being ${((C.level - C.inside) * 100).toFixed(2)}
        points optimistic under that much self selection is the honest version of
        "well calibrated".
      </div>`;
  }

  function renderConst() {
    const rows = data.constant.rows;
    const x = d3.scaleLog().domain([rows[0].c * 0.7, rows[rows.length - 1].c * 1.4])
      .range([0, w]);
    const y = d3.scaleLinear().domain([0, Math.max(...rows.map((d) => d.regret)) * 1.15])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: rows.map((d) => d.c), yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: rows.map((d) => d.c), yTicks: 5, xFmt: d3.format(',g') });
    axisLabels(g, w, h, { x: 'the constant in the bound', y: 'regret',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    const best = rows.reduce((a, b) => (b.regret < a.regret ? b : a), rows[0]);
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d) => x(d.c)).y((d) => y(d.regret))(rows));
    plot.selectAll('circle').data(rows).join('circle')
      .attr('cx', (d) => x(d.c)).attr('cy', (d) => y(d.regret)).attr('r', 4.5)
      .attr('fill', (d) => (d === best ? 'var(--primary)' : '#c9ced0'))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4);
    const proved = rows.find((d) => d.c === data.constant.proved_c);
    plot.append('line').attr('x1', x(proved.c)).attr('x2', x(proved.c))
      .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.6);
    plot.append('text').attr('class', 'chart-note').attr('x', x(proved.c) - 6).attr('y', 14)
      .attr('text-anchor', 'end').style('font-size', '10.5px').attr('fill', 'var(--cosmos)')
      .text('the proved value');
    key.selectAll('*').remove();
    title.text(`the size of the bonus, swept, over ${data.constant.horizon.toLocaleString('en-US')} pulls`);
    sub.text(`least regret at ${best.c}, against the ${proved.c} the proof uses`);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>constant</th>${rows.map((d) => `<th>${d.c}</th>`).join('')}</tr>
        <tr><td>regret</td>
            ${rows.map((d) => `<td>${d === best ? '<span class="value">' : ''}${d.regret}${d === best ? '</span>' : ''}</td>`).join('')}</tr>
        <tr><td>pulls on the best</td>
            ${rows.map((d) => `<td>${(d.share_best * 100).toFixed(0)}%</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        The constant that the proof needs is ${proved.c}, and the constant that
        works here is <span class="value">${best.c}</span>, eight times smaller,
        for ${(proved.regret / best.regret).toFixed(1)} times less regret. This is
        the same shape of finding as the explore then commit formula in the
        previous article and it has the same cause: a proof has to hold for every
        problem, so it buys a bound that is never wrong, and never wrong is
        expensive. The first view of this widget is that expense, measured
        directly: the interval failed
        ${(data.coverage.share_outside * 100).toFixed(4)}% of the time when it was
        allowed to fail ${(data.coverage.promised * 100).toFixed(2)}%.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    plot.selectAll('*').remove();
    if (view === 'bound') renderBound();
    else if (view === 'post') renderPost();
    else renderConst();
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
