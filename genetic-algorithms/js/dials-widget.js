/* The two settings that decide whether any of this works.
 *
 * Population size, swept at a fixed budget so that a bigger population buys
 * fewer generations rather than more compute; and selection pressure, measured
 * as how many generations the best individual needs to fill the population with
 * mutation and crossover switched off, which is the only way to see selection
 * on its own.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const VIEWS = [
  { key: 'pop', label: 'population size' },
  { key: 'press', label: 'tournament size' },
];

export function initDialsWidget(data) {
  const readout = document.querySelector('#dials-readout');
  const controls = d3.select('#dials-controls');
  let view = 'pop';

  const chart = makeChart('#dials-chart', {
    width: 640, height: 350, margin: { top: 50, right: 120, bottom: 54, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);

  function renderPop() {
    const rows = data.population.rows;
    const x = d3.scaleLog().domain([rows[0].pop * 0.8, rows[rows.length - 1].pop * 1.25])
      .range([0, w]);
    const y = d3.scaleLinear().domain([0, data.meta.seeds]).range([h, 0]);
    const xt = rows.map((d) => d.pop);
    drawGrid(g, x, y, w, h, { xValues: xt, yTicks: 4 });
    drawAxes(g, x, y, w, h, { xValues: xt, yTicks: 4, xFmt: d3.format(',d') });
    axisLabels(g, w, h, { x: 'population', y: `runs that solved it, of ${data.meta.seeds}`,
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d) => x(d.pop)).y((d) => y(d.success))(rows));
    plot.selectAll('circle').data(rows).join('circle')
      .attr('cx', (d) => x(d.pop)).attr('cy', (d) => y(d.success)).attr('r', 4)
      .attr('fill', 'var(--primary)');
    key.selectAll('*').remove();
    title.text(`the same ${data.meta.budget.toLocaleString('en-US')} evaluations, `
      + 'split differently');
    sub.text('a bigger population means fewer generations, not more work');

    const best = data.population.best_pop;
    const first = rows.find((d) => d.success === data.meta.seeds);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>size</th>${rows.map((d) => `<th>${d.pop}</th>`).join('')}</tr>
        <tr><td>solved of ${data.meta.seeds}</td>
            ${rows.map((d) => `<td>${d.pop === best ? '<span class="value">' : ''}${d.success}${d.pop === best ? '</span>' : ''}</td>`).join('')}</tr>
        <tr><td>generations</td>
            ${rows.map((d) => `<td>${Math.floor(data.meta.budget / d.pop)}</td>`).join('')}</tr>
        <tr><td>evals, median</td>
            ${rows.map((d) => `<td>${d.median_evals === null ? 'never' : d.median_evals.toLocaleString('en-US')}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Everything in this row got the same
        ${data.meta.budget.toLocaleString('en-US')} evaluations, so the choice is
        between many short generations and few long ones. Below
        ${rows[2].pop} the population agrees with itself before it has found
        anything worth agreeing on and the run is over:
        ${rows[0].success} of ${data.meta.seeds} at ${rows[0].pop}.
        ${first
    ? `It first solves every seed at <span class="value">${first.pop}</span>,
           with only ${Math.floor(data.meta.budget / first.pop)} generations to
           work in.`
    : `Even the largest population here does not solve every seed.`}
        The median evaluations column is over the runs that succeeded only, so a
        small population with one lucky run gets a flattering number: read the
        first row.
      </div>`;
  }

  function renderPress() {
    const rows = data.takeover.rows;
    const x = d3.scalePoint().domain(rows.map((d) => String(d.tournament)))
      .range([0, w]).padding(0.5);
    const top = Math.max(...rows.map((d) => d.median ?? 0)) * 1.3;
    const y = d3.scaleLinear().domain([0, top]).range([h, 0]);
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5 });
    axisLabels(g, w, h, { x: 'how many compete for each slot',
      y: 'generations to fill the population', leftBudget: margin.left });
    plot.selectAll('*').remove();
    const drawn = rows.filter((d) => d.median !== null);
    plot.selectAll('rect').data(drawn).join('rect')
      .attr('x', (d) => x(String(d.tournament)) - 20).attr('width', 40)
      .attr('y', (d) => y(d.median)).attr('height', (d) => h - y(d.median))
      .attr('fill', 'var(--primary)');
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
      .attr('d', d3.line().x((d) => x(String(d.tournament))).y((d) => y(d.predicted))(drawn));
    rows.filter((d) => d.median === null).forEach((d) => {
      plot.append('text').attr('class', 'chart-note')
        .attr('x', x(String(d.tournament))).attr('y', h - 8).attr('text-anchor', 'middle')
        .style('font-size', '10.5px').text('never');
    });
    key.selectAll('*').remove();
    key.append('rect').attr('width', 12).attr('height', 12).attr('fill', 'var(--primary)');
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 26)
      .style('font-size', '10.5px').text('measured');
    key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 48).attr('y2', 48)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 64)
      .style('font-size', '10.5px').text('log of size');
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 75)
      .style('font-size', '10.5px').text('over log of');
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 86)
      .style('font-size', '10.5px').text('the contest');
    title.text('selection with nothing else running');
    sub.text(`${data.takeover.runs} runs each, population ${data.takeover.pop}, `
      + 'no mutation and no crossover');

    const one = rows[0];
    const ratios = rows.filter((d) => d.median !== null && d.predicted)
      .map((d) => d.median / d.predicted);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>contest</th>${rows.map((d) => `<th>${d.tournament}</th>`).join('')}</tr>
        <tr><td>generations</td>
            ${rows.map((d) => `<td>${d.median === null ? 'never' : d.median}</td>`).join('')}</tr>
        <tr><td>predicted</td>
            ${rows.map((d) => `<td>${d.predicted === null ? 'undefined' : d.predicted}</td>`).join('')}</tr>
        <tr><td>best lost</td>
            ${rows.map((d) => `<td>${d.best_lost} of ${d.of}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        With a contest of ${one.tournament} there is no selection at all: a slot
        is filled by whoever is drawn, so the best individual was
        <span class="value">lost in ${one.best_lost} of ${one.of} runs</span> and
        never took over in any. From ${rows[1].tournament} upwards the population
        converges in ${rows[1].median} generations down to
        ${rows[rows.length - 1].median}, against a textbook prediction of
        log(population) over log(contest) that says ${rows[1].predicted} down to
        ${rows[rows.length - 1].predicted}: the right shape, low by a factor of
        ${(ratios.reduce((a, b) => a + b, 0) / ratios.length).toFixed(2)} on
        average. And the column nobody prints is the last one: even at
        ${rows[1].tournament} the best individual is thrown away in
        ${rows[1].best_lost} runs of ${rows[1].of}, which is the entire argument
        for keeping one copy of it by hand.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    if (view === 'pop') renderPop(); else renderPress();
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
