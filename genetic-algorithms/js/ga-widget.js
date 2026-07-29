/* A hundred generations, run in this page.
 *
 * Not a recording: the loop in gakit.js runs when you move a control, with the
 * same random number generator the generator used. At the settings it opens on
 * it fails, and it fails at the number the trap was built to attract it to,
 * which is the point. The controls that fix it are all here.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { runGA, fitnessOf } from './gakit.js';

const POPS = [25, 50, 100, 200, 400];
const MUTS = [0, 0.25, 0.5, 1, 2, 3, 4, 6, 8];      // multiples of 1 / bits
const CROSSES = [
  { key: 'one_point', label: 'one point' },
  { key: 'uniform', label: 'uniform' },
  { key: 'none', label: 'no crossover' },
];
const LANDS = [
  { key: 'trap_tight', label: 'traps side by side' },
  { key: 'trap_loose', label: 'traps shuffled' },
];

export function initGaWidget(data) {
  const P = data.probe;
  const base = data.reference.config;
  const readout = document.querySelector('#ga-readout');
  const popSlider = document.querySelector('#ga-pop');
  const mutSlider = document.querySelector('#ga-mut');
  const popLabel = document.querySelector('#ga-pop-label');
  const mutLabel = document.querySelector('#ga-mut-label');
  let cross = base.cross;
  let land = 'trap_tight';

  const chart = makeChart('#ga-chart', {
    width: 640, height: 360, margin: { top: 46, right: 124, bottom: 52, left: 60 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLinear().domain([0, base.gens]).range([0, w]);
  const y = d3.scaleLinear().domain([0, P.optima.trap_tight * 1.05]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'generation', y: 'fitness', leftBudget: margin.left });

  const optLine = g.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', y(P.optima.trap_tight)).attr('y2', y(P.optima.trap_tight))
    .attr('stroke', '#8e9aa6').attr('stroke-width', 1.4);
  const trapLine = g.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', y(P.deceptive_attractor)).attr('y2', y(P.deceptive_attractor))
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  const bestPath = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 2.6);
  const meanPath = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 1.4).attr('opacity', 0.55);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -28)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -11)
    .style('font-size', '11.5px');

  const key = g.append('g').attr('transform', `translate(${w + 8}, 4)`);
  /* the two lines of each entry are written out rather than split on spaces:
     splitting put "the" alone on one line and "trap 30" on the next */
  [['var(--primary)', ['best of the', 'population'], 2.6, null, 1],
    ['var(--primary)', ['the population', 'mean'], 1.4, null, 0.55],
    ['#8e9aa6', ['the optimum,', `${P.optima.trap_tight}`], 1.4, null, 1],
    ['var(--cosmos)', ['the trap,', `${P.deceptive_attractor}`], 1.4, '5 4', 1]]
    .forEach(([col, lines, sw, dash, op], i) => {
      key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 32 + 6).attr('y2', i * 32 + 6)
        .attr('stroke', col).attr('stroke-width', sw).attr('stroke-dasharray', dash)
        .attr('opacity', op);
      lines.forEach((line, k) => {
        key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 32 + 19 + k * 11)
          .style('font-size', '10.5px').text(line);
      });
    });

  function render() {
    const pop = POPS[Number(popSlider.value)];
    const mult = MUTS[Number(mutSlider.value)];
    const pmut = mult / base.bits;
    const cfg = { ...base, pop, cross, pmut };
    const fit = fitnessOf(land, P.block, P.shuffle);
    const run = runGA(cfg, fit);
    const hist = run.history;
    const evals = pop * (base.gens + 1);

    bestPath.attr('d', d3.line().x((d) => x(d.gen)).y((d) => y(d.best))(hist));
    meanPath.attr('d', d3.line().x((d) => x(d.gen)).y((d) => y(d.mean))(hist));
    popLabel.textContent = `${pop}`;
    mutLabel.textContent = mult === 0 ? 'none' : `${mult}/${base.bits}`;
    const solved = run.finalBest >= P.optima.trap_tight;
    title.text(`${pop} guesses, ${base.gens} generations, `
      + `${evals.toLocaleString('en-US')} evaluations`);
    sub.text(solved ? `solved: ${run.finalBest} of ${P.optima.trap_tight}`
      : `best ${run.finalBest} of ${P.optima.trap_tight}, `
        + `${run.finalBest > P.deceptive_attractor ? 'above' : 'at or below'} the trap`);

    const last = hist[hist.length - 1];
    const first = hist.find((d) => d.best === run.finalBest);
    const landLabel = LANDS.find((d) => d.key === land).label;
    const crossLabel = CROSSES.find((d) => d.key === cross).label;
    const table = data.arms[land].rows.find((r) => r.arm
      === (cross === 'none' ? 'mutation_only' : cross));
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>this run</th><th>evaluations</th><th>best</th>
            <th>mean</th><th>distinct</th></tr>
        <tr><td>${crossLabel}, ${landLabel}</td>
            <td>${evals.toLocaleString('en-US')}</td>
            <td><span class="value">${run.finalBest}</span> of ${P.optima.trap_tight}</td>
            <td>${last.mean.toFixed(2)}</td>
            <td>${last.unique} of ${pop}</td></tr>
      </table>
      <div class="gen-note">
        ${solved
    ? `Solved, first reaching ${run.finalBest} at generation ${first.gen}.`
    : `Stuck at <span class="value">${run.finalBest}</span>, with the population
           averaging ${last.mean.toFixed(2)} against the
           ${P.deceptive_attractor} the trap pays for getting every block exactly
           wrong. ${last.mean <= P.deceptive_attractor + 1
      ? 'That average is the trap: the population has agreed on the answer every single bit flip recommends.'
      : 'The population is above the trap but has not assembled all the blocks.'}`}
        The same settings measured over ${table.of} seeds at a budget of
        ${data.meta.budget.toLocaleString('en-US')} evaluations succeed
        <span class="value">${table.success} of ${table.of}</span> times with a
        population of ${data.meta.pop}, so one run is a run: what a control is
        worth has to be read off the table below, not off this line.
      </div>`;
  }

  const buttons = (sel, items, current, onPick) => {
    d3.select(sel).selectAll('button').data(items).join('button')
      .attr('class', (d) => `atlas-btn${d.key === current() ? '' : ' ghost'}`)
      .text((d) => d.label)
      .on('click', (_, d) => {
        onPick(d.key);
        d3.select(sel).selectAll('button')
          .attr('class', (q) => `atlas-btn${q.key === current() ? '' : ' ghost'}`);
        render();
      });
  };
  buttons('#ga-cross', CROSSES, () => cross, (k) => { cross = k; });
  buttons('#ga-land', LANDS, () => land, (k) => { land = k; });
  popSlider.addEventListener('input', render);
  mutSlider.addEventListener('input', render);
  render();
  return { render };
}
