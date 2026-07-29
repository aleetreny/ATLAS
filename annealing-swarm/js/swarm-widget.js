/* The cloud, and what it is worth.
 *
 * One view is the swarm itself over a function with a dent every unit, drawn
 * from the positions the generator recorded; the other is the scoreboard
 * against two searchers that are not swarms, on the same number of function
 * evaluations.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { rastrigin } from './sakit.js';

const VIEWS = [
  { key: 'cloud', label: 'the swarm moving' },
  { key: 'score', label: 'against two controls' },
];
const ARM_LABEL = { pso: 'the swarm', local: 'local search, restarts', random: 'random guessing' };

export function initSwarmWidget(data) {
  const S = data.shape;
  const W = data.swarm;
  const readout = document.querySelector('#swarm-readout');
  const controls = d3.select('#swarm-controls');
  const slider = document.querySelector('#swarm-step');
  const label = document.querySelector('#swarm-step-label');
  let view = 'cloud';

  const chart = makeChart('#swarm-chart', {
    width: 640, height: 400, margin: { top: 48, right: 30, bottom: 78, left: 64 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  /* the contour map is drawn once: a grid of squares shaded by the value of the
     function, which is cheap and reads better than contour lines at this size */
  const B = S.bound;
  const gridN = 60;
  const cloudX = d3.scaleLinear().domain([-B, B]).range([0, w]);
  const cloudY = d3.scaleLinear().domain([-B, B]).range([h, 0]);
  const field = [];
  for (let i = 0; i < gridN; i++) {
    for (let j = 0; j < gridN; j++) {
      const xv = -B + (2 * B * i) / (gridN - 1);
      const yv = -B + (2 * B * j) / (gridN - 1);
      field.push({ xv, yv, v: rastrigin(xv, yv) });
    }
  }
  const shade = d3.scaleLinear().domain([0, d3.max(field, (d) => d.v)])
    .range(['#ffffff', '#c9ced0']);

  function renderCloud() {
    const k = Number(slider.value);
    const pts = S.frames[k];
    drawGrid(g, cloudX, cloudY, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, cloudX, cloudY, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: 'first coordinate', y: 'second coordinate',
      leftBudget: margin.left });
    plot.selectAll('rect').data(field).join('rect')
      .attr('x', (d) => cloudX(d.xv) - w / (2 * gridN))
      .attr('y', (d) => cloudY(d.yv) - h / (2 * gridN))
      .attr('width', w / gridN + 1).attr('height', h / gridN + 1)
      .attr('fill', (d) => shade(d.v)).attr('opacity', 0.9);
    plot.selectAll('circle.p').data(pts).join('circle').attr('class', 'p')
      .attr('cx', (p) => cloudX(Math.max(-B, Math.min(B, p[0]))))
      .attr('cy', (p) => cloudY(Math.max(-B, Math.min(B, p[1]))))
      .attr('r', 4.5).attr('fill', 'var(--primary)').attr('fill-opacity', 0.85);
    plot.selectAll('circle.best').data([S.best]).join('circle').attr('class', 'best')
      .attr('cx', (p) => cloudX(p[0])).attr('cy', (p) => cloudY(p[1]))
      .attr('r', 7).attr('fill', 'none').attr('stroke', 'var(--cosmos)')
      .attr('stroke-width', 2.2);
    const outside = pts.filter((p) => Math.abs(p[0]) > B || Math.abs(p[1]) > B).length;
    title.text(`${S.particles} particles over a bowl with a dent every unit`);
    sub.text(`iteration ${k}: the cloud is ${S.spreads[k]} wide`
      + (outside ? `, ${outside} of them off the edge of this picture` : ''));
    label.textContent = `${k} of ${S.steps - 1}`;

    const shrink = S.spreads[0] / S.spreads[S.spreads.length - 1];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>iteration</th>${[0, 10, 20, 30, 40, 50, S.steps - 1].map((i) => `<th>${i}</th>`).join('')}</tr>
        <tr><td>width of the cloud</td>
            ${[0, 10, 20, 30, 40, 50, S.steps - 1].map((i) => `<td>${S.spreads[i]}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Each particle carries a velocity and adds two pulls to it every step, one
        towards the best place it has been and one towards the best place anybody
        has been. Nothing else. Over ${S.steps} iterations the cloud narrows from
        <span class="value">${S.spreads[0]}</span> to
        <span class="value">${S.spreads[S.spreads.length - 1]}</span>, a factor of
        ${shrink.toFixed(0)}, and it narrows whether or not it is anywhere good:
        that is the failure mode of the method and the reason every variant of it
        is about keeping the cloud open longer. Here it ends at
        (${S.best[0]}, ${S.best[1]}) with a value of ${S.best_value}, against an
        optimum of 0 at the origin.
      </div>`;
  }

  function renderScore() {
    const names = Object.keys(W.bench);
    const x = d3.scalePoint().domain(names).range([0, w]).padding(0.5);
    const vals = names.flatMap((n) => W.bench[n].rows.map((r) => Math.max(r.median, 1e-7)));
    const y = d3.scaleLog().domain([Math.min(...vals) * 0.5, Math.max(...vals) * 2])
      .range([h, 0]);
    const yt = [1e-6, 1e-4, 1e-2, 1, 100].filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { yValues: yt });
    drawAxes(g, x, y, w, h, { yValues: yt, yFmt: d3.format('.0e') });
    axisLabels(g, w, h, { y: 'best value found, median', leftBudget: margin.left });
    plot.selectAll('*').remove();
    const arms = ['pso', 'local', 'random'];
    const colours = { pso: 'var(--primary)', local: 'var(--anchor)', random: '#c9ced0' };
    arms.forEach((arm, ai) => {
      plot.selectAll(`rect.${arm}`).data(names).join('rect').attr('class', arm)
        .attr('x', (n) => x(n) - 33 + ai * 22).attr('width', 20)
        .attr('y', (n) => y(Math.max(W.bench[n].rows.find((r) => r.arm === arm).median, 1e-7)))
        .attr('height', (n) => h - y(Math.max(W.bench[n].rows.find((r) => r.arm === arm).median, 1e-7)))
        .attr('fill', colours[arm]);
      plot.append('rect').attr('x', ai * 150).attr('y', h + 40).attr('width', 11)
        .attr('height', 11).attr('fill', colours[arm]);
      plot.append('text').attr('class', 'chart-note').attr('x', ai * 150 + 15)
        .attr('y', h + 50).style('font-size', '10.5px').text(ARM_LABEL[arm]);
    });
    title.text(`four test functions in ${W.dim} dimensions, `
      + `${W.budget.toLocaleString('en-US')} evaluations each`);
    sub.text(`${W.seeds} runs per cell, bars are the median; lower is better`);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>function</th>${names.map((n) => `<th>${n}</th>`).join('')}</tr>
        ${arms.map((arm) => `
          <tr><td>${ARM_LABEL[arm]}</td>
              ${names.map((n) => {
    const row = W.bench[n].rows.find((r) => r.arm === arm);
    const best = Math.min(...W.bench[n].rows.map((r) => r.median));
    return `<td>${row.median === best ? '<span class="value">' : ''}${row.median}${row.median === best ? '</span>' : ''}</td>`;
  }).join('')}</tr>`).join('')}
      </table>
      <div class="gen-note">
        Two of these four go the other way, and that is the useful part. On the
        lattice of dents the swarm is
        ${(W.bench.rastrigin.rows.find((r) => r.arm === 'local').median
          / W.bench.rastrigin.rows.find((r) => r.arm === 'pso').median).toFixed(1)}
        times better than the local search, because a cloud that shares one best
        position does not have to walk out of a dent to leave it. In the narrow
        curved valley it is
        ${(W.bench.rosenbrock.rows.find((r) => r.arm === 'pso').median
          / W.bench.rosenbrock.rows.find((r) => r.arm === 'local').median).toFixed(0)}
        times worse, because following a curved floor needs small careful steps
        and a swarm does not take those. Random guessing is the floor everywhere
        and never close, which is the only thing on this page that all four
        functions agree about.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    plot.selectAll('*').remove();
    d3.select('#swarm-step').attr('disabled', view === 'score' ? 'disabled' : null);
    if (view === 'cloud') renderCloud(); else renderScore();
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  slider.addEventListener('input', render);
  render();
  return { render };
}
