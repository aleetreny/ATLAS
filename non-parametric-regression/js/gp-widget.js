/* The Gaussian process: a model that reports what it does not know.
 *
 * Click to place observations and watch the credible band collapse where you
 * put them and bloom where you did not. Changing the kernel changes the prior
 * belief about how the world behaves between observations, which is the whole
 * modelling decision in one control. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { gpPredict, rng } from './mathkit.js';

const TRUTH = (v) => Math.sin(v * 1.6) + v * 0.25;

export function initGpWidget() {
  const { svg, g, w, h } = makeChart('#gp-chart', {
    width: 680,
    height: 430,
    margin: { top: 30, right: 24, bottom: 58, left: 62 },
  });
  const x = d3.scaleLinear().domain([0, 10]).range([0, w]);
  const y = d3.scaleLinear().domain([-3.2, 4.6]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'y' });

  g.append('clipPath').attr('id', 'gp-clip').append('rect').attr('width', w).attr('height', h);
  const plot = g.append('g').attr('clip-path', 'url(#gp-clip)');

  const band2 = plot.append('path').attr('fill', 'var(--primary)').attr('fill-opacity', 0.12);
  const band1 = plot.append('path').attr('fill', 'var(--primary)').attr('fill-opacity', 0.2);
  const truthPath = plot.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2).attr('stroke-dasharray', '7 6').attr('opacity', 0.4);
  const meanHalo = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const meanPath = plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);
  const dotsG = g.append('g');

  const hint = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', 22).attr('text-anchor', 'middle')
    .style('letter-spacing', '0.5px').style('text-transform', 'none')
    .text('click anywhere to place an observation');

  const grid = d3.range(0, 10.001, 0.05);
  const truthLine = d3.line().x((v) => x(v)).y((v) => y(TRUTH(v)));
  truthPath.attr('d', truthLine(grid));

  const rand = rng(11);
  const obs = [];
  const state = { kernel: 'rbf', length: 1.2, noise: 0.25 };

  const lineGen = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
  const areaGen = d3.area().x((d) => x(d[0])).y0((d) => y(d[1])).y1((d) => y(d[2]));

  const statsEl = document.querySelector('#gp-stats');

  function render() {
    const { mean, sd } = gpPredict(
      obs.map((o) => o.x), obs.map((o) => o.y), grid,
      { kernel: state.kernel, length: state.length, noise: state.noise, amplitude: 1.6 }
    );
    band2.attr('d', areaGen(grid.map((v, i) => [v, mean[i] - 2 * sd[i], mean[i] + 2 * sd[i]])));
    band1.attr('d', areaGen(grid.map((v, i) => [v, mean[i] - sd[i], mean[i] + sd[i]])));
    const d = lineGen(grid.map((v, i) => [v, mean[i]]));
    meanHalo.attr('d', d);
    meanPath.attr('d', d);

    dotsG.selectAll('circle').data(obs).join('circle')
      .attr('cx', (o) => x(o.x)).attr('cy', (o) => y(o.y)).attr('r', 5.5)
      .attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 1.6);

    hint.attr('opacity', obs.length ? 0 : 1);

    const avgSd = d3.mean(sd);
    const maxSd = d3.max(sd);
    const worstAt = grid[sd.indexOf(maxSd)];
    statsEl.innerHTML =
      `<div class="slider-label">observations: <span class="value">${obs.length}</span>` +
      ` · average uncertainty <span class="value">${avgSd.toFixed(2)}</span></div>` +
      `<div class="slider-label">least confident at x = <span class="value">${worstAt.toFixed(1)}</span>` +
      ` (±${(2 * maxSd).toFixed(2)})</div>` +
      `<div class="slider-label" style="margin-top:.4rem;line-height:1.45">${
        obs.length === 0
          ? 'With no data the posterior is the prior: a flat mean and a band of constant width. It knows nothing, and says so.'
          : obs.length < 4
            ? 'The band pinches shut at each observation and swells between them. That shape is the model reporting where it is guessing.'
            : 'Every other method on this page would answer with equal confidence everywhere. This one will not.'
      }</div>`;
  }

  svg.style('cursor', 'crosshair').on('click', (event) => {
    const [px, py] = d3.pointer(event, g.node());
    if (px < 0 || px > w || py < 0 || py > h) return;
    obs.push({ x: x.invert(px), y: y.invert(py) });
    render();
  });

  document.querySelector('#gp-clear').addEventListener('click', () => {
    obs.length = 0;
    render();
  });
  document.querySelector('#gp-sample').addEventListener('click', () => {
    obs.length = 0;
    for (let i = 0; i < 7; i++) {
      const xv = 0.6 + rand() * 8.8;
      obs.push({ x: xv, y: TRUTH(xv) + (rand() - 0.5) * 0.5 });
    }
    render();
  });
  const lenSlider = document.querySelector('#gp-length');
  lenSlider.addEventListener('input', (e) => {
    state.length = +e.target.value;
    document.querySelector('#gp-length-label').textContent = state.length.toFixed(2);
    render();
  });
  for (const [k, id] of [['rbf', '#gp-rbf'], ['matern', '#gp-matern'], ['periodic', '#gp-periodic']]) {
    document.querySelector(id).addEventListener('click', () => {
      state.kernel = k;
      for (const [k2, id2] of [['rbf', '#gp-rbf'], ['matern', '#gp-matern'], ['periodic', '#gp-periodic']]) {
        document.querySelector(id2).classList.toggle('ghost', k2 !== k);
      }
      render();
    });
  }

  render();
}
