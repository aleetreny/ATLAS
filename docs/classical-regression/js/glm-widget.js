/* Poisson GLM vs OLS on count data, trained live by gradient descent. */
import { makeChart, drawAxes, drawGrid, axisLabels, ols1d } from '../../assets/js/chart.js';
import { poissonGLM } from './mathkit.js';

export function initGLMWidget(tips) {
  const pts = tips.points;
  const bills = pts.map((d) => d.bill);
  const sizes = pts.map((d) => d.size);
  const model = poissonGLM(bills, sizes, 0.1);
  const ols = ols1d(bills, sizes);
  const sigmaOLS = Math.sqrt(
    d3.mean(pts, (d) => (d.size - (ols.intercept + ols.slope * d.bill)) ** 2)
  );

  const { g, w, h } = makeChart('#glm-chart', {
    width: 640,
    height: 480,
    margin: { top: 24, right: 26, bottom: 60, left: 60 },
  });

  const x = d3.scaleLinear().domain([0, 55]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 7]).range([h, 0]);

  drawGrid(g, x, y, w, h, { yTicks: 7 });
  drawAxes(g, x, y, w, h, { yTicks: 7, yFmt: d3.format('d') });
  axisLabels(g, w, h, { x: 'total bill ($)', y: 'party size' });

  const grid = d3.range(0, 55.5, 0.5);
  const areaG = g.append('g');
  const dotsG = g.append('g');
  const linesG = g.append('g');

  // deterministic jitter so the discrete bands stay visible but stable
  const jitter = (i) => ((((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * 0.34;

  dotsG
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('cx', (d) => x(d.bill))
    .attr('cy', (d, i) => y(d.size + jitter(i)))
    .attr('r', 3.8)
    .attr('fill', 'var(--stone)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.8)
    .attr('opacity', 0.85);

  const poissonBand = areaG.append('path').attr('fill', 'var(--rind)').attr('opacity', 0);
  const olsBand = areaG.append('path').attr('fill', 'var(--stone)').attr('opacity', 0);

  const olsLine = linesG
    .append('line')
    .attr('stroke', 'var(--cosmos)')
    .attr('stroke-width', 3)
    .attr('stroke-dasharray', '8 6');

  const haloCurve = linesG.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7);
  const poissonCurve = linesG.append('path').attr('fill', 'none').attr('stroke', 'var(--galaxy)').attr('stroke-width', 3.5);

  const lineGen = d3.line().x((d) => x(d[0])).y((d) => y(Math.max(0, Math.min(7, d[1]))));
  const bandGen = d3
    .area()
    .x((d) => x(d[0]))
    .y0((d) => y(Math.max(0, Math.min(7, d[1]))))
    .y1((d) => y(Math.max(0, Math.min(7, d[2]))));

  const eqEl = document.querySelector('#glm-eq');
  const statusEl = document.querySelector('#glm-status');
  const state = { showOLS: true, showBand: false };

  function render() {
    const nat = model.natural();

    poissonCurve.attr('d', lineGen(grid.map((v) => [v, model.predict(v)])));
    haloCurve.attr('d', poissonCurve.attr('d'));

    olsLine
      .attr('x1', x(0)).attr('y1', y(Math.max(0, ols.intercept)))
      .attr('x2', x(55)).attr('y2', y(Math.min(7, ols.intercept + ols.slope * 55)))
      .attr('opacity', state.showOLS ? 1 : 0);

    poissonBand
      .attr('d', bandGen(grid.map((v) => {
        const mu = model.predict(v);
        const s = 1.96 * Math.sqrt(mu);
        return [v, Math.max(0, mu - s), mu + s];
      })))
      .attr('opacity', state.showBand ? 0.55 : 0);

    olsBand
      .attr('d', bandGen(grid.map((v) => {
        const mu = ols.intercept + ols.slope * v;
        return [v, mu - 1.96 * sigmaOLS, mu + 1.96 * sigmaOLS];
      })))
      .attr('opacity', state.showBand && state.showOLS ? 0.45 : 0);

    if (window.katex) {
      katex.render(
        `\\hat{\\mu} = e^{${nat.b0.toFixed(3)} + ${nat.b1.toFixed(4)} \\cdot \\text{bill}}`,
        eqEl,
        { displayMode: true, throwOnError: false }
      );
    }
    const ref = tips.ref.poisson;
    const converged =
      Math.abs(nat.b0 - ref.intercept) < 0.01 && Math.abs(nat.b1 - ref.coef) < 0.0005;
    statusEl.innerHTML =
      `iteration <span class="value">${model.iter}</span> · ` +
      `neg. log-likelihood <span class="value">${model.nll().toFixed(4)}</span>` +
      (converged
        ? ' · <span class="bold" style="color:var(--smile)">converged — matches statsmodels ✓</span>'
        : model.iter === 0
          ? ' · untrained: a flat guess of e⁰ = 1 diner'
          : '');
  }

  function animateSteps(total) {
    const chunk = Math.max(1, Math.round(total / 40));
    let done = 0;
    const timer = d3.timer(() => {
      model.step(Math.min(chunk, total - done));
      done += chunk;
      render();
      if (done >= total) timer.stop();
    });
  }

  document.querySelector('#glm-step1').addEventListener('click', () => { model.step(1); render(); });
  document.querySelector('#glm-step25').addEventListener('click', () => animateSteps(25));
  document.querySelector('#glm-step500').addEventListener('click', () => animateSteps(500));
  document.querySelector('#glm-reset').addEventListener('click', () => { model.reset(); render(); });
  document.querySelector('#glm-ols-toggle').addEventListener('change', (e) => { state.showOLS = e.target.checked; render(); });
  document.querySelector('#glm-band-toggle').addEventListener('change', (e) => { state.showBand = e.target.checked; render(); });

  render();
}
