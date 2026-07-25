/* The centrepiece: drag one point and watch four estimators disagree.
 *
 * The lesson is that WHERE a point sits matters more than how far it is. A
 * vertical outlier has neighbours to argue with it; a leverage point does not,
 * and Huber cannot see the difference because it judges by residual, which a
 * leverage point keeps small by winning. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { ols, theilSen, ransac, huber, leverage, rng } from './mathkit.js';

const N_CLEAN = 24;
const TRUE_SLOPE = 1.0;
const TRUE_INT = 1.0;

export function initLeverageLab() {
  // a fixed, reproducible clean cloud
  const rand = rng(2024);
  const gauss = () => {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const clean = [];
  for (let i = 0; i < N_CLEAN; i++) {
    const x = 1 + (8 * i) / (N_CLEAN - 1) + 0.18 * gauss();
    clean.push({ x, y: TRUE_INT + TRUE_SLOPE * x + 0.55 * gauss() });
  }

  const HOME = { x: 5, y: TRUE_INT + TRUE_SLOPE * 5 + 0.4 };
  const rogue = { ...HOME };
  // Companions matter: measured on the real star data, ONE leverage point only
  // bends Huber, but three reverse it outright. Robustness against leverage is
  // not a property Huber has, it is a margin it runs out of.
  const state = { companions: false };
  const companionsOf = () =>
    state.companions
      ? [
          { x: rogue.x - 0.55, y: rogue.y + 0.5 },
          { x: rogue.x + 0.5, y: rogue.y - 0.45 },
        ]
      : [];

  const { g, plot, w, h } = makeChart('#leverage-chart', {
    width: 620,
    height: 470,
    margin: { top: 30, right: 24, bottom: 58, left: 62 },
    clip: true,
  });

  const x = d3.scaleLinear().domain([0, 20]).range([0, w]);
  const y = d3.scaleLinear().domain([-4, 24]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'y' });

  const linesG = plot.append('g');
  const dotsG = g.append('g');

  const METHODS = [
    { id: 'clean', label: 'clean truth', color: 'var(--squidink)', dash: '8 6', width: 2.2 },
    { id: 'ols', label: 'ols', color: 'var(--cosmos)', dash: null, width: 3.2 },
    { id: 'huber', label: 'huber', color: 'var(--smile)', dash: null, width: 3.2 },
    { id: 'theilsen', label: 'theil-sen', color: 'var(--primary)', dash: null, width: 3.2 },
    { id: 'ransac', label: 'ransac', color: 'var(--galaxy)', dash: '3 4', width: 3 },
  ];
  const lines = {};
  for (const m of METHODS) {
    lines[m.id] = linesG
      .append('line')
      .attr('stroke', m.color)
      .attr('stroke-width', m.width)
      .attr('stroke-dasharray', m.dash);
  }

  dotsG
    .selectAll('circle.clean')
    .data(clean)
    .join('circle')
    .attr('class', 'clean')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 4.5)
    .attr('fill', 'var(--stone)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1);

  const companionDots = dotsG.append('g');

  const rogueDot = dotsG
    .append('circle')
    .attr('r', 11)
    .attr('fill', 'var(--cosmos)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2.5)
    .attr('cursor', 'grab');

  const hint = g
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('text-anchor', 'middle')
    .style('font-size', '0.64rem')
    .style('letter-spacing', '0.5px')
    .style('text-transform', 'none')
    .text('drag me');

  const statsEl = document.querySelector('#leverage-stats');

  function fits() {
    const extra = companionsOf();
    const xs = [...clean.map((d) => d.x), rogue.x, ...extra.map((d) => d.x)];
    const ys = [...clean.map((d) => d.y), rogue.y, ...extra.map((d) => d.y)];
    const cx = clean.map((d) => d.x);
    const cy = clean.map((d) => d.y);
    const spread = d3.deviation(ys) || 1;
    return {
      clean: ols(cx, cy),
      ols: ols(xs, ys),
      huber: huber(xs, ys, { delta: 1.35 }),
      theilsen: theilSen(xs, ys),
      ransac: ransac(xs, ys, { threshold: 0.9, iterations: 300, seed: 5 }),
      lev: leverage(xs),
      n: xs.length,
    };
  }

  function render() {
    const f = fits();
    const [x0, x1] = x.domain();
    for (const m of METHODS) {
      const fit = f[m.id];
      lines[m.id]
        .attr('x1', x(x0)).attr('y1', y(fit.intercept + fit.slope * x0))
        .attr('x2', x(x1)).attr('y2', y(fit.intercept + fit.slope * x1));
    }
    rogueDot.attr('cx', x(rogue.x)).attr('cy', y(rogue.y));
    hint.attr('x', x(rogue.x)).attr('y', y(rogue.y) - 18);

    const extra = companionsOf();
    companionDots
      .selectAll('circle')
      .data(extra)
      .join('circle')
      .attr('cx', (d) => x(d.x))
      .attr('cy', (d) => y(d.y))
      .attr('r', 9)
      .attr('fill', 'var(--cosmos)')
      .attr('opacity', 0.75)
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 2);

    const hRogue = f.lev[N_CLEAN];
    const cutoff = (2 * 2) / f.n;
    // report how far each method drifted as a share of the true slope, rather
    // than a binary verdict: the interesting behaviour is the gradient
    const drift = (fit) => (fit.slope - f.clean.slope) / Math.abs(f.clean.slope);
    const row = (m) => {
      const d = drift(f[m.id]);
      const pct = (d * 100).toFixed(0);
      const sev = Math.abs(d) > 0.5 ? 'var(--cosmos)' : Math.abs(d) > 0.15 ? 'var(--smile)' : 'inherit';
      const word = Math.abs(d) > 0.5 ? 'broken' : Math.abs(d) > 0.15 ? 'bending' : 'holding';
      return `<tr><td style="color:${m.color}">${m.label}</td>` +
        `<td>${f[m.id].slope.toFixed(2)}</td>` +
        `<td style="color:${sev}">${pct > 0 ? '+' : ''}${pct}% · ${word}</td></tr>`;
    };
    statsEl.innerHTML =
      `<div class="slider-label" style="margin-bottom:.5rem">leverage of the dragged point: ` +
      `<span class="value">${hRogue.toFixed(3)}</span> ` +
      `<span style="opacity:.7">(cutoff 2p/n = ${cutoff.toFixed(3)})</span>` +
      (hRogue > cutoff ? ' <span class="bold" style="color:var(--cosmos)">high</span>' : '') +
      `</div>` +
      `<table class="method-table">` +
      `<tr><th>method</th><th>slope</th><th>drift from ${f.clean.slope.toFixed(2)}</th></tr>` +
      METHODS.filter((m) => m.id !== 'clean').map(row).join('') +
      `</table>`;
  }

  rogueDot.call(
    d3
      .drag()
      .on('start', function () { d3.select(this).attr('cursor', 'grabbing'); hint.attr('opacity', 0); })
      .on('drag', (event) => {
        rogue.x = Math.max(x.domain()[0], Math.min(x.domain()[1], x.invert(event.x)));
        rogue.y = Math.max(y.domain()[0], Math.min(y.domain()[1], y.invert(event.y)));
        render();
      })
      .on('end', function () { d3.select(this).attr('cursor', 'grab'); })
  );

  const preset = (pos) => {
    rogue.x = pos.x;
    rogue.y = pos.y;
    hint.attr('opacity', 0);
    render();
  };
  document.querySelector('#lev-vertical').addEventListener('click', () => {
    state.companions = false;
    companionBtn.textContent = 'Give it two companions';
    preset({ x: 5, y: 20 });
  });
  document.querySelector('#lev-leverage').addEventListener('click', () => preset({ x: 18.5, y: 2 }));
  document.querySelector('#lev-reset').addEventListener('click', () => {
    state.companions = false;
    companionBtn.textContent = 'Give it two companions';
    preset(HOME);
  });

  const companionBtn = document.querySelector('#lev-companions');
  companionBtn.addEventListener('click', () => {
    state.companions = !state.companions;
    companionBtn.textContent = state.companions ? 'Send the companions away' : 'Give it two companions';
    companionBtn.classList.toggle('ghost', !state.companions);
    render();
  });

  render();
}
