/* The second-order picture XGBoost introduced, drawn.
 *
 * Move the threshold and watch the two leaf weights and the split gain change.
 * The point is that with a second-order approximation, "what should this leaf
 * say?" and "is this split worth making?" both become formulas rather than
 * searches, and lambda and gamma appear inside them as explicit prices. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { makeData, boost, ensemblePredict, splitGain } from './mathkit.js';

export function initGainWidget() {
  const { xs, ys } = makeData(70, 9);
  // work on the residual after a couple of rounds, which is what a real tree
  // deep in the ensemble would actually be handed
  const warm = boost(xs, ys, { rounds: 2, lr: 0.4 });
  const base = ensemblePredict(warm, xs, 2);
  const res = ys.map((v, i) => v - base[i]);

  const { g, w, h } = makeChart('#gain-chart', {
    width: 620, height: 400, margin: { top: 30, right: 24, bottom: 56, left: 58 },
  });
  const x = d3.scaleLinear().domain([-0.2, 9.8]).range([0, w]);
  const y = d3.scaleLinear().domain([-4, 4]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'x', y: 'residual handed to this tree' });

  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('opacity', 0.5);

  const shadeL = g.append('rect').attr('y', 0).attr('height', h).attr('fill', 'var(--anchor)').attr('fill-opacity', 0.07);
  const shadeR = g.append('rect').attr('y', 0).attr('height', h).attr('fill', 'var(--primary)').attr('fill-opacity', 0.07);
  const dots = g.append('g').selectAll('circle').data(xs).join('circle')
    .attr('cx', (v) => x(v)).attr('cy', (v, i) => y(res[i])).attr('r', 3.8)
    .attr('stroke', 'white').attr('stroke-width', 0.8);
  const wL = g.append('line').attr('stroke', 'var(--anchor)').attr('stroke-width', 3.4);
  const wR = g.append('line').attr('stroke', 'var(--primary)').attr('stroke-width', 3.4);
  const thrLine = g.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.5).attr('stroke-dasharray', '5 4');

  const thrSlider = document.querySelector('#gain-thr');
  const lamSlider = document.querySelector('#gain-lambda');
  const gamSlider = document.querySelector('#gain-gamma');
  const eqEl = document.querySelector('#gain-eq');
  const statsEl = document.querySelector('#gain-stats');

  function render() {
    const thr = +thrSlider.value;
    const lambda = +lamSlider.value;
    const gamma = +gamSlider.value;
    document.querySelector('#gain-thr-label').textContent = thr.toFixed(1);
    document.querySelector('#gain-lambda-label').textContent = lambda.toFixed(1);
    document.querySelector('#gain-gamma-label').textContent = gamma.toFixed(1);

    const leftRes = [];
    const rightRes = [];
    xs.forEach((v, i) => (v <= thr ? leftRes : rightRes).push(res[i]));
    if (!leftRes.length || !rightRes.length) return;

    const s = splitGain(leftRes, rightRes, lambda, gamma);
    dots.attr('fill', (v) => (v <= thr ? 'var(--anchor)' : 'var(--primary)'));
    shadeL.attr('x', 0).attr('width', Math.max(0, x(thr)));
    shadeR.attr('x', x(thr)).attr('width', Math.max(0, w - x(thr)));
    thrLine.attr('x1', x(thr)).attr('x2', x(thr));
    wL.attr('x1', 0).attr('x2', x(thr)).attr('y1', y(s.wL)).attr('y2', y(s.wL));
    wR.attr('x1', x(thr)).attr('x2', w).attr('y1', y(s.wR)).attr('y2', y(s.wR));

    if (window.katex) {
      katex.render(
        `\\text{gain} = \\tfrac{1}{2}\\left[\\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{G^2}{H+\\lambda}\\right] - \\gamma = ${s.gain.toFixed(2)}`,
        eqEl,
        { displayMode: true, throwOnError: false }
      );
    }
    statsEl.innerHTML =
      `<table class="gb-table">
        <tr><th></th><th style="color:var(--anchor)">left leaf</th><th style="color:var(--primary)">right leaf</th></tr>
        <tr><td>points</td><td>${leftRes.length}</td><td>${rightRes.length}</td></tr>
        <tr><td>G (sum of gradients)</td><td>${s.L.G.toFixed(1)}</td><td>${s.R.G.toFixed(1)}</td></tr>
        <tr><td>H (sum of hessians)</td><td>${s.L.H.toFixed(0)}</td><td>${s.R.H.toFixed(0)}</td></tr>
        <tr><td>weight w* = −G/(H+λ)</td><td>${s.wL.toFixed(3)}</td><td>${s.wR.toFixed(3)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">${
        s.gain <= 0
          ? 'Gain is not positive, so this split does not pay for itself and the tree stops here. That is pruning as arithmetic, not a heuristic.'
          : gamma > 0 && s.gain < 3
            ? 'Barely worth it. Raise γ a little further and the split disappears.'
            : 'A profitable split: the two halves explain enough more than the whole to cover γ.'
      }</div>`;
  }

  for (const el of [thrSlider, lamSlider, gamSlider]) el.addEventListener('input', render);
  render();
}
