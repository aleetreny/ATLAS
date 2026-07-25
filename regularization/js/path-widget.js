/* The regularization path plus the validation curve, driven by ONE slider so
 * the reader can see the trade in both views at once: which features are still
 * alive on the left, and whether that choice generalizes on the right. */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';
import { makeModel, predict, rmse, nonZero } from './mathkit.js';

const SERIES = [
  'var(--anchor)', 'var(--sky)', 'var(--cosmos)', 'var(--galaxy)', 'var(--aqua)',
  'var(--violet)', 'var(--magenta)', 'var(--seablue)', 'var(--smile)', '#8a5a2b',
];

export function initPathWidget(dia) {
  const alphas = d3.range(-3, 1.21, 0.035).map((v) => 10 ** v);
  const model = makeModel(dia.X, dia.y);

  const trainX = dia.train_idx.map((i) => dia.X[i]);
  const trainY = dia.train_idx.map((i) => dia.y[i]);
  const testX = dia.test_idx.map((i) => dia.X[i]);
  const testY = dia.test_idx.map((i) => dia.y[i]);
  const trainModel = makeModel(trainX, trainY);

  const state = { l1Ratio: 1 };
  let paths = null;
  let curve = null;

  function recompute() {
    paths = model.path(alphas, state.l1Ratio);
    const trainPaths = trainModel.path(alphas, state.l1Ratio);
    curve = trainPaths.map((b) => ({
      train: rmse(trainY, predict(trainX, b)),
      test: rmse(testY, predict(testX, b)),
    }));
  }
  recompute();

  const bestIdx = () => d3.minIndex(curve, (d) => d.test);

  /* ------------------------------ path chart ------------------------------ */
  const P = makeChart('#path-chart', {
    width: 620, height: 420, margin: { top: 30, right: 108, bottom: 58, left: 62 },
  });
  const px = d3.scaleLog().domain([alphas[0], alphas[alphas.length - 1]]).range([0, P.w]);
  const py = d3.scaleLinear().range([P.h, 0]);

  drawGrid(P.g, px, py, P.w, P.h);
  const pathLinesG = P.g.append('g');
  const pathRule = P.g.append('line').attr('y1', 0).attr('y2', P.h)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.5);
  const pathLabelsG = P.g.append('g');
  axisLabels(P.g, P.w, P.h, { x: 'penalty strength α (log)', y: 'coefficient' });

  const tip = tooltip();
  const lineGen = d3.line().x((d, i) => px(alphas[i])).y((d) => py(d));

  /* --------------------------- validation chart --------------------------- */
  const V = makeChart('#curve-chart', {
    width: 620, height: 300, margin: { top: 26, right: 108, bottom: 58, left: 62 },
  });
  const vx = d3.scaleLog().domain([alphas[0], alphas[alphas.length - 1]]).range([0, V.w]);
  const vy = d3.scaleLinear().range([V.h, 0]);
  drawGrid(V.g, vx, vy, V.w, V.h);
  const bestBand = V.g.append('rect').attr('y', 0).attr('height', V.h)
    .attr('fill', 'var(--lab)').attr('opacity', 0.18);
  const trainLine = V.g.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 2.5).attr('stroke-dasharray', '6 5');
  const testLine = V.g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 3.2);
  const curveRule = V.g.append('line').attr('y1', 0).attr('y2', V.h)
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.5);
  const vLabels = V.g.append('g');
  axisLabels(V.g, V.w, V.h, { x: 'penalty strength α (log)', y: 'RMSE' });

  const slider = document.querySelector('#path-alpha');
  const alphaLabel = document.querySelector('#path-alpha-label');
  const statsEl = document.querySelector('#path-stats');
  const btnL1 = document.querySelector('#path-l1');
  const btnL2 = document.querySelector('#path-l2');
  const btnBest = document.querySelector('#path-best');

  function renderStatic() {
    const flat = paths.flat();
    const ext = d3.extent(flat);
    const pad = (ext[1] - ext[0]) * 0.08;
    py.domain([ext[0] - pad, ext[1] + pad]);
    drawAxes(P.g, px, py, P.w, P.h, { xTicks: 5 });

    pathLinesG
      .selectAll('path')
      .data(dia.labels.map((_, j) => paths.map((b) => b[j])))
      .join('path')
      .attr('fill', 'none')
      .attr('stroke', (d, j) => SERIES[j])
      .attr('stroke-width', 2.6)
      .attr('opacity', 0.9)
      .attr('d', lineGen)
      .style('cursor', 'pointer')
      .on('mousemove', (event, d) => {
        const j = pathLinesG.selectAll('path').nodes().indexOf(event.currentTarget);
        tip.show(`<span style="color:var(--smile)">${dia.labels[j]}</span>`, event);
        pathLinesG.selectAll('path').attr('opacity', (dd, k) => (k === j ? 1 : 0.15));
      })
      .on('mouseleave', () => {
        tip.hide();
        pathLinesG.selectAll('path').attr('opacity', 0.9);
      });

    const vExt = [0, d3.max(curve, (d) => Math.max(d.train, d.test)) * 1.05];
    vy.domain(vExt);
    drawAxes(V.g, vx, vy, V.w, V.h, { xTicks: 5 });
    trainLine.attr('d', d3.line().x((d, i) => vx(alphas[i])).y((d) => vy(d.train))(curve));
    testLine.attr('d', d3.line().x((d, i) => vx(alphas[i])).y((d) => vy(d.test))(curve));

    const bi = bestIdx();
    bestBand.attr('x', vx(alphas[Math.max(0, bi - 2)]))
      .attr('width', Math.max(6, vx(alphas[Math.min(alphas.length - 1, bi + 2)]) - vx(alphas[Math.max(0, bi - 2)])));

    vLabels.selectAll('text').data(['train', 'held out']).join('text')
      .attr('class', 'chart-annotation')
      .attr('x', V.w + 8)
      .attr('y', (d, i) => (i === 0 ? vy(curve[curve.length - 1].train) : vy(curve[curve.length - 1].test)) + 4)
      .style('font-size', '0.68rem')
      .attr('fill', (d, i) => (i === 0 ? 'var(--squidink)' : 'var(--cosmos)'))
      .text((d) => d);
  }

  function renderAlpha() {
    const i = +slider.value;
    const alpha = alphas[i];
    alphaLabel.textContent = alpha.toPrecision(2);
    pathRule.attr('x1', px(alpha)).attr('x2', px(alpha));
    curveRule.attr('x1', vx(alpha)).attr('x2', vx(alpha));

    const b = paths[i];
    const alive = nonZero(b, 1e-8);

    // name the survivors right where their line is, only while they are alive
    pathLabelsG
      .selectAll('text')
      .data(dia.labels.map((l, j) => ({ l, j, v: b[j] })).filter((d) => Math.abs(d.v) > 1e-8))
      .join('text')
      .attr('class', 'chart-annotation')
      .attr('x', px(alpha) + 8)
      .attr('y', (d) => py(d.v) + 4)
      .attr('text-anchor', 'start')
      .style('font-size', '0.66rem')
      .attr('fill', (d) => SERIES[d.j])
      .text((d) => d.l);

    const bi = bestIdx();
    const c = curve[i];
    statsEl.innerHTML =
      `<div class="slider-label">features alive: <span class="value">${alive}/10</span></div>` +
      `<div class="slider-label">held-out RMSE: <span class="value">${c.test.toFixed(2)}</span>` +
      ` (best ${curve[bi].test.toFixed(2)} at α = ${alphas[bi].toPrecision(2)})</div>` +
      `<div class="slider-label" style="margin-top:.4rem">${
        i < bi - 6
          ? 'Too loose: the model is still chasing noise.'
          : i > bi + 6
            ? 'Too tight: real signal is being crushed.'
            : 'Right in the valley: this is the alpha you would ship.'
      }</div>`;
  }

  function setPenalty(ratio) {
    state.l1Ratio = ratio;
    btnL1.classList.toggle('ghost', ratio !== 1);
    btnL2.classList.toggle('ghost', ratio !== 0);
    recompute();
    renderStatic();
    renderAlpha();
  }

  btnL1.addEventListener('click', () => setPenalty(1));
  btnL2.addEventListener('click', () => setPenalty(0));
  btnBest.addEventListener('click', () => {
    slider.value = bestIdx();
    renderAlpha();
  });
  slider.max = alphas.length - 1;
  slider.value = Math.round(alphas.length * 0.28);
  slider.addEventListener('input', renderAlpha);

  // runtime check of the JS solver against the scikit-learn path
  const refA = dia.ref.alphas;
  const probe = Math.floor(refA.length / 2);
  const jsAt = model.fit(refA[probe], 1);
  const skAt = dia.ref.lasso[probe];
  const worst = d3.max(jsAt.map((v, j) => Math.abs(v - skAt[j])));
  if (worst > 0.25) console.warn(`lasso path mismatch vs sklearn: max |Δ| = ${worst.toFixed(3)}`);

  renderStatic();
  renderAlpha();
}
