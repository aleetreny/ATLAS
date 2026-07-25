/* Polynomial degree knob: honest train/test scoring, overfitting live. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { polyFit, r2Score } from './mathkit.js';

export function initPolyWidget(mpg) {
  const pts = mpg.points;
  const testSet = new Set(mpg.ref.test_idx);
  const train = pts.filter((_, i) => !testSet.has(i));
  const test = pts.filter((_, i) => testSet.has(i));

  // deterministic shuffle of the training pool, so the "starve the model"
  // slider always removes/restores the same cars (stable, replayable)
  const order = train
    .map((d, i) => [((i * 2654435761) >>> 0) % 9973, i])
    .sort((a, b) => a[0] - b[0])
    .map(([, i]) => i);

  const { svg, g, w, h } = makeChart('#poly-chart', {
    width: 640,
    height: 480,
    margin: { top: 24, right: 26, bottom: 60, left: 60 },
  });

  const x = d3.scaleLinear().domain([35, 235]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 50]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });

  // clip so degree-9 wiggles don't escape the plot
  const clipId = 'poly-clip';
  svg.append('clipPath').attr('id', clipId).append('rect').attr('width', w).attr('height', h);

  const trainDots = g
    .append('g')
    .selectAll('circle.train')
    .data(train)
    .join('circle')
    .attr('class', 'train')
    .attr('cx', (d) => x(d.hp))
    .attr('cy', (d) => y(d.mpg))
    .attr('r', 3.8)
    .attr('fill', 'var(--smile)')
    .attr('stroke', 'white')
    .attr('stroke-width', 0.8)
    .attr('opacity', 0.85);

  g.append('g')
    .selectAll('circle.test')
    .data(test)
    .join('circle')
    .attr('class', 'test')
    .attr('cx', (d) => x(d.hp))
    .attr('cy', (d) => y(d.mpg))
    .attr('r', 4.2)
    .attr('fill', 'none')
    .attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 2);

  const curveG = g.append('g').attr('clip-path', `url(#${clipId})`);
  const haloPath = curveG.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 7);
  const curvePath = curveG.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 3.5);

  const grid = d3.range(36, 234.5, 1);
  const lineGen = d3
    .line()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]))
    .curve(d3.curveMonotoneX);

  const barsEl = document.querySelector('#poly-r2-bars');
  const verdictEl = document.querySelector('#poly-verdict');
  const degLabel = document.querySelector('#poly-deg-label');
  const nLabel = document.querySelector('#poly-n-label');
  const state = { degree: 1, n: train.length };

  function verdictFor(deg, n, gap, r2Test) {
    if (r2Test < 0) return 'Below zero: worse than guessing the mean. Total memorization.';
    if (gap > 0.15) return `Overfitting: flexibility the ${n} cars cannot discipline.`;
    if (deg === 1) return 'Underfitting: the ruler cannot bend.';
    if (deg === 2) return 'The sweet spot — the curve speaks physics.';
    if (deg <= 4) return 'Still fine, but the extra wiggle buys almost nothing.';
    if (n === train.length) return 'High degree, but 294 cars keep the wiggles disciplined.';
    return 'The tail starts dancing. Fewer cars, more temptation.';
  }

  function render() {
    const { degree, n } = state;
    degLabel.textContent = degree;
    nLabel.textContent = n;
    const active = new Set(order.slice(0, n));
    const sub = train.filter((_, i) => active.has(i));

    trainDots.attr('opacity', (d, i) => (active.has(i) ? 0.85 : 0.12));

    const model = polyFit(sub.map((d) => d.hp), sub.map((d) => d.mpg), degree);
    const r2Train = r2Score(sub.map((d) => d.mpg), sub.map((d) => model.predict(d.hp)));
    const r2Test = r2Score(test.map((d) => d.mpg), test.map((d) => model.predict(d.hp)));

    // sanity check against the Python reference (full training set only)
    if (n === train.length) {
      const ref = mpg.ref.poly_degrees.find((r) => r.degree === degree);
      if (ref && Math.abs(ref.r2_train - r2Train) > 0.02) {
        console.warn(`poly degree ${degree}: JS r2_train ${r2Train.toFixed(4)} vs ref ${ref.r2_train}`);
      }
    }

    const pathData = lineGen(grid.map((v) => [v, model.predict(v)]));
    haloPath.transition().duration(350).attr('d', pathData);
    curvePath.transition().duration(350).attr('d', pathData);

    const bar = (label, val, color) => {
      const width = Math.max(0, Math.min(1, val)) * 100;
      const shown = val < -10 ? '≪ 0 (broken)' : val.toFixed(3);
      return (
        `<div class="slider-label" style="margin-top:.6rem">${label} r² = ` +
        `<span class="value" style="color:${color}">${shown}</span></div>` +
        `<div style="height:12px;background:var(--stone);max-width:260px">` +
        `<div style="height:12px;width:${width}%;background:${color};transition:width .35s"></div></div>`
      );
    };
    barsEl.innerHTML =
      bar('train', r2Train, 'var(--smile)') + bar('test&nbsp;', r2Test, 'var(--anchor)');
    verdictEl.textContent = verdictFor(degree, n, r2Train - r2Test, r2Test);
  }

  document.querySelector('#poly-deg').addEventListener('input', (e) => {
    state.degree = +e.target.value;
    render();
  });
  document.querySelector('#poly-n').addEventListener('input', (e) => {
    state.n = +e.target.value;
    render();
  });
  render();
}
