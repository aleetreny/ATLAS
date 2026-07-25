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
    .attr('fill', 'var(--primary)')
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

  /* The verdict has to be earned, not hardcoded to a degree. Calling degree 2
   * "the sweet spot" was a lie the widget itself refuted: on all 294 cars,
   * degree 7 scores 0.683 on held-out data against degree 2's 0.659, and both
   * numbers are printed two lines above the verdict. So sweep the degrees at
   * the current sample size and phrase everything relative to what actually
   * wins there. */
  function bestDegreeAt(sub) {
    const xs = sub.map((d) => d.hp);
    const ys = sub.map((d) => d.mpg);
    const ty = test.map((d) => d.mpg);
    let best = { degree: 1, r2: -Infinity };
    for (let deg = 1; deg <= 9; deg++) {
      const m = polyFit(xs, ys, deg);
      const s = r2Score(ty, test.map((d) => m.predict(d.hp)));
      if (s > best.r2) best = { degree: deg, r2: s };
    }
    return best;
  }

  function verdictFor(deg, n, gap, r2Test, best) {
    /* A straight line cannot overfit, whatever the train/test gap says. With
     * thirty cars the gap test fired on degree 1 and the widget called a ruler
     * "overfitting", so the degree check has to come first. */
    if (deg === 1 && deg !== best.degree) return 'Underfitting: the ruler cannot bend.';
    if (r2Test < 0) return 'Below zero: worse than guessing the mean. Total memorization.';
    if (gap > 0.15) return `Overfitting: flexibility the ${n} cars cannot discipline.`;
    const behind = best.r2 - r2Test;
    if (deg === best.degree) return `Nothing beats this on the held-out cars, at ${n} of them.`;
    if (behind < 0.01) return `As good as it gets here: degree ${best.degree} wins by ${behind.toFixed(3)}.`;
    if (deg === 1) return 'Underfitting: the ruler cannot bend.';
    if (deg < best.degree) return `More bend still pays: degree ${best.degree} scores ${behind.toFixed(3)} higher.`;
    return `Past the peak. Degree ${best.degree} scores ${behind.toFixed(3)} higher with less flexibility.`;
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

    /* Sanity check against the Python reference (full training set only).
     * The tolerance used to be 0.02, which was wide enough to swallow a
     * genuinely wrong degree-9 fit: 0.7048 against a reference of 0.7102.
     * Solved by QR the agreement is around 1e-4, so anything past 0.002 is a
     * real divergence and deserves to be shouted about. Grade the held-out
     * score too, since that is the number the reader is being asked to trust. */
    if (n === train.length) {
      const ref = mpg.ref.poly_degrees.find((r) => r.degree === degree);
      if (ref && Math.abs(ref.r2_train - r2Train) > 0.002) {
        console.warn(`poly degree ${degree}: JS r2_train ${r2Train.toFixed(4)} vs ref ${ref.r2_train}`);
      }
      if (ref && Math.abs(ref.r2_test - r2Test) > 0.002) {
        console.warn(`poly degree ${degree}: JS r2_test ${r2Test.toFixed(4)} vs ref ${ref.r2_test}`);
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
      bar('train', r2Train, 'var(--primary)') + bar('test&nbsp;', r2Test, 'var(--anchor)');
    verdictEl.textContent = verdictFor(degree, n, r2Train - r2Test, r2Test, bestDegreeAt(sub));
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
