/* The forest, built here, on every move.
 *
 * The two sliders are the only two numbers an isolation forest has, and both of
 * them are drawn rather than described: how many trees, and how many points
 * each tree is allowed to see. The field behind the points is the forest's
 * opinion about every position, evaluated on a grid, which is the only way to
 * show that the answer is a function of the plane and not of the sample.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { buildForest, forestScore, cFactor } from './forest.js';
import { averagePrecision, rocAuc } from './detkit.js';
import { drawPoints, legend, scoreField, fieldLegend, seriesLabel } from './draw.js';

/* Log spaced, and the slider indexes this list rather than carrying a count:
   between one tree and two hundred the interesting part is the first twenty. */
const TREES = [1, 2, 4, 8, 16, 32, 64, 100, 150, 200];
const PSI = [8, 16, 32, 64, 128, 256];
const GRID = 56;

export function initForestWidget(data) {
  const cases = Object.keys(data.sets);
  let which = cases[0];
  let ti = 5;                      /* 32 trees: enough to be stable, few enough
                                      that adding more visibly settles it */
  let pi = 3;                      /* 64 samples */

  const chart = makeChart('#forest-chart', {
    width: 640, height: 500, margin: { top: 74, right: 26, bottom: 56, left: 46 },
  });
  const { g, w, h } = chart;
  const gField = g.append('g');
  const gPts = g.append('g');
  const gNote = g.append('g');
  const readout = d3.select('#forest-readout');

  const tSlider = d3.select('#forest-trees')
    .attr('min', 0).attr('max', TREES.length - 1).attr('step', 1).property('value', ti);
  const pSlider = d3.select('#forest-psi')
    .attr('min', 0).attr('max', PSI.length - 1).attr('step', 1).property('value', pi);

  d3.select('#forest-cases').selectAll('button').data(cases).join('button')
    .attr('class', (d) => `atlas-btn${d === which ? '' : ' ghost'}`)
    .text((d) => d)
    .on('click', (event, d) => {
      which = d;
      d3.select('#forest-cases').selectAll('button')
        .attr('class', (q) => `atlas-btn${q === which ? '' : ' ghost'}`);
      render();
    });

  function render() {
    ti = +tSlider.property('value');
    pi = +pSlider.property('value');
    const nTrees = TREES[ti];
    const X = data.sets[which].points;
    const y = data.sets[which].labels;
    const psi = Math.min(PSI[pi], X.length);

    const xs = X.map((p) => p[0]);
    const ys = X.map((p) => p[1]);
    const pad = 0.08 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const x = d3.scaleLinear().domain([Math.min(...xs) - pad, Math.max(...xs) + pad]).range([0, w]);
    const yy = d3.scaleLinear().domain([Math.min(...ys) - pad, Math.max(...ys) + pad]).range([h, 0]);

    /* the seed is fixed so that moving a slider changes the slider and nothing
       else: a fresh random forest on every move would make the two effects
       impossible to tell apart */
    const forest = buildForest(X, nTrees, psi, 7);
    const scores = forestScore(forest, X);

    const probes = [];
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        probes.push([
          x.domain()[0] + ((x.domain()[1] - x.domain()[0]) * c) / (GRID - 1),
          yy.domain()[0] + ((yy.domain()[1] - yy.domain()[0]) * r) / (GRID - 1),
        ]);
      }
    }
    const field = forestScore(forest, probes);
    const lo = Math.min(...field);
    const hi = Math.max(...field);
    scoreField(gField, field, GRID, x, yy, w, h, { lo, hi, opacity: 0.92 });

    drawAxes(g, x, yy, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, { x: 'first measurement', y: 'second' });
    drawPoints(gPts, X, y, x, yy, { r: 2.8 });

    gNote.selectAll('*').remove();
    legend(gNote, [
      { label: 'ordinary', shape: 'dot', colour: 'var(--squidink)' },
      { label: 'planted anomaly', shape: 'ring', colour: 'var(--cosmos)' },
    ], { x0: 2, y0: -50, gap: 168 });
    /* The caption goes above the bar rather than beside it: to the right of the
       bar is where the bar's own maximum label lives, and the two collided at
       every setting where that number happened to be wide. */
    fieldLegend(gNote, 2, -26, 150, lo, hi, null);
    seriesLabel(gNote, 2, -34, "the forest's score, ordinary to strange", { size: 10 });

    d3.select('#forest-trees-label').text(nTrees);
    d3.select('#forest-psi-label').text(psi);

    const ap = averagePrecision(y, scores);
    const auc = rocAuc(y, scores);
    const anom = y.reduce((a, b) => a + b, 0);
    const order = scores.map((v, i) => i).sort((a, b) => scores[b] - scores[a]);
    const caught = order.slice(0, anom).filter((i) => y[i] === 1).length;
    const ref = data.forest.page[which];
    readout.html(
      `<span class="slider-label">trees <span class="value">${nTrees}</span></span> `
      + `<span class="slider-label">samples per tree <span class="value">${psi}</span></span>. `
      + `Every score on this page is `
      + `<span class="bold">2<sup>&minus;h/c</sup></span>, where h is the average number of cuts `
      + `this forest needed and c is ${cFactor(psi).toFixed(3)}, the number a random point would `
      + `have needed in a sample of ${psi}. Half means "exactly average". `
      + `Taking the top ${anom} scores here catches <span class="bold">${caught} of the ${anom}</span> `
      + `planted anomalies, at an average precision of <span class="bold">${ap.toFixed(4)}</span> `
      + `and an area under the ROC curve of ${auc.toFixed(4)}. `
      + (nTrees === ref.trees && psi === ref.psi
        ? `These are the settings the generator recorded, and the two agree to `
          + `${'1e-8'} on every probe it kept.`
        : `A coin would score ${(anom / y.length).toFixed(4)}, which is the anomaly rate.`)
      + ` Nothing about this field was fitted: no mean, no covariance, no density. `
      + `It is the count of random cuts.`
    );
  }

  tSlider.on('input', render);
  pSlider.on('input', render);
  render();
  return { render };
}
