/* The loop under the reader's finger: assign, update, run, reseed.
 *
 * The two step buttons enforce the actual alternation (assign is a no-op twice
 * in a row, and the readout says so) because the discipline of the loop IS the
 * algorithm. The inertia log keeps every value of the current run on screen,
 * since "it only ever falls" is a claim about the whole sequence.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { assign, update, initRandom, rng } from './kmeanskit.js';
import { drawClusteredPoints, drawCenters, drawBorders } from './clusterdraw.js';

export function initStepWidget(data) {
  const pts = data.blobs.points;
  const { g, w, h } = makeChart('#st-chart', {
    width: 560,
    height: 500,
    margin: { top: 30, right: 24, bottom: 54, left: 58 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const borderG = g.append('g');
  const dotG = g.append('g');
  const centreG = g.append('g');

  let seed = 7;
  const st = { centers: null, labels: null, log: [], phase: 'assign', done: false };
  const statsEl = document.querySelector('#st-stats');

  function reseed() {
    st.centers = initRandom(pts, 3, rng(seed));
    seed += 13;
    st.labels = null;
    st.log = [];
    st.phase = 'assign';
    st.done = false;
    render('new start: three random points are the centres');
  }

  function doAssign() {
    if (st.phase !== 'assign') {
      render('assign changes nothing until the centres move: update first');
      return;
    }
    const { labels, inertia } = assign(pts, st.centers);
    const prev = st.log.length ? st.log[st.log.length - 1] : Infinity;
    st.labels = labels;
    st.log.push(inertia);
    st.done = Math.abs(prev - inertia) < 1e-9;
    st.phase = 'update';
    render(st.done ? 'nothing changed: converged' : 'every point joined its nearest centre');
  }

  function doUpdate() {
    if (st.phase !== 'update') {
      render('update needs an assignment to average: assign first');
      return;
    }
    const means = update(pts, st.labels, 3);
    st.centers = means.map((m, c) => (m === null ? st.centers[c] : m));
    st.phase = 'assign';
    render('each centre moved to the mean of its points');
  }

  function runAll() {
    let guard = 0;
    while (!st.done && guard++ < 200) {
      if (st.phase === 'assign') doAssign();
      else doUpdate();
    }
    render(`converged after ${st.log.length} assignments`);
  }

  function render(msg) {
    drawClusteredPoints(dotG, pts, st.labels, x, y, { unlabelled: st.labels === null });
    drawCenters(centreG, st.centers, x, y);
    drawBorders(borderG, st.labels ? st.centers : [], x, y, w, h);

    const hist = st.log.map((v, i) => `${i ? ' → ' : ''}${v.toFixed(0)}`).join('');
    statsEl.innerHTML =
      `<div class="slider-label">inertia so far: <span class="value">${st.log.length ? hist : 'no assignment yet'}</span></div>` +
      `<div class="slider-label" style="margin-top:.5rem">next legal step: <span class="value">${st.done ? 'none, it is done' : st.phase}</span></div>` +
      `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">${msg}</div>` +
      (st.done
        ? `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">Converged at ` +
          `<span class="value">${st.log[st.log.length - 1].toFixed(1)}</span>. The good optimum on this data is ` +
          `<span class="value">${data.blobs.ref_inertia}</span>: ${
            Math.abs(st.log[st.log.length - 1] - data.blobs.ref_inertia) / data.blobs.ref_inertia < 0.01
              ? 'this run found it.'
              : 'this run is stuck somewhere worse, and no amount of further looping will fix it. That is the next section.'
          }</div>`
        : '');
  }

  document.querySelector('#st-assign').addEventListener('click', doAssign);
  document.querySelector('#st-update').addEventListener('click', doUpdate);
  document.querySelector('#st-run').addEventListener('click', runAll);
  document.querySelector('#st-reseed').addEventListener('click', reseed);
  reseed();
}
