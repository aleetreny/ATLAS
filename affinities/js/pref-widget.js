/* The two dials of affinity propagation.
 *
 * Left: the preference, the price of opening a cluster, on the exact grid the
 * generator swept so that the curve underneath and the live run are the same
 * experiment. Right: the damping, which is supposed to be a numerical detail.
 *
 * The whole algorithm re-runs in the page at every position of both sliders,
 * on all 300 points, which is where the numbers in the prose come from too.
 * That costs about a fifth of a second at 300 points and there is nothing
 * clever to be done about it: n squared message updates per iteration is the
 * article's subject. It is rendered straight from the input event rather than
 * through requestAnimationFrame, because a frame callback never fires when the
 * page is not being composited and the widget would look dead to an audit.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';
import { similarity, affinityPropagation, ari } from './graphkit.js';
import { drawPoints, drawExemplars, legend, colourOf } from './draw.js';

/* Enough iterations to show both outcomes: every run that settles at all
   settles well inside this, and one that does not is visibly still swinging. */
const LIVE_ITERS = 220;

export function initPrefWidget(data) {
  const cases = Object.keys(data.ap.sweep);
  const sim = {};
  const state = { name: cases[0], p: 0, d: 0 };

  const main = makeChart('#pref-chart', {
    width: 640, height: 400, margin: { top: 34, right: 26, bottom: 44, left: 52 },
  });
  const x = d3.scaleLinear().range([0, main.w]);
  const y = d3.scaleLinear().range([main.h, 0]);
  const dotG = main.g.append('g');
  const exG = main.g.append('g');
  legend(main.g, [{ colour: 'var(--squidink)', shape: 'ring', label: 'exemplars' }],
    { x0: 4, y0: -16, gap: 130 });

  /* Two panels under the picture: the swept staircase, and the trace of how
     many points are electing themselves at each iteration of the live run. */
  const cur = makeChart('#pref-curve', {
    width: 640, height: 250, margin: { top: 26, right: 150, bottom: 52, left: 56 },
  });
  const cx = d3.scaleLog().range([0, cur.w]);
  const cy = d3.scaleLog().range([cur.h, 0]);
  const ay = d3.scaleLinear().domain([-0.05, 1.02]).range([cur.h, 0]);
  const kPath = cur.g.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2);
  const aPath = cur.g.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--aqua)').attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
  const truthLine = cur.g.append('line')
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
    .attr('stroke-dasharray', '3 3').attr('opacity', 0.5);
  const marker = cur.g.append('line').attr('y1', 0).attr('y2', cur.h)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2).attr('opacity', 0.9);
  const liveDot = cur.g.append('circle').attr('r', 4.5).attr('fill', 'var(--primary)')
    .attr('stroke', 'white').attr('stroke-width', 1.5);
  const defLine = cur.g.append('line').attr('y1', 0).attr('y2', cur.h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('opacity', 0.75);
  const defLabel = cur.g.append('text').attr('class', 'chart-annotation')
    .attr('text-anchor', 'end').style('font-size', '0.62rem');
  const rightLabels = cur.g.append('g');

  const trace = makeChart('#pref-curve', {
    width: 640, height: 182, margin: { top: 26, right: 150, bottom: 56, left: 56 },
  });
  const tx = d3.scaleLinear().range([0, trace.w]);
  const ty = d3.scaleLog().range([trace.h, 0]);
  const tPath = trace.g.append('path').attr('fill', 'none')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 1.8);
  const tLabel = trace.g.append('text').attr('class', 'chart-annotation')
    .attr('x', trace.w + 8).attr('dy', '0.32em').style('font-size', '0.62rem');

  const pSlider = document.querySelector('#pref-p');
  const dSlider = document.querySelector('#pref-d');
  const readout = document.querySelector('#pref-readout');
  const damps = data.ap.damping.grid;
  dSlider.min = 0;
  dSlider.max = damps.length - 1;
  dSlider.step = 1;
  dSlider.value = damps.indexOf(0.9) >= 0 ? damps.indexOf(0.9) : damps.length - 2;

  /* dataset buttons */
  const wrap = document.querySelector('#pref-cases');
  cases.forEach((name) => {
    const b = document.createElement('button');
    b.className = `atlas-btn${name === state.name ? '' : ' ghost'}`;
    b.textContent = name;
    b.addEventListener('click', () => {
      state.name = name;
      wrap.querySelectorAll('button').forEach((o) => {
        o.className = `atlas-btn${o.textContent === name ? '' : ' ghost'}`;
      });
      setCase();
      render();
    });
    wrap.appendChild(b);
  });

  function simOf(name) {
    if (!sim[name]) {
      const pts = data.datasets[name].points;
      sim[name] = { pts, n: pts.length, S: similarity(pts) };
    }
    return sim[name];
  }

  function setCase() {
    const sw = data.ap.sweep[state.name];
    const { pts } = simOf(state.name);
    pSlider.min = 0;
    pSlider.max = sw.grid.length - 1;
    pSlider.step = 1;
    /* Opens where a reader who typed no arguments would land, which is the
       state the paragraph above is about, not at the answer. */
    pSlider.value = sw.grid.length - 1;
    x.domain(d3.extent(pts, (p) => p[0])).nice();
    y.domain(d3.extent(pts, (p) => p[1])).nice();
    drawGrid(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
    drawAxes(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });

    /* the swept staircase: k against the price of a cluster, both on log
       scales because both span two orders of magnitude */
    const mags = sw.grid.map((p) => -p);
    cx.domain([Math.max(...mags), Math.min(...mags)]);
    cy.domain([0.9, Math.max(...sw.k) * 1.15]);
    /* Powers of ten and their triples, and nothing else. d3.ticks on the
       exponent gives 316.228 and 31.6228, which nobody reads as a price, and
       the log scale's own ticks come back as eighteen values that print
       "1k 900 800 700 600 500 400 300 200 100" on top of each other. */
    const nice = [];
    for (let e = -2; e <= 7; e++) nice.push(10 ** e, 3 * 10 ** e);
    const XT = nice.filter((v) => v >= Math.min(...mags) && v <= Math.max(...mags))
      .sort((a, b) => a - b);
    /* 1, 2, 3, 5 crowd into the bottom eighth of a log axis and the labels
       touch; three of them are enough to read a staircase by. */
    const YT = [1, 2, 5, 10, 20].filter((v) => v <= Math.max(...sw.k) * 1.15);
    drawGrid(cur.g, cx, cy, cur.w, cur.h, { xValues: XT, yValues: YT });
    drawAxes(cur.g, cx, cy, cur.w, cur.h, {
      xValues: XT, yValues: YT, xFmt: d3.format('~s'), yFmt: d3.format('d'),
    });
    axisLabels(cur.g, cur.w, cur.h, { x: 'price of a cluster, |preference|', y: 'clusters found' });
    /* The ARI rides a second scale, so it gets a second axis. A series drawn
       against a scale nobody can read is decoration. */
    let gr = cur.g.select('g.axis.right');
    if (gr.empty()) gr = cur.g.append('g').attr('class', 'axis right');
    gr.attr('transform', `translate(${cur.w},0)`)
      .call(d3.axisRight(ay).tickValues([0, 0.5, 1]).tickFormat(d3.format('.1f')).tickSizeOuter(0));
    const line = d3.line().x((d, i) => cx(-sw.grid[i])).y((d) => cy(Math.max(d, 0.9)));
    kPath.datum(sw.k).attr('d', line);
    const aline = d3.line().x((d, i) => cx(-sw.grid[i])).y((d) => ay(Math.max(d, 0)));
    aPath.datum(sw.ari).attr('d', aline);
    truthLine.attr('x1', 0).attr('x2', cur.w)
      .attr('y1', cy(data.datasets[state.name].k)).attr('y2', cy(data.datasets[state.name].k));
    defLine.attr('x1', cx(-sw.grid[sw.grid.length - 1])).attr('x2', cx(-sw.grid[sw.grid.length - 1]));
    /* Along the bottom, not the top: at the top it lies across the two curves
       and the series labels at the right edge. */
    defLabel.attr('x', cx(-sw.grid[sw.grid.length - 1]) - 6).attr('y', cur.h - 8)
      .text('the library default');
    rightLabels.selectAll('text').data(spread([
      { t: `clusters`, c: 'var(--primary)', y: cy(sw.k[sw.k.length - 1]), yMax: cur.h },
      { t: `ARI, right axis`, c: 'var(--aqua)', y: ay(sw.ari[sw.ari.length - 1]), yMax: cur.h },
      { t: `true k = ${data.datasets[state.name].k}`, c: 'var(--squidink)', y: cy(data.datasets[state.name].k), yMax: cur.h },
    ], 16)).join('text').attr('class', 'chart-annotation')
      .attr('x', cur.w + 34).attr('y', (d) => d.y).attr('dy', '0.32em')
      .style('font-size', '0.6rem').style('letter-spacing', '0.5px')
      .style('fill', (d) => d.c).text((d) => d.t);
  }

  function render() {
    const sw = data.ap.sweep[state.name];
    const { pts, n, S } = simOf(state.name);
    const p = sw.grid[+pSlider.value];
    const damping = damps[+dSlider.value];
    const truth = data.datasets[state.name].true_labels;
    const trueK = data.datasets[state.name].k;

    const run = affinityPropagation(S, n, {
      preference: p, damping, maxIter: LIVE_ITERS, trace: true,
    });
    const a = run.exemplars.length ? ari(truth, run.labels) : 0;

    document.querySelector('#pref-p-label').textContent = p.toLocaleString('en-US');
    document.querySelector('#pref-d-label').textContent = damping.toFixed(2);

    drawPoints(dotG, pts, run.exemplars.length ? run.labels : null, x, y, { r: 3.4 });
    drawExemplars(exG, run.exemplars.map((i) => pts[i]), x, y, { r: 9 });

    marker.attr('x1', cx(-p)).attr('x2', cx(-p));
    liveDot.attr('cx', cx(-p)).attr('cy', cy(Math.max(run.exemplars.length, 0.9)));

    tx.domain([1, Math.max(20, run.counts.length)]);
    ty.domain([0.9, Math.max(10, ...run.counts) * 1.1]);
    const tTicks = [1, 2, 5, 10, 30, 100, 300].filter((v) => v <= Math.max(10, ...run.counts) * 1.1);
    drawGrid(trace.g, tx, ty, trace.w, trace.h, { xTicks: 5, yValues: tTicks });
    drawAxes(trace.g, tx, ty, trace.w, trace.h, {
      xTicks: 5, yValues: tTicks, xFmt: d3.format('d'), yFmt: d3.format('d'),
    });
    axisLabels(trace.g, trace.w, trace.h, { x: 'iteration', y: 'self-elected' });
    tPath.datum(run.counts).attr('d', d3.line()
      .x((d, i) => tx(i + 1)).y((d) => ty(Math.max(d, 0.9))));
    tLabel.attr('y', ty(Math.max(run.counts[run.counts.length - 1], 0.9)))
      .style('fill', run.converged ? 'var(--primary)' : 'var(--cosmos)')
      .text(run.converged ? `settled at ${run.exemplars.length}` : 'still swinging');

    const off = data.ap.default[state.name];
    const swK = sw.k[+pSlider.value];
    const agrees = swK === run.exemplars.length;
    readout.innerHTML =
      `<table class="aff-table">
        <tr><th></th><th>clusters</th><th>ARI against the truth</th><th>iterations</th></tr>
        <tr><td>here, at these two dials</td>
            <td><span class="value">${run.exemplars.length}</span></td>
            <td><span class="value">${a.toFixed(3)}</span></td>
            <td>${run.converged ? run.iters : `${LIVE_ITERS}, without settling`}</td></tr>
        <tr><td>at the library default preference</td>
            <td><span class="value">${off.k}</span></td>
            <td><span class="value">${off.ari.toFixed(3)}</span></td>
            <td>${off.iters}</td></tr>
        <tr><td>the truth</td><td><span class="value">${trueK}</span></td>
            <td>1.000</td><td></td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      (run.converged
        ? `${run.exemplars.length === trueK
          ? `Right number, and nobody said it: `
          : `${run.exemplars.length > trueK ? 'Too many' : 'Too few'}: `}`
          + `every one of those ${run.exemplars.length} exemplars is a point you could look up in `
          + `the table the data came from.`
        : `The messages never settled. What is drawn is iteration ${LIVE_ITERS} of a run that is `
          + `still moving, so it is not an answer, it is a snapshot.`)
      + ` ${agrees
        ? `The offline sweep at damping 0.90 agrees on ${swK}.`
        : `The curve says ${swK} at this preference, because it was swept at damping 0.90 and `
          + `this run is at ${damping.toFixed(2)}: the damping changed the answer, not just the `
          + `time it took.`}`
      + `</div>`;
  }

  pSlider.addEventListener('input', render);
  dSlider.addEventListener('input', render);
  setCase();
  render();
  return { render };
}
