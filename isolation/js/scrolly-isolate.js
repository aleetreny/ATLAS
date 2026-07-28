/* The opening story: one tree, two points, and a count of cuts.
 *
 * The tree is built in the page from the shared generator, so the cuts drawn
 * here are the cuts a real isolation tree made on this data, not a diagram of
 * what one might look like.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { LCG, buildForest, tracePath, treeCuts, cFactor } from './forest.js';
import { drawPoints, legend, seriesLabel } from './draw.js';

export function initScrollyIsolate(data) {
  const S = data.sets['two densities'];
  const X = S.points;
  const y = S.labels;

  /* One tree, from a seed chosen so that both stories are visible in it: the
     anomaly falls out early and the ordinary point does not. The seed is the
     only thing picked by hand here, and it is picked by measuring. */
  const forest = buildForest(X, 1, 128, 4242);
  const tree = forest.trees[0];
  const anomIdx = y.findIndex((v, i) => v === 1 && tracePath(tree, X[i]).depth <= 3);
  const anom = anomIdx >= 0 ? anomIdx : y.indexOf(1);
  const normIdx = y.findIndex((v, i) => v === 0 && tracePath(tree, X[i]).depth >= 5);
  const norm = normIdx >= 0 ? normIdx : y.indexOf(0);
  const traceA = tracePath(tree, X[anom]);
  const traceN = tracePath(tree, X[norm]);

  const chart = makeChart('#isolate-chart', {
    width: 640, height: 470, margin: { top: 60, right: 26, bottom: 56, left: 46 },
  });
  const { g, w, h } = chart;
  const xs = X.map((p) => p[0]);
  const ys = X.map((p) => p[1]);
  const pad = 0.6;
  const x = d3.scaleLinear().domain([Math.min(...xs) - pad, Math.max(...xs) + pad]).range([0, w]);
  const yy = d3.scaleLinear().domain([Math.min(...ys) - pad, Math.max(...ys) + pad]).range([h, 0]);
  const bounds = { x0: x.domain()[0], x1: x.domain()[1], y0: yy.domain()[0], y1: yy.domain()[1] };
  const allCuts = treeCuts(tree, bounds);

  const gCuts = g.append('g');
  const gPts = g.append('g');
  const gMark = g.append('g');
  const gNote = g.append('g');

  drawAxes(g, x, yy, w, h, { xTicks: 6, yTicks: 6 });
  axisLabels(g, w, h, { x: 'first measurement', y: 'second' });
  legend(g.append('g'), [
    { label: 'ordinary', shape: 'dot', colour: 'var(--squidink)' },
    { label: 'planted anomaly', shape: 'ring', colour: 'var(--cosmos)' },
    { label: 'a random cut', shape: 'line', colour: 'var(--primary)' },
  ], { x0: 2, y0: -34, gap: 168 });

  function seg(cut) {
    const b = cut.box;
    if (cut.f === 0) {
      return { x1: x(cut.s), x2: x(cut.s), y1: yy(Math.min(b.y1, bounds.y1)), y2: yy(Math.max(b.y0, bounds.y0)) };
    }
    return { x1: x(Math.max(b.x0, bounds.x0)), x2: x(Math.min(b.x1, bounds.x1)), y1: yy(cut.s), y2: yy(cut.s) };
  }

  function render(state) {
    const cuts = state.cuts;
    const sel = gCuts.selectAll('line.cut').data(cuts, (d, i) => i);
    sel.exit().remove();
    sel.enter().append('line').attr('class', 'cut').merge(sel)
      .attr('x1', (d) => seg(d).x1).attr('x2', (d) => seg(d).x2)
      .attr('y1', (d) => seg(d).y1).attr('y2', (d) => seg(d).y2)
      .attr('stroke', 'var(--primary)')
      .attr('stroke-width', (d) => (state.thin ? 0.8 : 1.6))
      .attr('opacity', (d) => (state.thin ? 0.35 : 0.9));

    drawPoints(gPts, X, y, x, yy, {
      r: 3, highlight: state.focus === null ? null : (i) => i === state.focus,
    });

    gMark.selectAll('*').remove();
    if (state.focus !== null) {
      const p = X[state.focus];
      gMark.append('circle')
        .attr('cx', x(p[0])).attr('cy', yy(p[1])).attr('r', 10)
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      const t = state.focus === anom ? traceA : traceN;
      const b = t.box;
      gMark.append('rect')
        .attr('x', x(Math.max(b.x0, bounds.x0)))
        .attr('y', yy(Math.min(b.y1, bounds.y1)))
        .attr('width', Math.abs(x(Math.min(b.x1, bounds.x1)) - x(Math.max(b.x0, bounds.x0))))
        .attr('height', Math.abs(yy(Math.max(b.y0, bounds.y0)) - yy(Math.min(b.y1, bounds.y1))))
        .attr('fill', 'var(--primary)').attr('opacity', 0.12)
        .attr('stroke', 'var(--primary)').attr('stroke-dasharray', '4 3');
    }

    gNote.selectAll('*').remove();
    if (state.note) seriesLabel(gNote, 0, -12, state.note, { size: 11.5, opacity: 0.95 });
  }

  const STATES = [
    { cuts: [], focus: null, thin: false, note: '' },
    { cuts: allCuts.slice(0, 1), focus: null, thin: false, note: 'one cut, chosen at random' },
    { cuts: traceA.cuts, focus: anom, thin: false,
      note: `the ringed point is alone after ${traceA.depth} cuts` },
    { cuts: traceN.cuts, focus: norm, thin: false,
      note: `an ordinary point takes ${traceN.depth}` },
    { cuts: allCuts, focus: null, thin: true,
      note: `the whole tree: ${allCuts.length} cuts, depth limit ${forest.limit}` },
  ];

  render(STATES[0]);
  const handlers = {};
  STATES.forEach((s, i) => {
    handlers[String(i)] = () => render(s);
  });
  return {
    scrolly: scrolly('#isolate-scrolly .step', handlers),
    facts: {
      anomDepth: traceA.depth,
      normDepth: traceN.depth,
      anomLength: traceA.length,
      normLength: traceN.length,
      anomLeaf: traceA.leafSize,
      normLeaf: traceN.leafSize,
      cuts: allCuts.length,
      limit: forest.limit,
      cPsi: cFactor(forest.psi),
      psi: forest.psi,
    },
  };
}
