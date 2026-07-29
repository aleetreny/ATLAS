/* Heterogeneous effects, scored against a function that was written down.
 *
 * The map view is the one that earns its place: a scoreboard says the centred
 * forest wins, and the map says what winning looks like, which is that the
 * shape of tau(V) is visible in the predictions rather than merely correlated
 * with them.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const f1 = (v) => v.toFixed(1);

export function initHetWidget(data) {
  const host = document.querySelector('#het-widget');
  if (!host) return;
  const H = data.heterogeneity;
  const C = data.cloud;
  const P = data.prognostic;
  const chart = makeChart('#het-chart', {
    width: 640, height: 420, margin: { top: 44, right: 30, bottom: 58, left: 70 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#het-readout');
  const views = d3.select('#het-views');
  let view = 'map';

  const buttons = [
    ['map', 'predicted against true, point by point'],
    ['tree', 'what one honest tree found'],
    ['prognostic', 'when the outcome is easy to predict'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const by = Object.fromEntries(H.rows.map((r) => [r.key, r]));

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    if (view === 'map') {
      const keys = ['cf', 's', 't'];
      const labels = { cf: 'causal forest', s: 'S-learner', t: 'T-learner' };
      const colours = { cf: 'var(--primary)', s: 'var(--cosmos)', t: 'var(--aqua)' };
      const all = keys.flatMap((k) => C.pred[k]);
      const lo = Math.min(d3.min(C.tau), d3.min(all));
      const hi = Math.max(d3.max(C.tau), d3.max(all));
      const x = d3.scaleLinear().domain([lo, hi]).nice().range([0, w]);
      const y = d3.scaleLinear().domain([lo, hi]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'the true effect for this person', y: 'what the method said' });
      g.append('line')
        .attr('x1', x(lo)).attr('y1', y(lo)).attr('x2', x(hi)).attr('y2', y(hi))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 4');
      keys.forEach((k) => {
        g.append('g').selectAll('circle')
          .data(C.tau.map((tv, i) => [tv, C.pred[k][i]]))
          .join('circle')
          .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 2.4)
          .attr('fill', colours[k]).attr('fill-opacity', 0.5);
      });
      let cursor = 0;
      keys.forEach((k) => {
        const label = annotate(g, cursor, -14, `${labels[k]}: ${f4(C.mse[k])}`,
          { anchor: 'start' }).style('font-size', '12px').attr('fill', colours[k]);
        cursor += label.node().getComputedTextLength() + 22;
      });
      readout.innerHTML =
        `<p>Each dot is one held out person: the true effect for them on the horizontal `
        + `axis, what a method said on the vertical. A perfect method would put every dot on `
        + `the dashed line. The forest tracks the shape, mean squared error `
        + `<span class="bold">${f4(C.mse.cf)}</span> on these ${C.tau.length} people; the `
        + `S-learner and the T-learner produce clouds that are much flatter than the truth, `
        + `at ${f4(C.mse.s)} and ${f4(C.mse.t)}. Flat is the characteristic failure here: `
        + `predicting a bit of everybody's effect towards the average is what a model does `
        + `when it is spending its capacity on predicting the outcome instead.</p>`;
    } else if (view === 'tree') {
      /* One tree, drawn as a tree. Depth three fits on a canvas and shows what
       * the criterion actually did: split on the variables that change the
       * effect, not on the ones that change the outcome. */
      const depth = 3;
      const leaves = [];
      const walk = (node, d, x0, x1, parent) => {
        const cx = (x0 + x1) / 2;
        const cy = 30 + d * ((h - 60) / depth);
        if (parent) {
          g.append('line')
            .attr('x1', parent[0]).attr('y1', parent[1]).attr('x2', cx).attr('y2', cy)
            .attr('stroke', 'var(--stone)').attr('stroke-width', 1.6);
        }
        if (node.leaf) {
          leaves.push([cx, cy, node]);
          return;
        }
        g.append('rect')
          .attr('x', cx - 40).attr('y', cy - 12).attr('width', 80).attr('height', 24)
          .attr('rx', 4).attr('fill', 'var(--paper)')
          .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4);
        g.append('text').attr('class', 'chart-note')
          .attr('x', cx).attr('y', cy + 4).attr('text-anchor', 'middle')
          .style('font-size', '11.5px')
          .text(`${node.name} < ${f2(node.t)}`);
        walk(node.left, d + 1, x0, cx, [cx, cy + 12]);
        walk(node.right, d + 1, cx, x1, [cx, cy + 12]);
      };
      walk(C.tree, 0, 0, w, null);
      const taus = leaves.map((l) => l[2].tau);
      const scale = d3.scaleLinear().domain([d3.min(taus), d3.max(taus)]).range([0.25, 1]);
      leaves.forEach(([cx, cy, node]) => {
        g.append('rect')
          .attr('x', cx - 26).attr('y', cy - 12).attr('width', 52).attr('height', 24)
          .attr('rx', 4)
          .attr('fill', 'var(--primary)').attr('fill-opacity', scale(node.tau));
        g.append('text').attr('class', 'chart-note')
          .attr('x', cx).attr('y', cy + 4).attr('text-anchor', 'middle')
          .style('font-size', '11.5px').attr('fill', 'var(--paper)')
          .text(f2(node.tau));
      });
      annotate(g, 0, -14, 'leaves: the effect, estimated on the held out half',
        { anchor: 'start' }).style('font-size', '12px');
      const used = new Set();
      const collect = (n) => { if (!n.leaf) { used.add(n.name); collect(n.left); collect(n.right); } };
      collect(C.tree);
      readout.innerHTML =
        `<p>One tree from the forest, depth ${depth}. It split on `
        + `<span class="bold">${[...used].join(', ')}</span>, and the effect in this world is `
        + `built from V0 and V1, so it found them. What makes it honest is that the numbers `
        + `in the leaves were computed from a different half of the sample than the one that `
        + `chose where to cut: without that, the effect in a leaf is the maximum of a lot of `
        + `noisy differences and comes out too extreme. The leaves run from `
        + `${f2(d3.min(taus))} to ${f2(d3.max(taus))}, against a true effect that ranges over `
        + `about the same interval.</p>`;
    } else {
      const x = d3.scaleLinear().domain(d3.extent(P, (r) => r.prognostic)).range([0, w]);
      const y = d3.scaleLinear()
        .domain([0, d3.max(P, (r) => Math.max(r.s, r.t, r.cf)) * 1.08]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, {
        x: 'how strongly the covariates move the outcome',
        y: 'error against the true effect',
      });
      const series = [['s', 'var(--cosmos)', 'S-learner'], ['t', 'var(--aqua)', 'T-learner'],
        ['cf', 'var(--primary)', 'causal forest']];
      series.forEach(([k, colour]) => {
        g.append('path')
          .attr('d', line(P.map((r) => [x(r.prognostic), y(r[k])])))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.4);
        P.forEach((r) => {
          g.append('circle').attr('cx', x(r.prognostic)).attr('cy', y(r[k])).attr('r', 3.6)
            .attr('fill', colour);
        });
      });
      let cursor = 0;
      series.forEach(([, colour, name]) => {
        const label = annotate(g, cursor, -14, name, { anchor: 'start' })
          .style('font-size', '12px').attr('fill', colour);
        cursor += label.node().getComputedTextLength() + 20;
      });
      const first = P[0];
      const last = P[P.length - 1];
      readout.innerHTML =
        `<p>The horizontal axis is how much of the outcome the covariates explain, which has `
        + `nothing to do with the treatment effect at all. With none of it, the S-learner is `
        + `at ${f4(first.s)} and the forest at ${f4(first.cf)}. Turn it up and the S-learner `
        + `goes to <span class="bold">${f4(last.s)}</span> while the forest goes to `
        + `${f4(last.cf)}. A model that predicts the outcome and reads the effect off the `
        + `difference between two predictions spends its capacity where the variance is, and `
        + `the variance is in the part that has nothing to do with the question.</p>`;
    }
  }

  render();
}
