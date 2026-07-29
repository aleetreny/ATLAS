/* Five steps through the mechanism, on one point of the actual dataset.
 *
 * Every panel is computed from the same 500 digits the widget below runs on, so
 * the bandwidth drawn in step two is the bandwidth used in step five.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));

export function initScrollyTsne(data, computed) {
  const chart = makeChart('#tsne-chart', {
    width: 640, height: 460, margin: { top: 62, right: 26, bottom: 76, left: 66 },
  });
  const { g, w, h } = chart;
  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', h + 58).attr('text-anchor', 'middle')
    .style('font-size', '11.5px').style('text-transform', 'none');

  const FOCUS = computed.focus;
  const P = computed.P;
  const betas = computed.betas;
  const D2 = computed.D2;
  const labels = data.digits.labels;
  const n = labels.length;

  function clear() {
    body.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();
  }

  /* the conditional distribution of one point, sorted, as a bar chart */
  function drawRow(i, showTail) {
    clear();
    const vals = [];
    for (let j = 0; j < n; j++) if (j !== i) vals.push({ j, p: P.cond[i][j], d: Math.sqrt(D2[i][j]) });
    vals.sort((a, b) => b.p - a.p);
    const K = 40;
    const top = vals.slice(0, K);
    const x = d3.scalePoint().domain(d3.range(K)).range([10, w - 10]);
    const y = d3.scaleLinear().domain([0, d3.max(top, (t) => t.p) * 1.1]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 0, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 0, yTicks: 5, yFmt: d3.format('.3f') });
    axisLabels(g, w, h, { x: 'the 40 likeliest neighbours, in order', y: 'probability' });
    body.selectAll('rect').data(top).join('rect')
      .attr('x', (t, k) => x(k) - 5).attr('width', 10)
      .attr('y', (t) => y(t.p)).attr('height', (t) => h - y(t.p))
      .attr('fill', (t) => (showTail ? DIGIT(labels[t.j]) : 'var(--primary)'))
      .attr('opacity', 0.9);
  }

  const steps = [
    () => {
      clear();
      const Y = computed.pca;
      const { x, y } = equalScales(Y, w, h, { pad: 1.08 });
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      axisLabels(g, w, h, { x: 'first component', y: 'second component' });
      body.selectAll('circle').data(Y).join('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
        .attr('fill', (d, i) => (i === FOCUS ? 'var(--squidink)' : DIGIT(labels[i])))
        .attr('opacity', (d, i) => (i === FOCUS ? 1 : 0.55));
      body.append('circle').attr('cx', x(Y[FOCUS][0])).attr('cy', y(Y[FOCUS][1])).attr('r', 11)
        .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.4);
      wrapLabel(title, `${n} digits, drawn by PCA for want of anything better, with one of them `
        + 'marked', w - 8);
      wrapLabel(caption, 'the question is not where this point goes, it is who its neighbours are',
        w - 8);
    },
    () => {
      drawRow(FOCUS, false);
      wrapLabel(title, `the marked point's neighbour distribution: how likely it is to pick each `
        + `other point, at a bandwidth of ${(1 / Math.sqrt(2 * betas[FOCUS])).toFixed(3)}`, w - 8);
      wrapLabel(caption, 'a gaussian centred on it, normalised so the probabilities add to one',
        w - 8);
    },
    () => {
      clear();
      const sig = Array.from(betas, (b) => 1 / Math.sqrt(2 * b));
      const x = d3.scaleLinear().domain([d3.min(sig) * 0.95, d3.max(sig) * 1.05]).range([0, w]);
      const bins = d3.bin().domain(x.domain()).thresholds(30)(sig);
      const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) * 1.1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
      axisLabels(g, w, h, { x: 'bandwidth chosen for that point', y: 'how many points' });
      body.selectAll('rect').data(bins).join('rect')
        .attr('x', (b) => x(b.x0) + 1).attr('width', (b) => Math.max(1, x(b.x1) - x(b.x0) - 2))
        .attr('y', (b) => y(b.length)).attr('height', (b) => h - y(b.length))
        .attr('fill', 'var(--primary)').attr('opacity', 0.85);
      body.append('line').attr('x1', x(sig[FOCUS])).attr('x2', x(sig[FOCUS]))
        .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
      wrapLabel(title, 'one bandwidth per point, each solved for by binary search so that every '
        + 'point has the same effective number of neighbours', w - 8);
      wrapLabel(caption, 'dashed line: the marked point. the spread is the algorithm adapting to '
        + 'density', w - 8);
    },
    () => {
      clear();
      const x = d3.scaleLinear().domain([0, 6]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
      axisLabels(g, w, h, { x: 'distance in the map', y: 'weight it is given' });
      const pts = d3.range(0, 6.01, 0.05);
      body.append('path').datum(pts).attr('fill', 'none')
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2.6)
        .attr('d', d3.line().x((t) => x(t)).y((t) => y(Math.exp(-t * t / 2))));
      body.append('path').datum(pts).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6)
        .attr('d', d3.line().x((t) => x(t)).y((t) => y(1 / (1 + t * t))));
      /* anchored at the right edge and measured: the fixed x = w - 304 start
         pushed the longer label 86px past the canvas */
      [['var(--anchor)', 'a gaussian, which is what the data end uses'],
        ['var(--primary)', 'a t with one degree of freedom, which is what the map uses']]
        .forEach(([c, t], i) => {
          const tx = body.append('text').attr('x', w - 8).attr('y', 22 + i * 18)
            .attr('text-anchor', 'end')
            .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
            .attr('fill', 'var(--squidink)').text(t);
          const len = tx.node().getComputedTextLength();
          body.append('line').attr('x1', w - 8 - len - 26).attr('x2', w - 8 - len - 6)
            .attr('y1', 18 + i * 18).attr('y2', 18 + i * 18)
            .attr('stroke', c).attr('stroke-width', 2.6);
        });
      wrapLabel(title, 'the map is allowed a heavier tail than the data', w - 8);
      wrapLabel(caption, 'at a distance of 4 the t gives 0.059 where the gaussian gives 0.000335, '
        + 'about 176 times more', w - 8);
    },
    () => {
      clear();
      const Y = computed.mid;
      const { x, y } = equalScales(Y, w, h, { pad: 1.1 });
      body.selectAll('circle').data(Y).join('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
        .attr('fill', (d, i) => DIGIT(labels[i])).attr('opacity', 0.85);
      wrapLabel(title, `${computed.midIter} steps in, still inside the early exaggeration`, w - 8);
      wrapLabel(caption, 'every neighbour probability multiplied by twelve, which packs the groups '
        + 'and leaves room between them', w - 8);
    },
    () => {
      clear();
      const Y = computed.final;
      const { x, y } = equalScales(Y, w, h, { pad: 1.1 });
      body.selectAll('circle').data(Y).join('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3)
        .attr('fill', (d, i) => DIGIT(labels[i])).attr('opacity', 0.85);
      wrapLabel(title, `${data.meta.iters} steps, colour by the digit each one really is`, w - 8);
      wrapLabel(caption, 'nothing in the calculation was ever shown a label', w - 8);
    },
  ];

  /* The last two panels show a run that is still going when the page loads, so
     the scrolly needs to be able to redraw the step it is on. Every handler is
     an idempotent state declaration, which is what makes that safe. */
  let last = 0;
  const wrapped = steps.map((fn, i) => () => {
    last = i;
    fn();
  });
  wrapped[0]();
  const sc = scrolly('#tsne-scrolly .step', wrapped);
  sc.refresh = () => wrapped[last]();
  return sc;
}
