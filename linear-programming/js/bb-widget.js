/* Branch and bound on a twelve item knapsack, replayed one node at a time.
 *
 * The whole search runs in the browser at load, both with the bound and with it
 * switched off, and the node counts are checked against the generator. The
 * slider indexes a list of measured stops rather than a range of numbers, so
 * every position it can take is a node that was really opened.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { branchAndBound, knapsackLP } from './lpkit.js';

export function initBbWidget(data) {
  const D = data.demo;
  const w8 = D.weights;
  const val = D.values;
  const cap = D.capacity;
  const readout = document.querySelector('#bb-readout');
  const slider = document.querySelector('#bb-node');
  const label = document.querySelector('#bb-node-label');
  const controls = d3.select('#bb-controls');

  const arms = {
    bound: branchAndBound(w8, val, cap, { useBound: true }),
    plain: branchAndBound(w8, val, cap, { useBound: false }),
  };
  const relax = knapsackLP(w8, val, cap);
  let arm = 'bound';

  /* Sixty stops spread geometrically over the search: the slider indexes this
     list, so every position is a node that was really opened, and a sweep of
     the control visits sixty real states instead of four thousand numbers. */
  const stopsFor = (n) => {
    const out = new Set();
    for (let k = 0; k <= 60; k++) out.add(Math.max(1, Math.min(n, Math.round(n ** (k / 60)))));
    return [...out].sort((a, b) => a - b);
  };

  const chart = makeChart('#bb-chart', {
    width: 640, height: 370, margin: { top: 48, right: 126, bottom: 54, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scaleLog().range([0, w]);
  const y = d3.scaleLinear().range([h, 0]);
  const paths = {};
  ['plain', 'bound'].forEach((k) => {
    paths[k] = g.append('path').attr('fill', 'none')
      .attr('stroke', k === 'bound' ? 'var(--primary)' : 'var(--cosmos)')
      .attr('stroke-width', k === 'bound' ? 2.6 : 1.8);
  });
  const lpLine = g.append('line').attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6)
    .attr('stroke-dasharray', '5 4');
  const optLine = g.append('line').attr('stroke', '#8e9aa6').attr('stroke-width', 1.4);
  const marker = g.append('circle').attr('r', 6).attr('fill', 'var(--primary)');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 0)`);

  /* the incumbent after each node, which the search already recorded */
  const series = (trace) => trace.map((node, i) => [i + 1, node.best]);
  const curves = { bound: series(arms.bound.trace), plain: series(arms.plain.trace) };

  function render() {
    const stops = stopsFor(arms[arm].trace.length);
    slider.min = '0';
    slider.max = String(stops.length - 1);
    if (Number(slider.value) > stops.length - 1) slider.value = String(stops.length - 1);
    const k = stops[Number(slider.value)];
    const node = arms[arm].trace[k - 1];
    const bestSoFar = curves[arm][k - 1][1];

    x.domain([1, Math.max(arms.bound.trace.length, arms.plain.trace.length)]);
    y.domain([0, Math.max(relax.value, D.optimum) * 1.06]);
    const xt = [1, 10, 100, 1000, 10000].filter((v) => v >= x.domain()[0] && v <= x.domain()[1]);
    drawGrid(g, x, y, w, h, { xValues: xt, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: xt, yTicks: 5, xFmt: d3.format(',d') });
    axisLabels(g, w, h, { x: 'nodes opened', y: 'best bag found so far',
      leftBudget: margin.left });

    ['plain', 'bound'].forEach((kk) => {
      paths[kk].attr('d', d3.line().curve(d3.curveStepAfter)
        .x((p) => x(p[0])).y((p) => y(p[1]))(curves[kk]))
        .attr('opacity', kk === arm ? 1 : 0.45);
    });
    lpLine.attr('x1', 0).attr('x2', w).attr('y1', y(relax.value)).attr('y2', y(relax.value));
    optLine.attr('x1', 0).attr('x2', w).attr('y1', y(D.optimum)).attr('y2', y(D.optimum));
    marker.attr('cx', x(k)).attr('cy', y(bestSoFar));

    key.selectAll('*').remove();
    const items = [
      ['var(--anchor)', `relaxation ${relax.value.toFixed(2)}`, '5 4'],
      ['#8e9aa6', `optimum ${D.optimum}`, null],
      ['var(--primary)', 'with the bound', null],
      ['var(--cosmos)', 'without it', null],
    ];
    items.forEach(([col, text, dash], i) => {
      key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 32 + 6).attr('y2', i * 32 + 6)
        .attr('stroke', col).attr('stroke-width', 2).attr('stroke-dasharray', dash);
      const words = text.split(' ');
      key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 32 + 19)
        .style('font-size', '10.5px').text(words.slice(0, 2).join(' '));
      if (words.length > 2) {
        key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 32 + 29)
          .style('font-size', '10.5px').text(words.slice(2).join(' '));
      }
    });

    title.text(`${D.n} items, a bag that holds ${cap}: `
      + `${arms[arm].trace.length.toLocaleString('en-US')} nodes `
      + `${arm === 'bound' ? 'with the bound' : 'with the bound switched off'}`);
    sub.text(`node ${k.toLocaleString('en-US')}: best bag so far ${bestSoFar.toFixed(0)}`);
    label.textContent = `${k.toLocaleString('en-US')} of ${arms[arm].trace.length.toLocaleString('en-US')}`;

    const why = {
      pruned: 'cut off, because even filling the rest of the bag with fractions of the '
        + 'best items left cannot beat what we already have',
      incumbent: 'the best complete bag so far, so it becomes the number everything '
        + 'else is measured against',
      leaf: 'the end of a branch: every item has been decided',
      open: 'still worth exploring, so both of its children go on the stack',
    }[node.why];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>with the bound</th><th>without it</th><th>every subset</th></tr>
        <tr><td>nodes opened</td>
            <td><span class="value">${arms.bound.nodes.toLocaleString('en-US')}</span></td>
            <td>${arms.plain.nodes.toLocaleString('en-US')}</td>
            <td>${D.brute_force.toLocaleString('en-US')}</td></tr>
        <tr><td>best bag found</td>
            <td>${arms.bound.best.toFixed(0)}</td>
            <td>${arms.plain.best.toFixed(0)}</td>
            <td>${D.optimum}</td></tr>
      </table>
      <div class="gen-note">
        Node <span class="value">${k.toLocaleString('en-US')}</span> is at depth
        ${node.depth} of ${D.n}, carrying ${node.taken.toFixed(0)} in the bag with
        ${node.room.toFixed(0)} kilos of room left, and its relaxation says it could
        reach at most <span class="value">${node.bound.toFixed(2)}</span>. It is
        ${why}. The relaxation of the whole problem,
        <span class="value">${relax.value.toFixed(3)}</span>, is
        ${((relax.value / D.optimum - 1) * 100).toFixed(2)}% above the true answer of
        ${D.optimum}, and that small gap is the entire reason the search finishes:
        every node whose fractional dream falls below the best real bag is thrown
        away unopened.
      </div>`;
  }

  controls.selectAll('button')
    .data([{ k: 'bound', t: 'with the bound' }, { k: 'plain', t: 'bound switched off' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.k === arm ? '' : ' ghost'}`)
    .text((d) => d.t)
    .on('click', (_, d) => {
      arm = d.k;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.k === arm ? '' : ' ghost'}`);
      render();
    });
  slider.addEventListener('input', render);
  render();
  return { render, arms, relax };
}
