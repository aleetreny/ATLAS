/* Depth, and the quantity that explains it.
 *
 * The first view is the accuracy of a trained stack against how many layers it
 * has. The second is the same operator applied with no training at all, which
 * is the honest version of the explanation: repeated averaging has a fixed
 * point, every node converges to it, and the Dirichlet energy that measures
 * how far apart two neighbours still are goes to zero whether or not anybody
 * is fitting weights.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initDepthWidget(data) {
  const D = data.depth;
  const buttons = document.querySelector('#depth-buttons');
  const readout = document.querySelector('#depth-readout');
  const st = { view: 'trained' };

  const chart = makeChart('#depth-chart', {
    width: 640, height: 380, margin: { top: 44, right: 112, bottom: 58, left: 74 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['trained', 'trained stacks'], ['smoothing', 'the operator alone']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();
    g.selectAll('g.axis.y2').remove();

    if (st.view === 'trained') {
      const rows = D.trained;
      const xs = rows.map((d) => d.layers);
      const x = d3.scalePoint().domain(xs).range([0, w]).padding(0.5);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      const lo = Math.max(1e-6, d3.min(rows, (d) => d.energy));
      const y2 = d3.scaleLog().domain([lo * 0.6, d3.max(rows, (d) => d.energy) * 1.6]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });

      plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
        .attr('stroke-width', 2.4)
        .attr('d', d3.line().x((d) => x(d.layers)).y((d) => y(d.acc))(rows));
      plot.selectAll(null).data(rows).enter().append('circle')
        .attr('cx', (d) => x(d.layers)).attr('cy', (d) => y(d.acc)).attr('r', 3.2)
        .attr('fill', 'var(--primary)');
      plot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
        .attr('stroke-width', 1.8).attr('stroke-dasharray', '5 3')
        .attr('d', d3.line().x((d) => x(d.layers)).y((d) => y2(Math.max(d.energy, lo * 0.61)))(rows));

      const ax2 = g.append('g').attr('class', 'axis y2').attr('transform', `translate(${w},0)`)
        .call(d3.axisRight(y2).ticks(4, '~e'));
      ax2.selectAll('text').style('fill', 'var(--anchor)');
      g.append('text').attr('class', 'chart-note')
        .attr('x', w + 6).attr('y', y(rows[rows.length - 1].acc) + 3.5)
        .style('font-size', '11px').style('fill', 'var(--primary)').text('accuracy');
      g.append('text').attr('class', 'chart-note')
        .attr('transform', `translate(${w + 62},${h / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--anchor)')
        .text('spread between neighbours');

      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.0%') });
      axisLabels(g, w, h, { y: 'test nodes correct' });
      rows.forEach((d) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(d.layers)).attr('y', h + 18).attr('text-anchor', 'middle')
          .style('font-size', '11px').text(String(d.layers));
      });
      g.append('text').attr('class', 'chart-note')
        .attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle').style('font-size', '11.5px')
        .text('layers');

      const best = rows.reduce((a, d) => (d.acc > a.acc ? d : a));
      const deepest = rows[rows.length - 1];
      readout.innerHTML =
        `Best at <span class="value">${best.layers}</span> layer${best.layers === 1 ? '' : 's'} `
        + `(<span class="value">${pct(best.acc)}</span>), down to `
        + `<span class="value">${pct(deepest.acc)}</span> at ${deepest.layers}. The dashed line is `
        + `why: the spread between neighbouring representations falls from `
        + `<span class="value">${rows[0].energy.toExponential(2)}</span> at ${rows[0].layers} layer `
        + `to <span class="value">${deepest.energy.toExponential(2)}</span> at ${deepest.layers}, a `
        + `factor of <span class="value">${(rows[0].energy / Math.max(deepest.energy, 1e-12)).toExponential(1)}</span>. `
        + `A deeper stack is not a bigger model here, it is a longer average.`;
    } else {
      const rows = D.smoothing;
      const x = d3.scaleLinear().domain([0, rows[rows.length - 1].steps]).range([0, w]);
      const lo = Math.max(1e-8, d3.min(rows, (d) => d.energy));
      const y = d3.scaleLog().domain([lo * 0.6, d3.max(rows, (d) => d.energy) * 1.6]).range([h, 0]);
      const yTicks = [1e-8, 1e-6, 1e-4, 1e-2, 1].filter(
        (v) => v >= y.domain()[0] && v <= y.domain()[1]);
      drawGrid(g, x, y, w, h, { xTicks: 6, yValues: yTicks });
      plot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.4)
        .attr('d', d3.line().x((d) => x(d.steps)).y((d) => y(Math.max(d.energy, lo * 0.61)))(rows));
      plot.selectAll(null).data(rows).enter().append('circle')
        .attr('cx', (d) => x(d.steps)).attr('cy', (d) => y(Math.max(d.energy, lo * 0.61))).attr('r', 3)
        .attr('fill', 'var(--anchor)');
      drawAxes(g, x, y, w, h, { xTicks: 6, yValues: yTicks, xFmt: (v) => String(v),
        yFmt: d3.format('.0e') });
      axisLabels(g, w, h, { x: 'times the operator is applied', y: 'spread between neighbours' });
      g.append('text').attr('class', 'chart-note')
        .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px')
        .text('the word vectors, averaged over and over, with no weights anywhere');

      const first = rows[1];
      const last = rows[rows.length - 1];
      readout.innerHTML =
        `No model at all here: the same mixing matrix applied to the raw word vectors, over and `
        + `over. The spread between neighbours falls from `
        + `<span class="value">${rows[0].energy.toExponential(2)}</span> to `
        + `<span class="value">${last.energy.toExponential(2)}</span> in `
        + `${last.steps} rounds, a factor of `
        + `<span class="value">${(rows[0].energy / Math.max(last.energy, 1e-12)).toExponential(1)}</span>. `
        + `That is a property of averaging on this graph, not of any architecture, and it is the `
        + `thing a deep stack is fighting. Every trick in the deep graph literature, residual `
        + `connections included, is about keeping that number away from zero.`;
    }
  }

  render();
  return { render, state: st };
}
