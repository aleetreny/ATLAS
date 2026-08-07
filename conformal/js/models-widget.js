/* Four models, one guarantee, four widths.
 *
 * Coverage and width on the same canvas rather than in two, because the whole
 * claim is that one of them is flat and the other is not, and two charts side
 * by side make that a comparison the reader has to perform instead of one they
 * can see.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initModelsWidget(data) {
  const host = document.querySelector('#models-widget');
  if (!host) return;
  const M = data.meta;
  const MO = data.models;
  const chart = makeChart('#models-chart', {
    width: 640, height: 400, margin: { top: 44, right: 30, bottom: 92, left: 74 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#models-readout');
  const views = d3.select('#models-views');
  let view = 'coverage';

  const buttons = [
    ['coverage', 'coverage: the promise'],
    ['width', 'width: the price'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const x = d3.scaleBand().domain(MO.rows.map((r) => r.key)).range([0, w]).padding(0.34);
    const target = 1 - M.alpha;
    const y = view === 'coverage'
      ? d3.scaleLinear().domain([target - 0.03, target + 0.03]).range([h, 0])
      : d3.scaleLinear().domain([0, d3.max(MO.rows, (r) => r.width) * 1.12]).range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h, {
      yFmt: view === 'coverage' ? d3.format('.1%') : null,
      xValues: [],
    });
    axisLabels(g, w, h, {
      y: view === 'coverage' ? 'coverage over 200 splits' : 'width of the interval',
      xOffset: 74,
    });

    if (view === 'coverage') {
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(target)).attr('y2', y(target))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4');
      /* Every dot sits on the promised line (that is the theorem), so a label
         hugging the line collides with the first value; it lives in the top
         margin, naming the line instead of touching it. */
      annotate(g, 4, -10, `dashed line: asked for ${pc(target, 0)}`, { anchor: 'start' })
        .style('font-size', '11.5px');
    }
    MO.rows.forEach((row) => {
      const v = view === 'coverage' ? row.coverage : row.width;
      const sd = view === 'coverage' ? row.coverage_sd : row.width_sd;
      const cx = x(row.key) + x.bandwidth() / 2;
      if (view === 'coverage') {
        g.append('line').attr('x1', cx).attr('x2', cx)
          .attr('y1', y(v - sd)).attr('y2', y(v + sd))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 7)
          .attr('stroke-linecap', 'round');
        g.append('circle').attr('cx', cx).attr('cy', y(v)).attr('r', 6)
          .attr('fill', 'var(--primary)');
      } else {
        g.append('rect')
          .attr('x', x(row.key)).attr('y', y(v))
          .attr('width', x.bandwidth()).attr('height', h - y(v))
          .attr('fill', row.key === 'forest' ? 'var(--primary)' : 'var(--cosmos)')
          .attr('fill-opacity', 0.85);
      }
      g.append('text').attr('class', 'chart-note')
        .attr('x', cx).attr('y', view === 'coverage' ? y(v) - 14 : y(v) - 8)
        .attr('text-anchor', 'middle').style('font-size', '12px')
        .text(view === 'coverage' ? f4(v) : f2(v));
      /* El nombre del modelo, en dos renglones si hace falta: "linear on one
       * feature" no cabe bajo una banda de ciento treinta unidades. */
      const words = row.label.split(' ');
      const lines = [];
      let cur = '';
      words.forEach((word) => {
        if ((`${cur} ${word}`).trim().length > 16) { lines.push(cur.trim()); cur = word; } else cur += ` ${word}`;
      });
      lines.push(cur.trim());
      lines.forEach((ln, k) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', cx).attr('y', h + 20 + k * 20)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(ln);
      });
    });

    const covs = MO.rows.map((r) => r.coverage);
    const widths = MO.rows.map((r) => r.width);
    readout.innerHTML = view === 'coverage'
      ? `<p>Four models on the same data, from a random forest to one that ignores the `
        + `features and returns the mean of the training set. Their coverages over `
        + `${MO.splits} splits are ${MO.rows.map((r) => f4(r.coverage)).join(', ')}, a range `
        + `of <span class="bold">${f4(Math.max(...covs) - Math.min(...covs))}</span>. The `
        + `guarantee never looked at the model, so it does not care that one of them is not `
        + `one.</p>`
      : `<p>And here is what the model is for. The forest's intervals are `
        + `<span class="bold">${f2(MO.rows[0].width)}</span> wide and the constant's are `
        + `<span class="bold">${f2(MO.rows[3].width)}</span>, a factor of `
        + `${f2(Math.max(...widths) / Math.min(...widths))}. Both are ${pc(1 - M.alpha, 0)} `
        + `intervals and both keep that promise. A useless model does not produce a wrong `
        + `interval here: it produces an honest one so wide that nobody would act on it, `
        + `which is the correct behaviour and not what an interval from a badly specified `
        + `regression would have done.</p>`;
  }

  render();
}
