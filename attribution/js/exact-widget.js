/* One prediction, taken apart, with the answer key drawn on top.
 *
 * A waterfall is the standard picture and it hides the thing worth seeing, so
 * the second view puts the exact value and the closed form side by side: a
 * waterfall always closes, whatever the numbers are, because closing is forced
 * by the definition.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));
const pretty = (s) => s.replace(/_/g, ' ').replace('od280/od315 of diluted wines', 'od280/od315');

export function initExactWidget(data) {
  const host = document.querySelector('#exact-widget');
  if (!host) return;
  const M = data.meta;
  const K = data.answer_key;
  const chart = makeChart('#exact-chart', {
    width: 640, height: 430, margin: { top: 42, right: 30, bottom: 60, left: 168 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#exact-readout');
  const label = document.querySelector('#exact-row-label');
  const slider = document.querySelector('#exact-row');
  const views = d3.select('#exact-views');
  let view = 'waterfall';
  let i = 0;

  const buttons = [
    ['waterfall', 'how the prediction is built'],
    ['key', 'against the closed form'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  slider.max = String(K.rows.length - 1);

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const row = K.rows[i];
    label.textContent = String(i + 1);
    const order = row.phi.map((v, j) => j).sort((a, b) => Math.abs(row.phi[b]) - Math.abs(row.phi[a]));
    const names = order.map((j) => pretty(M.columns[j]));
    const y = d3.scaleBand().domain(names).range([0, h]).padding(0.22);

    if (view === 'waterfall') {
      let cursor = K.base;
      const steps = order.map((j) => {
        const from = cursor;
        cursor += row.phi[j];
        return { j, from, to: cursor };
      });
      const lo = Math.min(K.base, ...steps.map((s) => Math.min(s.from, s.to)));
      const hi = Math.max(K.base, ...steps.map((s) => Math.max(s.from, s.to)));
      const x = d3.scaleLinear().domain([lo - 0.12, hi + 0.12]).nice().range([0, w]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'predicted flavanoids' });
      [[K.base, 'the average prediction'], [row.pred, 'this bottle']].forEach(([v, t], k) => {
        g.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', 0).attr('y2', h)
          .attr('stroke', k === 0 ? 'var(--anchor)' : 'var(--cosmos)')
          .attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
        /* El lado se elige por la posición, no por cuál de las dos líneas es:
         * la media de la predicción cae a la derecha en algunas botellas y la
         * etiqueta se salía 245 unidades del lienzo. */
        const right = x(v) > w * 0.6;
        annotate(g, x(v) + (right ? -6 : 6), k === 0 ? -12 : -28, `${t}: ${f3(v)}`,
          { anchor: right ? 'end' : 'start' })
          .style('font-size', '11.5px')
          .attr('fill', k === 0 ? 'var(--anchor)' : 'var(--cosmos)');
      });
      steps.forEach((s, k) => {
        const yy = y(names[k]);
        g.append('rect')
          .attr('x', x(Math.min(s.from, s.to))).attr('y', yy)
          .attr('width', Math.max(1.5, Math.abs(x(s.to) - x(s.from))))
          .attr('height', y.bandwidth())
          .attr('fill', row.phi[s.j] >= 0 ? 'var(--primary)' : 'var(--cosmos)')
          .attr('fill-opacity', 0.85);
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(Math.max(s.from, s.to)) + 7)
          .attr('y', yy + y.bandwidth() / 2 + 4)
          .style('font-size', '11.5px')
          .text(sgn(row.phi[s.j]));
      });
      const total = row.phi.reduce((a, b) => a + b, 0);
      readout.innerHTML =
        `<p>Bottle ${i + 1} of the held out set. The model predicts `
        + `<span class="bold">${f4(row.pred)}</span> where the average prediction over the `
        + `background is ${f4(K.base)}, and the ${M.p} bars are the exact Shapley values, `
        + `each one a sum over all ${(2 ** M.p).toLocaleString('en-US')} coalitions. They add `
        + `to <span class="bold">${f4(total)}</span> against a gap of `
        + `${f4(row.pred - K.base)}, a difference of `
        + `${Math.abs(total - (row.pred - K.base)).toExponential(0)}. That is the efficiency `
        + `axiom and it is worth being unimpressed by: it would hold just as neatly for a set `
        + `of wrong numbers whose errors cancelled. The other view is the one that cannot be `
        + `faked.</p>`;
    } else {
      const vals = order.flatMap((j) => [row.phi[j], row.key[j]]);
      const x = d3.scaleLinear()
        .domain([Math.min(0, d3.min(vals)) * 1.1, Math.max(0, d3.max(vals)) * 1.1])
        .nice().range([0, w]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, { x: 'attribution' });
      g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
      order.forEach((j, k) => {
        const yy = y(names[k]);
        g.append('rect')
          .attr('x', Math.min(x(0), x(row.phi[j]))).attr('y', yy + 2)
          .attr('width', Math.max(1.5, Math.abs(x(row.phi[j]) - x(0))))
          .attr('height', y.bandwidth() - 4)
          .attr('fill', 'var(--primary)').attr('fill-opacity', 0.8);
        g.append('line')
          .attr('x1', x(row.key[j])).attr('x2', x(row.key[j]))
          .attr('y1', yy - 1).attr('y2', yy + y.bandwidth() + 1)
          .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.6);
      });
      annotate(g, 0, -24, 'bars: the enumeration over every coalition', { anchor: 'start' })
        .style('font-size', '11.5px').attr('fill', 'var(--primary)');
      annotate(g, 0, -8, 'ticks: the additive model own term, in closed form',
        { anchor: 'start' }).style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      const worst = Math.max(...row.phi.map((v, j) => Math.abs(v - row.key[j])));
      readout.innerHTML =
        `<p>Every tick sits on its bar. The bars come from summing over `
        + `${(2 ** M.p).toLocaleString('en-US')} coalitions of a model that was never told `
        + `anything about Shapley values; the ticks come from reading the model's own term `
        + `for that column and centring it on the background. On this bottle the worst `
        + `disagreement is <span class="bold">${worst.toExponential(0)}</span>, and across `
        + `all ${K.rows.length} shown bottles it is ${K.worst.toExponential(0)}. This is the check that `
        + `matters: an attribution that summed correctly but distributed the credit wrongly `
        + `would pass the waterfall and fail here.</p>`;
    }
  }

  slider.addEventListener('input', (e) => { i = +e.target.value; render(); });
  render();
}
