/* Four bit weights: the levels, the error, and the assumption underneath.
 *
 * Three views. The first draws the sixteen levels of each format on a number
 * line, which is the whole difference between them: uniform steps, floating
 * point steps, or the quantiles of a normal. The second sweeps the block size,
 * which is the trade nobody mentions, because a smaller block means a better
 * fit and more scales to store, and the honest axis for that is bits per weight
 * and not bits per index. The third is the end of the argument: an adapter
 * trained on top of a base that has been squashed to four bits.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const FORMATS = [['nf4', 'var(--primary)'], ['int4', 'var(--anchor)'], ['fp4', 'var(--cosmos)']];

export function initQuantWidget(data) {
  const Q = data.quant;
  const buttons = document.querySelector('#quant-buttons');
  const readout = document.querySelector('#quant-readout');
  const st = { view: 'levels' };

  const chart = makeChart('#quant-chart', {
    width: 640, height: 360, margin: { top: 56, right: 34, bottom: 62, left: 80 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['levels', 'the sixteen levels'],
    ['blocks', 'block size against bits'], ['train', 'training on a squashed base']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'levels') {
      const x = d3.scaleLinear().domain([-1.05, 1.05]).range([0, w]);
      const y = d3.scalePoint().domain(FORMATS.map(([f]) => f)).range([30, h - 40]);
      drawAxes(g, x, y, w, h, { xTicks: 5, yValues: [] });
      axisLabels(g, w, h, { x: 'level, as a fraction of the block maximum', y: '' });
      FORMATS.forEach(([f, colour]) => {
        const lv = Q[f];
        layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(f)).attr('y2', y(f))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 1);
        lv.forEach((v) => {
          layer.append('line').attr('x1', x(v)).attr('x2', x(v))
            .attr('y1', y(f) - 9).attr('y2', y(f) + 9)
            .attr('stroke', colour).attr('stroke-width', 2);
        });
        layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', y(f) - 16)
          .style('font-size', '11px').attr('fill', colour)
          .text(`${f}, ${lv.length} distinct levels`);
      });
    } else if (st.view === 'blocks') {
      const B = Q.blocks;
      const x = d3.scaleLinear().domain([4, 4.75]).range([0, w]);
      const y = d3.scaleLinear()
        .domain([d3.min(B.map((b) => b.err)) * 0.92, d3.max(B.map((b) => b.err)) * 1.06])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, xFmt: d3.format('.2f'), yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'bits per weight, scales included', y: 'relative error of the matrix' });
      [['bits', 'var(--primary)', 'one scale per block, 32 bits each'],
        ['bits_double', 'var(--anchor)', 'the scales quantised too']]
        .forEach(([key, colour, name], i) => {
          const pts = B.map((b) => [b[key], b.err]);
          layer.append('path').attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts))
            .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.1);
          pts.forEach((p, k) => {
            layer.append('circle').attr('cx', x(p[0])).attr('cy', y(p[1])).attr('r', 4)
              .attr('fill', colour);
            if (i === 0) {
              layer.append('text').attr('class', 'chart-note').attr('x', x(p[0]))
                .attr('y', y(p[1]) - 11).attr('text-anchor', 'middle')
                .style('font-size', '10.5px').text(String(B[k].block));
            }
          });
          layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -40 + i * 15)
            .style('font-size', '11px').attr('fill', colour).text(name);
        });
    } else {
      const R = data.qlora;
      const ranks = Array.from(new Set(R.map((r) => r.rank))).sort((a, b) => a - b);
      const x = d3.scaleBand().domain(ranks.map(String)).range([0, w]).padding(0.3);
      const vals = R.flatMap((r) => [r.mean - r.sd, r.mean + r.sd]);
      const y = d3.scaleLinear().domain([d3.min(vals) - 0.008, d3.max(vals) + 0.006]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'rank of the adapter', y: 'accuracy on the new task' });
      const arms = [['float base', 'var(--anchor)'], ['nf4 base', 'var(--primary)']];
      arms.forEach(([arm, colour], i) => {
        ranks.forEach((rk) => {
          const row = R.find((r) => r.arm === arm && r.rank === rk);
          const bw = x.bandwidth() / 2 - 2;
          layer.append('rect').attr('x', x(String(rk)) + i * (bw + 4)).attr('width', bw)
            .attr('y', y(row.mean)).attr('height', h - y(row.mean)).attr('fill', colour);
        });
        layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -40 + i * 15)
          .style('font-size', '11px').attr('fill', colour)
          .text(arm === 'float base' ? 'adapter on the full precision base'
            : 'adapter on the four bit base');
      });
    }

    const rows = Q.rows.map((r) => `<tr><td>layer ${r.layer + 1}</td>`
      + `<td>${r.shape[0]} by ${r.shape[1]}</td><td>${r.kurtosis.toFixed(2)}</td>`
      + `<td>${r.ks.toFixed(4)}</td><td>${r.nf4.toFixed(4)}</td>`
      + `<td>${r.int4.toFixed(4)}</td><td>${r.fp4.toFixed(4)}</td></tr>`).join('');
    const best = ['nf4', 'int4', 'fp4'].map((f) => [f,
      Q.rows.reduce((s, r) => s + r[f], 0) / Q.rows.length]).sort((a, b) => a[1] - b[1]);
    const drop = Q.base_acc - Q.quant_acc;
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>layer</th><th>shape</th>`
      + `<th>kurtosis</th><th>distance from normal</th><th>nf4</th><th>int4</th><th>fp4</th>`
      + `</tr></thead><tbody>${rows}</tbody></table>`
      + `<div class="gen-note">The case for the quantile format rests on the weights being `
      + `roughly normal, so the first two columns check that on the weights it is being `
      + `applied to: a normal distribution has kurtosis 3, and the last column is the worst `
      + `gap between the empirical distribution and the fitted normal. Averaged over the `
      + `layers, the best of the three formats is <span class="bold">${best[0][0]}</span> at `
      + `${best[0][1].toFixed(4)} relative error, then ${best[1][0]} at `
      + `${best[1][1].toFixed(4)} and ${best[2][0]} at ${best[2][1].toFixed(4)}. Note that `
      + `fp4 has only ${Q.fp4.length} distinct levels rather than sixteen, because a `
      + `floating point format spends one of its codes on negative zero. Squashing the whole `
      + `network to four bits moves its accuracy from ${Q.base_acc.toFixed(4)} to `
      + `${Q.quant_acc.toFixed(4)}, a drop of ${drop.toFixed(4)}.</div>`;
  }

  render();
  return { render, state: st };
}
