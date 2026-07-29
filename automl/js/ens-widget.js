/* Averaging the ones that lost.
 *
 * Every automated system that has ever won a competition does this at the end,
 * and it is the cheapest part of the whole pipeline: the models are already
 * fitted. The selection is greedy with replacement on the out of fold
 * predictions, which is the only set that can choose an ensemble without
 * looking at the held out rows, and the held out curve is drawn afterwards
 * rather than optimised.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initEnsWidget(data) {
  const E = data.ensemble;
  const T = data.tables;
  const names = Object.keys(E);
  const buttons = document.querySelector('#ens-buttons');
  const readout = document.querySelector('#ens-readout');
  const st = { table: names[0] };

  const chart = makeChart('#ens-chart', {
    width: 640, height: 370, margin: { top: 56, right: 30, bottom: 58, left: 80 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  names.forEach((n) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = n;
    b.addEventListener('click', () => { st.table = n; render(); });
    buttons.appendChild(b);
    btns[n] = b;
  });

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.table));
    layer.selectAll('*').remove();
    const rows = E[st.table].rows;
    const tbl = T[st.table];

    const x = d3.scaleLinear().domain([1, rows.length]).range([0, w]);
    const all = rows.flatMap((r) => [r.oof, r.test]).concat([tbl.best.test, tbl.best.cv]);
    const y = d3.scaleLinear().domain([d3.min(all) - 0.01, d3.max(all) + 0.006]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, yFmt: d3.format('.3f') });
    axisLabels(g, w, h, { x: 'pipelines in the average', y: 'accuracy' });

    layer.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(tbl.best.test)).attr('y2', y(tbl.best.test))
      .attr('stroke', 'var(--smile)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
      .attr('y', y(tbl.best.test) - 6).attr('text-anchor', 'end')
      .style('font-size', '11px').attr('fill', 'var(--smile)')
      .text(`the single best pipeline, ${tbl.best.test.toFixed(4)}`);

    [['oof', 'var(--anchor)', 'out of fold, which is what chooses'],
      ['test', 'var(--primary)', 'held out, which is what counts']]
      .forEach(([key, colour, label], i) => {
        layer.append('path')
          .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r[key]))(rows))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
        layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -40 + i * 15)
          .style('font-size', '11px').attr('fill', colour).text(label);
      });

    const last = rows[rows.length - 1];
    const bestTest = rows.reduce((a, b) => (b.test > a.test ? b : a));
    const dips = rows.filter((r, i) => i > 0 && r.oof < rows[i - 1].oof - 1e-9).length;
    const gain = last.test - tbl.best.test;
    const sd = tbl.best.test_sd;
    readout.innerHTML = `<div class="gen-note">On <span class="bold">${st.table}</span>, `
      + `averaging ${last.k} picks (${last.distinct} distinct pipelines, chosen with `
      + `replacement so the good ones get more weight) reaches `
      + `<span class="bold">${last.test.toFixed(4)}</span> on held out rows against `
      + `${tbl.best.test.toFixed(4)} for the single best, a difference of `
      + `${gain >= 0 ? '+' : ''}${gain.toFixed(4)} `
      + (Math.abs(gain) <= sd
        ? `which is inside the ${sd.toFixed(4)} that this held out set wobbles under `
          + `resampling.`
        : `which is larger than the ${sd.toFixed(4)} that this held out set wobbles under `
          + `resampling.`)
      + ` The best held out score any size reached is ${bestTest.test.toFixed(4)} at `
      + `${bestTest.k}, and that number is not available in advance: the only curve the `
      + `system can see is the blue one. And the blue one is worth a second look, because it `
      + `is not monotone: ${dips} of its ${rows.length} steps go `
      + `<span class="bold">down</span> on the very set that chose them. Each step takes the `
      + `best move available, but a step is an average over one more member rather than an `
      + `extension of the previous one, so the best available move can still land lower. `
      + `Greedy selection with replacement is usually described as if it could not do that.`
      + `</div>`;
  }

  render();
  return { render, state: st };
}
