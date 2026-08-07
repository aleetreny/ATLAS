/* The same 2,000 digits, factorised twice: once whole, once with nine pixels
 * in ten hidden.
 *
 * The first view is the identity with the directions article: a truncated SVD
 * of the centred matrix is what that page called PCA, and the two curves are
 * the same curve. The second view is the whole of this article in one picture,
 * because the three ways of dealing with a hole are three different answers
 * and only one of them is fitting the model that was written down.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

export function initHolesWidget(data) {
  const D = data.digits;
  const buttons = document.querySelector('#hole-buttons');
  const readout = document.querySelector('#hole-readout');
  const st = { view: 'holed' };

  const chart = makeChart('#hole-chart', {
    width: 640, height: 380, margin: { top: 46, right: 132, bottom: 58, left: 74 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['holed', 'with nine pixels in ten hidden'],
    ['complete', 'with nothing missing']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const ranks = D.complete.map((row) => row.rank);

  /* The right margin is 132 units and `.chart-note` at 11px runs about 6.5 a
     character, so a label past eighteen characters leaves the canvas. And the
     three curves end within a few pixels of each other, so the labels are
     separated by value rather than by their own y. */
  const pending = [];
  function line(series, colour, dashed, label, x, y, rows, key) {
    plot.append('path').attr('fill', 'none').attr('stroke', colour)
      .attr('stroke-width', 2.2).attr('stroke-dasharray', dashed ? '5 3' : null)
      .attr('d', d3.line().x((d) => x(d.rank)).y((d) => y(d[key]))(rows));
    plot.selectAll(null).data(rows).enter().append('circle')
      .attr('cx', (d) => x(d.rank)).attr('cy', (d) => y(d[key])).attr('r', 3)
      .attr('fill', colour);
    pending.push({ label, colour, y: y(rows[rows.length - 1][key]) + 3.5, yMax: h + 8 });
  }

  function flushLabels() {
    spread(pending, 18).forEach((d) => {
      g.append('text').attr('class', 'chart-note').attr('x', w + 6).attr('y', d.y)
        .style('font-size', '11px').style('fill', d.colour).text(d.label);
    });
    pending.length = 0;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();
    pending.length = 0;

    const x = d3.scaleLog().domain([ranks[0] * 0.85, ranks[ranks.length - 1] * 1.15]).range([0, w]);

    if (st.view === 'complete') {
      const rows = D.complete;
      const vals = rows.flatMap((d) => [d.svd, d.published_pca]);
      const y = d3.scaleLinear().domain([d3.min(vals) * 0.9, d3.max(vals) * 1.05]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: ranks, yTicks: 5 });
      line(null, 'var(--primary)', false, 'SVD here', x, y, rows, 'svd');
      plot.selectAll(null).data(rows).enter().append('path')
        .attr('d', d3.symbol().type(d3.symbolCross).size(70))
        .attr('transform', (d) => `translate(${x(d.rank)},${y(d.published_pca)}) rotate(45)`)
        .attr('fill', 'var(--anchor)');
      pending.push({ label: 'PCA published', colour: 'var(--anchor)',
        y: y(rows[rows.length - 1].published_pca) + 16, yMax: h + 8 });
      flushLabels();
      drawAxes(g, x, y, w, h, { xValues: ranks, yTicks: 5, xFmt: (v) => String(v), yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'factors kept', y: 'mean squared error per pixel' });
      const worst = d3.max(rows, (d) => Math.abs(d.svd - d.published_pca));
      readout.innerHTML =
        `The same ${D.n.toLocaleString('en-US')} digits, the same five ranks, and the two curves are `
        + `one curve: the largest disagreement between the truncated SVD computed here and the `
        + `error <a href="../directions/">the directions article</a> published for PCA is `
        + `<span class="value">${worst.toExponential(2)}</span>. That is the point of the view, and `
        + `it is not a coincidence to be admired: minimising squared error at rank k over a `
        + `<span class="bold">complete</span> matrix has one answer, and everything else on this `
        + `page is about what happens when the matrix is not complete.`;
    } else {
      const rows = D.holed;
      const vals = rows.flatMap((d) => [d.zero, d.mean, d.observed]).concat([D.hidden_baseline]);
      const y = d3.scaleLinear().domain([d3.min(vals) * 0.94, d3.max(vals) * 1.04]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: ranks, yTicks: 5 });
      plot.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(D.hidden_baseline)).attr('y2', y(D.hidden_baseline))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 4');
      pending.push({ label: 'column average', colour: 'var(--squidink)',
        y: y(D.hidden_baseline) + 3.5, yMax: h + 8 });
      line(null, 'var(--cosmos)', false, 'holes as zero', x, y, rows, 'zero');
      line(null, 'var(--anchor)', false, 'holes averaged', x, y, rows, 'mean');
      line(null, 'var(--primary)', false, 'observed only', x, y, rows, 'observed');
      flushLabels();
      drawAxes(g, x, y, w, h, { xValues: ranks, yTicks: 5, xFmt: (v) => String(v), yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'factors kept', y: 'error on the hidden pixels' });
      const best = rows.reduce((a, d) => (d.observed < a.observed ? d : a));
      const same = rows.find((d) => d.rank === best.rank);
      readout.innerHTML =
        `Every pixel of the same ${D.n.toLocaleString('en-US')} digits is still there; `
        + `<span class="value">${(100 * D.hide).toFixed(0)}%</span> of them are simply not shown to `
        + `the factorisation, and the error is measured on those `
        + `${D.hidden_cells.toLocaleString('en-US')} hidden ones. Filling the holes with zero is the `
        + `thing a sparse matrix library does by default and it is the worst of the three at every `
        + `rank (<span class="value">${rows[rows.length - 1].zero.toFixed(4)}</span> at rank `
        + `${rows[rows.length - 1].rank}); filling them with the column average is better; fitting `
        + `only where there is data is best, `
        + `<span class="value">${best.observed.toFixed(4)}</span> at rank `
        + `<span class="value">${best.rank}</span> against `
        + `<span class="value">${same.mean.toFixed(4)}</span> and `
        + `<span class="value">${same.zero.toFixed(4)}</span> for the other two at the same rank. `
        + `All three are the same objective and the same rank. What differs is which entries the `
        + `sum runs over. `
        + `${rows[rows.length - 1].observed > best.observed
          ? `And the one that fits properly is the one that can overfit: by rank `
            + `${rows[rows.length - 1].rank} it is back up at `
            + `<span class="value">${rows[rows.length - 1].observed.toFixed(4)}</span>, `
            + `${rows[rows.length - 1].observed > D.hidden_baseline
              ? 'which is worse than answering the column average'
              : 'still under the column average but climbing'}. `
            + `Nothing here is regularised, and with ${(100 * D.hide).toFixed(0)}% of the entries `
            + `missing there is not much holding it down.`
          : 'The observed arm has not turned inside this range of ranks.'}`;
    }
  }

  render();
  return { render, state: st };
}
